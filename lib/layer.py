# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

import struct
import zlib
from numpy import *
import logging
import os
from cStringIO import StringIO
import time
import zipfile
logger = logging.getLogger(__name__)

from gettext import gettext as _

import tiledsurface
import strokemap
import mypaintlib
import helpers


## Module constants: compositing operations


COMPOSITE_OPS = [
    # (internal-name, display-name, description)
    ("svg:src-over", _("Normal"),
        _("The top layer only, without blending colors.")),
    ("svg:multiply", _("Multiply"),
        _("Similar to loading multiple slides into a single projector's slot "
          "and projecting the combined result.")),
    ("svg:screen", _("Screen"),
        _("Like shining two separate slide projectors onto a screen "
          "simultaneously. This is the inverse of 'Multiply'.")),
    ("svg:overlay", _("Overlay"),
        _("Overlays the backdrop with the top layer, preserving the backdrop's "
          "highlights and shadows. This is the inverse of 'Hard Light'.")),
    ("svg:darken", _("Darken"),
        _("The top layer is used only where it is darker than the backdrop.")),
    ("svg:lighten", _("Lighten"),
        _("The top layer is used only where it is lighter than the backdrop.")),
    ("svg:color-dodge", _("Dodge"),
        _("Brightens the backdrop using the top layer. The effect is similar "
          "to the photographic darkroom technique of the same name which is "
          "used for improving contrast in shadows.")),
    ("svg:color-burn", _("Burn"),
        _("Darkens the backdrop using the top layer. The effect looks similar "
          "to the photographic darkroom technique of the same name which is "
          "used for reducing over-bright highlights.")),
    ("svg:hard-light", _("Hard Light"),
        _("Similar to shining a harsh spotlight onto the backdrop.")),
    ("svg:soft-light", _("Soft Light"),
        _("Like shining a diffuse spotlight onto the backdrop.")),
    ("svg:difference", _("Difference"),
        _("Subtracts the darker color from the lighter of the two.")),
    ("svg:exclusion", _("Exclusion"),
        _("Similar to the 'Difference' mode, but lower in contrast.")),
    ("svg:hue", _("Hue"),
        _("Combines the hue of the top layer with the saturation and "
          "luminosity of the backdrop.")),
    ("svg:saturation", _("Saturation"),
        _("Applies the saturation of the top layer's colors to the hue and "
          "luminosity of the backdrop.")),
    ("svg:color", _("Color"),
        _("Applies the hue and saturation of the top layer to the luminosity "
          "of the backdrop.")),
    ("svg:luminosity", _("Luminosity"),
        _("Applies the luminosity of the top layer to the hue and saturation "
          "of the backdrop.")),
    ]

DEFAULT_COMPOSITE_OP = COMPOSITE_OPS[0][0]
VALID_COMPOSITE_OPS = set([n for n,d,s in COMPOSITE_OPS])


## Class defs


class Layer (object):
    """Surface-backed base layer implementation

    The base implementation is backed by a surface, and can be rendered by the
    main application. The actual content of the layer is held by the surface
    implementation. This is an internal detail that very few consumers should
    care about.
    """

    ## Initialization

    def __init__(self, name="", compositeop=DEFAULT_COMPOSITE_OP,
                 surface=None):
        """Construct a new Layer

        :param name: The name for the new layer.
        :param compositeop: Compositing operation to use.
        :param surface: Surface to use, overriding the default.

        If `surface` is specified, content observers will not be attached, and
        the layer will not be cleared during construction.
        """
        object.__init__(self)
        # Pluggable surface implementation
        # Only connect observers if using the default tiled surface
        if surface is None:
            self._surface = tiledsurface.Surface()
            self._surface.observers.append(self._notify_content_observers)
        else:
            self._surface = surface

        # Standard fields
        self.opacity = 1.0  #: Opacity of the layer (1 - alpha)
        self.name = name    #: The layer's name, for display
        self.visible = True #: Whether the layer is visible (forced opacity 0)
        self.locked = True  #: Whether the layer is locked (True by default)

        #: The compositing operation to use when displaying the layer
        self.compositeop = compositeop

        #: List of content observers (see _notify_content_observers())
        self.content_observers = []
        if surface is None:
            self.clear()


    def load_from_surface(self, surface):
        self._surface.load_from_surface(surface)


    def load_from_openraster(self, orazip, attrs, tempdir, feedback_cb):
        """Loads layer flags from XML attrs. Derived classes handle data.

        :param orazip: An OpenRaster zipfile, opened for extracting
        :param attrs: The XML attributes of the <layer/> tag.
        :param tempdir: A temporary working directory.
        :returns: True if the layer is marked as selected.
        :rtype: bool

        The base implementation does not attempt to load any surface image at
        all. That detail is left to the subclasses for now.
        """
        self.name = attrs.get('name', '')
        self.opacity = helpers.clamp(float(attrs.get('opacity', '1.0')),
                                     0.0, 1.0)
        self.compositeop = str(attrs.get('composite-op', DEFAULT_COMPOSITE_OP))
        if self.compositeop not in VALID_COMPOSITE_OPS:
            self.compositeop = DEFAULT_COMPOSITE_OP
        self.locked = helpers.xsd2bool(attrs.get("edit-locked", 'false'))
        self.visible = ('hidden' not in attrs.get('visibility', 'visible'))
        selected = helpers.xsd2bool(attrs.get("selected", 'false'))
        return selected


    def clear(self):
        """Clears the layer"""
        self._surface.clear()


    def _notify_content_observers(self, *args):
        """Notifies registered content observers

        Observer callbacks in `self.content_observers` are invoked via this
        method when the contents of the layer change, with the bounding box of
        the changed region (x, y, w, h).

        Only used for the default surface implmentation.
        """
        for func in self.content_observers:
            func(*args)

    @property
    def effective_opacity(self):
        """The opacity to use for rendering a layer: zero if invisible"""
        if self.visible:
            return self.opacity
        else:
            return 0.0

    def get_alpha(self, x, y, radius):
        """Gets the average alpha within a certain radius at a point

        Return value is not affected by the layer opacity, effective or
        otherwise. This is used by `Document.pick_layer()` and friends to test
        whether there's anything significant present at a particular point.
        """
        return self._surface.get_alpha(x, y, radius)


    def get_bbox(self):
        """Returns the inherent bounding box of the surface, tile aligned"""
        return self._surface.get_bbox()


    def is_empty(self):
        """Tests whether the surface is empty"""
        return self._surface.is_empty()


    def flood_fill(self, x, y, color, bbox, tolerance, dst_layer=None):
        """Fills a point on the surface with a colour

        :param x: Starting point X coordinate
        :param y: Starting point Y coordinate
        :param color: an RGB color
        :type color: tuple
        :param bbox: Bounding box: limits the fill
        :type bbox: lib.helpers.Rect or equivalent 4-tuple
        :param tolerance: how much filled pixels are permitted to vary
        :type tolerance: float [0.0, 1.0]
        :param dst_layer: Optional target layer (default is self!)
        :type dst_surface: Layer

        The `tolerance` parameter controls how much pixels are permitted to
        vary from the starting colour.  We use the 4D Euclidean distance from
        the starting point to each pixel under consideration as a metric,
        scaled so that its range lies between 0.0 and 1.0.

        The default target layer is `self`. This method invalidates the filled
        area of the target layer's surface, queueing a redraw if it is part of
        a visible document.
        """
        if dst_layer is None:
            dst_layer = self
        self._surface.flood_fill(x, y, color, bbox, tolerance,
                                 dst_surface=dst_layer._surface)


    ## Rendering


    def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0):
        """Composite one tile of the layer over a target array

        Composite one tile of the backing surface over the array dst, modifying
        only dst.
        """
        self._surface.composite_tile(
            dst, dst_has_alpha, tx, ty,
            mipmap_level=mipmap_level,
            opacity=self.effective_opacity,
            mode=self.compositeop
            )


    def render_as_pixbuf(self, *rect, **kwargs):
        """Renders this layer as a pixbuf

        :param *rect: rectangle to save, as a 4-tuple
        :param **kwargs: passed to pixbufsurface.render_as_pixbuf()
        :rtype: Gdk.Pixbuf
        """
        return self._surface.render_as_pixbuf(*rect, **kwargs)


    ## Translating


    def get_move(self, x, y):
        """Get a translation/move object for this layer

        :param x: Model X position of the start of the move
        :param y: Model X position of the start of the move
        :returns: A move object

        Subclasses should extend this base implementation to provide additional
        functionality for moving things other than the surface tiles around.
        """
        return self._surface.get_move(x, y)


    def translate(self, dx, dy):
        """Translate a layer non-interactively

        :param dx: Horizontal offset in model coordinates
        :param dy: Vertical offset in model coordinates

        This is implemented using `get_move()`.
        """
        move = self.get_move(0, 0)
        move.update(dx, dy)
        move.process(n=-1)
        move.cleanup()


    ## Pretty-printing


    def __repr__(self):
        x, y, w, h = self.get_bbox()
        l = self.locked and " locked" or ""
        v = self.visible and "" or " hidden"
        return ("<%s %r (%dx%d%+d%+d)%s%s %s %0.1f>"
                % (self.__class__.__name__, self.name, x, y, w, h, l, v,
                   self.compositeop, self.opacity))


    ## Saving


    def save_as_png(self, filename, *rect, **kwargs):
        """Save to a named PNG file

        :param filename: filename to save to
        :param *rect: rectangle to save, as a 4-tuple
        :param **kwargs: passed to pixbufsurface.save_as_png()
        :rtype: Gdk.Pixbuf
        """
        self._surface.save_as_png(filename, *rect, **kwargs)


    def _get_data_attrs(self, frame_bbox):
        """Internal: basic data attrs for OpenRaster saving"""
        fx, fy = frame_bbox[0:2]
        x, y, w, h = self.get_bbox()
        attrs = {}
        if self.name:
            attrs["name"] = str(self.name)
        attrs["x"] = str(x - fx)
        attrs["y"] = str(y - fy)
        attrs["opacity"] = str(self.opacity)
        if self.locked:
            attrs["edit-locked"] = "true"
        if self.visible:
            attrs["visibility"] = "visible"
        else:
            attrs["visibility"] = "hidden"
        compositeop = self.compositeop
        if compositeop not in VALID_COMPOSITE_OPS:
            compositeop = DEFAULT_COMPOSITE_OP
        attrs["composite-op"] = str(compositeop)
        return attrs


    def save_to_openraster(self, orazip, tmpdir, ref, selected,
                           canvas_bbox, frame_bbox, **kwargs):
        """Saves the layer's data into an open OpenRaster ZipFile

        :param orazip: a `zipfile.ZipFile` open for write
        :param tmpdir: path to a temp dir, removed after the save
        :param ref: Reference code for the layer, used for filenames
        :type ref: int or str
        :param selected: True if this layer is selected
        :param canvas_bbox: Bounding box of all tiles in all layers
        :param frame_bbox: Bounding box of the image being saved
        :param **kwargs: Keyword args used by the save implementation
        :returns: Attributes, for writing to the ``<layer/>`` record
        :rtype: dict, with string names and values

        There are three bounding boxes which need to considered. The inherent
        bbox of the layer as returned by `get_bbox()` is always tile aligned,
        as is `canvas_bbox`. The framing bbox, `frame_bbox`, is not tile
        aligned.

        All of the above bbox's coordinates are defined relative to the canvas
        origin. However, when saving, the data written must be translated so
        that `frame_bbox`'s top left corner defines the origin (0, 0), of the
        saved OpenRaster file. The width and height of `frame_bbox` determine
        the saved image's dimensions.

        More than one file may be written to the zipfile. The base
        implementation saves the surface only, as a PNG file.
        """
        rect = self.get_bbox()
        return self._save_rect_to_ora(orazip, tmpdir, ref, selected,
                                      canvas_bbox, frame_bbox, rect,
                                      **kwargs)

    def _save_rect_to_ora(self, orazip, tmpdir, ref, selected,
                          canvas_bbox, frame_bbox, rect, **kwargs):
        """Internal: saves a rectangle of the surface to an ORA zip"""
        # Write PNG data via a tempfile
        if type(ref) == int:
            pngname = "layer%03d.png" % (ref,)
        else:
            pngname = "%s.png" % (ref,)
        pngpath = os.path.join(tmpdir, pngname)
        t0 = time.time()
        self.save_as_png(pngpath, *rect, **kwargs)
        t1 = time.time()
        logger.debug('%.3fs surface saving %r', t1-t0, pngname)
        # Archive and remove
        storepath = "data/%s" % (pngname,)
        orazip.write(pngpath, storepath)
        os.remove(pngpath)
        # Return details
        attrs = self._get_data_attrs(frame_bbox)
        attrs["src"] = storepath
        return attrs


    ## Painting symmetry axis


    def set_symmetry_axis(self, center_x):
        """Sets the surface's painting symmetry axis

        :param center_x: Model X coordinate of the axis of symmetry. Set
               to None to remove the axis of symmetry
        :type x: `float` or `None`

        This is only useful for paintable layers.  Received strokes are
        reflected in the symmetry axis when it is set.
        """
        if center_x is None:
            self._surface.set_symmetry_state(False, 0.0)
        else:
            self._surface.set_symmetry_state(True, center_x)



class BackgroundLayer (Layer):
    """Background layer"""

    # NOTE: this could be generalized as a repeating tile for general use in
    # the layers stack, extending the ExternalLayer concept. Think textures!
    # Might need src-in compositing for that though.

    def __init__(self, bg):
        surface = tiledsurface.Background(bg)
        Layer.__init__(self, name="background", surface=surface)
        self.locked = False
        self.visible = True
        self.opacity = 1.0

    def set_surface(self, surface):
        """Sets the surface from a tiledsurface.Background"""
        assert isinstance(surface, tiledsurface.Background)
        self._surface = surface

    def save_to_openraster(self, orazip, tmpdir, ref, selected,
                           canvas_bbox, frame_bbox, **kwargs):
        # Normalize ref
        if type(ref) == int:
            ref = "background%03d"
        # Save as a regular layer for other apps.
        # Background surfaces repeat, so this will fit the frame.
        # XXX But we use the canvas bbox and always have. Why?
        # XXX - Presumably it's for origin alignment.
        # XXX - Inefficient for small frames.
        # XXX - I suspect rect should be redone with (w,h) granularity
        # XXX   and be based on the frame_bbox.
        rect = canvas_bbox
        attrs = Layer._save_rect_to_ora(self, orazip, tmpdir, ref, selected,
                                        canvas_bbox, frame_bbox, rect,
                                        **kwargs)
        # Also save as single pattern (with corrected origin)
        fx, fy = frame_bbox[0:2]
        x, y, w, h = self.get_bbox()
        rect = (x+fx, y+fy, w, h)

        pngname = '%s_tile.png' % (ref,)
        tmppath = os.path.join(tmpdir, pngname)
        t0 = time.time()
        self._surface.save_as_png(tmppath, *rect, **kwargs)
        t1 = time.time()
        storename = 'data/%s' % (pngname,)
        logger.debug('%.3fs surface saving %s', t1 - t0, storename)
        orazip.write(tmppath, storename)
        os.remove(tmppath)
        attrs['background_tile'] = storename
        return attrs



class ExternalLayer (Layer):
    """A layer which is stored as a tempfile in a non-MyPaint format

    External layers add the name of the tempfile to the base implementation.
    The internal surface is used to display a bitmap preview of the layer, but
    this cannot be edited.

    SVG files are the canonical example.
    """

    def __init__(self, name="", compositeop=DEFAULT_COMPOSITE_OP):
        """Construct, recording the filename, and its position"""
        Layer.__init__(self, name=name, compositeop=compositeop)
        self._filename = None
        self._tempdir = None
        self._x = None
        self._y = None
        self.locked = True


    ## Moving


    def get_move(self, x, y):
        """Start a new move for the external layer"""
        surface_move = Layer.get_move(self, x, y)
        return ExternalLayerMove(self, surface_move)


class ExternalLayerMove (object):
    """Move object wrapper for external layers"""

    def __init__(self, layer, surface_move):
        object.__init__(self)
        self._wrapped = surface_move
        self._layer = layer
        self._start_x = layer._x
        self._start_y = layer._y

    def update(self, dx, dy):
        self._layer._x = int(round(self._start_x + dx))
        self._layer._y = int(round(self._start_y + dy))
        return self._wrapped.update(dx, dy)

    def cleanup(self):
        return self._wrapped.cleanup()

    def process(self, n=200):
        return self._wrapped.process(n)



class PaintingLayer (Layer):
    """A paintable, bitmap layer

    Painting layers add a strokemap to the base implementation. The stroke map
    is a stack of `strokemap.StrokeShape` objects in painting order, allowing
    strokes and their associated brush and color information to be picked from
    the canvas.
    """


    ## Initializing


    def __init__(self, name="", compositeop=DEFAULT_COMPOSITE_OP):
        Layer.__init__(self, name=name, compositeop=compositeop)
        self.locked = False
        #: Stroke map.
        #: List of strokemap.StrokeShape instances (not stroke.Stroke), ordered
        #: by depth.
        self.strokes = []

 
    def clear(self):
        """Clear both the surface and the strokemap"""
        Layer.clear(self)
        self.strokes = []


    def load_from_surface(self, surface):
        Layer.load_from_surface(self, surface)
        self.strokes = []


    def load_from_openraster(self, orazip, attrs, tempdir, feedback_cb):
        # Load layer flags
        selected = Layer.load_from_openraster(self, orazip, attrs, tempdir,
                                              feedback_cb)
        # Read PNG content via tempdir
        src = attrs.get("src", None)
        src_basename, src_ext = os.path.splitext(src)
        src_basename = os.path.basename(src_basename)
        src_ext = src_ext.lower()
        x = int(attrs.get('x', 0))
        y = int(attrs.get('y', 0))
        t0 = time.time()
        if src_ext == '.png':
            orazip.extract(src, tempdir)
            tmp_filename = os.path.join(tempdir, src)
            surface = tiledsurface.Surface()
            surface.load_from_png(tmp_filename, x, y, feedback_cb)
            self.load_from_surface(surface)
            os.remove(tmp_filename)
        else:
            logger.error("Cannot load PaintingLayers from a %r", src_ext)
            raise NotImplementedError, "Only *.png is supported"
        t1 = time.time()
        logger.debug('%.3fs loading and converting src %r for %r',
                     t1 - t0, src_ext, src_basename)
        # Strokemap
        strokemap_name = attrs.get('mypaint_strokemap_v2', None)
        t2 = time.time()
        if strokemap_name is not None:
            sio = StringIO(orazip.read(strokemap_name))
            self.load_strokemap_from_file(sio, x, y)
            sio.close()
        t3 = time.time()
        logger.debug('%.3fs loading strokemap for %r',
                     t3 - t2, src_basename)
        # Return (TODO: notify needed here?)
        return selected

    ## Painting


    def stroke_to(self, brush, x, y, pressure, xtilt, ytilt, dtime):
        """Render a part of a stroke"""
        self._surface.begin_atomic()
        split = brush.stroke_to(self._surface.backend, x, y,
                                    pressure, xtilt, ytilt, dtime)
        self._surface.end_atomic()
        return split


    def add_stroke(self, stroke, snapshot_before):
        """Adds a stroke to the strokemap"""
        before = snapshot_before[1] # extract surface snapshot
        after  = self._surface.save_snapshot()
        shape = strokemap.StrokeShape()
        shape.init_from_snapshots(before, after)
        shape.brush_string = stroke.brush_settings
        self.strokes.append(shape)


    ## Snapshots


    def save_snapshot(self):
        return (self.strokes[:], self._surface.save_snapshot(), self.opacity)


    def load_snapshot(self, data):
        strokes, data, self.opacity = data
        self.strokes = strokes[:]
        self._surface.load_snapshot(data)


    ## Translating

    def get_move(self, x, y):
        """Get an interactive move object for the surface and its strokemap"""
        surface_move = Layer.get_move(self, x, y)
        return PaintingLayerMove(self, surface_move)


    ## Trimming


    def trim(self, rect):
        """Trim the layer to a rectangle, discarding data outside it

        :param rect: A trimming rectangle in model coordinates
        :type rect: tuple (x, y, w, h)

        Only complete tiles are discarded by this method.
        """
        self._surface.trim(rect)
        empty_strokes = []
        for stroke in self.strokes:
            if not stroke.trim(rect):
                empty_strokes.append(stroke)
        for stroke in empty_strokes:
            logger.debug("Removing emptied stroke %r", stroke)
            self.strokes.remove(stroke)

    ## Strokemap


    def load_strokemap_from_file(self, f, translate_x, translate_y):
        assert not self.strokes
        brushes = []
        N = tiledsurface.N
        x = int(translate_x//N) * N
        y = int(translate_y//N) * N
        dx = translate_x % N
        dy = translate_y % N
        while True:
            t = f.read(1)
            if t == 'b':
                length, = struct.unpack('>I', f.read(4))
                tmp = f.read(length)
                brushes.append(zlib.decompress(tmp))
            elif t == 's':
                brush_id, length = struct.unpack('>II', f.read(2*4))
                stroke = strokemap.StrokeShape()
                tmp = f.read(length)
                stroke.init_from_string(tmp, x, y)
                stroke.brush_string = brushes[brush_id]
                # Translate non-aligned strokes
                if (dx, dy) != (0, 0):
                    stroke.translate(dx, dy)
                self.strokes.append(stroke)
            elif t == '}':
                break
            else:
                assert False, 'invalid strokemap'


    def get_stroke_info_at(self, x, y):
        x, y = int(x), int(y)
        for s in reversed(self.strokes):
            if s.touches_pixel(x, y):
                return s


    def get_last_stroke_info(self):
        if not self.strokes:
            return None
        return self.strokes[-1]


    ## Layer merging


    def merge_into(self, dst, strokemap=True):
        """Merge this layer into another, modifying only the target

        :param dst: The target layer
        :param strokemap: Set to false to ignore the layers' strokemaps.

        The target layer must always have an alpha channel. After this
        operation, the target layer's opacity is set to 1.0 and it is made
        visible.
        """
        # Flood-fill uses this for its newly created and working layers,
        # but it should not construct a strokemap for what it does.
        if strokemap:
            dst.strokes.extend(self.strokes)
        # Normalize the target layer's effective opacity to 1.0 without
        # changing its appearance
        if dst.effective_opacity < 1.0:
            for tx, ty in dst._surface.get_tiles():
                with dst._surface.tile_request(tx, ty, readonly=False) as surf:
                    surf[:,:,:] = dst.effective_opacity * surf[:,:,:]
            dst.opacity = 1.0
            dst.visible = True
        # We must respect layer visibility, because saving a
        # transparent PNG just calls this function for each layer.
        src = self
        for tx, ty in src._surface.get_tiles():
            with dst._surface.tile_request(tx, ty, readonly=False) as surf:
                src._surface.composite_tile(surf, True, tx, ty,
                    opacity=self.effective_opacity,
                    mode=self.compositeop)


    def convert_to_normal_mode(self, get_bg):
        """Convert pixels to permit compositing with Normal mode

        Given a background, this layer is updated such that it can be
        composited over the background in normal blending mode. The
        result will look as if it were composited with the current
        blending mode.
        """
        if self.compositeop ==  "svg:src-over" and self.effective_opacity == 1.0:
            return # optimization for merging layers

        N = tiledsurface.N
        tmp = empty((N, N, 4), dtype='uint16')
        for tx, ty in self._surface.get_tiles():
            bg = get_bg(tx, ty)
            # tmp = bg + layer (composited with its mode)
            mypaintlib.tile_copy_rgba16_into_rgba16(bg, tmp)
            self.composite_tile(tmp, False, tx, ty)
            # overwrite layer data with composited result
            with self._surface.tile_request(tx, ty, readonly=False) as dst:

                mypaintlib.tile_copy_rgba16_into_rgba16(tmp, dst)
                dst[:,:,3] = 0 # minimize alpha (throw away the original alpha)

                # recalculate layer in normal mode
                mypaintlib.tile_flat2rgba(dst, bg)





    ## Saving


    @staticmethod
    def _write_file_str(z, filename, data):
        """Helper: write data to a zipfile with the right permissions"""
        # Work around a permission bug in the zipfile library:
        # http://bugs.python.org/issue3394
        zi = zipfile.ZipInfo(filename)
        zi.external_attr = 0100644 << 16
        z.writestr(zi, data)

    def _save_strokemap_to_file(self, f, translate_x, translate_y):
        brush2id = {}
        for stroke in self.strokes:
            s = stroke.brush_string
            # save brush (if not already known)
            if s not in brush2id:
                brush2id[s] = len(brush2id)
                s = zlib.compress(s)
                f.write('b')
                f.write(struct.pack('>I', len(s)))
                f.write(s)
            # save stroke
            s = stroke.save_to_string(translate_x, translate_y)
            f.write('s')
            f.write(struct.pack('>II', brush2id[stroke.brush_string], len(s)))
            f.write(s)
        f.write('}')


    def save_to_openraster(self, orazip, tmpdir, ref, selected,
                           canvas_bbox, frame_bbox, **kwargs):
        """Save the strokemap too, in addition to the base implementation"""
        # Normalize refs to strings
        # The *_strokemap.dat should agree with the layer name
        if type(ref) == int:
            ref = "layer%03d" % (ref,)
        # Save the layer normally
        attrs = Layer.save_to_openraster(self, orazip, tmpdir, ref, selected,
                                         canvas_bbox, frame_bbox, **kwargs)
        # Store stroke shape data too
        x, y, w, h = self.get_bbox()
        sio = StringIO()
        t0 = time.time()
        self._save_strokemap_to_file(sio, -x, -y)
        t1 = time.time()
        data = sio.getvalue()
        sio.close()
        datname = '%s_strokemap.dat' % (ref,)
        logger.debug("%.3fs strokemap saving %r", t1-t0, datname)
        storepath = "data/%s" % (datname,)
        self._write_file_str(orazip, storepath, data)
        # Return details
        attrs['mypaint_strokemap_v2'] = storepath
        return attrs

class PaintingLayerMove (object):
    """Move object wrapper for painting layers"""

    def __init__(self, layer, surface_move):
        object.__init__(self)
        self._wrapped = surface_move
        self._layer = layer
        self._final_dx = 0
        self._final_dy = 0

    def update(self, dx, dy):
        self._final_dx = dx
        self._final_dy = dy
        return self._wrapped.update(dx, dy)

    def cleanup(self):
        result = self._wrapped.cleanup()
        dx = self._final_dx
        dy = self._final_dy
        # Arrange for the strokemap to be moved too;
        # this happens in its own background idler.
        for stroke in self._layer.strokes:
            stroke.translate(dx, dy)
            # Minor problem: huge strokemaps take a long time to move, and the
            # translate must be forced to completion before drawing or any
            # further layer moves. This can cause apparent hangs for no
            # reason later on. Perhaps it would be better to process them
            # fully in this hourglass-cursor phase after all?

    def process(self, n=200):
        return self._wrapped.process(n)

