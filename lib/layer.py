# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

import gi
from gi.repository import GdkPixbuf

import struct
import zlib
import numpy
from numpy import *
import logging
import os
from cStringIO import StringIO
import time
import zipfile
logger = logging.getLogger(__name__)
import tempfile
import shutil

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

LOAD_CHUNK_SIZE = 64*1024


## Class defs

## Basic interface for a renderable layer & docs


class LayerBase (object):
    """Base class defining the layer API

    Layers support two similar tile-based methods which are used for two
    distinct rendering cases: _blitting_ (unconditional copying without flags)
    and _compositing_ (conditional alpha-compositing which respects flags like
    opacity and layer mode). Rendering for the display is supported using the
    compositing pathway and is coordinated via the `RootLayerStack`. Exporting
    layers is handled via the blitting pathway, which for layer stacks
    involves compositing the stacks' contents together to render an effective
    image.

    """

    ICON_NAME = None

    ## Construction, loading, other lifecycle stuff

    def __init__(self, name="", compositeop=DEFAULT_COMPOSITE_OP):
        """Construct a new layer

        :param name: The name for the new layer.
        :param compositeop: Compositing operation to use.
        :param **kwargs: Ignored.
        """
        super(LayerBase, self).__init__()
        #: Opacity of the layer (1 - alpha)
        self.opacity = 1.0
        #: The layer's name, for display purposes.
        self.name = name
        #: Whether the layer is visible (forced opacity 0 when invisible)
        self.visible = True
        #: Whether the layer is locked (locked layers cannot be changed)
        self.locked = False
        #: The compositing operation to use when displaying the layer
        self.compositeop = compositeop


    def copy(self):
        """Returns an independent copy of the layer, for Duplicate Layer

        Everything about the returned layer must be a completely independent
        copy of the original data. If the layer can be worked on, working on it
        must leave the original layer unaffected.

        This base class implementation can be reused/extended by subclasses if
        they support zero-argument construction. This implementation uses the
        `save_snapshot()` and `load_snapshot()` methods.
        """
        layer = self.__class__()
        layer.name = self.name
        layer.compositeop = self.compositeop
        layer.opacity = self.opacity
        layer.visible = self.visible
        layer.locked = self.locked
        layer.load_snapshot(self.save_snapshot())
        return layer


    def load_from_openraster(self, orazip, attrs, tempdir, feedback_cb):
        """Loads layer flags from XML attrs.

        :param orazip: An OpenRaster zipfile, opened for extracting
        :param attrs: The XML attributes of the <layer/> tag.
        :param tempdir: A temporary working directory.
        :returns: True if the layer is marked as selected.
        :rtype: bool
        """
        return False


    def clear(self):
        """Clears the layer"""
        pass


    ## Info methods

    @property
    def effective_opacity(self):
        """The opacity used when compositing a layer: zero if invisible

        This must match the appearance given by `composite_tile()` when it is
        called with no `layers` list, even if that method uses other means to
        determine how or whether to write its output. The base class's
        effective opacity is zero because the base `composite_tile()` does
        nothing.
        """
        return 0.0

    def get_alpha(self, x, y, radius):
        """Gets the average alpha within a certain radius at a point

        :param x: model X coordinate
        :param y: model Y coordinate
        :param radius: radius over which to average
        :rtype: float

        The return value is not affected by the layer opacity, effective or
        otherwise. This is used by `Document.pick_layer()` and friends to test
        whether there's anything significant present at a particular point.
        The default alpha at a point is zero.
        """
        return 0.0

    def get_bbox(self):
        """Returns the inherent bounding box of the surface, tile aligned

        :rtype: lib.helpers.Rect

        Just a default (zero-size) rect in the base implementation.
        """
        return helpers.Rect()

    def is_empty(self):
        """Tests whether the surface is empty

        Always true in the base implementation.
        """
        return True

    def get_paintable(self):
        """True if this layer currently accepts painting brushstrokes

        Always false in the base implementation.
        """
        return False

    def get_fillable(self):
        """True if this layer currently accepts flood fill

        Always false in the base implementation.
        """
        return False

    def get_stroke_info_at(self, x, y):
        """Return the brushstroke at a given point

        :param x: X coordinate to pick from, in model space.
        :param y: Y coordinate to pick from, in model space.
        :rtype: lib.strokemap.StrokeShape or None

        Returns None for the base class.
        """
        return None

    def get_last_stroke_info(self):
        """Return the most recently painted stroke

        :rtype lib.strokemap.StrokeShape or None

        Returns None for the base class.
        """
        return None


    ## Flood fill

    def flood_fill(self, x, y, color, bbox, tolerance, dst_layer=None):
        """Fills a point on the surface with a colour

        See `PaintingLayer.flood_fill() for parameters and semantics.
        """
        raise NotImplementedError


    ## Rendering


    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       **kwargs):
        """Unconditionally copy one tile's data into an array without options

        The visibility, opacity, and compositing mode flags of this layer must
        be ignored, or take default values. If the layer has sub-layers, they
        must be composited together as an isolated group (i.e. over an empty
        backdrop) using their `composite_tile()` method. It is the result of
        this compositing which is blitted, ignoring this layer's visibility,
        opacity, and compositing mode flags.

        :param dst: Target tile array (uint16, NxNx4, 15-bit scaled ints)
        :type dst: numpy.ndarray
        :param dst_has_alpha: the alpha channel in dst should be preserved
        :type dst_has_alpha: bool
        :param tx: Tile X coordinate, in model tile space
        :type tx: int
        :param ty: Tile Y coordinate, in model tile space
        :type ty: int
        :param mipmap_level: layer mipmap level to use
        :type mipmap_level: int
        :param **kwargs: extensibility...

        The base implementation does nothing.
        """
        pass


    def composite_tile( self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                        layers=None, **kwargs ):
        """Composite a tile's data into an array, respecting flags/layers list

        Unlike `blit_tile_into()`, the visibility, opacity, and compositing
        mode flags of this layer must be respected.  It otherwise works just
        like `blit_tile_into()`, but may make a local decision about whether
        to render as an isolated group.  This method uses the same parameters
        as `blit_tile_into()`, with one addition:

        :param layers: the set of layers to render
        :type layers: set of layers, or None

        If `layers` is defined, it identifies the layers which are to be
        rendered: certain special rendering modes require this. For layers
        other then the root stack, layers should not render themselves if
        omitted from a defined `layers`.

        The base implementation does nothing.
        """
        pass


    def render_as_pixbuf(self, *rect, **kwargs):
        """Renders this layer as a pixbuf

        :param *rect: rectangle to save, as a 4-tuple
        :param **kwargs: passed to pixbufsurface.render_as_pixbuf()
        :rtype: Gdk.Pixbuf
        """
        raise NotImplementedError


    ## Translation

    def get_move(self, x, y):
        """Get a translation/move object for this layer

        :param x: Model X position of the start of the move
        :param y: Model X position of the start of the move
        :returns: A move object
        """
        raise NotImplementedError


    def translate(self, dx, dy):
        """Translate a layer non-interactively

        :param dx: Horizontal offset in model coordinates
        :param dy: Vertical offset in model coordinates

        The base implementation uses `get_move()` and the object it returns.
        """
        move = self.get_move(0, 0)
        move.update(dx, dy)
        move.process(n=-1)
        move.cleanup()


    ## Standard stuff

    def __repr__(self):
        if self.name:
            return "<%s %r>" % (self.__class__.__name__, self.name)
        else:
            return "<%s>" % (self.__class__.__name__)

    def __nonzero__(self):
        return True

    def __eq__(self, layer):
        return self is layer


    ## Layer merging

    def merge_into(self, dst, **kwargs):
        """Merge this layer into another, modifying only the destination

        :param dst: The destination layer
        :param **kwargs: Ignored

        The base implementation does nothing.
        """
        pass


    def convert_to_normal_mode(self, get_bg):
        """Convert pixels to permit compositing with Normal mode

        :param get_bg: Callable accepting `tx, ty` params.

        Given a background, layer should be updated such that it can be
        composited over the background in normal blending mode. The result is
        intended to look as if it were composited with the current blending
        mode.

        The base implementation does nothing.
        """
        pass


    ## Saving


    def save_as_png(self, filename, *rect, **kwargs):
        """Save to a named PNG file

        :param filename: filename to save to
        :param *rect: rectangle to save, as a 4-tuple
        :param **kwargs: passed to pixbufsurface.save_as_png()
        :rtype: Gdk.Pixbuf

        The base implementation does nothing.
        """
        pass


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

        More than one file may be written to the zipfile.
        """
        raise NotImplementedError


    ## Painting symmetry axis


    def set_symmetry_axis(self, center_x):
        """Sets the surface's painting symmetry axis

        :param center_x: Model X coordinate of the axis of symmetry. Set
               to None to remove the axis of symmetry
        :type x: `float` or `None`

        This is only useful for paintable layers. Received strokes are
        reflected in the symmetry axis when it is set.

        the base implementation does nothing.
        """
        pass


    ## Snapshot


    def save_snapshot(self):
        """Snapshots the state of the layer, for undo purposes

        The returned data should be considered opaque, useful only as a
        memento to be restored with load_snapshot().
        """
        return _LayerBaseSnapshot(self)


    def load_snapshot(self, sshot):
        """Restores the layer from snapshot data"""
        sshot.restore_to_layer(self)


    ## Trimming


    def trim(self, rect):
        """Trim the layer to a rectangle, discarding data outside it

        :param rect: A trimming rectangle in model coordinates
        :type rect: tuple (x, y, w, h)

        The base implementation does nothing.
        """
        pass


class _LayerBaseSnapshot (object):
    """Base snapshot implementation

    Snapshots are stored in commands, and used to implement undo and redo.
    They must be independent copies of the data, although copy-on-write
    semantics are fine. Snapshot objects don't have to be _full and exact_
    clones of the layer's data, but they do need to capture _inherent_
    qualities of the layer. Mere metadata can be ignored. For the base
    layer implementation, this means only the layer's opacity.
    """

    def __init__(self, layer):
        super(_LayerBaseSnapshot, self).__init__()
        self.opacity = layer.opacity

    def restore_to_layer(self, layer):
        layer.opacity = self.opacity


## Stacks of layers


class LayerStack (LayerBase):
    """Reorderable stack of editable layers"""

    ICON_NAME = "mypaint-tool-layers"

    ## Construction and other lifecycle stuff


    def __init__(self, name='', compositeop=DEFAULT_COMPOSITE_OP):
        super(LayerStack, self).__init__(name=name, compositeop=compositeop)
        self._layers = []
        #: Explicit isolation flag
        self.isolated = False
        # Blank background, for use in rendering
        N = tiledsurface.N
        blank_arr = numpy.zeros((N, N, 4), dtype='uint16')
        self._blank_bg_surface = tiledsurface.Background(blank_arr)


    def copy(self):
        raise NotImplementedError


    def load_from_openraster(self, orazip, attrs, tempdir, feedback_cb):
        raise NotImplementedError


    def clear(self):
        super(LayerStack, self).clear()
        self._layers = []


    def __repr__(self):
        """String representation of a stack

        >>> repr(LayerStack(name='test'))
        "<LayerStack 'test' []>"
        """
        if self.name:
            return '<%s %r %r>' % (self.__class__.__name__, self.name,
                                   self._layers)
        else:
            return '<%s %r>' % (self.__class__.__name__, self._layers)


    ## Basic list-of-layers access

    def __len__(self):
        """Return the number of layers in the stack

        >>> stack = LayerStack()
        >>> len(stack)
        0
        >>> stack.append(LayerBase())
        >>> len(stack)
        1
        """
        return len(self._layers)


    def __iter__(self):
        return iter(self._layers)


    def append(self, layer):
        self._layers.append(layer)

    def remove(self, layer):
        return self._layers.remove(layer)

    def pop(self, index=None):
        if index is None:
            return self._layers.pop()
        else:
            return self._layers.pop(index)

    def insert(self, index, layer):
        self._layers.insert(index, layer)


    ## Info methods


    def get_effective_isolation(self):
        """True if the layer should be rendered as isolated"""
        return (self.isolated or self.opacity != 1.0 or
                self.compositeop != DEFAULT_COMPOSITE_OP)

    def get_bbox(self):
        result = helpers.Rect()
        for layer in self._layers:
            result.expandToIncludeRect(layer.get_bbox())
        return result

    def is_empty(self):
        return len(self._layers) == 0


    ## Rendering

    def blit_tile_into( self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                        **kwargs ):
        """Unconditionally copy one tile's data into an array"""
        raise NotImplementedError


    def composite_tile( self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                        layers=None, **kwargs):
        """Composite a tile's data into an array, respecting flags/layers list"""
        if layers and self in layers:
            layers.update(self._layers)   # blink all child layers too
            # But carry on when this layer is not in `layers`. Not doing that
            # would prevent child layers being blinked.
        elif not self.visible:
            return
        N = tiledsurface.N
        tmp = numpy.zeros((N, N, 4), dtype='uint16')
        for layer in self._layers:
            layer.composite_tile(tmp, True, tx, ty, mipmap_level,
                                 layers=layers, **kwargs)
        func = tiledsurface.SVG2COMPOSITE_FUNC[self.compositeop]
        func(tmp, dst, dst_has_alpha, self.opacity)


    def get_move(self, x, y):
        """Get a translation/move object for this layer"""
        return LayerStackMove(self, self._layers, x, y)


    ## Saving


    def save_as_png(self, filename, *rect, **kwargs):
        """Save to a named PNG file"""
        raise NotImplementedError


    def save_to_openraster(self, orazip, tmpdir, ref, selected,
                           canvas_bbox, frame_bbox, **kwargs):
        """Saves the layer's data into an open OpenRaster ZipFile"""
        raise NotImplementedError


    ## Snapshotting

    def save_snapshot(self):
        """Snapshots the state of the layer, for undo purposes"""
        return _LayerBaseSnapshot(self, self._layers)


    ## Trimming

    def trim(self, rect):
        """Trim the layer to a rectangle, discarding data outside it"""
        for layer in self._layers:
            layer.trim(rect)



class _LayerStackSnapshot (_LayerBaseSnapshot):

    def __init__(self, layer, layers):
        super(_LayerStackSnapshot, self).__init__(layer)
        raise NotImplementedError

    def restore_to_layer(self, layer):
        super(_LayerStackSnapshot, self).restore_to_layer(layer)
        raise NotImplementedError


class LayerStackMove (object):
    """Move object wrapper for layer stacks"""

    def __init__(self, layers, x, y):
        super(LayerStackMove, self).__init__(x, y)
        self._moves = []
        for layer in layers:
            self._moves.append(layer.get_move(x, y))

    def update(self, dx, dy):
        for move in self._moves:
            move.update(dx, dy)

    def cleanup(self):
        for move in self._moves:
            move.cleanup(dx, dy)

    def process(self, n=200):
        n = max(20, int(n / len(self._moves)))
        incomplete = False
        for move in self._moves:
            incomplete = move.process(n=n) or incomplete
        return incomplete


class RootLayerStack (LayerStack):
    """Layer stack with background, rendering loop, selection, & view modes"""

    ## Initialization


    def __init__(self, doc):
        """Construct, as part of a model

        :param doc: The model document. May be None for testing.
        :param doc: lib.document.Document
        """
        super(RootLayerStack, self).__init__(compositeop=DEFAULT_COMPOSITE_OP)
        self._doc = doc
        # Background
        self._default_background = (255, 255, 255)
        self._background_layer = BackgroundLayer(self._default_background)
        self._background_visible = True
        # Special rendering state
        self._current_layer_solo = False
        self._current_layer_previewing = False
        # Current layer
        self._current_path = ()


    def clear(self):
        super(RootLayerStack, self).clear()
        self.set_background(self._default_background)


    ## Rendering: root stack API


    def get_render_background(self):
        """True if the internal background will be rendered by render_into()

        The UI should draw its own checquered background if the background is
        not going to be rendered by the root stack, and expect `render_into()`
        to write RGBA data with lots of transparent areas.
        """

        # Layer-solo mode should probably *not* render without the background.
        # While it's intended to be used for showing what a layer contains by
        # itself, part of that involves showing what effect the the layer's
        # mode has. Layer-solo over real alpha checks doesn't permit that.
        # Users can always turn background visibility on or off with the UI if
        # they wish to preview it both ways, however.

        return self._background_visible and not self._current_layer_previewing

        # Conversely, current-layer-preview is intended to *blink* very
        # visibly to notify the user, so always turn off the background for
        # that.


    def get_render_layers(self, implicit=False):
        """Get the set of layers to be rendered as used by render_into()

        :param implicit: if true, return None if visible flags should be used
        :type implicit: bool
        :return: The set of layers which render_into() would use
        :rtype: set or None

        Implicit mode is used internally by render_into(). If it is enabled,
        this method returns ``None`` if each descendent layer's ``visible``
        flag is to be used to determine this.  When disabled, the flag is
        tested here, which requires an extra iteration.
        """
        if self._current_layer_previewing or self._current_layer_solo:
            return set([self.get_current()])
        elif implicit:
            return None
        else:
            return set((d for d in self.deepiter() if d.visible))


    def render_into(self, surface, tiles, mipmap_level, overlay=None):
        """Tiled rendering: used for display only

        :param surface: target rgba8 surface
        :type surface: lib.pixbufsurface.Surface
        :param tiles: tile coords, (tx, ty), to render
        :type tiles: list
        :param mipmap_level: layer and surface mipmap level to use
        :type mipmap_level: int
        :param overlay: overlay layer to render (stroke highlighting)
        :type overlay: SurfaceBackedLayer

        """
        # Decide a rendering mode
        background = None
        dst_has_alpha = False
        if not self.get_render_background():
            background = self._blank_bg_surface
            dst_has_alpha = True
        # TODO: Inject the overlay layer if we have one
        layers = self.get_render_layers(implicit=True)
        # Blit loop. Could this be done in C++?
        for tx, ty in tiles:
            with surface.tile_request(tx, ty, readonly=False) as dst:
                self.composite_tile( dst, dst_has_alpha, tx, ty, mipmap_level,
                                     layers=layers, background=background,
                                     overlay=overlay )

    ## Rendering


    def blit_tile_into( self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                        **kwargs ):
        """Unconditionally copy one tile's data into an array

        The root layer stack implementation just uses `composite_tile()` due
        to its lack of conditionality.
        """
        # NOTE: The background is always on when blitting;
        # NOTE:   - does this matter?
        # NOTE:   - should it be always-off?
        self.composite_tile( dst, dst_has_alpha, tx, ty,
                             mipmap_level=mipmap_level, **kwargs )


    def composite_tile( self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                        layers=None, background=None, overlay=None,
                        **kwargs ):
        """Composite a tile's data into an array, respecting flags/layers list

        The root layer stack implementation accepts the parameters documented
        in `BaseLayer.composite_tile()`, and also consumes:

        :param background: Surface supporting 15-bit scaled-int tile blits
        :param overlay: Layer supporting 15-bit scaled-int tile composition
        :type overlay: BaseLayer

        The root layer has flags which ensure it is always visible, so the
        result is generally indistinguishable from `blit_tile_into()`. However
        the rendering loop, `render_into()`, calls this method and sometimes
        passes in a zero-alpha `background` for special rendering modes which
        need isolated rendering.

        As a further extension to the base API, `dst` may be an 8bpp array. A
        temporary 15-bit scaled int array is used for compositing in this
        case, and the output is converted to 8bpp.
        """
        if background is None:
            background = self._background_layer._surface

        assert dst.shape[-1] == 4
        if dst.dtype == 'uint8':
            dst_8bit = dst
            N = tiledsurface.N
            dst = numpy.empty((N, N, 4), dtype='uint16')
        else:
            dst_8bit = None

        background.blit_tile_into(dst, dst_has_alpha, tx, ty, mipmap_level)

        for layer in self._layers:
            layer.composite_tile(dst, dst_has_alpha, tx, ty, mipmap_level,
                                 layers=layers, **kwargs)
        if overlay:
            overlay.composite_tile(dst, dst_has_alpha, tx, ty, mipmap_level,
                                   layers=set([overlay]), **kwargs)


        if dst_8bit is not None:
            if dst_has_alpha:
                mypaintlib.tile_convert_rgba16_to_rgba8(dst, dst_8bit)
            else:
                mypaintlib.tile_convert_rgbu16_to_rgbu8(dst, dst_8bit)




    ## Current layer

    def get_current_path(self):
        return self._current_path

    def set_current_path(self, path):
        p = tuple(path)
        while len(p) > 0:
            layer = self.deepget(p)
            if layer is not None:
                self._current_path = p
                return
            p = p[:-1]
        if len(self._layers) > 0:
            self._current_path = (0,)
        else:
            raise ValueError, 'Invalid path %r' % (path,)

    current_path = property(get_current_path, set_current_path)

    def get_current(self):
        layer = self.deepget(self._current_path)
        assert layer is not self
        assert layer is not None
        return layer

    current = property(get_current)


    ## The background layer

    @property
    def background_layer(self):
        """The background layer (accessor)"""
        return self._background_layer

    def set_background(self, obj, make_default=False):
        """Set the background layer's surface from an object

        :param obj: Background layer, or an RGB triple (uint8), or a
           HxWx4 or HxWx3 numpy array which can be either uint8 or uint16.
        :type obj: layer.BackgroundLayer or tuple or numpy array
        :param make_default: Whether to set the default background for
          clear() too.
        :type make_default: bool
        """
        if isinstance(obj, BackgroundLayer):
            obj = obj._surface
        if not isinstance(obj, tiledsurface.Background):
            if isinstance(obj, GdkPixbuf.Pixbuf):
                obj = helpers.gdkpixbuf2numpy(obj)
            obj = tiledsurface.Background(obj)
        self._background_layer.set_surface(obj)
        if make_default:
            self._default_background = obj
        if not self._background_visible:
            self._background_visible = True
            if self._doc:
                self._doc.call_doc_observers()
        if self._doc:
            self._doc.invalidate_all()


    def set_background_visible(self, value):
        """Sets whether the background is visible"""
        value = bool(value)
        old_value = self._background_visible
        self._background_visible = value
        if value != old_value and self._doc:
            self._doc.call_doc_observers()
            self._doc.invalidate_all()


    def get_background_visible(self):
        """Gets whether the background is visible"""
        return bool(self._background_visible)

    ## Layer Solo toggle (not saved)

    def get_current_layer_solo(self):
        """Layer-solo state for the document"""
        return self._current_layer_solo

    def set_current_layer_solo(self, value):
        """Layer-solo state for the document"""
        # TODO: use the user_initiated hack to make this undoable
        value = bool(value)
        old_value = self._current_layer_solo
        self._current_layer_solo = value
        if value != old_value:
            self._doc.call_doc_observers()
            self._doc.invalidate_all()


    ## Current layer temporary previewing state (not saved, used for blinking)


    def get_current_layer_previewing(self):
        """Layer-previewing state, as used when blinking a layer"""
        return self._current_layer_previewing


    def set_current_layer_previewing(self, value):
        """Layer-previewing state, as used when blinking a layer"""
        value = bool(value)
        old_value = self._current_layer_previewing
        self._current_layer_previewing = value
        if value != old_value:
            self._doc.call_doc_observers()
            self._doc.invalidate_all()


    ## Layer path manipulation


    def path_above(self, path): #FIXME: support walking.
        """Return the path for the layer stacked above a given path

        :param path: a layer path
        :type path: list or tuple
        :return: the layer above `path` in stacking order
        :rtype: tuple

        >>> root, leaves = _make_test_stack()
        >>> root.path_above([0, 0])
        (0, 1)
        >>> root.path_above([0, 1])
        >>> root.path_above([0])
        (1,)
        >>> root.path_above([1])

        """
        if len(path) == 0:
            return None
        layer = self.deepget(path)
        if layer is None:
            return None
        parent = self
        parent_path = ()
        if len(path) > 1:
            parent_path = path[:-1]
            parent = self.deepget(parent_path)
        idx = path[-1] + 1
        if idx >= len(parent._layers):
            return None
        return tuple(list(parent_path) + [idx])


    def path_below(self, path): #FIXME: support walking.
        """Return the path for the layer stacked below a given path

        :param path: a layer path
        :type path: list or tuple
        :return: the layer above `path` in stacking order
        :rtype: tuple or None

        >>> root, leaves = _make_test_stack()
        >>> root.path_below([0, 1])
        (0, 0)
        >>> root.path_below([0, 0])
        >>> root.path_below([1])
        (0,)
        >>> root.path_below((0,))
        """
        if len(path) == 0:
            return None
        layer = self.deepget(path)
        if layer is None:
            return None
        parent = self
        parent_path = ()
        if len(path) > 1:
            parent_path = path[:-1]
            parent = self.deepget(parent_path)
        idx = path[-1] - 1
        if idx < 0:
            return None
        return tuple(list(parent_path) + [idx])

    ## Simplified tree storage and access

    # We use a path concept that's similar to GtkTreePath's, but almost like a
    # key/value store if this is the root layer stack.

    def deepiter(self):
        """Iterates across all descendents of the stack

        >>> stack, leaves = _make_test_stack()
        >>> len(list(stack.deepiter()))
        6
        >>> len(set(stack.deepiter())) == len(list(stack.deepiter())) # no dups
        True
        >>> stack not in stack.deepiter()
        True
        >>> [] not in stack.deepiter()
        True
        >>> leaves[0] in stack.deepiter()
        True
        """
        queue = [self]
        while len(queue) > 0:
            layer = queue.pop(0)
            if layer is not self:
                yield layer
            if isinstance(layer, LayerStack):
                for child in reversed(layer._layers):
                    queue.insert(0, child)


    def deepenumerate(self):
        """Enumerates the structure of a stack in depth

        >>> stack, leaves = _make_test_stack()
        >>> [a[0] for a in stack.deepenumerate()]
        [(0,), (0, 0), (0, 1), (1,), (1, 0), (1, 1)]
        >>> set(leaves) - set([a[1] for a in stack.deepenumerate()])
        set([])
        """
        queue = [([], self)]
        while len(queue) > 0:
            path, layer = queue.pop(0)
            if layer is not self:
                yield (tuple(path), layer)
            if isinstance(layer, LayerStack):
                for i, child in enumerate(layer._layers):
                    queue.insert(i, (path + [i], child))


    def deepget(self, path, default=None):
        """Gets a layer based on its path

        >>> stack, leaves = _make_test_stack()
        >>> stack.deepget(()) is stack
        True
        >>> stack.deepget((0,1))
        <PaintingLayer '01'>
        >>> stack.deepget((0,))
        <LayerStack '0' [<PaintingLayer '00'>, ...]>
        >>> stack.deepget((0,11), "missing")
        'missing'

        """
        if len(path) == 0:
            return self
        unused_path = list(path)
        layer = self
        while len(unused_path) > 0:
            idx = unused_path.pop(0)
            if abs(idx) > len(layer._layers)-1:
                return default
            layer = layer._layers[idx]
            if unused_path:
                if not isinstance(layer, LayerStack):
                    return default
            else:
                return layer
        return default


    def deepinsert(self, path, layer):
        """Inserts a layer before the final index in path

        :param path: an insertion path: see below
        :type path: iterable of integers
        :param layer: the layer to insert
        :type layer: LayerBase

        Deepinsert cannot create sub-stacks. Every element of `path` before
        the final element must be a valid `list`-style ``[]`` index into an
        existing stack along the chain being addressed, starting with the
        root.  The final element may be any index which `list.insert()`
        accepts.  Negative final indices, and final indices greater than the
        number of layers in the addressed stack are quite valid in `path`.

        >>> stack, leaves = _make_test_stack()
        >>> layer = PaintingLayer('foo')
        >>> stack.deepinsert((0,2), layer)
        >>> stack.deepget((0,-1)) is layer
        True
        >>> stack = RootLayerStack(doc=None)
        >>> layer = PaintingLayer('bar')
        >>> stack.deepinsert([0], layer)
        >>> stack.deepget([0]) is layer
        True
        """
        if len(path) == 0:
            raise IndexError, 'Cannot insert after the root'
        unused_path = list(path)
        stack = self
        while len(unused_path) > 0:
            idx = unused_path.pop(0)
            if not isinstance(stack, LayerStack):
                raise IndexError, ("All nonfinal elements of %r must "
                                   "identify a stack" % (path,))
            if unused_path:
                stack = stack._layers[idx]
            else:
                stack.insert(idx, layer)
                return
        assert (len(unused_path) > 0), ("deepinsert() should never exhaust "
                                        "the path")


    def deeppop(self, path):
        """Removes a layer by its path

        >>> stack, leaves = _make_test_stack()
        >>> stack.deeppop(())
        Traceback (most recent call last):
        ...
        IndexError: Cannot pop the root stack
        >>> stack.deeppop([0])
        <LayerStack '0' [<PaintingLayer '00'>, <PaintingLayer '01'>]>
        >>> stack.deeppop((0,1))
        <PaintingLayer '11'>
        >>> stack.deeppop((0,2))
        Traceback (most recent call last):
        ...
        IndexError: ...
        """
        if len(path) == 0:
            raise IndexError, "Cannot pop the root stack"
        parent_path = path[:-1]
        child_index = path[-1]
        if len(parent_path) == 0:
            parent = self
        else:
            parent = self.deepget(parent_path)
        return parent.pop(child_index)


    def deepremove(self, layer):
        """Removes a layer from any of the root's descendents

        >>> stack, leaves = _make_test_stack()
        >>> stack.deepremove(leaves[3])
        >>> stack.deepremove(leaves[2])
        >>> stack.deepremove(stack.deepget([0]))
        >>> stack
        <RootLayerStack [<LayerStack '1' []>]>
        >>> stack.deepremove(leaves[3])
        Traceback (most recent call last):
        ...
        ValueError: Layer is not in the root stack or any descendent
        """
        if layer is self:
            raise ValueError, "Cannot remove the root stack"
        for path, descendent_layer in self.deepenumerate():
            assert len(path) > 0
            if descendent_layer is not layer:
                continue
            parent_path = path[:-1]
            if len(parent_path) == 0:
                parent = self
            else:
                parent = self.deepget(parent_path)
            return parent.remove(layer)
        raise ValueError, "Layer is not in the root stack or any descendent"


    def deepindex(self, layer):
        """Return a path for a layer by searching the stack tree

        >>> stack, leaves = _make_test_stack()
        >>> stack.deepindex(stack)
        []
        >>> [stack.deepindex(l) for l in leaves]
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        """
        if layer is self:
            return []
        for path, ly in self.deepenumerate():
            if ly is layer:
                return path
        return None


    ## Convenience methods for commands


    def canonpath(self, index=None, layer=None, path=None,
                  usecurrent=False, uselowest=False):
        """Verify and return the path for a layer from various criteria

        :param index: index of the layer in deepenumerate() order
        :param layer: a layer, which must be a descendent of this root
        :param path: a layer path
        :param usecurrent: if true, use the current path on failure/fallthru
        :return: a new, verified path referring to an existing layer
        :rtype: tuple

        The returned path is guaranteed to refer to an existing layer, and be
        the path in its most canonical form. If no matching layer exists, a
        ValueError is raised.
        """
        if path is not None:
            layer = self.deepget(path)
            if layer is self:
                raise ValueError, ("path=%r is root: must be descendent"
                                   % (path,))
            if layer is not None:
                path = self.deepindex(layer)
                assert self.deepget(path) is layer
                return path
            elif not usecurrent:
                raise ValueError, "layer not found with path=%r"
        elif index is not None:
            if index < 0:
                raise ValueError, "negative layer index %r" % (index,)
            for i, (path, layer) in enumerate(self.deepenumerate()):
                if i == index:
                    assert self.deepget(path) is layer
                    return path
            if not usecurrent:
                raise ValueError, "layer not found with index=%r" % (index,)
        elif layer is not None:
            if layer is self:
                raise ValueError, "layer is root stack: must be descendent"
            path = self.deepindex(layer)
            if path is not None:
                assert self.deepget(path) is layer
                return path
            elif not usecurrent:
                raise ValueError, "layer=%r not found" % (index,)
        # Criterion failed. Try fallbacks.
        if usecurrent:
            path = self.get_current_path()
            layer = self.deepget(path)
            if layer is not None:
                assert layer is not self, ("The current layer path refers to "
                                           "the root stack.")
                path = self.deepindex(layer)
                assert self.deepget(path) is layer
                return path
            if not uselowest:
                raise ValueError, ("Invalid current path; uselowest "
                                   "might work but not specified")
        if uselowest:
            if len(self) > 0:
                path = (0,)
                assert self.deepget(path) is not None
                return path
            else:
                raise ValueError, "Invalid current path; stack is empty"
        raise TypeError, ("No layer/index/path criterion, and no fallbacks")


## Layers with data


class SurfaceBackedLayer (LayerBase):
    """Minimal Surface-backed layer implementation

    This minimal implementation is backed by a surface, which is used for
    rendering by by the main application; subclasses are free to choose whether
    they consider the surface to be the canonical source of layer data or
    something else with the surface being just a preview.
    """

    ## Class constants: class capabilities & other meta-info

    IS_PAINTABLE = False
    IS_FILLABLE = False
    ICON_NAME = None


    ## Initialization

    def __init__(self, name="", compositeop=DEFAULT_COMPOSITE_OP,
                 surface=None):
        """Construct a new SurfaceBackedLayer

        :param name: The name for the new layer.
        :param compositeop: Compositing operation to use.
        :param surface: Surface to use, overriding the default.

        If `surface` is specified, content observers will not be attached, and
        the layer will not be cleared during construction.
        """
        super(SurfaceBackedLayer, self).__init__(name=name,
                                                 compositeop=compositeop)

        # Pluggable surface implementation
        # Only connect observers if using the default tiled surface
        if surface is None:
            self._surface = tiledsurface.Surface()
            self._surface.observers.append(self._notify_content_observers)
        else:
            self._surface = surface

        #: List of content observers (see _notify_content_observers())
        self.content_observers = []

        # Clear if we created our own surface
        if surface is None:
            self.clear()


    def load_from_surface(self, surface):
        """Load the backing surface image's tiles from another surface"""
        self._surface.load_from_surface(surface)

    def load_from_strokeshape(self, strokeshape):
        """Load image tiles from a strokemap.StrokeShape"""
        strokeshape.render_to_surface(self._surface)


    def copy(self):
        """Returns an independent copy of the layer, for Duplicate Layer"""
        layer = super(SurfaceBackedLayer, self).copy()
        layer.content_observers = self.content_observers[:]
        return layer


    ## Generic pixbuf loader methods

    @staticmethod
    def _pixbuf_from_stream(fp, feedback_cb=None):
        loader = GdkPixbuf.PixbufLoader()
        while True:
            if feedback_cb is not None:
                feedback_cb()
            buf = fp.read(LOAD_CHUNK_SIZE)
            if buf == '':
                break
            loader.write(buf)
        loader.close()
        return loader.get_pixbuf()

    def _load_surface_from_pixbuf_file(self, filename, x=0, y=0,
                                       feedback_cb=None):
        """Loads the layer's surface from any file which GdkPixbuf can open"""
        fp = open(filename, 'rb')
        pixbuf = self._pixbuf_from_stream(fp, feedback_cb)
        fp.close()
        return self._load_surface_from_pixbuf(pixbuf, x, y)

    def _load_surface_from_pixbuf(self, pixbuf, x=0, y=0):
        """Loads the layer's surface from a GdkPixbuf"""
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        surface = tiledsurface.Surface()
        bbox = surface.load_from_numpy(arr, x, y)
        self.load_from_surface(surface)
        return bbox

    def load_from_openraster(self, orazip, attrs, tempdir, feedback_cb):
        """Loads layer flags from XML attrs. Derived classes handle data.

        The minimal implementation does not attempt to load any surface image at
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

    ## Info methods

    @property
    def effective_opacity(self):
        """The opacity used when compositing a layer: zero if invisible"""
        # Mirror what composite_tile does.
        if self.visible:
            return self.opacity
        else:
            return 0.0

    def get_alpha(self, x, y, radius):
        """Gets the average alpha within a certain radius at a point"""
        return self._surface.get_alpha(x, y, radius)


    def get_bbox(self):
        """Returns the inherent bounding box of the surface, tile aligned"""
        return self._surface.get_bbox()


    def is_empty(self):
        """Tests whether the surface is empty"""
        return self._surface.is_empty()


    def get_paintable(self):
        """True if this layer currently accepts painting brushstrokes"""
        return self.IS_PAINTABLE and not self.locked


    def get_fillable(self):
        """True if this layer currently accepts flood fill"""
        return self.IS_FILLABLE and not self.locked

    ## Flood fill

    def flood_fill(self, x, y, color, bbox, tolerance, dst_layer=None):
        """Fills a point on the surface with a colour

        See `PaintingLayer.flood_fill() for parameters and semantics. This
        implementation does nothing.
        """
        pass


    ## Rendering


    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       **kwargs):
        """Unconditionally copy one tile's data into an array without options

        The minimal surface-based implementation composites one tile of the
        backing surface over the array dst, modifying only dst.
        """
        self._surface.composite_tile( dst, dst_has_alpha, tx, ty,
                                      mipmap_level=mipmap_level,
                                      opacity=1, mode=DEFAULT_COMPOSITE_OP )


    def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       layers=None, **kwargs):
        """Composite a tile's data into an array, respecting flags/layers list

        The minimal surface-based implementation composites one tile of the
        backing surface over the array dst, modifying only dst.
        """
        if layers is not None:
            if self not in layers:
                return
        elif not self.visible:
            return
        self._surface.composite_tile( dst, dst_has_alpha, tx, ty,
                                      mipmap_level=mipmap_level,
                                      opacity=self.opacity,
                                      mode=self.compositeop )


    def render_as_pixbuf(self, *rect, **kwargs):
        """Renders this layer as a pixbuf"""
        return self._surface.render_as_pixbuf(*rect, **kwargs)


    ## Translating


    def get_move(self, x, y):
        """Get a translation/move object for this layer

        :param x: Model X position of the start of the move
        :param y: Model X position of the start of the move
        :returns: A move object

        Subclasses should extend this minimal implementation to provide
        additional functionality for moving things other than the surface tiles
        around.
        """
        return self._surface.get_move(x, y)


    ## Layer merging

    def merge_into(self, dst, **kwargs):
        """Merge this layer into another, modifying only the destination

        :param dst: The destination layer
        :param **kwargs: Ignored

        The minimal implementation only merges surface tiles. The destination
        layer must always have an alpha channel. After this operation, the
        destination layer's opacity is set to 1.0 and it is made visible.
        """
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
        """Convert pixels to permit compositing with Normal mode"""
        if ( self.compositeop == "svg:src-over" and
             self.effective_opacity == 1.0 ):
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
                dst[:,:,3] = 0 # minimize alpha (discard original alpha)
                # recalculate layer in normal mode
                mypaintlib.tile_flat2rgba(dst, bg)


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
        """Saves the layer's data into an open OpenRaster ZipFile"""
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
        """Sets the surface's painting symmetry axis"""
        if center_x is None:
            self._surface.set_symmetry_state(False, 0.0)
        else:
            self._surface.set_symmetry_state(True, center_x)


    ## Snapshots


    def save_snapshot(self):
        """Snapshots the state of the layer, for undo purposes

        The returned data should be considered opaque, useful only as a
        memento to be restored with load_snapshot().  The base impementation
        snapshots only the surface tiles, and the layer's opacity.
        """
        return _SurfaceBackedLayerSnapshot(self)


    ## Trimming


    def trim(self, rect):
        """Trim the layer to a rectangle, discarding data outside it

        :param rect: A trimming rectangle in model coordinates
        :type rect: tuple (x, y, w, h)

        Only complete tiles are discarded by this method.
        """
        self._surface.trim(rect)


class _SurfaceBackedLayerSnapshot (_LayerBaseSnapshot):
    """Minimal layer implementation's snapshot

    Snapshots are stored in commands, and used to implement undo and redo.
    They must be independent copies of the data, although copy-on-write
    semantics are fine. Snapshot objects don't have to be _full and exact_
    clones of the layer's data, but they do need to capture _inherent_
    qualities of the layer. Mere metadata can be ignored. For the base
    layer implementation, this means the surface tiles and the layer's
    opacity.
    """

    def __init__(self, layer):
        _LayerBaseSnapshot.__init__(self, layer)
        self.surface_sshot = layer._surface.save_snapshot()

    def restore_to_layer(self, layer):
        _LayerBaseSnapshot.restore_to_layer(self, layer)
        layer._surface.load_snapshot(self.surface_sshot)


class BackgroundLayer (SurfaceBackedLayer):
    """Background layer, with a repeating tiled image"""

    # NOTE: by convention only, there is just a single non-editable background
    # layer. Background layers cannot be manipulated by the user except through
    # the background dialog.

    # NOTE: this could be generalized as a repeating tile for general use in
    # the layers stack, extending the ExternalLayer concept. Think textures!
    # Might need src-in compositing for that though.

    def __init__(self, bg):
        if isinstance(bg, tiledsurface.Background):
            surface = bg
        else:
            surface = tiledsurface.Background(bg)
        super(BackgroundLayer, self).__init__(name="background",
                                              surface=surface)
        self.locked = False
        self.visible = True
        self.opacity = 1.0

    def copy(self):
        raise NotImplementedError, "BackgroundLayer cannot be copied yet"

    def save_snapshot(self):
        raise NotImplementedError, "BackgroundLayer cannot be snapshotted yet"

    def load_snapshot(self):
        raise NotImplementedError, "BackgroundLayer cannot be snapshotted yet"

    def set_surface(self, surface):
        """Sets the surface from a tiledsurface.Background"""
        assert isinstance(surface, tiledsurface.Background)
        self._surface = surface

    def save_to_openraster(self, orazip, tmpdir, ref, selected,
                           canvas_bbox, frame_bbox, **kwargs):
        # Normalize ref
        if type(ref) == int:
            ref = "background%03d" % (ref,)
        ref = str(ref)

        # Save as a regular layer for other apps.
        # Background surfaces repeat, so this will fit the frame.
        # XXX But we use the canvas bbox and always have. Why?
        # XXX - Presumably it's for origin alignment.
        # XXX - Inefficient for small frames.
        # XXX - I suspect rect should be redone with (w,h) granularity
        # XXX   and be based on the frame_bbox.
        rect = canvas_bbox
        attrs = super(BackgroundLayer, self)\
            ._save_rect_to_ora( orazip, tmpdir, ref, selected,
                                canvas_bbox, frame_bbox, rect, **kwargs )
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



class ExternalLayer (SurfaceBackedLayer):
    """A layer which is stored as a tempfile in a non-MyPaint format

    External layers add the name of the tempfile to the base implementation.
    The internal surface is used to display a bitmap preview of the layer, but
    this cannot be edited.

    SVG files are the canonical example.
    """

    ## Class constants

    IS_FILLABLE = False
    IS_PAINTABLE = False
    ICON_NAME = "mypaint-layer-vector-symbolic"

    ## Construction

    def __init__(self, name="", compositeop=DEFAULT_COMPOSITE_OP):
        """Construct, with blank internal fields"""
        super(ExternalLayer, self).__init__( name=name,
                                             compositeop=compositeop )
        self._basename = None
        self._workdir = None
        self._x = None
        self._y = None


    def set_workdir(self, workdir):
        """Sets the working directory (i.e. to doc's tempdir)

        This is where working copies are created and cached during operation.
        """
        self._workdir = workdir


    def load_from_openraster(self, orazip, attrs, tempdir, feedback_cb):
        """Loads layer data and attrs from an OpenRaster zipfile

        Using this method also sets the working directory for the layer to
        tempdir.
        """
        # Load layer flags
        selected = super(ExternalLayer, self)\
                    .load_from_openraster(orazip, attrs, tempdir, feedback_cb)
        # Read SVG or whatever content via a tempdir
        src = attrs.get("src", None)
        src_rootname, src_ext = os.path.splitext(src)
        src_rootname = os.path.basename(src_rootname)
        src_ext = src_ext.lower()
        x = int(attrs.get('x', 0))
        y = int(attrs.get('y', 0))
        t0 = time.time()
        orazip.extract(src, path=tempdir)
        tmp_filename = os.path.join(tempdir, src)
        rev0_fd, rev0_filename = tempfile.mkstemp(suffix=src_ext, dir=tempdir)
        os.close(rev0_fd)
        os.rename(tmp_filename, rev0_filename)
        self._load_surface_from_pixbuf_file(rev0_filename, x, y, feedback_cb)
        self._basename = os.path.basename(rev0_filename)
        self._workdir = tempdir
        self._x = x
        self._y = y
        t1 = time.time()
        logger.debug('%.3fs loading and converting src %r for %r',
                     t1 - t0, src_ext, src_rootname)
        # Return
        return selected


    ## Snapshots

    def save_snapshot(self):
        """Snapshots the state of the layer and its strokemap for undo"""
        return _ExternalLayerSnapshot(self)

    ## Moving


    def get_move(self, x, y):
        """Start a new move for the external layer"""
        surface_move = super(ExternalLayer, self).get_move(x, y)
        return ExternalLayerMove(self, surface_move)


    ## Trimming (no-op for external layers)


    def trim(self, rect):
        """Override: external layers have no useful trim(), so do nothing"""
        pass


    ## Saving


    def save_to_openraster(self, orazip, tmpdir, ref, selected,
                           canvas_bbox, frame_bbox, **kwargs):
        """Saves the tempfile to an OpenRaster zipfile, & returns attrs"""
        # No supercall in this override, but the base implementation's
        # attributes method is useful.
        attrs = self._get_data_attrs(frame_bbox)
        # Store the managed layer position rather than one based on the
        # surface's tiles bbox, however.
        fx, fy = frame_bbox[0:2]
        attrs["x"] = self._x - fx
        attrs["y"] = self._y - fy
        # Pick a suitable name to store under.
        src_path = os.path.join(self._workdir, self._basename)
        src_rootname, src_ext = os.path.splitext(src_path)
        src_ext = src_ext.lower()
        if type(ref) == int:
            storename = "layer%03d%s" % (ref, src_ext)
        else:
            storename = "%s%s" % (ref)
        storepath = "data/%s" % (storename,)
        # Archive (but do not remove) the managed tempfile
        t0 = time.time()
        orazip.write(src_path, storepath)
        t1 = time.time()
        # Return details of what was written.
        attrs["src"] = storepath
        return attrs


class _ExternalLayerSnapshot (_SurfaceBackedLayerSnapshot):
    """Snapshot subclass for external layers"""

    def __init__(self, layer):
        super(_ExternalLayerSnapshot, self).__init__(layer)
        self.basename = self._copy_working_file( layer._basename,
                                                 layer._workdir )
        self.workdir = layer._workdir
        self.x = layer._x
        self.y = layer._y

    def restore_to_layer(self, layer):
        super(_ExternalLayerSnapshot, self).restore_to_layer(layer)
        layer._basename = self._copy_working_file( self.basename,
                                                   self.workdir )
        layer._workdir = self.workdir
        layer._x = self.x
        layer._y = self.y

    @staticmethod
    def _copy_working_file(old_basename, workdir):
        old_filename = os.path.join(workdir, old_basename)
        rootname, ext = os.path.splitext(old_basename)
        old_fp = open(old_filename, 'rb')
        new_fp = tempfile.NamedTemporaryFile( dir=workdir,
                                              suffix=ext, mode="w+b",
                                              delete=False )
        shutil.copyfileobj(old_fp, new_fp)
        new_basename = os.path.basename(new_fp.name)
        logger.debug( "Copied %r to %r within %r...",
                      old_basename, new_basename, workdir )
        new_fp.close()
        old_fp.close()
        return new_basename

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        sshot_copy = os.path.join(self.workdir, self.basename)
        if os.path.exists(sshot_copy):
            logger.debug("Cleanup: removing %r from %r",
                         self.basename, self.workdir)
            os.remove(sshot_copy)
        else:
            logger.debug("Cleanup: %r was already removed from %r",
                         self.basename, self.workdir)


class ExternalLayerMove (object):
    """Move object wrapper for external layers"""

    def __init__(self, layer, surface_move):
        super(ExternalLayerMove, self).__init__()
        self._wrapped = surface_move
        self._layer = layer
        self._start_x = layer._x
        self._start_y = layer._y

    def update(self, dx, dy):
        self._layer._x = int(round(self._start_x + dx))
        self._layer._y = int(round(self._start_y + dy))
        self._wrapped.update(dx, dy)

    def cleanup(self):
        self._wrapped.cleanup()

    def process(self, n=200):
        return self._wrapped.process(n)



class PaintingLayer (SurfaceBackedLayer):
    """A paintable, bitmap layer

    Painting layers add a strokemap to the base implementation. The stroke map
    is a stack of `strokemap.StrokeShape` objects in painting order, allowing
    strokes and their associated brush and color information to be picked from
    the canvas.
    """

    ## Class constants

    IS_PAINTABLE = True
    IS_FILLABLE = True


    ## Initializing & resetting


    def __init__(self, name="", compositeop=DEFAULT_COMPOSITE_OP):
        super(PaintingLayer, self).__init__( name=name,
                                             compositeop=compositeop )
        #: Stroke map.
        #: List of strokemap.StrokeShape instances (not stroke.Stroke), ordered
        #: by depth.
        self.strokes = []

 
    def clear(self):
        """Clear both the surface and the strokemap"""
        super(PaintingLayer, self).clear()
        self.strokes = []


    def load_from_surface(self, surface):
        """Load the surface image's tiles from another surface"""
        super(PaintingLayer, self).load_from_surface(surface)
        self.strokes = []


    def load_from_openraster(self, orazip, attrs, tempdir, feedback_cb):
        """Loads layer flags, PNG data, amd strokemap from a .ora zipfile"""
        # Load layer flags
        selected = super(PaintingLayer, self)\
            .load_from_openraster(orazip, attrs, tempdir, feedback_cb)
        # Read PNG content via tempdir
        src = attrs.get("src", None)
        src_rootname, src_ext = os.path.splitext(src)
        src_rootname = os.path.basename(src_rootname)
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
                     t1 - t0, src_ext, src_rootname)
        # Strokemap
        strokemap_name = attrs.get('mypaint_strokemap_v2', None)
        t2 = time.time()
        if strokemap_name is not None:
            sio = StringIO(orazip.read(strokemap_name))
            self.load_strokemap_from_file(sio, x, y)
            sio.close()
        t3 = time.time()
        logger.debug('%.3fs loading strokemap for %r',
                     t3 - t2, src_rootname)
        # Return (TODO: notify needed here?)
        return selected


    ## Flood fill

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
        :type dst_layer: SurfaceBackedLayer

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
        before = snapshot_before.surface_sshot
        after  = self._surface.save_snapshot()
        shape = strokemap.StrokeShape()
        shape.init_from_snapshots(before, after)
        shape.brush_string = stroke.brush_settings
        self.strokes.append(shape)


    ## Snapshots


    def save_snapshot(self):
        """Snapshots the state of the layer and its strokemap for undo"""
        return _PaintingLayerSnapshot(self)


    ## Translating

    def get_move(self, x, y):
        """Get an interactive move object for the surface and its strokemap"""
        surface_move = super(PaintingLayer, self).get_move(x, y)
        return PaintingLayerMove(self, surface_move)


    ## Trimming


    def trim(self, rect):
        """Trim the layer and its strokemap"""
        super(PaintingLayer, self).trim(rect)
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
        """Get the stroke at the given point"""
        x, y = int(x), int(y)
        for s in reversed(self.strokes):
            if s.touches_pixel(x, y):
                return s


    def get_last_stroke_info(self):
        if not self.strokes:
            return None
        return self.strokes[-1]


    ## Layer merging


    def merge_into(self, dst, strokemap=True, **kwargs):
        """Merge this layer into another, possibly including its strokemap

        :param dst: The target layer
        :param strokemap: Try to copy strokemap too
        :param **kwargs: passed to superclass

        If the destination layer is a PaintingLayer and `strokemap` is true,
        its strokemap will be extended with the data from this one.
        """
        # Flood-fill uses this for its newly created and working layers,
        # but it should not construct a strokemap for what it does.
        if strokemap and isinstance(dst, PaintingLayer):
            dst.strokes.extend(self.strokes)
        # Merge surface tiles
        super(PaintingLayer, self).merge_into(dst, **kwargs)


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

        attrs = super(PaintingLayer, self)\
            .save_to_openraster( orazip, tmpdir, ref, selected,
                                 canvas_bbox, frame_bbox, **kwargs )
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


class _PaintingLayerSnapshot (_SurfaceBackedLayerSnapshot):
    """Snapshot subclass for painting layers"""

    def __init__(self, layer):
        super(_PaintingLayerSnapshot, self).__init__(layer)
        self.strokes = layer.strokes[:]

    def restore_to_layer(self, layer):
        super(_PaintingLayerSnapshot, self).restore_to_layer(layer)
        layer.strokes = self.strokes[:]


class PaintingLayerMove (object):
    """Move object wrapper for painting layers"""

    def __init__(self, layer, surface_move):
        super(PaintingLayerMove, self).__init__()
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


## Module testing


def _test():
    """Run doctest strings"""
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)


def _make_test_stack():
    """Makes a simple test RootLayerStack with 2 branches of 2 leaves each

    :return: The root stack, and a list of its leaves.
    :rtype: tuple
    """
    layer = RootLayerStack(doc=None)
    layer0 = LayerStack('0'); layer.append(layer0)
    layer00 = PaintingLayer('00'); layer0.append(layer00)
    layer01 = PaintingLayer('01'); layer0.append(layer01)
    layer1 = LayerStack('1'); layer.append(layer1)
    layer10 = PaintingLayer('10'); layer1.append(layer10)
    layer11 = PaintingLayer('11'); layer1.append(layer11)
    return (layer, [layer00, layer01, layer10, layer11])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _test()
