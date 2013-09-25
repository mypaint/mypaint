# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import struct
import zlib
from numpy import *
import logging
logger = logging.getLogger(__name__)

from gettext import gettext as _

import tiledsurface
import strokemap
import mypaintlib

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

class Layer (object):
    """Representation of a layer in the document model.

    The actual content of the layer is held by the surface implementation.
    This is an internal detail that very few consumers should care about."""

    def __init__(self, name="", compositeop=DEFAULT_COMPOSITE_OP):
        object.__init__(self)
        self._surface = tiledsurface.Surface()
        self.opacity = 1.0
        self.name = name
        self.visible = True
        self.locked = False
        self.compositeop = compositeop

        #: List of content observers
        #: These callbacks are invoked when the contents of the layer change,
        #: with the bounding box of the changed region (x, y, w, h).
        self.content_observers = []

        # Forward from surface implementation
        self._surface.observers.append(self._notify_content_observers)

        self.clear()

    def _notify_content_observers(self, *args):
        for f in self.content_observers:
            f(*args)

    def get_effective_opacity(self):
        if self.visible:
            return self.opacity
        else:
            return 0.0
    effective_opacity = property(get_effective_opacity)

    def get_alpha(self, x, y, radius):
        return self._surface.get_alpha(x, y, radius)

    def get_bbox(self):
        return self._surface.get_bbox()

    def is_empty(self):
        return self._surface.is_empty()

    def save_as_png(self, filename, *args, **kwargs):
        self._surface.save_as_png(filename, *args, **kwargs)

    def stroke_to(self, brush, x, y, pressure, xtilt, ytilt, dtime):
        """Render a part of a stroke."""
        self._surface.begin_atomic()
        split = brush.stroke_to(self._surface.backend, x, y,
                                    pressure, xtilt, ytilt, dtime)
        self._surface.end_atomic()
        return split


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


    def clear(self):
        self.strokes = [] # contains StrokeShape instances (not stroke.Stroke)
        self._surface.clear()


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


    def load_from_surface(self, surface):
        self.strokes = []
        self._surface.load_from_surface(surface)

    def render_as_pixbuf(self, *rect, **kwargs):
        return self._surface.render_as_pixbuf(*rect, **kwargs)

    def save_snapshot(self):
        return (self.strokes[:], self._surface.save_snapshot(), self.opacity)

    def load_snapshot(self, data):
        strokes, data, self.opacity = data
        self.strokes = strokes[:]
        self._surface.load_snapshot(data)


    def translate(self, dx, dy):
        """Translate a layer non-interactively.
        """
        move = self.get_move(0, 0)
        move.update(dx, dy)
        move.process(n=-1)
        move.cleanup()
        for shape in self.strokes:
            shape.translate(dx, dy)


    def get_move(self, x, y):
        """Get a translation/move object for this layer.
        """
        return self._surface.get_move(x, y)
        # FIXME: really should return an object that does the strokemap
        # translates after the drag is complete.


    def add_stroke(self, stroke, snapshot_before):
        before = snapshot_before[1] # extract surface snapshot
        after  = self._surface.save_snapshot()
        shape = strokemap.StrokeShape()
        shape.init_from_snapshots(before, after)
        shape.brush_string = stroke.brush_settings
        self.strokes.append(shape)


    def save_strokemap_to_file(self, f, translate_x, translate_y):
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

    def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0):
        self._surface.composite_tile(
            dst, dst_has_alpha, tx, ty,
            mipmap_level=mipmap_level,
            opacity=self.effective_opacity,
            mode=self.compositeop
            )

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
        """
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

    def get_stroke_info_at(self, x, y):
        x, y = int(x), int(y)
        for s in reversed(self.strokes):
            if s.touches_pixel(x, y):
                return s

    def get_last_stroke_info(self):
        if not self.strokes:
            return None
        return self.strokes[-1]

    def set_symmetry_axis(self, center_x):
        if center_x is None:
            self._surface.set_symmetry_state(False, 0.0)
        else:
            self._surface.set_symmetry_state(True, center_x)
