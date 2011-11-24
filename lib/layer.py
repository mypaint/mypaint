# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import struct, zlib
from numpy import *
from gettext import gettext as _

import tiledsurface, strokemap

COMPOSITE_OPS = [
    # (internal-name, display-name)
    ("svg:src-over", _("Normal")),
    ("svg:multiply", _("Multiply")),
    ("svg:color-burn", _("Burn")),
    ("svg:color-dodge", _("Dodge")),
    ("svg:screen", _("Screen")),
    ]

DEFAULT_COMPOSITE_OP = COMPOSITE_OPS[0][0]
VALID_COMPOSITE_OPS = set([n for n, d in COMPOSITE_OPS])

class Layer:
    """Representation of a layer in the document model.

    The actual content of the layer is held by the surface implementation.
    This is an internal detail that very few consumers should care about."""

    def __init__(self, name="", compositeop=DEFAULT_COMPOSITE_OP):
        self._surface = tiledsurface.Surface()
        self.opacity = 1.0
        self.name = name
        self.visible = True
        self.locked = False
        self.compositeop = compositeop
        # Called when contents of layer changed,
        # with the bounding box of the changed region
        self.content_observers = []

        # Forward from surface implementation
        self._surface.observers.append(self._notify_content_observers)

        self.clear()

    def translate(self, dx, dy):
        self._surface.translate(dx, dy)

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
        split = brush.stroke_to(self._surface, x, y,
                                    pressure, xtilt, ytilt, dtime)
        self._surface.end_atomic()
        return split

    def clear(self):
        self.strokes = [] # contains StrokeShape instances (not stroke.Stroke)
        self._surface.clear()

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


    def begin_interactive_move(self, x, y):
        """Start an interactive move.

        Returns ``(snapshot, chunks)``, and blanks the current layer. In the
        current implementation, ``chunks`` is a list of tile indices which
        is sorted by proximity to the initial position.
        """
        return self._surface.begin_interactive_move(x, y)


    def update_interactive_move(self, dx, dy):
        """Update for a new offset during an interactive move.

        Call whenever there's a new pointer position in the drag. Returns a
        single offsets object. After calling, reprocess the chunks queue with
        the new offsets. This method blanks the current layer, so the chunks
        queue must be processed fully after the final call to it.
        """
        surf = self._surface
        return surf.update_interactive_move(dx, dy)


    def process_interactive_move_queue(self, snapshot, chunks, offsets):
        """Processes part of an interactive move.
        """
        surf = self._surface
        return surf.process_interactive_move_queue(snapshot, chunks, offsets)


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
                stroke.init_from_string(tmp, translate_x, translate_y)
                stroke.brush_string = brushes[brush_id]
                self.strokes.append(stroke)
            elif t == '}':
                break
            else:
                assert False, 'invalid strokemap'

    def merge_into(self, dst):
        """
        Merge this layer into dst, modifying only dst.
        """
        # We must respect layer visibility, because saving a
        # transparent PNG just calls this function for each layer.
        src = self
        dst.strokes.extend(self.strokes)
        for tx, ty in dst._surface.get_tiles():
            surf = dst._surface.get_tile_memory(tx, ty, readonly=False)
            surf[:,:,:] = dst.effective_opacity * surf[:,:,:]
        for tx, ty in src._surface.get_tiles():
            surf = dst._surface.get_tile_memory(tx, ty, readonly=False)
            src._surface.composite_tile(surf, tx, ty,
                opacity=self.effective_opacity,
                mode=self.compositeop)
        dst.opacity = 1.0

    def get_stroke_info_at(self, x, y):
        x, y = int(x), int(y)
        for s in reversed(self.strokes):
            if s.touches_pixel(x, y):
                return s

    def get_last_stroke_info(self):
        if not self.strokes:
            return None
        return self.strokes[-1]
