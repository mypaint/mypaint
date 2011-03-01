# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import struct, zlib
from numpy import *

import tiledsurface, strokemap

class Layer:
    def __init__(self,name=""):
        self.surface = tiledsurface.Surface()
        self.opacity = 1.0
        self.name = name
        self.visible = True
        self.locked = False
        self.clear()

    def get_effective_opacity(self):
        if self.visible:
            return self.opacity
        else:
            return 0.0
    effective_opacity = property(get_effective_opacity)

    def clear(self):
        self.strokes = [] # contains StrokeShape instances (not stroke.Stroke)
        self.surface.clear()

    def load_from_pixbuf(self, pixbuf):
        self.strokes = []
        self.surface.load_from_data(pixbuf)

    def save_snapshot(self):
        return (self.strokes[:], self.surface.save_snapshot(), self.opacity)

    def load_snapshot(self, data):
        strokes, data, self.opacity = data
        self.strokes = strokes[:]
        self.surface.load_snapshot(data)

    def add_stroke(self, stroke, snapshot_before):
        before = snapshot_before[1] # extract surface snapshot
        after  = self.surface.save_snapshot()
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
        for tx, ty in dst.surface.get_tiles():
            surf = dst.surface.get_tile_memory(tx, ty, readonly=False)
            surf[:,:,:] = dst.effective_opacity * surf[:,:,:]
        for tx, ty in src.surface.get_tiles():
            surf = dst.surface.get_tile_memory(tx, ty, readonly=False)
            src.surface.composite_tile_over(surf, tx, ty, opacity=self.effective_opacity)
        dst.opacity = 1.0

    def get_stroke_info_at(self, x, y):
        x, y = int(x), int(y)
        for s in reversed(self.strokes):
            if s.touches_pixel(x, y):
                return s



