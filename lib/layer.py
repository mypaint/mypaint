# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from numpy import *
import time

import tiledsurface
N = tiledsurface.N

class Layer:
    def __init__(self):
        self.surface = tiledsurface.Surface()
        self.clear()

    def clear(self):
        self.strokes = [] # contains StrokeInfo instances (not stroke.Stroke)
        self.surface.clear()

    def load_from_pixbuf(self, pixbuf):
        self.strokes = []
        self.surface.load_from_data(pixbuf)

    def save_snapshot(self):
        return (self.strokes[:], self.surface.save_snapshot())

    def load_snapshot(self, data):
        strokes, data = data
        self.strokes = strokes[:]
        self.surface.load_snapshot(data)

    def add_stroke(self, stroke, snapshot_before):
        before = snapshot_before[1] # extract surface snapshot
        after  = self.surface.save_snapshot()
        self.strokes.append(StrokeInfo(stroke, before, after))

    def merge_into(self, dst):
        """
        Merge this layer into dst, modifying only dst.
        """
        src = self
        dst.strokes.extend(self.strokes)
        for tx, ty in src.surface.get_tiles():
            src.surface.composite_tile_over(dst.surface.get_tile_memory(tx, ty, readonly=False), tx, ty)

    def get_brush_at(self, x, y):
        x, y = int(x), int(y)
        for s in reversed(self.strokes):
            if s.touches_pixel(x, y):
                return s.brush



class StrokeInfo:
    def __init__(self, stroke, snapshot_before, snapshot_after):
        self.brush = stroke.brush_settings
        t0 = time.time()
        # extract the layer from each snapshot
        a, b = snapshot_before.tiledict, snapshot_after.tiledict
        # enumerate all tiles that have changed
        a_tiles = set(a.items())
        b_tiles = set(b.items())
        changes = a_tiles.symmetric_difference(b_tiles)
        tiles_modified = set([pos for pos, data in changes])

        # for each tile, calculate the exact difference
        self.strokemap = {}
        for tx, ty in tiles_modified:
            # get the pixel data to compare
            a_data = a.get((tx, ty), tiledsurface.transparent_tile).rgba
            b_data = b.get((tx, ty), tiledsurface.transparent_tile).rgba

            # calculate the "perceptual" amount of difference
            absdiff = zeros((N, N), 'uint32')
            for i in range(4): # RGBA
                absdiff += abs(a_data[:,:,i].astype('uint32') - b_data[:,:,i])
            # ignore badly visible (parts of) strokes, eg. very faint strokes
            #
            # This is an arbitrary threshold. If it is too high, an
            # ink stroke with slightly different color than the one
            # below will not be pickable.  If it is too high, barely
            # visible strokes will make things below unpickable.
            #
            threshold = (1<<15)*4 / 16 # require 1/16 of the max difference (also not bad: 1/8)
            is_different = absdiff > threshold
            # except if there is no previous stroke below it
            is_different |= (absdiff > 0) #& (brushmap_data == 0) --- FIXME: not possible any more
            self.strokemap[tx, ty] = is_different

        print 'brushmap update took %.3f seconds' % (time.time() - t0)

    def touches_pixel(self, x, y):
        tile = self.strokemap.get((x/N, y/N))
        if tile is not None:
            return tile[y%N, x%N]
