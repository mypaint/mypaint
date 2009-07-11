# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from numpy import *
import time, zlib, gobject

import tiledsurface
N = tiledsurface.N

class Layer:
    def __init__(self):
        self.surface = tiledsurface.Surface()
        self.opacity = 1.0
        self.clear()

    def clear(self):
        self.strokes = [] # contains StrokeInfo instances (not stroke.Stroke)
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
        self.strokes.append(StrokeInfo(stroke, before, after))

    def merge_into(self, dst):
        """
        Merge this layer into dst, modifying only dst.
        """
        src = self
        dst.strokes.extend(self.strokes)
        for tx, ty in dst.surface.get_tiles():
            surf = dst.surface.get_tile_memory(tx, ty, readonly=False)
            surf[:,:,:] = dst.opacity * surf[:,:,:]
        for tx, ty in src.surface.get_tiles():
            src.surface.composite_tile_over(dst.surface.get_tile_memory(tx, ty, readonly=False), tx, ty, self.opacity)
        dst.opacity = 1.0

    def get_brush_at(self, x, y):
        x, y = int(x), int(y)
        for s in reversed(self.strokes):
            if s.touches_pixel(x, y):
                return s.brush



class StrokeInfo:
    processing_queue = [] # global (static) list
    def __init__(self, stroke, snapshot_before, snapshot_after):
        self.brush = stroke.brush_settings
        # extract the layer from each snapshot
        a, b = snapshot_before.tiledict, snapshot_after.tiledict
        # enumerate all tiles that have changed
        a_tiles = set(a.items())
        b_tiles = set(b.items())
        changes = a_tiles.symmetric_difference(b_tiles)
        tiles_modified = set([pos for pos, data in changes])

        # for each tile, calculate the exact difference (not now, later)
        self.strokemap = {}
        queue = []
        for tx, ty in tiles_modified:
            def work(tx=tx, ty=ty):
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
                strokemap_data = is_different.astype('uint8')

                # keep memory usage reasonable (data is highly redundant)
                strokemap_data = zlib.compress(strokemap_data.tostring())

                self.strokemap[tx, ty] = strokemap_data

            queue.append(work)
        self.processing_queue.append(queue)

        gobject.idle_add(self.idle_cb)

        # make sure we never lag too much behind with processing
        # (otherwise we waste way too much memory)
        self.process_pending_strokes(max_pending_strokes=6)

    def process_one_item(self):
        items = self.processing_queue[0]
        if items:
            func = items.pop(0)
            func()
        else:
            self.processing_queue.pop(0)
        
    def process_pending_strokes(self, max_pending_strokes=0):
        while len(self.processing_queue) > max_pending_strokes:
            self.process_one_item()

    def idle_cb(self):
        if not self.processing_queue:
            return False
        self.process_one_item()
        return True

    def touches_pixel(self, x, y):
        self.process_pending_strokes()
        data = self.strokemap.get((x/N, y/N))
        if data:
            data = fromstring(zlib.decompress(data), dtype='uint8')
            data.shape = (N, N)
            return data[y%N, x%N]
