# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import time
import gobject, zlib
from numpy import *

import tiledsurface, strokemap_pb2
N = tiledsurface.N

class StrokeInfo:
    """
    This class stores permanent (saved with image) information about a
    single stroke. Mainly this is the stroke shape and the brush
    settings that were used. Needed to pick brush from canvas.
    """
    processing_queue = [] # global (static) list
    def __init__(self):
        self.strokemap = {}
        self.brush = None

    def init_from_snapshots(self, brush_string, snapshot_before, snapshot_after):
        assert not self.strokemap
        self.brush_string = brush_string
        # extract the layer from each snapshot
        a, b = snapshot_before.tiledict, snapshot_after.tiledict
        # enumerate all tiles that have changed
        a_tiles = set(a.items())
        b_tiles = set(b.items())
        changes = a_tiles.symmetric_difference(b_tiles)
        tiles_modified = set([pos for pos, data in changes])

        # for each tile, calculate the exact difference (not now, later, when idle)
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
                data = is_different.astype('uint8')

                data_compressed = zlib.compress(data.tostring())
                self.strokemap[tx, ty] = data_compressed

            queue.append(work)
        self.processing_queue.append(queue)

        gobject.idle_add(self.idle_cb)

        # make sure we never lag too much behind with processing
        # (otherwise we waste way too much memory)
        self.process_pending_strokes(max_pending_strokes=6)

    def init_from_pb(self, stroke_pb, translate_x, translate_y):
        assert not self.strokemap
        assert translate_x % N == 0
        assert translate_y % N == 0
        translate_x /= N
        translate_y /= N
        for t in stroke_pb.tiles:
            self.strokemap[t.tx + translate_x, t.ty + translate_y] = t.data_compressed
        self.brush_string = zlib.decompress(stroke_pb.brush_string_compressed)

    def save_to_pb(self, stroke_pb, translate_x, translate_y):
        assert translate_x % N == 0
        assert translate_y % N == 0
        translate_x /= N
        translate_y /= N
        self.process_pending_strokes()
        for (tx, ty), data in self.strokemap.iteritems():
            t = stroke_pb.tiles.add()
            t.tx, t.ty = tx + translate_x, ty + translate_y
            t.data_compressed = data
        stroke_pb.brush_string_compressed = zlib.compress(self.brush_string)

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
