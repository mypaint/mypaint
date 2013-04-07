# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import time
import struct
import zlib
from numpy import *
import mypaintlib

import tiledsurface
import idletask

N = tiledsurface.N


class StrokeShape:
    """The shape of a single brushstroke.

    This class stores the shape of a stroke in as a 1-bit bitmap. The
    information is stored in compressed memory blocks of the size of a
    tile (for fast lookup).
    """
    def __init__(self):
        self.tasks = idletask.Processor(max_pending=6)
        self.strokemap = {}

    def init_from_snapshots(self, snapshot_before, snapshot_after):
        assert not self.strokemap
        # extract the layer from each snapshot
        a, b = snapshot_before.tiledict, snapshot_after.tiledict
        # enumerate all tiles that have changed
        a_tiles = set(a.iteritems())
        b_tiles = set(b.iteritems())
        changes = a_tiles.symmetric_difference(b_tiles)
        tiles_modified = set([pos for pos, data in changes])

        # for each tile, calculate the exact difference (not now, later, when idle)
        queue = []
        for tx, ty in tiles_modified:
            def work(tx=tx, ty=ty):
                # get the pixel data to compare
                a_data = a.get((tx, ty), tiledsurface.transparent_tile).rgba
                b_data = b.get((tx, ty), tiledsurface.transparent_tile).rgba

                data = empty((N, N), 'uint8')
                mypaintlib.tile_perceptual_change_strokemap(a_data, b_data, data)

                data_compressed = zlib.compress(data.tostring())
                self.strokemap[tx, ty] = data_compressed

            self.tasks.add_work(work, weight=1.0/len(tiles_modified))

    def init_from_string(self, data, translate_x, translate_y):
        assert not self.strokemap
        assert translate_x % N == 0
        assert translate_y % N == 0
        translate_x /= N
        translate_y /= N
        while data:
            tx, ty, size = struct.unpack('>iiI', data[:3*4])
            compressed_bitmap = data[3*4:size+3*4]
            self.strokemap[tx + translate_x, ty + translate_y] = compressed_bitmap
            data = data[size+3*4:]

    def save_to_string(self, translate_x, translate_y):
        assert translate_x % N == 0
        assert translate_y % N == 0
        translate_x /= N
        translate_y /= N
        self.tasks.finish_all()
        data = ''
        for (tx, ty), compressed_bitmap in self.strokemap.iteritems():
            tx, ty = tx + translate_x, ty + translate_y
            data += struct.pack('>iiI', tx, ty, len(compressed_bitmap))
            data += compressed_bitmap
        return data

    def touches_pixel(self, x, y):
        self.tasks.finish_all()
        data = self.strokemap.get((x/N, y/N))
        if data:
            data = fromstring(zlib.decompress(data), dtype='uint8')
            data.shape = (N, N)
            return data[y%N, x%N]

    def render_overlay(self, layer):
        surf = layer._surface # FIXME: Don't touch inner details of layer
        self.tasks.finish_all()
        for (tx, ty), data in self.strokemap.iteritems():
            data = fromstring(zlib.decompress(data), dtype='uint8')
            data.shape = (N, N)

            with surf.tile_request(tx, ty, readonly=False) as rgba:
                # neutral gray, 50% opaque
                rgba[:,:,3] = data.astype('uint16') * (1<<15)/2
                rgba[:,:,0] = rgba[:,:,3]/2
                rgba[:,:,1] = rgba[:,:,3]/2
                rgba[:,:,2] = rgba[:,:,3]/2

    def translate(self, dx, dy):
        """Translate the shape by (dx, dy).
        """
        # Finish any previous translations or handling of painted strokes
        self.tasks.finish_all()
        src_strokemap = self.strokemap
        self.strokemap = {}
        slices_x = tiledsurface.calc_translation_slices(int(dx))
        slices_y = tiledsurface.calc_translation_slices(int(dy))
        tmp_strokemap = {}
        is_integral = len(slices_x) == 1 and len(slices_y) == 1
        for (src_tx, src_ty), src in src_strokemap.iteritems():
            def __translate_tile(src_tx=src_tx, src_ty=src_ty, src=src):
                src = fromstring(zlib.decompress(src), dtype='uint8')
                src.shape = (N, N)
                for (src_x0, src_x1), (tmp_tdx, tmp_x0, tmp_x1) in slices_x:
                    for (src_y0, src_y1), (tmp_tdy, tmp_y0, tmp_y1) in slices_y:
                        tmp_tx = src_tx + tmp_tdx
                        tmp_ty = src_ty + tmp_tdy
                        if is_integral:
                            tmp_strokemap[tmp_tx, tmp_ty] = src
                        else:
                            tmp = tmp_strokemap.get((tmp_tx, tmp_ty), None)
                            if tmp is None:
                                tmp = zeros((N, N), 'uint8')
                                tmp_strokemap[tmp_tx, tmp_ty] = tmp
                            tmp[tmp_y0:tmp_y1, tmp_x0:tmp_x1] \
                              = src[src_y0:src_y1, src_x0:src_x1]
            self.tasks.add_work(__translate_tile, weight=0.1/len(src_strokemap))
        # Recompression of any tile can only start after all the above is
        # complete. Luckily the idle-processor does things in order.
        def __start_tile_recompression():
            for (tx, ty), data in tmp_strokemap.iteritems():
                def __recompress_tile(tx=tx, ty=ty, data=data):
                    if not data.any():
                        return
                    data_compressed = zlib.compress(data.tostring())
                    self.strokemap[tx, ty] = data_compressed
                self.tasks.add_work(__recompress_tile, weight=0.1/len(tmp_strokemap))
        self.tasks.add_work(__start_tile_recompression, weight=0)
