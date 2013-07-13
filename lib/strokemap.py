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
from logging import getLogger
logger = getLogger(__name__)

import mypaintlib

import tiledsurface
import idletask

TILE_SIZE = N = mypaintlib.TILE_SIZE

class StrokeShape (object):
    """The shape of a single brushstroke.

    This class stores the shape of a stroke in as a 1-bit bitmap. The
    information is stored in compressed memory blocks of the size of a
    tile (for fast lookup).
    """
    def __init__(self):
        object.__init__(self)
        self.tasks = idletask.Processor()
        self.strokemap = {}

    def init_from_snapshots(self, snapshot_before, snapshot_after):
        """Set the shape from a before- and after-stroke pair of snapshots

        :param snapshot_before: Snapshot state before the stroke was made
        :param snapshot_after: Snapshot state after the stroke was made
        """
        assert not self.strokemap
        # extract the layer from each snapshot
        a, b = snapshot_before.tiledict, snapshot_after.tiledict
        # enumerate all tiles that have changed
        a_tiles = set(a.iteritems())
        b_tiles = set(b.iteritems())
        changes = a_tiles.symmetric_difference(b_tiles)
        tiles_modified = set([pos for pos, data in changes])

        # for each tile, calculate the exact difference (not now, later, when idle)
        for tx, ty in tiles_modified:
            func = self._update_strokemap_with_percept_diff
            self.tasks.add_work(func, a, b, tx, ty)


    def _update_strokemap_with_percept_diff(self, before, after, tx, ty):
        # get the pixel data to compare
        data_before = before.get((tx, ty), tiledsurface.transparent_tile).rgba
        data_after = after.get((tx, ty), tiledsurface.transparent_tile).rgba
        # calculate pixel changes, and add to the stroke's tiled bitmap
        differences = empty((N, N), 'uint8')
        mypaintlib.tile_perceptual_change_strokemap(data_before, data_after,
                                                    differences)
        self.strokemap[tx, ty] = zlib.compress(differences.tostring())


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


    @staticmethod
    def _translate_tile(src, src_tx, src_ty, slices_x, slices_y,
                        targ_strokemap):
        """Idle task: translate a single tile into an output strokemap"""
        src = fromstring(zlib.decompress(src), dtype='uint8')
        src.shape = (N, N)
        is_integral = len(slices_x) == 1 and len(slices_y) == 1
        for (src_x0, src_x1), (targ_tdx, targ_x0, targ_x1) in slices_x:
            for (src_y0, src_y1), (targ_tdy, targ_y0, targ_y1) in slices_y:
                targ_tx = src_tx + targ_tdx
                targ_ty = src_ty + targ_tdy
                if is_integral:
                    targ_strokemap[targ_tx, targ_ty] = src
                else:
                    targ = targ_strokemap.get((targ_tx, targ_ty), None)
                    if targ is None:
                        targ = zeros((N, N), 'uint8')
                        targ_strokemap[targ_tx, targ_ty] = targ
                    targ[targ_y0:targ_y1, targ_x0:targ_x1] \
                      = src[src_y0:src_y1, src_x0:src_x1]


    def _recompress_tile(self, tx, ty, data):
        """Idle task: recompress a single translated tile's data"""
        if not data.any():
            return
        self.strokemap[tx, ty] = zlib.compress(data.tostring())


    def _start_tile_recompression(self, src_strokemap):
        """Idle task: starts recompressing data from the temp strokemap"""
        for (tx, ty), data in src_strokemap.iteritems():
            self.tasks.add_work(self._recompress_tile, tx, ty, data)


    def translate(self, dx, dy):
        """Translate the shape by (dx, dy)"""
        # Finish any previous translations or handling of painted strokes
        self.tasks.finish_all()
        # Source data
        src_strokemap = self.strokemap
        self.strokemap = {}
        slices_x = tiledsurface.calc_translation_slices(int(dx))
        slices_y = tiledsurface.calc_translation_slices(int(dy))
        # Temporary working strokemap, uncompressed
        tmp_strokemap = {}
        # Queue moves
        for (src_tx, src_ty), src in src_strokemap.iteritems():
            self.tasks.add_work(self._translate_tile, src,
                                src_tx, src_ty, slices_x, slices_y,
                                tmp_strokemap)
        # Recompression of any tile can only start after all the above is
        # complete. Luckily the idle-processor does things in order.
        self.tasks.add_work(self._start_tile_recompression, tmp_strokemap)


    def trim(self, rect):
        """Trim the shape to a rectangle, discarding data outside it

        :param rect: A trimming rectangle in model coordinates
        :type rect: tuple (x, y, w, h)
        :returns: Whether anything remains after the trim
        :rtype: bool

        Only complete tiles are discarded by this method.
        """
        self.tasks.finish_all()
        x, y, w, h = rect
        logger.debug("Trimming stroke to %dx%d%+d%+d", w, h, x, y)
        for tx, ty in list(self.strokemap.keys()):
            if tx*N+N < x or ty*N+N < y or tx*N > x+w or ty*N > y+h:
                self.strokemap.pop((tx, ty))
        return bool(self.strokemap)
