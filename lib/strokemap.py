# This file is part of MyPaint.
# Copyright (C) 2009-2011 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2011-2015 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

import time
import struct
import zlib
import numpy
from logging import getLogger
logger = getLogger(__name__)

import mypaintlib

import tiledsurface
import idletask

TILE_SIZE = N = mypaintlib.TILE_SIZE


## Class defs

class StrokeShape (object):
    """The shape of a single brushstroke.

    This class stores the shape of a stroke in as a 1-bit bitmap. The
    information is stored in compressed memory blocks of the size of a
    tile (for fast lookup).

    """
    def __init__(self):
        """Construct a new, blank StrokeShape."""
        object.__init__(self)
        self.tasks = idletask.Processor()
        self.strokemap = {}
        self.brush_string = None

    @classmethod
    def new_from_snapshots(cls, before, after):
        """Build a new StrokeShape from before+after pair of snapshots.

        :param before: snapshot of the layer before the stroke
        :type before: lib.tiledsurface._TiledSurfaceSnapshot
        :param after: snapshot of the layer after the stroke
        :type after: lib.tiledsurface._TiledSurfaceSnapshot
        :returns: A new StrokeShape, or None.

        If the snapshots haven't changed, None is returned. In this
        case, no StrokeShape should be recorded.

        """
        before_dict = before.tiledict
        after_dict = after.tiledict
        before_tiles = set(before_dict.iteritems())
        after_tiles = set(after_dict.iteritems())
        changed_idxs = set(
            pos for pos, data
            in before_tiles.symmetric_difference(after_tiles)
        )
        if not changed_idxs:
            return None
        shape = cls()
        assert not shape.strokemap
        shape.tasks.add_work(_TileDiffUpdateTask(
            before.tiledict,
            after.tiledict,
            changed_idxs,
            shape.strokemap,
        ))
        return shape

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
            data = numpy.fromstring(zlib.decompress(data), dtype='uint8')
            data.shape = (N, N)
            return data[y % N, x % N]

    def render_to_surface(self, surf):
        self.tasks.finish_all()
        for (tx, ty), data in self.strokemap.iteritems():
            data = numpy.fromstring(zlib.decompress(data), dtype='uint8')
            data.shape = (N, N)
            with surf.tile_request(tx, ty, readonly=False) as tile:
                # neutral gray, 50% opaque
                tile[:, :, 3] = data.astype('uint16') * (1 << 15)/2
                tile[:, :, 0] = tile[:, :, 3]/2
                tile[:, :, 1] = tile[:, :, 3]/2
                tile[:, :, 2] = tile[:, :, 3]/2

    def translate(self, dx, dy):
        """Translate the shape by (dx, dy)"""
        self.tasks.finish_all()
        tmp = {}
        self.tasks.add_work(_TileTranslateTask(self.strokemap, tmp, dx, dy))
        self.tasks.add_work(_TileRecompressTask(tmp, self.strokemap))

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


class _TileDiffUpdateTask:
    """Idle task: update strokemap with tile & pixel diffs of snapshots.

    This task is used during initialization of the StrokeShape.

    """

    def __init__(self, before, after, changed_idxs, targ):
        """Initialize, ready to update a target StrokeShape with diffs

        :param dict before: Complete pre-stroke tiledict (RO, {xy:Tile})
        :param dict after: Complete post-stroke tiledict (RO, {xy:Tile})
        :param set changed_idxs: RW set of (x,y) tile indexes to process
        :param dict targ: Target strokemap (WO, {xy: bytes})

        """
        self._before_dict = before
        self._after_dict = after
        self._targ_dict = targ
        self._remaining = changed_idxs

    def __repr__(self):
        return "<{name} remaining={remaining}>".format(
            name = self.__class__.__name__,
            remaining = len(self._remaining),
        )

    def __call__(self):
        """Diff and update one queued tile."""
        try:
            ti = self._remaining.pop()
        except KeyError:
            return False
        self._update_tile(ti)
        return bool(self._remaining)

    def _update_tile(self, ti):
        """Diff and update the tile at a specified position."""
        transparent = tiledsurface.transparent_tile
        data_before = self._before_dict.get(ti, transparent).rgba
        data_after = self._after_dict.get(ti, transparent).rgba
        differences = numpy.empty((N, N), 'uint8')
        mypaintlib.tile_perceptual_change_strokemap(
            data_before,
            data_after,
            differences,
        )
        self._targ_dict[ti] = zlib.compress(differences.tostring())


class _TileTranslateTask:
    """Translate/move tiles (compressed strokemap -> uncompressed tmp)

    Calling this task is destructive to the source strokemap, so it must
    be paired with a _TileRecompressTask queued up to fire when it has
    completely finished.

    Tiles are translated by slicing and recombining, so this task must
    be called to completion before the output tiledict will be ready for
    recompression.

    """

    def __init__(self, src, targ, dx, dy):
        """Initialize with source and target.

        :param dict src: compressed strokemap, RW {xy: bytes}
        :param dict targ: uncompressed tiledict, RW {xy: array}
        :param int dx: x offset for the translation, in pixels
        :param int dy: y offset for the translation, in pixels

        """
        self._src = src
        self._targ = targ
        self._dx = int(dx)
        self._dy = int(dy)
        self._slices_x = tiledsurface.calc_translation_slices(self._dx)
        self._slices_y = tiledsurface.calc_translation_slices(self._dy)

    def __repr__(self):
        return "<{name} dx={dx} dy={dy}>".format(
            name = self.__class__.__name__,
            dx = self._dx,
            dy = self._dy,
        )

    def __call__(self):
        """Idle task: translate a single tile into the output dict.

        """
        try:
            (src_tx, src_ty), src = self._src.popitem()
        except KeyError:
            return False
        src = numpy.fromstring(zlib.decompress(src), dtype='uint8')
        src.shape = (N, N)
        slices_x = self._slices_x
        slices_y = self._slices_y
        is_integral = len(slices_x) == 1 and len(slices_y) == 1
        for (src_x0, src_x1), (targ_tdx, targ_x0, targ_x1) in slices_x:
            for (src_y0, src_y1), (targ_tdy, targ_y0, targ_y1) in slices_y:
                targ_tx = src_tx + targ_tdx
                targ_ty = src_ty + targ_tdy
                if is_integral:
                    self._targ[targ_tx, targ_ty] = src
                else:
                    targ = self._targ.get((targ_tx, targ_ty), None)
                    if targ is None:
                        targ = numpy.zeros((N, N), 'uint8')
                        self._targ[targ_tx, targ_ty] = targ
                    targ[targ_y0:targ_y1, targ_x0:targ_x1] \
                        = src[src_y0:src_y1, src_x0:src_x1]
        return bool(self._src)


class _TileRecompressTask:
    """Re-compress data after a move (uncomp. tmp -> comp. strokemap)"""

    def __init__(self, src, targ):
        """Initialize with source and target.

        :param dict src: input uncompressed tiledict (WO, {xy:array})
        :param dict targ: output compressed strokemap (WO, {xy:bytes})

        """
        self._src_dict = src
        self._targ_dict = targ

    def __call__(self):
        """Compress & store an arbitrary queued tile's data."""
        try:
            ti, array = self._src_dict.popitem()
        except KeyError:
            return False
        self._compress_tile(ti, array)
        return len(self._src_dict) > 0

    def _compress_tile(self, ti, array):
        if not array.any():
            return
        self._targ_dict[ti] = zlib.compress(array.tostring())

    def __repr__(self):
        return "<{name} remaining={n}>".format(
            name = self.__class__.__name__,
            n = len(self._src_dict),
        )
