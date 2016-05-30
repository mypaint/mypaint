# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports
from __future__ import print_function

import sys
import contextlib
from logging import getLogger
logger = getLogger(__name__)

from gettext import gettext as _
from gi.repository import GdkPixbuf

import mypaintlib
import helpers
import lib.surface
from lib.surface import TileAccessible, TileBlittable
from lib.errors import FileHandlingError
from lib.errors import AllocationError
from lib.gettext import C_


## Module consts

TILE_SIZE = N = mypaintlib.TILE_SIZE

_POSSIBLE_OOM_USERTEXT = C_(
    "user-facing error texts",
    u"Unable to construct a vital internal object. "
    u"Your system may not have enough memory to perform "
    u"this operation."
)


## Class defs

class Surface (TileAccessible, TileBlittable):
    """Wrapper for a GdkPixbuf, with memory accessible by tile.

    Wraps a GdkPixbuf.Pixbuf (8 bit RGBU or RGBA data) with memory also
    accessible per-tile, compatible with tiledsurface.Surface.

    This class converts between linear 8bit RGB(A) and tiled RGBA storage. It
    is used for rendering updates, but also for save/load.

    """

    def __init__(self, x, y, w, h, data=None):
        super(Surface, self).__init__()
        assert w > 0 and h > 0
        # We create and use a pixbuf enlarged to the tile boundaries internally.
        # Variables ex, ey, ew, eh and epixbuf store the enlarged version.
        self.x, self.y, self.w, self.h = x, y, w, h
        #print x, y, w, h
        tx = self.tx = x/N
        ty = self.ty = y/N
        self.ex = tx*N
        self.ey = ty*N
        tw = (x+w-1)/N - tx + 1
        th = (y+h-1)/N - ty + 1

        self.ew = tw*N
        self.eh = th*N

        #print 'b:', self.ex, self.ey, self.ew, self.eh
        # OPTIMIZE: remove assertions here?
        assert self.ew >= w and self.eh >= h
        assert self.ex <= x and self.ey <= y

        # Tile-aligned pixbuf: also accessible by tile
        try:
            self.epixbuf = GdkPixbuf.Pixbuf.new(
                GdkPixbuf.Colorspace.RGB, True, 8,
                self.ew, self.eh
            )
        except Exception as te:
            logger.exception("GdkPixbuf.Pixbuf.new() failed")
            raise AllocationError(_POSSIBLE_OOM_USERTEXT)
        if self.epixbuf is None:
            logger.error("GdkPixbuf.Pixbuf.new() returned NULL")
            raise AllocationError(_POSSIBLE_OOM_USERTEXT)

        # External subpixbuf, also accessible by tile.
        dx = x-self.ex
        dy = y-self.ey
        try:
            self.pixbuf = self.epixbuf.new_subpixbuf(dx, dy, w, h)
        except Exception as te:
            logger.exception("GdkPixbuf.Pixbuf.new_subpixbuf() failed")
            raise AllocationError(_POSSIBLE_OOM_USERTEXT)
        if self.pixbuf is None:
            logger.error("GdkPixbuf.Pixbuf.new_subpixbuf() returned NULL")
            raise AllocationError(_POSSIBLE_OOM_USERTEXT)

        assert self.ew <= w + 2*N-2
        assert self.eh <= h + 2*N-2

        self.epixbuf.fill(0x00000000)  # keep undefined regions transparent

        # Make it accessible by tile
        arr = helpers.gdkpixbuf2numpy(self.epixbuf)
        assert len(arr) > 0

        discard_transparent = False

        if data is not None:
            dst = arr[dy:dy+h, dx:dx+w, :]
            if data.shape[2] == 4:
                dst[:, :, :] = data
                discard_transparent = True
            else:
                assert data.shape[2] == 3
                # no alpha channel
                dst[:, :, :3] = data
                dst[:, :, 3] = 255

        # Build (tx,ty)-indexed access struct
        self.tile_memory_dict = {}
        for ty in range(th):
            for tx in range(tw):
                buf = arr[ty*N:(ty+1)*N, tx*N:(tx+1)*N, :]
                if discard_transparent and not buf[:, :, 3].any():
                    continue
                self.tile_memory_dict[(self.tx+tx, self.ty+ty)] = buf

    def get_bbox(self):
        return lib.surface.get_tiles_bbox(self.get_tiles())

    def get_tiles(self):
        return self.tile_memory_dict

    @contextlib.contextmanager
    def tile_request(self, tx, ty, readonly):
        """Access memory by tile (lib.surface.TileAccessible impl.)"""
        numpy_tile = self._get_tile_numpy(tx, ty, readonly)
        yield numpy_tile
        self._set_tile_numpy(tx, ty, numpy_tile, readonly)

    def _get_tile_numpy(self, tx, ty, readonly):
        return self.tile_memory_dict[(tx, ty)]

    def _set_tile_numpy(self, tx, ty, arr, readonly):
        pass  # Data can be modified directly, no action needed

    def blit_tile_into(self, dst, dst_has_alpha, tx, ty):
        # (used mainly for loading transparent PNGs)
        assert dst_has_alpha is True
        assert dst.dtype == 'uint16', '16 bit dst expected'
        src = self.tile_memory_dict[(tx, ty)]
        assert src.shape[2] == 4, 'alpha required'
        mypaintlib.tile_convert_rgba8_to_rgba16(src, dst)


def render_as_pixbuf(surface, *rect, **kwargs):
    """Renders a surface within a given rectangle as a GdkPixbuf

    :param lib.surface.TileBlittable surface: source surface
    :param *rect: x, y, w, h positional args defining the render rectangle
    :param **kwargs: Keyword args are passed to ``surface.blit_tile_into()``
    :rtype: GdkPixbuf
    :raises: lib.errors.AllocationError
    :raises: MemoryError

    The keyword args ``alpha``, ``mipmap_level``, and ``feedback_cb`` are
    consumed here and removed from `**kwargs` before it is passed to the
    Surface's `blit_tile_into()`.
    """
    alpha = kwargs.pop('alpha', False)
    mipmap_level = kwargs.pop('mipmap_level', 0)
    feedback_cb = kwargs.pop('feedback_cb', None)
    if not rect:
        rect = surface.get_bbox()
    x, y, w, h, = rect
    s = Surface(x, y, w, h)
    tn = 0
    for tx, ty in s.get_tiles():
        with s.tile_request(tx, ty, readonly=False) as dst:
            surface.blit_tile_into(dst, alpha, tx, ty,
                                   mipmap_level=mipmap_level,
                                   **kwargs)
            if feedback_cb and tn % lib.surface.TILES_PER_CALLBACK == 0:
                feedback_cb()
            tn += 1
    return s.pixbuf








