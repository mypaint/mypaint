# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import sys
import contextlib
import numpy
from logging import getLogger
logger = getLogger(__name__)

from gi.repository import GdkPixbuf

import mypaintlib
import helpers

TILE_SIZE = N = mypaintlib.TILE_SIZE


class Surface (object):
    """Wrapper for a GdkPixbuf, with memory accessible by tile.

    Wraps a GdkPixbuf.Pixbuf (8 bit RGBU or RGBA data) with memory also
    accessible per-tile, compatible with tiledsurface.Surface.

    This class converts between linear 8bit RGB(A) and tiled RGBA storage. It
    is used for rendering updates, but also for save/load.

    """

    def __init__(self, x, y, w, h, data=None):
        object.__init__(self)
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

        self.epixbuf = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, True, 8,
                                            self.ew, self.eh)
        dx = x-self.ex
        dy = y-self.ey
        self.pixbuf = self.epixbuf.new_subpixbuf(dx, dy, w, h)

        assert self.ew <= w + 2*N-2
        assert self.eh <= h + 2*N-2

        self.epixbuf.fill(0x00000000)  # keep undefined regions transparent

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

        self.tile_memory_dict = {}
        for ty in range(th):
            for tx in range(tw):
                buf = arr[ty*N:(ty+1)*N, tx*N:(tx+1)*N, :]
                if discard_transparent and not buf[:, :, 3].any():
                    continue
                self.tile_memory_dict[(self.tx+tx, self.ty+ty)] = buf

    def get_tiles(self):
        return self.tile_memory_dict.keys()

    @contextlib.contextmanager
    def tile_request(self, tx, ty, readonly):
        # Interface compatible with that of TiledSurface
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

# throttle excesssive calls to the save/render feedback_cb
TILES_PER_CALLBACK = 256


def render_as_pixbuf(surface, *rect, **kwargs):
    """Renders a surface within a given rectangle as a GdkPixbuf

    :param surface: Any Surface-like object with a ``blit_tile_into()`` method
    :param *rect: x, y, w, h positional args defining the render rectangle
    :param **kwargs: Keyword args are passed to ``surface.blit_tile_into()``
    :rtype: GdkPixbuf

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
            if feedback_cb and tn % TILES_PER_CALLBACK == 0:
                feedback_cb()
            tn += 1
    return s.pixbuf


def save_as_png(surface, filename, *rect, **kwargs):
    """Saves a surface to a file in PNG format

    :param lib.pixbufsurface.Surface surface: Surface to save
    :param str filename: The file to wrote
    :param tuple \*rect: Rectangle (x, y, w, h) to save
    :param bool alpha: If true, write a PNG with alpha
    :param callable feedback_cb: Called every TILES_PER_CALLBACK tiles.
    :param bool single_tile_pattern: True if surface is a one tile only.
    :param tuple \*\*kwargs: Passed to blit_tile_into (minus the above)

    The `alpha` parameter is passed to the surface's `blit_tile_into()`
    method, as well as to `save_png_fast_progressive()`.  Rendering is
    skipped for all but the first line for single-tile patterns.
    If `*rect` is left unspecified, the surface's own bounding box will
    be used.

    """
    alpha = kwargs.pop('alpha', False)
    feedback_cb = kwargs.pop('feedback_cb', None)
    single_tile_pattern = kwargs.pop("single_tile_pattern", False)
    if not rect:
        rect = surface.get_bbox()
    x, y, w, h = rect
    if w == 0 or h == 0:
        # workaround to save empty documents
        x, y, w, h = 0, 0, 1, 1

    # calculate bounding box in full tiles
    render_tx = x/N
    render_ty = y/N
    render_tw = (x+w-1)/N - render_tx + 1
    render_th = (y+h-1)/N - render_ty + 1

    # buffer for rendering one tile row at a time
    arr = numpy.empty((1*N, render_tw*N, 4), 'uint8')  # rgba or rgbu
    # view into arr without the horizontal padding
    arr_xcrop = arr[:, x-render_tx*N:x-render_tx*N+w, :]

    first_row = render_ty
    last_row = render_ty+render_th-1

    def render_tile_scanlines():
        feedback_counter = 0
        for ty in range(render_ty, render_ty+render_th):
            skip_rendering = False
            if single_tile_pattern:
                # optimization for simple background patterns
                # e.g. solid color
                if ty != first_row:
                    skip_rendering = True

            for tx_rel in xrange(render_tw):
                # render one tile
                dst = arr[:, tx_rel*N:(tx_rel+1)*N, :]
                if not skip_rendering:
                    tx = render_tx + tx_rel
                    try:
                        surface.blit_tile_into(dst, alpha, tx, ty, **kwargs)
                    except Exception:
                        logger.exception("Failed to blit tile %r of %r",
                                         (tx, ty), surface)
                        mypaintlib.tile_clear_rgba8(dst)
                if feedback_cb and feedback_counter % TILES_PER_CALLBACK == 0:
                    feedback_cb()
                feedback_counter += 1

            # yield a numpy array of the scanline without padding
            res = arr_xcrop
            if ty == last_row:
                res = res[:y+h-ty*N, :, :]
            if ty == first_row:
                res = res[y-render_ty*N:, :, :]
            yield res

    filename_sys = filename.encode(sys.getfilesystemencoding())
    # FIXME: should not do that, should use open(unicode_object)
    mypaintlib.save_png_fast_progressive(filename_sys, w, h, alpha,
                                         render_tile_scanlines(),
                                         False)
