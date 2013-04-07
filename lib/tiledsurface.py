# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This module implements an unbounded tiled surface for painting.

import numpy
from numpy import *
import time
import sys
import os
import contextlib

import mypaintlib
import helpers
import math

TILE_SIZE = N = mypaintlib.TILE_SIZE
MAX_MIPMAP_LEVEL = 4

use_gegl = True if os.environ.get('MYPAINT_ENABLE_GEGL', 0) else False

from layer import DEFAULT_COMPOSITE_OP

import pixbufsurface


class Tile:
    def __init__(self, copy_from=None):
        # note: pixels are stored with premultiplied alpha
        #       15bits are used, but fully opaque or white is stored as 2**15 (requiring 16 bits)
        #       This is to allow many calcuations to divide by 2**15 instead of (2**16-1)
        if copy_from is None:
            self.rgba = zeros((N, N, 4), 'uint16')
        else:
            self.rgba = copy_from.rgba.copy()
        self.readonly = False

    def copy(self):
        return Tile(copy_from=self)


svg2composite_func = {
    'svg:src-over': mypaintlib.tile_composite_normal,
    'svg:multiply': mypaintlib.tile_composite_multiply,
    'svg:screen': mypaintlib.tile_composite_screen,
    'svg:overlay': mypaintlib.tile_composite_overlay,
    'svg:darken': mypaintlib.tile_composite_darken,
    'svg:lighten': mypaintlib.tile_composite_lighten,
    'svg:hard-light': mypaintlib.tile_composite_hard_light,
    'svg:soft-light': mypaintlib.tile_composite_soft_light,
    'svg:color-burn': mypaintlib.tile_composite_color_burn,
    'svg:color-dodge': mypaintlib.tile_composite_color_dodge,
    'svg:difference': mypaintlib.tile_composite_difference,
    'svg:exclusion': mypaintlib.tile_composite_exclusion,
    'svg:hue': mypaintlib.tile_composite_hue,
    'svg:saturation': mypaintlib.tile_composite_saturation,
    'svg:color': mypaintlib.tile_composite_color,
    'svg:luminosity': mypaintlib.tile_composite_luminosity,
    }

# tile for read-only operations on empty spots
transparent_tile = Tile()
transparent_tile.readonly = True

# tile with invalid pixel memory (needs refresh)
mipmap_dirty_tile = Tile()
del mipmap_dirty_tile.rgba

def get_tiles_bbox(tiles):
    res = helpers.Rect()
    for tx, ty in tiles:
        res.expandToIncludeRect(helpers.Rect(N*tx, N*ty, N, N))
    return res

class SurfaceSnapshot:
    pass

if use_gegl:

    class GeglSurface(mypaintlib.GeglBackedSurface):

        def __init__(self, mipmap_level=0):
            mypaintlib.GeglBackedSurface.__init__(self, self)
            self.observers = []

        def notify_observers(self, *args):
            for f in self.observers:
                f(*args)

        def get_bbox(self):
            rect = helpers.Rect(*self.get_bbox_c())
            return rect

        def clear(self):
            pass

        def save_as_png(self, path, *args, **kwargs):
            return self.save_as_png_c(str(path))

        def load_from_png(self, path, x, y, *args, **kwargs):
            return self.load_from_png_c(str(path))

        def save_snapshot(self):
            sshot = SurfaceSnapshot()
            sshot.tiledict = {}
            return sshot

        def load_snapshot(self, sshot):
            pass

        def is_empty(self):
            return False

        def remove_empty_tiles(self):
            pass

        def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0, opacity=1.0,
                           mode=DEFAULT_COMPOSITE_OP):
            pass

        def load_from_numpy(self, arr, x, y):
            return (0, 0, 0, 0)

        def load_from_surface(self, other):
            pass

        def get_tiles(self):
            return {}

        def set_symmetry_state(self, enabled, center_axis):
            pass

class MyPaintSurface(mypaintlib.TiledSurface):
    # the C++ half of this class is in tiledsurface.hpp
    def __init__(self, mipmap_level=0, looped=False, looped_size=(0,0)):
        mypaintlib.TiledSurface.__init__(self, self)
        self.tiledict = {}
        self.observers = []

        # Used to implement repeating surfaces, like Background
        if looped_size[0] % N or looped_size[1] % N:
            raise ValueError, 'Looped size must be multiples of tile size'
        self.looped = looped
        self.looped_size = looped_size

        self.mipmap_level = mipmap_level
        self.mipmap = None
        self.parent = None

        if mipmap_level < MAX_MIPMAP_LEVEL:
            self.mipmap = Surface(mipmap_level+1)
            self.mipmap.parent = self

    def notify_observers(self, *args):
        for f in self.observers:
            f(*args)

    def clear(self):
        tiles = self.tiledict.keys()
        self.tiledict = {}
        self.notify_observers(*get_tiles_bbox(tiles))
        if self.mipmap: self.mipmap.clear()

    @contextlib.contextmanager
    def tile_request(self, tx, ty, readonly):
        """Context manager that fetches a tile as a NumPy array,
        and then puts the potentially modified tile back into the
        tile backing store. To be used with the 'with' statement."""

        numpy_tile = self._get_tile_numpy(tx, ty, readonly)
        yield numpy_tile
        self._set_tile_numpy(tx, ty, numpy_tile, readonly)

    def _get_tile_numpy(self, tx, ty, readonly):
        # OPTIMIZE: do some profiling to check if this function is a bottleneck
        #           yes it is
        # Note: we must return memory that stays valid for writing until the
        # last end_atomic(), because of the caching in tiledsurface.hpp.

        if self.looped:
            tx = tx % (self.looped_size[0] / N)
            ty = ty % (self.looped_size[1] / N)

        t = self.tiledict.get((tx, ty))
        if t is None:
            if readonly:
                t = transparent_tile
            else:
                t = Tile()
                self.tiledict[(tx, ty)] = t
        if t is mipmap_dirty_tile:
            # regenerate mipmap
            t = Tile()
            self.tiledict[(tx, ty)] = t
            empty = True
            for x in xrange(2):
                for y in xrange(2):
                    with self.parent.tile_request(tx*2 + x, ty*2 + y, True) as src:
                        mypaintlib.tile_downscale_rgba16(src, t.rgba, x*N/2, y*N/2)
                        if src is not transparent_tile.rgba:
                            empty = False
            if empty:
                # rare case, no need to speed it up
                del self.tiledict[(tx, ty)]
                t = transparent_tile
        if t.readonly and not readonly:
            # shared memory, get a private copy for writing
            t = t.copy()
            self.tiledict[(tx, ty)] = t
        if not readonly:
            # assert self.mipmap_level == 0
            self._mark_mipmap_dirty(tx, ty)
        return t.rgba

    def _set_tile_numpy(self, tx, ty, obj, readonly):
        pass # Data can be modified directly, no action needed

    def _mark_mipmap_dirty(self, tx, ty):
        if self.mipmap_level > 0:
            self.tiledict[(tx, ty)] = mipmap_dirty_tile
        if self.mipmap:
            self.mipmap._mark_mipmap_dirty(tx/2, ty/2)

    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, mipmap_level=0):
        # used mainly for saving (transparent PNG)

        #assert dst_has_alpha is True

        if self.mipmap_level < mipmap_level:
            return self.mipmap.blit_tile_into(dst, dst_has_alpha, tx, ty, mipmap_level)

        assert dst.shape[2] == 4

        with self.tile_request(tx, ty, readonly=True) as src:

            if src is transparent_tile.rgba:
                #dst[:] = 0 # <-- notably slower than memset()
                mypaintlib.tile_clear(dst)
            else:

                if dst.dtype == 'uint16':
                    # this will do memcpy, not worth to bother skipping the u channel
                    mypaintlib.tile_copy_rgba16_into_rgba16(src, dst)
                elif dst.dtype == 'uint8':
                    if dst_has_alpha:
                        mypaintlib.tile_convert_rgba16_to_rgba8(src, dst)
                    else:
                        mypaintlib.tile_convert_rgbu16_to_rgbu8(src, dst)
                else:
                    raise ValueError, 'Unsupported destination buffer type'

    def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0, opacity=1.0,
                       mode=DEFAULT_COMPOSITE_OP):
        """Composite one tile of this surface over a NumPy array.

        Composite one tile of this surface over the array dst, modifying only dst.
        """
        if self.mipmap_level < mipmap_level:
            return self.mipmap.composite_tile(dst, dst_has_alpha, tx, ty, mipmap_level, opacity, mode)
        if not (tx,ty) in self.tiledict:
            return

        with self.tile_request(tx, ty, readonly=True) as src:
            func = svg2composite_func[mode]
            func(src, dst, dst_has_alpha, opacity)

    def save_snapshot(self):
        sshot = SurfaceSnapshot()
        for t in self.tiledict.itervalues():
            t.readonly = True
        sshot.tiledict = self.tiledict.copy()
        return sshot

    def load_snapshot(self, sshot):
        self._load_tiledict(sshot.tiledict)

    def _load_tiledict(self, d):
        if d == self.tiledict:
            # common case optimization, called from split_stroke() via stroke.redo()
            # testcase: comparison above (if equal) takes 0.6ms, code below 30ms
            return
        old = set(self.tiledict.iteritems())
        self.tiledict = d.copy()
        new = set(self.tiledict.iteritems())
        dirty = old.symmetric_difference(new)
        for pos, tile in dirty:
            self._mark_mipmap_dirty(*pos)
        bbox = get_tiles_bbox([pos for (pos, tile) in dirty])
        if not bbox.empty():
            self.notify_observers(*bbox)

    def load_from_surface(self, other):
        self.load_snapshot(other.save_snapshot())

    def _load_from_pixbufsurface(self, s):
        dirty_tiles = set(self.tiledict.keys())
        self.tiledict = {}

        for tx, ty in s.get_tiles():
            with self.tile_request(tx, ty, readonly=False) as dst:
                s.blit_tile_into(dst, True, tx, ty)

        dirty_tiles.update(self.tiledict.keys())
        bbox = get_tiles_bbox(dirty_tiles)
        self.notify_observers(*bbox)

    def load_from_numpy(self, arr, x, y):
        h, w, channels = arr.shape
        if h <= 0 or w <= 0:
            return (x, y, w, h)

        if arr.dtype == 'uint8':
            s = pixbufsurface.Surface(x, y, w, h, data=arr)
            self._load_from_pixbufsurface(s)
        elif arr.dtype == 'uint16':
            # We only support this for backgrounds, which are tile-aligned
            assert w % N == 0 and h % N == 0
            assert x == 0 and y == 0
            for ty in range(h/N):
                for tx in range(w/N):
                    with self.tile_request(tx, ty, readonly=False) as dst:
                        dst[:,:,:] = arr[ty*N:(ty+1)*N, tx*N:(tx+1)*N, :]
        else:
            raise ValueError

        return (x, y, w, h)

    def load_from_png(self, filename, x, y, feedback_cb=None):
        """Load from a PNG, one tilerow at a time, discarding empty tiles.
        """
        dirty_tiles = set(self.tiledict.keys())
        self.tiledict = {}

        state = {}
        state['buf'] = None # array of height N, width depends on image
        state['ty'] = y/N # current tile row being filled into buf
        state['frame_size'] = None

        def get_buffer(png_w, png_h):
            state['frame_size'] = x, y, png_w, png_h
            if feedback_cb:
                feedback_cb()
            buf_x0 = x/N*N
            buf_x1 = ((x+png_w-1)/N+1)*N
            buf_y0 = state['ty']*N
            buf_y1 = buf_y0+N
            buf_w = buf_x1-buf_x0
            buf_h = buf_y1-buf_y0
            assert buf_w % N == 0
            assert buf_h == N
            if state['buf'] is not None:
                consume_buf()
            else:
                state['buf'] = empty((buf_h, buf_w, 4), 'uint8')

            png_x0 = x
            png_x1 = x+png_w
            subbuf = state['buf'][:,png_x0-buf_x0:png_x1-buf_x0]
            if 1: # optimize: only needed for first and last
                state['buf'].fill(0)
                png_y0 = max(buf_y0, y)
                png_y1 = min(buf_y0+buf_h, y+png_h)
                assert png_y1 > png_y0
                subbuf = subbuf[png_y0-buf_y0:png_y1-buf_y0,:]

            state['ty'] += 1
            return subbuf

        def consume_buf():
            ty = state['ty']-1
            for i in xrange(state['buf'].shape[1]/N):
                tx = x/N + i
                src = state['buf'][:,i*N:(i+1)*N,:]
                if src[:,:,3].any():
                    with self.tile_request(tx, ty, readonly=False) as dst:
                        mypaintlib.tile_convert_rgba8_to_rgba16(src, dst)

        filename_sys = filename.encode(sys.getfilesystemencoding()) # FIXME: should not do that, should use open(unicode_object)
        flags = mypaintlib.load_png_fast_progressive(filename_sys, get_buffer)
        consume_buf() # also process the final chunk of data
        print flags

        dirty_tiles.update(self.tiledict.keys())
        bbox = get_tiles_bbox(dirty_tiles)
        self.notify_observers(*bbox)

        # return the bbox of the loaded image
        return state['frame_size']

    def render_as_pixbuf(self, *args, **kwargs):
        if not self.tiledict:
            print 'WARNING: empty surface'
        t0 = time.time()
        kwargs['alpha'] = True
        res = pixbufsurface.render_as_pixbuf(self, *args, **kwargs)
        print '  %.3fs rendering layer as pixbuf' % (time.time() - t0)
        return res

    def save_as_png(self, filename, *args, **kwargs):
        if not 'alpha' in kwargs:
            kwargs['alpha'] = True

        if len(self.tiledict) == 1:
            kwargs['single_tile_pattern'] = True
        pixbufsurface.save_as_png(self, filename, *args, **kwargs)

    def get_tiles(self):
        return self.tiledict

    def get_bbox(self):
        return get_tiles_bbox(self.tiledict)

    def is_empty(self):
        return not self.tiledict

    def remove_empty_tiles(self):
        # Only used in tests
        for pos, data in self.tiledict.items():
            if not data.rgba.any():
                self.tiledict.pop(pos)

    def get_move(self, x, y):
        return _InteractiveMove(self, x, y)


class _InteractiveMove:

    def __init__(self, surface, x, y):
        self.surface = surface
        self.snapshot = surface.save_snapshot()
        self.chunks = self.snapshot.tiledict.keys()
        # print "Number of Tiledict_keys", len(self.chunks)
        tx = x // N
        ty = y // N
        chebyshev = lambda p: max(abs(tx - p[0]), abs(ty - p[1]))
        manhattan = lambda p: abs(tx - p[0]) + abs(ty - p[1])
        euclidean = lambda p: math.sqrt((tx - p[0])**2 + (ty - p[1])**2)
        self.chunks.sort(key=manhattan)
        self.chunks_i = 0

    def update(self, dx, dy):
        # Tiles to be blanked at the end of processing
        self.blanked = set(self.surface.tiledict.keys())
        # Calculate offsets
        self.slices_x = calc_translation_slices(int(dx))
        self.slices_y = calc_translation_slices(int(dy))
        self.chunks_i = 0

    def cleanup(self):
        # called at the end of each set of processing batches
        for b in self.blanked:
            self.surface.tiledict.pop(b, None)
            self.surface._mark_mipmap_dirty(*b)
        bbox = get_tiles_bbox(self.blanked)
        self.surface.notify_observers(*bbox)
        # Remove empty tile created by Layer Move
        self.surface.remove_empty_tiles()

    def process(self, n=200):
        if self.chunks_i > len(self.chunks):
            return False
        written = set()
        if n <= 0:
            n = len(self.chunks)  # process all remaining
        for tile_pos in self.chunks[self.chunks_i : self.chunks_i + n]:
            src_tx, src_ty = tile_pos
            src_tile = self.snapshot.tiledict[(src_tx, src_ty)]
            is_integral = len(self.slices_x) == 1 and len(self.slices_y) == 1
            for (src_x0, src_x1), (targ_tdx, targ_x0, targ_x1) in self.slices_x:
                for (src_y0, src_y1), (targ_tdy, targ_y0, targ_y1) in self.slices_y:
                    targ_tx = src_tx + targ_tdx
                    targ_ty = src_ty + targ_tdy
                    if is_integral:
                        self.surface.tiledict[(targ_tx, targ_ty)] = src_tile.copy()
                    else:
                        targ_tile = None
                        if (targ_tx, targ_ty) in self.blanked:
                            targ_tile = Tile()
                            self.surface.tiledict[(targ_tx, targ_ty)] = targ_tile
                            self.blanked.remove( (targ_tx, targ_ty) )
                        else:
                            targ_tile = self.surface.tiledict.get((targ_tx, targ_ty), None)
                        if targ_tile is None:
                            targ_tile = Tile()
                            self.surface.tiledict[(targ_tx, targ_ty)] = targ_tile
                        targ_tile.rgba[targ_y0:targ_y1, targ_x0:targ_x1] = src_tile.rgba[src_y0:src_y1, src_x0:src_x1]
                    written.add((targ_tx, targ_ty))
        self.blanked -= written
        for pos in written:
            self.surface._mark_mipmap_dirty(*pos)
        bbox = get_tiles_bbox(written) # hopefully relatively contiguous
        self.surface.notify_observers(*bbox)
        self.chunks_i += n
        return self.chunks_i < len(self.chunks)


def calc_translation_slices(dc):
    """Returns a list of offsets and slice extents for a translation of `dc`.

    The returned slice list's members are of the form

        ((src_c0, src_c1), (targ_tdc, targ_c0, targ_c1))

    where ``src_c0`` and ``src_c1`` determine the extents of the source slice
    within a tile, their ``targ_`` equivalents specify where to put that slice
    in the target tile, and ``targ_tdc`` is the tile offset.
    """
    dcr = dc % N
    tdc = (dc // N)
    if dcr == 0:
        return [ ((0, N), (tdc, 0, N)) ]
    else:
        return [ ((0, N-dcr), (tdc, dcr, N)) ,
                 ((N-dcr, N), (tdc+1, 0, dcr)) ]

# Set which surface backend to use
Surface = GeglSurface if use_gegl else MyPaintSurface

def new_surface():
    return Surface()


class BackgroundError(Exception):
    pass

class Background(Surface):
    """ """

    def __init__(self, obj, mipmap_level=0):

        if not isinstance(obj, numpy.ndarray):
            r, g, b = obj
            obj = numpy.zeros((N, N, 3), dtype='uint8')
            obj[:,:,:] = r, g, b

        height, width = obj.shape[0:2]
        if height % N or width % N:
            raise BackgroundError, 'unsupported background tile size: %dx%d' % (width, height)

        Surface.__init__(self, mipmap_level=0,
                                      looped=True, looped_size=(width, height))
        self.load_from_numpy(obj, 0, 0)

        # Generate mipmap
        if mipmap_level <= MAX_MIPMAP_LEVEL:
            mipmap_obj = numpy.zeros((height, width, 4), dtype='uint16')
            for ty in range(height/N*2):
                for tx in range(width/N*2):
                    with self.tile_request(tx, ty, readonly=True) as src:
                        mypaintlib.tile_downscale_rgba16(src, mipmap_obj, tx*N/2, ty*N/2)

            self.mipmap = Background(mipmap_obj, mipmap_level+1)
            self.mipmap.parent = self
            self.mipmap_level = mipmap_level
