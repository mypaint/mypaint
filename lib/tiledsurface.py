# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This module implements an unbounded tiled surface for painting.

from numpy import *
import time, sys
import mypaintlib, helpers

TILE_SIZE = N = mypaintlib.TILE_SIZE
MAX_MIPMAP_LEVEL = mypaintlib.MAX_MIPMAP_LEVEL

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


def composite_array_src_over(dst, src, mipmap_level=0, opacity=1.0):
    """The default "svg:src-over" layer composite op implementation.
    """
    if dst.shape[2] == 4 and dst.dtype == 'uint16':
        # rarely used (for merging layers, also when exporting a transparent PNGs)
        # src (premultiplied) OVER dst (premultiplied)
        # cB = cA + (1.0 - aA) * cB
        #  where B = dst, A = src
        srcAlpha = src[:,:,3:4].astype('float') * opacity / (1 << 15)
        dstAlpha = dst[:,:,3:4].astype('float') / (1 << 15)
        src_premult = src[:,:,0:3].astype('float') * opacity / (1<<15)
        dst_premult = dst[:,:,0:3].astype('float') / (1<<15)
        dst_c = clip(src_premult + (1.0 - srcAlpha) * dst_premult, 0.0, 1.0)
        dst_a = clip(srcAlpha + dstAlpha - srcAlpha * dstAlpha, 0.0, 1.0)
        dst[:,:,0:3] = clip(dst_c * (1<<15), 0, (1<<15) - 1).astype('uint16')
        dst[:,:,3:4] = clip(dst_a * (1<<15), 0, (1<<15) - 1).astype('uint16')
    elif dst.shape[2] == 3 and dst.dtype == 'uint16':
        mypaintlib.tile_composite_rgba16_over_rgb16(src, dst, opacity)
    else:
        raise NotImplementedError


def composite_array_multiply(dst, src, mipmap_level=0, opacity=1.0):
    """The "svg:multiply" layer composite op implementation.
    """
    if dst.shape[2] == 4 and dst.dtype == 'uint16':
        # rarely used (for merging layers, also when exporting a transparent PNGs)
        # src (premultiplied) MULTIPLY dst (premultiplied)
        # cA * cB +  cA * (1 - aB) + cB * (1 - aA)
        srcAlpha = (opacity * src[:,:,3:4]).astype('float') / (1 << 15)
        dstAlpha = (dst[:,:,3:4]).astype('float') / (1 << 15)
        src_premult = src[:,:,0:3].astype('float') * opacity / (1<<15)
        dst_premult = dst[:,:,0:3].astype('float') / (1<<15)
        dst_c = clip(src_premult * dst_premult +  src_premult * (1.0 - dstAlpha) + dst_premult * (1.0 - srcAlpha),0.0, 1.0)
        dst_a = clip(srcAlpha + dstAlpha - srcAlpha * dstAlpha, 0.0, 1.0)
        dst[:,:,0:3] = clip(dst_c * (1<<15), 0, (1<<15) - 1).astype('uint16')
        dst[:,:,3:4] = clip(dst_a * (1<<15), 0, (1<<15) - 1).astype('uint16')
    elif dst.shape[2] == 3 and dst.dtype == 'uint16':
        mypaintlib.tile_composite_rgba16_multiply_rgb16(src, dst, opacity)
    else:
        raise NotImplementedError


def composite_array_screen(dst, src, mipmap_level=0, opacity=1.0):
    """The "svg:screen" layer composite op implementation.
    """
    if dst.shape[2] == 4 and dst.dtype == 'uint16':
        # rarely used (for merging layers, also when exporting a transparent PNGs)
        # src (premultiplied) SCREEN dst (premultiplied)
        # cA + cB - cA * cB
        srcAlpha = (opacity * src[:,:,3:4]).astype('float') / (1 << 15)
        dstAlpha = (dst[:,:,3:4]).astype('float') / (1 << 15)
        src_premult = src[:,:,0:3].astype('float') * opacity / (1<<15)
        dst_premult = dst[:,:,0:3].astype('float') / (1<<15)
        dst_c = clip(src_premult + dst_premult - src_premult * dst_premult, 0, 1)
        dst_a = clip(srcAlpha + dstAlpha - srcAlpha * dstAlpha, 0, 1)
        dst[:,:,0:3] = clip(dst_c * (1<<15),0, (1<<15) - 1).astype('uint16')
        dst[:,:,3:4] = clip(dst_a * (1<<15),0, (1<<15) - 1).astype('uint16')
    elif dst.shape[2] == 3 and dst.dtype == 'uint16':
        mypaintlib.tile_composite_rgba16_screen_rgb16(src, dst, opacity)
    else:
        raise NotImplementedError


def composite_array_burn(dst, src, mipmap_level=0, opacity=1.0):
    """The "svg:color-burn" layer composite op implementation.
    """
    if dst.shape[2] == 4 and dst.dtype == 'uint16':
        # rarely used (for merging layers, also when exporting a transparent PNGs)
        # src (premultiplied) OVER dst (premultiplied)
        # if cA * aB + cB * aA <= aA * aB :
        #   cA * (1 - aB) + cB * (1 - aA)
        #   (cA == 0 ? 1 : (aA * (cA * aB + cB * aA - aA * aB) / cA) + cA * (1 - aB) + cB * (1 - aA))
        #  where B = dst, A = src
        aA = (opacity * src[:,:,3:4]).astype('float') / (1 << 15)
        aB = (dst[:,:,3:4]).astype('float') / (1 << 15)
        cA = src[:,:,0:3].astype('float') * opacity / (1<<15)
        cB = dst[:,:,0:3].astype('float') / (1<<15)
        dst_c = where(cA * aB + cB * aA <= aA * aB,
                     cA * (1 - aB) + cB * (1 - aA),
                     where(cA == 0,
                           1.0,
                           (aA * (cA * aB + cB * aA - aA * aB) / cA) + cA * (1.0 - aB) + cB * (1.0 - aA)))
        dst_a = aA + aB - aA * aB
        dst[:,:,0:3] = clip(dst_c * (1<<15), 0, (1<<15) - 1).astype('uint16')
        dst[:,:,3:4] = clip(dst_a * (1<<15), 0, (1<<15) - 1).astype('uint16')
    elif dst.shape[2] == 3 and dst.dtype == 'uint16':
        mypaintlib.tile_composite_rgba16_burn_rgb16(src, dst, opacity)
    else:
        raise NotImplementedError


def composite_array_dodge(dst, src, mipmap_level=0, opacity=1.0):
    """The "svg:color-dodge" layer composite op implementation.
    """
    if dst.shape[2] == 4 and dst.dtype == 'uint16':
        # rarely used (for merging layers, also when exporting a transparent PNGs)
        # src (premultiplied) OVER dst (premultiplied)
        # if cA * aB + cB * aA >= aA * aB :
        #   aA * aB + cA * (1 - aB) + cB * (1 - aA)
        #   (cA == aA ? 1 : cB * aA / (aA == 0 ? 1 : 1 - cA / aA)) + cA * (1 - aB) + cB * (1 - aA)
        #  where B = dst, A = src
        aA = (opacity * src[:,:,3:4]).astype('float') / (1 << 15)
        aB = (dst[:,:,3:4]).astype('float') / (1 << 15)
        cA = src[:,:,0:3].astype('float') * opacity / (1<<15)
        cB = dst[:,:,0:3].astype('float') / (1<<15)
        dst_c = where(cA * aB + cB * aA >= aA * aB,
                     aA * aB + cA * (1 - aB) + cB * (1 - aA),
                     where(cA == aA, 1.0, cB * aA / where(aA == 0, 1.0, 1.0 - cA / aA)) + cA * (1.0 - aB) + cB * (1.0 - aA))
        dst_a = (aA + aB - aA * aB)
        dst[:,:,0:3] = clip(dst_c * (1<<15),0, (1<<15) - 1).astype('uint16')
        dst[:,:,3:4] = clip(dst_a * (1<<15),0, (1<<15) - 1).astype('uint16')
    elif dst.shape[2] == 3 and dst.dtype == 'uint16':
        mypaintlib.tile_composite_rgba16_dodge_rgb16(src, dst, opacity)
    else:
        raise NotImplementedError


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

class Surface(mypaintlib.TiledSurface):
    # the C++ half of this class is in tiledsurface.hpp
    def __init__(self, mipmap_level=0):
        mypaintlib.TiledSurface.__init__(self, self)
        self.tiledict = {}
        self.observers = []

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

    def get_tile_memory(self, tx, ty, readonly):
        # OPTIMIZE: do some profiling to check if this function is a bottleneck
        #           yes it is
        # Note: we must return memory that stays valid for writing until the
        # last end_atomic(), because of the caching in tiledsurface.hpp.
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
                    src = self.parent.get_tile_memory(tx*2 + x, ty*2 + y, True)
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
            assert self.mipmap_level == 0
            self._mark_mipmap_dirty(tx, ty)
        return t.rgba

    def _mark_mipmap_dirty(self, tx, ty):
        if self.mipmap_level > 0:
            self.tiledict[(tx, ty)] = mipmap_dirty_tile
        if self.mipmap:
            self.mipmap._mark_mipmap_dirty(tx/2, ty/2)

    def blit_tile_into(self, dst, tx, ty, mipmap_level=0):
        # used mainly for saving (transparent PNG)
        if self.mipmap_level < mipmap_level:
            return self.mipmap.blit_tile_into(dst, tx, ty, mipmap_level)
        assert dst.shape[2] == 4
        src = self.get_tile_memory(tx, ty, readonly=True)
        if src is transparent_tile.rgba:
            #dst[:] = 0 # <-- notably slower than memset()
            mypaintlib.tile_clear(dst)
        else:
            mypaintlib.tile_convert_rgba16_to_rgba8(src, dst)


    def composite_tile(self, dst, tx, ty, mipmap_level=0, opacity=1.0,
                       mode=DEFAULT_COMPOSITE_OP):
        """Composite one tile of this surface over a NumPy array.

        Composite one tile of this surface over the array dst, modifying only dst.
        """
        if self.mipmap_level < mipmap_level:
            return self.mipmap.composite_tile(dst, tx, ty, mipmap_level, opacity, mode)
        if not (tx,ty) in self.tiledict:
            return
        src = self.get_tile_memory(tx, ty, readonly=True)

        if mode == 'svg:src-over':
            return composite_array_src_over(dst, src, mipmap_level, opacity)
        elif mode == 'svg:multiply':
            return composite_array_multiply(dst, src, mipmap_level, opacity)
        elif mode == 'svg:screen':
            return composite_array_screen(dst, src, mipmap_level, opacity)
        elif mode == 'svg:color-burn':
            return composite_array_burn(dst, src, mipmap_level, opacity)
        elif mode == 'svg:color-dodge':
            return composite_array_dodge(dst, src, mipmap_level, opacity)
        else:
            raise NotImplementedError, mode

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
            dst = self.get_tile_memory(tx, ty, readonly=False)
            s.blit_tile_into(dst, tx, ty)

        dirty_tiles.update(self.tiledict.keys())
        bbox = get_tiles_bbox(dirty_tiles)
        self.notify_observers(*bbox)

    def load_from_numpy(self, arr, x, y):
        assert arr.dtype == 'uint8'
        h, w, channels = arr.shape
        s = pixbufsurface.Surface(x, y, w, h, alpha=True, data=arr)
        self._load_from_pixbufsurface(s)

    def load_from_png(self, filename, x, y, feedback_cb=None):
        """Load from a PNG, one tilerow at a time, discarding empty tiles.
        """
        dirty_tiles = set(self.tiledict.keys())
        self.tiledict = {}

        state = {}
        state['buf'] = None # array of height N, width depends on image
        state['ty'] = y/N # current tile row being filled into buf

        def get_buffer(png_w, png_h):
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
                    dst = self.get_tile_memory(tx, ty, readonly=False)
                    mypaintlib.tile_convert_rgba8_to_rgba16(src, dst)

        filename_sys = filename.encode(sys.getfilesystemencoding()) # FIXME: should not do that, should use open(unicode_object)
        mypaintlib.load_png_fast_progressive(filename_sys, get_buffer)
        consume_buf() # also process the final chunk of data
        
        dirty_tiles.update(self.tiledict.keys())
        bbox = get_tiles_bbox(dirty_tiles)
        self.notify_observers(*bbox)

    def render_as_pixbuf(self, *args, **kwargs):
        if not self.tiledict:
            print 'WARNING: empty surface'
        t0 = time.time()
        kwargs['alpha'] = True
        res = pixbufsurface.render_as_pixbuf(self, *args, **kwargs)
        print '  %.3fs rendering layer as pixbuf' % (time.time() - t0)
        return res

    def save_as_png(self, filename, *args, **kwargs):
        assert 'alpha' not in kwargs
        kwargs['alpha'] = True
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
