# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This module implements an unbounded tiled surface for painting.

from numpy import *
import time
import mypaintlib, helpers
from gettext import gettext as _

tilesize = N = mypaintlib.TILE_SIZE
MAX_MIPMAP_LEVEL = mypaintlib.MAX_MIPMAP_LEVEL

COMPOSITE_OPS = [
    # (internal-name, display-name)
    ("svg:src-over", _("Normal")),
    ("svg:multiply", _("Multiply")),
    ("svg:color-burn", _("Burn")),
    ("svg:color-dodge", _("Dodge")),
    ("svg:screen", _("Screen")),
    ]

VALID_COMPOSITE_OPS = set([n for n, d in COMPOSITE_OPS])
DEFAULT_COMPOSITE_OP = COMPOSITE_OPS[0][0]


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
            self.mark_mipmap_dirty(tx, ty)
        return t.rgba

    def mark_mipmap_dirty(self, tx, ty):
        if self.mipmap_level > 0:
            self.tiledict[(tx, ty)] = mipmap_dirty_tile
        if self.mipmap:
            self.mipmap.mark_mipmap_dirty(tx/2, ty/2)

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
        if mode == 'svg:src-over':
            return self.composite_tile_over(dst, tx, ty, mipmap_level, opacity)
        elif mode == 'svg:multiply':
            return self.composite_tile_multiply(dst, tx, ty, mipmap_level, opacity)
        elif mode == 'svg:screen':
            return self.composite_tile_screen(dst, tx, ty, mipmap_level, opacity)
        elif mode == 'svg:color-burn':
            return self.composite_tile_burn(dst, tx, ty, mipmap_level, opacity)
        elif mode == 'svg:color-dodge':
            return self.composite_tile_dodge(dst, tx, ty, mipmap_level, opacity)
        else:
            raise NotImplementedError, mode


    def composite_tile_over(self, dst, tx, ty, mipmap_level=0, opacity=1.0, mode=0):
        """The default "svg:src-over" layer composite op implementation.
        """
        if self.mipmap_level < mipmap_level:
            return self.mipmap.composite_tile_over(dst, tx, ty, mipmap_level, opacity)
        if not (tx,ty) in self.tiledict:
            return
        src = self.get_tile_memory(tx, ty, readonly=True)
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


    def composite_tile_multiply(self, dst, tx, ty, mipmap_level=0, opacity=1.0, mode=0):
        """The "svg:multiply" layer composite op implementation.
        """
        if self.mipmap_level < mipmap_level:
            return self.mipmap.composite_tile_multiply(dst, tx, ty, mipmap_level, opacity)
        if not (tx,ty) in self.tiledict:
            return
        src = self.get_tile_memory(tx, ty, readonly=True)
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


    def composite_tile_screen(self, dst, tx, ty, mipmap_level=0, opacity=1.0, mode=0):
        """The "svg:screen" layer composite op implementation.
        """
        if self.mipmap_level < mipmap_level:
            return self.mipmap.composite_tile_screen(dst, tx, ty, mipmap_level, opacity)
        if not (tx,ty) in self.tiledict:
            return
        src = self.get_tile_memory(tx, ty, readonly=True)
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


    def composite_tile_burn(self, dst, tx, ty, mipmap_level=0, opacity=1.0, mode=0):
        """The "svg:color-burn" layer composite op implementation.
        """
        if self.mipmap_level < mipmap_level:
            return self.mipmap.composite_tile_burn(dst, tx, ty, mipmap_level, opacity)
        if not (tx,ty) in self.tiledict:
            return
        src = self.get_tile_memory(tx, ty, readonly=True)
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


    def composite_tile_dodge(self, dst, tx, ty, mipmap_level=0, opacity=1.0, mode=0):
        """The "svg:color-dodge" layer composite op implementation.
        """
        if self.mipmap_level < mipmap_level:
            return self.mipmap.composite_tile_dodge(dst, tx, ty, mipmap_level, opacity)
        if not (tx,ty) in self.tiledict:
            return
        src = self.get_tile_memory(tx, ty, readonly=True)
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


    def save_snapshot(self):
        sshot = SurfaceSnapshot()
        for t in self.tiledict.itervalues():
            t.readonly = True
        sshot.tiledict = self.tiledict.copy()
        return sshot

    def load_snapshot(self, sshot):
        if sshot.tiledict == self.tiledict:
            # common case optimization, called from split_stroke() via stroke.redo()
            # testcase: comparison above (if equal) takes 0.6ms, code below 30ms
            return
        old = set(self.tiledict.iteritems())
        self.tiledict = sshot.tiledict.copy()
        new = set(self.tiledict.iteritems())
        dirty = old.symmetric_difference(new)
        for pos, tile in dirty:
            self.mark_mipmap_dirty(*pos)
        bbox = get_tiles_bbox([pos for (pos, tile) in dirty])
        if not bbox.empty():
            self.notify_observers(*bbox)

    def render_as_pixbuf(self, *args, **kwargs):
        if not self.tiledict:
            print 'WARNING: empty surface'
        t0 = time.time()
        kwargs['alpha'] = True
        res = pixbufsurface.render_as_pixbuf(self, *args, **kwargs)
        print '  %.3fs rendering layer as pixbuf' % (time.time() - t0)
        return res

    def save(self, filename, *args, **kwargs):
        assert 'alpha' not in kwargs
        kwargs['alpha'] = True
        pixbufsurface.save_as_png(self, filename, *args, **kwargs)

    def load_from_pixbufsurface(self, s):
        dirty_tiles = set(self.tiledict.keys())
        self.tiledict = {}

        for tx, ty in s.get_tiles():
            dst = self.get_tile_memory(tx, ty, readonly=False)
            s.blit_tile_into(dst, tx, ty)

        dirty_tiles.update(self.tiledict.keys())
        bbox = get_tiles_bbox(dirty_tiles)
        self.notify_observers(*bbox)
        
    def load_from_data(self, data):
        x, y, data = data
        assert data.dtype == 'uint8'
        h, w, channels = data.shape
        
        s = pixbufsurface.Surface(x, y, w, h, alpha=True, data=data)
        self.load_from_pixbufsurface(s)

    def get_tiles(self):
        return self.tiledict

    def get_bbox(self):
        return get_tiles_bbox(self.tiledict)

    def is_empty(self):
        return not self.tiledict

    def remove_empty_tiles(self):
        for pos, data in self.tiledict.items():
            if not data.rgba.any():
                self.tiledict.pop(pos)
