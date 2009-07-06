# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This module implements an an unbounded tiled surface for painting.

from numpy import *
import time
import mypaintlib, helpers

tilesize = N = mypaintlib.TILE_SIZE
MAX_MIPMAP_LEVEL = mypaintlib.MAX_MIPMAP_LEVEL

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
        self.mipmap_dirty = False

    def copy(self):
        return Tile(copy_from=self)
        
transparent_tile = Tile()

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
        # copy-on-write for readonly tiles
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
        if t.mipmap_dirty:
            # regenerate mipmap
            if self.mipmap_level > 0:
                for x in xrange(2):
                    for y in xrange(2):
                        src = self.parent.get_tile_memory(tx*2 + x, ty*2 + y, True)
                        mypaintlib.tile_downscale_rgba16(src, t.rgba, x*N/2, y*N/2)
            t.mipmap_dirty = False
        if t.readonly and not readonly:
            # OPTIMIZE: we could do the copying in save_snapshot() instead, this might reduce the latency while drawing
            #           (eg. tile.valid_copy = some_other_tile_instance; and valid_copy = None here)
            #           before doing this, measure the worst-case time of the call below; same thing with new tiles
            t = t.copy()
            self.tiledict[(tx, ty)] = t
        if not readonly:
            self.mark_mipmap_dirty(tx, ty, t)
        return t.rgba
        
    def mark_mipmap_dirty(self, tx, ty, t=None):
        if t is None:
            t = self.tiledict.get((tx,ty))
        if t is None:
            t = Tile()
            self.tiledict[(tx, ty)] = t
        if not t.mipmap_dirty:
            t.mipmap_dirty = True
            if self.mipmap:
                self.mipmap.mark_mipmap_dirty(tx/2, ty/2)

    def blit_tile_into(self, dst, tx, ty):
        # used mainly for saving (transparent PNG)
        assert dst.shape[2] == 4
        tmp = self.get_tile_memory(tx, ty, readonly=True)
        return mypaintlib.tile_convert_rgba16_to_rgba8(tmp, dst)

    def composite_tile_over(self, dst, tx, ty, mipmap_level=0, opac=1.0):
        """
        composite one tile of this surface over the array dst, modifying only dst
        """

        if self.mipmap_level < mipmap_level:
            return self.mipmap.composite_tile_over(dst, tx, ty, mipmap_level, opac)
        if not (tx,ty) in self.tiledict:
            return
        src = self.get_tile_memory(tx, ty, True)
        if dst.shape[2] == 3 and dst.dtype == 'uint8':
            mypaintlib.tile_composite_rgba16_over_rgb8(src, dst, opac)
        elif dst.shape[2] == 4 and dst.dtype == 'uint16':
            # rarely used (only for merging layers)
            # src (premultiplied) OVER dst (premultiplied)
            # dstColor = srcColor + (1.0 - srcAlpha) * dstColor
            one_minus_srcAlpha = (1<<15) - (opac * src[:,:,3:4]).astype('uint32')
            dst[:,:,:] = opac * src[:,:,:] + ((one_minus_srcAlpha * dst[:,:,:]) >> 15).astype('uint16')

    def save_snapshot(self):
        sshot = SurfaceSnapshot()
        for t in self.tiledict.itervalues():
            t.readonly = True
        sshot.tiledict = self.tiledict.copy()
        return sshot

    def load_snapshot(self, sshot):
        old = set(self.tiledict.items())
        self.tiledict = sshot.tiledict.copy()
        new = set(self.tiledict.items())
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

    def save(self, filename):
        pixbuf = self.render_as_pixbuf()
        pixbuf.save(filename, 'png')

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

