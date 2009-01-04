# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This module implements an an unbounded tiled surface for painting.

from numpy import *
import mypaintlib, helpers

tilesize = N = mypaintlib.TILE_SIZE

import pixbufsurface
from gtk import gdk


class Tile:
    def __init__(self):
        # note: pixels are stored with premultiplied alpha
        #       15bits are used, but fully opaque or white is stored as 2**15 (requiring 16 bits)
        #       This is to allow many calcuations to divide by 2**15 instead of (2**16-1)
        self.rgba   = zeros((N, N, 4), 'uint16')
        self.readonly = False

    def copy(self):
        t = Tile()
        t.rgba[:] = self.rgba[:]
        return t
        
transparentTile = Tile()

def get_tiles_bbox(tiles):
    res = helpers.Rect()
    for tx, ty in tiles:
        res.expandToIncludeRect(helpers.Rect(N*tx, N*ty, N, N))
    return res

class Surface(mypaintlib.TiledSurface):
    # the C++ half of this class is in tiledsurface.hpp
    def __init__(self):
        mypaintlib.TiledSurface.__init__(self, self)
        self.tiledict = {}
        self.observers = []

    def notify_observers(self, *args):
        for f in self.observers:
            f(*args)

    def clear(self):
        tiles = self.tiledict.keys()
        self.tiledict = {}
        self.notify_observers(*get_tiles_bbox(tiles))

    def get_tile_memory(self, tx, ty, readonly):
        # copy-on-write for readonly tiles
        # OPTIMIZE: do some profiling to check if this function is a bottleneck
        #           yes it is
        # Note: we must return memory that stays valid for writing until the
        # last end_atomic(), because of the caching in tiledsurface.hpp.
        t = self.tiledict.get((tx, ty))
        if t is None:
            if readonly:
                t = transparentTile
            else:
                t = Tile()
                self.tiledict[(tx, ty)] = t
        if t.readonly and not readonly:
            # OPTIMIZE: we could do the copying in save_snapshot() instead, this might reduce the latency while drawing
            #           (eg. tile.valid_copy = some_other_tile_instance; and valid_copy = None here)
            #           before doing this, measure the worst-case time of the call below; same thing with new tiles
            t = t.copy()
            self.tiledict[(tx, ty)] = t
        return t.rgba
        
    def iter_tiles_memory(self, x, y, w, h, readonly):
        for tx in xrange(x/N, (x+w-1)/N+1):
            for ty in xrange(y/N, (y+h-1)/N+1):
                # note, this is somewhat untested
                x_start = max(0, x - tx*N)
                y_start = max(0, y - ty*N)
                x_end = min(N, x+w - tx*N)
                y_end = min(N, y+h - ty*N)
                #print xx*N, yy*N
                #print x_start, y_start, x_end, y_end

                rgba = self.get_tile_memory(tx, ty, readonly)
                yield tx*N, ty*N, rgba[y_start:y_end,x_start:y_end]

    def blit_tile_into(self, dst, tx, ty):
        # rarely used
        assert dst.shape[2] == 4
        tmp = self.get_tile_memory(tx, ty, readonly=True)
        tmp = tmp.astype('float32') / (1<<15)
        tmp[:,:,0:3] /= tmp[:,:,3:].clip(0.0001, 1.0) # un-premultiply alpha
        tmp = tmp.clip(0.0,1.0)
        dst[:,:,:] = tmp * 255.0

    def composite_tile_over(self, dst, tx, ty):
        tile = self.tiledict.get((tx, ty))
        if tile is None:
            return
        assert dst.shape[2] == 3
        mypaintlib.tile_composite_rgba16_over_rgb8(tile.rgba, dst)

    def save_snapshot(self):
        for t in self.tiledict.itervalues():
            t.readonly = True
        return self.tiledict.copy()

    def load_snapshot(self, data):
        old = set(self.tiledict.items())
        self.tiledict = data.copy()
        new = set(self.tiledict.items())
        dirty = old.symmetric_difference(new)
        bbox = get_tiles_bbox([pos for (pos, tile) in dirty])
        self.notify_observers(*bbox)

    def render_as_pixbuf(self):
        if not self.tiledict:
            print 'WARNING: empty surface'
        return pixbufsurface.render_as_pixbuf(self, alpha=True)

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

