# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

# This module contains an infinite (unbounded) tiled surface for painting.
# It is the memory storage backend for one layer.

from numpy import *
import mypaintlib, helpers

tilesize = N = mypaintlib.TILE_SIZE

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
        
    def composite_over_RGB8(self, dst):
        mypaintlib.composite_tile_over_rgb8(self.rgba, dst)

    def copy_into_RGBA8(self, dst):
        # rarely used
        tmp = self.rgba.astype('float32') / (1<<15)
        tmp[:,:,0:3] /= tmp[:,:,3:].clip(0.0001, 1.0) # un-premultiply alpha
        tmp = tmp.clip(0.0,1.0)
        dst[:,:,:] = tmp * 255.0


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


    def composite_tile_over(self, dst, tx, ty):
        tile = self.tiledict.get((tx, ty))
        if tile is None:
            return
        tile.composite_over_RGB8(dst)
#
#    def composite_over_RGB8(self, dst, px, py):
#        h, w, channels = dst.shape
#        assert channels == 3
#
#        for (x0, y0), tile in self.tiledict.iteritems():
#            x0 = N*x0+px
#            y0 = N*y0+py
#            if x0 < 0 or y0 < 0: continue
#            if x0+N > w or y0+N > h: continue
#            tile.composite_over_RGB8(dst[y0:y0+N,x0:x0+N,:]) # OPTIMIZE: is this slower than without offsets?

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

    def save(self, filename):
        assert self.tiledict, 'cannot save empty surface'
        print 'SAVE'
        from PIL import Image
        a = array([xy for xy, tile in self.tiledict.iteritems()])
        minx, miny = N*a.min(0)
        sizex, sizey = N*(a.max(0) - a.min(0) + 1)
        buf = zeros((sizey, sizex, 4), 'uint8')

        for (x0, y0), tile in self.tiledict.iteritems():
            x0 = N*x0 - minx
            y0 = N*y0 - miny
            dst = buf[y0:y0+N,x0:x0+N,:]
            tile.copy_into_RGBA8(dst)

        im = Image.fromstring('RGBA', (sizex, sizey), buf.tostring())
        im.save(filename)

#     # planned
#     def load_from_surface(self, surface):
#         dirty_tiles = set(self.tiledict.keys())
#         self.tiledict = {}

#         for tx, ty, tile in surface.iter_tiles():
#             rgba = self.get_tile_memory(tx, ty, readonly=False)
#             tile.copy_into(rgba)

#         dirty_tiles.update(self.tiledict.keys())
#         bbox = get_tiles_bbox(dirty_tiles)
#         self.notify_observers(*bbox)
        

    def load_from_data(self, data):
        dirty_tiles = set(self.tiledict.keys())
        self.tiledict = {}

        # FIXME: rewrite this to use pixbufsurface
        if data.shape[0] % N or data.shape[1] % N:
            s = list(data.shape)
            print 'reshaping', s
            s[0] = ((s[0]+N-1) / N) * N
            s[1] = ((s[1]+N-1) / N) * N
            data_new = zeros(s, data.dtype)
            data_new[:data.shape[0],:data.shape[1],:] = data
            data = data_new

        for x, y, rgba in self.iter_tiles_memory(0, 0, data.shape[1], data.shape[0], readonly=False):
            # FIXME: will be buggy at border?
            tmp = data[y:y+N,x:x+N,:].astype('float32') / 255.0
            if data.shape[2] == 3:
                # no alpha channel loaded
                alpha = 1.0
            else:
                alpha = tmp[:,:,3:]
            tmp[:,:,0:3] *= alpha # premultiply alpha
            tmp *= 1<<15
            if data.shape[2] == 4:
                rgba[:,:,:] = tmp
            else:
                rgba[:,:,0:3] = tmp
                rgba[:,:,3]   = 1<<15

        dirty_tiles.update(self.tiledict.keys())
        bbox = get_tiles_bbox(dirty_tiles)
        self.notify_observers(*bbox)



    def get_bbox(self):
        # FIXME: should get precise bbox instead of tile bbox
        return get_tiles_bbox(self.tiledict)

