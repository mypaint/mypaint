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
from PIL import Image
import mypaintlib, helpers
import gtk
from gtk import gdk

tilesize = N = mypaintlib.TILE_SIZE

class Tile:
    def __init__(self):
        # note: pixels are stored with premultiplied alpha
        #self.rgb   = zeros((N, N, 3), 'uint8')
        #self.alpha = zeros((N, N, 1), 'uint8')
        self.pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, True, 8, N, N)
        self.rgba   = mypaintlib.gdkpixbuf2numpy(self.pixbuf.get_pixels_array())
        self.rgba[:,:,:] = 0
        # for finding bugs; this color is transparent and thus should never show up anywhere
        #self.rgba[:,:,:] = (255, 0, 255, 0)
        self.readonly = False

    def copy(self):
        t = Tile()
        t.rgba[:] = self.rgba[:]
        return t
        
    def composite_over_RGB8(self, dst):
        # OPTIMIZE: that's not how it is supposed to be done
        dst_pixbuf     = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, N, N)
        dst_pixbuf_rgb = mypaintlib.gdkpixbuf2numpy(dst_pixbuf.get_pixels_array())
        dst_pixbuf_rgb[:] = dst
        # un-premultiply alpha (argh!)
        rgba_orig = self.rgba.copy()
        self.rgba[:,:,0:3] = self.rgba[:,:,0:3] * 255 / clip(self.rgba[:,:,3:4], 1, 255)
        self.pixbuf.composite(dst_pixbuf, 0, 0, N, N, 0, 0, 1, 1, gdk.INTERP_NEAREST, 255)
        dst[:] = dst_pixbuf_rgb[:]

        self.rgba[:] = rgba_orig

transparentTile = Tile()

def get_tiles_bbox(tiles):
    res = helpers.Rect()
    for x, y in tiles:
        res.expandToIncludeRect(helpers.Rect(N*x, N*y, N, N))
    return res

class TiledSurface(mypaintlib.TiledSurface):
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

    def get_tile_memory(self, x, y, readonly):
        # copy-on-write for readonly tiles
        # OPTIMIZE: do some profiling to check if this function is a bottleneck
        t = self.tiledict.get((x, y))
        if t is None:
            if readonly:
                t = transparentTile
            else:
                t = Tile()
                self.tiledict[(x, y)] = t
        if t.readonly and not readonly:
            # OPTIMIZE: we could do the copying in save_snapshot() instead, this might reduce the latency while drawing
            #           (eg. tile.valid_copy = some_other_tile_instance; and valid_copy = None here)
            #           before doing this, measure the worst-case time of the call below; same thing with new tiles
            t = t.copy()
            self.tiledict[(x, y)] = t
        return t.rgba
        
    #def iter_existing_tiles(self, x, y, w, h):
    #    for xx in xrange(x/Tile.N, (x+w)/Tile.N+1):
    #        for yy in xrange(y/Tile.N, (x+h)/Tile.N+1):
    #            tile = self.tiledict.get((xx, yy), None)
    #            if tile is not None:
    #                yield xx*Tile.N, yy*Tile.N, tile

    def iter_tiles_memory(self, x, y, w, h, readonly):
        for xx in xrange(x/N, (x+w-1)/N+1):
            for yy in xrange(y/N, (y+h-1)/N+1):
                # note, this is somewhat untested
                x_start = max(0, x - xx*N)
                y_start = max(0, y - yy*N)
                x_end = min(N, x+w - xx*N)
                y_end = min(N, y+h - yy*N)
                #print xx*N, yy*N
                #print x_start, y_start, x_end, y_end

                rgb, alpha = self.get_tile_memory(xx, yy, readonly)
                yield xx*N, yy*N, (rgb[y_start:y_end,x_start:y_end], alpha[y_start:y_end,x_start:y_end])

    def composite_tile(self, dst, tx, ty):
        tile = self.tiledict.get((tx, ty))
        if tile is None:
            return
        tile.composite_over_RGB8(dst)

    def composite_over_RGB8(self, dst, px, py):
        h, w, channels = dst.shape
        assert channels == 3

        for (x0, y0), tile in self.tiledict.iteritems():
            x0 = N*x0+px
            y0 = N*y0+py
            if x0 < 0 or y0 < 0: continue
            if x0+N > w or y0+N > h: continue
            tile.composite_over_RGB8(dst[y0:y0+N,x0:x0+N,:]) # OPTIMIZE: is this slower than without offsets?

    def save(self, filename):
        assert self.tiledict, 'cannot save empty surface'
        a = array([xy for xy, tile in self.tiledict.iteritems()])
        minx, miny = N*a.min(0)
        sizex, sizey = N*(a.max(0) - a.min(0) + 1)
        buf = zeros((sizey, sizex, 4), 'uint8')

        for (x0, y0), tile in self.tiledict.iteritems():
            x0 = N*x0 - minx
            y0 = N*y0 - miny
            dst = buf[y0:y0+N,x0:x0+N,:]
            dst[:,:,:] = tile.rgba

        im = Image.fromstring('RGBA', (sizex, sizey), buf.tostring())
        im.save(filename)

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

    def load_from_data(self, data):
        self.clear()
        for x, y, rgba in self.iter_tiles_memory(0, 0, data.shape[1], data.shape[0], readonly=False):
            # FIXME: will be buggy at border?
            if data.shape[2] == 4:
                rgba[:,:,:] = data[y:y+N,x:x+N,:]
            else:
                rgba[:,:,0:3] = data[y:y+N,x:x+N,:]
                rgba[:,:,4]   = 255

    def get_bbox(self):
        return get_tiles_bbox(self.tiledict)

