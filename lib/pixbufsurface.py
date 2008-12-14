# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

# This class converts between linear 8bit RGB(A) and tiled RGBA storage.
# It is used for rendering updates, but also for save/load.

from gtk import gdk
import mypaintlib, tiledsurface

N = tiledsurface.N

class Surface:
    def __init__(self, x, y, w, h, alpha=False, data=None):
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

        self.epixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, alpha, 8, self.ew, self.eh)
        dx = x-self.ex
        dy = y-self.ey
        self.pixbuf  = self.epixbuf.subpixbuf(dx, dy, w, h)

        assert self.ew <= w + 2*N-2
        assert self.eh <= h + 2*N-2

        if not alpha:
            self.epixbuf.fill(0xff0088ff) # to detect uninitialized memory
        else:
            self.epixbuf.fill(0x00000000) # keep undefined region transparent

        arr = self.epixbuf.get_pixels_array()
        arr = mypaintlib.gdkpixbuf2numpy(arr)

        discard_transparent = False

        if data is not None:
            dst = arr[dy:dy+h,dx:dx+w,:]
            if data.shape[2] == 3:
                # no alpha
                dst[:,:,0:3] = data
                dst[:,:,3] = 255
            else:
                dst[:,:,:] = data
                # this surface will be used read-only
                discard_transparent = True

        self.tile_memory_dict = {}
        for ty in range(th):
            for tx in range(tw):
                buf = arr[ty*N:(ty+1)*N,tx*N:(tx+1)*N,:]
                if discard_transparent and not buf[:,:,3].any():
                    continue
                self.tile_memory_dict[(self.tx+tx, self.ty+ty)] = buf

    def get_tiles(self):
        return self.tile_memory_dict.keys()

    def get_tile_memory(self, tx, ty):
        return self.tile_memory_dict[(tx, ty)]

    def blit_tile_into(self, dst, tx, ty):
        assert dst.dtype == 'uint16', '16 bit dst expected'
        tmp = self.tile_memory_dict[(tx, ty)]
        assert tmp.shape[2] == 4, 'alpha required'
        tmp = tmp.astype('float32') / 255.0
        alpha = tmp[:,:,3:]
        tmp[:,:,0:3] *= alpha # premultiply alpha
        tmp *= 1<<15
        dst[:,:,:] = tmp

