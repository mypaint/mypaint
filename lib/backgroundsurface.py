# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import numpy, gtk
gdk = gtk.gdk

import mypaintlib, helpers
from tiledsurface import N, MAX_MIPMAP_LEVEL, get_tiles_bbox
import pixbufsurface

class BackgroundError(Exception):
    pass

class Background:
    def __init__(self, obj, mipmap_level=0):
        if isinstance(obj, gdk.Pixbuf):
            obj = helpers.gdkpixbuf2numpy(obj)
        elif not isinstance(obj, numpy.ndarray):
            r, g, b = obj
            obj = numpy.zeros((N, N, 3), dtype='uint8')
            obj[:,:,:] = r, g, b

        self.tw = obj.shape[1]/N
        self.th = obj.shape[0]/N
        if obj.shape[-1] == 4:
            raise BackgroundError, 'background tile with alpha channel is not allowed'
        if obj.shape != (self.th*N, self.tw*N, 3):
            raise BackgroundError, 'unsupported background tile size: %dx%d' % (obj.shape[0], obj.shape[1])
        if obj.dtype == 'uint8':
            obj = (obj.astype('uint32') * (1<<15) / 255).astype('uint16')

        self.tiles = {}
        for ty in range(self.th):
            for tx in range(self.tw):
                # make sure we have linear memory (optimization)
                tile = numpy.zeros((N, N, 3), dtype='uint16')
                tile[:,:,:] = obj[N*ty:N*(ty+1), N*tx:N*(tx+1), :]
                self.tiles[tx, ty] = tile
        
        # generate mipmap
        self.mipmap_level = mipmap_level
        if mipmap_level < MAX_MIPMAP_LEVEL:
            mipmap_obj = numpy.zeros((self.th*N, self.tw*N, 3), dtype='uint16')
            for ty in range(self.th*2):
                for tx in range(self.tw*2):
                    src = self.get_tile_memory(tx, ty)
                    mypaintlib.tile_downscale_rgb16(src, mipmap_obj, tx*N/2, ty*N/2)
            self.mipmap = Background(mipmap_obj, mipmap_level+1)

    def get_tile_memory(self, tx, ty):
        return self.tiles[(tx%self.tw, ty%self.th)]

    def blit_tile_into(self, dst, tx, ty, mipmap_level=0):
        if self.mipmap_level < mipmap_level:
            return self.mipmap.blit_tile_into(dst, tx, ty, mipmap_level)
        rgb = self.get_tile_memory(tx, ty)
        # render solid or tiled background
        #dst[:] = rgb # 13 times slower than below, with some bursts having the same speed as below (huh?)
        # note: optimization for solid colors is not worth it, it gives only 2x speedup (at best)
        if dst.dtype == 'uint16':
            mypaintlib.tile_blit_rgb16_into_rgb16(rgb, dst)
        else:
            # this case is for saving the background
            assert dst.dtype == 'uint8'
            # note: when saving the background layer we usually
            # convert here the same tile over and over again. But it
            # does help much to cache this conversion result. The
            # save_ora speedup when doing this is below 1%, even for a
            # single-layer ora.
            mypaintlib.tile_convert_rgb16_to_rgb8(rgb, dst)

    def get_pattern_bbox(self):
        return get_tiles_bbox(self.tiles)

    def save(self, filename, *rect, **kwargs):
        assert 'alpha' not in kwargs
        kwargs['alpha'] = False
        if len(self.tiles) == 1:
            kwargs['single_tile_pattern'] = True
        pixbufsurface.save_as_png(self, filename, *rect, **kwargs)
