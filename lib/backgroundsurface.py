# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import numpy

import mypaintlib, helpers
from tiledsurface import N

class Background:
    def __init__(self, obj):
        try:
            obj = helpers.gdkpixbuf2numpy(obj)
        except:
            # it was already an array (FIXME: is this codepath ever used?)
            pass
        try:
            r, g, b = obj
        except:
            pass
        else:
            obj = numpy.zeros((N, N, 3), dtype='uint8')
            obj[:,:,:] = r, g, b

        self.tw = obj.shape[1]/N
        self.th = obj.shape[0]/N
        assert obj.shape == (self.th*N, self.tw*N, 3), 'unsupported background pixmap type or dimensions'
        assert obj.dtype == 'uint8'

        self.tiles = {}
        for ty in range(self.th):
            for tx in range(self.tw):
                # make sure we have linear memory (optimization)
                tile = numpy.zeros((N, N, 3), dtype='uint8')
                tile[:,:,:] = obj[N*ty:N*(ty+1), N*tx:N*(tx+1), :]
                self.tiles[tx, ty] = tile
        
    def blit_tile_into(self, dst, tx, ty):
        rgb = self.tiles[tx%self.tw, ty%self.th]
        # render solid or tiled background
        #dst[:] = background_memory # 13 times slower than below, with some bursts having the same speed as below (huh?)
        # note: optimization for solid colors is not worth it any more now, even if it gives 2x speedup (at best)
        mypaintlib.tile_blit_rgb8_into_rgb8(rgb, dst)


