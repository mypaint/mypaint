# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
gdk = gtk.gdk

import numpy
import tiledsurface
N = tiledsurface.N
import helpers, mypaintlib

class BackgroundError(Exception):
    pass

class Background(tiledsurface.Surface):
    def __init__(self, obj, mipmap_level=0):

        if isinstance(obj, gdk.Pixbuf):
            obj = helpers.gdkpixbuf2numpy(obj)
        elif not isinstance(obj, numpy.ndarray):
            r, g, b = obj
            obj = numpy.zeros((N, N, 3), dtype='uint8')
            obj[:,:,:] = r, g, b

        height, width = obj.shape[0:2]
        if height % N or width % N:
            raise BackgroundError, 'unsupported background tile size: %dx%d' % (width, height)

        tiledsurface.Surface.__init__(self, mipmap_level=0,
                                      looped=True, looped_size=(width, height))
        self.load_from_numpy(obj, 0, 0)

        # Generate mipmap
        if mipmap_level < tiledsurface.MAX_MIPMAP_LEVEL:
            mipmap_obj = numpy.zeros((height, width, 4), dtype='uint16')
            for ty in range(height/N*2):
                for tx in range(width/N*2):
                    src = self.get_tile_memory(tx, ty, readonly=True)
                    mypaintlib.tile_downscale_rgba16(src, mipmap_obj, tx*N/2, ty*N/2)
            self.mipmap = Background(mipmap_obj, mipmap_level+1)
            self.mipmap.parent = self
            self.mipmap_level = mipmap_level
