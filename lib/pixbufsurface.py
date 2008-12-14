from gtk import gdk
import mypaintlib, tiledsurface

N = tiledsurface.N

class Tile:
    pass

class Surface:
    def __init__(self, x, y, w, h):
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

        self.epixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, self.ew, self.eh)
        self.pixbuf  = self.epixbuf.subpixbuf(x-self.ex, y-self.ey, w, h)

        assert self.ew <= w + 2*N-2
        assert self.eh <= h + 2*N-2

        self.epixbuf.fill(0xff0088ff) # to detect uninitialized memory

        arr = self.epixbuf.get_pixels_array()
        arr = mypaintlib.gdkpixbuf2numpy(arr)
        self.tile_memory_dict = {}
        for ty in range(th):
            for tx in range(tw):
                buf = arr[ty*N:(ty+1)*N,tx*N:(tx+1)*N,:]
                self.tile_memory_dict[(self.tx+tx, self.ty+ty)] = buf

    def get_tiles(self):
        return self.tile_memory_dict.keys()

    def get_tile_memory(self, tx, ty):
        return self.tile_memory_dict[(tx, ty)]

