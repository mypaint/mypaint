from numpy import *
from PIL import Image
import mypaintlib

tilesize = N = mypaintlib.TILE_SIZE

class Tile:
    def __init__(self):
        # note: pixels are stored with premultiplied alpha
        self.rgb   = zeros((N, N, 3), 'float32')
        self.alpha = zeros((N, N, 1), 'float32')
        self.changes = 0
        
    def compositeOverRGB8(self, dst):
        dst[:,:,0:3] = (dst[:,:,0:3] * (1.0-self.alpha)).round().astype('uint8')
        #dst[:,:,0:3] *= 1.0-self.alpha
        dst[:,:,0:3] = dst[:,:,0:3] + (255*self.rgb[:,:,0:3]).round().astype('uint8')
        
    #def composite(self, other):
        # resultColor = topColor + (1.0 - topAlpha) * bottomColor
    #    self.rgb = other.alpha * other.rgb + (1.0-other.alpha) * self.rgb
    #    self.alpha = other.alpha + (1.0-other.alpha)*self.alpha


class TiledLayer:
    def __init__(self):
        self.tiledict = {}
        self.alpha = 1.0

    def getTileMemory(self, x, y):
        t = self.tiledict.get((x, y))
        if t is None:
            t = Tile()
            self.tiledict[(x, y)] = t
        return t.rgb, t.alpha
        
    #def tiles(self, x, y, w, h):
    #    for xx in xrange(x/Tile.N, (x+w)/Tile.N+1):
    #        for yy in xrange(y/Tile.N, (x+h)/Tile.N+1):
    #            tile = self.tiledict.get((xx, yy), None)
    #            if tile is not None:
    #                yield xx*Tile.N, yy*Tile.N, tile

    def drawDab(self, *args):
       mypaintlib.tile_draw_dab(self, *args)


    def compositeOverRGB8(self, dst):
        h, w, channels = dst.shape
        assert channels == 3

        for (x0, y0), tile in self.tiledict.iteritems():
            x0 = N*x0
            y0 = N*y0
            if x0 < 0 or y0 < 0: continue
            if x0+N > w or y0+N > h: continue
            tile.compositeOverRGB8(dst[y0:y0+N,x0:x0+N,:])

    def save(self, filename):
        assert self.tiledict, 'cannot save empty layer'
        a = array([xy for xy, tile in self.tiledict.iteritems()])
        minx, miny = N*a.min(0)
        sizex, sizey = N*(a.max(0) - a.min(0) + 1)
        buf = zeros((sizey, sizex, 4), 'float32')

        for (x0, y0), tile in self.tiledict.iteritems():
            x0 = N*x0 - minx
            y0 = N*y0 - miny
            dst = buf[y0:y0+N,x0:x0+N,:]
            # un-premultiply alpha
            dst[:,:,0:3] = tile.rgb[:,:,0:3] / clip(tile.alpha, 0.000001, 1.0)
            dst[:,:,3:4] = tile.alpha

        buf = (buf*255).round().astype('uint8')
        im = Image.fromstring('RGBA', (sizex, sizey), buf.tostring())
        im.save(filename)
