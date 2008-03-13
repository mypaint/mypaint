from numpy import *
from PIL import Image
import _tilelib

tilesize = N = 64

class Tile:
    def __init__(self):
        self.rgb   = zeros((N, N, 3), 'float32')
        self.alpha = zeros((N, N, 1), 'float32')
        
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
            print 'allocating tile', (x, y)
            t = Tile()
            self.tiledict[(x, y)] = t
        return t.rgb, t.alpha
        
    def tiles(self, x, y, w, h):
        for xx in xrange(x/Tile.N, (x+w)/Tile.N+1):
            for yy in xrange(y/Tile.N, (x+h)/Tile.N+1):
                tile = self.tiledict.get((xx, yy), None)
                if tile is not None:
                    yield xx*Tile.N, yy*Tile.N, tile

    def drawDab(self, *args):
       _tilelib.tile_draw_dab(self, *args)

    def plot(self):
        from pylab import *

        import random
        xy, t = random.choice(self.tiledict.items())
        print len(self.tiledict)

        #print t.rgb
        subplot(121)
        #print t.rgb
        imshow(t.rgb)
        #imshow(t.rgb/(t.alpha+0.001))
        subplot(122)
        a = t.alpha.copy()
        a.shape = (N,N)
        imshow(a)

        show()
        raise SystemExit

    def save(self, filename):
        # only for debugging so far... should save alpha channel too
        a = array([(x0, y0) for (x0, y0), tile in self.tiledict.iteritems()])
        minx, miny = N*a.min(0)
        sizex, sizey = N*(a.max(0) - a.min(0) + 1)
        print sizex, sizey
        buf = zeros((sizey, sizex, 3), 'float32')
        for (x0, y0), tile in self.tiledict.iteritems():
            x0 = N*x0 - minx
            y0 = N*y0 - miny
            buf[y0:y0+N,x0:x0+N,:] = tile.rgb[:,:,:]
        buf *= 255
        buf = buf.astype('uint8')

        im = Image.fromstring('RGB', (sizex, sizey), buf.tostring())
        im.save(filename)
