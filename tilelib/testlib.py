from pylab import *

import tile

def directPaint():

    l = tile.TiledLayer()
    events = load('painting30sec.dat.gz')
    #events = events[:100,:]

    for t, x, y, pressure in events:
        r = g = b = 0.5*(1.0+sin(t))
        r *= 0.8
        l.drawDab(x, y, 12, r, g, b, pressure, 0.6)
    l.save('directPaint.png')


directPaint()
#interpolatedPaint()
