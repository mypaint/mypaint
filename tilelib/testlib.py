from pylab import *

import tile

def directPaint():

    l = tile.TiledLayer()
    events = load('painting30sec.dat.gz')
    #events = events[:100,:]

    for t, x, y, pressure in events:
        l.drawDab(x, y, 12, 1.0, 1.0, 1.0, pressure, 0.6)
    #l.plot()
    l.save('directPaint.png')


directPaint()
#interpolatedPaint()
