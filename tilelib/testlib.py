from numpy import *
from scipy import *
from pylab import *
import tile
import ctile

def blendtest():
    l = tile.TiledLayer()

    # creating a stroke
    N = 100
    t = linspace(0, 5, N)
    dabs = zeros((N, 8), 'float32')
    # x, y
    dabs[:,0] = sin(t*0.3)*10 + 16
    dabs[:,1] = t*8
    # radius
    dabs[:,2] = 2.0 + 5.0 * rand(N)
    # rgb
    dabs[:,3:6] = 0.0
    dabs[:,3] = 1.0
    # alpha
    dabs[:,6] = 0.1
    # hardness
    dabs[:,7] = 0.7


    ctile.render(l, dabs)

    l.plot()

def painttest():
    l = tile.TiledLayer()
    events = load('painting30sec.dat.gz')
    N = len(events)
    dabs = zeros((N, 8), 'float32')

    # x, y
    dabs[:,0:2] = events[:,1:3]
    # radius
    dabs[:,2] = 3.0 #+ 5.0 * rand(N)
    # rgb
    dabs[:,3:6] = 0.0
    dabs[:,3] = 1.0
    # alpha = pressure
    dabs[:,6] = events[:,3]
    # hardness
    dabs[:,7] = 0.7

    ctile.render(l, dabs)

    l.save('output.png')

    #l.plot()

#blendtest()
painttest()
