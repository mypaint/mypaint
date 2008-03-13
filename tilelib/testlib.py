from numpy import *
from scipy import *
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


def interpolatedPaint():
    OUTDATED
    l = tile.TiledLayer()
    events = load('painting30sec.dat.gz')

    direct_painttest(l, events)

    # states are:
    # 0 time
    # 1 filtered x position
    # 2 filtered y position

    def dstate(state, dab):
        dstate = ones_like(state)
        dt = 0.001
        dstate[0] = dt # time change per dab
        t = state[0]
        # find the last event that happened before time t (to be optimized...)
        i = argmax(events[:,0] > t) - 1
        if i < 0: i = 0
        #print dab, t, i, events[i,:], state
        # change of the filtered x/y position
        dstate[1:3] = (events[i,1:3] - state[1:3]) * 0.8 # per dab, could also do per time...

        return dstate


    N = 3000
    state0 = array([0.0, events[0,1], events[0,2]])
    print dstate(state0, 0.0)
    dabs = arange(N)
    states = integrate.odeint(dstate, state0, dabs)

    dabs = zeros((N, 8), 'float32')

    # x, y
    dabs[:,0:2] = states[:,1:3]
    # radius
    dabs[:,2] = 3.0 #+ 5.0 * rand(N)
    # rgb
    dabs[:,3:6] = 0.0
    dabs[:,3] = 1.0
    # alpha = pressure
    dabs[:,6] = 0.3 # TODO: need pressure state
    # hardness
    dabs[:,7] = 0.7

    ctile.render(l, dabs)

    l.save('output.png')

    #l.plot()

directPaint()
#interpolatedPaint()
