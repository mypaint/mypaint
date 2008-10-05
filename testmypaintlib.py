from pylab import *
OUTDATED

import mypaintlib, tiledsurface, brushsettings, brush

def directPaint():

    l = tiledsurface.TiledLayer()
    events = load('painting30sec.dat.gz')

    for t, x, y, pressure in events:
        r = g = b = 0.5*(1.0+sin(t))
        r *= 0.8
        l.drawDab(x, y, 12, r, g, b, pressure, 0.6)
    l.save('directPaint.png')

def brushPaint():

    l = tiledsurface.TiledLayer()
    b = brush.Brush_Lowlevel()
    #b.load_from_string(open('brushes/s006.myb').read())

    events = load('painting30sec.dat.gz')

    def f():
        print 'Stroke split callback.'
    b.set_split_stroke_callback (f)

    b.set_color_rgb((0.0, 0.9, 1.0))

    t_old = events[0][0]
    for t, x, y, pressure in events:
        dtime = t - t_old
        t_old = t
        b.tiled_surface_stroke_to (l, x, y, pressure, dtime)

    l.save('brushPaint.png')

directPaint()
brushPaint()
