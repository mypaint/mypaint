from pylab import *

import mypaintlib, tiledsurface, brushsettings, brush

def directPaint():

    s = tiledsurface.TiledSurface()
    events = load('painting30sec.dat.gz')

    for t, x, y, pressure in events:
        r = g = b = 0.5*(1.0+sin(t))
        r *= 0.8
        s.draw_dab(x, y, 12, r, g, b, pressure, 0.6)
    s.save('directPaint.png')

def brushPaint():

    s = tiledsurface.TiledSurface()
    b = brush.Brush_Lowlevel()
    b.load_from_string(open('brushes/s006.myb').read())

    events = load('painting30sec.dat.gz')

    b.set_color_rgb((0.0, 0.9, 1.0))

    t_old = events[0][0]
    for t, x, y, pressure in events:
        dtime = t - t_old
        t_old = t
        b.tiled_surface_stroke_to (s, x, y, pressure, dtime)

    s.save('brushPaint.png')

directPaint()
brushPaint()

