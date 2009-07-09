#!/usr/bin/env python

from pylab import *
import gtk
from time import time

wait_for_idle = -1

def paint30sec(app, FPS=30):
    dw = app.drawWindow
    tdw = dw.tdw

    dw.fullscreen_cb()

    yield wait_for_idle

    #tdw.rotate(46.0/360*2*math.pi)

    #wrap the ordinary function with one that counts repaints
    dw.repaints = 0
    oldfunc=tdw.repaint
    def count_repaints(*args, **kwargs):
        dw.repaints += 1
        return oldfunc(*args, **kwargs)
    tdw.repaint = count_repaints


    events = load('painting30sec.dat.gz')
    #events[:,0] *= 0.5 # speed
    events = list(events)
    t0 = time()
    t_old = 0.0
    t_last_redraw = 0.0
    for t, x, y, pressure in events:
        if t > t_last_redraw + 1.0/FPS:
            yield wait_for_idle
            t_last_redraw = t
        #sleeptime = t-(time()-t0)
        #if sleeptime > 0.001:
        #    yield sleeptime
        dtime = t - t_old
        t_old = t
        #dw.doc.stroke_to(dtime, x, y, pressure)
        dw.doc.stroke_to(dtime, x/tdw.scale, y/tdw.scale, pressure)
    #print 'replay done.', dw.repaints, 'repaints'
    print 'replay done, time:', time()-t0

    # cleanup
    dw.fullscreen_cb()
    tdw.repaint = oldfunc
    dw.doc.clear()

def paint30sec_zoomed(app):
    dw = app.drawWindow
    dw.zoom('ZoomOut')
    dw.zoom('ZoomOut')
    dw.zoom('ZoomOut')
    dw.zoom('ZoomOut')
    for res in paint30sec(app):
        yield res
    dw.reset_view_cb(None)


def clear_time(app):
    print 'Clear time test:'
    doc = app.drawWindow.doc
    t0 = time()
    for i in range(20):
        doc.clear()
        yield 0.0
    doc.clear()

def strokeperf():
    #from library
    raise NotImplementedError

def startuptime():
    #take a similar approach as renderperf() ?
    raise NotImplementedError

def test_gui_main(app):
    for res in paint30sec(app):
        yield res
    for res in paint30sec_zoomed(app):
        yield res
    gtk.main_quit(app)
    yield 10.0

def test_gui():
    t0 = time()
    from gui import main
    import mypaint
    data, conf = mypaint.get_paths()
    app = main.main(data, conf, standalone = False)
    
    def profiling_main():
        for res in test_gui_main(app):
            yield res

    import gobject
    p = profiling_main()
    def callback():
        res = p.next()
        if res == wait_for_idle:
            gobject.idle_add(callback)
        else:
            gobject.timeout_add(int(res*1000.0), callback)
    callback()
    gtk.main()

def test_nogui():
    pass

if __name__ == '__main__':
    test_gui()
    #print '%s repaints' % renderperf(rotate=False)
    #scrollperf()
