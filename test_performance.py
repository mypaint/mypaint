#!/usr/bin/env python

from pylab import *
import gtk
from time import time

wait_for_idle = -1
all_tests = {}

def run_gui_test(testfunction):
    """Setup the MyPaint GUI and run testfunction.
    testfunction must be a generator (using yield)
    """
    import mypaint
    data, conf = mypaint.get_paths()
    filenames = []
    from gui import application
    app = application.Application(data, conf, filenames)
    #app = main.main(data, conf, standalone = False)

    def profiling_main():
        for res in testfunction(app):
            yield res
        gtk.main_quit(app)
        yield 10.0

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

def gui_test(f):
    "decorator to declare GUI test functions"
    def run_f():
        run_gui_test(f)
    all_tests[f.__name__] = run_f
    return f

@gui_test
def paint30sec(app):
    FPS = 30
    dw = app.drawWindow
    tdw = dw.tdw

    dw.fullscreen_cb()

    yield wait_for_idle
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

@gui_test
def paint30sec_zoomed(app):
    dw = app.drawWindow
    dw.zoom('ZoomOut')
    dw.zoom('ZoomOut')
    dw.zoom('ZoomOut')
    dw.zoom('ZoomOut')
    for res in paint30sec(app):
        yield res

@gui_test
def paint30sec_rotated(app):
    app.drawWindow.tdw.rotate(46.0/360*2*math.pi)
    for res in paint30sec(app):
        yield res

@gui_test
def clear_time(app):
    print 'Clear time test:'
    doc = app.drawWindow.doc
    t0 = time()
    for i in range(20):
        doc.clear()
        yield 0.0

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser('usage: %prog [options] testname')
    parser.add_option('-l', '--list', action='store_true', default=False,
                    help='list all available tests')
    options, tests = parser.parse_args()

    if options.list:
        for name in sorted(all_tests.keys()):
            print name
        sys.exit(0)

    if len(tests) != 1:
        parser.print_help()
        sys.exit(1)

    for test in tests:
        if test not in all_tests:
            print 'Unknown test:', test
            sys.exit(1)
        func = all_tests[test]
        func()
