#!/usr/bin/env python

import sys, os, tempfile, subprocess
from time import time, sleep

from pylab import *
import gtk, glib

os.chdir(os.path.dirname(sys.argv[0]))
sys.path.insert(0, '..')

# wait until the last mypaint idle handler has finished
wait_for_idle = -1
# wait until all GUI updates are done, but don't wait for background tasks
wait_for_gui = -2

all_tests = {}

def run_gui_test(testfunction):
    """Setup the MyPaint GUI and run testfunction.
    testfunction must be a generator (using yield)
    """
    global gui_startup_time
    gui_startup_time = time()

    from gui import application
    tempdir = tempfile.mkdtemp()
    app = application.Application(datapath='..', confpath=tempdir, filenames=[])

    def profiling_main():
        for res in testfunction(app):
            yield res
        gtk.main_quit(app)
        yield 10.0

    import gobject
    p = profiling_main()
    def callback():
        res = p.next()
        if res < 0:
            if res == wait_for_idle:
                priority = gobject.PRIORITY_LOW + 50
            elif res == wait_for_gui:
                priority = gobject.PRIORITY_DEFAULT_IDLE - 1
            gobject.idle_add(callback, priority=priority)
        else:
            gobject.timeout_add(int(res*1000.0), callback)

    # fatal exceptions, please
    def excepthook(exctyp, value, tb):
        import traceback
        traceback.print_exception (exctyp, value, tb, None, sys.stderr)
        sys.exit(1)
    sys.excepthook = excepthook

    callback()
    gtk.main()
    os.system('rm -rf ' + tempdir)

def gui_test(f):
    "decorator to declare GUI test functions"
    def run_f():
        run_gui_test(f)
    all_tests[f.__name__] = run_f
    return f

def nogui_test(f):
    "decorator for test functions that require no gui"
    all_tests[f.__name__] = f
    return f



@gui_test
def startup(app):
    yield wait_for_idle
    print 'result = %.3f' % (time() - gui_startup_time)

@gui_test
def paint(app):
    """
    Paint with a constant number of frames per recorded second.
    Not entirely realistic, but gives good and stable measurements.
    """
    FPS = 30
    dw = app.drawWindow
    tdw = dw.tdw

    for b in app.brushes:
        if b.name == 'redbrush':
            app.select_brush(b)

    dw.fullscreen_cb()
    yield wait_for_idle

    events = load('painting30sec.dat.gz')
    events = list(events)
    t0 = time()
    t_old = 0.0
    t_last_redraw = 0.0
    for t, x, y, pressure in events:
        if t > t_last_redraw + 1.0/FPS:
            yield wait_for_gui
            t_last_redraw = t
        dtime = t - t_old
        t_old = t
        cr = tdw.get_model_coordinates_cairo_context()
        x, y = cr.device_to_user(x, y)
        dw.doc.stroke_to(dtime, x, y, pressure)
    print 'result =', time()-t0

@gui_test
def paint_zoomed_out_5x(app):
    dw = app.drawWindow
    for i in range(5):
        dw.zoom('ZoomOut')
    for res in paint(app):
        yield res

@gui_test
def layerpaint_nozoom(app):
    dw = app.drawWindow
    dw.open_file('bigimage.ora')
    dw.doc.select_layer(len(dw.doc.layers)/2)
    for res in paint(app):
        yield res

@gui_test
def layerpaint_zoomed_out_5x(app):
    dw = app.drawWindow
    dw.open_file('bigimage.ora')
    dw.tdw.scroll(800, 1000)
    dw.doc.select_layer(len(dw.doc.layers)/3)
    for i in range(5):
        dw.zoom('ZoomOut')
    for res in paint(app):
        yield res

@gui_test
def paint_rotated(app):
    app.drawWindow.tdw.rotate(46.0/360*2*math.pi)
    for res in paint(app):
        yield res

@nogui_test
def saveload():
    from lib import document
    d = document.Document()
    t0 = t1 = time()
    d.load('bigimage.ora')
    print 'ora load time %.3f' % (time() - t1)
    t1 = time()
    d.save('test_save.ora')
    print 'ora save time %.3f' % (time() - t1)
    t1 = time()
    d.save('test_save.png')
    print 'png save time %.3f' % (time() - t1)

    print 'result = %.3f' % (time() - t0)

def scroll(app, zoom_func):
    dw = app.drawWindow
    dw.fullscreen_cb()
    dw.open_file('bigimage.ora')
    zoom_func()
    yield wait_for_idle

    t0 = time()
    N = 20
    dx = linspace(-30, 30, N)
    dy = linspace(-10, 60, N)
    for i in xrange(N):
        dw.tdw.scroll(int(dx[i]), int(dy[i]))
        yield wait_for_idle

    print 'result = %.3f' % (time() - t0)

@gui_test
def scroll_nozoom(app):
    def f(): pass
    for res in scroll(app, f):
        yield res

@gui_test
def scroll_zoomed_out_5x(app):
    dw = app.drawWindow
    def f():
        for i in range(5):
            dw.zoom('ZoomOut')
    for res in scroll(app, f):
        yield res

@gui_test
def memory_zoomed_out_5x(app):
    dw = app.drawWindow
    dw.fullscreen_cb()
    dw.open_file('bigimage.ora')
    for i in range(5):
        dw.zoom('ZoomOut')
    yield wait_for_idle
    dw.tdw.scroll(100, 120)
    yield wait_for_idle
    dw.tdw.scroll(-80, -500)
    yield wait_for_idle
    print 'result =', open('/proc/self/statm').read().split()[0]

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'SINGLE_TEST_RUN':
        func = all_tests[sys.argv[2]]
        func()
        sys.exit(0)

    from optparse import OptionParser
    parser = OptionParser('usage: %prog [options] [test1 test2 test3 ...]')
    parser.add_option('-a', '--all', action='store_true', default=False, 
                      help='run all tests')
    parser.add_option('-l', '--list', action='store_true', default=False,
                    help='list all available tests')
    parser.add_option('-c', '--count', metavar='N', type='int', default=3, 
                      help='number of repetitions (default: 3)')
    options, tests = parser.parse_args()

    if options.list:
        for name in sorted(all_tests.keys()):
            print name
        sys.exit(0)

    if not tests:
        if options.all:
            tests = list(all_tests)
        else:
            parser.print_help()
            sys.exit(1)

    for t in tests:
        if t not in all_tests:
            print 'Unknown test:', t
            sys.exit(1)

    results = []
    for t in tests:
        result = []
        for i in range(options.count):
            print '---'
            print 'running test "%s" (run %d of %d)' % (t, i+1, options.count)
            print '---'
            # spawn a new process for each test, to ensure proper cleanup
            child = subprocess.Popen(['./test_performance.py', 'SINGLE_TEST_RUN', t], stdout=subprocess.PIPE)
            output, trash = child.communicate()
            if child.returncode != 0:
                print 'FAILED'
                break
            else:
                print output,
                try:
                    value = float(output.split('result = ')[-1].strip())
                except:
                    print 'FAILED to find result in test output.'
                    result = None
                    break
                else:
                    result.append(value)
        # some time to press ctrl-c
        sleep(1.0)
        if result is None:
            sleep(3.0)
        results.append(result)
    print
    print '=== DETAILS ==='
    print 'tests =', repr(tests)
    print 'results =', repr(results)
    print
    print '=== SUMMARY ==='
    for t, result in zip(tests, results):
        if not result:
            print t, 'FAILED'
        else:
            print '%s %.3f' % (t, min(result))
