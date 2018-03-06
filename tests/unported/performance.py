#!/usr/bin/env python
from __future__ import division, print_function

import sys
import os
import subprocess
import cProfile
from time import time, sleep
import distutils.spawn

import numpy as np

import guicontrol

start_measurement = -1
stop_measurement = -2

all_tests = {}


def run_test(testfunction, profile=None):
    """Run a single test
    testfunction must be a generator (using yield)
    """
    tst = testfunction()

    time_total = 0.0
    for res in tst:
        assert res == start_measurement, res

        def run_function_under_test():
            res = next(tst)
            assert res == stop_measurement

        t0 = time()
        if profile:
            profile.runcall(run_function_under_test)
        else:
            run_function_under_test()
        time_total += time() - t0

    if time_total:
        print('result =', time_total)
    else:
        # test did not make time measurements,
        # it will print its own result (eg. memory)
        pass


def nogui_test(f):
    "decorator for test functions that require no gui"
    all_tests[f.__name__] = f
    return f


def gui_test(f):
    "decorator for test functions that require a gui"
    def f2():
        gui = guicontrol.GUI()
        for action in f(gui):
            yield action
    all_tests[f.__name__] = f2
    return f


@gui_test
def startup(gui):
    yield start_measurement
    gui.wait_for_idle()
    yield stop_measurement


@gui_test
def paint(gui):
    """
    Paint with a constant number of frames per recorded second.
    Not entirely realistic, but gives good and stable measurements.
    """
    gui.wait_for_gui()
    FPS = 30
    dw = gui.app.drawWindow
    gui_doc = gui.app.doc
    tdw = gui_doc.tdw

    b = gui.app.brushmanager.get_brush_by_name('redbrush')
    assert b, 'brush not found'

    dw.fullscreen_cb()
    gui.wait_for_idle()
    gui.app.brushmanager.select_brush(b)

    # fullscreen seems to take some time to get through...
    gui.wait_for_duration(1.5)

    gui.wait_for_idle()

    events = np.loadtxt('painting30sec.dat')
    yield start_measurement
    t_old = 0.0
    t_last_redraw = 0.0
    mode = gui_doc.modes.top
    model = gui_doc.model
    for t, x, y, pressure in events:
        if t > t_last_redraw + 1.0 / FPS:
            gui.wait_for_gui()
            t_last_redraw = t
        dtime = t - t_old
        t_old = t
        x, y = tdw.display_to_model(x, y)

        mode.stroke_to(model, dtime, x, y, pressure, 0.0, 0.0)
    yield stop_measurement


@gui_test
def paint_zoomed_out_5x(gui):
    gui.wait_for_idle()
    gui.zoom_out(5)
    for res in paint(gui):
        yield res


@gui_test
def layerpaint_nozoom(gui):
    gui.wait_for_idle()
    gui.app.filehandler.open_file('bigimage.ora')
    gui_doc = gui.app.doc
    gui_doc.model.select_layer(index=len(gui_doc.model.layer_stack) // 2)
    for res in paint(gui):
        yield res


@gui_test
def layerpaint_zoomed_out_5x(gui):
    gui.wait_for_idle()
    gui_doc = gui.app.doc
    gui.app.filehandler.open_file('bigimage.ora')
    gui_doc.tdw.scroll(800, 1000)
    gui_doc.model.select_layer(index=len(gui_doc.model.layer_stack) // 3)
    gui.zoom_out(5)
    for res in paint(gui):
        yield res


@gui_test
def paint_rotated(gui):
    gui.wait_for_idle()
    gui.app.doc.tdw.rotate(46.0 / 360 * 2 * np.pi)
    for res in paint(gui):
        yield res


@nogui_test
def load_ora():
    from lib import document
    d = document.Document()
    yield start_measurement
    d.load('bigimage.ora')
    yield stop_measurement


@nogui_test
def save_ora():
    from lib import document
    d = document.Document()
    d.load('bigimage.ora')
    yield start_measurement
    d.save('test_save.ora')
    yield stop_measurement


@nogui_test
def save_ora_again():
    from lib import document
    d = document.Document()
    d.load('bigimage.ora')
    d.save('test_save.ora')
    yield start_measurement
    d.save('test_save.ora')
    yield stop_measurement


@nogui_test
def save_png():
    from lib import document
    d = document.Document()
    d.load('bigimage.ora')
    yield start_measurement
    d.save('test_save.png')
    yield stop_measurement


@nogui_test
def save_png_layer():
    from lib import document
    d = document.Document()
    d.load('biglayer.png')
    yield start_measurement
    d.layer_stack.current.save_as_png('test_save.png')
    yield stop_measurement


@nogui_test
def brushengine_paint_hires():
    from lib import tiledsurface, brush
    s = tiledsurface.Surface()
    with open('brushes/v2/watercolor.myb') as fp:
        bi = brush.BrushInfo(fp.read())
    b = brush.Brush(bi)

    events = np.loadtxt('painting30sec.dat')
    t_old = events[0][0]
    yield start_measurement
    s.begin_atomic()
    trans_time = 0.0
    for t, x, y, pressure in events:
        dtime = t - t_old
        t_old = t
        b.stroke_to(s.backend, x * 5, y * 5, pressure, 0.0, 0.0, dtime)

        trans_time += dtime
        if trans_time > 0.05:
            trans_time = 0.0
            s.end_atomic()
            s.begin_atomic()

    s.end_atomic()
    yield stop_measurement
    # s.save('test_paint_hires.png') # approx. 3000x3000


@gui_test
def scroll_nozoom(gui):
    gui.wait_for_idle()
    dw = gui.app.drawWindow
    dw.fullscreen_cb()
    gui.app.filehandler.open_file('bigimage.ora')
    gui.wait_for_idle()
    yield start_measurement
    gui.scroll()
    yield stop_measurement


@gui_test
def scroll_nozoom_onelayer(gui):
    gui.wait_for_idle()
    dw = gui.app.drawWindow
    dw.fullscreen_cb()
    gui.app.filehandler.open_file('biglayer.png')
    gui.wait_for_idle()
    yield start_measurement
    gui.scroll()
    yield stop_measurement


@gui_test
def scroll_zoomed_out_1x_onelayer(gui):
    gui.wait_for_idle()
    dw = gui.app.drawWindow
    dw.fullscreen_cb()
    gui.app.filehandler.open_file('biglayer.png')
    gui.zoom_out(1)
    gui.wait_for_idle()
    yield start_measurement
    gui.scroll()
    yield stop_measurement


@gui_test
def scroll_zoomed_out_2x_onelayer(gui):
    gui.wait_for_idle()
    dw = gui.app.drawWindow
    dw.fullscreen_cb()
    gui.app.filehandler.open_file('biglayer.png')
    gui.zoom_out(2)
    gui.wait_for_idle()
    yield start_measurement
    gui.scroll()
    yield stop_measurement


@gui_test
def scroll_zoomed_out_5x(gui):
    gui.wait_for_idle()
    dw = gui.app.drawWindow
    dw.fullscreen_cb()
    gui.app.filehandler.open_file('bigimage.ora')
    gui.zoom_out(5)
    gui.wait_for_idle()
    yield start_measurement
    gui.scroll()
    yield stop_measurement


@gui_test
def memory_zoomed_out_5x(gui):
    gui.wait_for_idle()
    dw = gui.app.drawWindow
    dw.fullscreen_cb()
    gui.app.filehandler.open_file('bigimage.ora')
    gui.zoom_out(5)
    gui.wait_for_idle()
    gui.scroll()
    with open('/proc/self/statm') as statm:
        print('result =', statm.read().split()[0])
    if False:
        yield None  # just to make this function iterator


@gui_test
def memory_after_startup(gui):
    gui.wait_for_idle()
    sleep(1)
    gui.wait_for_idle()
    sleep(1)
    gui.wait_for_idle()
    with open('/proc/self/statm') as statm:
        print('result =', statm.read().split()[0])
    if False:
        yield None  # just to make this function iterator


if __name__ == '__main__':

    os.chdir(os.path.dirname(__file__))
    sys.path.insert(0, '..')
    sys.path.insert(0, '.')

    if len(sys.argv) == 4 and sys.argv[1] == 'SINGLE_TEST_RUN':
        func = all_tests[sys.argv[2]]
        if sys.argv[3] == 'NONE':
            run_test(func)
        else:
            profile = cProfile.Profile()
            run_test(func, profile)
            profile.dump_stats(sys.argv[3])
        sys.exit(0)

    from optparse import OptionParser
    parser = OptionParser('usage: %prog [options] [test1 test2 test3 ...]')
    parser.add_option(
        '-a',
        '--all',
        action='store_true',
        default=False,
        help='run all tests'
    )
    parser.add_option(
        '-l',
        '--list',
        action='store_true',
        default=False,
        help='list all available tests'
    )
    parser.add_option(
        '-c',
        '--count',
        metavar='N',
        type='int',
        default=3,
        help='number of repetitions (default: 3)'
    )
    parser.add_option(
        '-p',
        '--profile',
        metavar='PREFIX',
        help='dump cProfile info to PREFIX_TESTNAME_N.pstats'
    )
    parser.add_option(
        '-s',
        '--show-profile',
        action='store_true',
        default=False,
        help='run cProfile, gprof2dot.py and show last result'
    )
    options, tests = parser.parse_args()

    if options.list:
        for name in sorted(all_tests.keys()):
            print(name)
        sys.exit(0)

    if not tests:
        if options.all:
            tests = list(all_tests)
        else:
            parser.print_help()
            sys.exit(1)

    for t in tests:
        if t not in all_tests:
            print('Unknown test:', t)
            sys.exit(1)

    results = []
    for t in tests:
        result = []
        for i in range(options.count):
            print('---')
            print('running test "%s" (run %d of %d)'
                  % (t, i + 1, options.count))
            print('---')
            # spawn a new process for each test, to ensure proper cleanup
            args = [sys.executable, './test_performance.py',
                    'SINGLE_TEST_RUN', t, 'NONE']
            if options.profile or options.show_profile:
                if options.show_profile:
                    fname = 'tmp.pstats'
                else:
                    fname = '%s_%s_%d.pstats' % (options.profile, t, i)
                args[4] = fname
            child = subprocess.Popen(args, stdout=subprocess.PIPE)
            output, junk = child.communicate()
            if child.returncode != 0:
                print('FAILED')
                break
            else:
                print(output, end=' ')
                try:
                    value = float(output.split('result = ')[-1].strip())
                except:
                    print('FAILED to find result in test output.')
                    result = None
                    break
                else:
                    result.append(value)
        # some time to press ctrl-c
        sleep(1.0)
        if result is None:
            sleep(3.0)
        results.append(result)
    print()
    print('=== DETAILS ===')
    print('tests =', repr(tests))
    print('results =', repr(results))
    print()
    print('=== SUMMARY ===')
    fail = False
    for t, result in zip(tests, results):
        if not result:
            print(t, 'FAILED')
            fail = True
        else:
            print('%s %.3f' % (t, min(result)))
    if fail:
        sys.exit(1)

    if options.show_profile:
        gprof2dot = "gprof2dot.py" \
            if distutils.spawn.find_executable("gprof2dot.py") \
            else "gprof2dot"
        viewer = "feh" \
            if distutils.spawn.find_executable("feh") \
            else "eog"
        # FIXME: use gui.profiling's improved code somehow
        os.system(
            '%s -f pstats tmp.pstats | dot -Tpng -o tmp.png && %s tmp.png'
            % (gprof2dot, viewer)
        )
