## Quick Profiling HOWTO

Install gprof2dot.py into your $PATH:
<https://github.com/jrfonseca/gprof2dot>
Also install graphviz (to render PNGs),
and an image viewer.

Now run

    tests/test_performance.py -c 1 -s load_ora

For more options see

    test/test_performance.py -h

You can also start the profiler from within MyPaint (Menu→Help→Debug).
Works best with a keyboard shortcut assigned through the menu.

Use the results with care. There is profiler overhead, the work being
done while idling will sum up (e.g. just hovering with the stylus), and
X11 async stuff is probably filtered out completely.

To profile the code written in C you have to use something else
(e.g. `oprofile`).
