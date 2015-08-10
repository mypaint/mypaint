## Debugging

By default, our use of Python's `logging` module
is noisy about errors, warnings, and general informational stuff,
but silent about anything with a lower priority.
To see all messages, set the `MYPAINT_DEBUG` environment variable.

    MYPAINT_DEBUG=1 ./mypaint -c /tmp/cfgtmp_throwaway_1

MyPaint normally logs Python exception backtraces to the terminal
and to a dialog within the application.

To debug segfaults in C/C++ code, use `gdb` with a debug build,
after first making sure you have debugging symbols for Python and GTK3.

    sudo apt-get install gdb python2.7-dbg libgtk-3-0-dbg
    scons debug=1
    export MYPAINT_DEBUG=1
    gdb -ex r --args python ./mypaint -c /tmp/cfgtmp_throwaway_2

Execute ``bt`` within the gdb environment for a full backtrace.
See also: https://wiki.python.org/moin/DebuggingWithGdb

## Profiling

MyPaint can run the cProfile code profiler interactively if you find
that performance is lagging in a particular area, and want to figure out
what functions need optimizing. Assigning the "Start/Stop Profiling..."
command to a spare function key like `F9` allows profiling to be flipped
on and off quickly while you do something that needs profiling.

When profiling stops, MyPaint creates a temporary output folder and
writes its output files there. It then opens the temp folder in your
desktop environment's file and directory browser. You get a single
uniquely named temporary profiling folder for each instance of MyPaint
you run.

One word of warning: the temporary folders are removed at the end of the
MyPaint session to avoid tempdir clutter. If you need to keep the
profiler's output, be sure to copy them safely somewhere first.

The most convenient output we support is graphical (PNG format).
Install [gprof2dot.py](https://github.com/jrfonseca/gprof2dot)
and [graphviz](http://www.graphviz.org/) for the prettiest results.
If you need to run the commands manually on `.pstat` output,
try:

    gprof2dot.py -f pstats -o output.dot output.pstat
    dot -Tpng -o output.png output.dot

Our profiler tries to do this for you.
