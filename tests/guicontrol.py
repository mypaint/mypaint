import gtk, glib, gobject
import traceback, tempfile, os, sys
from numpy import *

class GUI:
    """
    Class for driving the MyPaint GUI.
    """
    def __init__(self):
        self.app = None
        self.tempdir = None

    def __del__(self):
        if self.tempdir:
            os.system('rm -rf ' + self.tempdir)

    def setup(self):
        self.tempdir = tempfile.mkdtemp()
        from gui import application
        self.app = application.Application(datapath='..', confpath=self.tempdir, filenames=[])

        # fatal exceptions, please
        def excepthook(exctyp, value, tb):
            traceback.print_exception (exctyp, value, tb, None, sys.stderr)
            sys.exit(1)
        sys.excepthook = excepthook

    def signal_cb(self):
        gtk.main_quit()

    def wait_for_idle(self):
        "wait until the last mypaint idle handler has finished"
        if not self.app: self.setup()
        self.signal = False
        gobject.idle_add(self.signal_cb, priority=gobject.PRIORITY_LOW + 50)
        gtk.main()

    def wait_for_gui(self):
        "wait until all GUI updates are done, but don't wait for background tasks"
        if not self.app: self.setup()
        self.signal = False
        gobject.idle_add(self.signal_cb, priority=gobject.PRIORITY_DEFAULT_IDLE - 1)
        gtk.main()

    def wait_for_duration(self, duration):
        if not self.app: self.setup()
        self.signal = False
        gobject.timeout_add(int(duration*1000.0), self.signal_cb)
        gtk.main()

    def scroll(self, N=20):
        tdw = self.app.doc.tdw
        dx = linspace(-30, 30, N)
        dy = linspace(-10, 60, N)
        for i in xrange(N):
            tdw.scroll(int(dx[i]), int(dy[i]))
            self.wait_for_idle()
        # jump back to the start
        for i in xrange(N):
            tdw.scroll(-int(dx[i]), -int(dy[i]))

