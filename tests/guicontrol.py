import gtk, glib, gobject
import traceback, tempfile, os, sys

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
        self.signal = True

    def wait_for_idle(self):
        "wait until the last mypaint idle handler has finished"
        if not self.app: self.setup()
        self.signal = False
        gobject.idle_add(self.signal_cb, priority=gobject.PRIORITY_LOW + 50)
        while not self.signal:
            gtk.main_iteration()

    def wait_for_gui(self):
        "wait until all GUI updates are done, but don't wait for background tasks"
        if not self.app: self.setup()
        self.signal = False
        gobject.idle_add(self.signal_cb, priority=gobject.PRIORITY_DEFAULT_IDLE - 1)
        while not self.signal:
            gtk.main_iteration()

    def wait_for_duration(self, duration):
        if not self.app: self.setup()
        self.signal = False
        gobject.timeout_add(int(duration*1000.0), self.signal_cb)
        while not self.signal:
            gtk.main_iteration()

