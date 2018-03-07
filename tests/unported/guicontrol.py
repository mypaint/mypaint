from __future__ import division, print_function

import traceback
import tempfile
import os
import sys

import numpy as np

import gi
try:
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk
    from gi.repository import GObject
except:
    raise


class GUI:
    """
    Class for driving the MyPaint GUI.
    """
    def __init__(self):
        self.app = None
        self.tempdir = None
        sys.argv_unicode = sys.argv
        # FileHandler.save_file passes this to gtk recent_manager

    def __del__(self):
        if self.tempdir:
            os.system('rm -rf ' + self.tempdir)

    def setup(self):
        self.tempdir = tempfile.mkdtemp()
        from gui import application
        os.system('cp -a brushes ' + self.tempdir)

        app_statedirs = application.StateDirs(
            app_data = u'..',
            app_icons = u'../desktop',
            user_data = unicode(self.tempdir),
            user_config = unicode(self.tempdir),
        )
        self.app = application.Application(
            filenames = [],
            state_dirs = app_statedirs,
            version = 'guicontrol_testing',
        )

        # ignore mouse movements during testing (creating extra strokes)
        def motion_ignore_cb(*junk1, **junk2):
            pass
        self.app.doc.tdw.motion_notify_cb = motion_ignore_cb

        # fatal exceptions, please
        def excepthook(exctyp, value, tb):
            traceback.print_exception(exctyp, value, tb, None, sys.stderr)
            sys.exit(1)
        sys.excepthook = excepthook

    def signal_cb(self):
        self.waiting = False

    def wait_for_idle(self):
        "wait until the last mypaint idle handler has finished"
        if not self.app:
            self.setup()
        self.signal = False
        GObject.idle_add(self.signal_cb, priority=GObject.PRIORITY_LOW + 50)
        self.waiting = True
        while self.waiting:
            Gtk.main_iteration()

    def wait_for_gui(self):
        "wait until all GUI updates are done, but don't wait for bg tasks"
        if not self.app:
            self.setup()
        self.signal = False
        GObject.idle_add(
            self.signal_cb,
            priority=GObject.PRIORITY_DEFAULT_IDLE - 1,
        )
        self.waiting = True
        while self.waiting:
            Gtk.main_iteration()

    def wait_for_duration(self, duration):
        if not self.app:
            self.setup()
        self.signal = False
        GObject.timeout_add(int(duration * 1000.0), self.signal_cb)
        self.waiting = True
        while self.waiting:
            Gtk.main_iteration()

    def scroll(self, N=20):
        tdw = self.app.doc.tdw
        dx = np.linspace(-30, 30, N)
        dy = np.linspace(-10, 60, N)
        for i in range(N):
            tdw.scroll(int(dx[i]), int(dy[i]))
            self.wait_for_idle()
        # jump back to the start
        for i in xrange(N):
            tdw.scroll(-int(dx[i]), -int(dy[i]))

    def zoom_out(self, steps):
        doc = self.app.doc
        for i in range(steps):
            doc.zoom(doc.ZOOM_OUTWARDS)
