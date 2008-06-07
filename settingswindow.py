# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

"preferences dialog"
import gtk, os
from functionwindow import CurveWidget
import mydrawwidget

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app

        self.set_title('Settings')
        self.connect('delete-event', self.app.hide_window_cb)

        v = gtk.VBox()
        self.add(v)

        l = gtk.Label('Global Pressure Mapping')
        l.set_alignment(0.0, 0.0)
        v.pack_start(l, expand=False)


        t = gtk.Table(4, 4)
        self.cv = CurveWidget(self.pressure_curve_changed_cb, magnetic=False)
        t.attach(self.cv, 0, 3, 0, 3, gtk.EXPAND | gtk.FILL, gtk.EXPAND | gtk.FILL, 5, 0)
        l1 = gtk.Label('1.0')
        l2 = gtk.Label('')
        l3 = gtk.Label('0.0')
        t.attach(l1, 3, 4, 0, 1, 0, 0, 5, 0)
        t.attach(l2, 3, 4, 1, 2, 0, gtk.EXPAND, 5, 0)
        t.attach(l3, 3, 4, 2, 3, 0, 0, 5, 0)
        l4 = gtk.Label('0.0')
        l5 = gtk.Label('pressure reported by tablet')
        l6 = gtk.Label('1.0')
        t.attach(l4, 0, 1, 3, 4, 0, 0, 5, 0)
        t.attach(l5, 1, 2, 3, 4, gtk.EXPAND, 0, 5, 0)
        t.attach(l6, 2, 3, 3, 4, 0, 0, 5, 0)
        v.pack_start(t)

        self.ip_cb = gtk.CheckButton('ignore pressure (use click state instead)')
        self.ip_cb.connect('toggled', self.ignore_pressure_cb)
        v.pack_start(self.ip_cb, expand=False)

        h = gtk.HBox()

        b = gtk.Button("Revert")
        b.connect('clicked', self.load_settings)
        h.pack_start(b, expand=False)

        b = gtk.Button("Save")
        b.connect('clicked', self.save_settings)
        h.pack_end(b, expand=False)

        v.pack_start(h, expand=False)

        self.filename = os.path.join(self.app.confpath, 'settings.conf')

        self.load_settings()

    def save_settings(self, *trash):
        f = open(self.filename, 'w')
        print >>f, 'global_pressure_mapping =', self.cv.points
        print >>f, 'ignore_pressure =', self.ignore_pressure
        f.close()

    def load_settings(self, *trash):
        # 1. set defaults
        self.global_pressure_mapping = [(0.0, 1.0), (1.0, 0.0)]
        self.ignore_pressure = 0
        # 2. parse config file
        if os.path.exists(self.filename):
            exec open(self.filename) in self.__dict__
        # 3. apply
        self.apply_settings()

    def apply_settings(self):
        p = self.cv.points = self.global_pressure_mapping
        self.cv.queue_draw()
        if len(p) == 2 and abs(p[0][1]-1.0)+abs(p[1][1]-0.0) < 0.0001:
            # 1:1 mapping (mapping disabled)
            mydrawwidget.global_pressure_mapping_set_n(0)
        else:
            mydrawwidget.global_pressure_mapping_set_n(len(p))
            for i, (x, y) in enumerate(p):
                mydrawwidget.global_pressure_mapping_set_point(i, x, 1.0-y)
        mydrawwidget.global_ignore_pressure_set(self.ignore_pressure)

        self.ip_cb.set_active(self.ignore_pressure)

    def pressure_curve_changed_cb(self, widget):
        self.global_pressure_mapping = self.cv.points[:]
        self.apply_settings()


    def ignore_pressure_cb(self, widget):
        self.ignore_pressure = self.ip_cb.get_active()
        self.apply_settings()
