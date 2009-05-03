# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"preferences dialog"
import gtk, os
gdk = gtk.gdk

from functionwindow import CurveWidget
from lib import mypaintlib

device_modes = ['disabled','screen','window']

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app

        self.set_title('Settings')
        self.connect('delete-event', self.app.hide_window_cb)

        self.applying = True

        v_outside = gtk.VBox()
        self.add(v_outside)

        nb = gtk.Notebook()
        v_outside.pack_start(nb, expand=True, padding=5)

        ### pressure tab

        v = gtk.VBox()
        nb.append_page(v, gtk.Label('Pressure'))

        l = gtk.Label()
        l.set_alignment(0.0, 0.0)
        l.set_markup('<b><span size="large">Global Pressure Mapping</span></b>')
        v.pack_start(l, expand=False, padding=5)

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
        l5 = gtk.Label('0.5\n(mouse button)')
        l5.set_justify(gtk.JUSTIFY_CENTER)
        l6 = gtk.Label('1.0')
        t.attach(l4, 0, 1, 3, 4, 0, 0, 5, 0)
        t.attach(l5, 1, 2, 3, 4, gtk.EXPAND, 0, 5, 0)
        t.attach(l6, 2, 3, 3, 4, 0, 0, 5, 0)
        v.pack_start(t)

        v.pack_start(gtk.HSeparator(), expand=False, padding=5)

        h = gtk.HBox()
        h.pack_start(gtk.Label('Mode for input devices: '), expand=False)
        combo = self.input_devices_combo = gtk.combo_box_new_text()
        for s in device_modes:
            combo.append_text(s)
        combo.connect('changed', self.input_devices_combo_changed_cb)
        h.pack_start(combo, expand=True)
        v.pack_start(h, expand=False)

        ### paths tab

        v = gtk.VBox()
        nb.append_page(v, gtk.Label('Paths'))

        l = gtk.Label()
        l.set_alignment(0.0, 0.0)
        l.set_markup('<b><span size="large">Save as Scrap</span></b>')
        v.pack_start(l, expand=False, padding=5)
        l = gtk.Label('Path or filename prefix for "Save as Scrap"')
        l.set_alignment(0.0, 0.0)
        v.pack_start(l, expand=False)

        self.prefix_entry = gtk.Entry()
        self.prefix_entry.connect('changed', self.prefix_entry_changed_cb)
        v.pack_start(self.prefix_entry, expand=False)

        ### end tabs

        h = gtk.HBox()
        h.set_border_width(3)

        b = gtk.Button("Save")
        b.connect('clicked', self.save_settings_and_hide)
        h.pack_end(b, expand=False)

        b = gtk.Button("Revert")
        b.connect('clicked', self.load_settings)
        h.pack_end(b, expand=False)

        v_outside.pack_start(h, expand=False)

        self.filename = os.path.join(self.app.confpath, 'settings.conf')

        self.applying = False
        self.load_settings(None, startup=True)

    def save_settings_and_hide(self, *trash):
        f = open(self.filename, 'w')
        print >>f, 'global_pressure_mapping =', self.cv.points
        print >>f, 'save_scrap_prefix =', repr(self.save_scrap_prefix)
        print >>f, 'input_devices_mode =', repr(self.input_devices_mode)
        f.close()
        self.hide()

    def load_settings(self, widget, startup=False):
        # 1. set defaults
        self.global_pressure_mapping = [(0.0, 1.0), (1.0, 0.0)]
        self.save_scrap_prefix = 'scrap'
        self.input_devices_mode = 'screen'
        # 2. parse config file
        if os.path.exists(self.filename):
            exec open(self.filename) in self.__dict__
        # 3. apply
        self.apply_settings()

        if startup and not self.pressure_devices:
            print 'No pressure sensitive devices found.'

    def apply_settings(self):
        if self.applying:
            return
        self.applying = True
        p = self.cv.points = self.global_pressure_mapping
        self.cv.queue_draw()
        if len(p) == 2 and abs(p[0][1]-1.0)+abs(p[1][1]-0.0) < 0.0001:
            # 1:1 mapping (mapping disabled)
            self.app.global_pressure_mapping = None
            self.app.drawWindow.tdw.pressure_mapping = None
        else:
            # TODO: maybe replace this stupid mapping by a hard<-->soft slider?
            m = mypaintlib.Mapping(1)
            m.set_n(0, len(p))
            for i, (x, y) in enumerate(p):
                m.set_point(0, i, x, 1.0-y)

            def mapping(pressure):
                return m.calculate_single_input(pressure)
            self.app.drawWindow.tdw.pressure_mapping = mapping

        self.prefix_entry.set_text(self.save_scrap_prefix)

        self.input_devices_combo.set_active(device_modes.index(self.input_devices_mode))

        # init extended input devices
        self.pressure_devices = []
        for device in gdk.devices_list():
            #if device.source in [gdk.SOURCE_PEN, gdk.SOURCE_ERASER]:
            # The above seems to be True sometimes for a normal
            # USB Mouse (bug #11215). Using different check now:
            for use, val_min, val_max in device.axes:
                if use == gdk.AXIS_PRESSURE:
                    self.pressure_devices.append(device.name)
                    mode = getattr(gdk, 'MODE_' + self.input_devices_mode.upper())
                    if device.mode != mode:
                        print 'Setting %s mode for %s' % (self.input_devices_mode, device.name)
                        device.set_mode(mode)
                    break
        self.applying = False

    def input_devices_combo_changed_cb(self, window):
        self.input_devices_mode = self.input_devices_combo.get_active_text()
        self.apply_settings()

    def pressure_curve_changed_cb(self, widget):
        self.global_pressure_mapping = self.cv.points[:]
        self.apply_settings()

    def prefix_entry_changed_cb(self, widget):
        self.save_scrap_prefix = widget.get_text()

