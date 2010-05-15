# This file is part of MyPaint.
# Copyright (C) 2008-2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"preferences dialog"
from gettext import gettext as _
import gtk, os
gdk = gtk.gdk

from functionwindow import CurveWidget
from lib import mypaintlib
import windowing

device_modes = ['disabled','screen','window']
RESPONSE_REVERT = 1

class Window(windowing.Dialog):
    def __init__(self, app):
        flags = gtk.DIALOG_DESTROY_WITH_PARENT
        buttons = (gtk.STOCK_REVERT_TO_SAVED, RESPONSE_REVERT,
                   gtk.STOCK_OK, gtk.RESPONSE_ACCEPT)
        windowing.Dialog.__init__(self, app=app, title=_('Preferences'),
                                  parent=app.drawWindow, flags=flags,
                                  buttons=buttons)
        self.filename = os.path.join(self.app.confpath, 'settings.conf')
        self.applying = True

        self.connect('delete-event', self.app.hide_window_cb)
        self.connect('response', self.on_response)


        nb = gtk.Notebook()
        self.vbox.pack_start(nb, expand=True, padding=5)

        ### pressure tab

        v = gtk.VBox()
        nb.append_page(v, gtk.Label(_('Pressure')))

        l = gtk.Label()
        l.set_alignment(0.0, 0.0)
        l.set_markup(_('<b><span size="large">Global Pressure Mapping</span></b>'))
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
        l5 = gtk.Label(_('0.5\n(mouse button)'))
        l5.set_justify(gtk.JUSTIFY_CENTER)
        l6 = gtk.Label('1.0')
        t.attach(l4, 0, 1, 3, 4, 0, 0, 5, 0)
        t.attach(l5, 1, 2, 3, 4, gtk.EXPAND, 0, 5, 0)
        t.attach(l6, 2, 3, 3, 4, 0, 0, 5, 0)
        v.pack_start(t)

        v.pack_start(gtk.HSeparator(), expand=False, padding=5)

        h = gtk.HBox()
        h.pack_start(gtk.Label(_('Mode for input devices: ')), expand=False)
        combo = self.input_devices_combo = gtk.combo_box_new_text()
        for s in device_modes:
            combo.append_text(s)
        combo.connect('changed', self.input_devices_combo_changed_cb)
        h.pack_start(combo, expand=True)
        v.pack_start(h, expand=False)

        ### paths tab

        v = gtk.VBox()
        nb.append_page(v, gtk.Label(_('Paths')))

        l = gtk.Label()
        l.set_alignment(0.0, 0.0)
        l.set_markup(_('<b><span size="large">Save as Scrap</span></b>'))
        v.pack_start(l, expand=False, padding=5)
        l = gtk.Label(_('Path and filename prefix for "Save Next Scrap"'))
        l.set_alignment(0.0, 0.0)
        v.pack_start(l, expand=False)

        self.prefix_entry = gtk.Entry()
        self.prefix_entry.connect('changed', self.prefix_entry_changed_cb)
        v.pack_start(self.prefix_entry, expand=False)

        ### View tab
        view_vbox = gtk.VBox()
        l = gtk.Label('Default zoom')
        l.set_alignment(0.0, 0.0)
        nb.append_page(view_vbox, gtk.Label(_('View'))) 
        combo = self.defaultzoom_combo = gtk.combo_box_new_text()
        # Different from doc.zoomlevel_values because we only want a subset
        self.defaultzoom_values = [0.25, 0.50, 1.0, 2.0]
        for val in self.defaultzoom_values:
            combo.append_text('%d%%' % (val*100))
        combo.connect('changed', self.defaultzoom_combo_changed_cb)
        view_vbox.pack_start(l, expand=False)
        view_vbox.pack_start(combo, expand=False)

        self.applying = False
        self.load_settings(startup=True)

    def on_response(self, dialog, response, *args):
        if response == gtk.RESPONSE_ACCEPT:
            self.save_settings_and_hide()
        elif response == RESPONSE_REVERT:
            self.load_settings()

    def save_settings_and_hide(self, *trash):
        self.app.save_settings()
        self.hide()

    def load_settings(self, startup=False):
        self.app.load_settings()
        self.apply_settings()

        if startup and not self.app.pressure_devices:
            print 'No pressure sensitive devices found.'

    def apply_settings(self):
        if self.applying:
            return
        self.applying = True
        self.update_input_mapping()
        self.update_ui()
        self.update_input_devices()
        self.applying = False

    def update_input_mapping(self):
        p = self.app.preferences['input.global_pressure_mapping']
        if len(p) == 2 and abs(p[0][1]-1.0)+abs(p[1][1]-0.0) < 0.0001:
            # 1:1 mapping (mapping disabled)
            self.app.doc.tdw.pressure_mapping = None
        else:
            # TODO: maybe replace this stupid mapping by a hard<-->soft slider?
            m = mypaintlib.Mapping(1)
            m.set_n(0, len(p))
            for i, (x, y) in enumerate(p):
                m.set_point(0, i, x, 1.0-y)

            def mapping(pressure):
                return m.calculate_single_input(pressure)
            self.app.doc.tdw.pressure_mapping = mapping

    def update_ui(self):
        p = self.app.preferences
        self.cv.points = p['input.global_pressure_mapping']
        self.prefix_entry.set_text(p['saving.scrap_prefix'])
        mode = device_modes.index(p['input.device_mode'])
        self.input_devices_combo.set_active(mode)
        zoom = self.app.doc.zoomlevel_values[self.app.doc.zoomlevel]
        zoomlevel = self.defaultzoom_values.index(zoom)
        self.defaultzoom_combo.set_active(zoomlevel)

        self.cv.queue_draw()

    def update_input_devices(self):
        # init extended input devices
        self.app.pressure_devices = []
        for device in gdk.devices_list():
            #print device.name, device.source

            #if device.source in [gdk.SOURCE_PEN, gdk.SOURCE_ERASER]:
            # The above contition is True sometimes for a normal USB
            # Mouse. https://gna.org/bugs/?11215
            # In fact, GTK also just guesses this value from device.name.

            last_word = device.name.split()[-1].lower()
            if last_word == 'pad':
                # Setting the intuos3 pad into "screen mode" causes
                # glitches when you press a pad-button in mid-stroke,
                # and it's not a pointer device anyway. But it reports
                # axes almost identical to the pen and eraser.
                #
                # device.name is usually something like "wacom intuos3 6x8 pad" or just "pad"
                print 'Ignoring "%s" (probably wacom keypad device)' % device.name
                continue
            if last_word == 'cursor':
                # this is a "normal" mouse and does not work in screen mode
                print 'Ignoring "%s" (probably wacom mouse device)' % device.name
                continue

            for use, val_min, val_max in device.axes:
                # Some mice have a third "pressure" axis, but without
                # minimum or maximum. https://gna.org/bugs/?14029
                if use == gdk.AXIS_PRESSURE and val_min != val_max:
                    if 'mouse' in device.name.lower():
                        # Real fix for the above bug https://gna.org/bugs/?14029
                        print 'Ignoring "%s" (probably a mouse, but it reports extra axes)' % device.name
                        continue

                    self.app.pressure_devices.append(device.name)
                    modesetting = self.app.preferences['input.device_mode']
                    mode = getattr(gdk, 'MODE_' + modesetting.upper())
                    if device.mode != mode:
                        print 'Setting %s mode for "%s"' % (modesetting, device.name)
                        device.set_mode(mode)
                    break

    # Callbacks for widgets that manipulate settings
    def input_devices_combo_changed_cb(self, window):
        mode = self.input_devices_combo.get_active_text()
        self.app.preferences['input.device_mode'] = mode
        self.apply_settings()

    def pressure_curve_changed_cb(self, widget):
        self.app.preferences['input.global_pressure_mapping'] = self.cv.points[:]
        self.apply_settings()

    def prefix_entry_changed_cb(self, widget):
        self.app.preferences['saving.scrap_prefix'] = widget.get_text()

    def defaultzoom_combo_changed_cb(self, widget):
        zoomlevel = self.defaultzoom_combo.get_active()
        zoom = self.defaultzoom_values[zoomlevel]
        self.app.preferences['view.default_zoom'] = zoom
