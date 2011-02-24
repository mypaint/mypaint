# This file is part of MyPaint.
# Copyright (C) 2008-2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"preferences dialog"
from bisect import bisect_left
from gettext import gettext as _
import gtk, os
gdk = gtk.gdk

from functionwindow import CurveWidget
from lib import mypaintlib
import windowing, filehandling

device_modes = ['disabled','screen','window']
RESPONSE_REVERT = 1

class Window(windowing.Dialog):
    '''Window for manipulating preferences.'''

    def __init__(self, app):
        flags = gtk.DIALOG_DESTROY_WITH_PARENT
        buttons = (gtk.STOCK_REVERT_TO_SAVED, RESPONSE_REVERT,
                   gtk.STOCK_OK, gtk.RESPONSE_ACCEPT)
        windowing.Dialog.__init__(self, app=app, title=_('Preferences'),
                                  parent=app.drawWindow, flags=flags,
                                  buttons=buttons)
        self.connect('delete-event', self.app.hide_window_cb)
        self.connect('response', self.on_response)

        # Set up widgets
        nb = gtk.Notebook()
        self.vbox.pack_start(nb, expand=True, padding=5)

        ### Input tab
        v = gtk.VBox()
        nb.append_page(v, gtk.Label(_('Input')))

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

        self.enable_history_popup_checkbox = c = gtk.CheckButton(_('Enable right-click color history'))
        c.connect('toggled', self.enable_history_popup_toggled_cb)
        v.pack_start(c, expand=False)

        ### Saving tab
        saving_vbox = gtk.VBox()
        nb.append_page(saving_vbox, gtk.Label(_('Saving')))

        l = gtk.Label(_('Default file format for saving'))
        l.set_alignment(0.0, 0.0)
        combo = self.defaultsaveformat_combo = gtk.combo_box_new_text()
        self.defaultsaveformat_values = [filehandling.SAVE_FORMAT_ORA, 
            filehandling.SAVE_FORMAT_PNGSOLID, filehandling.SAVE_FORMAT_JPEG]
        for saveformat in self.defaultsaveformat_values:
            format_desc = self.app.filehandler.saveformats[saveformat][0]
            combo.append_text(format_desc)
        combo.connect('changed', self.defaultsaveformat_combo_changed_cb)
        saving_vbox.pack_start(l, expand=False)
        saving_vbox.pack_start(combo, expand=False)

        l = gtk.Label(_('Path and filename prefix for "Save Next Scrap"'))
        l.set_alignment(0.0, 0.0)
        saving_vbox.pack_start(l, expand=False)
        self.prefix_entry = gtk.Entry()
        self.prefix_entry.connect('changed', self.prefix_entry_changed_cb)
        saving_vbox.pack_start(self.prefix_entry, expand=False)

        ### View tab
        view_vbox = gtk.VBox()
        l = gtk.Label(_('Default zoom'))
        l.set_alignment(0.0, 0.0)
        nb.append_page(view_vbox, gtk.Label(_('View'))) 
        combo = self.defaultzoom_combo = gtk.combo_box_new_text()
        # Different from doc.zoomlevel_values because we only want a subset
        # - keep sorted for bisect
        self.defaultzoom_values = [0.25, 0.50, 1.0, 2.0]
        for val in self.defaultzoom_values:
            combo.append_text('%d%%' % (val*100))
        combo.connect('changed', self.defaultzoom_combo_changed_cb)
        view_vbox.pack_start(l, expand=False)
        view_vbox.pack_start(combo, expand=False)

    def on_response(self, dialog, response, *args):
        if response == gtk.RESPONSE_ACCEPT:
            self.app.save_settings()
            self.hide()
        elif response == RESPONSE_REVERT:
            self.app.load_settings()
            self.app.apply_settings()

    def update_ui(self):
        """Update the preferences window to reflect the current settings."""
        p = self.app.preferences
        self.cv.points = p['input.global_pressure_mapping']
        self.prefix_entry.set_text(p['saving.scrap_prefix'])
        mode = device_modes.index(p['input.device_mode'])
        self.input_devices_combo.set_active(mode)
        zoom = self.app.doc.zoomlevel_values[self.app.doc.zoomlevel]
        zoomlevel = min(bisect_left(self.defaultzoom_values, zoom),
                        len(self.defaultzoom_values) - 1)
        self.defaultzoom_combo.set_active(zoomlevel)
        saveformat_config = p['saving.default_format']
        saveformat_idx = self.app.filehandler.config2saveformat[saveformat_config]
        idx = self.defaultsaveformat_values.index(saveformat_idx)
        # FIXME: ^^^^^^^^^ try/catch/default may be more tolerant & futureproof
        self.defaultsaveformat_combo.set_active(idx)
        self.enable_history_popup_checkbox.set_active(p['input.enable_history_popup'])

        self.cv.queue_draw()

    # Callbacks for widgets that manipulate settings
    def input_devices_combo_changed_cb(self, window):
        mode = self.input_devices_combo.get_active_text()
        self.app.preferences['input.device_mode'] = mode
        self.app.apply_settings()

    def enable_history_popup_toggled_cb(self, widget):
        self.app.preferences['input.enable_history_popup'] = widget.get_active()
        self.app.apply_settings()

    def pressure_curve_changed_cb(self, widget):
        self.app.preferences['input.global_pressure_mapping'] = self.cv.points[:]
        self.app.apply_settings()

    def prefix_entry_changed_cb(self, widget):
        self.app.preferences['saving.scrap_prefix'] = widget.get_text()

    def defaultzoom_combo_changed_cb(self, widget):
        zoomlevel = self.defaultzoom_combo.get_active()
        zoom = self.defaultzoom_values[zoomlevel]
        self.app.preferences['view.default_zoom'] = zoom

    def defaultsaveformat_combo_changed_cb(self, widget):
        idx = self.defaultsaveformat_combo.get_active()
        saveformat = self.defaultsaveformat_values[idx]
        # Reverse lookup
        for key, val in self.app.filehandler.config2saveformat.iteritems():
            if val == saveformat:
                formatstr = key
        self.app.preferences['saving.default_format'] = formatstr

