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
        self.connect('response', self.on_response)

        # Set up widgets
        nb = gtk.Notebook()
        nb.set_border_width(12)
        self.vbox.pack_start(nb, expand=True, padding=0)

        ### Input tab
        table = gtk.Table(5, 3)
        table.set_border_width(12)
        table.set_col_spacing(0, 12)
        table.set_col_spacing(1, 12)
        table.set_row_spacings(6)
        current_row = 0
        # TRANSLATORS: Tab label
        nb.append_page(table, gtk.Label(_('Input')))
        xopt = gtk.FILL | gtk.EXPAND
        yopt = gtk.FILL

        l = gtk.Label()
        l.set_alignment(0.0, 0.5)
        l.set_markup(_('<b>Input Device</b>'))
        table.attach(l, 0, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        l = gtk.Label()
        l.set_alignment(0.0, 0.5)
        l.set_line_wrap(True)
        l.set_markup(_('Scale input pressure to brush pressure. This is applied to all input devices. The mouse button have an input pressure of 0.5 when pressed.'))
        table.attach(l, 1, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        t = gtk.Table(4, 4)
        self.cv = CurveWidget(self.pressure_curve_changed_cb, magnetic=False)
        t.attach(self.cv, 0, 3, 0, 3, gtk.EXPAND | gtk.FILL, gtk.EXPAND | gtk.FILL, 5, 0)
        l1 = gtk.Label('1.0')
        if l1.set_angle:
            # TRANSLATORS: Graph y-axis label
            l2 = gtk.Label(_('Brush Pressure'))
            l2.set_angle(90)
        else:
            l2 = gtk.Label('')
        l3 = gtk.Label('0.0')
        t.attach(l1, 3, 4, 0, 1, 0, 0, 5, 0)
        t.attach(l2, 3, 4, 1, 2, 0, gtk.EXPAND, 5, 0)
        t.attach(l3, 3, 4, 2, 3, 0, 0, 5, 0)
        l4 = gtk.Label('0.0')
        # TRANSLATORS: Graph x-axis label
        l5 = gtk.Label(_('Input Pressure'))
        l5.set_justify(gtk.JUSTIFY_CENTER)
        l6 = gtk.Label('1.0')
        t.attach(l4, 0, 1, 3, 4, 0, 0, 5, 0)
        t.attach(l5, 1, 2, 3, 4, gtk.EXPAND, 0, 5, 0)
        t.attach(l6, 2, 3, 3, 4, 0, 0, 5, 0)
        table.attach(t, 1, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        l = gtk.Label(_('Mode: '))
        l.set_alignment(0.0, 0.5)
        table.attach(l, 1, 2, current_row, current_row + 1, xopt, yopt)
        combo = self.input_devices_combo = gtk.combo_box_new_text()
        for s in device_modes:
            combo.append_text(s)
        combo.connect('changed', self.input_devices_combo_changed_cb)
        table.attach(combo, 2, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        self.enable_history_popup_checkbox = c = gtk.CheckButton(_('Show color history on right-click'))
        c.connect('toggled', self.enable_history_popup_toggled_cb)
        table.attach(c, 1, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        ### Saving tab
        table = gtk.Table(5, 3)
        table.set_border_width(12)
        table.set_col_spacing(0, 12)
        table.set_col_spacing(1, 12)
        table.set_row_spacings(6)
        current_row = 0
        nb.append_page(table, gtk.Label(_('Saving')))
        xopt = gtk.FILL | gtk.EXPAND
        yopt = gtk.FILL

        l = gtk.Label()
        l.set_alignment(0.0, 0.5)
        l.set_markup(_('<b>Saving</b>'))
        table.attach(l, 0, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        l = gtk.Label(_('Default file format:'))
        l.set_alignment(0.0, 0.5)
        combo = self.defaultsaveformat_combo = gtk.combo_box_new_text()
        self.defaultsaveformat_values = [filehandling.SAVE_FORMAT_ORA, 
            filehandling.SAVE_FORMAT_PNGSOLID, filehandling.SAVE_FORMAT_JPEG]
        for saveformat in self.defaultsaveformat_values:
            format_desc = self.app.filehandler.saveformats[saveformat][0]
            combo.append_text(format_desc)
        combo.connect('changed', self.defaultsaveformat_combo_changed_cb)
        table.attach(l, 1, 2, current_row, current_row + 1, xopt, yopt)
        table.attach(combo, 2, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        l = gtk.Label()
        l.set_alignment(0.0, 0.5)
        l.set_markup(_('<b>Save Next Scrap</b>'))
        table.attach(l, 0, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        l = gtk.Label(_('Path and filename prefix:'))
        l.set_alignment(0.0, 0.5)
        self.prefix_entry = gtk.Entry()
        self.prefix_entry.connect('changed', self.prefix_entry_changed_cb)
        table.attach(l, 1, 2, current_row, current_row + 1, xopt, yopt)
        table.attach(self.prefix_entry, 2, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        ### View tab
        table = gtk.Table(2, 3)
        table.set_border_width(12)
        table.set_col_spacing(0, 12)
        table.set_col_spacing(1, 12)
        table.set_row_spacings(6)
        current_row = 0
        nb.append_page(table, gtk.Label(_('View')))
        xopt = gtk.FILL | gtk.EXPAND
        yopt = gtk.FILL

        l = gtk.Label()
        l.set_alignment(0.0, 0.5)
        l.set_markup(_('<b>Default View</b>'))
        table.attach(l, 0, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        l = gtk.Label(_('Default zoom:'))
        l.set_alignment(0.0, 0.5)
        combo = self.defaultzoom_combo = gtk.combo_box_new_text()
        # Different from doc.zoomlevel_values because we only want a subset
        self.defaultzoom_values = [0.25, 0.50, 1.0, 2.0]
        for val in self.defaultzoom_values:
            combo.append_text('%d%%' % (val*100))
        combo.connect('changed', self.defaultzoom_combo_changed_cb)
        table.attach(l, 1, 2, current_row, current_row + 1, xopt, yopt)
        table.attach(combo, 2, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

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
        zoomlevel = self.defaultzoom_values.index(zoom)
        self.defaultzoom_combo.set_active(zoomlevel)
        saveformat_config = p['saving.default_format']
        saveformat_idx = self.app.filehandler.config2saveformat[saveformat_config]
        idx = self.defaultsaveformat_values.index(saveformat_idx)
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

