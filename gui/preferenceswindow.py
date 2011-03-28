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

device_modes = [
    ('disabled', _("Disabled (no pressure sensitivity)")),
    ('screen', _("Screen (normal)")),
    ('window', _("Window (not recommended)")),  ]

RESPONSE_REVERT = 1

# Rebindable mouse buttons
mouse_button_actions = [
    # These can be names of actions within ActionGroups defined elsewhere,
    # or names of actions the handler interprets itself.
    # NOTE: The translatable strings for actions are duplicated from
    # their action definition. Please keep in sync (or refactor to get the string from there)
    # (action_or_whatever, label)
    ('no_action', _("No action")),  #[0] is the default for the comboboxes
    ('popup_menu', _("Menu")),
    ('ToggleSubwindows', _("Toggle Subwindows")),
    ('ColorPickerPopup', _("Pick Color")),
    ('PickContext', _('Pick Context (layer, brush and color)')),
    ('PickLayer', _('Select Layer at Cursor')),
    ('pan_canvas', _("Pan")),
    ('zoom_canvas', _("Zoom")),
    ('rotate_canvas', _("Rotate")),
    ('straight_line', _("Straight Line")),
    ('straight_line_sequence', _("Sequence of Straight Lines")),
    ('ColorChangerPopup', _("Color Changer")),
    ('ColorRingPopup', _("Color Ring")),
    ('ColorHistoryPopup', _("Color History")),
    ]
mouse_button_prefs = [
    # Used for creating the menus,
    # (pref_name, label)
    ("input.button1_shift_action", _("Button 1 + Shift")),
    ("input.button1_ctrl_action",  _("Button 1 + Ctrl (or Alt)")),
    ("input.button2_action",       _("Button 2")),
    ("input.button2_shift_action", _("Button 2 + Shift")),
    ("input.button2_ctrl_action",  _("Button 2 + Ctrl (or Alt)")),
    ("input.button3_action",       _("Button 3")),
    ("input.button3_shift_action", _("Button 3 + Shift")),
    ("input.button3_ctrl_action",  _("Button 3 + Ctrl (or Alt)")),
    ]

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

        self.in_update_ui = False

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
        nb.append_page(table, gtk.Label(_('Pen Input')))
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
        l.set_markup(_('Scale input pressure to brush pressure. This is applied to all input devices. The mouse button has an input pressure of 0.5 when pressed.'))
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
        for m, s in device_modes:
            combo.append_text(s)
        combo.connect('changed', self.input_devices_combo_changed_cb)
        table.attach(combo, 2, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        ### Buttons tab
        table = gtk.Table(5, 3)
        table.set_border_width(12)
        table.set_col_spacing(0, 12)
        table.set_col_spacing(1, 12)
        table.set_row_spacings(6)
        current_row = 0
        nb.append_page(table, gtk.Label(_('Buttons')))
        xopt = gtk.FILL | gtk.EXPAND
        yopt = gtk.FILL

        l = gtk.Label()
        l.set_alignment(0.0, 0.5)
        l.set_markup(_('<b>Pen and mouse button mappings</b>'))
        table.attach(l, 0, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        # Mouse button actions
        self.mouse_action_comboboxes = {}
        for pref_name, label_str in mouse_button_prefs:
            l = gtk.Label(label_str)
            l.set_alignment(0.0, 0.5)
            table.attach(l, 1, 2, current_row, current_row + 1, xopt, yopt)
            action_name = self.app.preferences.get(pref_name, None)
            c = gtk.combo_box_new_text()
            self.mouse_action_comboboxes[pref_name] = c
            for a, s in mouse_button_actions:
                c.append_text(s)
            c.connect("changed", self.mouse_button_action_changed, pref_name)
            table.attach(c, 2, 3, current_row, current_row + 1, xopt, yopt)
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
        # - keep sorted for bisect
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
        if self.in_update_ui:
            return
        self.in_update_ui = True
        p = self.app.preferences
        self.cv.points = p['input.global_pressure_mapping']
        self.prefix_entry.set_text(p['saving.scrap_prefix'])
        # Device mode
        mode_config = p.get("input.device_mode", None)
        mode_idx = i = 0
        for mode_name, junk in device_modes:
            if mode_config == mode_name:
                mode_idx = i
                break
            i += 1
        self.input_devices_combo.set_active(mode_idx)
        zoom = p['view.default_zoom']
        zoomlevel = min(bisect_left(self.defaultzoom_values, zoom),
                        len(self.defaultzoom_values) - 1)
        self.defaultzoom_combo.set_active(zoomlevel)
        saveformat_config = p['saving.default_format']
        saveformat_idx = self.app.filehandler.config2saveformat[saveformat_config]
        idx = self.defaultsaveformat_values.index(saveformat_idx)
        # FIXME: ^^^^^^^^^ try/catch/default may be more tolerant & futureproof
        self.defaultsaveformat_combo.set_active(idx)
        # Mouse button
        for pref_name, junk in mouse_button_prefs:
            action_config = p.get(pref_name, None)
            action_idx = i = 0
            for action_name, junk in mouse_button_actions:
                if action_config == action_name:
                    action_idx = i
                    break
                i += 1
            combobox = self.mouse_action_comboboxes[pref_name]
            combobox.set_active(action_idx)
        self.cv.queue_draw()
        self.in_update_ui = False

    # Callbacks for widgets that manipulate settings
    def input_devices_combo_changed_cb(self, widget):
        i = widget.get_property("active")
        mode = device_modes[i][0]
        self.app.preferences['input.device_mode'] = mode
        self.app.apply_settings()

    def mouse_button_action_changed(self, widget, pref_name):
        i = widget.get_property("active")
        action = mouse_button_actions[i][0]
        self.app.preferences[pref_name] = action
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

