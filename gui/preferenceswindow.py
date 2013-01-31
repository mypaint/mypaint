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
import gtk
gdk = gtk.gdk

from curve import CurveWidget
import windowing, filehandling
import canvasevent
from buttonmap import button_press_parse, button_press_name, ButtonMappingEditor


device_modes = [
    ('disabled', _("Disabled (no pressure sensitivity)")),
    ('screen', _("Screen (normal)")),
    ('window', _("Window (not recommended)")),  ]

cursor_presets = [
    ('thin', _("Circle (light outline)")),
    ('medium', _("Circle (medium outline)")),
    ('thick', _("Circle (heavy outline)")),
    ('crosshair', _("Crosshair")),  ]

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

        ### Pointer actions tab
        assert app.preferences.has_key("input.button_mapping")
        vbox = gtk.VBox()

        actions_possible = canvasevent.ModeRegistry.get_action_names()
        actions_possible = [n for n in actions_possible
          if issubclass(canvasevent.ModeRegistry.get_mode_class(n),
                        canvasevent.SpringLoadedModeMixin) ]
        actions_possible += canvasevent.extra_actions
        self.button_map_editor = ButtonMappingEditor(app=app,
                bindings=app.preferences["input.button_mapping"],
                actions_possible=actions_possible)
        self.button_map_editor.bindings_observers.append(
                self.button_mapping_editor_bindings_edited_cb)
        vbox.set_border_width(12)
        vbox.set_spacing(12)
        vbox.pack_start(self.button_map_editor, True, True)
        button_map_label = gtk.Label()
        button_map_label.set_markup(
            _("<small>Space can be used like Button2. Note that some pads "
              "have buttons that cannot be held down.</small>"))
        vbox.pack_start(button_map_label, False, False)
        nb.append_page(vbox, gtk.Label(_("Buttons")))

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
        table = gtk.Table(2, 4)
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
        l.set_markup(_('<b>Zoom</b>'))
        table.attach(l, 0, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        l = gtk.Label(_('Default zoom level:'))
        l.set_alignment(0.0, 0.5)
        combo = self.defaultzoom_combo = gtk.combo_box_new_text()
        # Different from doc.zoomlevel_values because we only want a subset
        # - keep sorted for bisect
        self.defaultzoom_values = [0.25, 0.50, 1.0, 2.0]
        for val in self.defaultzoom_values:
            combo.append_text('%d%%' % (val*100))
        combo.connect('changed', self.defaultzoom_combo_changed_cb)
        table.attach(l, 1, 2, current_row, current_row + 1, gtk.FILL, yopt)
        table.attach(combo, 2, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        b = self.highqualityzoom_checkbox = gtk.CheckButton(_('High quality zoom (may result in slow scrolling)'))
        b.connect('toggled', self.highqualityzoom_checkbox_changed_cb)
        table.attach(b, 2, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        l = gtk.Label()
        l.set_alignment(0.0, 0.5)
        l.set_markup(_('<b>Fullscreen</b>'))
        table.attach(l, 0, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        b = self.fullscreenhidemenubar_checkbox = gtk.CheckButton(_('Hide menu bar'))
        b.connect('toggled', self.fullscreenhidemenubar_checkbox_changed_cb)
        table.attach(b, 1, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        b = self.fullscreenhidetoolbar_checkbox = gtk.CheckButton(_('Hide toolbar'))
        b.connect('toggled', self.fullscreenhidetoolbar_checkbox_changed_cb)
        table.attach(b, 1, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        b = self.fullscreenhidesubwindows_checkbox = gtk.CheckButton(_('Hide tools'))
        b.connect('toggled', self.fullscreenhidesubwindows_checkbox_changed_cb)
        table.attach(b, 1, 3, current_row, current_row + 1, xopt, yopt)
        current_row += 1

        #### Cursor tab
        #table = gtk.Table(2, 1)
        #table.set_border_width(12)
        ##table.set_col_spacing(0, 12)
        #table.set_row_spacings(6)
        #current_row = 0
        #nb.append_page(table, gtk.Label(_('Cursor')))
        #xopt = gtk.FILL | gtk.EXPAND
        #yopt = gtk.FILL

        l = gtk.Label()
        l.set_alignment(0.0, 0.5)
        l.set_markup(_('<b>Freehand cursor</b>'))
        table.attach(l, 0, 3, current_row, current_row + 1, xopt, yopt)
        table.set_row_spacing(current_row-1, 18)
        current_row += 1

        self.cursor_radio_buttons = {}
        b = None
        for cname, label_text in cursor_presets:
            b = gtk.RadioButton(group=b, label=label_text, use_underline=False)
            b.connect("toggled", self.cursor_radio_toggled_cb, cname)
            self.cursor_radio_buttons[cname] = b
            table.attach(b, 1, 3, current_row, current_row+1, xopt, yopt)
            current_row += 1


    def cursor_radio_toggled_cb(self, togglebutton, cname):
        if not togglebutton.get_active():
            return
        if self.in_update_ui:
            return
        p = self.app.preferences
        p["cursor.freehand.style"] = cname
        if cname == 'thin':
            # The default.
            del p["cursor.freehand.min_size"]
            del p["cursor.freehand.outer_line_width"]
            del p["cursor.freehand.inner_line_width"]
            del p["cursor.freehand.inner_line_inset"]
            del p["cursor.freehand.outer_line_color"]
            del p["cursor.freehand.inner_line_color"]
        elif cname == "medium":
            p["cursor.freehand.min_size"] = 5
            p["cursor.freehand.outer_line_width"] = 2.666
            p["cursor.freehand.inner_line_width"] = 1.333
            p["cursor.freehand.inner_line_inset"] = 2
            p["cursor.freehand.outer_line_color"] = (0, 0, 0, 1)
            p["cursor.freehand.inner_line_color"] = (1, 1, 1, 1)
        elif cname == "thick":
            p["cursor.freehand.min_size"] = 7
            p["cursor.freehand.outer_line_width"] = 3.75
            p["cursor.freehand.inner_line_width"] = 2.25
            p["cursor.freehand.inner_line_inset"] = 3
            p["cursor.freehand.outer_line_color"] = (0, 0, 0, 1)
            p["cursor.freehand.inner_line_color"] = (1, 1, 1, 1)


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
        self.fullscreenhidemenubar_checkbox.set_active(p['ui.hide_menubar_in_fullscreen'])
        self.fullscreenhidetoolbar_checkbox.set_active(p['ui.hide_toolbar_in_fullscreen'])
        self.fullscreenhidesubwindows_checkbox.set_active(p['ui.hide_subwindows_in_fullscreen'])
        self.highqualityzoom_checkbox.set_active(p['view.high_quality_zoom'])
        saveformat_config = p['saving.default_format']
        saveformat_idx = self.app.filehandler.config2saveformat[saveformat_config]
        idx = self.defaultsaveformat_values.index(saveformat_idx)
        # FIXME: ^^^^^^^^^ try/catch/default may be more tolerant & futureproof
        self.defaultsaveformat_combo.set_active(idx)
        # Button mapping
        self.button_map_editor.set_bindings(p["input.button_mapping"])
        # Input curve
        self.cv.queue_draw()
        # Cursor presets
        self.update_cursor_settings()
        self.in_update_ui = False


    def update_cursor_settings(self):
        if not self.in_update_ui:
            return
        p = self.app.preferences
        style = self.app.preferences.get("cursor.freehand.style", "thin")
        b = self.cursor_radio_buttons.get(style, None)
        if not b:
            return
        b.set_active(True)


    # Callbacks for widgets that manipulate settings

    def input_devices_combo_changed_cb(self, widget):
        i = widget.get_property("active")
        mode = device_modes[i][0]
        self.app.preferences['input.device_mode'] = mode
        self.app.apply_settings()

    def button_mapping_editor_bindings_edited_cb(self, editor):
        self.app.button_mapping.update(editor.bindings)

    def pressure_curve_changed_cb(self, widget):
        self.app.preferences['input.global_pressure_mapping'] = self.cv.points[:]
        self.app.apply_settings()

    def prefix_entry_changed_cb(self, widget):
        self.app.preferences['saving.scrap_prefix'] = widget.get_text()

    def defaultzoom_combo_changed_cb(self, widget):
        zoomlevel = self.defaultzoom_combo.get_active()
        zoom = self.defaultzoom_values[zoomlevel]
        self.app.preferences['view.default_zoom'] = zoom

    def fullscreenhidemenubar_checkbox_changed_cb(self, widget):
        self.app.preferences['ui.hide_menubar_in_fullscreen'] = bool(widget.get_active())

    def fullscreenhidetoolbar_checkbox_changed_cb(self, widget):
        self.app.preferences['ui.hide_toolbar_in_fullscreen'] = bool(widget.get_active())

    def fullscreenhidesubwindows_checkbox_changed_cb(self, widget):
        self.app.preferences['ui.hide_subwindows_in_fullscreen'] = bool(widget.get_active())

    def highqualityzoom_checkbox_changed_cb(self, widget):
        self.app.preferences['view.high_quality_zoom'] = bool(widget.get_active())
        self.app.doc.tdw.queue_draw()

    def defaultsaveformat_combo_changed_cb(self, widget):
        idx = self.defaultsaveformat_combo.get_active()
        saveformat = self.defaultsaveformat_values[idx]
        # Reverse lookup
        for key, val in self.app.filehandler.config2saveformat.iteritems():
            if val == saveformat:
                formatstr = key
        self.app.preferences['saving.default_format'] = formatstr


