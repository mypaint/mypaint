# This file is part of MyPaint.
# Copyright (C) 2008-2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Preferences dialog.
"""

import os.path
from logging import getLogger
logger = getLogger(__name__)

from gettext import gettext as _
import gtk
from gtk import gdk

import windowing
import canvasevent


RESPONSE_REVERT = 1


class PreferencesWindow (windowing.Dialog):
    """Window for manipulating preferences.
    """

    def __init__(self):
        import application
        app = application.get_app()
        assert app is not None
        flags = gtk.DIALOG_DESTROY_WITH_PARENT
        buttons = (gtk.STOCK_REVERT_TO_SAVED, RESPONSE_REVERT,
                   gtk.STOCK_OK, gtk.RESPONSE_ACCEPT)
        windowing.Dialog.__init__(self, app=app, title=_('Preferences'),
                                  parent=app.drawWindow, flags=flags,
                                  buttons=buttons)
        self.connect('response', self.on_response)

        self.in_update_ui = False

        # Set up widgets
        builder = gtk.Builder()
        xml_path = os.path.join(app.datapath, 'gui/preferenceswindow.glade')
        builder.add_from_file(xml_path)
        self._builder = builder

        # Notebook
        nb = builder.get_object("prefs_notebook")
        self.nb = nb
        self.vbox.pack_start(nb, expand=True, padding=0)

        # Curve init
        curve = builder.get_object("mapping_curve")
        curve.changed_cb = self.pressure_curve_changed_cb
        curve.magnetic = False
        self._pressure_curve = curve

        # Button mappings editor
        assert app.preferences.has_key("input.button_mapping")
        reg = canvasevent.ModeRegistry
        actions_possible = [n for n in reg.get_action_names()
                            if issubclass(reg.get_mode_class(n),
                                          canvasevent.SpringLoadedModeMixin) ]
        actions_possible += canvasevent.extra_actions
        bm_ed = builder.get_object("button_mapping_editor")
        bm_ed.set_bindings(app.preferences["input.button_mapping"])
        bm_ed.set_actions(actions_possible)
        bm_ed.bindings_observers.append(self.button_mapping_edited_cb)

        # Signal hookup now everything is in the right initial state
        self._builder.connect_signals(self)


    def on_response(self, dialog, response, *args):
        if response == gtk.RESPONSE_ACCEPT:
            self.app.save_settings()
            self.hide()
        elif response == RESPONSE_REVERT:
            self.app.load_settings()
            self.app.apply_settings()


    def update_ui(self):
        """Update the preferences window to reflect the current settings.
        """
        if self.in_update_ui:
            return
        self.in_update_ui = True

        p = self.app.preferences

        # Pen input curve
        self._pressure_curve.points = p['input.global_pressure_mapping']

        # prefix for saving scarps
        entry = self._builder.get_object("scrap_prefix_entry")
        entry.set_text(p['saving.scrap_prefix'])

        # Device mode
        mode_config = p["input.device_mode"]
        mode_combo = self._builder.get_object("input_mode_combobox")
        mode_combo.set_active_id(mode_config)

        # Zoom
        zoom_float = p.get('view.default_zoom', 1.0)
        zoom_idcolstr = "%0.2f" % (zoom_float,)
        zoom_combo = self._builder.get_object("default_zoom_combobox")
        zoom_combo.set_active_id(zoom_idcolstr)

        # Toolbar icon size radios
        size = str(p.get("ui.toolbar_icon_size", "large")).lower()
        for size_name in ["small", "large"]:
            if size_name != size:
                continue
            radio_name = "toolbar_icon_size_%s_radio" % (size,)
            radio = self._builder.get_object(radio_name)
            radio.set_active(True)
            logger.debug("Set %r active", radio_name)
            break

        # Dark theme
        dark = bool(p.get("ui.dark_theme_variant", True))
        dark_checkbutton = self._builder.get_object("dark_theme_checkbutton")
        dark_checkbutton.set_active(dark)

        # High-quality zoom
        hq_zoom_checkbutton = self._builder.get_object("hq_zoom_checkbutton")
        hq_zoom_checkbutton.set_active(p['view.high_quality_zoom'])

        # Default save format
        fmt_config = p['saving.default_format']
        fmt_combo = self._builder.get_object("default_save_format_combobox")
        fmt_combo.set_active_id(fmt_config)

        # Button mapping
        bm_ed = self._builder.get_object("button_mapping_editor")
        bm_ed.set_bindings(p.get("input.button_mapping", {}))

        # Input curve
        self._pressure_curve.queue_draw()

        # Cursor presets
        cursor_config = p.get("cursor.freehand.style", "thin")
        cursor_combo = self._builder.get_object("freehand_cursor_combobox")
        cursor_combo.set_active_id(cursor_config)

        # Colour wheel type
        wheel_config = self.app.brush_color_manager.get_wheel_type()
        wheel_radiobutton_name = "color_wheel_%s_radiobutton"
        wheel_radiobutton = self._builder.get_object(wheel_radiobutton_name)
        if wheel_radiobutton:
            wheel_radiobutton.set_active(True)

        self.in_update_ui = False


    # Callbacks for widgets that manipulate settings

    def input_mode_combobox_changed_cb(self, combobox):
        mode = combobox.get_active_id()
        self.app.preferences['input.device_mode'] = mode
        self.app.apply_settings()


    def button_mapping_edited_cb(self, editor):
        self.app.button_mapping.update(editor.bindings)


    def pressure_curve_changed_cb(self, widget):
        points = self._pressure_curve.points[:]
        self.app.preferences['input.global_pressure_mapping'] = points
        self.app.apply_settings()


    def scrap_prefix_entry_changed_cb(self, widget):
        self.app.preferences['saving.scrap_prefix'] = widget.get_text()


    def default_zoom_combobox_changed_cb(self, combobox):
        zoom_idcolstr = combobox.get_active_id()
        zoom = float(zoom_idcolstr)
        self.app.preferences['view.default_zoom'] = zoom


    def toolbar_icon_size_small_toggled_cb(self, radio):
        if not radio.get_active():
            return
        self.app.preferences["ui.toolbar_icon_size"] = "small"


    def toolbar_icon_size_large_toggled_cb(self, radio):
        if not radio.get_active():
            return
        self.app.preferences["ui.toolbar_icon_size"] = "large"


    def dark_theme_toggled_cb(self, checkbut):
        dark = bool(checkbut.get_active())
        self.app.preferences["ui.dark_theme_variant"] = dark


    def hq_zoom_checkbutton_toggled_cb(self, button):
        hq_zoom = bool(button.get_active())
        self.app.preferences['view.high_quality_zoom'] = hq_zoom


    def default_save_format_combobox_changed_cb(self, combobox):
        formatstr = combobox.get_active_id()
        self.app.preferences['saving.default_format'] = formatstr


    def color_wheel_rgb_radiobutton_toggled_cb(self, radiobtn):
        if self.in_update_ui or not radiobtn.get_active():
            return
        cm = self.app.brush_color_manager
        cm.set_wheel_type("rgb")


    def color_wheel_ryb_radiobutton_toggled_cb(self, radiobtn):
        if self.in_update_ui or not radiobtn.get_active():
            return
        cm = self.app.brush_color_manager
        cm.set_wheel_type("ryb")


    def color_wheel_rygb_radiobutton_toggled_cb(self, radiobtn):
        if self.in_update_ui or not radiobtn.get_active():
            return
        cm = self.app.brush_color_manager
        cm.set_wheel_type("rygb")


    def freehand_cursor_combobox_changed_cb(self, combobox):
        cname = combobox.get_active_id()
        if self.in_update_ui:
            return
        p = self.app.preferences
        p["cursor.freehand.style"] = cname
        if cname == 'thin':
            # The default.
            p.pop("cursor.freehand.min_size", None)
            p.pop("cursor.freehand.outer_line_width", None)
            p.pop("cursor.freehand.inner_line_width", None)
            p.pop("cursor.freehand.inner_line_inset", None)
            p.pop("cursor.freehand.outer_line_color", None)
            p.pop("cursor.freehand.inner_line_color", None)
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


