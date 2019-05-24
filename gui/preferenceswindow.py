# This file is part of MyPaint.
# Copyright (C) 2010-2018 by the MyPaint Development Team
# Copyright (C) 2008-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Preferences dialog."""

from __future__ import division, print_function

import os.path
from logging import getLogger
from gettext import gettext as _

from gi.repository import Gtk

from . import windowing


logger = getLogger(__name__)
RESPONSE_REVERT = 1


class PreferencesWindow (windowing.Dialog):
    """Window for manipulating preferences.
    """

    def __init__(self):
        import gui.application
        app = gui.application.get_app()
        assert app is not None

        super(PreferencesWindow, self).__init__(
            app=app,
            title=_('Preferences'),
            transient_for=app.drawWindow,
            destroy_with_parent=True,
        )
        self.add_buttons(
            Gtk.STOCK_REVERT_TO_SAVED, RESPONSE_REVERT,
            Gtk.STOCK_OK, Gtk.ResponseType.ACCEPT,
        )

        self.connect('response', self.on_response)

        self.in_update_ui = False

        # Set up widgets
        builder = Gtk.Builder()
        builder.set_translation_domain("mypaint")
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(ui_dir, 'preferenceswindow.glade')
        builder.add_from_file(xml_path)
        self._builder = builder

        getobj = builder.get_object

        # Notebook
        nb = getobj("prefs_notebook")
        self.nb = nb
        self.vbox.pack_start(nb, True, True, 0)

        # Curve init
        curve = getobj("mapping_curve")
        curve.changed_cb = self.pressure_curve_changed_cb
        curve.magnetic = False
        self._pressure_curve = curve

        # Button mappings editor
        assert "input.button_mapping" in app.preferences
        reg = gui.mode.ModeRegistry
        actions_possible = [n for n in reg.get_action_names()
                            if issubclass(reg.get_mode_class(n),
                                          gui.mode.DragMode)]
        actions_possible += gui.mode.BUTTON_BINDING_ACTIONS
        bm_ed = getobj("button_mapping_editor")
        bm_ed.set_bindings(app.preferences["input.button_mapping"])
        bm_ed.set_actions(actions_possible)
        bm_ed.bindings_observers.append(self.button_mapping_edited_cb)

        # Autosave controls
        autosave_interval_spinbut = getobj("autosave_interval_spinbutton")
        self._autosave_interval_spinbutton = autosave_interval_spinbut

        # Signal hookup now everything is in the right initial state
        self._builder.connect_signals(self)

    def on_response(self, dialog, response, *args):
        if response == Gtk.ResponseType.ACCEPT:
            self.app.save_settings()
            self.app.apply_settings()
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
        getobj = self._builder.get_object

        # Pen input curve
        self._pressure_curve.points = p['input.global_pressure_mapping']

        # prefix for saving scarps
        entry = getobj("scrap_prefix_entry")
        entry.set_text(p['saving.scrap_prefix'])

        # Zoom
        zoom_float = p.get('view.default_zoom', 1.0)
        zoom_idcolstr = "%0.2f" % (zoom_float,)
        zoom_combo = getobj("default_zoom_combobox")
        zoom_combo.set_active_id(zoom_idcolstr)

        # Toolbar icon size radios
        size = str(p.get("ui.toolbar_icon_size", "large")).lower()
        for size_name in ["small", "large"]:
            if size_name != size:
                continue
            radio_name = "toolbar_icon_size_%s_radio" % (size,)
            radio = getobj(radio_name)
            radio.set_active(True)
            logger.debug("Set %r active", radio_name)
            break

        # Dark theme
        dark = bool(p.get("ui.dark_theme_variant", True))
        dark_checkbutton = getobj("dark_theme_checkbutton")
        dark_checkbutton.set_active(dark)

        # Smooth scrolling
        smoothsc = bool(p.get("ui.support_smooth_scrolling", True))
        smoothsc_checkbutton = getobj("smooth_scrolling_checkbutton")
        smoothsc_checkbutton.set_active(smoothsc)

        # Blink layers on selection/creation
        blink_layers = bool(p.get("ui.blink_layers", True))
        blink_layers_checkbutton = getobj("blink_layers_checkbutton")
        blink_layers_checkbutton.set_active(blink_layers)

        # Use real or faked alpha checks (faked is faster...)
        real_alpha_checks_checkbutton = getobj("real_alpha_checks_checkbutton")
        real_alpha_checks_checkbutton.set_active(p['view.real_alpha_checks'])

        # Hide cursor when painting
        hidecsr = bool(p.get("ui.hide_cursor_while_painting", False))
        hidecsr_checkbut = getobj("hide_cursor_while_painting_checkbutton")
        hidecsr_checkbut.set_active(hidecsr)

        # Default save format
        fmt_config = p['saving.default_format']
        fmt_combo = getobj("default_save_format_combobox")
        fmt_combo.set_active_id(fmt_config)

        # Display colorspace setting
        # Only affects loading and saving PNGs and ORAs currently,
        # so it's located on the Load & Save tab for now.
        disp_colorspace_setting = p["display.colorspace"]
        disp_colorspace_radiobtn = getobj(
            "display_colorspace_%s_radiobutton" % (disp_colorspace_setting,)
        )
        if disp_colorspace_radiobtn:
            disp_colorspace_radiobtn.set_active(True)

        # Button mapping
        bm_ed = getobj("button_mapping_editor")
        bm_ed.set_bindings(p.get("input.button_mapping", {}))

        # Input curve
        self._pressure_curve.queue_draw()

        # Cursor presets
        cursor_config = p.get("cursor.freehand.style", "thin")
        cursor_combo = getobj("freehand_cursor_combobox")
        cursor_combo.set_active_id(cursor_config)

        # Color wheel type
        cm = self.app.brush_color_manager

        wheel_radiobutton_name = "color_wheel_%s_radiobutton" \
            % (cm.get_wheel_type(),)
        wheel_radiobutton = getobj(wheel_radiobutton_name)
        if wheel_radiobutton:
            wheel_radiobutton.set_active(True)

        # Autosave
        autosave = bool(p["document.autosave_backups"])
        autosave_switch = getobj("autosave_backups_switch")
        autosave_switch.set_active(autosave)
        autosave_interval = int(p["document.autosave_interval"])
        autosave_interval_adj = getobj("autosave_interval_adjustment")
        autosave_interval_adj.set_value(autosave_interval)
        self._autosave_interval_spinbutton.set_sensitive(autosave)

        self.in_update_ui = False

    ## Callbacks for widgets that manipulate settings

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
        scrap_prefix = widget.get_text()
        if isinstance(scrap_prefix, bytes):
            scrap_prefix = scrap_prefix.decode("utf-8")
        self.app.preferences['saving.scrap_prefix'] = scrap_prefix

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

    def real_alpha_checks_checkbutton_toggled_cb(self, button):
        real = bool(button.get_active())
        self.app.preferences['view.real_alpha_checks'] = real

    def default_save_format_combobox_changed_cb(self, combobox):
        formatstr = combobox.get_active_id()
        self.app.preferences['saving.default_format'] = formatstr

    def display_colorspace_unknown_radiobutton_toggled_cb(self, radiobtn):
        if self.in_update_ui or not radiobtn.get_active():
            return
        p = self.app.preferences
        p["display.colorspace"] = "unknown"

    def display_colorspace_srgb_radiobutton_toggled_cb(self, radiobtn):
        if self.in_update_ui or not radiobtn.get_active():
            return
        p = self.app.preferences
        p["display.colorspace"] = "srgb"

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

    def autosave_backups_switch_active_notify_cb(self, switch, param):
        active = bool(switch.props.active)
        self.app.preferences["document.autosave_backups"] = active
        self._autosave_interval_spinbutton.set_sensitive(active)

    def autosave_interval_adjustment_value_changed_cb(self, adj):
        interval = int(round(adj.get_value()))
        self.app.preferences["document.autosave_interval"] = interval

    def smooth_scrolling_toggled_cb(self, checkbut):
        smoothsc = bool(checkbut.get_active())
        self.app.preferences["ui.support_smooth_scrolling"] = smoothsc

    def _hide_cursor_while_painting_toggled_cb(self, checkbut):
        hide = bool(checkbut.get_active())
        self.app.preferences["ui.hide_cursor_while_painting"] = hide

    def blink_layers_toggled_cb(self, checkbut):
        blink = bool(checkbut.get_active())
        self.app.preferences["ui.blink_layers"] = blink
