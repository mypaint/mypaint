# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2012-2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


## Imports
from __future__ import division, print_function

import cairo
import math

import gui.overlays
import gui.mode
import gui.cursor
import gui.style
import gui.widgets
import gui.windowing
import gui.tileddrawwidget
import lib.alg
from lib.helpers import clamp
import lib.mypaintlib
from lib.mypaintlib import (
    SymmetryHorizontal, SymmetryVertical, SymmetryVertHorz,
    SymmetryRotational, SymmetrySnowflake
)
import lib.tiledsurface
import gui.drawutils
from lib.gettext import C_

from lib.gibindings import Gdk
from lib.gibindings import Gtk


## Module settings

_DEFAULT_ALPHA = 0.333
_ALPHA_PREFS_KEY = 'symmetry.line_alpha'


## Class defs

class _EditZone:

    NONE = 0
    CREATE_AXIS = 1
    MOVE_AXIS = 2
    MOVE_CENTER = 3
    DISABLE = 4


class SymmetryEditMode (gui.mode.ScrollableModeMixin, gui.mode.DragMode):
    """Tool/mode for editing the axis of symmetry used when painting"""

    ## Class-level config

    ACTION_NAME = 'SymmetryEditMode'

    pointer_behavior = gui.mode.Behavior.EDIT_OBJECTS
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    # These will be overridden on enter()
    inactive_cursor = None
    active_cursor = None

    unmodified_persist = True
    permitted_switch_actions = {
        'ShowPopupMenu', 'RotateViewMode', 'ZoomViewMode', 'PanViewMode',
    }

    _GRAB_SENSITIVITY = 8  # pixels

    DISABLE_BUTTON_RADIUS = gui.style.FLOATING_BUTTON_RADIUS
    CENTER_BUTTON_RADIUS = gui.style.FLOATING_BUTTON_RADIUS / 2

    # Statusbar stuff
    _STATUSBAR_CONTEXT = 'symmetry-mode'
    _STATUSBAR_CREATE_AXIS_MSG = C_(
        "symmetry axis edit mode: instructions shown in statusbar",
        u"Place axis",
    )
    _STATUSBAR_MOVE_AXIS_MSG = C_(
        "symmetry axis edit mode: instructions shown in statusbar",
        u"Move axis",
    )
    _STATUSBAR_DELETE_AXIS_MSG = C_(
        "symmetry axis edit mode: instructions shown in statusbar",
        u"Remove axis",
    )

    # Options widget singleton
    _OPTIONS_WIDGET = None

    ## Info strings

    @classmethod
    def get_name(cls):
        return C_(
            "symmetry axis edit mode: mode name (tooltips)",
            u"Edit Symmetry Axis",
        )

    def get_usage(self):
        return C_(
            "symmetry axis edit mode: mode description (tooltips)",
            u"Adjust the painting symmetry axis.",
        )

    ## Initization and mode interface

    def __init__(self, **kwds):
        """Initialize."""
        super(SymmetryEditMode, self).__init__(**kwds)
        from gui.application import get_app
        app = get_app()
        self.app = app
        self.zone = _EditZone.NONE
        self.active_axis = 0
        self.button_pos = None
        self.center_pos = None
        # Whether to render the actual alpha value, or a clamped value.
        # The clamped value should be used when moving the cursor across
        # the canvas when editing, whereas the real alpha should be used
        # when actively editing the alpha value.
        self.real_alpha = False

        statusbar_cid = app.statusbar.get_context_id(self._STATUSBAR_CONTEXT)
        self._statusbar_context_id = statusbar_cid
        self._drag_start_pos = None
        self._drag_axis_p2 = None
        self._last_msg_zone = None
        self._click_info = None
        self._entered_before = False

    def _get_cursor(self, name):
        return self.app.cursors.get_action_cursor(self.ACTION_NAME, name)

    def enter(self, doc, **kwds):
        """Enter the mode"""
        super(SymmetryEditMode, self).enter(doc, **kwds)

        # Initialize/fetch cursors
        self.cursor_remove = self._get_cursor(gui.cursor.Name.ARROW)
        self.cursor_add = self._get_cursor(gui.cursor.Name.ADD)
        self.cursor_normal = self._get_cursor(gui.cursor.Name.ARROW)
        self.cursor_movable = self._get_cursor(gui.cursor.Name.HAND_OPEN)
        self.cursor_moving = self._get_cursor(gui.cursor.Name.HAND_CLOSED)

        # Turn on the axis, if it happens to be off right now
        if not self._entered_before:
            action = self.app.find_action("SymmetryActive")
            action.set_active(True)
            self._entered_before = True

    def _update_statusbar(self):
        if self.in_drag:
            return
        if self._last_msg_zone == self.zone:
            return
        self._last_msg_zone = self.zone
        statusbar = self.app.statusbar
        statusbar_cid = self._statusbar_context_id
        statusbar.remove_all(statusbar_cid)
        msgs = {
            _EditZone.CREATE_AXIS: self._STATUSBAR_CREATE_AXIS_MSG,
            _EditZone.MOVE_AXIS: self._STATUSBAR_MOVE_AXIS_MSG,
            _EditZone.MOVE_CENTER: self._STATUSBAR_MOVE_AXIS_MSG,
            _EditZone.DISABLE: self._STATUSBAR_DELETE_AXIS_MSG,
        }
        msg = msgs.get(self.zone, None)
        if msg:
            statusbar.push(statusbar_cid, msg)

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = SymmetryEditOptionsWidget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET

    ## Events and internals

    def button_press_cb(self, tdw, event):
        if self.zone in (_EditZone.CREATE_AXIS, _EditZone.DISABLE):
            button = event.button
            if button == 1 and event.type == Gdk.EventType.BUTTON_PRESS:
                self._click_info = (button, self.zone)
                return False
        return super(SymmetryEditMode, self).button_press_cb(tdw, event)

    def button_release_cb(self, tdw, event):
        if self._click_info is not None:
            button0, zone0 = self._click_info
            if event.button == button0:
                if self.zone == zone0:
                    layer_stack = tdw.doc.layer_stack
                    if zone0 == _EditZone.DISABLE:
                        layer_stack.symmetry_active = False
                    elif zone0 == _EditZone.CREATE_AXIS:
                        x, y = tdw.display_to_model(event.x, event.y)
                        x, y = int(round(x)), int(round(y))
                        if layer_stack.symmetry_unset:
                            layer_stack.symmetry_unset = False
                        layer_stack.set_symmetry_state(True, center=(x, y))
                self._click_info = None
                self._update_zone_and_cursor(tdw, event.x, event.y)
                return False
        return super(SymmetryEditMode, self).button_release_cb(tdw, event)

    def _update_zone_and_cursor(self, tdw, x, y):
        """Update UI & some internal zone flags from pointer position

        :param tdw: canvas widget
        :param x: cursor x position
        :param y: cursor y position

        See also: `SymmetryOverlay`.

        """
        if self.in_drag:
            return
        new_zone = None
        layer_stack = tdw.doc.layer_stack
        if not layer_stack.symmetry_active:
            self.active_cursor = self.cursor_add
            self.inactive_cursor = self.cursor_add
            new_zone = _EditZone.CREATE_AXIS

        # Button hits - prioritize moving over disabling
        if new_zone is None and self.center_pos:
            cx, cy = self.center_pos
            if math.hypot(cx - x, cy - y) <= self.CENTER_BUTTON_RADIUS:
                self.active_cursor = self.cursor_moving
                self.inactive_cursor = self.cursor_movable
                new_zone = _EditZone.MOVE_CENTER

        if new_zone is None and self.button_pos:
            bx, by = self.button_pos
            if math.hypot(bx - x, by - y) <= self.DISABLE_BUTTON_RADIUS:
                self.active_cursor = self.cursor_remove
                self.inactive_cursor = self.cursor_remove
                new_zone = _EditZone.DISABLE

        axis_changed = False
        if new_zone is None:
            new_zone, axis_changed = self._update_axis_status(
                layer_stack, tdw, x, y)

        if new_zone is None:
            new_zone = _EditZone.NONE
            self.active_cursor = self.cursor_normal
            self.inactive_cursor = self.cursor_normal

        if new_zone != self.zone or axis_changed:
            self.zone = new_zone
            self._update_statusbar()
            tdw.queue_draw()

    def _update_axis_status(self, stack, tdw, x, y):
        """Check and record if cursor is within grabbing distance of an axis

        :param lib.layer.tree.RootLayerStack stack:
        :param gui.tileddrawwidget.TiledDrawWidget tdw:
        :return: (zone, active axis changed), or (None, None) if the cursor was
        not within grabbing distaance of any visible axis line.
        """
        # TODO: Change to NOT recalculate intersections every time
        corners = tdw.get_corners_model_coords()
        xm, ym = stack.symmetry_center
        intersections = get_viewport_intersections(
            stack.symmetry_type, xm, ym,
            stack.symmetry_angle, stack.symmetry_lines, corners)
        for i, p1, p2 in intersections:
            cursor_name = tdw.get_move_cursor_name_for_edge(
                (x, y), p1, p2, self._GRAB_SENSITIVITY)
            if cursor_name:
                a = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
                self._drag_axis_p2 = (
                    xm + math.cos(a),
                    ym + math.sin(a)
                )
                axis_changed = self.active_axis != i
                self.active_axis = i
                cursor = self._get_cursor(cursor_name)
                self.active_cursor = self.inactive_cursor = cursor
                return _EditZone.MOVE_AXIS, axis_changed

        return None, None

    def motion_notify_cb(self, tdw, event):
        if not self.in_drag:
            self.real_alpha = False
            self._update_zone_and_cursor(tdw, event.x, event.y)
            tdw.set_override_cursor(self.inactive_cursor)
        return super(SymmetryEditMode, self).motion_notify_cb(tdw, event)

    def drag_start_cb(self, tdw, event):
        if self.zone in {_EditZone.MOVE_AXIS, _EditZone.MOVE_CENTER}:
            self._drag_start_pos = tdw.doc.layer_stack.symmetry_center
        else:
            self._update_zone_and_cursor(tdw, self.start_x, self.start_y)
        return super(SymmetryEditMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, dx, dy):
        xm, ym = tdw.display_to_model(event.x, event.y)
        stack = tdw.doc.layer_stack
        if self.zone == _EditZone.MOVE_CENTER:
            stack.symmetry_center = (xm, ym)
        elif self.zone == _EditZone.MOVE_AXIS:
            xs, ys = self._drag_start_pos
            xmc, ymc = lib.alg.nearest_point_on_line(
                (xs, ys), self._drag_axis_p2, (xm, ym))
            stack.symmetry_center = ((xs - (xmc - xm)), (ys - (ymc - ym)))
        return super(SymmetryEditMode, self).drag_update_cb(tdw, event, dx, dy)

    def drag_stop_cb(self, tdw):
        tdw.queue_draw()
        return super(SymmetryEditMode, self).drag_stop_cb(tdw)


def is_symmetry_edit_mode(mode):
    return isinstance(mode, SymmetryEditMode)


class SymmetryEditOptionsWidget (Gtk.Alignment):

    _POSITION_LABEL_X_TEXT = C_(
        "symmetry axis options panel: labels",
        u"X Position:",
    )
    _POSITION_LABEL_Y_TEXT = C_(
        "symmetry axis options panel: labels",
        u"Y Position:",
    )
    _ANGLE_LABEL_TEXT = C_(
        "symmetry axis options panel: labels",
        u"Angle: %.2f°",
    )
    _POSITION_BUTTON_TEXT_INACTIVE = C_(
        "symmetry axis options panel: position button: no axis pos.",
        u"None",
    )
    _POSITION_BUTTON_TEXT_TEMPLATE = C_(
        "symmetry axis options panel: position button: axis pos. in pixels",
        u"%d px",
    )
    _ALPHA_LABEL_TEXT = C_(
        "symmetry axis options panel: labels",
        u"Alpha:",
    )
    _SYMMETRY_TYPE_TEXT = C_(
        "symmetry axis options panel: labels",
        u"Symmetry Type:",
    )
    _SYMMETRY_ROT_LINES_TEXT = C_(
        "symmetry axis options panel: labels",
        u"Rotational lines:",
    )

    def __init__(self):
        super(SymmetryEditOptionsWidget, self).__init__(
            xalign=0.5,
            yalign=0.5,
            xscale=1.0,
            yscale=1.0,
        )
        self._axis_pos_x_dialog = None
        self._axis_pos_x_button = None
        self._axis_pos_y_dialog = None
        self._axis_pos_y_button = None
        self._symmetry_type_combo = None
        self._axis_sym_lines_entry = None
        from gui.application import get_app
        self.app = get_app()
        rootstack = self.app.doc.model.layer_stack
        x, y = rootstack.symmetry_center
        self._axis_pos_adj_x = Gtk.Adjustment(
            value=x,
            upper=32000,
            lower=-32000,
            step_increment=1,
            page_increment=100,
        )
        self._xpos_cb_id = self._axis_pos_adj_x.connect(
            'value-changed',
            self._axis_pos_adj_x_changed,
        )
        self._axis_pos_adj_y = Gtk.Adjustment(
            value=y,
            upper=32000,
            lower=-32000,
            step_increment=1,
            page_increment=100,
        )
        self._ypos_cb_id = self._axis_pos_adj_y.connect(
            'value-changed',
            self._axis_pos_adj_y_changed,
        )
        self._axis_angle = Gtk.Adjustment(
            value=rootstack.symmetry_angle,
            upper=180,
            lower=0,
            step_increment=1,
            page_increment=15,
        )
        self._angle_cb_id = self._axis_angle.connect(
            "value-changed", self._angle_value_changed)
        self._axis_symmetry_lines = Gtk.Adjustment(
            value=rootstack.symmetry_lines,
            upper=50,
            lower=2,
            step_increment=1,
            page_increment=3,
        )
        self._lines_cb_id = self._axis_symmetry_lines.connect(
            'value-changed',
            self._axis_rot_symmetry_lines_changed,
        )

        self._init_ui()
        rootstack.symmetry_state_changed += self._symmetry_state_changed_cb
        self._update_button_labels(rootstack)

    def _init_ui(self):
        app = self.app

        # Dialog for showing and editing the axis value directly
        buttons = (Gtk.STOCK_OK, Gtk.ResponseType.ACCEPT)
        dialog = gui.windowing.Dialog(
            app, C_(
                "symmetry axis options panel: "
                "axis position dialog: window title",
                u"X axis Position",
            ),
            app.drawWindow,
            buttons=buttons,
        )
        dialog.connect('response', self._axis_pos_dialog_response_cb)
        grid = Gtk.Grid()
        grid.set_border_width(gui.widgets.SPACING_LOOSE)
        grid.set_column_spacing(gui.widgets.SPACING)
        grid.set_row_spacing(gui.widgets.SPACING)
        label = Gtk.Label(label=self._POSITION_LABEL_X_TEXT)
        label.set_hexpand(False)
        label.set_vexpand(False)
        grid.attach(label, 0, 0, 1, 1)
        entry = Gtk.SpinButton(
            adjustment=self._axis_pos_adj_x,
            climb_rate=0.25,
            digits=0
        )
        entry.set_hexpand(True)
        entry.set_vexpand(False)
        grid.attach(entry, 1, 0, 1, 1)
        dialog_content_box = dialog.get_content_area()
        dialog_content_box.pack_start(grid, True, True, 0)
        self._axis_pos_x_dialog = dialog

        # Dialog for showing and editing the axis value directly
        buttons = (Gtk.STOCK_OK, Gtk.ResponseType.ACCEPT)
        dialog = gui.windowing.Dialog(
            app, C_(
                "symmetry axis options panel: "
                "axis position dialog: window title",
                u"Y axis Position",
            ),
            app.drawWindow,
            buttons=buttons,
        )
        dialog.connect('response', self._axis_pos_dialog_response_cb)
        grid = Gtk.Grid()
        grid.set_border_width(gui.widgets.SPACING_LOOSE)
        grid.set_column_spacing(gui.widgets.SPACING)
        grid.set_row_spacing(gui.widgets.SPACING)
        label = Gtk.Label(label=self._POSITION_LABEL_Y_TEXT)
        label.set_hexpand(False)
        label.set_vexpand(False)
        grid.attach(label, 0, 0, 1, 1)
        entry = Gtk.SpinButton(
            adjustment=self._axis_pos_adj_y,
            climb_rate=0.25,
            digits=0
        )
        entry.set_hexpand(True)
        entry.set_vexpand(False)
        grid.attach(entry, 1, 0, 1, 1)
        dialog_content_box = dialog.get_content_area()
        dialog_content_box.pack_start(grid, True, True, 0)
        self._axis_pos_y_dialog = dialog

        # Layout grid
        row = 0
        grid = Gtk.Grid()
        grid.set_border_width(gui.widgets.SPACING_CRAMPED)
        grid.set_row_spacing(gui.widgets.SPACING_CRAMPED)
        grid.set_column_spacing(gui.widgets.SPACING_CRAMPED)
        self.add(grid)

        row += 1
        label = Gtk.Label(label=self._ALPHA_LABEL_TEXT)
        label.set_hexpand(False)
        label.set_halign(Gtk.Align.START)
        grid.attach(label, 0, row, 1, 1)
        scale = Gtk.Scale.new_with_range(
            orientation = Gtk.Orientation.HORIZONTAL,
            min = 0,
            max = 1,
            step = 0.1,
        )
        scale.set_draw_value(False)
        line_alpha = self.app.preferences.get(_ALPHA_PREFS_KEY, _DEFAULT_ALPHA)
        scale.set_value(line_alpha)
        scale.set_hexpand(True)
        scale.set_vexpand(False)
        scale.connect("value-changed", self._scale_value_changed_cb)
        grid.attach(scale, 1, row, 1, 1)

        row += 1
        store = Gtk.ListStore(int, str)
        rootstack = self.app.doc.model.layer_stack
        for _type in lib.tiledsurface.SYMMETRY_TYPES:
            store.append([_type, lib.tiledsurface.SYMMETRY_STRINGS[_type]])
        self._symmetry_type_combo = Gtk.ComboBox()
        self._symmetry_type_combo.set_model(store)
        self._symmetry_type_combo.set_active(rootstack.symmetry_type)
        self._symmetry_type_combo.set_hexpand(True)
        cell = Gtk.CellRendererText()
        self._symmetry_type_combo.pack_start(cell, True)
        self._symmetry_type_combo.add_attribute(cell, "text", 1)
        self._type_cb_id = self._symmetry_type_combo.connect(
            'changed',
            self._symmetry_type_combo_changed_cb
        )
        label = Gtk.Label(label=self._SYMMETRY_TYPE_TEXT)
        label.set_hexpand(False)
        label.set_halign(Gtk.Align.START)
        grid.attach(label, 0, row, 1, 1)
        grid.attach(self._symmetry_type_combo, 1, row, 1, 1)

        row += 1
        label = Gtk.Label(label=self._SYMMETRY_ROT_LINES_TEXT)
        label.set_hexpand(False)
        label.set_halign(Gtk.Align.START)
        self._axis_sym_lines_entry = Gtk.SpinButton(
            adjustment=self._axis_symmetry_lines,
            climb_rate=0.25
        )
        self._update_num_lines_sensitivity(rootstack.symmetry_type)
        grid.attach(label, 0, row, 1, 1)
        grid.attach(self._axis_sym_lines_entry, 1, row, 1, 1)

        row += 1
        label = Gtk.Label(label=self._POSITION_LABEL_X_TEXT)
        label.set_hexpand(False)
        label.set_halign(Gtk.Align.START)
        button = Gtk.Button(label=self._POSITION_BUTTON_TEXT_INACTIVE)
        button.set_vexpand(False)
        button.connect("clicked", self._axis_pos_x_button_clicked_cb)
        button.set_hexpand(True)
        button.set_vexpand(False)
        grid.attach(label, 0, row, 1, 1)
        grid.attach(button, 1, row, 1, 1)
        self._axis_pos_x_button = button

        row += 1
        label = Gtk.Label(label=self._POSITION_LABEL_Y_TEXT)
        label.set_hexpand(False)
        label.set_halign(Gtk.Align.START)
        button = Gtk.Button(label=self._POSITION_BUTTON_TEXT_INACTIVE)
        button.set_vexpand(False)
        button.connect("clicked", self._axis_pos_y_button_clicked_cb)
        button.set_hexpand(True)
        button.set_vexpand(False)
        grid.attach(label, 0, row, 1, 1)
        grid.attach(button, 1, row, 1, 1)
        self._axis_pos_y_button = button

        row += 1
        label = Gtk.Label()
        label.set_hexpand(False)
        label.set_halign(Gtk.Align.START)
        self._angle_label = label
        self._update_angle_label()
        grid.attach(label, 0, row, 1, 1)
        scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=self._axis_angle)
        scale.set_draw_value(False)
        scale.set_hexpand(True)
        scale.set_vexpand(False)
        grid.attach(scale, 1, row, 1, 1)

        row += 1
        button = Gtk.CheckButton()
        toggle_action = self.app.find_action("SymmetryActive")
        button.set_related_action(toggle_action)
        button.set_label(C_(
            "symmetry axis options panel: axis active checkbox",
            u'Enabled',
        ))
        button.set_hexpand(True)
        button.set_vexpand(False)
        grid.attach(button, 1, row, 2, 1)
        self._axis_active_button = button

    def _update_angle_label(self):
        self._angle_label.set_text(
            self._ANGLE_LABEL_TEXT % self._axis_angle.get_value()
        )

    def _symmetry_state_changed_cb(
            self, stack, active, center, sym_type, sym_lines, sym_angle):

        if center:
            cx, cy = center
            with self._axis_pos_adj_x.handler_block(self._xpos_cb_id):
                self._axis_pos_adj_x.set_value(cx)
            with self._axis_pos_adj_y.handler_block(self._ypos_cb_id):
                self._axis_pos_adj_y.set_value(cy)
        if sym_type is not None:
            with self._symmetry_type_combo.handler_block(self._type_cb_id):
                self._symmetry_type_combo.set_active(sym_type)
            self._update_num_lines_sensitivity(sym_type)
        if sym_lines is not None:
            with self._axis_symmetry_lines.handler_block(self._lines_cb_id):
                self._axis_symmetry_lines.set_value(sym_lines)
        if sym_angle is not None:
            with self._axis_angle.handler_block(self._angle_cb_id):
                self._axis_angle.set_value(sym_angle)
            self._update_angle_label()
        if center or stack.symmetry_unset:
            self._update_button_labels(stack)

    def _update_num_lines_sensitivity(self, sym_type):
        self._axis_sym_lines_entry.set_sensitive(
            sym_type in {SymmetryRotational, SymmetrySnowflake}
        )

    def _update_button_labels(self, stack):
        if stack.symmetry_unset:
            x, y = None, None
        else:
            x, y = stack.symmetry_center
        self._update_axis_button_label(self._axis_pos_x_button, x)
        self._update_axis_button_label(self._axis_pos_y_button, y)

    def _update_axis_button_label(self, button, value):
        if value is None:
            button.set_label(self._POSITION_BUTTON_TEXT_INACTIVE)
        else:
            button.set_label(self._POSITION_BUTTON_TEXT_TEMPLATE % value)

    def _axis_pos_adj_x_changed(self, adj):
        self.app.doc.model.layer_stack.symmetry_x = int(adj.get_value())

    def _axis_rot_symmetry_lines_changed(self, adj):
        self.app.doc.model.layer_stack.symmetry_lines = int(adj.get_value())

    def _axis_pos_adj_y_changed(self, adj):
        self.app.doc.model.layer_stack.symmetry_y = int(adj.get_value())

    def _angle_value_changed(self, adj):
        self._update_angle_label()
        self.app.doc.model.layer_stack.symmetry_angle = adj.get_value()

    def _axis_pos_x_button_clicked_cb(self, button):
        self._axis_pos_x_dialog.show_all()

    def _axis_pos_y_button_clicked_cb(self, button):
        self._axis_pos_y_dialog.show_all()

    def _symmetry_type_combo_changed_cb(self, combo):
        sym_type = combo.get_model()[combo.get_active()][0]
        self.app.doc.model.layer_stack.symmetry_type = sym_type

    def _axis_pos_dialog_response_cb(self, dialog, response_id):
        if response_id == Gtk.ResponseType.ACCEPT:
            dialog.hide()

    def _scale_value_changed_cb(self, alpha_scale):
        self.app.preferences[_ALPHA_PREFS_KEY] = alpha_scale.get_value()
        edit_mode = self.app.doc.modes.top
        edit_mode.real_alpha = True
        for tdw in self._tdws_with_symmetry_overlays():
            tdw.queue_draw()

    @staticmethod
    def _tdws_with_symmetry_overlays():
        for tdw in gui.tileddrawwidget.TiledDrawWidget.get_visible_tdws():
            for ov in tdw.display_overlays:
                if isinstance(ov, SymmetryOverlay):
                    yield tdw


class SymmetryOverlay (gui.overlays.Overlay):
    """Symmetry overlay, operating in display coordinates"""

    _DASH_PATTERN = [10, 7]
    _DASH_OFFSET = 5
    _EDIT_MODE_MIN_ALPHA = 0.25

    def __init__(self, doc):
        gui.overlays.Overlay.__init__(self)
        self.doc = doc
        self.tdw = self.doc.tdw
        self.tdw.connect("enter-notify-event", self._enter_notify_cb)
        rootstack = doc.model.layer_stack
        rootstack.symmetry_state_changed += self._symmetry_state_changed_cb
        doc.modes.changed += self._active_mode_changed_cb
        self._trash_icon_pixbuf = None

    def _enter_notify_cb(self, tdw, event):
        edit_mode = self._get_edit_mode()
        if edit_mode and edit_mode.real_alpha:
            edit_mode.real_alpha = False
            self.tdw.queue_draw()

    @property
    def trash_icon_pixbuf(self):
        if not self._trash_icon_pixbuf:
            self._trash_icon_pixbuf = gui.drawutils.load_symbolic_icon(
                icon_name="mypaint-trash-symbolic",
                size=SymmetryEditMode.DISABLE_BUTTON_RADIUS,
                fg=(0, 0, 0, 1),
            )
        return self._trash_icon_pixbuf

    def _symmetry_state_changed_cb(self, *args, **kwargs):
        self.tdw.queue_draw()

    def _active_mode_changed_cb(self, mode_stack, old, new):
        if any(map(is_symmetry_edit_mode, (old, new))):
            self.tdw.queue_draw()

    def paint(self, cr):
        """Paint the overlay, in display coordinates"""

        # The symmetry axis is a line (x==self.axis) in model coordinates
        model = self.doc.model
        if not model.layer_stack.symmetry_active:
            return

        # allocation, in display coordinates
        alloc = self.tdw.get_allocation()
        vx0, vy0 = alloc.x, alloc.y
        vx1, vy1 = vx0 + alloc.width, vy0 + alloc.height

        corners_m = self.tdw.get_corners_model_coords()
        mx, my = model.layer_stack.symmetry_center
        angle = model.layer_stack.symmetry_angle
        symmetry_type = model.layer_stack.symmetry_type
        num_lines = model.layer_stack.symmetry_lines

        intersections = get_viewport_intersections(
            symmetry_type, mx, my, angle, num_lines, corners_m)

        edit_mode = self._get_edit_mode()

        if intersections:
            self._render_axis_lines(cr, intersections, edit_mode)

        if not edit_mode:
            return

        self._draw_disable_button(
            edit_mode, mx, my, corners_m, cr, vx0, vx1, vy0, vy1)

        # Draw center button if it intersects the viewport
        r = SymmetryEditMode.CENTER_BUTTON_RADIUS
        dx, dy = self.tdw.model_to_display(mx, my)
        if (vx0 - r) < dx < (vx1 + r) and (vy0 - r) < dy < (vy1 + r):
            col = self._item_color(edit_mode.zone == _EditZone.MOVE_CENTER)
            gui.drawutils.render_round_floating_color_chip(cr, dx, dy, col, r)
            edit_mode.center_pos = (dx, dy)
        else:
            edit_mode.center_pos = None

    def _get_edit_mode(self):
        edit_mode = tuple(filter(is_symmetry_edit_mode, self.doc.modes))
        return edit_mode[0] if edit_mode else None

    def _draw_disable_button(
            self, edit_mode, x, y,
            corners_model, cr, vx0, vx1, vy0, vy1):
        # Positioning strategy: If the center is within the viewport,
        # the button is placed to the left of the center. Otherwise it is
        # placed in the location closest to the center. That way it can also
        # be used to locate the center visually.

        button_pos_m = lib.alg.nearest_point_in_poly(
            corners_model, (x, y))
        bpx, bpy = self.tdw.model_to_display(*button_pos_m)

        # Constrain position to viewport (display coordinates)
        margin = 2 * SymmetryEditMode.DISABLE_BUTTON_RADIUS
        bpx = clamp(bpx - margin, vx0 + margin, vx1 - margin)
        bpy = clamp(bpy, vy0 + margin, vy1 - margin)

        gui.drawutils.render_round_floating_button(
            cr, bpx, bpy,
            self._item_color(edit_mode.zone == _EditZone.DISABLE),
            pixbuf=self.trash_icon_pixbuf,
        )
        edit_mode.button_pos = (bpx, bpy)

    def _render_axis_lines(self, cr, intersections, mode):
        # Convert to display coordinates, with rounding and pixel centering
        def convert(n):
            x, y = self.tdw.model_to_display(*n)
            return math.floor(x), math.floor(y)
        points = [(i, convert(p1), convert(p2)) for i, p1, p2 in intersections]

        prefs = self.tdw.app.preferences
        line_alpha = float(prefs.get(_ALPHA_PREFS_KEY, _DEFAULT_ALPHA))
        if mode and mode.zone in {_EditZone.MOVE_AXIS, _EditZone.MOVE_CENTER}:
            line_alpha = 1.0
        elif mode and not mode.real_alpha:
            line_alpha = max(self._EDIT_MODE_MIN_ALPHA, line_alpha)

        if line_alpha <= 0:
            return

        # Paint the symmetry axis
        cr.save()
        cr.push_group()
        cr.set_line_cap(cairo.LINE_CAP_SQUARE)
        cr.set_dash(self._DASH_PATTERN, self._DASH_OFFSET)
        cr.set_line_width(gui.style.DRAGGABLE_EDGE_WIDTH)
        for i, (x0, y0), (x1, y1) in points:
            # Draw all axes as active if center is being moved, otherwise only
            # draw an axis as active if it is currently being moved.
            active = mode and (mode.zone == _EditZone.MOVE_CENTER
                               or mode.active_axis == i
                               and mode.zone == _EditZone.MOVE_AXIS)
            line_color = SymmetryOverlay._item_color(active)
            cr.set_source_rgb(*line_color.get_rgb())

            cr.move_to(x0, y0)
            cr.line_to(x1, y1)
            gui.drawutils.render_drop_shadow(cr, z=1)
            cr.stroke()
        cr.pop_group_to_source()
        cr.paint_with_alpha(line_alpha)
        cr.restore()

    @staticmethod
    def _item_color(active):
        if active:
            return gui.style.ACTIVE_ITEM_COLOR
        else:
            return gui.style.EDITABLE_ITEM_COLOR


def get_viewport_intersections(symm_type, x, y, angle, num_lines, corners_m):
    """Get indexed tuples with pairs of coordinates for each intersection

    The returned data is of the form [(index, (x0, y0), (x1, y1)), ...] where
    the index canonically identifies an axis, when there are multiple. Index
    values are partially ordered, but not always contiguous.

    If there are no intersections, the empty list is returned.
    """
    intersections = []
    p1 = (x, y)
    angle = math.pi * ((angle % 360) / 180)

    def append(a, **kwargs):
        # Reflected on y axis, due to display direction
        inter = lib.alg.intersection_of_vector_and_poly(
            corners_m, p1, (x + math.cos(a), y - math.sin(a)), **kwargs)
        intersections.append(inter)

    if symm_type in {SymmetryHorizontal, SymmetryVertHorz}:
        append(angle)
    if symm_type in {SymmetryVertical, SymmetryVertHorz}:
        append(angle + math.pi / 2)
    elif symm_type in {SymmetryRotational, SymmetrySnowflake}:
        delta = (math.pi * 2) / num_lines
        for i in range(num_lines):
            a = angle + delta * i
            append(a, line_type=lib.alg.LineType.DIRECTIONAL)
    return [(i, p[0], p[1]) for i, p in enumerate(intersections) if p]
