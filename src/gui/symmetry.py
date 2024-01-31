# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2012-2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (C) 2017-2020 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


# Imports
from __future__ import division, print_function

import cairo
import math

import gui.overlays
import gui.mode
import gui.cursor
import gui.drawutils
import gui.style
import gui.widgets
import gui.windowing
import gui.tileddrawwidget
from gui.sliderwidget import InputSlider

import lib.alg
import lib.mypaintlib
import lib.tiledsurface
from lib.mypaintlib import (
    SymmetryHorizontal, SymmetryVertical, SymmetryVertHorz,
    SymmetryRotational, SymmetrySnowflake
)
from lib.helpers import Rect
from lib.gettext import C_

from lib.gibindings import Gdk
from lib.gibindings import GLib
from lib.gibindings import Gtk


# Module settings

_DEFAULT_ALPHA = 0.333
_ALPHA_PREFS_KEY = 'symmetry.line_alpha'


# Class defs

class _EditZone:

    NONE = 0
    CREATE_AXIS = 1
    MOVE_AXIS = 2
    MOVE_CENTER = 3
    DISABLE = 4


class SymmetryEditMode (gui.mode.ScrollableModeMixin, gui.mode.DragMode):
    """Tool/mode for editing the axis of symmetry used when painting"""

    # Class-level config

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

    # Info strings

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

    # Initization and mode interface

    def __init__(self, **kwds):
        """Initialize."""
        super(SymmetryEditMode, self).__init__(**kwds)
        from gui.application import get_app
        app = get_app()
        self.app = app

        # The overlay is always present and stores the information required to
        # draw the axes, as well as information about what the active zone is.
        self._overlay = [
            o for o in app.doc.tdw.display_overlays
            if isinstance(o, SymmetryOverlay)][0]

        statusbar_cid = app.statusbar.get_context_id(self._STATUSBAR_CONTEXT)
        self._statusbar_context_id = statusbar_cid
        self._last_msg_zone = None
        self._zone = None

        # Symmetry center location at the beginning of the drag
        self._drag_start_pos = None
        self._drag_prev_pos = None
        self._active_axis_points = None
        self._drag_factors = None
        self._click_info = None
        self._entered_before = False

        self._move_item = None
        self._move_timeout_id = None

        # Initialize/fetch cursors
        self.cursor_remove = self._get_cursor(gui.cursor.Name.ARROW)
        self.cursor_add = self._get_cursor(gui.cursor.Name.ADD)
        self.cursor_normal = self._get_cursor(gui.cursor.Name.ARROW)
        self.cursor_movable = self._get_cursor(gui.cursor.Name.HAND_OPEN)
        self.cursor_moving = self._get_cursor(gui.cursor.Name.HAND_CLOSED)

    def _get_cursor(self, name):
        return self.app.cursors.get_action_cursor(self.ACTION_NAME, name)

    def enter(self, doc, **kwds):
        """Enter the mode"""
        super(SymmetryEditMode, self).enter(doc, **kwds)
        # Set overlay to draw edit controls (center point, disable button)
        self._overlay.enable_edit_mode()
        # Turn on the axis, if it happens to be off right now
        if not self._entered_before:
            self.app.find_action("SymmetryActive").set_active(True)
            self._entered_before = True

    def popped(self):
        # Set overlay to draw normally, without controls
        self._overlay.disable_edit_mode()
        super(SymmetryEditMode, self).popped()

    def _update_statusbar(self, zone):
        if self.in_drag:
            return
        if self._last_msg_zone == zone:
            return
        self._last_msg_zone = zone
        statusbar = self.app.statusbar
        statusbar_cid = self._statusbar_context_id
        statusbar.remove_all(statusbar_cid)
        msgs = {
            _EditZone.CREATE_AXIS: self._STATUSBAR_CREATE_AXIS_MSG,
            _EditZone.MOVE_AXIS: self._STATUSBAR_MOVE_AXIS_MSG,
            _EditZone.MOVE_CENTER: self._STATUSBAR_MOVE_AXIS_MSG,
            _EditZone.DISABLE: self._STATUSBAR_DELETE_AXIS_MSG,
        }
        msg = msgs.get(zone, None)
        if msg:
            statusbar.push(statusbar_cid, msg)

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = SymmetryEditOptionsWidget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET

    # Events and internals

    def button_press_cb(self, tdw, event):
        if self._zone in (_EditZone.CREATE_AXIS, _EditZone.DISABLE):
            button = event.button
            if button == 1 and event.type == Gdk.EventType.BUTTON_PRESS:
                self._click_info = (button, self._zone)
                return False
        return super(SymmetryEditMode, self).button_press_cb(tdw, event)

    def button_release_cb(self, tdw, event):
        # If the corresponding press was not on a clickable zone, or the cursor
        # was moved away from it before the button was released.
        info = self._click_info
        if not info or (event.button, self._zone) != info:
            return super(SymmetryEditMode, self).button_release_cb(tdw, event)

        _, zone_pressed = info
        self._click_info = None
        layer_stack = tdw.doc.layer_stack
        # Disable button clicked
        if zone_pressed == _EditZone.DISABLE:
            layer_stack.symmetry_active = False
        # Symmetry was inactive - create axis based on cursor position
        elif zone_pressed == _EditZone.CREATE_AXIS:
            new_center = tuple(
                int(round(c)) for c in
                tdw.display_to_model(event.x, event.y))
            if layer_stack.symmetry_unset:
                layer_stack.symmetry_unset = False
            layer_stack.set_symmetry_state(True, center=new_center)
        self._update_zone_and_cursor(event.x, event.y)
        return False

    def _update_zone_and_cursor(self, x, y):
        """Update UI & some internal zone flags from pointer position

        :param x: cursor x position
        :param y: cursor y position

        See also: `SymmetryOverlay`.

        """
        if self.in_drag:
            return

        zone_cursors = {  # Active and inactive cursors respectively
            _EditZone.CREATE_AXIS: (self.cursor_add, self.cursor_add),
            _EditZone.MOVE_CENTER: (self.cursor_moving, self.cursor_movable),
            _EditZone.DISABLE: (self.cursor_remove, self.cursor_remove),
            _EditZone.NONE: (self.cursor_normal, self.cursor_normal),
        }

        changed, zone, data = self._overlay.update_zone_data(x, y)
        self._zone = zone

        if changed:
            self._update_statusbar(zone)
            if zone in zone_cursors:
                self.active_cursor, self.inactive_cursor = zone_cursors[zone]
            elif data:
                active, inactive, axis_points = data
                self._active_axis_points = axis_points
                self.active_cursor = self._get_cursor(active)
                self.inactive_cursor = self._get_cursor(inactive)

    def motion_notify_cb(self, tdw, event):
        if not self.in_drag:
            self._update_zone_and_cursor(event.x, event.y)
            tdw.set_override_cursor(self.inactive_cursor)
        return super(SymmetryEditMode, self).motion_notify_cb(tdw, event)

    def drag_start_cb(self, tdw, event):
        tdw.renderer.defer_hq_rendering(10)
        if self._zone in (_EditZone.MOVE_AXIS, _EditZone.MOVE_CENTER):
            self._drag_start_pos = tdw.doc.layer_stack.symmetry_center
            if self._zone == _EditZone.MOVE_AXIS:
                p1, p2 = self._active_axis_points
                # Calculate how the pixel offsets (display coordinates) relate
                # to the symmetry center offsets (model coordinates).
                # Sloppy way to do it, but it isn't done often.
                offs = 1000.0
                dx, dy = tdw.model_to_display(*p1)
                # Horizontal and vertical display offsets -> model coordinates
                xrefx, xrefy = tdw.display_to_model(dx - offs, dy)
                yrefx, yrefy = tdw.display_to_model(dx, dy + offs)
                # Display x offsets -> model x,y offsets
                xc, yc = lib.alg.nearest_point_on_line(p1, p2, (xrefx, xrefy))
                xx, xy = (xc - xrefx) / offs, (yc - xrefy) / offs
                # Display y offsets -> model x,y offsets
                xc, yc = lib.alg.nearest_point_on_line(p1, p2, (yrefx, yrefy))
                yx, yy = (yrefx - xc) / offs, (yrefy - yc) / offs
                self._drag_factors = xx, xy, yx, yy
        else:
            self._update_zone_and_cursor(self.start_x, self.start_y)
        return super(SymmetryEditMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, ev_x, ev_y, dx, dy):
        zone = self._zone
        if zone == _EditZone.MOVE_CENTER:
            self._queue_movement(zone, (ev_x, ev_y, tdw))
        elif zone == _EditZone.MOVE_AXIS:
            self._queue_movement(
                zone, (ev_x - self.start_x, ev_y - self.start_y, tdw))

    def _queue_movement(self, zone, args):
        self._move_item = (zone, args)
        if not self._move_timeout_id:
            self._move_timeout_id = GLib.timeout_add(
                interval=16.66,  # 60 fps cap
                function=self._do_move,
            )

    def _do_move(self):
        if self._move_item:
            zone, args = self._move_item
            self._move_item = None
            if zone == _EditZone.MOVE_AXIS:
                dx, dy, tdw = args
                self._move_axis(dx, dy, tdw.doc.layer_stack)
            elif zone == _EditZone.MOVE_CENTER:
                x, y, tdw = args
                self._move_center(x, y, tdw)
        self._move_timeout_id = None

    def _move_center(self, x, y, tdw):
        xm, ym = tdw.display_to_model(x, y)
        tdw.doc.layer_stack.symmetry_center = (xm, ym)

    def _move_axis(self, dx_full, dy_full, stack):
        xs, ys = self._drag_start_pos
        xx, xy, yx, yy = self._drag_factors
        xm = round(xs + (dx_full * xx + dy_full * yx))
        ym = round(ys + (dx_full * xy + dy_full * yy))
        new_pos = xm, ym
        if self._drag_prev_pos != new_pos:
            self._drag_prev_pos = new_pos
            stack.symmetry_center = new_pos

    def drag_stop_cb(self, tdw):
        if self._move_item and not self._move_timeout_id:
            self._do_move()
        tdw.renderer.defer_hq_rendering(0)
        return super(SymmetryEditMode, self).drag_stop_cb(tdw)


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
        self._symmetry_type_combo = None
        self._axis_sym_lines_entry = None
        from gui.application import get_app
        self.app = get_app()
        rootstack = self.app.doc.model.layer_stack
        x, y = rootstack.symmetry_center

        def pos_adj(start_val):
            return Gtk.Adjustment(
                value=start_val,
                upper=32000,
                lower=-32000,
                step_increment=1,
                page_increment=100,
            )

        self._axis_pos_adj_x = pos_adj(x)
        self._xpos_cb_id = self._axis_pos_adj_x.connect(
            'value-changed',
            self._axis_pos_adj_x_changed,
        )
        self._axis_pos_adj_y = pos_adj(y)
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

    def _init_ui(self):

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
        scale = InputSlider()
        scale.set_range(0, 1)
        scale.set_round_digits(1)
        scale.set_draw_value(False)
        line_alpha = self.app.preferences.get(_ALPHA_PREFS_KEY, _DEFAULT_ALPHA)
        scale.set_value(line_alpha)
        scale.set_hexpand(True)
        scale.set_vexpand(False)
        scale.scale.connect("value-changed", self._scale_value_changed_cb)
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
        entry = Gtk.SpinButton(
            adjustment=self._axis_pos_adj_x,
            climb_rate=0.25,
            digits=0
        )
        entry.set_hexpand(True)
        entry.set_vexpand(False)
        grid.attach(label, 0, row, 1, 1)
        grid.attach(entry, 1, row, 1, 1)

        row += 1
        label = Gtk.Label(label=self._POSITION_LABEL_Y_TEXT)
        label.set_hexpand(False)
        label.set_halign(Gtk.Align.START)
        entry = Gtk.SpinButton(
            adjustment=self._axis_pos_adj_y,
            climb_rate=0.25,
            digits=0
        )
        entry.set_hexpand(True)
        entry.set_vexpand(False)
        grid.attach(label, 0, row, 1, 1)
        grid.attach(entry, 1, row, 1, 1)

        row += 1
        label = Gtk.Label()
        label.set_hexpand(False)
        label.set_halign(Gtk.Align.START)
        self._angle_label = label
        self._update_angle_label()
        grid.attach(label, 0, row, 1, 1)
        scale = InputSlider(self._axis_angle)
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

    def _update_num_lines_sensitivity(self, sym_type):
        self._axis_sym_lines_entry.set_sensitive(
            sym_type in (SymmetryRotational, SymmetrySnowflake)
        )

    def _axis_pos_adj_x_changed(self, adj):
        self.app.doc.model.layer_stack.symmetry_x = int(adj.get_value())

    def _axis_rot_symmetry_lines_changed(self, adj):
        self.app.doc.model.layer_stack.symmetry_lines = int(adj.get_value())

    def _axis_pos_adj_y_changed(self, adj):
        self.app.doc.model.layer_stack.symmetry_y = int(adj.get_value())

    def _angle_value_changed(self, adj):
        self._update_angle_label()
        self.app.doc.model.layer_stack.symmetry_angle = adj.get_value()

    def _symmetry_type_combo_changed_cb(self, combo):
        sym_type = combo.get_model()[combo.get_active()][0]
        self.app.doc.model.layer_stack.symmetry_type = sym_type

    def _scale_value_changed_cb(self, alpha_scale):
        self.app.preferences[_ALPHA_PREFS_KEY] = alpha_scale.get_value()
        for overlay in self._symmetry_overlays():
            overlay.set_line_alpha(alpha_scale.get_value())

    @staticmethod
    def _symmetry_overlays():
        for tdw in gui.tileddrawwidget.TiledDrawWidget.get_visible_tdws():
            for ov in tdw.display_overlays:
                if isinstance(ov, SymmetryOverlay):
                    yield ov


class SymmetryOverlay (gui.overlays.Overlay):
    """Symmetry overlay, operating in display coordinates"""

    _LINE_COL1 = gui.style.EDITABLE_ITEM_COLOR
    _LINE_COL2 = _LINE_COL1.to_contrasting()
    _LINE_COLS = [_LINE_COL1.get_rgb(), _LINE_COL2.get_rgb()]
    _DASH_LENGTH = 10
    _DASH_GAP = 7
    _DASH_PATTERN = [_DASH_LENGTH, 2 * _DASH_GAP + _DASH_LENGTH]
    _DASH_OFFSET = 5
    _EDIT_MODE_MIN_ALPHA = 0.25

    _GRAB_SENSITIVITY = 8  # pixels

    _DISABLE_RADIUS = gui.style.FLOATING_BUTTON_RADIUS
    _CENTER_RADIUS = gui.style.FLOATING_BUTTON_RADIUS / 2

    def __init__(self, doc):
        gui.overlays.Overlay.__init__(self)
        self.doc = doc
        self.tdw = self.doc.tdw
        self.tdw.connect("enter-notify-event", self._enter_notify_cb)
        self.tdw.connect("leave-notify-event", self._leave_notify_cb)
        self.tdw.connect("size-allocate", self._size_allocate_cb)
        rootstack = doc.model.layer_stack
        rootstack.symmetry_state_changed += self._symmetry_state_changed_cb
        self.tdw.transformation_updated += self._transformation_updated_cb
        self._prev_active = rootstack.symmetry_active
        self._trash_icon_pixbuf = None
        self._intersections = None
        self._intersections_view = []
        # When in edit mode, the change callback is ignored
        # and updates should be done manually from whatever
        # is currently doing the edits.
        self._edit_mode = False
        # Whether to render the actual alpha value, or a clamped value.
        # The clamped value should be used when moving the cursor across
        # the canvas when editing, whereas the real alpha should be used
        # when actively editing the alpha value.
        self._real_alpha = True
        prefs = doc.app.preferences
        self._line_alpha = float(prefs.get(_ALPHA_PREFS_KEY, _DEFAULT_ALPHA))
        self._zone = None
        self._active_axis = None
        self._disable_pos = None
        self._center_pos = None
        # View corners - used to calculate axis intersections
        # These are cached since they are surprisingly expensive to calculate
        self._corners = None
        self._alloc = None
        # Invalidation areas (rectangles)
        self._prev_rectangles = []
        self._axis_rectangles = None
        self._disable_rectangle = None
        self._center_rectangle = None
        self._prev_full_redraw = False

    @property
    def view_corners(self):
        """Corners of the current viewport, in model coordinates"""
        if not self._corners:
            self._corners = self.tdw.get_corners_model_coords()
        return self._corners

    @property
    def zone(self):
        """The most recently active edit zone"""
        return self._zone

    @property
    def trash_icon_pixbuf(self):
        if not self._trash_icon_pixbuf:
            self._trash_icon_pixbuf = gui.drawutils.load_symbolic_icon(
                icon_name="mypaint-trash-symbolic",
                size=self._DISABLE_RADIUS,
                fg=(0, 0, 0, 1),
            )
        return self._trash_icon_pixbuf

    def enable_edit_mode(self):
        stack = self.doc.model.layer_stack
        cx, cy = stack.symmetry_center
        self._real_alpha = False
        self._edit_mode = True
        self._recalculate_edit_data(cx, cy)
        self._redraw()

    def disable_edit_mode(self):
        was_true = self._edit_mode
        self._edit_mode = False
        # invalidate any data specific to edit mode
        self._invalidate_edit_data()
        self._disable_pos = None
        self._center_pos = None
        self._real_alpha = True
        if was_true:
            self.tdw.queue_draw()

    def _invalidate_edit_data(self):
        self._active_axis = None
        self._zone = None

    def set_line_alpha(self, alpha):
        invis = not self.line_alpha()
        alpha_changed = self._line_alpha != alpha
        self._line_alpha = alpha
        if self._edit_mode and not self._real_alpha:
            self._real_alpha = True
            self._redraw()
        elif alpha_changed:
            if invis:
                self._recalculate_gui_data()
            self._redraw()

    def update_zone_data(self, x, y):

        axis_changed = False
        data = {}
        layer_stack = self.doc.model.layer_stack

        def cursor_in_circle(center, radius):
            if center:
                cx, cy = center
                return math.hypot(cx - x, cy - y) <= radius
            else:
                return False

        if not layer_stack.symmetry_active:
            new_zone = _EditZone.CREATE_AXIS
        # Canvas button hits - prioritize moving over disabling
        elif cursor_in_circle(self._center_pos, self._CENTER_RADIUS):
            new_zone = _EditZone.MOVE_CENTER
        elif cursor_in_circle(self._disable_pos, self._DISABLE_RADIUS):
            new_zone = _EditZone.DISABLE
        # Check if pointer is on a visible axis
        else:
            axis_data = self._axis_check(x, y)
            if axis_data:
                new_zone = _EditZone.MOVE_AXIS
                axis_changed, cursor, factors = axis_data
                if axis_changed:
                    data = cursor, cursor, factors
            else:
                new_zone = _EditZone.NONE

        if new_zone != _EditZone.MOVE_AXIS:
            self._active_axis = None

        changed = axis_changed or self._zone != new_zone
        button_zones = (_EditZone.NONE, _EditZone.DISABLE)
        only_button = self._zone in button_zones and new_zone in button_zones
        self._zone = new_zone
        if changed:
            self._redraw(only_button=only_button)
        return changed, new_zone, data

    def _calculate_axis_rectangles(self):
        if self._axis_rectangles:
            self._prev_rectangles.extend(self._axis_rectangles)
        self._axis_rectangles = []
        # One rectangle per view intersection - simple and mostly ok, in terms
        # of redraw performance. Should be split up further for long angled
        # lines when we bump requirements so that cairo Regions can be used.
        margin = 2 * gui.style.DRAGGABLE_EDGE_WIDTH
        for i, (x0, y0), (x1, y1) in self._intersections_view:
            m = margin
            if self._edit_mode and i == self._active_axis:
                # compensate for drop shadow
                m *= 2
            w = abs(x1 - x0) + 2 * m
            h = abs(y1 - y0) + 2 * m
            x = min(x0, x1) - m
            y = min(y0, y1) - m
            self._axis_rectangles.append((x, y, w, h))

    def _axis_check(self, x, y):

        for i, p1, p2 in self._intersections:
            cursor_name = self.tdw.get_move_cursor_name_for_edge(
                (x, y), p1, p2, self._GRAB_SENSITIVITY)
            if cursor_name:
                if self._active_axis == i:
                    return False, None, None
                self._active_axis = i
                return True, cursor_name, (p1, p2)

    def _transformation_updated_cb(self, *args):
        self._corners = None
        self._invalidate_edit_data()
        if self._prev_active:
            self._recalculate_gui_data()

    def _recalculate_gui_data(self):
        stack = self.doc.model.layer_stack
        cx, cy = stack.symmetry_center
        if stack.symmetry_active and self.line_alpha():
            self._intersections = get_viewport_intersections(
                stack.symmetry_type, cx, cy,
                stack.symmetry_angle, stack.symmetry_lines, self.view_corners,
            )

            def rounded_view_coords(n):
                x, y = self.tdw.model_to_display(*n)
                return round(x), round(y)
            self._intersections_view = [
                (i, rounded_view_coords(p1), rounded_view_coords(p2))
                for i, p1, p2 in self._intersections]

            if not self._full_redraw():
                self._calculate_axis_rectangles()
        if self._edit_mode:
            self._recalculate_edit_data(cx, cy)

    @property
    def tdw_allocation(self):
        if not self._alloc:
            self._alloc = Rect.new_from_gdk_rectangle(
                self.tdw.get_allocation())
        return self._alloc

    def _recalculate_edit_data(self, cx, cy):
        assert self._edit_mode

        # TiledDrawWidget allocation
        alloc = self.tdw_allocation

        # Position of disable-button, which is always visible when editing
        # Positioning strategy: If the symmetry center is inside the viewport,
        # the button is placed to one side of the center. If not, the button is
        # placed in the location closest to the center. That way it can also be
        # used to locate the center visually, when the center is out of view.
        bp_x, bp_y = self.tdw.model_to_display(
            *lib.alg.nearest_point_in_poly(self.view_corners, (cx, cy)))
        margin = 2 * self._DISABLE_RADIUS
        # Position to left or right of the center, depending on whether
        # the center is closer to the left or the right edge of the view.
        bp_x += math.copysign(margin, alloc.x + alloc.w / 2 - bp_x)
        # Constrain to viewport with margins (so the button is fully shown).
        new_pos = alloc.expanded(-margin).clamped_point(bp_x, bp_y)
        if self._disable_pos != new_pos:
            self._disable_pos = new_pos
            if self._disable_rectangle:
                self._prev_rectangles.append(self._disable_rectangle)
            offs = self._DISABLE_RADIUS + 4  # margin for drop shadow
            x, y = new_pos
            new_rect = (x - offs, y - offs, offs * 2, offs * 2)
            self._disable_rectangle = new_rect

        # Symmetry center symbol position. Not constrained to the viewport,
        # but won't be drawn, or tested for, when outside of it.
        d_cx, d_cy = self.tdw.model_to_display(cx, cy)
        if alloc.expanded(self._CENTER_RADIUS).contains_pixel(d_cx, d_cy):
            if self._center_pos != (d_cx, d_cy):
                if self._center_rectangle:
                    self._prev_rectangles.append(self._center_rectangle)
                offs = self._CENTER_RADIUS + 4
                center_rect = (d_cx - offs, d_cy - offs, offs * 2, offs * 2)
                self._center_rectangle = center_rect
                self._center_pos = d_cx, d_cy
        else:
            self._center_pos = None
            if self._center_rectangle:
                self._prev_rectangles.append(self._center_rectangle)
                self._center_rectangle = None

    def _enter_notify_cb(self, tdw, event):
        if self._edit_mode and self._real_alpha:
            self._real_alpha = False
            if not self._line_alpha:
                self._recalculate_gui_data()
            self._redraw()

    def _leave_notify_cb(self, twd, event):
        if self._edit_mode:
            self._zone = None
            self._real_alpha = True
            self._redraw()

    def _size_allocate_cb(self, tdw, allocation):
        # The value is only needed in edit mode, so the cached value is
        # invalidated and the new one fetched when needed.
        self._alloc = None

    def _symmetry_state_changed_cb(
            self, stack, active, center, symmetry_type, symmetry_lines, angle):
        redraw, recalc = False, False
        params = (center, symmetry_type, symmetry_lines, angle)
        if stack.symmetry_active and [p for p in params if p is not None]:
            redraw, recalc = True, True
        if active is not None and active != self._prev_active:
            self._prev_active = active
            redraw = True
            recalc = recalc or active
        if recalc:
            self._recalculate_gui_data()
        if redraw:
            if symmetry_type is not None:
                self._prev_full_redraw = True
            self._redraw()

    def _full_redraw(self):
        return len(self._intersections_view) > 4

    def _redraw(self, only_button=False):

        if only_button:
            self.tdw.queue_draw_area(*self._disable_rectangle)
            return

        # When there are lots of intersections, just redraw the entire screen
        if self._full_redraw():
            self._prev_full_redraw = True
            self.tdw.queue_draw()
            return
        elif self._prev_full_redraw:
            self._prev_full_redraw = False
            self.tdw.queue_draw()
            return

        # If there was no full redraw, invalidate individual areas.
        areas = []
        if self._prev_rectangles:
            areas = self._prev_rectangles
        self._prev_rectangles = []
        if self._edit_mode:
            if self._disable_rectangle:
                areas.append(self._disable_rectangle)
            if self._center_pos:
                areas.append(self._center_rectangle)
        axis_rectangles = self._axis_rectangles
        if axis_rectangles:
            self._prev_rectangles[:] = axis_rectangles
            areas.extend(axis_rectangles)
        for area in areas:
            self.tdw.queue_draw_area(*area)

    def paint(self, cr):
        """Paint the overlay, in display coordinates"""
        if self.doc.model.layer_stack.symmetry_active:
            self._render_axis_lines(cr)
            if self._edit_mode:
                if self._disable_pos:
                    self._draw_disable_button(cr)
                if self._center_pos:
                    self._draw_center_button(cr)

    def _draw_disable_button(self, cr):
        x, y = self._disable_pos
        col = self._item_color(self._zone == _EditZone.DISABLE)
        gui.drawutils.render_round_floating_button(
            cr, x, y, col, pixbuf=self.trash_icon_pixbuf,
        )

    def _draw_center_button(self, cr):
        radius, (x, y) = self._CENTER_RADIUS, self._center_pos
        col = self._item_color(self._zone == _EditZone.MOVE_CENTER)
        gui.drawutils.render_round_floating_color_chip(
            cr, x, y, col, radius, z=0)

    def line_alpha(self):
        opaque_line_zones = (_EditZone.MOVE_AXIS, _EditZone.MOVE_CENTER)
        if self._edit_mode and self._zone in opaque_line_zones:
            return 1.0
        elif self._edit_mode and not self._real_alpha:
            return max(self._EDIT_MODE_MIN_ALPHA, self._line_alpha)
        else:
            return self._line_alpha

    def _render_axis_lines(self, cr):
        # Convert to display coordinates, with rounding and pixel centering

        line_alpha = self.line_alpha()
        if line_alpha <= 0:
            return

        # Paint the symmetry axis lines
        cr.save()
        cr.push_group()
        cr.set_line_cap(cairo.LINE_CAP_SQUARE)
        cr.set_line_width(gui.style.DRAGGABLE_EDGE_WIDTH)
        for i, (x0, y0), (x1, y1) in self._intersections_view:
            # Draw all axes as active if center is being moved, otherwise only
            # draw an axis as active if it is currently being moved.
            zone = self._zone
            active = self._edit_mode and (
                zone == _EditZone.MOVE_CENTER or
                self._active_axis == i and zone == _EditZone.MOVE_AXIS)
            if not active:
                for offs in (0, 1):
                    dash_offset = self._DASH_OFFSET + (
                        self._DASH_LENGTH + self._DASH_GAP) * offs
                    cr.set_dash(self._DASH_PATTERN, dash_offset)
                    cr.set_source_rgb(*self._LINE_COLS[offs])
                    cr.move_to(x0, y0)
                    cr.line_to(x1, y1)
                    cr.stroke()
            else:
                cr.set_dash([])
                cr.set_source_rgb(*gui.style.ACTIVE_ITEM_COLOR.get_rgb())
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

    if symm_type in (SymmetryHorizontal, SymmetryVertHorz):
        append(angle)
    if symm_type in (SymmetryVertical, SymmetryVertHorz):
        append(angle + math.pi / 2)
    elif symm_type in (SymmetryRotational, SymmetrySnowflake):
        delta = (math.pi * 2) / num_lines
        for i in range(num_lines):
            a = angle + delta * i
            append(a, line_type=lib.alg.LineType.DIRECTIONAL)
    return [(i, p[0], p[1]) for i, p in enumerate(intersections) if p]
