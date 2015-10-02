# This file is part of MyPaint.
# Copyright (C) 2012-2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


## Imports

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
from lib.color import RGBColor
import lib.helpers
import gui.drawutils
from lib.gettext import C_

from gi.repository import Gdk
from gi.repository import Gtk


## Module settings

_DEFAULT_ALPHA = 0.333
_ALPHA_PREFS_KEY = 'symmetry.line_alpha'


## Class defs

class _EditZone:

    UNKNOWN = 0
    CREATE_AXIS = 1
    MOVE_AXIS = 2
    DELETE_AXIS = 3


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
    permitted_switch_actions = set([
        'ShowPopupMenu',
        'RotateViewMode',
        'ZoomViewMode',
        'PanViewMode',
    ])

    _GRAB_SENSITIVITY = 8  # pixels

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
        from application import get_app
        app = get_app()
        self.app = app
        statusbar_cid = app.statusbar.get_context_id(self._STATUSBAR_CONTEXT)
        self._statusbar_context_id = statusbar_cid
        self._drag_start_axis = None
        self._drag_start_model_x = None
        self.zone = _EditZone.UNKNOWN
        self._last_msg_zone = None
        self._click_info = None
        self.button_pos = None
        self._entered_before = False
        self.line_alphafrac = 0.0

    def enter(self, doc, **kwds):
        """Enter the mode"""
        super(SymmetryEditMode, self).enter(doc, **kwds)
        # Initialize/fetch cursors
        mkcursor = lambda name: doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            name,
        )
        self._move_cursors = {}
        self.cursor_remove = mkcursor(gui.cursor.Name.ARROW)
        self.cursor_add = mkcursor(gui.cursor.Name.ADD)
        self.cursor_normal = mkcursor(gui.cursor.Name.ARROW)
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
        statusbar = self.app.statusbar
        statusbar_cid = self._statusbar_context_id
        statusbar.remove_all(statusbar_cid)
        msgs = {
            _EditZone.CREATE_AXIS: self._STATUSBAR_CREATE_AXIS_MSG,
            _EditZone.MOVE_AXIS: self._STATUSBAR_MOVE_AXIS_MSG,
            _EditZone.DELETE_AXIS: self._STATUSBAR_DELETE_AXIS_MSG,
        }
        msg = msgs.get(self.zone, None)
        if msg:
            statusbar.push(statusbar_cid, msg)
            self._last_msg_zone = self.zone

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = SymmetryEditOptionsWidget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET

    ## Events and internals

    def button_press_cb(self, tdw, event):
        if self.zone in (_EditZone.CREATE_AXIS, _EditZone.DELETE_AXIS):
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
                    model = tdw.doc
                    layer_stack = model.layer_stack
                    if zone0 == _EditZone.DELETE_AXIS:
                        layer_stack.symmetry_active = False
                    elif zone0 == _EditZone.CREATE_AXIS:
                        x, y = tdw.display_to_model(event.x, event.y)
                        layer_stack.symmetry_axis = x
                        layer_stack.symmetry_active = True
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
        old_zone = self.zone
        new_zone = None
        new_alphafrac = self.line_alphafrac
        xm, ym = tdw.display_to_model(x, y)
        model = tdw.doc
        layer_stack = model.layer_stack
        axis = layer_stack.symmetry_axis
        if not layer_stack.symmetry_active:
            self.active_cursor = self.cursor_add
            self.inactive_cursor = self.cursor_add
            new_zone = _EditZone.CREATE_AXIS

        # Button hits.
        # NOTE: the position is calculated by the related overlay,
        # in its paint() method.
        if new_zone is None and self.button_pos:
            bx, by = self.button_pos
            d = math.hypot(bx-x, by-y)
            if d <= gui.style.FLOATING_BUTTON_RADIUS:
                self.active_cursor = self.cursor_remove
                self.inactive_cursor = self.cursor_remove
                new_zone = _EditZone.DELETE_AXIS

        if new_zone is None:
            move_cursor_name, perp_dist = tdw.get_move_cursor_name_for_edge(
                (x, y),
                (axis, 0),
                (axis, 1000),
                tolerance=self._GRAB_SENSITIVITY,
                finite=False,
            )
            if move_cursor_name:
                move_cursor = self._move_cursors.get(move_cursor_name)
                if not move_cursor:
                    move_cursor = self.doc.app.cursors.get_action_cursor(
                        self.ACTION_NAME,
                        move_cursor_name,
                    )
                    self._move_cursors[move_cursor_name] = move_cursor
                self.active_cursor = move_cursor
                self.inactive_cursor = move_cursor
                new_zone = _EditZone.MOVE_AXIS
            dfrac = lib.helpers.clamp(
                perp_dist / (10.0 * self._GRAB_SENSITIVITY),
                0.0, 1.0,
            )
            new_alphafrac = 1.0 - dfrac

        if new_zone is None:
            new_zone = _EditZone.UNKNOWN
            self.active_cursor = self.cursor_normal
            self.inactive_cursor = self.cursor_normal

        if new_zone != old_zone:
            self.zone = new_zone
            self._update_statusbar()
            tdw.queue_draw()
        elif new_alphafrac != self.line_alphafrac:
            tdw.queue_draw()
            self.line_alphafrac = new_alphafrac

    def motion_notify_cb(self, tdw, event):
        if not self.in_drag:
            self._update_zone_and_cursor(tdw, event.x, event.y)
            tdw.set_override_cursor(self.inactive_cursor)
        return super(SymmetryEditMode, self).motion_notify_cb(tdw, event)

    def drag_start_cb(self, tdw, event):
        model = tdw.doc
        layer_stack = model.layer_stack
        self._update_zone_and_cursor(tdw, event.x, event.y)
        if self.zone == _EditZone.MOVE_AXIS:
            x0, y0 = self.start_x, self.start_y
            self._drag_start_axis = int(round(layer_stack.symmetry_axis))
            x0_m, y0_m = tdw.display_to_model(x0, y0)
            self._drag_start_model_x = x0_m
        return super(SymmetryEditMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, dx, dy):
        if self.zone == _EditZone.MOVE_AXIS:
            x_m, y_m = tdw.display_to_model(event.x, event.y)
            axis = self._drag_start_axis + x_m - self._drag_start_model_x
            axis = int(round(axis))
            if axis != self._drag_start_axis:
                model = tdw.doc
                layer_stack = model.layer_stack
                layer_stack.symmetry_axis = axis
        return super(SymmetryEditMode, self).drag_update_cb(tdw, event, dx, dy)

    def drag_stop_cb(self, tdw):
        if self.zone == _EditZone.MOVE_AXIS:
            tdw.queue_draw()
        return super(SymmetryEditMode, self).drag_stop_cb(tdw)


class SymmetryEditOptionsWidget (Gtk.Alignment):

    _POSITION_LABEL_TEXT = C_(
        "symmetry axis options panel: labels",
        u"Position:",
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

    def __init__(self):
        Gtk.Alignment.__init__(self, 0.5, 0.5, 1.0, 1.0)
        self._axis_pos_dialog = None
        self._axis_pos_button = None
        from application import get_app
        self.app = get_app()
        rootstack = self.app.doc.model.layer_stack
        self._axis_pos_adj = Gtk.Adjustment(
            rootstack.symmetry_axis,
            upper=32000,
            lower=-32000,
            step_incr=1,
            page_incr=100,
        )
        self._axis_pos_adj.connect(
            'value-changed',
            self._axis_pos_adj_changed,
        )
        self._init_ui()
        rootstack.symmetry_state_changed += self._symmetry_state_changed_cb
        self._update_axis_pos_button_label(rootstack.symmetry_axis)

    def _init_ui(self):
        app = self.app

        # Dialog for showing and editing the axis value directly
        buttons = (Gtk.STOCK_OK, Gtk.ResponseType.ACCEPT)
        dialog = gui.windowing.Dialog(
            app, C_(
                "symmetry axis options panel: axis position dialog: window title",
                u"Axis Position",
            ),
            app.drawWindow,
            buttons=buttons,
        )
        dialog.connect('response', self._axis_pos_dialog_response_cb)
        grid = Gtk.Grid()
        grid.set_border_width(gui.widgets.SPACING_LOOSE)
        grid.set_column_spacing(gui.widgets.SPACING)
        grid.set_row_spacing(gui.widgets.SPACING)
        label = Gtk.Label(self._POSITION_LABEL_TEXT)
        label.set_hexpand(False)
        label.set_vexpand(False)
        grid.attach(label, 0, 0, 1, 1)
        entry = Gtk.SpinButton(
            adjustment=self._axis_pos_adj,
            climb_rate=0.25,
            digits=0
        )
        entry.set_hexpand(True)
        entry.set_vexpand(False)
        grid.attach(entry, 1, 0, 1, 1)
        dialog_content_box = dialog.get_content_area()
        dialog_content_box.pack_start(grid, True, True)
        self._axis_pos_dialog = dialog

        # Layout grid
        row = 0
        grid = Gtk.Grid()
        grid.set_border_width(gui.widgets.SPACING_CRAMPED)
        grid.set_row_spacing(gui.widgets.SPACING_CRAMPED)
        grid.set_column_spacing(gui.widgets.SPACING_CRAMPED)
        self.add(grid)

        row += 1
        label = Gtk.Label(self._ALPHA_LABEL_TEXT)
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
        label = Gtk.Label(self._POSITION_LABEL_TEXT)
        label.set_hexpand(False)
        label.set_halign(Gtk.Align.START)
        button = Gtk.Button(self._POSITION_BUTTON_TEXT_INACTIVE)
        button.set_vexpand(False)
        button.connect("clicked", self._axis_pos_button_clicked_cb)
        button.set_hexpand(True)
        button.set_vexpand(False)
        grid.attach(label, 0, row, 1, 1)
        grid.attach(button, 1, row, 1, 1)
        self._axis_pos_button = button

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


    def _symmetry_state_changed_cb(self, rootstack, active, x):
        self._update_axis_pos_button_label(x)
        dialog = self._axis_pos_dialog
        dialog_content_box = dialog.get_content_area()
        if x is None:
            dialog_content_box.set_sensitive(False)
        else:
            dialog_content_box.set_sensitive(True)
            adj = self._axis_pos_adj
            adj_pos = int(adj.get_value())
            model_pos = int(x)
            if adj_pos != model_pos:
                adj.set_value(model_pos)

    def _update_axis_pos_button_label(self, x):
        if x is None:
            text = self._POSITION_BUTTON_TEXT_INACTIVE
        else:
            text = self._POSITION_BUTTON_TEXT_TEMPLATE % (x,)
        self._axis_pos_button.set_label(text)

    def _axis_pos_adj_changed(self, adj):
        rootstack = self.app.doc.model.layer_stack
        model_pos = int(rootstack.symmetry_axis)
        adj_pos = int(adj.get_value())
        if adj_pos != model_pos:
            rootstack.symmetry_axis = adj_pos

    def _axis_pos_button_clicked_cb(self, button):
        self._axis_pos_dialog.show_all()

    def _axis_pos_dialog_response_cb(self, dialog, response_id):
        if response_id == Gtk.ResponseType.ACCEPT:
            dialog.hide()

    def _scale_value_changed_cb(self, scale):
        alpha = scale.get_value()
        prefs = self.app.preferences
        prefs[_ALPHA_PREFS_KEY] = alpha
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

    def __init__(self, doc):
        gui.overlays.Overlay.__init__(self)
        self.doc = doc
        self.tdw = self.doc.tdw
        rootstack = doc.model.layer_stack
        rootstack.symmetry_state_changed += self._symmetry_state_changed_cb
        doc.modes.changed += self._active_mode_changed_cb
        self._trash_icon_pixbuf = None

    def _symmetry_state_changed_cb(self, rootstack, active, x):
        self.tdw.queue_draw()

    def _active_mode_changed_cb(self, mode_stack, old, new):
        for mode in (old, new):
            if isinstance(mode, SymmetryEditMode):
                self.tdw.queue_draw()
                break

    def paint(self, cr):
        """Paint the overlay, in display coordinates"""

        # The symmetry axis is a line (x==self.axis) in model coordinates
        model = self.doc.model
        if not model.layer_stack.symmetry_active:
            return
        axis_x_m = model.layer_stack.symmetry_axis

        # allocation, in display coords
        alloc = self.tdw.get_allocation()
        view_x0, view_y0 = alloc.x, alloc.y
        view_x1, view_y1 = view_x0+alloc.width, view_y0+alloc.height
        view_center = ((view_x1-view_x0)/2.0, (view_y1-view_y0)/2.0)

        # Viewing rectangle extents, in model coords
        viewport_corners = [
            (view_x0, view_y0),
            (view_x0, view_y1),
            (view_x1, view_y1),
            (view_x1, view_y0),
        ]
        viewport_corners_m = [
            self.tdw.display_to_model(*c)
            for c in viewport_corners
        ]

        # Viewport extent in x in model space
        min_corner_y_m = min([c_m[1] for c_m in viewport_corners_m])
        max_corner_y_m = max([c_m[1] for c_m in viewport_corners_m])
        axis_p_min = (axis_x_m, min_corner_y_m)
        axis_p_max = (axis_x_m, max_corner_y_m)

        # The two places where the axis intersects the viewing rectangle
        intersections = [
            lib.alg.intersection_of_segments(p1, p2, axis_p_min, axis_p_max)
            for (p1, p2) in lib.alg.pairwise(viewport_corners_m)
        ]
        intersections = [p for p in intersections if p is not None]
        if len(intersections) != 2:
            return
        intsc0_m, intsc1_m = intersections

        # Back to display coords, with rounding and pixel centring
        ax_x0, ax_y0 = [math.floor(c) for c in
                        self.tdw.model_to_display(*intsc0_m)]
        ax_x1, ax_y1 = [math.floor(c) for c in
                        self.tdw.model_to_display(*intsc1_m)]

        # Paint the symmetry axis
        cr.save()

        cr.push_group()

        cr.set_line_cap(cairo.LINE_CAP_SQUARE)
        cr.set_dash(self._DASH_PATTERN, self._DASH_OFFSET)

        mode_stack = self.doc.modes
        active_edit_mode = None

        for mode in reversed(list(mode_stack)):
            if isinstance(mode, SymmetryEditMode):
                active_edit_mode = mode
                break

        prefs = self.tdw.app.preferences
        min_alpha = float(prefs.get(_ALPHA_PREFS_KEY, _DEFAULT_ALPHA))
        max_alpha = 1.0

        if not active_edit_mode:
            line_color = gui.style.EDITABLE_ITEM_COLOR
            line_alpha = min_alpha
        elif mode.zone == _EditZone.MOVE_AXIS:
            line_color = gui.style.ACTIVE_ITEM_COLOR
            line_alpha = max_alpha
        else:
            line_color = gui.style.EDITABLE_ITEM_COLOR
            line_alpha = min_alpha + (
                active_edit_mode.line_alphafrac * (max_alpha-min_alpha)
            )

        line_width = gui.style.DRAGGABLE_EDGE_WIDTH
        if line_width % 2 != 0:
            ax_x0 += 0.5
            ax_y0 += 0.5
            ax_x1 += 0.5
            ax_y1 += 0.5

        cr.move_to(ax_x0, ax_y0)
        cr.line_to(ax_x1, ax_y1)
        cr.set_line_width(line_width)
        gui.drawutils.render_drop_shadow(cr, z=1)
        cr.set_source_rgb(*line_color.get_rgb())
        cr.stroke()

        cr.pop_group_to_source()
        cr.paint_with_alpha(line_alpha)

        cr.restore()

        if not active_edit_mode:
            return

        # Remove button

        # Positioning strategy: the point on the axis line
        # which is closest to the centre of the viewport.
        button_pos = lib.alg.nearest_point_in_segment(
            seg_start=(ax_x0, ax_y0),
            seg_end=(ax_x1, ax_y1),
            point=view_center,
        )

        if button_pos is None:
            d0 = math.hypot(view_center[0]-ax_x0, view_center[1]-ax_y0)
            d1 = math.hypot(view_center[0]-ax_x1, view_center[1]-ax_y1)
            if d0 < d1:
                button_pos = (ax_x0, ax_y0)
            else:
                button_pos = (ax_x1, ax_y1)
        assert button_pos is not None
        button_pos = [math.floor(c) for c in button_pos]

        # Constrain the position so that it appears within the viewport
        margin = 2 * gui.style.FLOATING_BUTTON_RADIUS
        button_pos = [
            lib.helpers.clamp(
                button_pos[0],
                view_x0 + margin,
                view_x1 - margin,
            ),
            lib.helpers.clamp(
                button_pos[1],
                view_y0 + margin,
                view_y1 - margin,
            ),
        ]

        if not self._trash_icon_pixbuf:
            self._trash_icon_pixbuf = gui.drawutils.load_symbolic_icon(
                icon_name="mypaint-trash-symbolic",
                size=gui.style.FLOATING_BUTTON_ICON_SIZE,
                fg=(0, 0, 0, 1),
            )
        icon_pixbuf = self._trash_icon_pixbuf

        if active_edit_mode.zone == _EditZone.DELETE_AXIS:
            button_color = gui.style.ACTIVE_ITEM_COLOR
        else:
            button_color = gui.style.EDITABLE_ITEM_COLOR

        gui.drawutils.render_round_floating_button(
            cr=cr,
            x=button_pos[0],
            y=button_pos[1],
            color=button_color,
            radius=gui.style.FLOATING_BUTTON_RADIUS,
            pixbuf=icon_pixbuf,
        )
        active_edit_mode.button_pos = button_pos
