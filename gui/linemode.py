# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2012 by Richard Jones
# Copyright (C) 2012-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# NOTE: Much of this was written before the new modes system.
# NOTE: The InteractionMode stuff was sort of bolted on after the fact.

# TODO: Understand it more; devolve specifics to specific subclasses.
# TODO: Document stuff properly.


## Imports

from __future__ import division, print_function
import math
import logging
from gettext import gettext as _

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib

from .curve import CurveWidget
import gui.mode
import gui.cursor
from lib.pycompat import xrange


logger = logging.getLogger(__name__)

## Module constants

# internal-name, display-name, constant, minimum, default, maximum, tooltip
_LINE_MODE_SETTINGS_LIST = [
    ['entry_pressure', _('Entrance Pressure'), False, 0.0001, 0.3, 1.0,
     _("Stroke entrance pressure for line tools")],
    ['midpoint_pressure', _('Midpoint Pressure'), False, 0.0001, 1.0, 1.0,
     _("Mid-Stroke pressure for line tools")],
    ['exit_pressure', _('Exit Pressure'), False, 0.0001, 0.3, 1.0,
     _("Stroke exit pressure for line tools")],
    ['line_head', _('Head'), False, 0.0001, 0.25, 1.0,
     _("Stroke lead-in end")],
    ['line_tail', _('Tail'), False, 0.0001, 0.75, 1.0,
     _("Stroke trail-off beginning")],
]


## Line pressure settings

class LineModeSettings (object):
    """Manage GtkAdjustments for tweaking LineMode settings.

    An instance resides in the main application singleton. Changes to the
    adjustments are reflected into the app preferences.
    """

    def __init__(self, app):
        """Initializer; initial settings are loaded from the app prefs"""
        object.__init__(self)
        self.app = app
        self.adjustments = {}  #: Dictionary of GtkAdjustments
        self.observers = []  #: List of callbacks
        self._idle_srcid = None
        self._changed_settings = set()
        for line_list in _LINE_MODE_SETTINGS_LIST:
            cname, name, const, min_, default, max_, tooltip = line_list
            prefs_key = "linemode.%s" % cname
            value = float(self.app.preferences.get(prefs_key, default))
            adj = Gtk.Adjustment(value=value, lower=min_, upper=max_,
                                 step_increment=0.01, page_increment=0.1)
            adj.connect("value-changed", self._value_changed_cb, prefs_key)
            self.adjustments[cname] = adj

    def _value_changed_cb(self, adj, prefs_key):
        # Direct GtkAdjustment callback for a single adjustment being changed.
        value = float(adj.get_value())
        self.app.preferences[prefs_key] = value
        self._changed_settings.add(prefs_key)
        if self._idle_srcid is None:
            self._idle_srcid = GLib.idle_add(self._values_changed_idle_cb)

    def _values_changed_idle_cb(self):
        # Aggregate, idle-state callback for multiple adjustments being changed
        # in a single event. Queues redraws, and runs observers. The curve sets
        # multiple settings at once, and we might as well not queue too many
        # redraws.
        if self._idle_srcid is not None:
            current_mode = self.app.doc.modes.top
            if isinstance(current_mode, LineModeBase):
                # Redraw last_line when settings are adjusted
                # in the adjustment Curve
                GLib.idle_add(current_mode.redraw_line_cb)
            for func in self.observers:
                func(self._changed_settings)
            self._changed_settings = set()
            self._idle_srcid = None
        return False


class LineModeCurveWidget (CurveWidget):
    """Graph of pressure by distance, tied to the central LineModeSettings"""

    _SETTINGS_COORDINATE = [('entry_pressure', (0, 1)),
                            ('midpoint_pressure', (1, 1)),
                            ('exit_pressure', (3, 1)),
                            ('line_head', (1, 0)),
                            ('line_tail', (2, 0))]

    def __init__(self):
        from gui.application import get_app
        self.app = get_app()
        CurveWidget.__init__(self, npoints=4, ylockgroups=((1, 2),),
                             changed_cb=self._changed_cb)
        self.app.line_mode_settings.observers.append(self._adjs_changed_cb)
        self._update()

    def _adjs_changed_cb(self, changed):
        logger.debug("Updating curve (changed: %r)", changed)
        self._update()

    def _update(self, from_defaults=False):
        if from_defaults:
            self.points = [(0.0, 0.2), (0.33, 0.5), (0.66, 0.5), (1.0, 0.33)]
        for setting, coord_pair in self._SETTINGS_COORDINATE:
            if not from_defaults:
                adj = self.app.line_mode_settings.adjustments[setting]
                value = adj.get_value()
            else:
                # TODO: move this case into the base settings object
                defaults = [a[4] for a in _LINE_MODE_SETTINGS_LIST
                            if a[0] == setting]
                assert len(defaults) == 1
                value = defaults[0]
            index, subindex = coord_pair
            if not setting.startswith('line'):
                value = 1.0 - value
            if subindex == 0:
                coord = (value, self.points[index][1])
            else:
                coord = (self.points[index][0], value)
            self.set_point(index, coord)
        if from_defaults:
            self._changed_cb(self)
        self.queue_draw()

    def _changed_cb(self, curve):
        """Updates the linemode pressure settings when the curve is altered"""
        for setting, coord_pair in self._SETTINGS_COORDINATE:
            index, subindex = coord_pair
            value = self.points[index][subindex]
            if not setting.startswith('line'):
                value = 1.0 - value
            value = max(0.0001, value)
            adj = self.app.line_mode_settings.adjustments[setting]
            adj.set_value(value)


## Options UI

class LineModeOptionsWidget (gui.mode.PaintingModeOptionsWidgetBase):
    """Options widget for geometric line modes"""

    def init_specialized_widgets(self, row=0):
        curve = LineModeCurveWidget()
        curve.set_size_request(175, 125)
        self._curve = curve
        exp = Gtk.Expander()
        exp.set_label(_(u"Pressure variation…"))
        exp.set_use_markup(False)
        exp.add(curve)
        self.attach(exp, 0, row, 2, 1)
        row += 1
        return row

    def reset_button_clicked_cb(self, button):
        super(LineModeOptionsWidget, self).reset_button_clicked_cb(button)
        self._curve._update(from_defaults=True)


## Interaction modes for making lines

class LineModeBase (gui.mode.ScrollableModeMixin,
                    gui.mode.BrushworkModeMixin,
                    gui.mode.DragMode):
    """Draws geometric lines (base class)"""

    ## Class constants

    _OPTIONS_WIDGET = None

    ## Class configuration.

    permitted_switch_actions = {
        "PanViewMode",
        "ZoomViewMode",
        "RotateViewMode",
        'BrushResizeMode',
    }.union(gui.mode.BUTTON_BINDING_ACTIONS)

    pointer_behavior = gui.mode.Behavior.PAINT_CONSTRAINED
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    @property
    def active_cursor(self):
        cursor_name = gui.cursor.Name.PENCIL
        if not self._line_possible:
            cursor_name = gui.cursor.Name.FORBIDDEN_EVERYWHERE
        return self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            cursor_name
        )

    @classmethod
    def get_name(cls):
        return _(u"Lines and Curves")

    def get_usage(self):
        # TRANSLATORS: users should never see this message
        return _(u"Generic line/curve mode")

    @property
    def inactive_cursor(self):
        cursor_name = gui.cursor.Name.CROSSHAIR_OPEN_PRECISE
        if not self._line_possible:
            cursor_name = gui.cursor.Name.FORBIDDEN_EVERYWHERE
        return self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            cursor_name
        )

    unmodified_persist = True

    # FIXME: all of the logic resides in the base class, for historical
    # reasons, and is decided by line_mode. The differences should be
    # factored out to the user-facing mode subclasses at some point.
    line_mode = None

    ## Initialization

    def __init__(self, **kwds):
        """Initialize"""
        super(LineModeBase, self).__init__(**kwds)
        self.app = None
        self.last_line_data = None
        self.idle_srcid = None
        self._line_possible = False

    ## InteractionMode/DragMode implementation

    def enter(self, doc, **kwds):
        """Enter the mode.

        If modifiers are held when the mode is entered, the mode is a oneshot
        mode and is popped from the mode stack automatically at the end of the
        drag. Without modifiers, line modes may be continued, and some
        subclasses offer additional options for adjusting control points.

        """
        super(LineModeBase, self).enter(doc, **kwds)
        self.app = self.doc.app
        rootstack = self.doc.model.layer_stack
        rootstack.current_path_updated += self._update_cursors
        rootstack.layer_properties_changed += self._update_cursors
        self._update_cursors()

    def leave(self, **kwds):
        rootstack = self.doc.model.layer_stack
        rootstack.current_path_updated -= self._update_cursors
        rootstack.layer_properties_changed -= self._update_cursors
        return super(LineModeBase, self).leave(**kwds)

    def _update_cursors(self, *_ignored):
        if self.in_drag:
            return   # defer update to the end of the drag
        layer = self.doc.model.layer_stack.current
        self._line_possible = layer.get_paintable()
        self.doc.tdw.set_override_cursor(self.inactive_cursor)

    def drag_start_cb(self, tdw, event):
        super(LineModeBase, self).drag_start_cb(tdw, event)
        if self._line_possible:
            self.start_command(self.initial_modifiers)

    def drag_update_cb(self, tdw, event, ev_x, ev_y, dx, dy):
        if self._line_possible:
            self.update_position(ev_x, ev_y)
            if self.idle_srcid is None:
                self.idle_srcid = GLib.idle_add(self._drag_idle_cb)
        return super(LineModeBase, self).drag_update_cb(
            tdw, event, ev_x, ev_y, dx, dy)

    def drag_stop_cb(self, tdw):
        if self._line_possible:
            self.idle_srcid = None
            self.stop_command()
        self._update_cursors()
        return super(LineModeBase, self).drag_stop_cb(tdw)

    def _drag_idle_cb(self):
        # Updates the on-screen line during drags.
        if self.idle_srcid is not None:
            self.idle_srcid = None
            self.process_line()

    def checkpoint(self, flush=True, **kwargs):
        # Only push outstanding changes to the document's undo stack on
        # a request for a flushing sync. Without this, users can't curve
        # the previously drawn line. See also inkmode.py and
        # https://github.com/mypaint/mypaint/issues/262
        if flush:
            super(LineModeBase, self).checkpoint(flush=True, **kwargs)

    ## Options panel

    def get_options_widget(self):
        """Get the (base class singleton) options widget"""
        cls = LineModeBase
        if cls._OPTIONS_WIDGET is None:
            widget = LineModeOptionsWidget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET

    ### Draw dynamic Line, Curve, or Ellipse

    def start_command(self, modifier):
        # :param modifier: the keyboard modifiers which ere in place
        #                   when the mode was created

        active_tdw = self.app.doc.tdw.__class__.get_active_tdw()
        assert active_tdw is not None
        if active_tdw is self.app.scratchpad_doc.tdw:
            self.model = self.app.scratchpad_doc.model
            self.tdw = self.app.scratchpad_doc.tdw
        else:
            # unconditionally
            self.model = self.app.doc.model
            self.tdw = self.app.doc.tdw

        self.done = False
        self.brushwork_begin(self.model, abrupt=False,
                             description=self.get_name())
        layer = self.model.layer_stack.current

        x, y, kbmods = self.local_mouse_state()
        # ignore the modifier used to start this action (don't make it
        # change the action)
        self.invert_kbmods = modifier
        kbmods ^= self.invert_kbmods  # invert using bitwise xor
        shift = kbmods & Gdk.ModifierType.SHIFT_MASK

        # line_mode is the type of line to be drawn eg. "EllipseMode"
        self.mode = self.line_mode
        assert self.mode is not None

        # Ignore slow_tracking.
        # There are some other sttings that interfere
        # with the workings of the Line Tools,
        # but slowtracking is the main one.
        self.adj = self.app.brush_adjustment['slow_tracking']
        self.slow_tracking = self.adj.get_value()
        self.adj.set_value(0)

        # Throughout this module these conventions are used:
        # sx, sy = starting point
        # ex, ey = end point
        # kx, ky = curve point from last line
        # lx, ly = last point from InteractionMode update
        self.sx, self.sy = x, y
        self.lx, self.ly = x, y

        if self.mode == "EllipseMode":
            # Rotation angle of ellipse.
            self.angle = 90
            # Vector to measure any rotation from.
            # Assigned when ratation begins.
            self.ellipse_vec = None
            return
        # If not Ellipse, command must be Straight Line or Sequence
        # First check if the user intends to Curve or move an existing Line
        if shift:
            last_line = self.last_line_data
            last_stroke = layer.get_last_stroke_info()
            if last_line is not None:
                if last_line[1] == last_stroke:
                    self.mode = last_line[0]
                    self.sx, self.sy = last_line[2], last_line[3]
                    self.ex, self.ey = last_line[4], last_line[5]
                    if self.mode == "CurveLine2":
                        length_a = distance(x, y, self.sx, self.sy)
                        length_b = distance(x, y, self.ex, self.ey)
                        self.flip = length_a > length_b
                        if self.flip:
                            self.kx, self.ky = last_line[6], last_line[7]
                            self.k2x, self.k2y = last_line[8], last_line[9]
                        else:
                            self.k2x, self.k2y = last_line[6], last_line[7]
                            self.kx, self.ky = last_line[8], last_line[9]
                    self.model.undo()
                    self.process_line()
                    return

        if self.mode == "SequenceMode":
            if not self.tdw.last_painting_pos:
                return
            else:
                self.sx, self.sy = self.tdw.last_painting_pos

    def update_position(self, x, y):
        self.lx, self.ly = self.tdw.display_to_model(x, y)

    def stop_command(self):
        # End mode
        self.done = True
        x, y = self.process_line()
        self.brushwork_commit(self.model, abrupt=False)
        cmd = self.mode
        self.record_last_stroke(cmd, x, y)

    def record_last_stroke(self, cmd, x, y):
        """ Store last stroke data

        Stroke data is used for redraws and modifications of the line.

        :param str cmd: name of the last command
        :param int x: last cursor x-coordinate
        :param int y: last cursor y-coordinate
        """
        last_line = None
        self.tdw.last_painting_pos = x, y
        # FIXME: should probably not set that from here

        layer = self.model.layer_stack.current
        last_stroke = layer.get_last_stroke_info()
        sx, sy = self.sx, self.sy

        if cmd == "CurveLine1":
            last_line = [
                "CurveLine2", last_stroke,
                sx, sy,
                self.ex, self.ey,
                x, y, x, y,
            ]
            self.tdw.last_painting_pos = self.ex, self.ey

        if cmd == "CurveLine2":
            if self.flip:
                last_line = [
                    cmd, last_stroke,
                    sx, sy,
                    self.ex, self.ey,
                    self.kx, self.ky, self.k2x, self.k2y,
                ]
            else:
                last_line = [
                    cmd, last_stroke,
                    sx, sy,
                    self.ex, self.ey,
                    self.k2x, self.k2y, self.kx, self.ky,
                ]
            self.tdw.last_painting_pos = self.ex, self.ey

        if cmd == "StraightMode" or cmd == "SequenceMode":
            last_line = ["CurveLine1", last_stroke, sx, sy, x, y]

        if cmd == "EllipseMode":
            last_line = [cmd, last_stroke, sx, sy, x, y, self.angle]
            self.tdw.last_painting_pos = sx, sy

        self.last_line_data = last_line
        self.adj.set_value(self.slow_tracking)
        self.model.brush.reset()

    def local_mouse_state(self, last_update=False):
        tdw_win = self.tdw.get_window()
        display = self.tdw.get_display()
        devmgr = display and display.get_device_manager() or None
        coredev = devmgr and devmgr.get_client_pointer() or None
        if coredev and tdw_win:
            win_, x, y, kbmods = tdw_win.get_device_position_double(coredev)
        else:
            x, y, kbmods = (0., 0., Gdk.ModifierType(0))
        if last_update:
            return self.lx, self.ly, kbmods
        x, y = self.tdw.display_to_model(x, y)
        return x, y, kbmods

    def process_line(self):
        sx, sy = self.sx, self.sy
        x, y, kbmods = self.local_mouse_state(last_update=True)
        kbmods ^= self.invert_kbmods  # invert using bitwise xor
        ctrl = kbmods & Gdk.ModifierType.CONTROL_MASK
        shift = kbmods & Gdk.ModifierType.SHIFT_MASK

        if self.mode == "CurveLine1":
            self.dynamic_curve_1(x, y, sx, sy, self.ex, self.ey)

        elif self.mode == "CurveLine2":
            ex, ey = self.ex, self.ey
            kx, ky = self.kx, self.ky
            k2x, k2y = self.k2x, self.k2y
            if shift and ctrl:
                # moved line end
                if not self.flip:
                    self.dynamic_curve_2(k2x, k2y, x, y, ex, ey, kx, ky)
                    self.sx, self.sy = x, y
                else:
                    self.dynamic_curve_2(kx, ky, sx, sy, x, y, k2x, k2y)
                    self.ex, self.ey = x, y
            else:
                # changed curve shape
                self.k2x, self.k2y = x, y
                if not self.flip:
                    self.dynamic_curve_2(x, y, sx, sy, ex, ey, kx, ky)
                else:
                    self.dynamic_curve_2(kx, ky, sx, sy, ex, ey, x, y)

        elif self.mode == "EllipseMode":
            constrain = False
            if ctrl:
                x, y = constrain_to_angle(x, y, sx, sy)
                constrain = True
            if shift:
                self.ellipse_rotation_angle(x, y, sx, sy, constrain)
            else:
                self.ellipse_vec = None
            self.dynamic_ellipse(x, y, sx, sy)

        else:  # if "StraightMode" or "SequenceMode"
            if ctrl or shift:
                x, y = constrain_to_angle(x, y, sx, sy)
            self.dynamic_straight_line(x, y, sx, sy)
        return x, y

    def ellipse_rotation_angle(self, x, y, sx, sy, constrain):
        x1, y1 = normal(sx, sy, x, y)
        if self.ellipse_vec is None:
            self.ellipse_vec = x1, y1
            self.last_angle = self.angle
        x2, y2 = self.ellipse_vec
        px, py = perpendicular(x2, y2)
        pangle = get_angle(x1, y1, px, py)
        angle = get_angle(x1, y1, x2, y2)
        if pangle > 90.0:
            angle = 360 - angle
        angle += self.last_angle
        if constrain:
            angle = constraint_angle(angle)
        self.angle = angle

    ### Line Functions

    # Straight Line
    def dynamic_straight_line(self, x, y, sx, sy):
        self.brush_prep(sx, sy)
        entry_p, midpoint_p, junk, prange2, head, tail = self.line_settings()
        # Beginning
        length, nx, ny = length_and_normal(sx, sy, x, y)
        mx, my = multiply_add(sx, sy, nx, ny, 0.25)
        self._stroke_to(mx, my, entry_p)
        # Middle start
        # length = length/2
        mx, my = multiply_add(sx, sy, nx, ny, head * length)
        self._stroke_to(mx, my, midpoint_p)
        # Middle end
        mx, my = multiply_add(sx, sy, nx, ny, tail * length)
        self._stroke_to(mx, my, midpoint_p)
        # End
        self._stroke_to(x, y, self.exit_pressure)

    # Ellipse
    def dynamic_ellipse(self, x, y, sx, sy):
        points_in_curve = 360
        x1, y1 = difference(sx, sy, x, y)
        x1, y1, sin, cos = starting_point_for_ellipse(x1, y1, self.angle)
        rx, ry = point_in_ellipse(x1, y1, sin, cos, 0)
        self.brush_prep(sx+rx, sy+ry)
        entry_p, midpoint_p, prange1, prange2, h, t = self.line_settings()
        head = points_in_curve * h
        head_range = int(head)+1
        tail = points_in_curve * t
        tail_range = int(tail)+1
        tail_length = points_in_curve - tail
        # Beginning
        px, py = point_in_ellipse(x1, y1, sin, cos, 1)
        length, nx, ny = length_and_normal(rx, ry, px, py)
        mx, my = multiply_add(rx, ry, nx, ny, 0.25)
        self._stroke_to(sx+mx, sy+my, entry_p)
        pressure = abs(1/head * prange1 + entry_p)
        self._stroke_to(sx+px, sy+py, pressure)
        for degree in xrange(2, head_range):
            px, py = point_in_ellipse(x1, y1, sin, cos, degree)
            pressure = abs(degree/head * prange1 + entry_p)
            self._stroke_to(sx+px, sy+py, pressure)
        # Middle
        for degree in xrange(head_range, tail_range):
            px, py = point_in_ellipse(x1, y1, sin, cos, degree)
            self._stroke_to(sx+px, sy+py, midpoint_p)
        # End
        for degree in xrange(tail_range, points_in_curve+1):
            px, py = point_in_ellipse(x1, y1, sin, cos, degree)
            pressure = abs((degree-tail)/tail_length * prange2 + midpoint_p)
            self._stroke_to(sx+px, sy+py, pressure)

    def dynamic_curve_1(self, cx, cy, sx, sy, ex, ey):
        self.brush_prep(sx, sy)
        self.draw_curve_1(cx, cy, sx, sy, ex, ey)

    def dynamic_curve_2(self, cx, cy, sx, sy, ex, ey, kx, ky):
        self.brush_prep(sx, sy)
        self.draw_curve_2(cx, cy, sx, sy, ex, ey, kx, ky)

    # Curve Straight Line
    # Found this page helpful:
    # http://www.caffeineowl.com/graphics/2d/vectorial/bezierintro.html
    def draw_curve_1(self, cx, cy, sx, sy, ex, ey):
        points_in_curve = 100
        entry_p, midpoint_p, prange1, prange2, h, t = self.line_settings()
        mx, my = midpoint(sx, sy, ex, ey)
        length, nx, ny = length_and_normal(mx, my, cx, cy)
        cx, cy = multiply_add(mx, my, nx, ny, length*2)
        x1, y1 = difference(sx, sy, cx, cy)
        x2, y2 = difference(cx, cy, ex, ey)
        head = points_in_curve * h
        head_range = int(head)+1
        tail = points_in_curve * t
        tail_range = int(tail)+1
        tail_length = points_in_curve - tail
        # Beginning
        px, py = point_on_curve_1(1, cx, cy, sx, sy, x1, y1, x2, y2)
        length, nx, ny = length_and_normal(sx, sy, px, py)
        bx, by = multiply_add(sx, sy, nx, ny, 0.25)
        self._stroke_to(bx, by, entry_p)
        pressure = abs(1/head * prange1 + entry_p)
        self._stroke_to(px, py, pressure)
        for i in xrange(2, head_range):
            px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
            pressure = abs(i/head * prange1 + entry_p)
            self._stroke_to(px, py, pressure)
        # Middle
        for i in xrange(head_range, tail_range):
            px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
            self._stroke_to(px, py, midpoint_p)
        # End
        for i in xrange(tail_range, points_in_curve+1):
            px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
            pressure = abs((i-tail)/tail_length * prange2 + midpoint_p)
            self._stroke_to(px, py, pressure)

    def draw_curve_2(self, cx, cy, sx, sy, ex, ey, kx, ky):
        points_in_curve = 100
        self.brush_prep(sx, sy)
        entry_p, midpoint_p, prange1, prange2, h, t = self.line_settings()
        mx, my = (cx+sx+ex+kx)/4.0, (cy+sy+ey+ky)/4.0
        length, nx, ny = length_and_normal(mx, my, cx, cy)
        cx, cy = multiply_add(mx, my, nx, ny, length*2)
        length, nx, ny = length_and_normal(mx, my, kx, ky)
        kx, ky = multiply_add(mx, my, nx, ny, length*2)
        x1, y1 = difference(sx, sy, cx, cy)
        x2, y2 = difference(cx, cy, kx, ky)
        x3, y3 = difference(kx, ky, ex, ey)
        head = points_in_curve * h
        head_range = int(head)+1
        tail = points_in_curve * t
        tail_range = int(tail)+1
        tail_length = points_in_curve - tail
        # Beginning
        px, py = point_on_curve_2(1, cx, cy, sx, sy, kx, ky,
                                  x1, y1, x2, y2, x3, y3)
        length, nx, ny = length_and_normal(sx, sy, px, py)
        bx, by = multiply_add(sx, sy, nx, ny, 0.25)
        self._stroke_to(bx, by, entry_p)
        pressure = abs(1/head * prange1 + entry_p)
        self._stroke_to(px, py, pressure)
        for i in xrange(2, head_range):
            px, py = point_on_curve_2(i, cx, cy, sx, sy, kx, ky,
                                      x1, y1, x2, y2, x3, y3)
            pressure = abs(i/head * prange1 + entry_p)
            self._stroke_to(px, py, pressure)
        # Middle
        for i in xrange(head_range, tail_range):
            px, py = point_on_curve_2(i, cx, cy, sx, sy, kx, ky,
                                      x1, y1, x2, y2, x3, y3)
            self._stroke_to(px, py, midpoint_p)
        # End
        for i in xrange(tail_range, points_in_curve+1):
            px, py = point_on_curve_2(i, cx, cy, sx, sy, kx, ky,
                                      x1, y1, x2, y2, x3, y3)
            pressure = abs((i-tail)/tail_length * prange2 + midpoint_p)
            self._stroke_to(px, py, pressure)

    def _stroke_to(self, x, y, pressure):
        # FIXME add control for time, similar to inktool
        duration = 1.0

        self.stroke_to(self.model, duration, x, y, pressure, 0.0, 0.0,
                       self.doc.tdw.scale, self.doc.tdw.rotation, 0.0,
                       auto_split=False)

    def brush_prep(self, sx, sy):
        # Send brush to where the stroke will begin
        self.model.brush.reset()
        self.brushwork_rollback(self.model)

        self.stroke_to(self.model, 10.0, sx, sy, 0.0, 0.0, 0.0,
                       self.doc.tdw.scale, self.doc.tdw.rotation, 0.0,
                       auto_split=False)

    ## Line mode settings

    @property
    def entry_pressure(self):
        adj = self.app.line_mode_settings.adjustments["entry_pressure"]
        return adj.get_value()

    @property
    def midpoint_pressure(self):
        adj = self.app.line_mode_settings.adjustments["midpoint_pressure"]
        return adj.get_value()

    @property
    def exit_pressure(self):
        adj = self.app.line_mode_settings.adjustments["exit_pressure"]
        return adj.get_value()

    @property
    def head(self):
        adj = self.app.line_mode_settings.adjustments["line_head"]
        return adj.get_value()

    @property
    def tail(self):
        adj = self.app.line_mode_settings.adjustments["line_tail"]
        return adj.get_value()

    def line_settings(self):
        p1 = self.entry_pressure
        p2 = self.midpoint_pressure
        p3 = self.exit_pressure
        if self.head == 0.0001:
            p1 = p2
        prange1 = p2 - p1
        prange2 = p3 - p2
        return p1, p2, prange1, prange2, self.head, self.tail

    def redraw_line_cb(self):
        # Redraws the line when the line_mode_settings change
        last_line = self.last_line_data
        if last_line is not None:
            current_layer = self.model.layer_stack.current
            last_stroke = current_layer.get_last_stroke_info()
            if last_line[1] is last_stroke:
                # ignore slow_tracking
                self.done = True
                self.adj = self.app.brush_adjustment['slow_tracking']
                self.slow_tracking = self.adj.get_value()
                self.adj.set_value(0)
                self.brushwork_rollback(self.model)
                self.model.undo()
                command = last_line[0]
                self.sx, self.sy = last_line[2], last_line[3]
                self.ex, self.ey = last_line[4], last_line[5]
                x, y = self.ex, self.ey
                if command == "EllipseMode":
                    self.angle = last_line[6]
                    self.dynamic_ellipse(self.ex, self.ey,
                                         self.sx, self.sy)
                if command == "CurveLine1":
                    self.dynamic_straight_line(self.ex, self.ey,
                                               self.sx, self.sy)
                    command = "StraightMode"
                if command == "CurveLine2":
                    x, y = last_line[6], last_line[7]
                    self.kx, self.ky = last_line[8], last_line[9]
                    self.k2x, self.k2y = x, y
                    if (x, y) == (self.kx, self.ky):
                        self.dynamic_curve_1(x, y, self.sx, self.sy,
                                             self.ex, self.ey)
                        command = "CurveLine1"
                    else:
                        self.flip = False
                        self.dynamic_curve_2(x, y, self.sx, self.sy,
                                             self.ex, self.ey,
                                             self.kx, self.ky)
                self.model.sync_pending_changes()
                self.record_last_stroke(command, x, y)


class StraightMode (LineModeBase):
    ACTION_NAME = "StraightMode"
    line_mode = "StraightMode"

    @classmethod
    def get_name(cls):
        return _(u"Lines and Curves")

    def get_usage(self):
        return _(u"Draw straight lines; Shift adds curves, "
                 "Shift + Ctrl moves line ends, "
                 "Ctrl constrains angle")


class SequenceMode (LineModeBase):
    ACTION_NAME = "SequenceMode"
    line_mode = "SequenceMode"

    @classmethod
    def get_name(cls):
        return _(u"Connected Lines")

    def get_usage(cls):
        return _("Draw a sequence of lines; Shift adds curves, "
                 "Ctrl constrains angle")


class EllipseMode (LineModeBase):
    ACTION_NAME = "EllipseMode"
    line_mode = "EllipseMode"

    @classmethod
    def get_name(cls):
        return _(u"Ellipses and Circles")

    def get_usage(self):
        return _(u"Draw ellipses; Shift rotates, Ctrl constrains ratio/angle")


## Curve Math
def point_on_curve_1(t, cx, cy, sx, sy, x1, y1, x2, y2):
    ratio = t/100.0
    x3, y3 = multiply_add(sx, sy, x1, y1, ratio)
    x4, y4 = multiply_add(cx, cy, x2, y2, ratio)
    x5, y5 = difference(x3, y3, x4, y4)
    x, y = multiply_add(x3, y3, x5, y5, ratio)
    return x, y


def point_on_curve_2(t, cx, cy, sx, sy, kx, ky, x1, y1, x2, y2, x3, y3):
    ratio = t/100.0
    x4, y4 = multiply_add(sx, sy, x1, y1, ratio)
    x5, y5 = multiply_add(cx, cy, x2, y2, ratio)
    x6, y6 = multiply_add(kx, ky, x3, y3, ratio)
    x1, y1 = difference(x4, y4, x5, y5)
    x2, y2 = difference(x5, y5, x6, y6)
    x4, y4 = multiply_add(x4, y4, x1, y1, ratio)
    x5, y5 = multiply_add(x5, y5, x2, y2, ratio)
    x1, y1 = difference(x4, y4, x5, y5)
    x, y = multiply_add(x4, y4, x1, y1, ratio)
    return x, y


## Ellipse Math
def starting_point_for_ellipse(x, y, rotate):
    # Rotate starting point
    r = math.radians(rotate)
    sin = math.sin(r)
    cos = math.cos(r)
    x, y = rotate_ellipse(x, y, cos, sin)
    return x, y, sin, cos


def point_in_ellipse(x, y, r_sin, r_cos, degree):
    # Find point in ellipse
    r2 = math.radians(degree)
    cos = math.cos(r2)
    sin = math.sin(r2)
    x = x * cos
    y = y * sin
    # Rotate Ellipse
    x, y = rotate_ellipse(y, x, r_sin, r_cos)
    return x, y


def rotate_ellipse(x, y, sin, cos):
    x1, y1 = multiply(x, y, sin)
    x2, y2 = multiply(x, y, cos)
    x = x2 - y1
    y = y2 + x1
    return x, y


## Vector Math
def get_angle(x1, y1, x2, y2):
    dot = dot_product(x1, y1, x2, y2)
    if abs(dot) < 1.0:
        angle = math.acos(dot) * 180/math.pi
    else:
        angle = 0.0
    return angle


def constrain_to_angle(x, y, sx, sy):
    length, nx, ny = length_and_normal(sx, sy, x, y)
    # dot = nx*1 + ny*0 therefore nx
    angle = math.acos(nx) * 180/math.pi
    angle = constraint_angle(angle)
    ax, ay = angle_normal(ny, angle)
    x = sx + ax*length
    y = sy + ay*length
    return x, y


def constraint_angle(angle):
    n = angle//15
    n1 = n*15
    rem = angle - n1
    if rem < 7.5:
        angle = n*15.0
    else:
        angle = (n+1)*15.0
    return angle


def angle_normal(ny, angle):
    if ny < 0.0:
        angle = 360.0 - angle
    radians = math.radians(angle)
    x = math.cos(radians)
    y = math.sin(radians)
    return x, y


def length_and_normal(x1, y1, x2, y2):
    x, y = difference(x1, y1, x2, y2)
    length = vector_length(x, y)
    if length == 0.0:
        x, y = 0.0, 0.0
    else:
        x, y = x/length, y/length
    return length, x, y


def normal(x1, y1, x2, y2):
    junk, x, y = length_and_normal(x1, y1, x2, y2)
    return x, y


def vector_length(x, y):
    length = math.sqrt(x*x + y*y)
    return length


def distance(x1, y1, x2, y2):
    x, y = difference(x1, y1, x2, y2)
    length = vector_length(x, y)
    return length


def dot_product(x1, y1, x2, y2):
    return x1*x2 + y1*y2


def multiply_add(x1, y1, x2, y2, d):
    x3, y3 = multiply(x2, y2, d)
    x, y = add(x1, y1, x3, y3)
    return x, y


def multiply(x, y, d):
    # Multiply vector
    x = x*d
    y = y*d
    return x, y


def add(x1, y1, x2, y2):
    # Add vectors
    x = x1+x2
    y = y1+y2
    return x, y


def difference(x1, y1, x2, y2):
    # Difference in x and y between two points
    x = x2-x1
    y = y2-y1
    return x, y


def midpoint(x1, y1, x2, y2):
    # Midpoint between to points
    x = (x1+x2)/2.0
    y = (y1+y2)/2.0
    return x, y


def perpendicular(x1, y1):
    # Swap x and y, then flip one sign to give vector at 90 degree
    x = -y1
    y = x1
    return x, y
