# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


## Imports

from __future__ import division, print_function

import math
import collections
import weakref
from logging import getLogger

from gettext import gettext as _
from lib.gibindings import Gdk
from lib.gibindings import GLib
import numpy as np

import gui.mode
import gui.overlays
import gui.style
import gui.drawutils
import lib.helpers
import gui.cursor
import lib.observable
import gui.mvp
from lib.pycompat import xrange


## Module constants

logger = getLogger(__name__)


## Class defs


class _Phase:
    """Enumeration of the states that an InkingMode can be in"""
    CAPTURE = 0
    ADJUST = 1


_NODE_FIELDS = ("x", "y", "pressure", "xtilt", "ytilt", "time", "viewzoom", "viewrotation", "barrel_rotation")


class _Node (collections.namedtuple("_Node", _NODE_FIELDS)):
    """Recorded control point, as a namedtuple.

    Node tuples have the following 6 fields, in order

    * x, y: model coords, float
    * pressure: float in [0.0, 1.0]
    * xtilt, ytilt: float in [-1.0, 1.0]
    * time: absolute seconds, float
    * viewzoom: current zoom level [0.0, 64]
    * viewrotation: current view rotation [-180.0, 180.0]
    * barrel_rotation: float in [0.0, 1.0]
    """


class _EditZone:
    """Enumeration of what the pointer is on in the ADJUST phase"""
    EMPTY_CANVAS = 0  #: Nothing, empty space
    CONTROL_NODE = 1  #: Any control node; see target_node_index
    REJECT_BUTTON = 2  #: On-canvas button that abandons the current line
    ACCEPT_BUTTON = 3  #: On-canvas button that commits the current line


class InkingMode (gui.mode.ScrollableModeMixin,
                  gui.mode.BrushworkModeMixin,
                  gui.mode.DragMode):

    ## Metadata properties

    ACTION_NAME = "InkingMode"
    pointer_behavior = gui.mode.Behavior.PAINT_FREEHAND
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW
    permitted_switch_actions = (
        set(gui.mode.BUTTON_BINDING_ACTIONS).union([
            'RotateViewMode',
            'ZoomViewMode',
            'PanViewMode',
            'BrushResizeMode',
        ])
    )

    ## Metadata methods

    @classmethod
    def get_name(cls):
        return _(u"Inking")

    def get_usage(self):
        return _(u"Draw, and then adjust smooth lines")

    @property
    def inactive_cursor(self):
        return None

    @property
    def active_cursor(self):
        if self.phase == _Phase.ADJUST:
            if self.zone == _EditZone.CONTROL_NODE:
                return self._crosshair_cursor
            elif self.zone != _EditZone.EMPTY_CANVAS:  # assume button
                return self._arrow_cursor
        return None

    ## Class config vars

    # Input node capture settings:
    MAX_INTERNODE_DISTANCE_MIDDLE = 30   # display pixels
    MAX_INTERNODE_DISTANCE_ENDS = 10   # display pixels
    MAX_INTERNODE_TIME = 1 / 100.0   # seconds

    # Captured input nodes are then interpolated with a spline.
    # The code tries to make nice smooth input for the brush engine,
    # but avoids generating too much work.
    INTERPOLATION_MAX_SLICE_TIME = 1 / 200.0   # seconds
    INTERPOLATION_MAX_SLICE_DISTANCE = 20   # model pixels
    INTERPOLATION_MAX_SLICES = MAX_INTERNODE_DISTANCE_MIDDLE * 5
    # In other words, limit to a set number of interpolation slices
    # per display pixel at the time of stroke capture.

    # Node value adjustment settings
    MIN_INTERNODE_TIME = 1 / 200.0   # seconds (used to manage adjusting)

    ## Other class vars

    _OPTIONS_PRESENTER = None   #: Options presenter singleton

    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(InkingMode, self).__init__(**kwargs)
        self.phase = _Phase.CAPTURE
        self.zone = _EditZone.EMPTY_CANVAS
        self.current_node_index = None  #: Node active in the options ui
        self.target_node_index = None  #: Node that's prelit
        self._overlays = {}  # keyed by tdw
        self._reset_nodes()
        self._reset_capture_data()
        self._reset_adjust_data()
        self._task_queue = collections.deque()  # (cb, args, kwargs)
        self._task_queue_runner_id = None
        self._click_info = None   # (button, zone)
        self._current_override_cursor = None
        # Button pressed while drawing
        # Not every device sends button presses, but evdev ones
        # do, and this is used as a workaround for an evdev bug:
        # https://github.com/mypaint/mypaint/issues/223
        self._button_down = None
        self._last_good_raw_pressure = 0.0
        self._last_good_raw_xtilt = 0.0
        self._last_good_raw_ytilt = 0.0
        self._last_good_raw_viewzoom = 0.0
        self._last_good_raw_viewrotation = 0.0
        self._last_good_raw_barrel_rotation = 0.0

    def _reset_nodes(self):
        self.nodes = []  # nodes that met the distance+time criteria

    def _reset_capture_data(self):
        self._last_event_node = None  # node for the last event
        self._last_node_evdata = None  # (xdisp, ydisp, tmilli) for nodes[-1]

    def _reset_adjust_data(self):
        self.zone = _EditZone.EMPTY_CANVAS
        self.current_node_index = None
        self.target_node_index = None
        self._dragged_node_start_pos = None

    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = Overlay(self, tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay

    def _is_active(self):
        for mode in self.doc.modes:
            if mode is self:
                return True
        return False

    def _discard_overlays(self):
        for tdw, overlay in self._overlays.items():
            tdw.display_overlays.remove(overlay)
            tdw.queue_draw()
        self._overlays.clear()

    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
        super(InkingMode, self).enter(doc, **kwds)
        if not self._is_active():
            self._discard_overlays()
        self._ensure_overlay_for_tdw(self.doc.tdw)
        self._arrow_cursor = self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            gui.cursor.Name.ARROW,
        )
        self._crosshair_cursor = self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            gui.cursor.Name.CROSSHAIR_OPEN_PRECISE,
        )

    def leave(self, **kwds):
        """Leaves the mode: called by `ModeStack.pop()` etc."""
        if not self._is_active():
            self._discard_overlays()
        self._stop_task_queue_runner(complete=True)
        super(InkingMode, self).leave(**kwds)  # supercall will commit

    def checkpoint(self, flush=True, **kwargs):
        """Sync pending changes from (and to) the model

        If called with flush==False, this is an override which just
        redraws the pending stroke with the current brush settings and
        color. This is the behavior our testers expect:
        https://github.com/mypaint/mypaint/issues/226

        When this mode is left for another mode (see `leave()`), the
        pending brushwork is committed properly.

        """
        if flush:
            # Commit the pending work normally
            self._start_new_capture_phase(rollback=False)
            super(InkingMode, self).checkpoint(flush=flush, **kwargs)
        else:
            # Queue a re-rendering with any new brush data
            # No supercall
            self._stop_task_queue_runner(complete=False)
            self._queue_draw_buttons()
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()

    def _start_new_capture_phase(self, rollback=False):
        """Let the user capture a new ink stroke"""
        if rollback:
            self._stop_task_queue_runner(complete=False)
            self.brushwork_rollback_all()
        else:
            self._stop_task_queue_runner(complete=True)
            self.brushwork_commit_all()
        self.options_presenter.target = (self, None)
        self._queue_draw_buttons()
        self._queue_redraw_all_nodes()
        self._reset_nodes()
        self._reset_capture_data()
        self._reset_adjust_data()
        self.phase = _Phase.CAPTURE

    ## Raw event handling (prelight & zone selection in adjust phase)

    def button_press_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
        self._update_zone_and_target(tdw, event.x, event.y)
        self._update_current_node_index()
        button = event.button
        if self.phase == _Phase.ADJUST:
            if self.zone in (_EditZone.REJECT_BUTTON,
                             _EditZone.ACCEPT_BUTTON):
                if button == 1 and event.type == Gdk.EventType.BUTTON_PRESS:
                    self._click_info = (button, self.zone)
                    return False
                # FALLTHRU: *do* allow drags to start with other buttons
            elif self.zone == _EditZone.EMPTY_CANVAS and button == 1:
                self._start_new_capture_phase(rollback=False)
                assert self.phase == _Phase.CAPTURE
                # FALLTHRU: *do* start a drag
        elif self.phase == _Phase.CAPTURE:
            # Only allow capturing with the primary mouse button
            if not (button == 1 and event.type == Gdk.EventType.BUTTON_PRESS):
                # Don't bubble up - no drag should be started
                return False
            # XXX Not sure what to do here.
            # XXX Click to append nodes?
            # XXX  but how to stop that and enter the adjust phase?
            # XXX Click to add a 1st & 2nd (=last) node only?
            # XXX  but needs to allow a drag after the 1st one's placed.
            pass
        else:
            raise NotImplementedError("Unrecognized zone %r", self.zone)
        # Update workaround state for evdev dropouts
        self._button_down = button
        self._last_good_raw_pressure = 0.0
        self._last_good_raw_xtilt = 0.0
        self._last_good_raw_ytilt = 0.0
        self._last_good_raw_viewzoom = 0.0
        self._last_good_raw_viewrotation = 0.0
        self._last_good_raw_barrel_rotation = 0.0
        # Supercall: start drags etc
        return super(InkingMode, self).button_press_cb(tdw, event)

    def button_release_cb(self, tdw, event):
        if event.button == self._button_down:
            self._button_down = None
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
        if self.phase == _Phase.ADJUST:
            if self._click_info:
                button0, zone0 = self._click_info
                if event.button == button0:
                    if self.zone == zone0:
                        if zone0 == _EditZone.REJECT_BUTTON:
                            self._start_new_capture_phase(rollback=True)
                            assert self.phase == _Phase.CAPTURE
                        elif zone0 == _EditZone.ACCEPT_BUTTON:
                            self._start_new_capture_phase(rollback=False)
                            assert self.phase == _Phase.CAPTURE
                    self._click_info = None
                    self._update_zone_and_target(tdw, event.x, event.y)
                    self._update_current_node_index()
                    return False
            # (otherwise fall through and end any current drag)
        elif self.phase == _Phase.CAPTURE:
            # XXX Not sure what to do here: see above
            # Update options_presenter when capture phase end
            self.options_presenter.target = (self, None)
        else:
            raise NotImplementedError("Unrecognized zone %r", self.zone)
        # Update workaround state for evdev dropouts
        self._last_good_raw_pressure = 0.0
        self._last_good_raw_xtilt = 0.0
        self._last_good_raw_ytilt = 0.0
        self._last_good_raw_viewzoom = 0.0
        self._last_good_raw_viewrotation = 0.0
        self._last_good_raw_barrel_rotation = 0.0
        # Supercall: stop current drag
        return super(InkingMode, self).button_release_cb(tdw, event)

    def motion_notify_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
        self._update_zone_and_target(tdw, event.x, event.y)
        return super(InkingMode, self).motion_notify_cb(tdw, event)

    def _update_current_node_index(self):
        """Updates current_node_index from target_node_index & redraw"""
        new_index = self.target_node_index
        old_index = self.current_node_index
        if new_index == old_index:
            return
        self.current_node_index = new_index
        self.current_node_changed(new_index)
        self.options_presenter.target = (self, new_index)
        for i in (old_index, new_index):
            if i is not None:
                self._queue_draw_node(i)

    @lib.observable.event
    def current_node_changed(self, index):
        """Event: current_node_index was changed"""

    def _update_zone_and_target(self, tdw, x, y):
        """Update the zone and target node under a cursor position"""
        self._ensure_overlay_for_tdw(tdw)
        new_zone = _EditZone.EMPTY_CANVAS
        if self.phase == _Phase.ADJUST and not self.in_drag:
            new_target_node_index = None
            # Test buttons for hits
            overlay = self._ensure_overlay_for_tdw(tdw)
            hit_dist = gui.style.FLOATING_BUTTON_RADIUS
            button_info = [
                (_EditZone.ACCEPT_BUTTON, overlay.accept_button_pos),
                (_EditZone.REJECT_BUTTON, overlay.reject_button_pos),
            ]
            for btn_zone, btn_pos in button_info:
                if btn_pos is None:
                    continue
                btn_x, btn_y = btn_pos
                d = math.hypot(btn_x - x, btn_y - y)
                if d <= hit_dist:
                    new_target_node_index = None
                    new_zone = btn_zone
                    break
            # Test nodes for a hit, in reverse draw order
            if new_zone == _EditZone.EMPTY_CANVAS:
                hit_dist = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 12
                new_target_node_index = None
                for i, node in reversed(list(enumerate(self.nodes))):
                    node_x, node_y = tdw.model_to_display(node.x, node.y)
                    d = math.hypot(node_x - x, node_y - y)
                    if d > hit_dist:
                        continue
                    new_target_node_index = i
                    new_zone = _EditZone.CONTROL_NODE
                    break
            # Update the prelit node, and draw changes to it
            if new_target_node_index != self.target_node_index:
                if self.target_node_index is not None:
                    self._queue_draw_node(self.target_node_index)
                self.target_node_index = new_target_node_index
                if self.target_node_index is not None:
                    self._queue_draw_node(self.target_node_index)
        # Update the zone, and assume any change implies a button state
        # change as well (for now...)
        if self.zone != new_zone:
            self.zone = new_zone
            self._ensure_overlay_for_tdw(tdw)
            self._queue_draw_buttons()
        # Update the "real" inactive cursor too:
        if not self.in_drag:
            cursor = None
            if self.phase == _Phase.ADJUST:
                if self.zone == _EditZone.CONTROL_NODE:
                    cursor = self._crosshair_cursor
                elif self.zone != _EditZone.EMPTY_CANVAS:  # assume button
                    cursor = self._arrow_cursor
            if cursor is not self._current_override_cursor:
                tdw.set_override_cursor(cursor)
                self._current_override_cursor = cursor

    ## Redraws

    def _queue_draw_buttons(self):
        """Redraws the accept/reject buttons on all known view TDWs"""
        for tdw, overlay in self._overlays.items():
            overlay.update_button_positions()
            positions = (
                overlay.reject_button_pos,
                overlay.accept_button_pos,
            )
            for pos in positions:
                if pos is None:
                    continue
                r = gui.style.FLOATING_BUTTON_ICON_SIZE
                r += max(
                    gui.style.DROP_SHADOW_X_OFFSET,
                    gui.style.DROP_SHADOW_Y_OFFSET,
                )
                r += gui.style.DROP_SHADOW_BLUR
                x, y = pos
                tdw.queue_draw_area(x - r, y - r, (2 * r) + 1, (2 * r) + 1)

    def _queue_draw_node(self, i):
        """Redraws a specific control node on all known view TDWs"""
        for tdw in self._overlays:
            node = self.nodes[i]
            x, y = tdw.model_to_display(node.x, node.y)
            x = math.floor(x)
            y = math.floor(y)
            size = math.ceil(gui.style.DRAGGABLE_POINT_HANDLE_SIZE * 2)
            tdw.queue_draw_area(
                x - size, y - size,
                (size * 2) + 1, (size * 2) + 1,
            )

    def _queue_redraw_all_nodes(self):
        """Redraws all nodes on all known view TDWs"""
        for i in xrange(len(self.nodes)):
            self._queue_draw_node(i)

    def _queue_redraw_curve(self):
        """Redraws the entire curve on all known view TDWs"""
        self._stop_task_queue_runner(complete=False)
        for tdw in self._overlays:
            model = tdw.doc
            if len(self.nodes) < 2:
                continue
            self._queue_task(self.brushwork_rollback, model)
            self._queue_task(
                self.brushwork_begin, model,
                description=_("Inking"),
                abrupt=True,
            )
            interp_state = {"t_abs": self.nodes[0].time}
            for p_1, p0, p1, p2 in gui.drawutils.spline_iter(self.nodes):
                self._queue_task(
                    self._draw_curve_segment,
                    model,
                    p_1, p0, p1, p2,
                    state=interp_state
                )
        self._start_task_queue_runner()

    def _draw_curve_segment(self, model, p_1, p0, p1, p2, state):
        """Draw the curve segment between the middle two points"""
        last_t_abs = state["t_abs"]
        dtime_p0_p1_real = p1[-1] - p0[-1]
        steps_t = dtime_p0_p1_real / self.INTERPOLATION_MAX_SLICE_TIME
        dist_p1_p2 = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        steps_d = dist_p1_p2 / self.INTERPOLATION_MAX_SLICE_DISTANCE
        steps = math.ceil(min(self.INTERPOLATION_MAX_SLICES,
                              max(2, steps_t, steps_d)))
        for i in xrange(int(steps) + 1):
            t = i / steps
            point = gui.drawutils.spline_4p(t, p_1, p0, p1, p2)
            x, y, pressure, xtilt, ytilt, t_abs, viewzoom, viewrotation, barrel_rotation = point
            pressure = lib.helpers.clamp(pressure, 0.0, 1.0)
            xtilt = lib.helpers.clamp(xtilt, -1.0, 1.0)
            ytilt = lib.helpers.clamp(ytilt, -1.0, 1.0)
            t_abs = max(last_t_abs, t_abs)
            dtime = t_abs - last_t_abs
            viewzoom = self.doc.tdw.scale
            viewrotation = self.doc.tdw.rotation
            barrel_rotation = 0.0
            self.stroke_to(
                model, dtime, x, y, pressure, xtilt, ytilt, viewzoom, viewrotation, barrel_rotation,
                auto_split=False,
            )
            last_t_abs = t_abs
        state["t_abs"] = last_t_abs

    def _queue_task(self, callback, *args, **kwargs):
        """Append a task to be done later in an idle cycle"""
        self._task_queue.append((callback, args, kwargs))

    def _start_task_queue_runner(self):
        """Begin processing the task queue, if not already going"""
        if self._task_queue_runner_id is not None:
            return
        idler_id = GLib.idle_add(self._task_queue_runner_cb)
        self._task_queue_runner_id = idler_id

    def _stop_task_queue_runner(self, complete=True):
        """Halts processing of the task queue, and clears it"""
        if self._task_queue_runner_id is None:
            return
        if complete:
            for (callback, args, kwargs) in self._task_queue:
                callback(*args, **kwargs)
        self._task_queue.clear()
        GLib.source_remove(self._task_queue_runner_id)
        self._task_queue_runner_id = None

    def _task_queue_runner_cb(self):
        """Idle runner callback for the task queue"""
        try:
            callback, args, kwargs = self._task_queue.popleft()
        except IndexError:  # queue empty
            self._task_queue_runner_id = None
            return False
        else:
            callback(*args, **kwargs)
            return True

    ## Drag handling (both capture and adjust phases)

    def drag_start_cb(self, tdw, event):
        # A drag started with the space key will bypass the check in
        # the button_press_cb, so we check for them here and cancel
        # those drags for the capture phase.
        if self.phase == _Phase.CAPTURE and not self._button_down:
            self._stop_drag()
            return
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.CAPTURE:
            self._reset_nodes()
            self._reset_capture_data()
            self._reset_adjust_data()
            x, y = self.start_x, self.start_y
            node = self._get_event_data(tdw, event, x, y)
            self.nodes.append(node)
            self._queue_draw_node(0)
            self._last_node_evdata = (x, y, event.time)
            self._last_event_node = node
        elif self.phase == _Phase.ADJUST:
            if self.target_node_index is not None:
                node = self.nodes[self.target_node_index]
                self._dragged_node_start_pos = (node.x, node.y)
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    def drag_update_cb(self, tdw, event, ev_x, ev_y, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.CAPTURE:
            node = self._get_event_data(tdw, event, ev_x, ev_y)
            evdata = (ev_x, ev_y, event.time)
            if not self._last_node_evdata:  # e.g. after an undo while dragging
                append_node = True
            elif evdata == self._last_node_evdata:
                logger.debug(
                    "Capture: ignored successive events "
                    "with identical position and time: %r",
                    evdata,
                )
                append_node = False
            else:
                dx = ev_x - self._last_node_evdata[0]
                dy = ev_y - self._last_node_evdata[1]
                dist = math.hypot(dy, dx)
                dt = event.time - self._last_node_evdata[2]
                max_dist = self.MAX_INTERNODE_DISTANCE_MIDDLE
                if len(self.nodes) < 2:
                    max_dist = self.MAX_INTERNODE_DISTANCE_ENDS
                append_node = (
                    dist > max_dist and
                    dt > self.MAX_INTERNODE_TIME
                )
            if append_node:
                self.nodes.append(node)
                self._queue_draw_node(len(self.nodes) - 1)
                self._queue_redraw_curve()
                self._last_node_evdata = evdata
            self._last_event_node = node
        elif self.phase == _Phase.ADJUST:
            if self._dragged_node_start_pos:
                x0, y0 = self._dragged_node_start_pos
                disp_x, disp_y = tdw.model_to_display(x0, y0)
                disp_x += ev_x - self.start_x
                disp_y += ev_y - self.start_y
                x, y = tdw.display_to_model(disp_x, disp_y)
                self.update_node(self.target_node_index, x=x, y=y)
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.CAPTURE:
            if not self.nodes:
                return
            node = self._last_event_node
            # TODO: maybe rewrite the last node here so it's the right
            # TODO: distance from the end?
            if self.nodes[-1] is not node:
                self.nodes.append(node)
            self._reset_capture_data()
            self._reset_adjust_data()
            if len(self.nodes) > 1:
                self.phase = _Phase.ADJUST
                self._queue_redraw_all_nodes()
                self._queue_redraw_curve()
                self._queue_draw_buttons()
            else:
                self._reset_nodes()
                tdw.queue_draw()
        elif self.phase == _Phase.ADJUST:
            self._dragged_node_start_pos = None
            self._queue_redraw_curve()
            self._queue_draw_buttons()
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    ## Interrogating events

    def _get_event_data(self, tdw, event, x, y):
        xm, ym = tdw.display_to_model(x, y)
        xtilt, ytilt = self._get_event_tilt(tdw, event)
        return _Node(
            x=xm, y=ym,
            pressure=self._get_event_pressure(event),
            xtilt=xtilt, ytilt=ytilt,
            time=(event.time / 1000.0),
            viewzoom = self.doc.tdw.scale,
            viewrotation = self.doc.tdw.rotation,
            barrel_rotation = 0.0,
        )

    def _get_event_pressure(self, event):
        # FIXME: CODE DUPLICATION: copied from freehand.py
        pressure = event.get_axis(Gdk.AxisUse.PRESSURE)
        if pressure is not None:
            if not np.isfinite(pressure):
                pressure = None
            else:
                pressure = lib.helpers.clamp(pressure, 0.0, 1.0)

        if pressure is None:
            pressure = 0.0
            if event.state & Gdk.ModifierType.BUTTON1_MASK:
                pressure = 0.5

        # Workaround for buggy evdev behaviour.
        # Events sometimes get a zero raw pressure reading when the
        # pressure reading has not changed. This results in broken
        # lines. As a workaround, forbid zero pressures if there is a
        # button pressed down, and substitute the last-known good value.
        # Detail: https://github.com/mypaint/mypaint/issues/223
        if self._button_down is not None:
            if pressure == 0.0:
                pressure = self._last_good_raw_pressure
            elif pressure is not None and np.isfinite(pressure):
                self._last_good_raw_pressure = pressure
        return pressure


    def _get_event_tilt(self, tdw, event):
        # FIXME: CODE DUPLICATION: copied from freehand.py
        xtilt = event.get_axis(Gdk.AxisUse.XTILT)
        ytilt = event.get_axis(Gdk.AxisUse.YTILT)
        if xtilt is None or ytilt is None or not np.isfinite(xtilt + ytilt):
            return (0.0, 0.0)

        # Switching from a non-tilt device to a device which reports
        # tilt can cause GDK to return out-of-range tilt values, on X11.
        xtilt = lib.helpers.clamp(xtilt, -1.0, 1.0)
        ytilt = lib.helpers.clamp(ytilt, -1.0, 1.0)

        # Evdev workaround. X and Y tilts suffer from the same
        # problem as pressure for fancier devices.
        if self._button_down is not None:
            if xtilt == 0.0:
                xtilt = self._last_good_raw_xtilt
            else:
                self._last_good_raw_xtilt = xtilt
            if ytilt == 0.0:
                ytilt = self._last_good_raw_ytilt
            else:
                self._last_good_raw_ytilt = ytilt

        if tdw.mirrored:
            xtilt *= -1.0

        return (xtilt, ytilt)

    ## Node editing

    @property
    def options_presenter(self):
        """MVP presenter object for the node editor panel"""
        cls = self.__class__
        if cls._OPTIONS_PRESENTER is None:
            cls._OPTIONS_PRESENTER = OptionsUI()
        return cls._OPTIONS_PRESENTER

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        return self.options_presenter.widget

    def update_node(self, i, **kwargs):
        """Updates properties of a node, and redraws it"""
        changing_pos = bool({"x", "y"}.intersection(kwargs))
        oldnode = self.nodes[i]
        if changing_pos:
            self._queue_draw_node(i)
        self.nodes[i] = oldnode._replace(**kwargs)
        # FIXME: The curve redraw is a bit flickery.
        #   Perhaps dragging to adjust should only draw an
        #   armature during the drag, leaving the redraw to
        #   the stop handler.
        self._queue_redraw_curve()
        if changing_pos:
            self._queue_draw_node(i)

    def get_node_dtime(self, i):
        if not (0 < i < len(self.nodes)):
            return 0.0
        n0 = self.nodes[i - 1]
        n1 = self.nodes[i]
        dtime = n1.time - n0.time
        dtime = max(dtime, self.MIN_INTERNODE_TIME)
        return dtime

    def set_node_dtime(self, i, dtime):
        dtime = max(dtime, self.MIN_INTERNODE_TIME)
        nodes = self.nodes
        if not (0 < i < len(nodes)):
            return
        old_dtime = nodes[i].time - nodes[i - 1].time
        for j in range(i, len(nodes)):
            n = nodes[j]
            new_time = n.time + dtime - old_dtime
            self.update_node(j, time=new_time)

    def can_delete_node(self, i):
        if i is None:
            return False
        return 0 < i < len(self.nodes) - 1

    def delete_node(self, i):
        """Delete a node, and issue redraws & updates"""
        assert self.can_delete_node(i), "Can't delete endpoints"
        # Redraw old locations of things while the node still exists
        self._queue_draw_buttons()
        self._queue_draw_node(i)
        # Remove the node
        self.nodes.pop(i)
        # Limit the current node
        new_cn = self.current_node_index
        if (new_cn is not None) and new_cn >= len(self.nodes):
            new_cn = len(self.nodes) - 2
            self.current_node_index = new_cn
            self.current_node_changed(new_cn)
        # Options panel update
        self.options_presenter.target = (self, new_cn)
        # Issue redraws for the changed on-canvas elements
        self._queue_redraw_curve()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

    def delete_current_node(self):
        if self.can_delete_node(self.current_node_index):
            self.delete_node(self.current_node_index)

            # FIXME: Quick hack,to avoid indexerror(very rare case)
            self.target_node_index = None

    def can_insert_node(self, i):
        if i is None:
            return False
        return 0 <= i < (len(self.nodes) - 1)

    def insert_node(self, i):
        """Insert a node, and issue redraws & updates"""
        assert self.can_insert_node(i), "Can't insert back of the endpoint"
        # Redraw old locations of things while the node still exists
        self._queue_draw_buttons()
        self._queue_draw_node(i)
        # Create the new e
        cn = self.nodes[i]
        nn = self.nodes[i + 1]

        newnode = _Node(
            x=(cn.x + nn.x) / 2.0, y=(cn.y + nn.y) / 2.0,
            pressure=(cn.pressure + nn.pressure) / 2.0,
            xtilt=(cn.xtilt + nn.xtilt) / 2.0,
            ytilt=(cn.ytilt + nn.ytilt) / 2.0,
            time=(cn.time + nn.time) / 2.0,
            viewzoom=(cn.viewzoom + nn.viewzoom) / 2.0,
            viewrotation=(cn.viewrotation + nn.viewrotation) / 2.0,
            barrel_rotation=(cn.barrel_rotation + nn.barrel_rotation) / 2.0,
        )
        self.nodes.insert(i + 1, newnode)

        # Issue redraws for the changed on-canvas elements
        self._queue_redraw_curve()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

    def insert_current_node(self):
        if self.can_insert_node(self.current_node_index):
            self.insert_node(self.current_node_index)

    def _simplify_nodes(self, tolerance):
        """Internal method of simplify nodes.

        Algorithm: Reumann-Witkam.

        """
        i = 0
        oldcnt = len(self.nodes)
        while i < len(self.nodes) - 2:
            try:
                vsx = self.nodes[i + 1].x - self.nodes[i].x
                vsy = self.nodes[i + 1].y - self.nodes[i].y
                ss = math.sqrt((vsx * vsx) + (vsy * vsy))
                nsx = vsx / ss
                nsy = vsy / ss
                while (i + 2) < len(self.nodes):
                    vex = self.nodes[i + 2].x - self.nodes[i].x
                    vey = self.nodes[i + 2].y - self.nodes[i].y
                    es = math.sqrt((vex * vex) + (vey * vey))
                    px = nsx * es
                    py = nsy * es
                    dp = (px * (vex / es) + py * (vey / es)) / es
                    hx = (vex * dp) - px
                    hy = (vey * dp) - py

                    if math.sqrt((hx * hx) + (hy * hy)) < tolerance:
                        self.nodes.pop(i + 1)
                    else:
                        break

            except ValueError:
                pass
            except ZeroDivisionError:
                pass
            finally:
                i += 1

        return oldcnt - len(self.nodes)

    def _cull_nodes(self):
        """Internal method of cull nodes."""
        curcnt = len(self.nodes)
        lastnode = self.nodes[-1]
        self.nodes = self.nodes[:-1:2]
        self.nodes.append(lastnode)
        return curcnt - len(self.nodes)

    def _nodes_deletion_operation(self, func, args):
        """Internal method for delete-related operation of multiple nodes."""
        # To ensure redraw entire overlay,avoiding glitches.
        self._queue_redraw_curve()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

        if func(*args) > 0:

            new_cn = self.current_node_index
            if (new_cn is not None) and new_cn >= len(self.nodes):
                new_cn = len(self.nodes) - 2
                self.current_node_index = new_cn
                self.current_node_changed(new_cn)
                self.options_presenter.target = (self, new_cn)

            # FIXME: Quick hack,to avoid indexerror
            self.target_node_index = None

            # Issue redraws for the changed on-canvas elements
            self._queue_redraw_curve()
            self._queue_redraw_all_nodes()
            self._queue_draw_buttons()

    def simplify_nodes(self):
        """User interface method of simplify nodes."""
        # For now, parameter is fixed value.
        # tolerance is 8, in model coords.
        self._nodes_deletion_operation(self._simplify_nodes, (8,))

    def cull_nodes(self):
        """User interface method of cull nodes."""
        self._nodes_deletion_operation(self._cull_nodes, ())


class Overlay (gui.overlays.Overlay):
    """Overlay for an InkingMode's adjustable points"""

    def __init__(self, inkmode, tdw):
        super(Overlay, self).__init__()
        self._inkmode = weakref.proxy(inkmode)
        self._tdw = weakref.proxy(tdw)
        self._button_pixbuf_cache = {}
        self.accept_button_pos = None
        self.reject_button_pos = None

    def update_button_positions(self):
        """Recalculates the positions of the mode's buttons."""
        nodes = self._inkmode.nodes
        num_nodes = len(nodes)
        if num_nodes == 0:
            self.reject_button_pos = None
            self.accept_button_pos = None
            return

        button_radius = gui.style.FLOATING_BUTTON_RADIUS
        margin = 1.5 * button_radius
        alloc = self._tdw.get_allocation()
        view_x0, view_y0 = alloc.x, alloc.y
        view_x1, view_y1 = (view_x0 + alloc.width), (view_y0 + alloc.height)

        # Force-directed layout: "wandering nodes" for the buttons'
        # eventual positions, moving around a constellation of "fixed"
        # points corresponding to the nodes the user manipulates.
        fixed = []

        for i, node in enumerate(nodes):
            x, y = self._tdw.model_to_display(node.x, node.y)
            fixed.append(_LayoutNode(x, y))

        # The reject and accept buttons are connected to different nodes
        # in the stroke by virtual springs.
        stroke_end_i = len(fixed) - 1
        stroke_start_i = 0
        stroke_last_quarter_i = int(stroke_end_i * 3.0 // 4.0)
        assert stroke_last_quarter_i < stroke_end_i
        reject_anchor_i = stroke_start_i
        accept_anchor_i = stroke_end_i

        # Classify the stroke direction as a unit vector
        stroke_tail = (
            fixed[stroke_end_i].x - fixed[stroke_last_quarter_i].x,
            fixed[stroke_end_i].y - fixed[stroke_last_quarter_i].y,
        )
        stroke_tail_len = math.hypot(*stroke_tail)
        if stroke_tail_len <= 0:
            stroke_tail = (0., 1.)
        else:
            stroke_tail = tuple(c / stroke_tail_len for c in stroke_tail)

        # Initial positions.
        accept_button = _LayoutNode(
            fixed[accept_anchor_i].x + stroke_tail[0] * margin,
            fixed[accept_anchor_i].y + stroke_tail[1] * margin,
        )
        reject_button = _LayoutNode(
            fixed[reject_anchor_i].x - stroke_tail[0] * margin,
            fixed[reject_anchor_i].y - stroke_tail[1] * margin,
        )

        # Constraint boxes. They mustn't share corners.
        # Natural hand strokes are often downwards,
        # so let the reject button to go above the accept button.
        reject_button_bbox = (
            view_x0 + margin, view_x1 - margin,
            view_y0 + margin, view_y1 - (2.666 * margin),
        )
        accept_button_bbox = (
            view_x0 + margin, view_x1 - margin,
            view_y0 + (2.666 * margin), view_y1 - margin,
        )

        # Force-update constants
        k_repel = -25.0
        k_attract = 0.05

        # Let the buttons bounce around until they've settled.
        for iter_i in xrange(100):
            accept_button \
                .add_forces_inverse_square(fixed, k=k_repel) \
                .add_forces_inverse_square([reject_button], k=k_repel) \
                .add_forces_linear([fixed[accept_anchor_i]], k=k_attract)
            reject_button \
                .add_forces_inverse_square(fixed, k=k_repel) \
                .add_forces_inverse_square([accept_button], k=k_repel) \
                .add_forces_linear([fixed[reject_anchor_i]], k=k_attract)
            reject_button \
                .update_position() \
                .constrain_position(*reject_button_bbox)
            accept_button \
                .update_position() \
                .constrain_position(*accept_button_bbox)
            settled = [(p.speed < 0.5) for p in [accept_button, reject_button]]
            if all(settled):
                break
        self.accept_button_pos = accept_button.x, accept_button.y
        self.reject_button_pos = reject_button.x, reject_button.y

    def _get_button_pixbuf(self, name):
        """Loads the pixbuf corresponding to a button name (cached)"""
        cache = self._button_pixbuf_cache
        pixbuf = cache.get(name)
        if not pixbuf:
            pixbuf = gui.drawutils.load_symbolic_icon(
                icon_name=name,
                size=gui.style.FLOATING_BUTTON_ICON_SIZE,
                fg=(0, 0, 0, 1),
            )
            cache[name] = pixbuf
        return pixbuf

    def _get_onscreen_nodes(self):
        """Iterates across only the on-screen nodes."""
        mode = self._inkmode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        for i, node in enumerate(mode.nodes):
            x, y = self._tdw.model_to_display(node.x, node.y)
            node_on_screen = (
                x > alloc.x - (radius * 2) and
                y > alloc.y - (radius * 2) and
                x < alloc.x + alloc.width + (radius * 2) and
                y < alloc.y + alloc.height + (radius * 2)
            )
            if node_on_screen:
                yield (i, node, x, y)

    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        # Control nodes
        mode = self._inkmode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        for i, node, x, y in self._get_onscreen_nodes():
            color = gui.style.EDITABLE_ITEM_COLOR
            if mode.phase == _Phase.ADJUST:
                if i == mode.current_node_index:
                    color = gui.style.ACTIVE_ITEM_COLOR
                elif i == mode.target_node_index:
                    color = gui.style.PRELIT_ITEM_COLOR
            gui.drawutils.render_round_floating_color_chip(
                cr=cr, x=x, y=y,
                color=color,
                radius=radius,
            )
        # Buttons
        if mode.phase == _Phase.ADJUST and not mode.in_drag:
            self.update_button_positions()
            radius = gui.style.FLOATING_BUTTON_RADIUS
            button_info = [
                (
                    "mypaint-ok-symbolic",
                    self.accept_button_pos,
                    _EditZone.ACCEPT_BUTTON,
                ),
                (
                    "mypaint-trash-symbolic",
                    self.reject_button_pos,
                    _EditZone.REJECT_BUTTON,
                ),
            ]
            for icon_name, pos, zone in button_info:
                if pos is None:
                    continue
                x, y = pos
                if mode.zone == zone:
                    color = gui.style.ACTIVE_ITEM_COLOR
                else:
                    color = gui.style.EDITABLE_ITEM_COLOR
                icon_pixbuf = self._get_button_pixbuf(icon_name)
                gui.drawutils.render_round_floating_button(
                    cr=cr, x=x, y=y,
                    color=color,
                    pixbuf=icon_pixbuf,
                    radius=radius,
                )


class _LayoutNode (object):
    """Vertex/point for the button layout algorithm."""

    def __init__(self, x, y, force=(0., 0.), velocity=(0., 0.)):
        self.x = float(x)
        self.y = float(y)
        self.force = tuple(float(c) for c in force[:2])
        self.velocity = tuple(float(c) for c in velocity[:2])

    def __repr__(self):
        return "_LayoutNode(x=%r, y=%r, force=%r, velocity=%r)" % (
            self.x, self.y, self.force, self.velocity,
        )

    @property
    def pos(self):
        return (self.x, self.y)

    @property
    def speed(self):
        return math.hypot(*self.velocity)

    def add_forces_inverse_square(self, others, k=20.0):
        """Adds inverse-square components to the effective force.

        :param [_LayoutNode] others: _LayoutNodes affecting this one
        :param float k: scaling factor
        :returns: self

        The forces applied are proportional to k, and inversely
        proportional to the square of the distances. Examples:
        gravity, electrostatic repulsion.

        With the default arguments, the added force components are
        attractive. Use negative k to simulate repulsive forces.

        """
        fx, fy = self.force
        for other in others:
            if other is self:
                continue
            rsquared = (self.x - other.x) ** 2 + (self.y - other.y) ** 2
            if rsquared == 0:
                continue
            else:
                fx += k * (other.x - self.x) / rsquared
                fy += k * (other.y - self.y) / rsquared
        self.force = (fx, fy)
        return self

    def add_forces_linear(self, others, k=0.05):
        """Adds linear components to the total effective force.

        :param [_LayoutNode] others: _LayoutNodes affecting this one
        :param float k: scaling factor
        :returns: self

        The forces applied are proportional to k, and to the distance.
        Example: springs.

        With the default arguments, the added force components are
        attractive. Use negative k to simulate repulsive forces.

        """
        fx, fy = self.force
        for other in others:
            if other is self:
                continue
            fx += k * (other.x - self.x)
            fy += k * (other.y - self.y)
        self.force = (fx, fy)
        return self

    def update_position(self, damping=0.85):
        """Updates velocity & position from total force, then resets it.

        :param float damping: Damping factor for velocity/speed.
        :returns: self

        Calling this method should be done just once per iteration,
        after all the force components have been added in. The effective
        force is reset to zero after calling this method.

        """
        fx, fy = self.force
        self.force = (0., 0.)
        vx, vy = self.velocity
        vx = (vx + fx) * damping
        vy = (vy + fy) * damping
        self.velocity = (vx, vy)
        self.x += vx
        self.y += vy
        return self

    def constrain_position(self, x0, x1, y0, y1):
        vx, vy = self.velocity
        if self.x < x0:
            self.x = x0
            vx = 0
        elif self.x > x1:
            self.x = x1
            vx = 0
        if self.y < y0:
            self.y = y0
            vy = 0
        elif self.y > y1:
            self.y = y1
            vy = 0
        self.velocity = (vx, vy)
        return self


class OptionsUI (gui.mvp.BuiltUIPresenter, object):
    """Presents UI for directly editing point values etc."""

    def __init__(self):
        super(OptionsUI, self).__init__()
        self._target = (None, None)

    def init_view(self):
        self.view.point_values_grid.set_sensitive(False)
        self.view.insert_point_button.set_sensitive(False)
        self.view.delete_point_button.set_sensitive(False)
        self.view.simplify_points_button.set_sensitive(False)
        self.view.cull_points_button.set_sensitive(False)

    @property
    def widget(self):
        return self.view.options_grid

    @property
    def target(self):
        """The active mode and its current node index

        :returns: a pair of the form (inkmode, node_idx)
        :rtype: tuple

        Updating this pair via the property also updates the options UI
        view, shortly afterwards. The target mode must be an InkingTool
        instance.

        """
        mode_ref, node_idx = self._target
        mode = None
        if mode_ref is not None:
            mode = mode_ref()
        return (mode, node_idx)

    @target.setter
    def target(self, targ):
        inkmode, cn_idx = targ
        inkmode_ref = None
        if inkmode:
            inkmode_ref = weakref.ref(inkmode)
        self._target = (inkmode_ref, cn_idx)

        GLib.idle_add(self._update_ui_for_current_target)

    @gui.mvp.view_updater(default=False)
    def _update_ui_for_current_target(self):
        (inkmode, cn_idx) = self.target
        if (cn_idx is not None) and (0 <= cn_idx < len(inkmode.nodes)):
            cn = inkmode.nodes[cn_idx]
            self.view.pressure_adj.set_value(cn.pressure)
            self.view.xtilt_adj.set_value(cn.xtilt)
            self.view.ytilt_adj.set_value(cn.ytilt)
            if cn_idx > 0:
                sensitive = True
                dtime = inkmode.get_node_dtime(cn_idx)
            else:
                sensitive = False
                dtime = 0.0
            for w in (self.view.dtime_scale, self.view.dtime_label):
                w.set_sensitive(sensitive)
            self.view.dtime_adj.set_value(dtime)
            self.view.point_values_grid.set_sensitive(True)
        else:
            self.view.point_values_grid.set_sensitive(False)
        button_sensitivities = [
            (self.view.insert_point_button, inkmode.can_insert_node(cn_idx)),
            (self.view.delete_point_button, inkmode.can_delete_node(cn_idx)),
            (self.view.simplify_points_button, (len(inkmode.nodes) > 3)),
            (self.view.cull_points_button, (len(inkmode.nodes) > 2)),
        ]
        for button, sens in button_sensitivities:
            button.set_sensitive(sens)
        return False

    @gui.mvp.model_updater
    def _pressure_adj_value_changed_cb(self, adj):
        inkmode, node_idx = self.target
        inkmode.update_node(node_idx, pressure=float(adj.get_value()))

    @gui.mvp.model_updater
    def _dtime_adj_value_changed_cb(self, adj):
        inkmode, node_idx = self.target
        inkmode.set_node_dtime(node_idx, adj.get_value())

    @gui.mvp.model_updater
    def _xtilt_adj_value_changed_cb(self, adj):
        value = float(adj.get_value())
        inkmode, node_idx = self.target
        inkmode.update_node(node_idx, xtilt=value)

    @gui.mvp.model_updater
    def _ytilt_adj_value_changed_cb(self, adj):
        value = float(adj.get_value())
        inkmode, node_idx = self.target
        inkmode.update_node(node_idx, ytilt=value)

    @gui.mvp.model_updater
    def _insert_point_button_clicked_cb(self, button):
        inkmode, node_idx = self.target
        if inkmode.can_insert_node(node_idx):
            inkmode.insert_node(node_idx)

    @gui.mvp.model_updater
    def _delete_point_button_clicked_cb(self, button):
        inkmode, node_idx = self.target
        if inkmode.can_delete_node(node_idx):
            inkmode.delete_node(node_idx)

    @gui.mvp.model_updater
    def _simplify_points_button_clicked_cb(self, button):
        inkmode, node_idx = self.target
        if len(inkmode.nodes) > 3:
            inkmode.simplify_nodes()

    @gui.mvp.model_updater
    def _cull_points_button_clicked_cb(self, button):
        inkmode, node_idx = self.target
        if len(inkmode.nodes) > 2:
            inkmode.cull_nodes()
