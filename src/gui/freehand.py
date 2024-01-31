# This file is part of MyPaint.
# Copyright (C) 2008-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2013-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Freehand drawing modes"""

## Imports

from __future__ import division, print_function
import math
import logging
from collections import deque
from gettext import gettext as _

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib
import numpy as np

from gui.tileddrawwidget import TiledDrawWidget

from lib.helpers import clamp
import gui.mode
from .drawutils import spline_4p
from .sliderwidget import InputSlider

logger = logging.getLogger(__name__)


## Class defs

class FreehandMode (gui.mode.BrushworkModeMixin,
                    gui.mode.ScrollableModeMixin,
                    gui.mode.InteractionMode):
    """Freehand drawing mode

    To improve application responsiveness, this mode uses an internal
    queue for capturing input data. The raw motion data from the stylus
    is queued; an idle routine then tidies up this data and feeds it
    onward. The presence of an input capture queue means that long
    queued strokes can be terminated by entering a new mode, or by
    pressing Escape.

    This is the default mode in MyPaint.

    """

    ## Class constants & instance defaults

    ACTION_NAME = 'FreehandMode'
    permitted_switch_actions = set()   # Any action is permitted

    _OPTIONS_WIDGET = None

    IS_LIVE_UPDATEABLE = True

    # Motion queue processing (raw data capture)

    # This controls processing of an internal queue of event data such
    # as the x and y coords, pressure and tilt prior to the strokes
    # rendering.

    MOTION_QUEUE_PRIORITY = GLib.PRIORITY_DEFAULT_IDLE

    # The Right Thing To Do generally is to spend as little time as
    # possible directly handling each event received. Disconnecting
    # stroke rendering from event processing buys the user the ability
    # to quit out of a slowly/laggily rendering stroke if desired.

    ## Initialization

    def __init__(self, ignore_modifiers=True, **args):
        # Ignore the additional arg that flip actions feed us
        super(FreehandMode, self).__init__(**args)
        self._cursor_hidden_tdws = set()
        self._cursor_hidden = None

    ## Metadata

    pointer_behavior = gui.mode.Behavior.PAINT_FREEHAND
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    @classmethod
    def get_name(cls):
        return _(u"Freehand Drawing")

    def get_usage(self):
        return _(u"Paint free-form brush strokes")

    ## Per-TDW drawing state

    class _DrawingState (object):
        """Per-canvas drawing state

        Various kinds of queue for raw data capture or interpolation of
        pressure and tilt.
        """

        def __init__(self):
            object.__init__(self)

            self.last_event_had_pressure = False

            # Raw data which was delivered with an identical timestamp
            # to the previous one.  Happens on Windows due to differing
            # clock granularities (at least, under GTK2).
            self._zero_dtime_motions = []

            # Motion Queue

            # Combined, cleaned-up motion data queued ready for
            # interpolation of missing pressures and tilts, then
            # subsequent rendering. Using a queue makes rendering
            # independent of data gathering.
            self.motion_queue = deque()
            self.motion_processing_cbid = None
            self._last_queued_event_time = 0

            # Queued Event Handling

            # Pressure and tilt interpolation for events which
            # don't have pressure or tilt data.
            self.interp = PressureAndTiltInterpolator()

            # Time of the last-processed event
            self.last_handled_event_time = 0

            # Debugging: number of events processed each second,
            # average times.
            self.avgtime = None

            # Button pressed while drawing
            # Not every device sends button presses, but evdev ones
            # do, and this is used as a workaround for an evdev bug:
            # https://github.com/mypaint/mypaint/issues/29
            self.button_down = None
            self.last_good_raw_pressure = 0.0
            self.last_good_raw_xtilt = 0.0
            self.last_good_raw_ytilt = 0.0
            self.last_good_raw_viewzoom = 0.0
            self.last_good_raw_viewrotation = 0.0
            self.last_good_raw_barrel_rotation = 0.0

        def queue_motion(self, event_data):
            """Append one raw motion event to the motion queue

            :param event_data: Extracted data from an event.
            :type event_data: tuple

            Events are tuples of the form ``(time, x, y, pressure,
            xtilt, ytilt, viewzoom, viewrotation, barrel_rotation)``.
            Times are in milliseconds, and are expressed as ints. ``x``
            and ``y`` are ordinary Python floats, and refer to model
            coordinates. The pressure and tilt values have the meaning
            assigned to them by GDK; if ```pressure`` is None, pressure
            and tilt values will be interpolated from surrounding
            defined values.

            Zero-dtime events are detected and cleaned up here.

            """

            (time, x, y, pressure,
             xtilt, ytilt, viewzoom,
             viewrotation, barrel_rotation) = event_data
            if time < self._last_queued_event_time:
                logger.warning('Time is running backwards! Corrected.')
                time = self._last_queued_event_time

            if time == self._last_queued_event_time:
                # On Windows, GTK timestamps have a resolution around
                # 15ms, but tablet events arrive every 8ms.
                # https://gna.org/bugs/index.php?16569
                zdata = (x, y, pressure, xtilt, ytilt, viewzoom,
                         viewrotation, barrel_rotation)
                self._zero_dtime_motions.append(zdata)
            else:
                # Queue any previous events that had identical
                # timestamps, linearly interpolating their times.
                if self._zero_dtime_motions:
                    dtime = time - self._last_queued_event_time
                    if dtime > 100:
                        # Really old events; don't associate them with
                        # the new one.
                        zt = time - 100.0
                        interval = 100.0
                    else:
                        zt = self._last_queued_event_time
                        interval = dtime
                    step = interval / (len(self._zero_dtime_motions) + 1)

                    for (zx, zy, zp, zxt,
                         zyt, zvz, zvr, zbr) in self._zero_dtime_motions:
                        zt += step
                        zevent_data = (zt, zx, zy, zp, zxt, zyt, zvz, zvr, zbr)
                        self.motion_queue.append(zevent_data)
                    # Reset the backlog buffer
                    self._zero_dtime_motions = []
                # Queue this event too
                self.motion_queue.append(event_data)
                # Update the timestamp used above
                self._last_queued_event_time = time

        def next_processing_events(self):
            """Fetches zero or more events to process from the queue"""
            if len(self.motion_queue) > 0:
                event = self.motion_queue.popleft()
                for ievent in self.interp.feed(*event):
                    yield ievent

    def _reset_drawing_state(self):
        """Resets all per-TDW drawing state"""
        self._drawing_state = {}

    def _get_drawing_state(self, tdw):
        drawstate = self._drawing_state.get(tdw, None)
        if drawstate is None:
            drawstate = self._DrawingState()
            self._drawing_state[tdw] = drawstate
        return drawstate

    ## Mode stack & current mode

    def enter(self, doc, **kwds):
        """Enter freehand mode"""
        super(FreehandMode, self).enter(doc, **kwds)
        self._drawing_state = {}
        self._reset_drawing_state()
        self._debug = (logger.getEffectiveLevel() == logging.DEBUG)

    def leave(self, **kwds):
        """Leave freehand mode"""
        self._reset_drawing_state()
        self._reinstate_drawing_cursor(tdw=None)
        super(FreehandMode, self).leave(**kwds)

    ## Special cursor state while there's pressure

    def _hide_drawing_cursor(self, tdw):
        """Hide the cursor while painting, if configured to.

        :param tdw: Canvas widget to hide the cursor on.
        :type tdw: TiledDrawWidget

        """
        if tdw in self._cursor_hidden_tdws:
            return
        if not tdw.app:
            return
        if not tdw.app.preferences.get("ui.hide_cursor_while_painting"):
            return

        if self._cursor_hidden is None:
            window = tdw.get_window()
            cursor = Gdk.Cursor.new_for_display(
                window.get_display(), Gdk.CursorType.BLANK_CURSOR)
            self._cursor_hidden = cursor

        tdw.set_override_cursor(self._cursor_hidden)
        self._cursor_hidden_tdws.add(tdw)

    def _reinstate_drawing_cursor(self, tdw=None):
        """Un-hide any hidden cursors.

        :param tdw: Canvas widget to reset. None means all affected.
        :type tdw: TiledDrawWidget

        """
        if tdw is None:
            for tdw in self._cursor_hidden_tdws:
                tdw.set_override_cursor(None)
            self._cursor_hidden_tdws.clear()
        elif tdw in self._cursor_hidden_tdws:
            tdw.set_override_cursor(None)
            self._cursor_hidden_tdws.remove(tdw)

    ## Input handlers

    def button_press_cb(self, tdw, event):
        result = False
        current_layer = tdw.doc.layer_stack.current
        if (current_layer.get_paintable() and event.button == 1
                and event.type == Gdk.EventType.BUTTON_PRESS):
            # Single button press
            # Stroke started, notify observers
            self.doc.input_stroke_started(event)
            # Mouse button pressed (while painting without pressure
            # information)
            drawstate = self._get_drawing_state(tdw)
            if not drawstate.last_event_had_pressure:
                # For the mouse we don't get a motion event for
                # "pressure" changes, so we simulate it. (Note: we can't
                # use the event's button state because it carries the
                # old state.)
                self.motion_notify_cb(tdw, event,
                                      fakepressure=tdw.app.fakepressure)

            drawstate.button_down = event.button
            drawstate.last_good_raw_pressure = 0.0
            drawstate.last_good_raw_xtilt = 0.0
            drawstate.last_good_raw_ytilt = 0.0
            drawstate.last_good_raw_viewzoom = 0.0
            drawstate.last_good_raw_viewrotation = 0.0
            drawstate.last_good_raw_barrel_rotation = 0.0

            # Hide the cursor if configured to
            self._hide_drawing_cursor(tdw)

            result = True
        return (super(FreehandMode, self).button_press_cb(tdw, event)
                or result)

    def button_release_cb(self, tdw, event):
        result = False
        current_layer = tdw.doc.layer_stack.current
        if current_layer.get_paintable() and event.button == 1:
            # See comment above in button_press_cb.
            drawstate = self._get_drawing_state(tdw)
            if not drawstate.last_event_had_pressure:
                self.motion_notify_cb(tdw, event, fakepressure=0.0)
            # Notify observers after processing the event
            self.doc.input_stroke_ended(event)

            drawstate.button_down = None
            drawstate.last_good_raw_pressure = 0.0
            drawstate.last_good_raw_xtilt = 0.0
            drawstate.last_good_raw_ytilt = 0.0

            # Reinstate the normal cursor if it was hidden
            self._reinstate_drawing_cursor(tdw)

            result = True
        return (super(FreehandMode, self).button_release_cb(tdw, event)
                or result)

    def motion_notify_cb(self, tdw, event, fakepressure=None):
        """Motion event handler: queues raw input and returns

        :param tdw: The TiledDrawWidget receiving the event
        :param event: the MotionNotify event being handled
        :param fakepressure: fake pressure to use if no real pressure

        Fake pressure is passed with faked motion events, e.g.
        button-press and button-release handlers for mouse events.

        """

        # Do nothing if painting is inactivated
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

        # If the device has changed and the last pressure value from the
        # previous device is not equal to 0.0, this can leave a visible
        # stroke on the layer even if the 'new' device is not pressed on
        # the tablet and has a pressure axis == 0.0.  Resetting the brush
        # when the device changes fixes this issue, but there may be a
        # much more elegant solution that only resets the brush on this
        # edge-case.
        same_device = True
        if tdw.app is not None:
            device = event.get_source_device()
            same_device = tdw.app.device_monitor.device_used(device)
            if not same_device:
                tdw.doc.brush.reset()

        # Extract the raw readings for this event
        x = event.x
        y = event.y
        time = event.time
        pressure = event.get_axis(Gdk.AxisUse.PRESSURE)
        xtilt = event.get_axis(Gdk.AxisUse.XTILT)
        ytilt = event.get_axis(Gdk.AxisUse.YTILT)
        viewzoom = tdw.scale
        viewrotation = tdw.rotation
        barrel_rotation = event.get_axis(Gdk.AxisUse.WHEEL)
        state = event.state

        # Workaround for buggy evdev behaviour.
        # Events sometimes get a zero raw pressure reading when the
        # pressure reading has not changed. This results in broken
        # lines. As a workaround, forbid zero pressures if there is a
        # button pressed down, and substitute the last-known good value.
        # Detail: https://github.com/mypaint/mypaint/issues/29
        drawstate = self._get_drawing_state(tdw)
        if drawstate.button_down is not None:
            if pressure == 0.0:
                pressure = drawstate.last_good_raw_pressure
            elif pressure is not None and np.isfinite(pressure):
                drawstate.last_good_raw_pressure = pressure

        # Ensure each event has a defined pressure
        if pressure is not None:
            # Using the reported pressure. Apply some sanity checks
            if not np.isfinite(pressure):
                # infinity/nan: use button state (instead of clamping in
                # brush.hpp) https://gna.org/bugs/?14709
                pressure = None
            else:
                pressure = clamp(pressure, 0.0, 1.0)
            drawstate.last_event_had_pressure = True

        # Fake the pressure if we have none, or if infinity was reported
        if pressure is None:
            if fakepressure is not None:
                pressure = clamp(fakepressure, 0.0, 1.0)
            else:
                pressure = (
                    (state & Gdk.ModifierType.BUTTON1_MASK) and
                    tdw.app.fakepressure or 0.0)
            drawstate.last_event_had_pressure = False

        # Check whether tilt is present.  For some tablets without
        # tilt support GTK reports a tilt axis with value nan, instead
        # of None.  https://gna.org/bugs/?17084
        if xtilt is None or ytilt is None or not np.isfinite(xtilt + ytilt):
            xtilt = 0.0
            ytilt = 0.0

        # Switching from a non-tilt device to a device which reports
        # tilt can cause GDK to return out-of-range tilt values, on X11.
        xtilt = clamp(xtilt, -1.0, 1.0)
        ytilt = clamp(ytilt, -1.0, 1.0)

        tilt_ascension = 0.5 * math.atan2(-xtilt, ytilt) / math.pi

        # Offset barrel rotation if wanted
        # This could be used to correct for different devices,
        # Left vs Right handed, etc.
        b_offset = tdw.app.preferences.get("input.barrel_rotation_offset")
        if (barrel_rotation is not None):
            barrel_rotation = (barrel_rotation + b_offset) % 1.0

        # barrel_rotation is likely affected by ascension (a bug?)
        # lets compensate but allow disabling
        if (barrel_rotation is not None and tdw.app.preferences.get(
           "input.barrel_rotation_subtract_ascension")):
            barrel_rotation = (barrel_rotation - tilt_ascension) % 1.0

        # If WHEEL is missing (barrel_rotation)
        # Use the fakerotation controller to allow keyboard control
        # We can't trust None so also look at preference
        if (barrel_rotation is None
           or not tdw.app.preferences.get("input.use_barrel_rotation")):
            barrel_rotation = (tdw.app.fakerotation + b_offset) % 1.0

        # Evdev workaround. X and Y tilts suffer from the same
        # problem as pressure for fancier devices.
        if drawstate.button_down is not None:
            if xtilt == 0.0:
                xtilt = drawstate.last_good_raw_xtilt
            else:
                drawstate.last_good_raw_xtilt = xtilt
            if ytilt == 0.0:
                ytilt = drawstate.last_good_raw_ytilt
            else:
                drawstate.last_good_raw_ytilt = ytilt

        if tdw.mirrored:
            xtilt *= -1.0
            barrel_rotation *= -1.0

        # Apply pressure mapping if we're running as part of a full
        # MyPaint application (and if there's one defined).
        if tdw.app is not None and tdw.app.pressure_mapping:
            pressure = tdw.app.pressure_mapping(pressure)

        # Apply any configured while-drawing cursor
        if pressure > 0:
            self._hide_drawing_cursor(tdw)
        else:
            self._reinstate_drawing_cursor(tdw)

        # Queue this event
        x, y = tdw.display_to_model(x, y)

        event_data = (time, x, y, pressure,
                      xtilt, ytilt, viewzoom,
                      viewrotation, barrel_rotation)
        drawstate.queue_motion(event_data)
        # Start the motion event processor, if it isn't already running
        if not drawstate.motion_processing_cbid:
            cbid = GLib.idle_add(
                self._motion_queue_idle_cb,
                tdw,
                priority = self.MOTION_QUEUE_PRIORITY,
            )
            drawstate.motion_processing_cbid = cbid

    ## Motion queue processing

    def _motion_queue_idle_cb(self, tdw):
        """Idle callback; processes each queued event"""
        drawstate = self._get_drawing_state(tdw)
        # Stop if asked to stop
        if drawstate.motion_processing_cbid is None:
            drawstate.motion_queue = deque()
            return False
        # Forward one or more motion events to the canvas
        for event in drawstate.next_processing_events():
            self._process_queued_event(tdw, event)
        # Stop if the queue is now empty
        if len(drawstate.motion_queue) == 0:
            drawstate.motion_processing_cbid = None
            return False
        # Otherwise, continue being invoked
        return True

    def _process_queued_event(self, tdw, event_data):
        """Process one motion event from the motion queue"""
        drawstate = self._get_drawing_state(tdw)
        (time, x, y, pressure, xtilt, ytilt, viewzoom,
         viewrotation, barrel_rotation) = event_data
        model = tdw.doc

        # Calculate time delta for the brush engine
        last_event_time = drawstate.last_handled_event_time
        drawstate.last_handled_event_time = time
        if not last_event_time:
            return
        dtime = (time - last_event_time) / 1000.0
        if self._debug:
            cavg = drawstate.avgtime
            if cavg is not None:
                tavg, nevents = cavg
                nevents += 1
                tavg += (dtime - tavg) / nevents
            else:
                tavg = dtime
                nevents = 1
            if ((nevents * tavg) > 1.0) and nevents > 20:
                logger.debug("Processing at %d events/s (t_avg=%0.3fs)",
                             nevents, tavg)
                drawstate.avgtime = None
            else:
                drawstate.avgtime = (tavg, nevents)

        current_layer = model._layers.current
        if not current_layer.get_paintable():
            return

        # Feed data to the brush engine.  Pressure and tilt cleanup
        # needs to be done here to catch all forwarded data after the
        # earlier interpolations. The interpolation method used for
        # filling in missing axis data is known to generate
        # OverflowErrors for legitimate but pathological input streams.
        # https://github.com/mypaint/mypaint/issues/344

        pressure = clamp(pressure, 0.0, 1.0)
        xtilt = clamp(xtilt, -1.0, 1.0)
        ytilt = clamp(ytilt, -1.0, 1.0)
        self.stroke_to(model, dtime, x, y, pressure,
                       xtilt, ytilt, viewzoom,
                       viewrotation, barrel_rotation)

        # Update the TDW's idea of where we last painted
        # FIXME: this should live in the model, not the view
        if pressure:
            tdw.set_last_painting_pos((x, y))

    ## Mode options

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = FreehandOptionsWidget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET


class FreehandOptionsWidget (gui.mode.PaintingModeOptionsWidgetBase):
    """Configuration widget for freehand mode"""

    def init_specialized_widgets(self, row):
        cname = "slow_tracking"
        label = Gtk.Label()
        # TRANSLATORS: Short alias for "Slow position tracking". This is
        # TRANSLATORS: used on the options panel.
        label.set_text(_("Smooth:"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.adjustable_settings.add(cname)
        adj = self.app.brush_adjustment[cname]
        scale = InputSlider(adj)
        scale.set_draw_value(False)
        scale.set_hexpand(True)
        self.attach(label, 0, row, 1, 1)
        self.attach(scale, 1, row, 1, 1)
        row += 1
        cname = "fakepressure"
        label = Gtk.Label()
        # TRANSLATORS: Short alias for "Fake Pressure (for mouse)". This is
        # TRANSLATORS: used on the options panel.
        label.set_text(_("Pressure:"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        changed_cb = self._fakepressure_value_changed_cb
        adj = Gtk.Adjustment(value=0.5, lower=0.0, upper=1.0,
                             step_increment=0.01, page_increment=0.1)
        self.app.fake_adjustment['fakepressure'] = adj
        adj.connect("value-changed", changed_cb)
        scale = InputSlider(adj)
        scale.set_draw_value(False)
        scale.set_hexpand(True)
        self.attach(label, 0, row, 1, 1)
        self.attach(scale, 1, row, 1, 1)
        row += 1
        cname = "fakerotation"
        label = Gtk.Label()
        # TRANSLATORS: Short alias for "Fake Barrel Rotation (for mouse)".
        # TRANSLATORS: This is used on the options panel.
        label.set_text(_("Twist:"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        changed_cb = self._fakerotation_value_changed_cb
        adj = Gtk.Adjustment(value=0.5, lower=0.0, upper=1.0,
                             step_increment=0.0625, page_increment=0.25)
        self.app.fake_adjustment['fakerotation'] = adj
        adj.connect("value-changed", changed_cb)
        scale = InputSlider(adj)
        scale.set_draw_value(False)
        scale.set_hexpand(True)
        self.attach(label, 0, row, 1, 1)
        self.attach(scale, 1, row, 1, 1)
        row += 1
        return row

    def _fakepressure_value_changed_cb(self, adj):
        """Updates fakepressure when the user tweaks it using a scale"""
        newvalue = adj.get_value()
        self.app.fakepressure = newvalue

    def fakepressure_modified_cb(self, value):
        """Updates the fakepressure slider when changed elsewhere"""
        adj = self.app.fake_adjustment.get('fakepressure', None)
        if adj is not None:
            adj.set_value(value)

    def _fakerotation_value_changed_cb(self, adj):
        """Updates fakerotation when the user tweaks it using a scale"""
        newvalue = adj.get_value()
        self.app.fakerotation = newvalue

    def fakerotation_modified_cb(self, value):
        """Updates the fakerotation slider when changed elsewhere"""
        adj = self.app.fake_adjustment.get('fakerotation', None)
        if adj is not None:
            adj.set_value(value)


class PressureAndTiltInterpolator (object):
    """Interpolates event sequences, filling in null pressure/tilt data

    The interpolator operates almost as a filter. Feed the interpolator
    an extra zero-pressure event at button-release time to generate a
    nice tailoff for mouse users. The interpolator is sensitive to
    transitions between nonzero and zero effective pressure in both
    directions. These transitions clear out just enough history to avoid
    hook-off and lead-in artifacts.

    >>> interp = PressureAndTiltInterpolator()
    >>> raw_data = interp._TEST_DATA
    >>> all([len(t) == 9 for t in raw_data])
    True
    >>> any([t for t in raw_data if None in t[3:]])
    True
    >>> cooked_data = []
    >>> for raw_event in raw_data:
    ...    for cooked_event in interp.feed(*raw_event):
    ...        cooked_data.append(cooked_event)
    >>> any([t for t in cooked_data if None in t[3:]])
    False
    >>> len(cooked_data) <= len(raw_data)
    True
    >>> all([(t in cooked_data) for t in raw_data
    ...      if None not in t[3:]])
    True
    >>> any([t for t in cooked_data if 70 < t[0] < 110])
    False
    >>> len([t for t in cooked_data if t[0] in (70, 110)]) == 2
    True

    """

    # Test data:

    _TEST_DATA = [
        # These 2 events will be dropped (no prior state with pressure).
        (3, 0.3, 0.3, None, None, None, None, None, None),
        (7, 0.7, 0.7, None, None, None, None, None, None),
        (10, 1.0, 1.0, 0.33, 0.0, 0.5, 1.0, 0.0, 0.0),
        # Gaps between defined data like this one will have those
        # None entries filled in.
        (13, 1.3, 1.3, None, None, None, None, None, None),
        (15, 1.5, 1.5, None, None, None, None, None, None),
        (17, 1.7, 1.7, None, None, None, None, None, None),
        (20, 2.0, 2.0, 0.45, 0.1, 0.4, 1.0, 0.0, 0.0),
        (23, 2.3, 2.3, None, None, None, None, None, None),
        (27, 2.7, 2.7, None, None, None, None, None, None),
        (30, 3.0, 3.0, 0.50, 0.2, 0.3, 1.0, 0.0, 0.0),
        (33, 3.3, 3.3, None, None, None, None, None, None),
        (37, 3.7, 3.7, None, None, None, None, None, None),
        (40, 4.0, 4.0, 0.40, 0.3, 0.2, 1.0, 0.0, 0.0),
        (44, 4.4, 4.4, None, None, None, None, None, None),
        (47, 4.7, 4.7, None, None, None, None, None, None),
        (50, 5.0, 5.0, 0.30, 0.5, 0.1, 1.0, 0.0, 0.0),
        (53, 5.3, 5.3, None, None, None, None, None, None),
        (57, 5.7, 5.7, None, None, None, None, None, None),
        (60, 6.0, 6.0, 0.11, 0.4, 0.0, 1.0, 0.0, 0.0),
        (63, 6.3, 6.3, None, None, None, None, None, None),
        (67, 6.7, 6.7, None, None, None, None, None, None),
        # Down to zero pressure...
        (70, 7.0, 7.0, 0.00, 0.2, 0.0, 1.0, 0.0, 0.0),
        # .. followed by a null-pressure sequence.
        # That means that this gap will be skipped over till an
        # event with a defined pressure comes along.
        (73, 7.0, 7.0, None, None, None, None, None, None),
        (78, 50.0, 50.0, None, None, None, None, None, None),
        (83, 110.0, 110.0, None, None, None, None, None, None),
        (88, 120.0, 120.0, None, None, None, None, None, None),
        (93, 130.0, 130.0, None, None, None, None, None, None),
        (98, 140.0, 140.0, None, None, None, None, None, None),
        (103, 150.0, 150.0, None, None, None, None, None, None),
        (108, 160.0, 160.0, None, None, None, None, None, None),
        # Normally, event tuples won't be altered or have extra events
        # inserted between them.
        (110, 170.0, 170.0, 0.11, 0.1, 0.0, 1.0, 0.0, 0.0),
        (120, 171.0, 171.0, 0.33, 0.0, 0.0, 1.0, 0.0, 0.0),
        (130, 172.0, 172.0, 0.00, 0.0, 0.0, 1.0, 0.0, 0.0)
    ]

    # Construction:

    def __init__(self):
        """Instantiate with a clear internal state"""
        object.__init__(self)
        # Events with all axis data present, forming control points
        self._pt0_prev = None
        self._pt0 = None
        self._pt1 = None
        self._pt1_next = None
        # Null-axis event sequences
        self._np = []
        self._np_next = []

    # Internals:

    def _clear(self):
        """Reset to the initial clean state"""
        self._pt0_prev = None
        self._pt0 = None
        self._pt1 = None
        self._pt1_next = None
        self._np = []
        self._np_next = []

    def _step(self):
        """Step the interpolation parameters forward"""
        self._pt0_prev = self._pt0
        self._pt0 = self._pt1
        self._pt1 = self._pt1_next
        self._pt1_next = None
        self._np = self._np_next
        self._np_next = []

    def _interpolate_p0_p1(self):
        """Interpolate between p0 and p1, but do not step or clear"""
        pt0p, pt0 = self._pt0_prev, self._pt0
        pt1, pt1n = self._pt1, self._pt1_next
        can_interp = (pt0 is not None and pt1 is not None and
                      len(self._np) > 0)
        if can_interp:
            if pt0p is None:
                pt0p = pt0
            if pt1n is None:
                pt1n = pt1
            t0 = pt0[0]
            t1 = pt1[0]
            dt = t1 - t0
            can_interp = dt > 0
        if can_interp:
            for event in self._np:
                t, x, y = event[0:3]
                p, xt, yt, vz, vr, br = spline_4p(
                    (t - t0) / dt,
                    np.array(pt0p[3:]), np.array(pt0[3:]),
                    np.array(pt1[3:]), np.array(pt1n[3:])
                )
                yield (t, x, y, p, xt, yt, vz, vr, br)
        if pt1 is not None:
            yield pt1

    def _interpolate_and_step(self):
        """Internal: interpolate & step forward or clear"""
        for ievent in self._interpolate_p0_p1():
            yield ievent
        if ((self._pt1_next[3] > 0.0) and
                (self._pt1 is not None) and
                (self._pt1[3] <= 0.0)):
            # Transitions from zero to nonzero pressure
            # Clear history to avoid artifacts
            self._pt0_prev = None   # ignore the current pt0
            self._pt0 = self._pt1
            self._pt1 = self._pt1_next
            self._pt1_next = None
            self._np = []           # drop the buffer we've built up too
            self._np_next = []
        elif ((self._pt1_next[3] <= 0.0) and
              (self._pt1 is not None) and (self._pt1[3] > 0.0)):
            # Transitions from nonzero to zero pressure
            # Tail off neatly by doubling the zero-pressure event
            self._step()
            self._pt1_next = self._pt1
            for ievent in self._interpolate_p0_p1():
                yield ievent
            # Then clear history
            self._clear()
        else:
            # Normal forward of control points and event buffers
            self._step()

    # Public methods:

    def feed(self, time, x, y, pressure, xtilt, ytilt, viewzoom,
             viewrotation, barrel_rotation):
        """Feed in an event, yielding zero or more interpolated events

        :param time: event timestamp, integer number of milliseconds
        :param x: Horizontal coordinate of the event, in model space
        :type x: float
        :param y: Vertical coordinate of the event, in model space
        :type y: float
        :param pressure: Effective pen pressure, [0.0, 1.0]
        :param xtilt: Pen tilt in the model X direction, [-1.0, 1.0]
        :param ytilt: Pen tilt in the model's Y direction, [-1.0, 1.0]
        :param viewzoom: The view's current zoom level, [0, 64]
        :param viewrotation: The view's current rotation, [-180.0, 180.0]
        :param barrel_rotation: The stylus barrel rotation, [0.0, 1.0]
        :returns: Iterator of event tuples

        Event tuples have the form (TIME, X, Y, PRESSURE, XTILT, YTILT,
        VIEWZOOM, VIEWROTATION, BARREL_ROTATION).
        """
        if None in (pressure, xtilt, ytilt, viewzoom,
                    viewrotation, barrel_rotation):
            self._np_next.append((time, x, y, pressure, xtilt, ytilt, viewzoom,
                                  viewrotation, barrel_rotation))
        else:
            self._pt1_next = (time, x, y, pressure, xtilt, ytilt, viewzoom,
                              viewrotation, barrel_rotation)
            for t, x, y, p, xt, yt, vz, vr, br in self._interpolate_and_step():
                yield (t, x, y, p, xt, yt, vz, vr, br)


## Module tests

def _test():
    import doctest
    doctest.testmod()
    interp = PressureAndTiltInterpolator()
    # Emit CSV for ad-hoc plotting
    print("time,x,y,pressure,xtilt,ytilt,viewzoom,viewrotation,barrel_rotation")
    for event in interp._TEST_DATA:
        for data in interp.feed(*event):
            print(",".join([str(c) for c in data]))


if __name__ == '__main__':
    _test()
