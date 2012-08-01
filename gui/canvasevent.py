# This file is part of MyPaint.
# Copyright (C) 2008-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Canvas input event handling.
"""

import math

from numpy import isfinite

import pygtkcompat
import gobject
import gtk
from gtk import gdk
from gtk import keysyms


class InteractionMode (object):
    """Base class for temporary interaction modes.

    Active interaction objects process input events, and manipulate ether
    document views (TiledDrawWidget), the document model data (lib.document),
    or the mode stack they sit on. Interactions encapsulate state about their
    particular kind of interaction; for example a drag interaction typically
    contains the starting position for the drag.

    Event handler methods can create new sub-modes and `push()` or `replace()`
    them to the stack. It is conventional to pass the current event to the
    equivalent method on the new object when this transfer of control happens.

    """


    is_live_updateable = False


    def __init__(self):
        object.__init__(self)

        #: The `gui.document.Document` this mode should affect.
        #: Updated by enter()/leave().
        self.doc = None


    def enter(self, doc):
        """Enters the mode: called by `ModeStack.push()` etc.

        This is called when the mode becomes active, i.e. when it is the tip of
        the stack and before input is sent to it.

        :parameter doc: the `gui.document.Document` this mode should affect.

        """
        self.doc = doc


    # TODO: decide if the semantics when pushing/popping should involve
    # pause/resume. For now the stack uses enter/leave when a mode is revealed
    # or hidden.

    def leave(self):
        """Leaves the mode: called by `ModeStack.pop()` etc.

        This is called when an active mode becomes inactive, i.e. when it is
        popped from the stack.

        """
        self.doc = None


    def button_press_cb(self, tdw, event):
        """Handler for ``button-press-event``s."""
        pass


    def motion_notify_cb(self, tdw, event):
        """Handler for ``motion-notify-event``s."""
        pass


    def button_release_cb(self, tdw, event):
        """Handler for ``button-release-event``s."""
        pass


    def scroll_cb(self, tdw, event):
        """Handler for ``scroll-event``s.

        The base class implements some immediate rotation, and zoom commands
        for the main up-down scroll wheel: these should be useful to most
        modes.

        """
        d = event.direction
        if d == gdk.SCROLL_UP:
            if event.state & gdk.SHIFT_MASK:
                self.doc.rotate('RotateLeft')
            else:
                self.doc.zoom('ZoomIn')
        elif d == gdk.SCROLL_DOWN:
            if event.state & gdk.SHIFT_MASK:
                self.doc.rotate('RotateRight')
            else:
                self.doc.zoom('ZoomOut')
        elif d == gdk.SCROLL_RIGHT:
            self.doc.rotate('RotateRight')
        elif d == gdk.SCROLL_LEFT:
            self.doc.rotate('RotateLeft')
        return True


    def key_press_cb(self, win, tdw, event):
        """Handler for ``key-press-event``s.

        The base class implementation does nothing.
        Keypresses are received by the main window only, but at this point it
        has applied some heuristics to determine the active doc and view.
        These are passed through to the active mode and are accessible to
        keypress handlers via `self.doc` and the `tdw` argument.

        """
        pass


    def key_release_cb(self, win, tdw, event):
        """Handler for ``key-release-event``s.

        The base class implementation does nothing. See `key_press_cb` for
        details of the additional arguments.

        """
        pass


class FreehandOnlyMode (InteractionMode):
    """A freehand-only drawing mode, which cannot be switched with modifiers.

    This mode can be used with the basic CanvasController, and in the absence
    of the main application.

    """

    is_live_updateable = True


    def enter(self, doc):
        self.reset_drawing_state()
        InteractionMode.enter(self, doc)


    def leave(self):
        self.reset_drawing_state()
        InteractionMode.leave(self)


    def reset_drawing_state(self):
        self.last_event_had_pressure_info = False
        self.last_painting_pos = None
        # Windows stuff
        self.motions = []


    def button_press_cb(self, tdw, event):
        if event.type != gdk.BUTTON_PRESS:
            # Ignore the extra double-click event
            return
        if event.button == 1:
            # Stroke started, notify observers
            try:
                observers = self.doc.input_stroke_started_observers
            except AttributeError:
                pass
            else:
                for func in observers:
                    func(event)
            # Mouse button pressed (while painting without pressure
            # information)
            if not self.last_event_had_pressure_info:
                # For the mouse we don't get a motion event for "pressure"
                # changes, so we simulate it. (Note: we can't use the
                # event's button state because it carries the old state.)
                self.motion_notify_cb(tdw, event, button1_pressed=True)


    def button_release_cb(self, tdw, event):
        # (see comment above in button_press_cb)
        if event.button == 1:
            if not self.last_event_had_pressure_info:
                self.motion_notify_cb(tdw, event, button1_pressed=False)
            # Notify observers after processing the event
            try:
                observers = self.doc.input_stroke_ended_observers
            except AttributeError:
                pass
            else:
                for func in observers:
                    func(event)


    def motion_notify_cb(self, tdw, event, button1_pressed=None):
        # Purely responsible for drawing.
        if not tdw.is_sensitive:
            return

        model = tdw.doc
        app = tdw.app

        last_event_time, last_x, last_y = self.doc.get_last_event_info(tdw)
        if last_event_time:
            dtime = (event.time - last_event_time)/1000.0
            dx = event.x - last_x
            dy = event.y - last_y
        else:
            dtime = None
        if dtime is None:
            return

        same_device = True
        if app is not None:
            same_device = app.device_monitor.device_used(event.device)

        # Refuse drawing if the layer is locked or hidden
        if model.layer.locked or not model.layer.visible:
            return
            # TODO: some feedback, maybe

        x, y = tdw.display_to_model(event.x, event.y)

        pressure = event.get_axis(gdk.AXIS_PRESSURE)

        if pressure is not None and (   pressure > 1.0
                                     or pressure < 0.0
                                     or not isfinite(pressure)):
            if event.device.name not in self.bad_devices:
                print 'WARNING: device "%s" is reporting bad pressure %+f' \
                    % (event.device.name, pressure)
                self.bad_devices.append(event.device.name)
            if not isfinite(pressure):
                # infinity/nan: use button state (instead of clamping in
                # brush.hpp) https://gna.org/bugs/?14709
                pressure = None

        # Fake pressure if we have none based on the extra argument passed if
        # this is a fake.
        if pressure is None:
            self.last_event_had_pressure_info = False
            if button1_pressed is None:
                button1_pressed = event.state & gdk.BUTTON1_MASK
            if button1_pressed:
                pressure = 0.5
            else:
                pressure = 0.0
        else:
            self.last_event_had_pressure_info = True

        xtilt = event.get_axis(gdk.AXIS_XTILT)
        ytilt = event.get_axis(gdk.AXIS_YTILT)
        # Check whether tilt is present.  For some tablets without
        # tilt support GTK reports a tilt axis with value nan, instead
        # of None.  https://gna.org/bugs/?17084
        if xtilt is None or ytilt is None or not isfinite(xtilt+ytilt):
            xtilt = 0.0
            ytilt = 0.0

        if event.state & gdk.CONTROL_MASK or event.state & gdk.MOD1_MASK:
            # HACK: color picking, do not paint
            # Don't simply return; this is a workaround for unwanted lines
            # in https://gna.org/bugs/?16169
            pressure = 0.0

        if app is not None and app.pressure_mapping:
            pressure = app.pressure_mapping(pressure)

        if event.state & gdk.SHIFT_MASK:
            pressure = 0.0

        if pressure:
            self.last_painting_pos = x, y

        # If the device has changed and the last pressure value from the
        # previous device is not equal to 0.0, this can leave a visible stroke
        # on the layer even if the 'new' device is not pressed on the tablet
        # and has a pressure axis == 0.0.  Reseting the brush when the device
        # changes fixes this issue, but there may be a much more elegant
        # solution that only resets the brush on this edge-case.

        if not same_device:
            model.brush.reset()

        # On Windows, GTK timestamps have a resolution around
        # 15ms, but tablet events arrive every 8ms.
        # https://gna.org/bugs/index.php?16569
        # TODO: proper fix in the brush engine, using only smooth,
        #       filtered speed inputs, will make this unneccessary
        if dtime < 0.0:
            print 'Time is running backwards, dtime=%f' % dtime
            dtime = 0.0
        data = (x, y, pressure, xtilt, ytilt)
        if dtime == 0.0:
            self.motions.append(data)
        elif dtime > 0.0:
            if self.motions:
                # replay previous events that had identical timestamp
                if dtime > 0.1:
                    # really old events, don't associate them with the new one
                    step = 0.1
                else:
                    step = dtime
                step /= len(self.motions)+1
                for data_old in self.motions:
                    model.stroke_to(step, *data_old)
                    dtime -= step
                self.motions = []
            model.stroke_to(dtime, *data)


class SwitchableFreehandMode (FreehandOnlyMode):
    """The default mode: freehand drawing, accepting modifiers to switch modes.
    """

    def button_press_cb(self, tdw, event):
        app = tdw.app
        drawwindow = app.drawWindow

        #print event.device, event.button
        ## Ignore accidentals
        # Single button-presses only, not 2ble/3ple

        if event.type != gdk.BUTTON_PRESS:
            # pass through the extra double-click event
            return FreehandOnlyMode.button_press_cb(self, tdw, event)

        if event.button != 1:
            # check whether we are painting (accidental)
            if event.state & gdk.BUTTON1_MASK:
                # Do not allow dragging in the middle of
                # painting. This often happens by accident with wacom
                # tablet's stylus button.
                #
                # However we allow dragging if the user's pressure is
                # still below the click threshold.  This is because
                # some tablet PCs are not able to produce a
                # middle-mouse click without reporting pressure.
                # https://gna.org/bugs/index.php?15907
                return FreehandOnlyMode.button_press_cb(self, tdw, event)

        # Line Mode event
        if event.button == 1:
            line_mode = app.linemode.line_mode
            # Dynamic Line events from toolbar settings
            if line_mode != "FreehandMode":
                mode = DynamicLineMode(modifier=0)
                self.doc.modes.push(mode)
                mode.button_press_cb(tdw, event)
                return True

        # Pick a suitable config option
        ctrl = event.state & gdk.CONTROL_MASK
        alt  = event.state & gdk.MOD1_MASK
        shift = event.state & gdk.SHIFT_MASK
        if shift:
            modifier_str = "_shift"
            modifier = gdk.SHIFT_MASK
        elif alt or ctrl:
            modifier_str = "_ctrl"
            if alt:
                modifier = gdk.MOD1_MASK
            elif ctrl:
                modifier = gdk.CONTROL_MASK
        else:
            modifier_str = ""
            modifier = 0
        prefs_name = "input.button%d%s_action" % (event.button, modifier_str)
        action_name = app.preferences.get(prefs_name, "no_action")

        # No-ops
        if action_name == 'no_action':
            return FreehandOnlyMode.button_press_cb(self, tdw, event)

        # Line Mode event triggered by preferenced modifier button
        mode = None
        if action_name == 'straight_line':
            mode = DynamicLineMode(mode="StraightMode", modifier=modifier)
        elif action_name == 'straight_line_sequence':
            mode = DynamicLineMode(mode="SequenceMode", modifier=modifier)
        elif action_name == 'ellipse':
            mode = DynamicLineMode(mode="EllipseMode", modifier=modifier)
        if mode is not None:
            self.doc.modes.push(mode)
            mode.button_press_cb(tdw, event)
            return True

        # View control
        if action_name.endswith("_canvas"):
            mode = None
            if action_name == "pan_canvas":
                mode = PanViewMode()
            elif action_name == "zoom_canvas":
                mode = ZoomViewMode()
            elif action_name == "rotate_canvas":
                mode = RotateViewMode()
            assert mode is not None
            self.doc.modes.push(mode)
            mode.button_press_cb(tdw, event)
            return True

        # TODO: add frame manipulation here too

        if action_name == 'move_layer':
            mode = LayerMoveMode()
            self.doc.modes.push(mode)
            mode.button_press_cb(tdw, event)
            return True

        # Application menu
        if action_name == 'popup_menu':
            drawwindow.show_popupmenu(event=event)
            return True

        if action_name in drawwindow.popup_states:
            state = drawwindow.popup_states[action_name]
            state.activate(event)
            return True

        # Dispatch regular GTK events.
        action = app.find_action(action_name)
        if action is not None:
            action.activate()
            return True


    def key_press_cb(self, win, tdw, event):
        key = event.keyval
        ctrl = event.state & gdk.CONTROL_MASK
        shift = event.state & gdk.SHIFT_MASK
        alt = event.state & gdk.MOD1_MASK
        if key == keysyms.space:
            if (shift and ctrl) or alt:
                mode = MoveFrameMode()
            elif shift:
                mode = RotateViewMode()
            elif ctrl:
                mode = ZoomViewMode()
            else:
                mode = PanViewMode()
            self.doc.modes.push(mode)
            mode.key_press_cb(win, tdw, event)


class ModeStack (object):
    """A stack of InteractionModes. The top of the stack is the active mode.

    The stack can never be empty: if the final element is popped, it will be
    replaced with a new instance of its `default_mode_class`.

    """

    #: Class to instantiate if stack is empty: callable with 0 args.
    default_mode_class = SwitchableFreehandMode


    def __init__(self, doc):
        object.__init__(self)
        self._stack = []
        self._doc = doc

    @property
    def top(self):
        self._check()
        return self._stack[-1]

    def pop(self):
        old_mode = self._stack.pop(-1)
        old_mode.leave()
        self._check()
        self.top.enter(self._doc)

    def push(self, mode):
        self.top.leave()
        self._stack.append(mode)
        mode.enter(self._doc)

    def replace(self, mode):
        old_mode = self._stack.pop(-1)
        old_mode.leave()
        self._stack.append(mode)
        mode.enter(self._doc)

    def reset(self):
        """Clears the stack, popping the final element and replacing it.
        """
        while True:
            self.pop()
            if len(self._stack) == 1:
                break

    def _check(self):
        if len(self._stack) > 0:
            return
        mode = self.default_mode_class()
        self._stack.append(mode)
        mode.enter(self._doc)


class DragMode (InteractionMode):
    """Base class for drag activities.

    The drag can be entered when the pen is up or down: if the pen is down, the
    initial position will be determined from the first motion event.

    """

    cursor = gdk.Cursor(gdk.BOGOSITY)
    ## XXX two cursors? One for without the button pressed and one for with.

    def __init__(self):
        InteractionMode.__init__(self)
        self.last_x = None
        self.last_y = None
        self.start_x = None
        self.start_y = None
        self._start_keyval = None
        self._start_button = None
        self._grab_widget = None

    def _stop_if_started(self, t=gdk.CURRENT_TIME):
        if self._grab_widget is None:
            return
        tdw = self._grab_widget
        tdw.grab_remove()
        gdk.keyboard_ungrab(t)
        gdk.pointer_ungrab(t)
        self._grab_widget = None
        self.drag_stop_cb()

    def _start_unless_started(self, tdw, event):
        if self._grab_widget is not None:
            return
        if hasattr(event, "x"):
            self.start_x = event.x
            self.start_y = event.y
        else:
            last_t, last_x, last_y = self.doc.get_last_event_info(tdw)
            self.start_x = last_x
            self.start_y = last_y
        tdw_window = tdw.get_window()
        event_mask = gdk.BUTTON_PRESS_MASK | gdk.BUTTON_RELEASE_MASK \
                   | gdk.POINTER_MOTION_MASK
        grab_status = gdk.pointer_grab(tdw_window, False, event_mask, None,
                                       self.cursor, event.time)
        if grab_status != gdk.GRAB_SUCCESS:
            print "pointer grab failed:", grab_status
            return
        grab_status = gdk.keyboard_grab(tdw_window, False, event.time)
        if grab_status != gdk.GRAB_SUCCESS:
            print "keyboard grab failed:", grab_status
            gdk.pointer_ungrab()
            return
        tdw.grab_add()
        self._grab_widget = tdw
        self.drag_start_cb(tdw, event)

    def enter(self, doc):
        InteractionMode.enter(self, doc)
        self.doc.tdw.set_override_cursor(self.cursor)

    def leave(self):
        self._stop_if_started()
        self.doc.tdw.set_override_cursor(None)
        InteractionMode.leave(self)

    def scroll_cb(self, tdw, event):
        return True

    def button_press_cb(self, tdw, event):
        if self._start_keyval:  # not if keypress-initiated
            return
        self._start_unless_started(tdw, event)
        if self._grab_widget is None:
            return
        self.last_x = event.x
        self.last_y = event.y
        self._start_button = event.button
        return True

    def button_release_cb(self, tdw, event):
        if self._grab_widget is None:
            return
        if event.button == self._start_button:
            self._stop_if_started()
            self._start_button = None

    def motion_notify_cb(self, tdw, event):
        if self._grab_widget is None:
            return
        if not (self._start_button or self._start_keyval):
            # We might be here because an Action manipulated the modes stack
            # but if that's the case then we should wait for a button or
            # a keypress to initiate the drag.
            return
        self._start_unless_started(tdw, event)
        if self.last_x is not None:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            self.drag_update_cb(tdw, event, dx, dy)
        self.last_x = event.x
        self.last_y = event.y
        return True

    def key_press_cb(self, win, tdw, event):
        if self._start_button:  # not if button-initiated
            return
        if event.keyval == self._start_keyval:   # ignore repeats
            return
        if event.keyval == keysyms.space:
            self._start_keyval = event.keyval
            self._start_unless_started(tdw, event)

    def key_release_cb(self, win, tdw, event):
        if self._grab_widget is None:
            return
        if event.keyval == self._start_keyval:
            self._stop_if_started()
            self._start_keyval = None

    ## Default no-op callbacks for the drag sub-API

    def drag_start_cb(self, tdw, event):
        pass

    def drag_update_cb(self, tdw, event, dx, dy):
        pass

    def drag_stop_cb(self):
        pass



class OneshotDragModeMixin:
    """Oneshot drag modes exit & pop the mode stack when the drag stops."""

    def drag_stop_cb(self):
        self.doc.modes.pop()


class PanViewMode (OneshotDragModeMixin, DragMode):

    cursor = gdk.Cursor(gdk.FLEUR)

    def __init__(self):
        DragMode.__init__(self)

    def drag_update_cb(self, tdw, event, dx, dy):
        tdw.scroll(-dx, -dy)


class ZoomViewMode (OneshotDragModeMixin, DragMode):

    cursor = gdk.Cursor(gdk.SIZING)

    def __init__(self):
        DragMode.__init__(self)

    def drag_update_cb(self, tdw, event, dx, dy):
        if True:
            # Old style: zoom at wherever the pointer is
            tdw.scroll(-dx, -dy)
            tdw.zoom(math.exp(dy/100.0), center=(event.x, event.y))
        else:
            # Experimental: zoom around wherever the drag started
            start_pos = self.start_x, self.start_y
            tdw.zoom(math.exp(dy/100.0), center=start_pos)


class RotateViewMode (OneshotDragModeMixin, DragMode):

    cursor = gdk.Cursor(gdk.EXCHANGE)

    def __init__(self):
        DragMode.__init__(self)

    def drag_update_cb(self, tdw, event, dx, dy):
        # calculate angular velocity from the rotation center
        x, y = event.x, event.y
        cx, cy = tdw.get_center()
        x, y = x-cx, y-cy
        phi2 = math.atan2(y, x)
        x, y = x-dx, y-dy
        phi1 = math.atan2(y, x)
        tdw.rotate(phi2-phi1, center=(cx, cy))


class MoveFrameMode (OneshotDragModeMixin, DragMode):

    cursor = gdk.Cursor(gdk.ICON)

    def __init__(self):
        DragMode.__init__(self)

    def drag_update_cb(self, tdw, event, dx, dy):
        model = self.doc.model
        if not model.frame_enabled:
            # FIXME: this may be a little unintuitive and minimal
            return
        x, y, w, h = model.get_frame()
        x0, y0 = tdw.display_to_model(x, y)
        x1, y1 = tdw.display_to_model(x+dx, y+dy)
        model.move_frame(dx=x1-x0, dy=y1-y0)



class DynamicLineMode (DragMode):
    """Geometric line drawing."""

    # FIXME: split into multiple modes, one for each thing the linemode code
    # can do. Maybe some sub-modes for altering Bezier lines.

    cursor = gdk.Cursor(gdk.CROSS)

    def __init__(self, mode=None, modifier=0):
        DragMode.__init__(self)
        self.idle_srcid = None
        self._mode = mode
        self._modifier = modifier

    def drag_start_cb(self, tdw, event):
        modifier = self._modifier
        self.lm = tdw.app.linemode
        self.lm.start_command(self._mode, modifier)

    def drag_update_cb(self, tdw, event, dx, dy):
        self.lm.update_position(event.x, event.y)
        if self.idle_srcid is None:
            self.idle_srcid = gobject.idle_add(self.idle_cb)

    def drag_stop_cb(self):
        self.idle_srcid = None
        self.lm.stop_command()
        self.doc.modes.pop()

    def idle_cb(self):
        if self.idle_srcid is not None:
            self.idle_srcid = None
            self.lm.process_line()


class LayerMoveMode (DragMode):
    """Moving a layer interactively.

    MyPaint is tile-based, and tiles must align between layers, so moving
    layers involves copying data around. This is slow for very large layers, so
    the work is broken into chunks and processed in the idle phase of the GUI
    for greater responsivity.

    """

    cursor = gdk.Cursor(gdk.FLEUR)


    def __init__(self):
        DragMode.__init__(self)
        self.model_x0 = None
        self.model_y0 = None
        self.final_model_dx = None
        self.final_model_dy = None
        self.drag_update_idler_srcid = None
        self.layer = None
        self.move = None


    def drag_start_cb(self, tdw, event):
        if self.layer is not None:
            return
        self.layer = self.doc.model.get_current_layer()
        model_x, model_y = tdw.display_to_model(event.x, event.y)
        self.model_x0 = model_x
        self.model_y0 = model_y
        self.drag_start_tdw = tdw
        self.move = None


    def drag_update_cb(self, tdw, event, dx, dy):
        if self.layer is None:
            return

        # Begin moving, if we're not already
        if self.move is None:
            self.move = self.layer.get_move(self.model_x0, self.model_y0)

        # Update the active move 
        model_x, model_y = tdw.display_to_model(event.x, event.y)
        model_dx = model_x - self.model_x0
        model_dy = model_y - self.model_y0
        self.final_model_dx = model_dx
        self.final_model_dy = model_dy
        self.move.update(model_dx, model_dy)

        # Keep showing updates in the background for feedback.
        if self.drag_update_idler_srcid is None:
            self.drag_update_idler_srcid = gobject.idle_add(
                                            self.drag_update_idler)


    def drag_update_idler(self):
        # Process tile moves in chunks in a background idler
        # Terminate if asked
        if self.drag_update_idler_srcid is None:
            self.move.cleanup()
            return False
        # Process some tile moves, and carry on if there's more to do
        if self.move.process():
            return True
        # Nothing more to do for this move
        self.move.cleanup()
        self.drag_update_idler_srcid = None
        return False


    def drag_stop_cb(self):
        self.drag_update_idler_srcid = None   # ask it to finish
        if self.move is None:
            return

        # Arrange for the background work to be done, and look busy
        tdw = self.drag_start_tdw
        tdw.set_sensitive(False)
        tdw.set_override_cursor(gdk.Cursor(gdk.WATCH))
        gobject.idle_add(self.finalize_move_idler)


    def finalize_move_idler(self):
        # Finalize everything once the drag's finished.

        # Keep processing until the move queue is done.
        if self.move.process():
            return True

        # Cleanup tile moves
        self.move.cleanup()
        tdw = self.drag_start_tdw
        dx = self.final_model_dx
        dy = self.final_model_dy
        self.drag_start_tdw = self.move = None
        self.final_model_dx = self.final_model_dy = None

        # Arrange for the strokemap to be moved too;
        # this happens in its own background idler.
        for stroke in self.layer.strokes:
            stroke.translate(dx, dy)

        # Record move so it can be undone
        self.doc.model.record_layer_move(self.layer, dx, dy)
        self.layer = None

        # Restore sensitivity
        tdw.set_sensitive(True)
        tdw.set_override_cursor(None)

        # Leave mode
        # Moving a layer's content probably should act as a oneshot, even if
        # the leave is deferred.
        self.doc.modes.pop()
        return False
