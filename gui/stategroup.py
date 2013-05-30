# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import logging
logger = logging.getLogger(__name__)

import gtk2compat
import gobject
import gtk
from gtk import gdk

class StateGroup():
    """Supervisor instance for GUI states.

    This class mainly deals with the various ways how the user can
    leave such a mode, eg. if the mode is entered by holding down a
    key long enough, it will be left when the key is released.
    """

    def __init__(self):
        self.states = []
        self.keys_pressed = {}

    def get_active_states(self):
        return [s for s in self.states if s.active]
    active_states = property(get_active_states)

    def create_state(self, enter, leave, popup=None):
        s = State(self, popup)
        s.popup = None # FIXME: who uses this? hack?
        s.on_enter = enter
        s.on_leave = leave
        self.states.append(s)
        return s

    def create_popup_state(self, popup):
        return self.create_state(popup.enter, popup.leave, popup)

class State:
    """A GUI state.

    A GUI state is a mode which the GUI is in, for example an active
    popup window or a special (usually short-lived) view on the
    document. The application defines functions to be called when the
    state is entered or left.
    """

    #: How long a key can be held down to go through as single hit (and not
    #: press-and-hold)
    max_key_hit_duration = 0.250

    #: The state is automatically left after this time (ignored during
    #: press-and-hold)
    autoleave_timeout = 0.800

    ##: popups only: how long the cursor is allowed outside before closing
    ##: (ignored during press-and-hold)"
    #outside_popup_timeout = 0.050

    #: state to activate when this state is activated while already active
    #: (None = just leave this state)
    next_state = None

    #: Allowed buttons and their masks for starting and continuing states
    #: triggered by gdk button press events.
    allowed_buttons_masks = {
        1: gdk.BUTTON1_MASK,
        2: gdk.BUTTON2_MASK,
        3: gdk.BUTTON3_MASK, }

    #: Human-readable display string for the state.
    label = None

    def __init__(self, stategroup, popup):
        self.sg = stategroup
        self.active = False
        self.popup = popup
        self.autoleave_timer = None
        self.outside_popup_timer = None
        if popup:
            popup.connect("enter-notify-event", self.popup_enter_notify_cb)
            popup.connect("leave-notify-event", self.popup_leave_notify_cb)
            popup.popup_state = self # FIXME: hacky?
            self.outside_popup_timeout = popup.outside_popup_timeout

    def enter(self):
        logger.debug('Entering State, calling %s', self.on_enter.__name__)
        assert not self.active
        self.active = True
        self.enter_time = gtk.get_current_event_time()/1000.0
        self.connected_motion_handler = None
        if self.autoleave_timeout:
            self.autoleave_timer = gobject.timeout_add(int(1000*self.autoleave_timeout), self.autoleave_timeout_cb)
        self.on_enter()

    def leave(self, reason=None):
        logger.debug('Leaving State, calling %s', self.on_leave.__name__)
        assert self.active
        self.active = False
        if self.autoleave_timer:
            gobject.source_remove(self.autoleave_timer)
            self.autoleave_timer = None
        if self.outside_popup_timer:
            gobject.source_remove(self.outside_popup_timer)
            self.outside_popup_timer = None
        self.disconnect_motion_handler()
        self.on_leave(reason)

    def activate(self, action_or_event=None):
        """Activate a State from an action or a button press event.

        Only button press events are supported by this code.  When a GtkAction
        is activated, custom attributes are used to figure out whether the
        action was invoked from a menu, or using a keypress.  This requires the
        action to have been registered with the app's keyboard manager: see
        `keyboard.KeyboardManager.takeover_event()`.

        """
        if self.active:
            # pressing the key again
            if self.next_state:
                self.leave()
                self.next_state.activate(action_or_event)
                return

        # first leave other active states from the same stategroup
        for state in self.sg.active_states:
            state.leave()

        self.keydown = False
        self.mouse_button = None

        if action_or_event:
            if not isinstance(action_or_event, gtk.Action):
                e = action_or_event
                # currently, we only support mouse buttons being pressed here
                assert e.type == gdk.BUTTON_PRESS
                # let's just note down what mous button that was
                assert e.button
                if e.button in self.allowed_buttons_masks:
                    self.mouse_button = e.button

            else:
                a = action_or_event
                # register for key release events, see keyboard.py
                if a.keydown:
                    a.keyup_callback = self.keyup_cb
                    self.keydown = True
        self.activated_by_keyboard = self.keydown # FIXME: should probably be renamed (mouse button possible)
        self.enter()

    def toggle(self, action=None):
        if isinstance(action, gtk.ToggleAction):
            want_active = action.get_active()
        else:
            want_active = not self.active
        if want_active:
            if not self.active:
                self.activate(action)
        else:
            if self.active:
                self.leave()

    def keyup_cb(self, widget, event):
        if not self.active:
            return
        self.keydown = False
        if event.time/1000.0 - self.enter_time < self.max_key_hit_duration:
            pass # accept as one-time hit
        else:
            if self.outside_popup_timer:
                self.leave('outside')
            else:
                self.leave('keyup')

    def autoleave_timeout_cb(self):
        if not self.keydown:
            self.leave('timeout')
    def outside_popup_timeout_cb(self):
        if not self.keydown:
            self.leave('outside')

    def popup_enter_notify_cb(self, widget, event):
        if not self.active:
            return
        if self.outside_popup_timer:
            gobject.source_remove(self.outside_popup_timer)
            self.outside_popup_timer = None

    def popup_leave_notify_cb(self, widget, event):
        if not self.active:
            return
        # allow to leave the window for a short time
        if self.outside_popup_timer:
            gobject.source_remove(self.outside_popup_timer)
        self.outside_popup_timer = gobject.timeout_add(int(1000*self.outside_popup_timeout), self.outside_popup_timeout_cb)



    # ColorPicker-only stuff (for now)


    def motion_notify_cb(self, widget, event):
        assert self.keydown

        # We can't leave the state yet if button 1 is being pressed without
        # risking putting an accidental dab on the canvas. This happens with
        # some resistive touchscreens where a logical "button 3" is physically
        # a stylus button 3 plus a nib press (which is also a button 1).
        pressure = event.get_axis(gdk.AXIS_PRESSURE)
        button1_down = event.state & gdk.BUTTON1_MASK
        if pressure or button1_down:
            return

        # Leave if the button we started with is no longer being pressed.
        button_mask = self.allowed_buttons_masks.get(self.mouse_button, 0)
        if not event.state & button_mask:
            self.disconnect_motion_handler()
            self.keyup_cb(widget, event)

    def disconnect_motion_handler(self):
        if not self.connected_motion_handler:
            return
        widget, handler_id = self.connected_motion_handler
        widget.disconnect(handler_id)
        self.connected_motion_handler = None

    def register_mouse_grab(self, widget):
        assert self.active

        # fix for https://gna.org/bugs/?14871 (problem with tablet and pointer grabs)
        widget.add_events(gdk.POINTER_MOTION_MASK
                          # proximity mask might help with scrollwheel events (https://gna.org/bugs/index.php?16253)
                          | gdk.PROXIMITY_OUT_MASK
                          | gdk.PROXIMITY_IN_MASK
                          )
        if gtk2compat.USE_GTK3:
            pass
        else:
            widget.set_extension_events (gdk.EXTENSION_EVENTS_ALL)

        if self.keydown:
            # we are reacting to a keyboard event, we will not be
            # waiting for a mouse button release
            assert not self.mouse_button
            return
        if self.mouse_button:
            # we are able to wait for a button release now
            self.keydown = True
            # register for events
            assert self.mouse_button in self.allowed_buttons_masks
            handler_id = widget.connect("motion-notify-event", self.motion_notify_cb)
            assert not self.connected_motion_handler
            self.connected_motion_handler = (widget, handler_id)
        else:
            # The user is neither pressing a key, nor holding down a button.
            # This happens when activating the color picker from the menu.
            #
            # Stop everything, release the pointer grab.
            # (TODO: wait for a click instead, or show an instruction dialog)
            logger.warning('Releasing grab ("COV")')
            self.leave(None)

