# This file is part of MyPaint.
# Copyright (C) 2009-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2011-2015 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function

import logging

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib


logger = logging.getLogger(__name__)


class StateGroup (object):
    """Supervisor instance for GUI states.

    This class mainly deals with the various ways how the user can
    leave such a mode, eg. if the mode is entered by holding down a
    key long enough, it will be left when the key is released.
    """

    def __init__(self):
        super(StateGroup, self).__init__()
        self.states = []
        self.keys_pressed = {}

    @property
    def active_states(self):
        return [s for s in self.states if s.active]

    def create_state(self, enter, leave, popup=None):
        s = State(self, popup)
        s.popup = None  # FIXME: who uses this? hack?
        s.on_enter = enter
        s.on_leave = leave
        self.states.append(s)
        return s

    def create_popup_state(self, popup):
        return self.create_state(popup.enter, popup.leave, popup)


class State (object):
    """A GUI state.

    A GUI state is a mode which the GUI is in, for example an active
    popup window or a special (usually short-lived) view on the
    document. The application defines functions to be called when the
    state is entered or left.
    """

    ## Class consts and instance defaults

    #: How long a key can be held down to go through as single hit (and not
    #: press-and-hold)
    max_key_hit_duration = 0.250

    #: The state is automatically left after this time (ignored during
    #: press-and-hold)
    autoleave_timeout = 0.800

    # : popups only: how long the cursor is allowed outside before closing
    # : (ignored during press-and-hold)"
    #outside_popup_timeout = 0.050

    #: state to activate when this state is activated while already active
    #: (None = just leave this state)
    next_state = None

    #: Allowed buttons and their masks for starting and continuing states
    #: triggered by gdk button press events.
    allowed_buttons_masks = {
        1: Gdk.ModifierType.BUTTON1_MASK,
        2: Gdk.ModifierType.BUTTON2_MASK,
        3: Gdk.ModifierType.BUTTON3_MASK, }

    #: Human-readable display string for the state.
    label = None


    ## Methods

    def __init__(self, stategroup, popup):
        super(State, self).__init__()
        self.sg = stategroup
        self.active = False
        self.popup = popup
        self._autoleave_timeout_id = None
        self._outside_popup_timeout_id = None
        if popup:
            popup.connect("enter-notify-event", self._popup_enter_notify_cb)
            popup.connect("leave-notify-event", self._popup_leave_notify_cb)
            popup.popup_state = self  # FIXME: hacky?
            self.outside_popup_timeout = popup.outside_popup_timeout

    def enter(self, **kwargs):
        logger.debug('Entering State, calling %s', self.on_enter.__name__)
        assert not self.active
        self.active = True
        self._enter_time = Gtk.get_current_event_time()/1000.0
        try:
            self.on_enter(**kwargs)
        except:
            logger.exception("State on_enter method failed")
            raise
        self._restart_autoleave_timeout()

    def leave(self, reason=None):
        logger.debug(
            'Leaving State (reason=%r), calling %s',
            reason,
            self.on_leave.__name__,
        )
        assert self.active
        self.active = False
        self._stop_autoleave_timeout()
        self._stop_outside_popup_timeout()
        self._enter_time = None
        try:
            self.on_leave(reason)
        except:
            logger.exception("State on_leave method failed")
            raise

    def activate(self, action_or_event=None, **kwargs):
        """Activate a State from an action or a button press event.

        :param action_or_event: A Gtk.Action, or a Gdk.Event.
        :param \*\*kwargs: passed to enter().

        For events, only button press events are supported by this code.

        When a Gtk.Action is activated, custom attributes are used to
        figure out whether the action was invoked from a menu, or using
        a keypress.  This requires the action to have been registered
        with the app's keyboard manager.

        See also `keyboard.KeyboardManager.takeover_event()`.

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
            if not isinstance(action_or_event, Gtk.Action):
                e = action_or_event
                # eat any multiple clicks. TODO should possibly try to e.g.
                # split a triple click into three clicks in the future.
                if (e.type == Gdk.EventType.DOUBLE_BUTTON_PRESS or
                        e.type == Gdk.EventType.TRIPLE_BUTTON_PRESS):
                    e.type = Gdk.EventType.BUTTON_PRESS

                # currently we only support mouse buttons being single-pressed.
                assert e.type == Gdk.EventType.BUTTON_PRESS
                # let's just note down what mouse button that was
                assert e.button
                if e.button in self.allowed_buttons_masks:
                    self.mouse_button = e.button

            else:
                a = action_or_event
                # register for key release events, see keyboard.py
                if a.keydown:
                    a.keyup_callback = self._keyup_cb
                    self.keydown = True
        self.activated_by_keyboard = self.keydown  # FIXME: should probably be renamed (mouse button possible)
        self.enter(**kwargs)

    def toggle(self, action=None):
        if isinstance(action, Gtk.ToggleAction):
            want_active = action.get_active()
        else:
            want_active = not self.active
        if want_active:
            if not self.active:
                self.activate(action)
        else:
            if self.active:
                self.leave()

    def _keyup_cb(self, widget, event):
        if not self.active:
            return
        self.keydown = False
        if event.time/1000.0 - self._enter_time < self.max_key_hit_duration:
            pass  # accept as one-time hit
        else:
            if self._outside_popup_timeout_id:
                self.leave('outside')
            else:
                self.leave('keyup')

    ## Auto-leave timeout

    def _stop_autoleave_timeout(self):
        if not self._autoleave_timeout_id:
            return
        GLib.source_remove(self._autoleave_timeout_id)
        self._autoleave_timeout_id = None

    def _restart_autoleave_timeout(self):
        if not self.autoleave_timeout:
            return
        self._stop_autoleave_timeout()
        self._autoleave_timeout_id = GLib.timeout_add(
            int(1000*self.autoleave_timeout),
            self._autoleave_timeout_cb,
        )

    def _autoleave_timeout_cb(self):
        if not self.keydown:
            self.leave('timeout')
        return False

    ## Outside-popup timer

    def _stop_outside_popup_timeout(self):
        if not self._outside_popup_timeout_id:
            return
        GLib.source_remove(self._outside_popup_timeout_id)
        self._outside_popup_timeout_id = None

    def _restart_outside_popup_timeout(self):
        if not self.outside_popup_timeout:
            return
        self._stop_outside_popup_timeout()
        self._outside_popup_timeout_id = GLib.timeout_add(
            int(1000*self.outside_popup_timeout),
            self._outside_popup_timeout_cb,
        )

    def _outside_popup_timeout_cb(self):
        if not self._outside_popup_timeout_id:
            return False
        self._outside_popup_timeout_id = None
        if not self.keydown:
            self.leave('outside')
        return False

    def _popup_enter_notify_cb(self, widget, event):
        if not self.active:
            return
        if self._outside_popup_timeout_id:
            GLib.source_remove(self._outside_popup_timeout_id)
            self._outside_popup_timeout_id = None

    def _popup_leave_notify_cb(self, widget, event):
        if not self.active:
            return
        # allow to leave the window for a short time
        if self._outside_popup_timeout_id:
            GLib.source_remove(self._outside_popup_timeout_id)
            self._outside_popup_timeout_id = None
        self._outside_popup_timeout_id = GLib.timeout_add(
            int(1000*self.outside_popup_timeout),
            self._outside_popup_timeout_cb,
        )
