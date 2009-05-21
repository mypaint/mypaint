# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk, gobject
gdk = gtk.gdk
import time

class StateGroup():
    """
    Supervisor instance for GUI states.

    A GUI state is a mode which the GUI is in, for example an active
    popup window or a special (usually short-lived) view on the
    document. The application defines functions to be called when the
    state is entered or left.

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
    "how long a key can be held down to go through as single hit (and not press-and-hold)"
    max_key_hit_duration = 0.200
    "the state is automatically left after this time (ignored during press-and-hold)"
    autoleave_timeout = 0.800
    #"popups only: how long the cursor is allowed outside before closing (ignored during press-and-hold)"
    #outside_popup_timeout = 0.050
    "state to activate when this state is activated while already active (None = just leave this state)"
    next_state = None

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
        #print 'entering state, calling', self.on_enter.__name__
        assert not self.active
        self.active = True
        self.enter_time = time.time()
        if self.autoleave_timeout:
            self.autoleave_timer = gobject.timeout_add(int(1000*self.autoleave_timeout), self.autoleave_timeout_cb)
        self.on_enter()

    def leave(self, reason=None):
        #print 'leaving state, calling', self.on_leave.__name__
        assert self.active
        self.active = False
        if self.autoleave_timer:
            gobject.source_remove(self.autoleave_timer)
            self.autoleave_timer = None
        if self.outside_popup_timer:
            gobject.source_remove(self.outside_popup_timer)
            self.outside_popup_timer = None
        self.on_leave(reason)

    def activate(self, action=None):
        """
        Called from the GUI code, eg. when a gtk.Action is
        activated. The action is used to figure out the key.
        """
        if self.active:
            # pressing the key again
            if self.next_state:
                self.leave()
                self.next_state.activate()
                return

        # first leave other active states from the same stategroup
        for state in self.sg.active_states:
            state.leave()

        self.keydown = False
        if action and action.keydown:
            self.keydown = True
            action.keyup_callback = self.keyup_cb
        self.enter()

    def toggle(self, action=None):
        if not self.active:
            self.activate(action)
        else:
            self.leave()

    def keyup_cb(self, widget, event):
        if not self.active:
            return
        self.keydown = False
        if time.time() - self.enter_time < self.max_key_hit_duration:
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

