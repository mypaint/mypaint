# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
from gtk import gdk
import gtk2compat


class KeyboardManager:
    """Application-wide key event dispatch.

    This class represents all keyboard shortcuts (similar to
    gtk.AccelGroup). It connects to keyboard events of various
    gtk.Window instances to handle hotkeys. It synchronizes with the
    global gtk accelmap to figure out what keyboard shortcuts the user
    has assigned through the menu.

    The point of the whole exercise (instead of just using gtk
    standard tools) is to allow the action handlers to wait for the
    corresponding key release event.

    This class adds extra state attributes to every gtk.Action.
    """

    def __init__(self, app):
        self.app = app
        self.enabled = True
        self.actions = []

        # Keymap hashes:  (keyval, modifiers) --> GtkAction
        self.keymap  = {}
        self.keymap2 = {} # 2nd priority; for hardcoded keys

        # Keypress state
        self.pressed = {} # hardware_keycode -> GtkAction (while held down)

        # Window-specific sets of actions which can be invoked.
        # If one of these exists for a window (see `add_window()`), then
        # only these actions can be dispatched. Other events fall through to
        # the window.
        self.window_actions = {} # GtkWindow -> set(['ActionName1', ...)


    def start_listening(self):
        """Begin listening for changes to the keymap.
        """
        accel_map = gtk2compat.gtk.accel_map_get()
        accel_map.connect('changed', self.accel_map_changed_cb)


    def accel_map_changed_cb(self, object, accel_path, accel_key, accel_mods):
        self.update_keymap(accel_path)


    def update_keymap(self, accel_path):
        if not accel_path:
            return
        for k, v in self.keymap.items():
            if v.get_accel_path() == accel_path:
                del self.keymap[k]

        shortcut = gtk2compat.gtk.accel_map_lookup_entry(accel_path)
        if shortcut:
            for action in self.actions:
                if action.get_accel_path() == accel_path:
                    self.keymap[shortcut] = action
                    return
            print 'Ignoring keybinding for', accel_path


    def key_press_cb(self, widget, event):
        """App-wide keypress handler for toplevel windows.
        """
        if not self.enabled:
            return
        # See gtk sourcecode in gtkmenu.c function gtk_menu_key_press,
        # which uses the same code as below when changing an accelerator.
        keymap = gtk2compat.gdk.keymap_get_default()

        # Instead of using event.keyval, we do it the lowlevel way.
        # Reason: ignoring CAPSLOCK and checking if SHIFT was pressed
        state = event.state & ~gdk.LOCK_MASK
        if gtk2compat.USE_GTK3:
            state = gdk.ModifierType(state)
        res = keymap.translate_keyboard_state(event.hardware_keycode, state,
                                              event.group)
        if not res:
            # PyGTK returns None when gdk_keymap_translate_keyboard_state()
            # returns false.  Not sure if this is a bug or a feature - the only
            # time I have seen this happen is when I put my laptop into sleep
            # mode.
            print 'Warning: translate_keyboard_state() returned None. ' \
                  'Strange key pressed?'
            return

        keyval_offset = 1 if gtk2compat.USE_GTK3 else 0
        keyval = res[keyval_offset]
        consumed_modifiers = res[keyval_offset+3]

        # We want to ignore irrelevant modifiers like ScrollLock.  The stored
        # key binding does not include modifiers that affected its keyval.
        modifiers = event.state & gtk.accelerator_get_default_mod_mask() \
                  & ~consumed_modifiers

        # Except that key bindings are always stored in lowercase.
        keyval_lower = gdk.keyval_to_lower(keyval)
        if keyval_lower != keyval:
            modifiers |= gdk.SHIFT_MASK
        action = self.keymap.get((keyval_lower, modifiers))
        if not action:
            # try hardcoded keys
            action = self.keymap2.get((keyval_lower, modifiers))

        # Don't dispatch if the window is only sensitive to a subset of
        # actions, and the action is not in that set.
        if action is not None and isinstance(action, gtk.Action):
            win_actions = self.window_actions.get(widget, None)
            if win_actions is not None:
                if action.get_name() not in win_actions:
                    return False

        # Otherwise, dispatch via our handler.
        return self.activate_keydown_event(action, event)


    def activate_keydown_event(self, action, event):
        # The kbm is responsible for activating events which correspond to
        # keypresses so that it can keep track of which keys are pressed.
        # Expose this part on a separate method so that canvas "pointer" events
        # using the Space=Button2 equivalence can invoke popup states via their
        # action as proper keypresses.
        if not action:
            return False

        def activate():
            action.keydown = True
            action.keyup_callback = None
            action.activate()
            action.keydown = False

        if event.hardware_keycode in self.pressed:
            # allow keyboard autorepeating only if the action
            # handler is not waiting for the key release event
            if not action.keyup_callback:
                activate()
        else:
            #print 'PRESS', action.get_name()
            self.pressed[event.hardware_keycode] = action
            ## make sure we also get the corresponding key release event
            #gdk.keyboard_grab(widget.window, False, event.time)
            #widget.grab_add() hm? what would this do?
            activate()
        return True


    def key_release_cb(self, widget, event):
        """Application-wide key release handler.
        """

        def released(hardware_keycode):
            #gdk.keyboard_ungrab(event.time)
            action = self.pressed[hardware_keycode]
            del self.pressed[hardware_keycode]
            #print 'RELEASE', action.get_name()
            if action.keyup_callback:
                action.keyup_callback(widget, event)
                action.keyup_callback = None

        if event.keyval == gtk.keysyms.Escape:
            # emergency exit in case of bugs
            for hardware_keycode in self.pressed.keys():
                released(hardware_keycode)
            # Pop all stacked modes; they should release grabs
            self.app.doc.modes.reset()
            # Just in case...
            gdk.pointer_ungrab(event.time)
        else:
            # note: event.keyval would not be suited for this because
            # it can be different from the one we have seen in
            # key_press_cb if the user has released a modifier first
            if event.hardware_keycode in self.pressed:
                released(event.hardware_keycode)
                return True


    def add_window(self, window, actions=None):
        """Set up app-wide key event handling for a toplevel window.

        If `actions` is set to an iterable list of names, only those actions
        will be dispatched if the window has focus. Other keypresses and
        releases fall through to the window's normal handlers. Ideal for modal
        dialogs which want keyboard navigation, but also want to pop down when
        their action is invoked by a keypress.

        """
        handler_ids = []
        for name, cb in [ ("key-press-event", self.key_press_cb),
                          ("key-release-event", self.key_release_cb), ]:
            handler_id = window.connect(name, cb)
            handler_ids.append(handler_id)
        if actions is not None:
            action_names = [str(n) for n in actions]
            self.window_actions[window] = set(action_names)
        handler_id = window.connect("destroy", self._added_window_destroy_cb,
                                    handler_ids)
        handler_ids.append(handler_id)


    def _added_window_destroy_cb(self, window, handler_ids):
        """Clean up references to a window when it's destroyed.

        The main Workspace's undocked toolstack windows are created and
        destroyed as needed, so we need this.

        """
        self.window_actions.pop(window, None)
        for handler_id in handler_ids:
            window.disconnect(handler_id)   # is this needed?


    def add_extra_key(self, keystring, action):
        keyval, modifiers = gtk.accelerator_parse(keystring)
        if callable(action):
            # construct an action-like object from a function
            self.add_custom_attributes(action)
            action.activate = lambda: action(action)
            #action.get_name = lambda: action.__name__
        else:
            # find an existing gtk.Action by name
            res = [a for a in self.actions if a.get_name() == action]
            assert len(res) == 1, \
              'action %s not found, or found more than once' % action
            action = res[0]
        self.keymap2[(keyval, modifiers)] = action


    def takeover_action(self, action):
        assert action not in self.actions
        self.add_custom_attributes(action)
        self.actions.append(action)
        self.update_keymap(action.get_accel_path())


    def add_custom_attributes(self, action):
        assert not hasattr(action, 'keydown')
        assert not hasattr(action, 'keyup_callback')
        action.keydown = False
        action.keyup_callback = None

