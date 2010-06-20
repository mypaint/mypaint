# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
gdk = gtk.gdk

class KeyboardManager:
    """
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
    def __init__(self):
        self.enabled = True
        self.actions = []
        self.keymap  = {} # (keyval, modifiers) --> gtk.Action
        self.keymap2 = {} # (keyval, modifiers) --> gtk.Action (2nd priority; for hardcoded keys)
        self.pressed = {} # hardware_keycode --> gtk.Action (while holding it down)

    def start_listening(self):
        gtk.accel_map_get().connect('changed', self.accel_map_changed_cb)

    def accel_map_changed_cb(self, object, accel_path, accel_key, accel_mods):
        self.update_keymap(accel_path)

    def update_keymap(self, accel_path):
        if not accel_path:
            return
        for k, v in self.keymap.items():
            if v.get_accel_path() == accel_path:
                del self.keymap[k]
        shortcut = gtk.accel_map_lookup_entry(accel_path)
        if shortcut:
            for action in self.actions:
                if action.get_accel_path() == accel_path:
                    self.keymap[shortcut] = action
                    return
            print 'Ignoring keybinding for', accel_path

    def key_press_cb(self, widget, event):
        if not self.enabled:
            return
        # See gtk sourcecode in gtkmenu.c function gtk_menu_key_press,
        # which uses the same code as below when changing an accelerator.
        keymap = gdk.keymap_get_default()
        # Instead of using event.keyval, we do it the lowlevel way.
        # Reason: ignoring CAPSLOCK and checking if SHIFT was pressed
        res = keymap.translate_keyboard_state(event.hardware_keycode, event.state & ~gdk.LOCK_MASK, event.group)
        if not res:
            # PyGTK returns None when gdk_keymap_translate_keyboard_state() returns false.
            # Not sure if this is a bug or a feature - the only time I have seen this
            # happen is when I put my laptop into sleep mode.
            print 'Warning: translate_keyboard_state() returned None. Strange key pressed?'
            return
        keyval, trash2, trash3, consumed_modifiers = res
        # We want to ignore irrelevant modifiers like ScrollLock.
        # The stored key binding does not include modifiers that affected its keyval.
        modifiers = event.state & gtk.accelerator_get_default_mod_mask() & ~consumed_modifiers
        # Except that key bindings are always stored in lowercase.
        keyval_lower = gdk.keyval_to_lower(keyval)
        if keyval_lower != keyval:
            modifiers |= gdk.SHIFT_MASK
        action = self.keymap.get((keyval_lower, modifiers))
        if not action:
            # try hardcoded keys
            action = self.keymap2.get((keyval_lower, modifiers))

        if action:
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
            gdk.pointer_ungrab()
        else:
            # note: event.keyval would not be suited for this because
            # it can be different from the one we have seen in
            # key_press_cb if the user has released a modifier first
            if event.hardware_keycode in self.pressed:
                released(event.hardware_keycode)
                return True
    
    def add_window(self, window):
        window.connect("key-press-event", self.key_press_cb)
        window.connect("key-release-event", self.key_release_cb)

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
            assert len(res) == 1, 'action %s not found, or found more than once' % action
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

