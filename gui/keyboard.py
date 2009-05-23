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
    global gtk accelmap to figure out what the keyboard shortcuts the
    user has assigned in the menu.

    The point of the whole exercise (instead of just using gtk
    standard tools) is to make it possible for the action handler to
    wait for the key release event that has activated the
    action. Autorepeated key press events are blocked in this case.
    """
    def __init__(self):
        gtk.accel_map_get().connect('changed', self.accel_map_changed_cb)
        self.actions = []
        self.keymap = {} # (keyval, modifiers) --> gtk.Action
        self.pressed = {} # hardware_keycode --> gtk.Action (while holding it down)

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
            raise RuntimeError

    def key_press_cb(self, widget, event):
        # See gtk sourcecode in gtkmenu.c function gtk_menu_key_press,
        # which uses the same code as below when changing an accelerator.
        keymap = gdk.keymap_get_default()
        # figure out what modifiers went into determing the keyval
        trash1, trash2, trash3, consumed_modifiers = keymap.translate_keyboard_state(
                         event.hardware_keycode, event.state, event.group)
        # We want to ignore irrelevant modifiers like ScrollLock.
        # The stored key binding does not include modifiers that affected its keyval.
        modifiers = event.state & gtk.accelerator_get_default_mod_mask() & ~consumed_modifiers
        # Except that key bindings are always stored in lowercase.
        keyval = gdk.keyval_to_lower(event.keyval)
        if keyval != event.keyval:
            modifiers |= gdk.SHIFT_MASK
        #print 'You are pressing keyval', event.keyval, 'with hardware code', event.hardware_keycode
        #print 'Which has the lowercase form', keyval
        action = self.keymap.get((keyval, modifiers))

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
        #print 'You are releasing keyval', event.keyval, 'with hardware code', event.hardware_keycode
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

    def takeover_action(self, action):
        assert action not in self.actions
        # custom action attributes:
        assert not hasattr(action, 'keydown')
        assert not hasattr(action, 'keyup_callback')
        action.keydown = False
        action.keyup_callback = None
        self.actions.append(action)
        self.update_keymap(action.get_accel_path())

