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
        self.pressed = {} # keyval --> gtk.Action

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
        # This modifier checking code is from an example in the PyGTK docu:
        # http://www.pygtk.org/pygtk2reference/class-gdkkeymap.html
        keymap = gdk.keymap_get_default()
        # We want to ignore irrelevant modifiers like ScrollLock
        ALL_ACCELS_MASK = gtk.accelerator_get_default_mod_mask()
        keyval, egroup, level, consumed = keymap.translate_keyboard_state(
                         event.hardware_keycode, event.state, event.group)
        assert keyval == event.keyval
        modifiers = event.state & ~consumed & ALL_ACCELS_MASK
        action = self.keymap.get((keyval, modifiers))

        if action:
            def activate():
                action.keydown = True
                action.keyup_callback = None
                action.activate()
                action.keydown = False

            if keyval in self.pressed:
                # allow keyboard autorepeating only if the action
                # handler is not waiting for the key release event
                if not action.keyup_callback:
                    activate()
            else:
                #print 'PRESS', action.get_name()
                self.pressed[keyval] = action
                ## make sure we also get the corresponding key release event
                #gdk.keyboard_grab(widget.window, False, event.time)
                #widget.grab_add() hm? what would this do?
                activate()
            return True

    def key_release_cb(self, widget, event):
        def released(keyval):
            #gdk.keyboard_ungrab(event.time)
            action = self.pressed[keyval]
            del self.pressed[keyval]
            #print 'RELEASE', action.get_name()
            if action.keyup_callback:
                action.keyup_callback(widget, event)
                action.keyup_callback = None

        if event.keyval == gtk.keysyms.Escape:
            # emergency exit in case of bugs
            for keyval in self.pressed.keys():
                released(keyval)
        else:
            if event.keyval in self.pressed:
                released(event.keyval)
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

