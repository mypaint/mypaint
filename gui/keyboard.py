# This file is part of MyPaint.
# Copyright (C) 2009-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2009-2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Standardized, app-wide keyboard handling.

All actions, and almost all toplevel windows need to be registered here
for consistent keyboard handling.

"""

from __future__ import division, print_function
import logging

from lib.gibindings import Gtk
from lib.gibindings import Gdk
import lib.keyboard_util as kb

import gui.document
import gui.tileddrawwidget

logger = logging.getLogger(__name__)


class KeyboardManager:
    """Application-wide key event dispatch.

    This class represents all keyboard shortcuts (similar to
    Gtk.AccelGroup). It connects to keyboard events of various
    Gtk.Window instances to handle hotkeys. It synchronizes with the
    global gtk accelmap to figure out what keyboard shortcuts the user
    has assigned through the menu.

    The point of the whole exercise (instead of just using gtk
    standard tools) is to allow the action handlers to wait for the
    corresponding key release event.

    This class adds extra state attributes to every Gtk.Action.
    """

    ## Initialization

    def __init__(self, app):
        self.app = app
        self.enabled = True
        self.actions = []

        # Keymap hashes:  (keyval, modifiers) --> GtkAction
        self.keymap = {}
        self.keymap2 = {}  # 2nd priority; for hardcoded keys

        # Keypress state
        self.pressed = {}  # hardware_keycode -> GtkAction (while held down)

        # Window-specific sets of actions which can be invoked.
        # If one of these exists for a window (see `add_window()`), then
        # only these actions can be dispatched. Other events fall through to
        # the window.
        self.window_actions = {}  # GtkWindow -> set(['ActionName1', ...)

    def start_listening(self):
        """Begin listening for changes to the keymap.
        """
        accel_map = Gtk.AccelMap.get()
        accel_map.connect('changed', self._accel_map_changed_cb)

    ## Handle changes to the user-defined keymap

    def _accel_map_changed_cb(self, object, accel_path, accel_key, accel_mods):
        self._update_keymap(accel_path)

    def _update_keymap(self, accel_path):
        if not accel_path:
            return
        for k, v in list(self.keymap.items()):
            if v.get_accel_path() == accel_path:
                del self.keymap[k]

        found, key = Gtk.AccelMap.lookup_entry(accel_path)
        if found:
            for action in self.actions:
                if action.get_accel_path() == accel_path:
                    shortcut = (key.accel_key, key.accel_mods)
                    self.keymap[shortcut] = action
                    return
            logger.warning('Ignoring keybinding for %r', accel_path)

    ## Keyboard handling

    def _key_press_cb(self, widget, event):
        """App-wide keypress handler for toplevel windows."""

        # If an input widget has focus - their key handling is prioritized.
        consumed = widget.propagate_key_event(event)
        if consumed:
            return True
        if not self.enabled:
            return

        # Instead of using event.keyval, we do it the lowlevel way.
        # Reason: ignoring CAPSLOCK and checking if SHIFT was pressed
        keyval, keyval_lower, accel_label, modifiers = kb.translate(
            event.hardware_keycode,
            event.state,
            event.group
        )

        if not keyval:
            return

        # Except that key bindings are always stored in lowercase.
        keyval_lower = Gdk.keyval_to_lower(keyval)
        if keyval_lower != keyval:
            modifiers |= Gdk.ModifierType.SHIFT_MASK
        action = self.keymap.get((keyval_lower, modifiers))
        if not action and not self.fallbacks_disabled():
            # try hardcoded keys
            action = self.keymap2.get((keyval_lower, modifiers))

        # Don't dispatch if the window is only sensitive to a subset of
        # actions, and the action is not in that set.
        if action is not None and isinstance(action, Gtk.Action):
            win_actions = self.window_actions.get(widget, None)
            if win_actions is not None:
                if action.get_name() not in win_actions:
                    return False

        # If the lookup succeeded, activate the corresponding action.
        if action:
            return self.activate_keydown_event(action, event)

        # Otherwise, dispatch the event to the active doc.
        return self._dispatch_fallthru_key_press_event(widget, event)

    def fallbacks_disabled(self):
        return self.app.preferences.get('keyboard.disable_fallbacks', False)

    def activate_keydown_event(self, action, event):
        """Activate a looked-up action triggered by an event

        :param Gtk.Action action: action looked up in some keymap
        :param Gdk.Event: the triggering event
        :returns: True if the event should not be propagated further.
        :rtype: bool

        The KeyboardManager is responsible for activating events which
        correspond to keypresses so that it can keep track of which keys
        are pressed.  This part is exposed on a public method so that
        canvas "pointer" events using the Space=Button2 equivalence can
        invoke popup states via their action as proper keypresses.

        """
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
            self.pressed[event.hardware_keycode] = action
            # Make sure we also get the corresponding key release event
            activate()
        return True

    def _key_release_cb(self, widget, event):
        """Application-wide key release handler."""

        consumed = widget.propagate_key_event(event)
        if consumed:
            return True

        if not self.enabled:
            return

        def released(hardware_keycode):
            action = self.pressed[hardware_keycode]
            del self.pressed[hardware_keycode]
            if action.keyup_callback:
                action.keyup_callback(widget, event)
                action.keyup_callback = None

        if event.keyval == Gdk.KEY_Escape:
            # emergency exit in case of bugs
            for hardware_keycode in list(self.pressed.keys()):
                released(hardware_keycode)
            # Pop all stacked modes; they should release grabs
            self.app.doc.modes.reset()
            # Just in case...
            Gdk.pointer_ungrab(event.time)
        else:
            # note: event.keyval would not be suited for this because
            # it can be different from the one we have seen in
            # key_press_cb if the user has released a modifier first
            if event.hardware_keycode in self.pressed:
                released(event.hardware_keycode)
                return True

        # Fallthru handler: dispatch doc-specific stuff.
        return self._dispatch_fallthru_key_release_event(widget, event)

    def _get_active_doc(self):
        # Determines which is the active doc for the purposes of keyboard
        # event dispatch.
        active_tdw = gui.tileddrawwidget.TiledDrawWidget.get_active_tdw()
        for doc in gui.document.Document.get_instances():
            if doc.tdw is active_tdw:
                return (doc, doc.tdw)
        return (None, None)

    def _dispatch_fallthru_key_press_event(self, win, event):
        # Fall-through behavior: handle via the active document.
        target_doc, target_tdw = self._get_active_doc()
        if target_doc is None:
            return False
        return target_doc.key_press_cb(win, target_tdw, event)

    def _dispatch_fallthru_key_release_event(self, win, event):
        # Fall-through behavior: handle via the active document.
        target_doc, target_tdw = self._get_active_doc()
        if target_doc is None:
            return False
        return target_doc.key_release_cb(win, target_tdw, event)

    ## Toplevel window registration

    def add_window(self, window, actions=None):
        """Set up app-wide key event handling for a toplevel window.

        :param Gtk.Window window: the toplevel to handle key events of
        :param iterable actions: optional action names to handle only

        If action names are specified, then *only* those actions will be
        dispatched if the window has focus. Other keypresses and
        releases fall through to the usual handlers. Ideal for modal
        dialogs which want keyboard navigation, but also want to pop
        down when their action is invoked by a keypress.

        """
        handler_ids = []
        for name, cb in [("key-press-event", self._key_press_cb),
                         ("key-release-event", self._key_release_cb)]:
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

    ## Hardcoded fallback keymap

    def add_extra_key(self, keystring, action):
        """Adds a hardcoded keymap definition.

        These are processed as fallbacks and are used for things like
        the Tab or menu keys, or the cursor keys. The user-definable
        keymap overrides these.

        """
        keyval, modifiers = Gtk.accelerator_parse(keystring)
        if callable(action):
            # construct an action-like object from a function
            self._add_custom_attributes(action)
            action.activate = lambda: action(action)
        else:
            # find an existing Gtk.Action by name
            res = [a for a in self.actions if a.get_name() == action]
            assert len(res) == 1, \
                'action %s not found, or found more than once' % action
            action = res[0]
        self.keymap2[(keyval, modifiers)] = action

    ## Action registration

    def takeover_action(self, action):
        """Registers a GtkAction, and sets up custom attributes on it.

        The custom attributes are used internally by the kbm.

        """
        assert action not in self.actions
        self._add_custom_attributes(action)
        self.actions.append(action)
        self._update_keymap(action.get_accel_path())

    def _add_custom_attributes(self, action):
        assert not hasattr(action, 'keydown')
        assert not hasattr(action, 'keyup_callback')
        action.keydown = False
        action.keyup_callback = None
