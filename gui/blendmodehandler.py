# This file is part of MyPaint.
# Copyright (C) 2019 by The MyPaint Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from lib.observable import event


class BlendMode(object):
    """A local on-off switch for a blend mode, with a change event"""

    def __init__(self, name, setting_name, active=False):
        self.name = name
        self.setting_name = setting_name
        self._active = active
        self.enabled = True

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, active):
        old_active = self._active
        self._active = active
        if old_active != active:
            self.changed()

    @event
    def changed(self):
        """Event dispatched triggered by a
        change in activation status"""


class BlendModes(object):
    """Proxy values for tools with individual blend mode states

    Used by tool modes to maintain their own instances of active
    blend modes and blend mode history.

    In order to enable individual blend modes for a tool mode,
    simply create its own BlendModes instance in a property named
    blend_modes and bind any listeners to it as needed.

    This class ensures that only a single mode is active at any one time
    and maintains a stack of modes to determine which it should switch to
    on deactivation.
    """

    NORMAL = 0
    ERASE = 1
    LOCK_ALPHA = 2
    COLORIZE = 3

    def __init__(self):
        self.eraser_mode = BlendMode("eraser_mode", "eraser")
        self.eraser_mode.mode_type = self.ERASE
        self.lock_alpha_mode = BlendMode("lock_alpha_mode", "lock_alpha")
        self.lock_alpha_mode.mode_type = self.LOCK_ALPHA
        self.colorize_mode = BlendMode("colorize_mode", "colorize")
        self.colorize_mode.mode_type = self.COLORIZE
        self.normal_mode = BlendMode("normal_mode", None, True)
        self.normal_mode.mode_type = self.NORMAL

        self.modes = [
            self.eraser_mode, self.lock_alpha_mode,
            self.colorize_mode, self.normal_mode
        ]
        self.mode_names = map(lambda x: x.name, self.modes)
        # Keep track of the active mode
        self.active_mode = self.normal_mode
        self.history = []

        for m in self.modes:
            m.changed += self._update

    def _push_history(self, mode):
        if mode is self.normal_mode:
            del self.history[:]
            return
        if mode in self.history:
            self.history.remove(mode)
        self.history.append(mode)

    def _pop_history(self, removed):
        while len(self.history) > 0:
            mode = self.history.pop()
            if mode is not removed:
                mode.active = True
                return
        self.normal_mode.active = True

    def _update(self, mode):
        old = self.active_mode
        old._active = False
        if mode.active:
            self._push_history(mode)
            self.active_mode = mode
            self.mode_changed(old, mode)
        else:
            self._pop_history(mode)

    @event
    def mode_changed(self, old_mode, new_mode):
        """Triggers when a mode changes,
        passing an instance of the changed mode"""


class BlendModeManager(object):
    """Manages blend mode actions by updating and redirecting
    callbacks based on BlendModes state models.

    A single instance "blendmodemanager" of this class is
    accessible from the Application singleton.

    This class allows multiple tool modes to make use of the same
    blend mode actions (same hotkeys etc.) by maintaining their
    own state in a BlendModes object, and taking control of the actions
    by registering that state when necessary.
    """

    def __init__(self, app):
        self.app = app
        self.delegates = []
        self.action_group = self.app.builder.get_object("BrushModifierActions")
        self.eraser_mode = app.find_action("BlendModeEraser")
        self.lock_alpha_mode = app.find_action("BlendModeLockAlpha")
        self.normal_mode = app.find_action("BlendModeNormal")
        self.colorize_mode = app.find_action("BlendModeColorize")
        self.actions = {
            "eraser_mode": self.eraser_mode,
            "lock_alpha_mode": self.lock_alpha_mode,
            "normal_mode": self.normal_mode,
            "colorize_mode": self.colorize_mode
        }

        for name in self.actions:
            action = self.actions[name]
            action.set_draw_as_radio(True)
            self.app.kbm.takeover_action(action)

        self._bm = None

    def register(self, bm):
        """Connect the blend mode actions to the given BlendModes object"""
        assert isinstance(bm, BlendModes)
        self.delegates.insert(0, bm)
        self._setup(bm)

    def update(self, bm, old, new):
        """Update actions without triggering change listeners"""
        old_action = self.actions[old.name]
        old_action.block_activate()
        old_action.set_active(False)
        old_action.unblock_activate()

        new_action = self.actions[new.name]
        new_action.block_activate()
        new_action.set_active(True)
        new_action.unblock_activate()

    def _setup(self, bm):
        """Set up listener and controls for new model and
        remove listener for old model"""
        # Deregister old change listener
        if self._bm:
            self._bm.mode_changed -= self.update
        # Update change listener
        self._bm = bm
        self._bm.mode_changed += self.update
        # Set up actions based on new BlendModes object
        for mode in bm.modes:
            action = self.actions[mode.name]
            action.block_activate()
            action.set_active(mode.active)
            action.set_sensitive(mode.enabled)
            action.unblock_activate()

    def deregister(self, bm):
        """Disconnect the blend mode actions from the given BlendModes
        if it is active.

        Remove the object from the stack and connect the next object in
        line for control, if such an object exists.
        """
        assert(isinstance(bm, BlendModes))
        if bm in self.delegates:
            self.delegates.remove(bm)
        else:
            return
        if self.delegates:
            self._setup(self.delegates[0])
        else:
            self._bm = None
            for action in self.actions.values():
                action.set_active(False)
                action.set_enabled(False)

    def blend_mode_normal_cb(self, action):
        if self._bm:
            self._bm.normal_mode.active = action.get_active()

    def blend_mode_eraser_cb(self, action):
        if self._bm:
            self._bm.eraser_mode.active = action.get_active()

    def blend_mode_lock_alpha_cb(self, action):
        if self._bm:
            self._bm.lock_alpha_mode.active = action.get_active()

    def blend_mode_colorize_cb(self, action):
        if self._bm:
            self._bm.colorize_mode.active = action.get_active()
