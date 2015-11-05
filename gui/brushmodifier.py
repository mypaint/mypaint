# This file is part of MyPaint.
# Copyright (C) 2011 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


#import stategroup
import gtk
from gettext import gettext as _
from lib.helpers import rgb_to_hsv, hsv_to_rgb


class BrushModifier (object):
    """Applies changed brush settings to the active brush, with overrides.

    A single instance of this lives within the main `application.Application`
    instance. The BrushModifier tracks brush settings like color, eraser and
    lock alpha mode that can be overridden by the GUI.::

      BrushManager ---select_brush---> BrushModifier --> TiledDrawWidget

    The `BrushManager` provides the brush settings as stored on disk, and
    the BrushModifier passes them via `TiledDrawWidget` to the brush engine.
    """

    MODE_FORCED_ON_SETTINGS = [1.0, {}]
    MODE_FORCED_OFF_SETTINGS = [0.0, {}]

    def __init__(self, app):
        object.__init__(self)
        self.app = app
        app.brushmanager.brush_selected += self.brush_selected_cb
        app.brush.observers.append(self.brush_modified_cb)
        self.unmodified_brushinfo = app.brush.clone()
        self._in_brush_selected_cb = False
        self._last_selected_color = app.brush.get_color_hsv()
        self._init_actions()

    def _init_actions(self):
        self.action_group = self.app.builder.get_object("BrushModifierActions")

        # ToggleActions defined in resources.xml
        self.eraser_mode = self.app.find_action("BlendModeEraser")
        self.lock_alpha_mode = self.app.find_action("BlendModeLockAlpha")
        self.normal_mode = self.app.find_action("BlendModeNormal")
        self.colorize_mode = self.app.find_action("BlendModeColorize")

        # Each mode ToggleAction has a corresponding setting
        self.eraser_mode.setting_name = "eraser"
        self.lock_alpha_mode.setting_name = "lock_alpha"
        self.colorize_mode.setting_name = "colorize"
        self.normal_mode.setting_name = None

        # The set of toggle actions our fakery swicthes between
        self._toggle_actions = (self.eraser_mode, self.lock_alpha_mode,
                                self.colorize_mode, self.normal_mode)

        for action in self._toggle_actions:
            action.set_draw_as_radio(True)
            self.app.kbm.takeover_action(action)

        # Faking radio items, but ours can be disabled by re-activating them
        # and backtrack along their history.
        self._hist = []

    def _push_hist(self, justentered):
        if justentered in self._hist:
            self._hist.remove(justentered)
        self._hist.append(justentered)

    def _pop_hist(self, justleft):
        for mode in self._toggle_actions:
            if mode.get_active():
                return
        while len(self._hist) > 0:
            mode = self._hist.pop()
            if mode is not justleft:
                mode.set_active(True)
                return
        self.normal_mode.set_active(True)

    def set_override_setting(self, setting_name, override):
        """Overrides a boolean setting currently in effect.

        If `override` is true, the named setting will be forced to a base value
        greater than 0.9, and if it is false a base value less than 0.1 will be
        applied. Where possible, values from the base brush will be used. The
        setting from `unmodified_brushinfo`, including any input mapping, will
        be used if its base value is suitably large (or small). If not, a base
        value of either 1 or 0 and no input mapping will be applied.
        """
        unmod_b = self.unmodified_brushinfo
        modif_b = self.app.brush
        if override:
            if not modif_b.has_large_base_value(setting_name):
                settings = self.MODE_FORCED_ON_SETTINGS
                if unmod_b.has_large_base_value(setting_name):
                    settings = unmod_b.get_setting(setting_name)
                modif_b.set_setting(setting_name, settings)
        else:
            if not modif_b.has_small_base_value(setting_name):
                settings = self.MODE_FORCED_OFF_SETTINGS
                if unmod_b.has_small_base_value(setting_name):
                    settings = unmod_b.get_setting(setting_name)
                modif_b.set_setting(setting_name, settings)

    def _cancel_other_modes(self, action):
        for other_action in self._toggle_actions:
            if action is other_action:
                continue
            setting_name = other_action.setting_name
            if setting_name:
                self.set_override_setting(setting_name, False)
            if other_action.get_active():
                other_action.block_activate()
                other_action.set_active(False)
                other_action.unblock_activate()

    def blend_mode_normal_cb(self, action):
        """Callback for the ``BlendModeNormal`` action.
        """
        normal_wanted = action.get_active()
        if normal_wanted:
            self._cancel_other_modes(action)
            self._push_hist(action)
        else:
            # Disallow cancelling Normal mode unless something else
            # has become active.
            other_active = self.eraser_mode.get_active()
            other_active |= self.lock_alpha_mode.get_active()
            other_active |= self.colorize_mode.get_active()
            if not other_active:
                self.normal_mode.set_active(True)

    def blend_mode_eraser_cb(self, action):
        """Callback for the ``BlendModeEraser`` action.

        This manages the size difference between the eraser-mode version of a
        normal brush and its normal state.
        """
        eraser_wanted = action.get_active()
        if eraser_wanted:
            self._cancel_other_modes(action)
            self._push_hist(action)
        else:
            self._pop_hist(action)
        self.set_override_setting("eraser", eraser_wanted)

    def blend_mode_lock_alpha_cb(self, action):
        """Callback for the ``BlendModeLockAlpha`` action.
        """
        lock_alpha_wanted = action.get_active()
        if lock_alpha_wanted:
            self._cancel_other_modes(action)
            self._push_hist(action)
        else:
            self._pop_hist(action)
        self.set_override_setting("lock_alpha", lock_alpha_wanted)

    def blend_mode_colorize_cb(self, action):
        """Callback for the ``BlendModeColorize`` action.
        """
        colorize_wanted = action.get_active()
        if colorize_wanted:
            self._cancel_other_modes(action)
            self._push_hist(action)
        else:
            self._pop_hist(action)
        self.set_override_setting("colorize", colorize_wanted)

    def restore_context_of_selected_brush(self):
        """Restores color from the unmodified base brush.

        After a brush has been selected, restore additional brush settings -
        currently just color - from `unmodified_brushinfo`. This is called
        after selecting a brush by picking a stroke from the canvas.
        """
        c = self.unmodified_brushinfo.get_color_hsv()
        self.app.brush.set_color_hsv(c)

    def brush_selected_cb(self, bm, managed_brush, brushinfo):
        """Responds to the user changing their brush.

        This observer callback is responsible for allocating the current brush
        settings to the current brush singleton in `self.app`. The Brush
        Selector, the Pick Context action, and the Brushkeys and
        Device-specific brush associations all cause this to be invoked.
        """
        self._in_brush_selected_cb = True
        b = self.app.brush
        prev_lock_alpha = b.is_alpha_locked()

        # Changing the effective brush
        b.begin_atomic()
        color = b.get_color_hsv()

        mix_old = b.get_base_value('restore_color')
        b.load_from_brushinfo(brushinfo)
        self.unmodified_brushinfo = b.clone()

        # Preserve color
        mix = b.get_base_value('restore_color')
        if mix:
            c1 = hsv_to_rgb(*color)
            c2 = hsv_to_rgb(*b.get_color_hsv())
            c3 = [(1.0-mix)*v1 + mix*v2 for v1, v2 in zip(c1, c2)]
            color = rgb_to_hsv(*c3)
        elif mix_old and self._last_selected_color:
            # switching from a brush with fixed color back to a normal one
            color = self._last_selected_color
        b.set_color_hsv(color)

        b.set_string_property("parent_brush_name", managed_brush.name)

        if b.is_eraser():
            # User picked a dedicated eraser brush
            # Unset any lock_alpha state (necessary?)
            self.set_override_setting("lock_alpha", False)
        else:
            # Preserve the old lock_alpha state
            self.set_override_setting("lock_alpha", prev_lock_alpha)

        b.end_atomic()

        # Updates the blend mode buttons to match the new settings.
        # First decide which blend mode is active and which aren't.
        active_blend_mode = self.normal_mode
        blend_modes = []
        for mode_action in self._toggle_actions:
            setting_name = mode_action.setting_name
            if setting_name is not None:
                if b.has_large_base_value(setting_name):
                    active_blend_mode = mode_action
            blend_modes.append(mode_action)
        blend_modes.remove(active_blend_mode)

        # Twiddle the UI to match without emitting "activate" signals.
        active_blend_mode.block_activate()
        active_blend_mode.set_active(True)
        active_blend_mode.unblock_activate()
        for other in blend_modes:
            other.block_activate()
            other.set_active(False)
            other.unblock_activate()

        self._in_brush_selected_cb = False

    def _brush_is_dedicated_eraser(self):
        if self.unmodified_brushinfo is None:
            return False
        return self.unmodified_brushinfo.is_eraser()

    def brush_modified_cb(self, changed_settings):
        """Responds to changes of the brush settings.
        """
        if self._brush_is_dedicated_eraser():
            return

        if changed_settings.intersection(('color_h', 'color_s', 'color_v')):
            # Cancel eraser mode on ordinary brushes
            if self.eraser_mode.get_active() and 'eraser_mode' not in changed_settings:
                self.eraser_mode.set_active(False)

            if not self._in_brush_selected_cb:
                self._last_selected_color = self.app.brush.get_color_hsv()
