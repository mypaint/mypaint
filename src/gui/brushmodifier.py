# This file is part of MyPaint.
# Copyright (C) 2011 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function

from gettext import gettext as _
from lib.helpers import rgb_to_hsv, hsv_to_rgb

import gui.blendmodehandler


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
        self.bm = gui.blendmodehandler.BlendModes()
        self.bm.mode_changed += self.update_blendmodes

    def update_blendmodes(self, bm, old, new):
        if old is new:
            return
        if old.setting_name:
            self.set_override_setting(old.setting_name, False)
        if new.setting_name:
            self.set_override_setting(new.setting_name, True)

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
        active_blend_mode = self.bm.normal_mode
        for mode in self.bm.modes:
            setting_name = mode.setting_name
            if setting_name is not None:
                if b.has_large_base_value(setting_name):
                    active_blend_mode = mode
        active_blend_mode.active = True

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
            em = self.bm.eraser_mode
            if em.active and 'eraser_mode' not in changed_settings:
                em.active = False

            if not self._in_brush_selected_cb:
                self._last_selected_color = self.app.brush.get_color_hsv()
