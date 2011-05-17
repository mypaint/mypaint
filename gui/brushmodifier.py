# This file is part of MyPaint.
# Copyright (C) 2011 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""
The BrushModifier tracks brush settings like color, eraser and
blending modes that can be overridden by the GUI.

BrushManager ----select_brush----> BrushModifier --> TiledDrawWidget

The BrushManager provides the brush settings as stored on disk, and
the BrushModifier passes them via TiledDrawWidget to the brush engine.
"""

import stategroup

class BrushModifier:
    def __init__(self, app):
        self.app = app

        self.app.brushmanager.selected_brush_observers.append(self.brush_selected_cb)
        self.app.brush.observers.append(self.brush_modified_cb)

        self.unmodified_brushinfo = None

        sg = stategroup.StateGroup()
        self.eraser_mode = sg.create_state(enter=self.enter_eraser_mode_cb,
                                           leave=self.leave_eraser_mode_cb)
        self.eraser_mode.autoleave_timeout = None
        sg = stategroup.StateGroup()
        self.lock_alpha = sg.create_state(enter=self.enter_lock_alpha_cb,
                                          leave=self.leave_lock_alpha_cb)
        self.lock_alpha.autoleave_timeout = None

    def enter_eraser_mode_cb(self):
        b = self.app.brush

        self.lock_alpha_before_eraser_mode = self.lock_alpha.active
        if self.lock_alpha.active:
            self.lock_alpha.leave()

        if b.get_base_value('eraser') > 0.9:
            # we are entering eraser mode because the selected brush is an eraser
            self.eraser_mode_original_brush = None
        else:
            self.eraser_mode_original_brush = b.clone()
            # create an eraser version of the current brush
            b.set_base_value('eraser', 1.0)

            # change brush radius
            r = b.get_base_value('radius_logarithmic')
            default = 3*(0.3) # this value allows the user to go back to the exact original size with brush_smaller_cb()
            dr = self.app.preferences.get('document.eraser_mode_radius_change', default)
            b.set_base_value('radius_logarithmic', r + dr)

    def leave_eraser_mode_cb(self, reason):
        if self.eraser_mode_original_brush:
            b = self.app.brush
            # remember radius change
            dr = b.get_base_value('radius_logarithmic') - self.eraser_mode_original_brush.get_base_value('radius_logarithmic')
            self.app.preferences['document.eraser_mode_radius_change'] = dr 
            b.load_from_brushinfo(self.eraser_mode_original_brush)
        del self.eraser_mode_original_brush

        if self.lock_alpha_before_eraser_mode:
            self.lock_alpha.enter()
        del self.lock_alpha_before_eraser_mode

    def enter_lock_alpha_cb(self):
        self.enforce_lock_alpha()

    def enforce_lock_alpha(self):
        b = self.app.brush
        b.begin_atomic()
        if b.get_base_value('eraser') < 0.9:
            b.reset_setting('lock_alpha')
            b.set_base_value('lock_alpha', 1.0)
        b.end_atomic()

    def leave_lock_alpha_cb(self, reason):
        data = self.unmodified_brushinfo.get_setting('lock_alpha')
        self.app.brush.set_setting('lock_alpha', data)

    def restore_context_of_selected_brush(self):
        """
        After a brush has been selected, restore additional brush
        settings (eg. color) from the brushinfo. This is called after
        selecting a brush by picking a stroke from the canvas.
        """
        c = self.unmodified_brushinfo.get_color_hsv()
        self.app.brush.set_color_hsv(c)

    def brush_modified_cb(self, settings):
        if settings.intersection(('color_h', 'color_s', 'color_v')):
            if self.eraser_mode.active:
                if 'eraser_mode' not in settings:
                    self.eraser_mode.leave()

    def brush_selected_cb(self, managed_brush):
        """
        Copies a ManagedBrush's settings into the brush settings currently used
        for painting. Sets the parent brush name to the closest ancestor brush
        currently in the brushlist.
        """
        if not managed_brush:
            return

        b = self.app.brush

        b.begin_atomic()

        if self.eraser_mode.active:
            self.eraser_mode.leave()

        color = b.get_color_hsv()

        b.load_from_brushinfo(managed_brush.brushinfo)
        self.unmodified_brushinfo = b.clone()

        b.set_color_hsv(color)

        parent_name = None
        list_brush = self.app.brushmanager.find_brushlist_ancestor(managed_brush)
        if list_brush and list_brush.name is not None:
            parent_name = list_brush.name
        b.set_string_property("parent_brush_name", parent_name)

        if self.lock_alpha.active:
            self.enforce_lock_alpha()

        b.end_atomic()





