# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2020 by the Mypaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Modes for manipulating brush properties"""

from __future__ import print_function

from math import ceil, hypot, log, pi

import gui.overlays
import gui.mode

from lib.brush import brush_visual_radius
from lib.gibindings import Gdk
from lib.gettext import C_


class BrushSizeOverlay(gui.overlays.Overlay):
    """Indicate the radius of the brush by a circular outline"""

    HANDLE_RADIUS = 3

    def __init__(self, doc, tdw, x, y, max_radius, radius, handle_x, handle_y):
        super(BrushSizeOverlay, self).__init__()
        self._max_radius = max_radius
        self._tdw = tdw
        self._x = int(round(x))
        self._y = int(round(y))
        self._hx = handle_x
        self._hy = handle_y
        self._radius = int(radius)
        self._old_radius = -1

        # Get style values from preferences - used when drawing the outlines
        prefs = doc.app.preferences
        self.col_fg = tuple(
            prefs.get("cursor.freehand.outer_line_color", (0, 0, 0, 1)))
        self.col_bg = tuple(
            prefs.get("cursor.freehand.inner_line_color", (1, 1, 1, 0.75)))
        lwi = float(prefs.get("cursor.freehand.inner_line_width", 1.25))
        lwo = float(prefs.get("cursor.freehand.outer_line_width", 1.25))
        self._line_width_inner = lwi
        self._line_width_outer = lwo

        # Calculate offsets - used when calculating invalidation rectangles
        hr = self.HANDLE_RADIUS
        self._handle_offs = hr + ceil(lwi / 2)
        self._inset = int(prefs.get("cursor.freehand.inner_line_inset", 2))
        self._lwi_offs = max(self._handle_offs,
                             self._inset + ceil(hypot(lwi/2, lwi/2)))
        self._lwo_offs = max(self._handle_offs, ceil(hypot(lwo/2, lwo/2)))

        # Keeps track of areas that were painted in the previous redraw,
        # and need to be invalidated for the next one (or when cleaning up).
        self._prev_areas = ()
        self._prev_handle_area = ()

        tdw.display_overlays.append(self)
        self._queue_tdw_redraw()

    def cleanup(self):
        self._tdw.display_overlays.remove(self)
        self._queue_tdw_redraw(clear=True)

    def _queue_tdw_redraw(self, clear=False):
        regions = []
        if self._prev_handle_area:
            regions.append(self._prev_handle_area)
        # When radius is unchanged (only the handle moved), don't redraw the
        # regions of the main circle. When clearing, only invalidate the last
        # region used for the main circle (after overlay is removed).
        if self._old_radius != self._radius or clear:
            if self._prev_areas:
                regions.extend(self._prev_areas)
            if not clear:
                new_areas = self._get_areas()
                regions.extend(new_areas)
                self._prev_areas = new_areas
        # Invalidate handle areas when they are/were outside the main circle,
        # or if the circle is at its maximum size (and won't be invalidated).
        if self._radius >= self._max_radius or clear:
            m = self._handle_offs
            handle = (self._hx - m, self._hy - m, 2 * m, 2 * m)
            regions.append(handle)
            self._prev_handle_area = handle
        else:
            self._prev_handle_area = None
        for region in regions:
            self._tdw.queue_draw_area(*region)

    def _get_areas(self):
        big_r = max(self._radius, self._old_radius)
        if big_r <= 32:
            # Small radius: return a rectangle covering the circle.
            self._prev_areas = ()  # Old area is always covered
            offs = big_r + self._lwo_offs
            return ((self._x - offs, self._y - offs, 2 * offs, 2 * offs),)
        else:
            # Large radius: return multiple rectangles covering
            # the outline (four slices based on the n * 45° points).
            r = self._radius
            # 0.2928932188134524 = 1 - cos(pi /4)
            short = ceil(0.2928932188134524 * r)  # short side
            # full width + margins
            offs_o = int(r + self._lwo_offs)
            # inner width - margins
            offs_i = int(r - short - self._lwi_offs)
            x, y, = self._x, self._y
            top = (x - offs_o, y - offs_o, 2 * offs_o, offs_o - offs_i)
            bot = (x - offs_o, y + offs_i, 2 * offs_o, offs_o - offs_i)
            lft = (x - offs_o, y - offs_i, offs_o - offs_i, 2 * offs_i)
            rgt = (x + offs_i, y - offs_i, offs_o - offs_i, 2 * offs_i)
            return top, lft, rgt, bot

    def update(self, radius, handle_x, handle_y):
        self._old_radius = self._radius
        self._radius = int(round(radius))
        self._hx = handle_x
        self._hy = handle_y
        self._queue_tdw_redraw()

    def paint(self, cr):
        cx = self._x
        cy = self._y

        r0 = self._radius
        r = r0 - self._line_width_outer / 2.0
        cr.set_source_rgba(*self.col_fg)
        cr.set_line_width(self._line_width_outer)
        cr.arc(cx, cy, r, 0, pi*2)
        cr.stroke()

        r = r0 - self._inset + self._line_width_inner / 2.0
        cr.set_source_rgba(*self.col_bg)
        cr.set_line_width(self._line_width_inner)
        cr.arc(cx, cy, r, 0, pi*2)
        cr.stroke()

        # Draw handle marker
        cr.set_source_rgba(*self.col_fg)
        cr.arc(self._hx, self._hy, self.HANDLE_RADIUS, 0, pi*2)
        cr.fill()
        cr.set_source_rgba(*self.col_bg)
        cr.arc(self._hx, self._hy, self.HANDLE_RADIUS - 1, 0, pi*2)
        cr.fill()


class BrushResizeMode(gui.mode.OneshotDragMode):
    """Mode for changing the size of the active brush by dragging on the canvas
    """

    # Does not actually refer to a GtkAction, only used to register the class
    ACTION_NAME = "BrushResizeMode"

    SPRING_LOADED = False

    mod_key_mask = Gdk.ModifierType.SHIFT_MASK

    pointer_behavior = gui.mode.Behavior.EDIT_OBJECTS
    supports_button_switching = True

    permitted_switch_actions = {None}

    def __init__(self, *args, **kwargs):
        super(BrushResizeMode, self).__init__(*args, **kwargs)
        self._prev_mode = None
        self._overlay = None
        self._radius_factor = None
        self._handle_x = None
        self._handle_y = None
        self._max_radius = None
        self._max_px_radius = None
        self._base_radius_adj = None
        self._new_radius = None
        self._precision_mode = False
        self._mod_pressed_initially = False

    def get_icon_name(self):
        return self._prev_mode.get_icon_name()

    @classmethod
    def get_name(cls):
        return C_(
            "brush resize mode - name",
            u"Resize Brush")

    def get_usage(self):
        return C_(
            "brush resize mode - usage",
            u"Change brush size by dragging on the canvas")

    @property
    def inactive_cursor(self):
        return None

    @property
    def active_cursor(self):
        return Gdk.Cursor.new(Gdk.CursorType.BLANK_CURSOR)

    def enter(self, doc, **kwds):
        super(BrushResizeMode, self).enter(doc, **kwds)
        # Record the mode that was on the top of the stack before entering
        # resize mode. It's used to retain the icon and options widget of
        # the "parent" mode (whether freehand, line modes, inking etc.).
        if not self._prev_mode:
            self._prev_mode = doc.modes[-2]
            self._prev_mode.doc = doc

    def drag_start_cb(self, tdw, event):
        super(BrushResizeMode, self).drag_start_cb(tdw, event)
        x, y = self.start_x, self.start_y
        brush_info = tdw.doc.brush.brushinfo
        visual_radius = brush_info.get_visual_radius()
        radius_px = visual_radius * tdw.scale
        by_random = brush_info.get_base_value('offset_by_random')
        self._base_radius_adj = tdw.app.brush_adjustment['radius_logarithmic']
        # This factor determines how to adjust the base radius value,
        # based on the visual radius (based on the cursor size equation).
        self._radius_factor = 1 / (1 + by_random * 2)
        self._handle_x = x + radius_px
        self._handle_y = y
        self._mod_pressed_initially = (
            self.current_modifiers() & self.mod_key_mask)
        self._max_radius = self._base_radius_adj.get_upper()
        self._max_px_radius = (
            brush_visual_radius(self._max_radius, by_random) * tdw.scale)
        self._overlay = BrushSizeOverlay(
            self.doc, tdw, x, y, self._max_px_radius,
            radius_px, self._handle_x, self._handle_y)

    def key_press_cb(self, win, tdw, event):
        self._update_precision_state()
        return super(BrushResizeMode, self).key_press_cb(win, tdw, event)

    def key_release_cb(self, win, tdw, event):
        self._update_precision_state()
        return super(BrushResizeMode, self).key_release_cb(win, tdw, event)

    def _update_precision_state(self):
        mod_pressed = self.current_modifiers() & self.mod_key_mask
        self._precision_mode = self._mod_pressed_initially ^ mod_pressed

    def leave(self, **kwds):
        self._overlay.cleanup()
        self._overlay = None
        self._prev_mode.doc = None
        self._prev_mode = None
        return super(BrushResizeMode, self).leave(**kwds)

    def drag_update_cb(self, tdw, event, ev_x, ev_y, dx, dy):
        # Change size at a quarter of the normal rate when
        # high-precision mode is enabled.
        move_factor = 0.125 if self._precision_mode else 0.5
        self._handle_x += dx * move_factor
        self._handle_y += dy * move_factor

        dx = self._handle_x - self.start_x
        dy = self._handle_y - self.start_y

        radius_px = hypot(dx, dy)
        new_radius = log((radius_px / tdw.scale) * self._radius_factor)

        if new_radius >= self._max_radius:
            new_radius = self._max_radius
            radius_px = self._max_px_radius
        self._new_radius = new_radius
        self._overlay.update(radius_px, self._handle_x, self._handle_y)

    def drag_stop_cb(self, tdw):
        if self._base_radius_adj and self._new_radius is not None:
            self._base_radius_adj.set_value(self._new_radius)
        return super(BrushResizeMode, self).drag_stop_cb(tdw)

    def get_options_widget(self):
        return self._prev_mode.get_options_widget()
