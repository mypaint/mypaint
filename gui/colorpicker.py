# This file is part of MyPaint.
# Copyright (C) 2010-2020 by the MyPaint Development Team
# Copyright (C) 2009-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

from __future__ import division, print_function
from gettext import gettext as _

import gui.mode
from .overlays import Overlay
from .overlays import rounded_box
from lib.color import HCYColor, HSVColor

from lib.gibindings import GLib


## Color picking mode, with a preview rectangle overlay

class ColorPickMode (gui.mode.OneshotDragMode):
    """Mode for picking colors from the screen, with a preview

    This can be invoked in quite a number of ways:

    * The keyboard hotkey ("R" by default)
    * Modifier and pointer button: (Ctrl+Button1 by default)
    * From the toolbar or menu

    The first two methods pick immediately. Moving the mouse with the
    initial keys or buttons held down keeps picking with a little
    preview square appearing.

    The third method doesn't pick immediately: you have to click on the
    canvas to start picking.

    While the preview square is visible, it's possible to pick outside
    the window. This "hidden" functionality may not work at all with
    more modern window managers and DEs, and may be removed if it proves
    slow or faulty.

    """
    # Class configuration
    ACTION_NAME = 'ColorPickMode'

    # Keyboard activation behaviour (instance defaults)
    # See keyboard.py and doc.mode_flip_action_activated_cb()
    keyup_timeout = 0   # don't change behaviour by timeout

    pointer_behavior = gui.mode.Behavior.EDIT_OBJECTS
    scroll_behavior = gui.mode.Behavior.NONE
    # XXX ^^^^^^^ grabs ptr, so no CHANGE_VIEW
    supports_button_switching = False

    @property
    def inactive_cursor(self):
        return self.doc.app.cursor_color_picker

    @classmethod
    def get_name(cls):
        return _(u"Pick Color")

    def get_usage(self):
        return _(u"Set the color used for painting")

    def __init__(self, ignore_modifiers=False, **kwds):
        super(ColorPickMode, self).__init__(**kwds)
        self._overlay = None
        self._started_from_key_press = ignore_modifiers
        self._start_drag_on_next_motion_event = False
        self._color = None
        self._queued_data = None
        self._timeout_id = None

    def enter(self, doc, **kwds):
        """Enters the mode, arranging for necessary grabs ASAP"""
        super(ColorPickMode, self).enter(doc, **kwds)
        self._color = doc.app.brush_color_manager.get_color()
        if self._started_from_key_press:
            # Pick now using the last recorded event position
            doc = self.doc
            tdw = self.doc.tdw
            t, x, y = doc.get_last_event_info(tdw)
            if None not in (x, y):
                self._pick_color(tdw, x, y, direct=True)
            # Start the drag when possible
            self._start_drag_on_next_motion_event = True

    def leave(self, **kwds):
        if self._queued_data and not self._timeout_id:
            self._change_color()
        self._remove_overlay()
        super(ColorPickMode, self).leave(**kwds)

    def button_press_cb(self, tdw, event):
        self._pick_color(tdw, event.x, event.y, direct=True)
        # Supercall will start the drag normally
        self._start_drag_on_next_motion_event = False
        return super(ColorPickMode, self).button_press_cb(tdw, event)

    def motion_notify_cb(self, tdw, event):
        if self._start_drag_on_next_motion_event:
            self._start_drag(tdw, event)
            self._start_drag_on_next_motion_event = False
        return super(ColorPickMode, self).motion_notify_cb(tdw, event)

    def drag_stop_cb(self, tdw):
        self._remove_overlay()
        super(ColorPickMode, self).drag_stop_cb(tdw)

    def drag_update_cb(self, tdw, event, ev_x, ev_y, dx, dy):
        self._pick_color(tdw, ev_x, ev_y)
        self._update_overlay(tdw, ev_x, ev_y, self._color)
        return super(ColorPickMode, self).drag_update_cb(
            tdw, event, ev_x, ev_y, dx, dy)

    def _update_overlay(self, tdw, x, y, col):
        if self._overlay is None:
            self._overlay = ColorPickPreviewOverlay(self.doc, tdw, x, y, col)
        else:
            self._overlay.update(x, y, col)

    def _remove_overlay(self):
        if self._overlay is not None:
            self._overlay.cleanup()
            self._overlay = None

    def get_options_widget(self):
        return None

    def get_new_color(self, pick_color, brush_color):
        # Normal pick mode, but preserve hue for achromatic colors.
        pick_h, pick_s, pick_v = pick_color.get_hsv()
        if pick_s == 0 or pick_v == 0:
            return HSVColor(brush_color.h, pick_s, pick_v)
        else:
            return pick_color

    def _pick_color(self, tdw, x, y, direct=False):
        cm = self.doc.app.brush_color_manager
        pick_color = tdw.pick_color(x, y)
        if pick_color != self._color:
            new_color = self.get_new_color(pick_color, self._color)
            self._color = new_color
            if direct:
                cm.set_color(new_color)
            else:
                self._queue_color_change(new_color, cm)

    def _queue_color_change(self, new_col, cm):
        self._queued_data = new_col, cm
        if not self._timeout_id:
            self._timeout_id = GLib.timeout_add(
                interval=16.66,  # 60 fps cap
                function=self._change_color,
            )

    def _change_color(self):
        if self._queued_data:
            col, cm = self._queued_data
            self._queued_data = None
            cm.set_color(col)
        self._timeout_id = None


# HCY chromaticity thresholds

_C_MIN = 0.0001
_Y_MIN = 0.0001
_Y_MAX = 0.9999


class ColorPickModeHCYBase (ColorPickMode):

    def get_new_color(self, pick_color, brush_color):
        new_col_hcy = self.get_new_hcy_color(
            HCYColor(color=pick_color), HCYColor(color=brush_color))
        new_col_hcy.c = max(_C_MIN, new_col_hcy.c)
        new_col_hcy.y = min(_Y_MAX, max(_Y_MIN, new_col_hcy.y))
        return new_col_hcy

    def get_new_hcy_color(self, *args):
        raise NotImplementedError


class ColorPickModeH (ColorPickModeHCYBase):

    # Class configuration
    ACTION_NAME = 'ColorPickModeH'

    @property
    def inactive_cursor(self):
        return self.doc.app.cursor_color_picker_h

    @classmethod
    def get_name(cls):
        return _(u"Pick Hue")

    def get_usage(self):
        return _(u"Set the color Hue used for painting")

    def get_new_hcy_color(self, pick_hcy, brush_hcy):
        if pick_hcy.c >= _C_MIN and pick_hcy.y >= _Y_MIN:
            brush_hcy.h = pick_hcy.h
        return brush_hcy


class ColorPickModeC (ColorPickModeHCYBase):
    # Class configuration
    ACTION_NAME = 'ColorPickModeC'

    @property
    def inactive_cursor(self):
        return self.doc.app.cursor_color_picker_c

    @classmethod
    def get_name(cls):
        return _(u"Pick Chroma")

    def get_usage(self):
        return _(u"Set the color Chroma used for painting")

    def get_new_hcy_color(self, pick_hcy, brush_hcy):
        brush_hcy.c = pick_hcy.c
        return brush_hcy


class ColorPickModeY (ColorPickModeHCYBase):
    # Class configuration
    ACTION_NAME = 'ColorPickModeY'

    @property
    def inactive_cursor(self):
        return self.doc.app.cursor_color_picker_y

    @classmethod
    def get_name(cls):
        return _(u"Pick Luma")

    def get_usage(self):
        return _(u"Set the color Luma used for painting")

    def get_new_hcy_color(self, pick_hcy, brush_hcy):
        brush_hcy.y = pick_hcy.y
        return brush_hcy


class ColorPickPreviewOverlay (Overlay):
    """Preview overlay during color picker mode.

    This is only shown when dragging the pointer with a button or the
    hotkey held down, to avoid flashing and distraction.

    """

    PREVIEW_SIZE = 70
    OUTLINE_WIDTH = 3
    CORNER_RADIUS = 10

    def __init__(self, doc, tdw, x, y, color):
        """Initialize, attaching to the brush and to the tdw.

        Observer callbacks and canvas overlays are registered by this
        constructor, so cleanup() must be called when the owning mode leave()s.

        """
        Overlay.__init__(self)
        self._doc = doc
        self._tdw = tdw
        self._x = int(x)+0.5
        self._y = int(y)+0.5
        self._color = color
        alloc = tdw.get_allocation()
        self._tdw_w = alloc.width
        self._tdw_h = alloc.height
        tdw.display_overlays.append(self)
        self._previous_area = None
        self._queue_tdw_redraw()

    def cleanup(self):
        """Cleans up temporary observer stuff, allowing garbage collection.
        """
        self._tdw.display_overlays.remove(self)
        self._queue_tdw_redraw()

    def update(self, x, y, color):
        """Update the overlay's position and color"""
        self._x = int(x)+0.5
        self._y = int(y)+0.5
        self._color = color
        self._queue_tdw_redraw()

    def _queue_tdw_redraw(self):
        if self._previous_area is not None:
            self._tdw.queue_draw_area(*self._previous_area)
            self._previous_area = None
        area = self._get_area()
        if area is not None:
            self._tdw.queue_draw_area(*area)

    def _get_area(self):
        # Returns the drawing area for the square
        size = self.PREVIEW_SIZE

        # Start with the pointer location
        x = self._x
        y = self._y

        offset = size // 2

        # Only show if the pointer is inside the tdw
        alloc = self._tdw.get_allocation()
        if x < 0 or y < 0 or y > alloc.height or x > alloc.width:
            return None

        # Convert to preview location
        # Pick a direction - N,W,E,S - in which to offset the preview
        if y + size > alloc.height - offset:
            x -= offset
            y -= size + offset
        elif x < offset:
            x += offset
            y -= offset
        elif x > alloc.width - offset:
            x -= size + offset
            y -= offset
        else:
            x -= offset
            y += offset

        ## Correct to place within the tdw
        #   if x < 0:
        #       x = 0
        #   if y < 0:
        #       y = 0
        #   if x + size > alloc.width:
        #       x = alloc.width - size
        #   if y + size > alloc.height:
        #       y = alloc.height - size

        return (int(x), int(y), size, size)

    def paint(self, cr):
        area = self._get_area()
        if area is not None:
            x, y, w, h = area

            cr.set_source_rgb(*self._color.get_rgb())
            x += (self.OUTLINE_WIDTH // 2) + 1.5
            y += (self.OUTLINE_WIDTH // 2) + 1.5
            w -= self.OUTLINE_WIDTH + 3
            h -= self.OUTLINE_WIDTH + 3
            rounded_box(cr, x, y, w, h, self.CORNER_RADIUS)
            cr.fill_preserve()

            cr.set_source_rgb(0, 0, 0)
            cr.set_line_width(self.OUTLINE_WIDTH)
            cr.stroke()

        self._previous_area = area
