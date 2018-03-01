# This file is part of MyPaint.
# Copyright (C) 2010-2018 by the MyPaint Development Team
# Copyright (C) 2009-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

from __future__ import division, print_function
from gettext import gettext as _
import colorsys

import gui.mode
from .overlays import Overlay
from .overlays import rounded_box
import lib.color


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
    PICK_SIZE = 6

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

    def __init__(self, ignore_modifiers=False, pickmode="PickAll", **kwds):
        super(ColorPickMode, self).__init__(**kwds)
        self._overlay = None
        self._started_from_key_press = ignore_modifiers
        self._start_drag_on_next_motion_event = False
        self._pickmode = pickmode

    def enter(self, doc, **kwds):
        """Enters the mode, arranging for necessary grabs ASAP"""
        super(ColorPickMode, self).enter(doc, **kwds)
        if self._started_from_key_press:
            # Pick now using the last recorded event position
            doc = self.doc
            tdw = self.doc.tdw
            t, x, y = doc.get_last_event_info(tdw)
            if None not in (x, y):
                self._pick_color_mode(tdw, x, y, self._pickmode)
            # Start the drag when possible
            self._start_drag_on_next_motion_event = True
            self._needs_drag_start = True

    def leave(self, **kwds):
        self._remove_overlay()
        super(ColorPickMode, self).leave(**kwds)

    def button_press_cb(self, tdw, event):
        self._pick_color_mode(tdw, event.x, event.y, self._pickmode)
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

    def drag_update_cb(self, tdw, event, dx, dy):
        self._pick_color_mode(tdw, event.x, event.y, self._pickmode)
        self._place_overlay(tdw, event.x, event.y)
        return super(ColorPickMode, self).drag_update_cb(tdw, event, dx, dy)

    def _place_overlay(self, tdw, x, y):
        if self._overlay is None:
            self._overlay = ColorPickPreviewOverlay(self.doc, tdw, x, y)
        else:
            self._overlay.move(x, y)

    def _remove_overlay(self):
        if self._overlay is None:
            return
        self._overlay.cleanup()
        self._overlay = None

    def get_options_widget(self):
        return None

    def _pick_color_mode(self, tdw, x, y, mode):
        # Init shared variables between normal and HCY modes.
        pickcolor = tdw.pick_color(x, y)
        brushcolor = self._get_app_brush_color()
        brushcolor_rgb = brushcolor.get_rgb()
        pickcolor_rgb = pickcolor.get_rgb()

        # If brush and pick colors are the same, nothing to do.
        if brushcolor_rgb != pickcolor_rgb:
            pickcolor_hsv = pickcolor.get_hsv()
            brushcolor_hsv = brushcolor.get_hsv()
            cm = self.doc.app.brush_color_manager
            c_min = 0.0001
            y_min = 0.0001
            y_max = 0.9999

            # Normal pick mode, but preserve hue for achromatic colors.
            # Easy because we are staying in HSV
            # and not converting to another color space.

            if mode == "PickAll":
                if (pickcolor_hsv[1] == 0) or (pickcolor_hsv[2] == 0):
                    brushcolornew_hsv = lib.color.HSVColor(
                        brushcolor_hsv[0],
                        pickcolor_hsv[1],
                        pickcolor_hsv[2],
                    )
                    pickcolor = brushcolornew_hsv
                cm.set_color(pickcolor)
            else:
                # Pick H, C, or Y independently.
                # Deal with scenarios to avoid achromatic conversions
                # from HCY to RGB to HSV losing hue information.
                # Basically prevent C from reaching 0
                # and Y from reaching 0.0 or 1.0.
                brushcolor_hcy = lib.color.RGB_to_HCY(brushcolor_rgb)
                pickcolor_hcy = lib.color.RGB_to_HCY(pickcolor_rgb)

                if mode == "PickHue":
                    brushcolornew_hcy = (
                        pickcolor_hcy[0],
                        brushcolor_hcy[1],
                        brushcolor_hcy[2],
                    )
                    if brushcolor_hcy[1] < c_min:
                        brushcolornew_hcy = (
                            brushcolornew_hcy[0],
                            c_min,
                            brushcolornew_hcy[2],
                        )
                    if pickcolor_hcy[2] < y_min:
                        brushcolornew_hcy = (
                            brushcolornew_hcy[0],
                            brushcolornew_hcy[1],
                            y_min,
                        )
                    if pickcolor_hcy[2] > y_max:
                        brushcolornew_hcy = (
                            brushcolornew_hcy[0],
                            brushcolornew_hcy[1],
                            y_max,
                        )
                    if (pickcolor_hsv[1] == 0) or (pickcolor_hsv[2] == 0):
                        brushcolornew_hcy = (
                            brushcolor_hsv[0],
                            brushcolor_hcy[1],
                            brushcolor_hcy[2],
                        )
                elif mode == "PickLuma":
                    brushcolornew_hcy = (
                        brushcolor_hsv[0],
                        brushcolor_hcy[1],
                        pickcolor_hcy[2],
                    )
                    if brushcolor_hcy[1] < c_min:
                        brushcolornew_hcy = (
                            brushcolornew_hcy[0],
                            c_min,
                            pickcolor_hcy[2],
                        )
                    if pickcolor_hcy[2] < y_min:
                        brushcolornew_hcy = (
                            brushcolornew_hcy[0],
                            brushcolornew_hcy[1],
                            y_min,
                        )
                    if pickcolor_hcy[2] > y_max:
                        brushcolornew_hcy = (
                            brushcolornew_hcy[0],
                            brushcolornew_hcy[1],
                            y_max,
                        )
                elif mode == "PickChroma":
                    brushcolornew_hcy = (
                        brushcolor_hsv[0],
                        pickcolor_hcy[1],
                        brushcolor_hcy[2],
                    )
                    if pickcolor_hcy[1] < c_min:
                        brushcolornew_hcy = (
                            brushcolornew_hcy[0],
                            c_min,
                            brushcolornew_hcy[2],
                        )
                    if brushcolor_hcy[2] < y_min:
                        brushcolornew_hcy = (
                            brushcolornew_hcy[0],
                            brushcolornew_hcy[1],
                            y_min,
                        )
                    if brushcolor_hcy[2] > y_max:
                        brushcolornew_hcy = (
                            brushcolornew_hcy[0],
                            brushcolornew_hcy[1],
                            y_max,
                        )

                brushcolornew = lib.color.HCY_to_RGB(brushcolornew_hcy)
                brushcolornew = colorsys.rgb_to_hsv(*brushcolornew)
                brushcolornew = lib.color.HSVColor(*brushcolornew)
                cm.set_color(brushcolornew)
        return None

    def _get_app_brush_color(self):
        app = self.doc.app
        return lib.color.HSVColor(*app.brush.get_color_hsv())


class ColorPickModeH (ColorPickMode):

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

    def __init__(self, ignore_modifiers=False, pickmode="PickHue", **kwds):
        super(ColorPickModeH, self).__init__(**kwds)
        self._overlay = None
        self._started_from_key_press = ignore_modifiers
        self._start_drag_on_next_motion_event = False
        self._pickmode = pickmode


class ColorPickModeC (ColorPickMode):
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

    def __init__(self, ignore_modifiers=False, pickmode="PickChroma", **kwds):
        super(ColorPickModeC, self).__init__(**kwds)
        self._overlay = None
        self._started_from_key_press = ignore_modifiers
        self._start_drag_on_next_motion_event = False
        self._pickmode = pickmode


class ColorPickModeY (ColorPickMode):
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

    def __init__(self, ignore_modifiers=False, pickmode="PickLuma", **kwds):
        super(ColorPickModeY, self).__init__(**kwds)
        self._overlay = None
        self._started_from_key_press = ignore_modifiers
        self._start_drag_on_next_motion_event = False
        self._pickmode = pickmode


class ColorPickPreviewOverlay (Overlay):
    """Preview overlay during color picker mode.

    This is only shown when dragging the pointer with a button or the
    hotkey held down, to avoid flashing and distraction.

    """

    PREVIEW_SIZE = 70
    OUTLINE_WIDTH = 3
    CORNER_RADIUS = 10

    def __init__(self, doc, tdw, x, y):
        """Initialize, attaching to the brush and to the tdw.

        Observer callbacks and canvas overlays are registered by this
        constructor, so cleanup() must be called when the owning mode leave()s.

        """
        Overlay.__init__(self)
        self._doc = doc
        self._tdw = tdw
        self._x = int(x)+0.5
        self._y = int(y)+0.5
        alloc = tdw.get_allocation()
        self._tdw_w = alloc.width
        self._tdw_h = alloc.height
        self._color = self._get_app_brush_color()
        app = doc.app
        app.brush.observers.append(self._brush_color_changed_cb)
        tdw.display_overlays.append(self)
        self._previous_area = None
        self._queue_tdw_redraw()

    def cleanup(self):
        """Cleans up temporary observer stuff, allowing garbage collection.
        """
        app = self._doc.app
        app.brush.observers.remove(self._brush_color_changed_cb)
        self._tdw.display_overlays.remove(self)
        assert self._brush_color_changed_cb not in app.brush.observers
        assert self not in self._tdw.display_overlays
        self._queue_tdw_redraw()

    def move(self, x, y):
        """Moves the preview square to a new location, in tdw pointer coords.
        """
        self._x = int(x)+0.5
        self._y = int(y)+0.5
        self._queue_tdw_redraw()

    def _get_app_brush_color(self):
        app = self._doc.app
        return lib.color.HSVColor(*app.brush.get_color_hsv())

    def _brush_color_changed_cb(self, settings):
        if not settings.intersection(('color_h', 'color_s', 'color_v')):
            return
        self._color = self._get_app_brush_color()
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
