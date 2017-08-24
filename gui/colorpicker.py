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

import time

from gettext import gettext as _
import colorsys
import colour
import numpy as np
import cairo
import copy
import logging

import gui.mode
from .overlays import Overlay
from .overlays import rounded_box, rounded_box_hole
from lib.color import (CAM16Color, HSVColor, HCYColor,
                       RGB_to_CCT, CCT_to_RGB, PigmentColor,
                       RGBColor, color_diff)

logger = logging.getLogger(__name__)

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
    MIN_PREVIEW_SIZE = 70

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
        self.app = gui.application.get_app()
        # interactive blend mode blends brush+canvas using ratio
        # based on distance from starting position
        self.starting_position = None
        self.starting_color = None
        self.blending_color = None
        self.blending_ratio = None

    def enter(self, doc, **kwds):
        """Enters the mode, arranging for necessary grabs ASAP"""
        super(ColorPickMode, self).enter(doc, **kwds)
        if self._started_from_key_press:
            # Pick now using the last recorded event position
            doc = self.doc
            tdw = self.doc.tdw
            t, x, y, p = doc.get_last_event_info(tdw)

            if None not in (x, y):
                self.starting_position = (x, y)
                self._pick_color_mode(tdw, x, y, self._pickmode)
            # Start the drag when possible
            self._start_drag_on_next_motion_event = True
            self._needs_drag_start = True

    def leave(self, **kwds):
        self._remove_overlay()
        # if we're interactively blending, set the brushcolor when leaving
        if self.blending_color is not None:
            app = self.doc.app
            app.brush.set_color_hsv(self.blending_color.get_hsv())
            app.brush.set_cam16_color(CAM16Color(
                                       color=self.blending_color))
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
            self._overlay = ColorPickPreviewOverlay(
                self.doc, tdw, x, y, self._pickmode,
                blending_color=self.blending_color,
                blending_ratio=self.blending_ratio)
        else:
            # don't move if pick and blend
            if self._pickmode != "PickandBlend":
                self._overlay.move(x, y)
            if self._pickmode == "PickandBlend":
                self._overlay.blending_color = self.blending_color
                self._overlay.blending_ratio = self.blending_ratio
                self._overlay._queue_tdw_redraw()

    def _remove_overlay(self):
        if self._overlay is None:
            return
        self._overlay.cleanup()
        self._overlay = None

    def get_options_widget(self):
        return None

    def _pick_color_mode(self, tdw, x, y, mode):
        # init shared variables between pick modes
        doc = self.doc
        tdw = self.doc.tdw
        app = self.doc.app
        p = self.app.preferences
        elapsed = None
        t, x, y, pressure = doc.get_last_event_info(tdw)
        # TODO configure static pressure as a slider?
        # This would allow non-pressure devices to use
        # pressures besides 50% for brushes too
        if pressure is None:
            pressure = 0.5
        if p['color.pick_blend_use_pressure'] is False:
            pressure = 0.0
        if t <= doc.last_colorpick_time:
            t = (time.time() * 1000)

        # limit rate for performance
        min_wait = p['color.adjuster_min_wait']
        if doc.last_colorpick_time:
            elapsed = t - doc.last_colorpick_time
            if elapsed < min_wait:
                return

        cm = app.brush_color_manager
        prefs = cm.get_prefs()
        illuminant = prefs['color.dimension_illuminant']
        tune_model = prefs['color.tune_model']

        if illuminant == "custom_XYZ":
            illuminant = prefs['color.dimension_illuminant_XYZ']
        else:
            illuminant = colour.ILLUMINANTS['cie_2_1931'][illuminant]

        doc.last_colorpick_time = t
        pickcolor = tdw.pick_color(x, y, size=int(3/tdw.renderer.scale))
        brushcolor = copy.copy(self.app.brush.CAM16Color)
        brushcolor_rgb = brushcolor.get_rgb()
        pickcolor_rgb = pickcolor.get_rgb()

        # if brush and pick colors are the same, nothing to do
        if brushcolor_rgb != pickcolor_rgb:
            pickcolor_hsv = pickcolor.get_hsv()
            brushcolor_hsv = brushcolor.get_hsv()
            cm = self.doc.app.brush_color_manager
            try:
                color_class = eval(tune_model + 'Color')
            except:
                logger.error('Incorrect color model "%s"' % tune_model)
                return

            # normal pick mode
            if mode == "PickAll":
                cm.set_color(pickcolor)
            elif mode == "PickIlluminant":
                ill = colour.sRGB_to_XYZ(np.array(pickcolor_rgb))*100
                if ill[1] <= 0:
                    return
                fac = 1/ill[1]*100

                p['color.dimension_illuminant'] = "custom_XYZ"
                p['color.dimension_illuminant_XYZ'] = (
                    ill[0]*fac,
                    ill[1]*fac,
                    ill[2]*fac
                )
                # update pref ui
                app.preferences_window.update_ui()

                # reset the brush color with the same color
                # under the new illuminant
                brushcolor.illuminant = ill * fac

                app.brush.set_color_hsv(brushcolor.get_hsv())
                app.brush.set_cam16_color(brushcolor)
            elif mode == "PickandBlend":
                alloc = self.doc.tdw.get_allocation()
                size = max(int(p['color.preview_size'] * .01 *  alloc.height),
                           self.MIN_PREVIEW_SIZE)
                if self.starting_color is None:
                    self.starting_color = pickcolor
                dist = np.linalg.norm(
                    np.array(self.starting_position) - np.array((x, y)))
                dist = np.clip(dist / size + pressure, 0, 1)
                if p['color.pick_blend_reverse'] is True:
                    dist = 1 - dist
                self.blending_ratio = dist
                brushcolor_start = color_class(color=brushcolor)
                pickcolor = color_class(
                    color=self.starting_color)
                self.blending_color = brushcolor_start.mix(pickcolor, dist)
            elif mode == "PickTarget":
                doc.last_color_target = pickcolor
            else:
                # pick V, S, H independently
                if tune_model == 'HSV':
                    brushcolornew = color_class(color=brushcolor)
                    pickcolor = color_class(color=pickcolor)
                    if mode == "PickHue":
                        brushcolornew.h = pickcolor.h
                    elif mode == "PickLuma":
                        brushcolornew.v = pickcolor.v
                    elif mode == "PickChroma":
                        brushcolornew.s = pickcolor.s
                elif tune_model == 'HCY':
                    brushcolornew = color_class(color=brushcolor)
                    pickcolor = color_class(color=pickcolor)
                    if mode == "PickHue":
                        brushcolornew.h = pickcolor.h
                    elif mode == "PickLuma":
                        brushcolornew.y = pickcolor.y
                    elif mode == "PickChroma":
                        brushcolornew.c = pickcolor.c
                elif tune_model == 'CAM16' or tune_model == 'Pigment':
                    brushcolornew = CAM16Color(color=brushcolor)
                    pickcolor = CAM16Color(color=pickcolor)
                    if mode == "PickHue":
                        brushcolornew.h = pickcolor.h
                    elif mode == "PickLuma":
                        brushcolornew.v = pickcolor.v
                    elif mode == "PickChroma":
                        brushcolornew.s = pickcolor.s

                app.brush.set_color_hsv(brushcolornew.get_hsv())
                app.brush.set_cam16_color(CAM16Color(color=brushcolornew))

        return None


class ColorPickModeH(ColorPickMode):

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


class ColorPickModeBlend (ColorPickMode):
    # Class configuration
    ACTION_NAME = 'ColorPickModeBlend'

    @property
    def inactive_cursor(self):
        return self.doc.app.cursor_color_picker

    @classmethod
    def get_name(cls):
        return _(u"Pick and Blend")

    def get_usage(self):
        return _(u"Blend the canvas color with brush color-> drag distance")

    def __init__(self, ignore_modifiers=False, pickmode="PickandBlend",
                 **kwds):
        super(ColorPickModeBlend, self).__init__(**kwds)
        self._overlay = None
        self._started_from_key_press = ignore_modifiers
        self._start_drag_on_next_motion_event = False
        self._pickmode = pickmode
        self.starting_position = None


class ColorPickModeSetTarget (ColorPickMode):
    # Class configuration
    ACTION_NAME = 'ColorPickModeSetTarget'

    @property
    def inactive_cursor(self):
        return self.doc.app.cursor_color_picker

    @classmethod
    def get_name(cls):
        return _(u"Set Color Target")

    def get_usage(self):
        return _(u"Set the target color for re-saturation")

    def __init__(self, ignore_modifiers=False, pickmode="PickTarget",
                 **kwds):
        super(ColorPickModeSetTarget, self).__init__(**kwds)
        self._overlay = None
        self._started_from_key_press = ignore_modifiers
        self._start_drag_on_next_motion_event = False
        self._pickmode = pickmode
        self.starting_position = None


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


class ColorPickModeIlluminant(ColorPickMode):
    # Class configuration
    ACTION_NAME = 'ColorPickModeIlluminant'

    @property
    def inactive_cursor(self):
        return self.doc.app.cursor_color_picker_illuminant

    @classmethod
    def get_name(cls):
        return _(u"Pick Illuminant")

    def get_usage(self):
        return _(u"Set the illuminant used for color adjusters")

    def __init__(self, ignore_modifiers=False, pickmode="PickIlluminant",
                 **kwds):
        super(ColorPickModeIlluminant, self).__init__(**kwds)
        self._overlay = None
        self._started_from_key_press = ignore_modifiers
        self._start_drag_on_next_motion_event = False
        self._pickmode = pickmode


class ColorPickPreviewOverlay (Overlay):
    """Preview overlay during color picker mode.

    This is only shown when dragging the pointer with a button or the
    hotkey held down, to avoid flashing and distraction.

    """
    # make relative sizes
    MIN_PREVIEW_SIZE = 70
    OUTLINE_WIDTH = 3

    def __init__(self, doc, tdw, x, y, pickmode,
                 blending_color=None, blending_ratio=None):
        """Initialize, attaching to the brush and to the tdw.

        Observer callbacks and canvas overlays are registered by this
        constructor, so cleanup() must be called when the owning mode leave()s.

        """
        Overlay.__init__(self)
        self._pickmode = pickmode
        self.app = gui.application.get_app()
        p = self.app.preferences
        self.preview_size = p['color.preview_size']
        self.blending_color = blending_color
        self.blending_ratio = blending_ratio
        self._doc = doc
        self._tdw = tdw
        self._x = int(x)+0.5
        self._y = int(y)+0.5
        alloc = tdw.get_allocation()
        self._tdw_w = alloc.width
        self._tdw_h = alloc.height
        self.corner_radius = None
        self._color = copy.copy(self.app.brush.CAM16Color)
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
        return HSVColor(*app.brush.get_color_hsv())

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
        alloc = self._tdw.get_allocation()
        if self._pickmode == "PickandBlend":
            size = max(int(self.preview_size * .01 * alloc.height * 2),
                       self.MIN_PREVIEW_SIZE * 2)
        else:
            size = max(int(self.preview_size * .01 * alloc.height),
                       self.MIN_PREVIEW_SIZE)
        self.corner_radius = size * 0.1
        # Start with the pointer location
        x = self._x
        y = self._y

        offset = size // 2

        # Only show if the pointer is inside the tdw
        if x < 0 or y < 0 or y > alloc.height or x > alloc.width:
            return None

        # Convert to preview location
        # Pick a direction - N,W,E,S - in which to offset the preview
        if self._pickmode == "PickandBlend":
            x -= offset
            y -= offset
        else:
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
            # if we're picking an illuminant splash that instead of brush color
            if self._pickmode == "PickIlluminant":
                p = self.app.preferences
                xyz = p['color.dimension_illuminant_XYZ']
                ill = colour.XYZ_to_sRGB(np.array(xyz)/100.0)
                cr.set_source_rgb(*ill)
            elif self.blending_color is not None:
                cr.set_source_rgb(*self.blending_color.get_rgb())
            elif self._pickmode == "PickTarget":
                cr.set_source_rgb(*self._doc.last_color_target.get_rgb())
            else:
                cr.set_source_rgb(*self._color.get_rgb())
            x += (self.OUTLINE_WIDTH // 2) + 1.5
            y += (self.OUTLINE_WIDTH // 2) + 1.5
            w -= self.OUTLINE_WIDTH + 3
            h -= self.OUTLINE_WIDTH + 3

            if self._pickmode == "PickandBlend":
                rounded_box_hole(cr, x, y, w, h, self.corner_radius)
                cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            else:
                rounded_box(cr, x, y, w, h, self.corner_radius)
            cr.fill_preserve()
            # don't outline when blending, will detract from
            # color comparisons
            if self.blending_ratio is None:
                cr.set_source_rgb(0, 0, 0)
                cr.set_line_width(self.OUTLINE_WIDTH)
            cr.stroke()

        self._previous_area = area
