# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Overlays for TDWs showing information about the TDW state."""


## Imports

from __future__ import division, print_function
from math import pi
from gettext import gettext as _

from gi.repository import Pango
from gi.repository import PangoCairo
from gi.repository import GLib
import cairo

from lib.helpers import clamp
import gui.style


## Base classes and utils

class Overlay (object):
    """Base class/interface for objects which paint things over a TDW."""

    def paint(self, cr):
        """Paint information onto a TiledDrawWidget.

        The drawing interface is very simple. `cr` is a Cairo context in either
        display coordinates or model coordinates: which one you get depends on
        which list the Overlay is appended to on its tdw.
        """
        pass


class FadingOverlay (Overlay):
    """Base class for temporary overlays which fade to alpha over a short time
    """

    # Overridable animation controls
    fade_fps = 20  #: Nominal frames per second
    fade_duration = 1.5  #: Time for fading entirely to zero, in seconds

    # Animation and redrawing state
    tdw = None
    alpha = 1.0
    __area = None
    __anim_srcid = None

    def __init__(self, doc):
        Overlay.__init__(self)
        self.tdw = doc.tdw

    def paint(self, cr):
        """Repaint the overlay and start animating if necessary.

        Individual frames are handled by `paint_frame()`.
        """
        if self.overlay_changed():
            self.alpha = 1.0
        elif self.alpha <= 0:
            # No need to draw anything.
            return
        self.__restart_anim_if_needed()
        self.__area = self.paint_frame(cr)

    def anim_cb(self):
        """Animation callback.

        Each step fades the alpha multiplier slightly and invalidates the area
        last painted.
        """
        self.alpha -= 1 / (self.fade_fps * self.fade_duration)
        self.alpha = clamp(self.alpha, 0.0, 1.0)

        if self.__area:
            self.tdw.queue_draw_area(*self.__area)
        if self.alpha <= 0.0:
            self.__anim_srcid = None
            return False
        else:
            return True

    def __restart_anim_if_needed(self):
        """Restart if not currently running, without changing the alpha.
        """
        if self.__anim_srcid is None:
            delay = int(1000 // self.fade_fps)
            self.__anim_srcid = GLib.timeout_add(delay, self.anim_cb)

    def stop_anim(self):
        """Stops the animation after the next frame is drawn.
        """
        self.alpha = 0.0

    def start_anim(self):
        """Restarts the animation, setting alpha to 1.
        """
        self.alpha = 1.0
        self.__restart_anim_if_needed()

    def paint_frame(self, cr):
        """Paint a single frame.
        """
        raise NotImplementedError

    def overlay_changed(self):
        """Return true if the overlay has changed.

        This virtual method is called by paint() to determine whether the
        alpha should be reset to 1.0 and the fade begun anew.
        """
        raise NotImplementedError


def rounded_box(cr, x, y, w, h, r):
    """Paint a rounded box path into a Cairo context.

    The position is given by `x` and `y`, and the size by `w` and `h`. The
    cornders are of radius `r`, and must be smaller than half the minimum
    dimension. The path is created as a new, closed subpath.
    """
    assert r <= min(w, h) / 2
    cr.new_sub_path()
    cr.arc(x+r, y+r, r, pi, pi*1.5)
    cr.line_to(x+w-r, y)
    cr.arc(x+w-r, y+r, r, pi*1.5, pi*2)
    cr.line_to(x+w, y+h-r)
    cr.arc(x+w-r, y+h-r, r, 0, pi*0.5)
    cr.line_to(x+r, y+h)
    cr.arc(x+r, y+h-r, r, pi*0.5, pi)
    cr.close_path()


def rounded_box_hole(cr, x, y, w, h, r):
    """Paint a rounded box path with a hole into a Cairo context.

    The position is given by `x` and `y`, and the size by `w` and `h`. The
    cornders are of radius `r`, and must be smaller than half the minimum
    dimension. The path is created as a new, nested closed subpaths
    """
    assert r <= min(w, h) / 2
    cr.new_sub_path()
    cr.arc(x+r, y+r, r, pi, pi*1.5)
    cr.line_to(x+w-r, y)
    cr.arc(x+w-r, y+r, r, pi*1.5, pi*2)
    cr.line_to(x+w, y+h-r)
    cr.arc(x+w-r, y+h-r, r, 0, pi*0.5)
    cr.line_to(x+r, y+h)
    cr.arc(x+r, y+h-r, r, pi*0.5, pi)
    cr.close_path()

    cr.new_sub_path()
    cr.arc(x+w/2, y+h/2, w/4, 0, 2*pi)
    cr.close_path()

## Minor builtin overlays


class ScaleOverlay (FadingOverlay):
    """Overlays its TDW's current zoom, fading to transparent.

    The animation is started by the normal full canvas repaint which happens
    after the scale changes.
    """

    vmargin = 6
    hmargin = 12
    padding = 6
    shown_scale = None

    def overlay_changed(self):
        return self.tdw.scale != self.shown_scale

    def paint_frame(self, cr):
        self.shown_scale = self.tdw.scale
        text = _("Zoom: %.01f%%") % (100*self.shown_scale)
        layout = self.tdw.create_pango_layout(text)

        # Set a bold font
        font = layout.get_font_description()
        if font is None:  # inherited from context
            font = layout.get_context().get_font_description()
            font = font.copy()
        font.set_weight(Pango.Weight.BOLD)
        layout.set_font_description(font)

        # General dimensions
        alloc = self.tdw.get_allocation()
        lw, lh = layout.get_pixel_size()

        # Background rectangle
        hm = self.hmargin
        vm = self.hmargin
        w = alloc.width
        p = self.padding
        area = bx, by, bw, bh = w-lw-hm-p-p, vm, lw+p+p, lh+p+p
        rounded_box(cr, bx, by, bw, bh, p)
        rgba = list(gui.style.TRANSIENT_INFO_BG_RGBA)
        rgba[3] *= self.alpha
        cr.set_source_rgba(*rgba)
        cr.fill()

        # Text
        cr.translate(w-lw-hm-p, vm+p)
        rgba = list(gui.style.TRANSIENT_INFO_RGBA)
        rgba[3] *= self.alpha
        cr.set_source_rgba(*rgba)
        PangoCairo.show_layout(cr, layout)

        # Where to invalidate
        return area


class LastPaintPosOverlay (FadingOverlay):
    """Displays the last painting position after a stroke has finished.

    Not especially useful, but serves as an example of how to drive an overlay
    from user input events.
    """

    inner_line_rgba = gui.style.TRANSIENT_INFO_RGBA
    inner_line_width = 6
    outer_line_rgba = gui.style.TRANSIENT_INFO_BG_RGBA
    outer_line_width = 8
    radius = 4.0

    def __init__(self, doc):
        FadingOverlay.__init__(self, doc)
        doc.input_stroke_started += self.input_stroke_started
        doc.input_stroke_ended += self.input_stroke_ended
        self.current_marker_pos = None
        self.in_input_stroke = False

    def input_stroke_started(self, doc, event):
        self.in_input_stroke = True
        if self.current_marker_pos is None:
            return
        # Clear the current marker
        model_x, model_y = self.current_marker_pos
        x, y = self.tdw.model_to_display(model_x, model_y)
        area = self._calc_area(x, y)
        self.tdw.queue_draw_area(*area)
        self.current_marker_pos = None
        self.stop_anim()

    def input_stroke_ended(self, doc, event):
        self.in_input_stroke = False
        if self.tdw.last_painting_pos is None:
            return
        # Record the new marker position
        model_x, model_y = self.tdw.last_painting_pos
        x, y = self.tdw.model_to_display(model_x, model_y)
        area = self._calc_area(x, y)
        self.tdw.queue_draw_area(*area)
        self.current_marker_pos = model_x, model_y
        self.start_anim()

    def overlay_changed(self):
        return False

    def _calc_area(self, x, y):
        r = self.radius
        lw = max(self.inner_line_width, self.outer_line_width)
        return (int(x-r-lw), int(y-r-lw), int(2*(r+lw)), int(2*(r+lw)))

    def paint_frame(self, cr):
        if self.in_input_stroke:
            return
        if self.current_marker_pos is None:
            return
        x, y = self.tdw.model_to_display(*self.current_marker_pos)
        area = self._calc_area(x, y)
        x = int(x) + 0.5
        y = int(y) + 0.5
        r = self.radius
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        rgba = list(self.outer_line_rgba)
        rgba[3] *= self.alpha
        cr.set_source_rgba(*rgba)
        cr.set_line_width(self.outer_line_width)
        cr.move_to(x-r, y-r)
        cr.line_to(x+r, y+r)
        cr.move_to(x+r, y-r)
        cr.line_to(x-r, y+r)
        cr.stroke_preserve()
        rgba = list(self.inner_line_rgba)
        rgba[3] *= self.alpha
        cr.set_source_rgba(*rgba)
        cr.set_line_width(self.inner_line_width)
        cr.stroke()
        return area


class ColorAdjustOverlay (FadingOverlay):
    """Preview overlay during color Adjustments.

    Important- no border.  Like a paint swatch sample
    For comparing to area on canvas
    Mostly a copy of ColorPickerOverlay
    """
    fade_fps = 30   #: Nominal frames per second
    fade_duration = 5  #: Time for fading entirely to zero, in seconds
    MIN_PREVIEW_SIZE = 140
    # CORNER_RADIUS = 10

    def __init__(self, doc, x, y, r, g, b):
        FadingOverlay.__init__(self, doc)
        doc.input_stroke_started += self.input_stroke_started
        doc.input_stroke_ended += self.input_stroke_ended
        self.in_input_stroke = False
        self.app = gui.application.get_app()
        p = self.app.preferences
        self.preview_size = p['color.preview_size']
        self._doc = doc
        self._tdw = doc.tdw
        self._r = r
        self._g = g
        self._b = b
        self._x = int(x)+0.5
        self._y = int(y)+0.5
        self.corner_radius = None
        alloc = doc.tdw.get_allocation()
        self._tdw_w = alloc.width
        self._tdw_h = alloc.height
        doc.tdw.display_overlays.append(self)
        self._previous_area = None
        self._placed_in_stroke = False
        self._queue_tdw_redraw()

    def input_stroke_started(self, doc, event):
        self.in_input_stroke = True

    def input_stroke_ended(self, doc, event):
        self.in_input_stroke = False
        self._placed_in_stroke = self.in_input_stroke

    def overlay_changed(self):
        return False

    def move(self, x, y):
        """Moves the preview square to a new location, in tdw pointer coords.
        """
        self._x = int(x)+0.5
        self._y = int(y)+0.5
        self._placed_in_stroke = self.in_input_stroke
        self._queue_tdw_redraw()

    def cleanup(self):
        """Cleans up temporary observer stuff, allowing garbage collection.
        """
        self._tdw.display_overlays.remove(self)
        assert self not in self._tdw.display_overlays
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
        size = max(int(self.preview_size * .01 * alloc.height * 2),
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
        x -= offset
        y -= offset
        return (int(x), int(y), size, size)

    def paint_frame(self, cr):
        # Cleanup if starting stroke regardless of prefs
        # Tricky due to prefs for in_stroke placement
        if self.in_input_stroke is True and self._placed_in_stroke is False:
            self.alpha = 0

        area = self._get_area()
        if area is not None:
            x, y, w, h = area
            # keep opacity at 100% for a while
            if self.alpha > 0.5:
                actual_alpha = 1.0
            else:
                actual_alpha = pow(self.alpha, 2)
            cr.set_source_rgba(self._r, self._g, self._b, actual_alpha)
            rounded_box_hole(cr, x, y, w, h, self.corner_radius)
            cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            cr.fill()
        self._previous_area = area
        return area
