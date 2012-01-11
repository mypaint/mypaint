# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Overlays for TDWs showing information about the TDW state.
"""

import gtk
from gtk import gdk
import gobject
import cairo
import pango
from math import floor, ceil, pi
from tileddrawwidget import Overlay
from lib.helpers import clamp

from gettext import gettext as _


class FadingOverlay (Overlay):
    """A temporary overlay which fades to alpha over about a second.
    """

    fade_step_delay = 50
    fade_step_amount = 0.075

    def __init__(self, tdw):
        self.tdw = tdw
        self.anim_srcid = None
        self.area = None
        self.alpha = 1.0

    def paint(self, cr):
        """Repaint the overlay and start animating if necessary.
        """
        if self.overlay_changed():
            self.alpha = 1.0
        elif self.alpha <= 0:
            # No need to draw anything.
            return
        self.restart_anim_if_needed()
        self.area = self.paint_frame(cr)

    def anim_cb(self):
        """Animation callback.

        Each step fades the alpha multiplier slightly and invalidates the area
        last painted.
        """
        self.alpha -= self.fade_step_amount
        win = self.tdw.get_window()
        if win is not None:
            if self.area:
                win.invalidate_rect(gdk.Rectangle(*self.area), True)
        if self.alpha <= 0.0:
            self.anim_srcid = None
            return False
        else:
            return True

    def restart_anim_if_needed(self):
        if self.anim_srcid is None:
            delay = self.fade_step_delay
            self.anim_srcid = gobject.timeout_add(delay, self.anim_cb)

    def paint_frame(self, cr):
        raise NotImplementedError

    def overlay_changed(self):
        raise NotImplementedError


class ScaleOverlay (FadingOverlay):
    """Overlays its TDW's current zoom, fading to transparent.

    The animation is started by the normal full canvas repaint which happens
    after the scale changes.
    """

    background_rgba = [0, 0, 0, 0.666]
    text_rgba = [1, 1, 1, 1.0]
    vmargin = 5
    hmargin = 10
    padding = 5
    shown_scale = None


    def overlay_changed(self):
        return self.tdw.scale != self.shown_scale


    def paint_frame(self, cr):
        self.shown_scale = self.tdw.scale
        text = _("%.01f%%") % (100*self.shown_scale)
        layout = self.tdw.create_pango_layout(text)
        attrs = pango.AttrList()
        attrs.insert(pango.AttrWeight(pango.WEIGHT_BOLD, 0, -1))
        layout.set_attributes(attrs)
        alloc = self.tdw.get_allocation()
        lw, lh = layout.get_pixel_size()
        alpha = clamp(self.alpha, 0.0, 1.0)

        # Background rectangle
        hm = self.hmargin
        vm = self.hmargin
        w = alloc.width
        h = alloc.height
        p = self.padding
        area = w-lw-hm-p-p, vm, lw+p+p, lh+p+p
        cr.rectangle(*area)
        rgba = self.background_rgba[:]
        rgba[3] *= alpha
        cr.set_source_rgba(*rgba)
        cr.fill()

        # Text
        cr.translate(w-lw-hm-p, vm+p)
        rgba = self.text_rgba[:]
        rgba[3] *= alpha
        cr.set_source_rgba(*rgba)
        cr.show_layout(layout)

        # Where to invalidate
        return area



class LastPaintPosOverlay (FadingOverlay):
    """Displays the last painting position after a stroke has finished.

    Not especially useful, but serves as an example of how to drive an overlay
    from user input events.
    """

    inner_line_rgba = [1, 1, 1, 1]
    inner_line_width = 6
    outer_line_rgba = [0, 0, 0, 0.666]
    outer_line_width = 8
    radius = 4.0


    def __init__(self, doc):
        FadingOverlay.__init__(self, doc.tdw)
        doc.input_stroke_started_observers.append(self.input_stroke_started)
        doc.input_stroke_ended_observers.append(self.input_stroke_ended)
        self.current_marker_pos = None
        self.in_input_stroke = False

    def input_stroke_started(self, event):
        self.in_input_stroke = True
        if self.current_marker_pos is None:
            return
        # Clear the current marker
        model_x, model_y = self.current_marker_pos
        model_cr = self.tdw.get_model_coordinates_cairo_context()
        x, y = model_cr.user_to_device(model_x, model_y)
        area = self._calc_area(x, y)
        self.tdw.queue_draw_area(*area)
        self.current_marker_pos = None
        self.alpha = 0.0

    def input_stroke_ended(self, event):
        self.in_input_stroke = False
        if self.tdw.last_painting_pos is None:
            return
        # Record the new marker position
        model_x, model_y = self.tdw.last_painting_pos
        model_cr = self.tdw.get_model_coordinates_cairo_context()
        x, y = model_cr.user_to_device(model_x, model_y)
        area = self._calc_area(x, y)
        self.tdw.queue_draw_area(*area)
        self.current_marker_pos = model_x, model_y
        self.alpha = 1.0
        self.restart_anim_if_needed()

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
        model_cr = self.tdw.get_model_coordinates_cairo_context()
        x, y = model_cr.user_to_device(*self.current_marker_pos)
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


