# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team.
# Copyright (C) 2015 by ShadowKyogre <shadowkyogre@aim.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Axis-aligned planar slice of an HSV color cube, and a depth slider.
"""

from __future__ import division, print_function
import math
from gettext import gettext as _

from lib.gibindings import Gtk
import cairo

from .util import clamp
from .util import draw_marker_circle
from lib.color import RGBColor, HSVColor
from .bases import IconRenderable
from .adjbases import ColorAdjusterWidget
from .adjbases import ColorAdjuster
from .adjbases import IconRenderableColorAdjusterWidget
from .adjbases import HueSaturationWheelAdjuster
from .combined import CombinedAdjusterPage

from lib.pycompat import xrange


class HSVSquarePage (CombinedAdjusterPage, IconRenderable):
    """Hue ring and Sat+Val square: page for `CombinedAdjuster`."""

    def __init__(self):
        self._faces = ['h', 's', 'v']
        table = Gtk.Table(n_rows=1, n_columns=1)

        xopts = Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND
        yopts = Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND

        self.__adj = HSVSquare()

        table.attach(self.__adj, 0, 1, 0, 1, xopts, yopts, 3, 3)
        self.__table = table
        self.__adj._update_tooltips()

    @classmethod
    def get_page_icon_name(self):
        return 'mypaint-tool-hsvsquare'

    @classmethod
    def get_page_title(self):
        return _('HSV Square')

    @classmethod
    def get_page_description(self):
        return _("An HSV Square which can be rotated to show different hues.")

    def get_page_widget(self):
        return self.__table

    def set_color_manager(self, manager):
        ColorAdjuster.set_color_manager(self, manager)
        self.__adj.set_color_manager(manager)

    def render_as_icon(self, cr, size):
        """Renders as an icon into a Cairo context.
        """
        # Strategy: construct tmp R,G,B sliders with a color that shows off
        # their primary a bit. Render carefully (might need special handling
        # for the 16px size).
        from adjbases import ColorManager
        mgr = ColorManager(prefs={}, datapath=".")
        mgr.set_color(RGBColor(0.3, 0.3, 0.4))
        ring_adj = _HSVSquareOuterRing(self)
        ring_adj.set_color_manager(mgr)
        square_adj = _HSVSquareInnerSquare(self)
        square_adj.set_color_manager(mgr)
        if size <= 16:
            cr.save()
            ring_adj.render_background_cb(cr, wd=16, ht=16)
            cr.translate(-6, -6)
            square_adj.render_background_cb(cr, wd=12, ht=12)
            cr.restore()
        else:
            cr.save()
            square_offset = int(size/5.0 * 1.6)
            square_dim = int(size * 0.64)
            ring_adj.render_background_cb(cr, wd=size, ht=size)
            # do minor rounding adjustments for hsvsquare icons at this size
            if size == 24:
                cr.translate(-1, -1)
                square_dim += 1
            cr.translate(-square_offset, -square_offset)
            square_adj.render_background_cb(cr, wd=square_dim, ht=square_dim)
            cr.restore()
        ring_adj.set_color_manager(None)
        square_adj.set_color_manager(None)


class HSVSquare(Gtk.VBox, ColorAdjuster):
    """Combined Sat+Val square and Hue ring color adjuster"""

    __gtype_name__ = 'HSVSquare'

    def __init__(self):
        super(HSVSquare, self).__init__()
        self._faces = ['h', 's', 'v']

        self.__square = _HSVSquareInnerSquare(self)
        self.__ring = _HSVSquareOuterRing(self)

        s_align = Gtk.Alignment(
            xalign=0.5, yalign=0.5,
            xscale=0.54, yscale=0.54,
        )
        plz_be_square = Gtk.AspectFrame()
        plz_be_square.set_shadow_type(Gtk.ShadowType.NONE)
        s_align.add(plz_be_square)
        plz_be_square.add(self.__square)
        self.__ring.add(s_align)
        self.pack_start(self.__ring, True, True, 0)

    def set_color_manager(self, manager):
        super(HSVSquare, self).set_color_manager(manager)
        self.__square.set_color_manager(manager)
        self.__ring.set_color_manager(manager)

    def _update_tooltips(self):
        self.__ring.set_tooltip_text(_("HSV Hue"))
        self.__square.set_tooltip_text(_("HSV Saturation and Value"))


class _HSVSquareOuterRing (HueSaturationWheelAdjuster):
    """Outer color ring"""

    vertical = True
    samples = 4

    def __init__(self, cube):
        HueSaturationWheelAdjuster.__init__(self)
        self.__cube = cube

    def get_pos_for_color(self, col):
        nr, ntheta = self.get_normalized_polar_pos_for_color(col)
        mgr = self.get_color_manager()
        if mgr:
            ntheta = mgr.distort_hue(ntheta)
        nr **= 1.0/self.SAT_GAMMA
        alloc = self.get_allocation()
        wd, ht = alloc.width, alloc.height
        radius = self.get_radius(wd, ht, self.BORDER_WIDTH)
        cx, cy = self.get_center(wd, ht)
        r = radius * clamp(nr, 0, 1)
        t = clamp(ntheta, 0, 1) * 2 * math.pi
        x = int(cx + r*math.cos(t)) + 0.5
        y = int(cy + r*math.sin(t)) + 0.5
        return x, y

    def get_color_at_position(self, x, y):
        """Gets the color at a position, for `ColorAdjusterWidget` impls.
        """
        alloc = self.get_allocation()
        cx, cy = self.get_center(alloc=alloc)
        # Normalized radius
        r = math.sqrt((x-cx)**2 + (y-cy)**2)
        radius = self.get_radius(alloc=alloc)
        if r > radius:
            r = radius
        r /= radius
        r **= self.SAT_GAMMA
        # Normalized polar angle
        theta = 1.25 - (math.atan2(x-cx, y-cy) / (2*math.pi))
        while theta <= 0:
            theta += 1.0
        theta %= 1.0
        mgr = self.get_color_manager()
        if mgr:
            theta = mgr.undistort_hue(theta)
        return self.color_at_normalized_polar_pos(r, theta)

    def get_normalized_polar_pos_for_color(self, col):
        col = HSVColor(color=col)
        return col.s, col.h

    def color_at_normalized_polar_pos(self, r, theta):
        col = HSVColor(color=self.get_managed_color())
        col.h = theta
        return col

    def get_background_validity(self):
        col = HSVColor(color=self.get_managed_color())
        f0, f1, f2 = self.__cube._faces
        return f0, getattr(col, f0)

    def render_background_cb(self, cr, wd, ht, icon_border=None):
        """Renders the offscreen bg, for `ColorAdjusterWidget` impls.
        """
        cr.save()

        border = icon_border
        if border is None:
            border = self.BORDER_WIDTH
        radius = self.get_radius(wd, ht, border)

        steps = self.HUE_SLICES

        # Move to the centre
        cx, cy = self.get_center(wd, ht)
        cr.translate(cx, cy)

        # Clip, for a slight speedup
        cr.arc(0, 0, radius+border, 0, 2*math.pi)
        cr.clip()

        # Tangoesque outer border
        cr.set_line_width(self.OUTLINE_WIDTH)
        cr.arc(0, 0, radius, 0, 2*math.pi)
        cr.set_source_rgba(*self.OUTLINE_RGBA)
        cr.stroke()

        # Each slice in turn
        cr.save()
        cr.set_line_width(1.0)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        step_angle = 2.0*math.pi/steps
        mgr = self.get_color_manager()

        for ih in xrange(steps+1):  # overshoot by 1, no solid bit for final
            h = ih / steps
            if mgr:
                h = mgr.undistort_hue(h)
            edge_col = self.color_at_normalized_polar_pos(1.0, h)
            edge_col.s = 1.0
            edge_col.v = 1.0
            rgb = edge_col.get_rgb()

            if ih > 0:
                # Backwards gradient
                cr.arc_negative(0, 0, radius, 0, -step_angle)
                x, y = cr.get_current_point()
                cr.line_to(0, 0)
                cr.close_path()
                lg = cairo.LinearGradient(radius, 0, (x + radius) / 2, y)
                lg.add_color_stop_rgba(0, rgb[0], rgb[1], rgb[2], 1.0)
                lg.add_color_stop_rgba(1, rgb[0], rgb[1], rgb[2], 0.0)
                cr.set_source(lg)
                cr.fill()

            if ih < steps:
                # Forward solid
                cr.arc(0, 0, radius, 0, step_angle)
                x, y = cr.get_current_point()
                cr.line_to(0, 0)
                cr.close_path()
                cr.set_source_rgb(*rgb)
                cr.stroke_preserve()
                cr.fill()
            cr.rotate(step_angle)

        cr.restore()

        # Tangoesque inner border
        cr.set_source_rgba(*self.EDGE_HIGHLIGHT_RGBA)
        cr.set_line_width(self.EDGE_HIGHLIGHT_WIDTH)
        cr.arc(0, 0, radius, 0, 2*math.pi)
        cr.stroke()

        cr.set_line_width(self.OUTLINE_WIDTH)
        cr.arc(0, 0, radius*0.8, 0, 2*math.pi)
        cr.set_source_rgba(0, 0, 0, 1)
        cr.set_operator(cairo.OPERATOR_DEST_OUT)
        cr.fill()
        cr.set_operator(cairo.OPERATOR_OVER)
        cr.set_source_rgba(*self.OUTLINE_RGBA)
        cr.stroke()

        cr.set_source_rgba(*self.EDGE_HIGHLIGHT_RGBA)
        cr.set_line_width(self.EDGE_HIGHLIGHT_WIDTH)
        cr.arc(0, 0, radius*0.8, 0, 2*math.pi)
        cr.stroke()

    def paint_foreground_cb(self, cr, wd, ht):
        """Fg marker painting, for `ColorAdjusterWidget` impls.
        """
        col = HSVColor(color=self.get_managed_color())
        col.s = 1.0
        radius = self.get_radius(wd, ht, self.BORDER_WIDTH)
        cx = int(wd // 2)
        cy = int(ht // 2)
        cr.arc(cx, cy, radius+0.5, 0, 2*math.pi)
        cr.clip()
        x, y = self.get_pos_for_color(col)
        col.s = 0.70
        ex, ey = self.get_pos_for_color(col)

        cr.set_line_width(5)
        cr.move_to(x, y)
        cr.line_to(ex, ey)
        cr.set_source_rgb(0, 0, 0)
        cr.stroke_preserve()

        cr.set_source_rgb(1, 1, 1)
        cr.set_line_width(3.5)
        cr.stroke_preserve()

        cr.set_source_rgb(*col.get_rgb())
        cr.set_line_width(0.25)
        cr.stroke()


class _HSVSquareInnerSquare (IconRenderableColorAdjusterWidget):
    """Inner saturation & value square"""

    def __init__(self, cube):
        ColorAdjusterWidget.__init__(self)
        self.__cube = cube
        self.connect('button-press-event', self.stop_fallthrough)

    def stop_fallthrough(self, widget, event):
        return True

    def __get_faces(self):
        f1 = self.__cube._faces[1]
        f2 = self.__cube._faces[2]
        if f2 == 'h':
            f1, f2 = f2, f1
        return f1, f2

    def render_background_cb(self, cr, wd, ht, icon_border=None):
        col = HSVColor(color=self.get_managed_color())
        b = icon_border
        if b is None:
            b = self.BORDER_WIDTH
        eff_wd = int(wd - 2*b)
        eff_ht = int(ht - 2*b)
        f1, f2 = self.__get_faces()

        step = max(1, int(eff_wd // 128))

        rect_x, rect_y = int(b)+0.5, int(b)+0.5
        rect_w, rect_h = int(eff_wd)-1, int(eff_ht)-1

        # Paint the central area offscreen
        cr.push_group()
        for x in xrange(0, eff_wd, step):
            amt = x / eff_wd
            setattr(col, f1, amt)
            setattr(col, f2, 1.0)
            lg = cairo.LinearGradient(b+x, b, b+x, b+eff_ht)
            lg.add_color_stop_rgb(*([0.0] + list(col.get_rgb())))
            setattr(col, f2, 0.0)
            lg.add_color_stop_rgb(*([1.0] + list(col.get_rgb())))
            cr.rectangle(b+x, b, step, eff_ht)
            cr.set_source(lg)
            cr.fill()
        slice_patt = cr.pop_group()

        # Tango-like outline
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.rectangle(rect_x, rect_y, rect_w, rect_h)
        cr.set_line_width(self.OUTLINE_WIDTH)
        cr.set_source_rgba(*self.OUTLINE_RGBA)
        cr.stroke()

        # The main area
        cr.set_source(slice_patt)
        cr.paint()

        # Tango-like highlight over the top
        cr.rectangle(rect_x, rect_y, rect_w, rect_h)
        cr.set_line_width(self.EDGE_HIGHLIGHT_WIDTH)
        cr.set_source_rgba(*self.EDGE_HIGHLIGHT_RGBA)
        cr.stroke()

    def get_background_validity(self):
        col = HSVColor(color=self.get_managed_color())
        f0 = self.__cube._faces[0]
        return f0, getattr(col, f0)

    def get_color_at_position(self, x, y):
        alloc = self.get_allocation()
        b = self.BORDER_WIDTH
        wd = alloc.width
        ht = alloc.height
        eff_wd = wd - 2*b
        eff_ht = ht - 2*b
        f1_amt = clamp((x-b) / eff_wd, 0, 1)
        f2_amt = clamp((y-b) / eff_ht, 0, 1)
        col = HSVColor(color=self.get_managed_color())
        f1, f2 = self.__get_faces()
        f2_amt = 1.0 - f2_amt
        setattr(col, f1, f1_amt)
        setattr(col, f2, f2_amt)
        return col

    def get_position_for_color(self, col):
        col = HSVColor(color=col)
        f1, f2 = self.__get_faces()
        f1_amt = getattr(col, f1)
        f2_amt = getattr(col, f2)
        f2_amt = 1.0 - f2_amt
        alloc = self.get_allocation()
        b = self.BORDER_WIDTH
        wd = alloc.width
        ht = alloc.height
        eff_wd = wd - 2*b
        eff_ht = ht - 2*b
        x = b + f1_amt*eff_wd
        y = b + f2_amt*eff_ht
        return x, y

    def paint_foreground_cb(self, cr, wd, ht):
        x, y = self.get_position_for_color(self.get_managed_color())
        draw_marker_circle(cr, x, y)


if __name__ == '__main__':
    import os
    import sys
    from adjbases import ColorManager
    mgr = ColorManager(prefs={}, datapath='.')
    cube = HSVSquarePage()
    cube.set_color_manager(mgr)
    mgr.set_color(RGBColor(0.3, 0.6, 0.7))
    if len(sys.argv) > 1:
        icon_name = cube.get_page_icon_name()
        for dir_name in sys.argv[1:]:
            cube.save_icon_tree(dir_name, icon_name)
    else:
        # Interactive test
        window = Gtk.Window()
        window.add(cube.get_page_widget())
        window.set_title(os.path.basename(sys.argv[0]))
        window.connect("destroy", lambda *a: Gtk.main_quit())
        window.show_all()
        Gtk.main()
