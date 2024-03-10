# This file is part of MyPaint.
# Copyright (C) 2012-2013 by Andrew Chadwick <a.t.chadwickgmail.com>
# Copyright (C) 2014-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Axis-aligned planar slice of an HSV color cube, and a depth slider.
"""

from __future__ import division, print_function

from gettext import gettext as _

import cairo
from lib.gibindings import Gtk

from .util import clamp
from .util import draw_marker_circle
from lib.color import HSVColor
from lib.color import RGBColor
from .adjbases import ColorAdjusterWidget
from .adjbases import ColorAdjuster
from .adjbases import SliderColorAdjuster
from .adjbases import IconRenderableColorAdjusterWidget
from .combined import CombinedAdjusterPage
from .uimisc import borderless_button
from .uimisc import PRIMARY_ADJUSTERS_MIN_WIDTH
from .uimisc import PRIMARY_ADJUSTERS_MIN_HEIGHT

from lib.pycompat import xrange


class HSVCubePage (CombinedAdjusterPage):
    """Slice+depth view through an HSV cube: page for `CombinedAdjuster`.

    The page includes a button for tumbling the cube, i.e. changing which of
    the color components the slice and the depth slider refer to.

    """

    # Tooltip mappings, indexed by whatever the slider currently represents
    _slider_tooltip_map = dict(h=_("HSV Hue"),
                               s=_("HSV Saturation"),
                               v=_("HSV Value"))
    _slice_tooltip_map = dict(h=_("HSV Saturation and Value"),
                              s=_("HSV Hue and Value"),
                              v=_("HSV Hue and Saturation"))

    def __init__(self):
        self._faces = ['h', 's', 'v']
        button = borderless_button(
            icon_name='mypaint-hsv-rotate-symbolic',
            size=Gtk.IconSize.MENU,
            tooltip=_("Rotate cube (show different axes)")
        )
        button.connect("clicked", lambda *a: self.tumble())
        self.__slice = HSVCubeSlice(self)
        self.__slider = HSVCubeSlider(self)
        s_align = Gtk.Alignment(xalign=0.5, yalign=0, xscale=0, yscale=1)
        s_align.add(self.__slider)

        table = Gtk.Table(n_rows=2, n_columns=2)

        xopts = Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND
        yopts = Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND

        table.attach(s_align, 0, 1, 0, 1, Gtk.AttachOptions.FILL, yopts, 3, 3)
        table.attach(
            button, 0, 1, 1, 2,
            Gtk.AttachOptions.FILL, Gtk.AttachOptions.FILL, 3, 3)
        table.attach(self.__slice, 1, 2, 0, 2, xopts, yopts, 3, 3)
        self.__table = table
        self._update_tooltips()

    @classmethod
    def get_page_icon_name(self):
        return 'mypaint-tool-hsvcube'

    @classmethod
    def get_page_title(self):
        return _('HSV Cube')

    @classmethod
    def get_page_description(self):
        return _("An HSV cube which can be rotated to show different "
                 "planar slices.")

    def get_page_widget(self):
        return self.__table

    def tumble(self):
        f0 = self._faces.pop(0)
        self._faces.append(f0)
        self.__slider.queue_draw()
        self.__slice.queue_draw()
        self._update_tooltips()

    def _update_tooltips(self):
        f0 = self._faces[0]
        self.__slice.set_tooltip_text(self._slice_tooltip_map[f0])
        self.__slider.set_tooltip_text(self._slider_tooltip_map[f0])

    def set_color_manager(self, manager):
        ColorAdjuster.set_color_manager(self, manager)
        self.__slider.set_color_manager(manager)
        self.__slice.set_color_manager(manager)


class HSVCubeSlider (SliderColorAdjuster):
    """Depth of the planar slice of a cube.
    """

    vertical = True
    samples = 4

    def __init__(self, cube):
        SliderColorAdjuster.__init__(self)
        self.__cube = cube

    def get_background_validity(self):
        col = HSVColor(color=self.get_managed_color())
        f0, f1, f2 = self.__cube._faces
        return f0, getattr(col, f1), getattr(col, f2)

    def get_color_for_bar_amount(self, amt):
        col = HSVColor(color=self.get_managed_color())
        f0 = self.__cube._faces[0]
        setattr(col, f0, amt)
        return col

    def get_bar_amount_for_color(self, col):
        f0 = self.__cube._faces[0]
        amt = getattr(col, f0)
        return amt


class HSVCubeSlice (IconRenderableColorAdjusterWidget):
    """Planar slice through an HSV cube.
    """

    def __init__(self, cube):
        ColorAdjusterWidget.__init__(self)
        w = PRIMARY_ADJUSTERS_MIN_WIDTH
        h = PRIMARY_ADJUSTERS_MIN_HEIGHT
        self.set_size_request(w, h)
        self.__cube = cube

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
    cube = HSVCubePage()
    cube.set_color_manager(mgr)
    mgr.set_color(RGBColor(0.3, 0.6, 0.7))
    if len(sys.argv) > 1:
        slice = HSVCubeSlice(cube)
        slice.set_color_manager(mgr)
        icon_name = cube.get_page_icon_name()
        for dir_name in sys.argv[1:]:
            slice.save_icon_tree(dir_name, icon_name)
    else:
        # Interactive test
        window = Gtk.Window()
        window.add(cube.get_page_widget())
        window.set_title(os.path.basename(sys.argv[0]))
        window.connect("destroy", lambda *a: Gtk.main_quit())
        window.show_all()
        Gtk.main()
