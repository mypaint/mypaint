# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Hue, Saturation and Value wheel.
"""

from __future__ import division, print_function

from gettext import gettext as _

from lib.gibindings import Gtk
from lib.gibindings import Gdk

from .adjbases import HueSaturationWheelAdjuster
from .sliders import HSVValueSlider
from lib.color import HSVColor
from .util import clamp
from .combined import CombinedAdjusterPage


class HSVHueSaturationWheel (HueSaturationWheelAdjuster):
    """Hue, Saturation and Value wheel.
    """

    STATIC_TOOLTIP_TEXT = _("HSV Hue and Saturation")

    def __init__(self):
        HueSaturationWheelAdjuster.__init__(self)
        self.connect("scroll-event", self.__scroll_cb)
        self.add_events(Gdk.EventMask.SCROLL_MASK)

    def __scroll_cb(self, widget, event):
        d = self.SCROLL_DELTA
        if event.direction in (
                Gdk.ScrollDirection.DOWN, Gdk.ScrollDirection.LEFT):
            d *= -1
        col = HSVColor(color=self.get_managed_color())
        v = clamp(col.v+d, 0.0, 1.0)
        if col.v != v:
            col.v = v
            self.set_managed_color(col)
        return True

    def get_normalized_polar_pos_for_color(self, col):
        col = HSVColor(color=col)
        return col.s, col.h

    def color_at_normalized_polar_pos(self, r, theta):
        col = HSVColor(color=self.get_managed_color())
        col.h = theta
        col.s = r
        return col


class HSVAdjusterPage (CombinedAdjusterPage):
    """Page details for the HSV wheel.
    """

    def __init__(self):

        self.__v_adj = HSVValueSlider()
        self.__v_adj.vertical = True
        v_align = Gtk.Alignment(xalign=1, yalign=0, xscale=0, yscale=1)
        v_align.add(self.__v_adj)

        self.__hs_adj = HSVHueSaturationWheel()

        table = Gtk.Table(n_rows=1, n_columns=2)
        row = 0
        xopts = Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND
        yopts = Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND
        table.attach(
            v_align, 0, 1, row, row+1,
            Gtk.AttachOptions.FILL, yopts, 3, 3)
        table.attach(self.__hs_adj, 1, 2, row, row+1, xopts, yopts, 3, 3)

        self.__table = table

    @classmethod
    def get_page_icon_name(self):
        return 'mypaint-tool-hsvwheel'

    @classmethod
    def get_page_title(self):
        return _('HSV Wheel')

    @classmethod
    def get_page_description(self):
        return _("Saturation and Value color changer.")

    def get_page_widget(self):
        frame = Gtk.AspectFrame(obey_child=True)
        frame.set_shadow_type(Gtk.ShadowType.NONE)
        frame.add(self.__table)
        return frame

    def set_color_manager(self, manager):
        CombinedAdjusterPage.set_color_manager(self, manager)
        self.__v_adj.set_color_manager(manager)
        self.__hs_adj.set_color_manager(manager)


if __name__ == '__main__':
    import os
    import sys
    from adjbases import ColorManager
    mgr = ColorManager(prefs={}, datapath='.')
    if len(sys.argv) > 1:
        # Generate icons
        mgr.set_color(HSVColor(0.0, 0.0, 0.8))
        wheel = HSVHueSaturationWheel()
        wheel.set_color_manager(mgr)
        icon_name = HSVAdjusterPage.get_page_icon_name()
        for dir_name in sys.argv[1:]:
            wheel.save_icon_tree(dir_name, icon_name)
    else:
        # Interactive test
        mgr.set_color(HSVColor(0.333, 0.6, 0.5))
        page = HSVAdjusterPage()
        page.set_color_manager(mgr)
        window = Gtk.Window()
        window.add(page.get_page_widget())
        window.set_title(os.path.basename(sys.argv[0]))
        window.set_border_width(6)
        window.connect("destroy", lambda *a: Gtk.main_quit())
        window.show_all()
        Gtk.main()
