# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Hue, Saturation and Value wheel.
"""

import math

import gtk
from gtk import gdk
import cairo
from gettext import gettext as _

from adjbases import ColorAdjusterWidget
from adjbases import HueSaturationWheelAdjuster
from sliders import HSVValueSlider
from uicolor import HSVColor
from util import clamp
from combined import CombinedAdjusterPage


class HSVHueSaturationWheel (HueSaturationWheelAdjuster):
    """Hue, Saturation and Value wheel.
    """

    tooltip_text = _("HSV Hue and Saturation")


    def __init__(self):
        HueSaturationWheelAdjuster.__init__(self)
        self.connect("scroll-event", self.__scroll_cb)
        self.add_events(gdk.SCROLL_MASK)


    def __scroll_cb(self, widget, event):
        d = self.scroll_delta
        if event.direction in (gdk.SCROLL_DOWN, gdk.SCROLL_LEFT):
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
        table = gtk.Table(rows=1, columns=2)

        self.__v_adj = HSVValueSlider()
        self.__v_adj.vertical = True
        self.__hs_adj = HSVHueSaturationWheel()

        row = 0
        xopts = gtk.FILL|gtk.EXPAND
        yopts = gtk.FILL|gtk.EXPAND
        v_align = gtk.Alignment(xalign=1, yalign=0, xscale=0, yscale=1)
        v_align.add(self.__v_adj)
        table.attach(v_align, 0,1,  row,row+1,  gtk.FILL, yopts,  3, 3)
        table.attach(self.__hs_adj, 1,2,  row,row+1,  xopts, yopts,  3, 3)

        self.__table = table

    @classmethod
    def get_page_icon_name(self):
        return 'mypaint-tool-hsvwheel'

    @classmethod
    def get_page_title(self):
        return _('HSV wheel')

    @classmethod
    def get_page_description(self):
        return _("Saturation and Value colour changer.")

    def get_page_widget(self):
        frame = gtk.AspectFrame(obey_child=True)
        frame.set_shadow_type(gtk.SHADOW_NONE)
        frame.add(self.__table)
        return frame

    def set_color_manager(self, manager):
        CombinedAdjusterPage.set_color_manager(self, manager)
        self.__v_adj.set_color_manager(manager)
        self.__hs_adj.set_color_manager(manager)


if __name__ == '__main__':
    import os, sys
    from adjbases import ColorManager
    mgr = ColorManager()
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
        window = gtk.Window()
        window.add(page.get_page_widget())
        window.set_title(os.path.basename(sys.argv[0]))
        window.set_border_width(6)
        window.connect("destroy", lambda *a: gtk.main_quit())
        window.show_all()
        gtk.main()

