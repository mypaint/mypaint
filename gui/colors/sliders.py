# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Component sliders for power users.
"""

import gtk
from gtk import gdk
from gettext import gettext as _

from util import *
from uicolor import *
from bases import IconRenderable
from adjbases import ColorAdjusterWidget
from adjbases import ColorAdjuster
from adjbases import SliderColorAdjuster
from combined import CombinedAdjusterPage


class ComponentSlidersAdjusterPage (CombinedAdjusterPage, IconRenderable):
    """Component sliders for precise adjustment: page for `CombinedAdjuster`.
    """

    # Instance data (defaults, docs)
    __sliders = None  #: List of slider widgets
    __table = None  #: Page table.


    def __init__(self):
        table = gtk.Table(rows=6, columns=2)
        table.set_size_request(100, -1)
        self.__sliders = []
        xpad = 3
        ypad = 3
        table_layout = [
            [(RGBRedSlider,        1, 2,   'R', 0, 1),
             (RGBGreenSlider,      1, 2,   'G', 0, 1),
             (RGBBlueSlider,       1, 2,   'B', 0, 1)],
            #[(HSVHueSlider,        1, 2,   'H', 0, 1),
            # (HSVSaturationSlider, 1, 2,   'S', 0, 1),
            # (HSVValueSlider,      1, 2,   'V', 0, 1)],
            [(HCYHueSlider,        1, 2,   'H', 0, 1),
             (HCYChromaSlider,     1, 2,   'C', 0, 1),
             (HCYLumaSlider,       1, 2,   'Y', 0, 1)],
            ]
        row = 0
        for adj_triple in table_layout:
            component_num = 1
            for ( slider_class, slider_l, slider_r,
                  label_text, label_l, label_r  ) in adj_triple:
                yopts = gtk.FILL
                slider = slider_class()
                self.__sliders.append(slider)
                label = gtk.Label()
                label.set_text(label_text)
                label.set_alignment(1.0, 0.5)
                if component_num in (1, 3) and row != 0:
                    yalign = (component_num == 1) and 1 or 0
                    align = gtk.Alignment(xalign=0, yalign=yalign,
                                          xscale=1, yscale=0)
                    align.add(label)
                    label = align
                    align = gtk.Alignment(xalign=0, yalign=yalign,
                                          xscale=1, yscale=0)
                    align.add(slider)
                    slider = align
                    yopts |= gtk.EXPAND
                table.attach(label, label_l, label_r, row, row+1,
                             gtk.SHRINK|gtk.FILL, yopts, xpad, ypad)
                table.attach(slider, slider_l, slider_r, row, row+1,
                             gtk.EXPAND|gtk.SHRINK|gtk.FILL, yopts, xpad, ypad)
                row += 1
                component_num += 1
        self.__table = table

    @classmethod
    def get_page_icon_name(self):
        return 'mypaint-tool-component-sliders'

    @classmethod
    def get_page_title(self):
        return _('Component sliders')

    @classmethod
    def get_page_description(self):
        return _('Adjust individual components of the colour.')

    def get_page_widget(self):
        return self.__table

    def set_color_manager(self, manager):
        ColorAdjuster.set_color_manager(self, manager)
        for slider in self.__sliders:
            slider.set_color_manager(manager)


    def render_as_icon(self, cr, size):
        """Renders as an icon into a Cairo context.
        """
        # Strategy: construct tmp R,G,B sliders with a colour that shows off
        # their primary a bit. Render carefully (might need special handling for
        # the 16px size).
        from adjbases import ColorManager
        mgr = ColorManager()
        mgr.set_color(RGBColor(0.3, 0.3, 0.4))
        adjs = [RGBRedSlider(), RGBGreenSlider(), RGBBlueSlider()]
        for adj in adjs:
            adj.set_color_manager(mgr)
        if size <= 16:
            cr.save()
            for adj in adjs:
                adj.border = 1
                adj.render_background_cb(cr, wd=16, ht=5)
                cr.translate(0, 5)
            cr.restore()
        else:
            cr.save()
            bar_ht = int(size/3)
            offset = int((size - bar_ht*3) / 2)
            cr.translate(0, offset)
            for adj in adjs:
                adj.border = max(2, int(size/16))
                adj.render_background_cb(cr, wd=size, ht=bar_ht)
                cr.translate(0, bar_ht)
            cr.restore()
        for adj in adjs:
            adj.set_color_manager(None)



class RGBRedSlider (SliderColorAdjuster):
    tooltip_text = _("RGB Red")
    def get_background_validity(self):
        col = self.get_managed_color()
        r,g,b = col.get_rgb()
        return g, b
    def get_color_for_bar_amount(self, amt):
        col = RGBColor(color=self.get_managed_color())
        col.r = amt
        return col
    def get_bar_amount_for_color(self, col):
        return col.r

class RGBGreenSlider (SliderColorAdjuster):
    tooltip_text = _("RGB Green")
    def get_background_validity(self):
        col = self.get_managed_color()
        r,g,b = col.get_rgb()
        return r, b
    def get_color_for_bar_amount(self, amt):
        col = RGBColor(color=self.get_managed_color())
        col.g = amt
        return col
    def get_bar_amount_for_color(self, col):
        return col.g

class RGBBlueSlider (SliderColorAdjuster):
    tooltip_text = _("RGB Blue")
    def get_background_validity(self):
        col = self.get_managed_color()
        r,g,b = col.get_rgb()
        return r, g
    def get_color_for_bar_amount(self, amt):
        col = RGBColor(color=self.get_managed_color())
        col.b = amt
        return col
    def get_bar_amount_for_color(self, col):
        return col.b


class HSVHueSlider (SliderColorAdjuster):
    tooltip_text = _("HSV Hue")
    samples = 4
    def get_color_for_bar_amount(self, amt):
        col = HSVColor(color=self.get_managed_color())
        col.h = amt
        return col
    def get_bar_amount_for_color(self, col):
        return col.h

class HSVSaturationSlider (SliderColorAdjuster):
    tooltip_text = _("HSV Saturation")
    def get_color_for_bar_amount(self, amt):
        col = HSVColor(color=self.get_managed_color())
        col.s = amt
        return col
    def get_bar_amount_for_color(self, col):
        return col.s

class HSVValueSlider (SliderColorAdjuster):
    tooltip_text = _("HSV Value")
    def get_color_for_bar_amount(self, amt):
        col = HSVColor(color=self.get_managed_color())
        col.v = amt
        return col
    def get_bar_amount_for_color(self, col):
        return col.v


class HCYHueSlider (SliderColorAdjuster):
    tooltip_text = _("HCY Hue")
    samples = 4
    def get_color_for_bar_amount(self, amt):
        col = HCYColor(color=self.get_managed_color())
        col.h = amt
        return col
    def get_bar_amount_for_color(self, col):
        col = HCYColor(color=col)
        return col.h

class HCYChromaSlider (SliderColorAdjuster):
    tooltip_text = _("HCY Chroma")

    def get_color_for_bar_amount(self, amt):
        col = HCYColor(color=self.get_managed_color())
        col.c = amt
        return col

    def get_bar_amount_for_color(self, col):
        col = HCYColor(color=col)
        return col.c

class HCYLumaSlider (SliderColorAdjuster):
    tooltip_text = _("HCY Luma (Y')")

    @property
    def samples(self):
        alloc = self.get_allocation()
        len = self.vertical and alloc.height or alloc.width
        len -= self.border * 2
        return min(int(len / 3), 64)

    def get_color_for_bar_amount(self, amt):
        col = HCYColor(color=self.get_managed_color())
        col.y = amt
        return col

    def get_bar_amount_for_color(self, col):
        col = HCYColor(color=col)
        return col.y

    def get_background_validity(self):
        col = HCYColor(color=self.get_managed_color())
        return int(col.h * 1000), int(col.c * 1000)


if __name__ == '__main__':
    import os, sys
    from adjbases import ColorManager
    mgr = ColorManager()
    cs_adj = ComponentSlidersAdjusterPage()
    cs_adj.set_color_manager(mgr)
    cs_adj.set_managed_color(RGBColor(0.3, 0.6, 0.7))
    if len(sys.argv) > 1:
        icon_name = cs_adj.get_page_icon_name()
        for dir_name in sys.argv[1:]:
            cs_adj.save_icon_tree(dir_name, icon_name)
    else:
        # Interactive test
        window = gtk.Window()
        window.add(cs_adj.get_page_widget())
        window.set_title(os.path.basename(sys.argv[0]))
        window.connect("destroy", lambda *a: gtk.main_quit())
        window.show_all()
        gtk.main()

