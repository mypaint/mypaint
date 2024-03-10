# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Component sliders for power users.
"""

from __future__ import division, print_function

from lib.gibindings import Gtk

from lib.color import RGBColor
from lib.color import HSVColor
from lib.color import HCYColor
from .bases import IconRenderable
from .adjbases import ColorAdjuster
from .adjbases import SliderColorAdjuster
from .combined import CombinedAdjusterPage
from lib.gettext import C_


class ComponentSlidersAdjusterPage (CombinedAdjusterPage, IconRenderable):
    """Component sliders for precise adjustment: page for `CombinedAdjuster`.
    """

    def __init__(self):
        CombinedAdjusterPage.__init__(self)
        grid = Gtk.Grid()
        grid.set_size_request(150, -1)
        grid.set_row_spacing(6)
        grid.set_column_spacing(0)
        grid.set_border_width(6)
        self._sliders = []   #: List of slider widgets.
        grid.set_valign(0.5)
        grid.set_halign(0.5)
        grid.set_hexpand(True)
        grid.set_vexpand(False)
        row_defs = [
            (
                C_("color sliders panel: red/green/blue: slider label", "R"),
                RGBRedSlider,
                0,
            ), (
                C_("color sliders panel: red/green/blue: slider label", "G"),
                RGBGreenSlider,
                0,
            ), (
                C_("color sliders panel: red/green/blue: slider label", "B"),
                RGBBlueSlider,
                0,
            ), (
                C_("color sliders panel: hue/chroma/luma: slider label", "H"),
                HCYHueSlider,
                12,
            ), (
                C_("color sliders panel: hue/chroma/luma: slider label", "C"),
                HCYChromaSlider,
                0,
            ), (
                C_("color sliders panel: hue/chroma/luma: slider label", "Y"),
                HCYLumaSlider,
                0,
            ),
        ]
        row = 0
        for row_def in row_defs:
            label_text, adj_class, margin_top = row_def
            label = Gtk.Label()
            label.set_text(label_text)
            label.set_tooltip_text(adj_class.STATIC_TOOLTIP_TEXT)
            label.set_vexpand(True)
            label.set_hexpand(False)
            label.set_valign(0.0)
            label.set_margin_top(margin_top)
            label.set_margin_start(3)
            label.set_margin_end(3)
            adj = adj_class()
            adj.set_size_request(100, 22)
            adj.set_vexpand(False)
            adj.set_hexpand(True)
            adj.set_margin_top(margin_top)
            adj.set_margin_left(3)
            adj.set_margin_right(3)
            adj.set_valign(0.0)
            self._sliders.append(adj)
            grid.attach(label, 0, row, 1, 1)
            grid.attach(adj, 1, row, 1, 1)
            row += 1
        align = Gtk.Alignment(
            xalign=0.5, yalign=0.5,
            xscale=1.0, yscale=0.0,
        )
        align.add(grid)
        self._page_widget = align  #: Page's layout widget

    @classmethod
    def get_page_icon_name(self):
        return 'mypaint-tool-component-sliders'

    @classmethod
    def get_page_title(self):
        return C_(
            "color sliders panel: tab title (in tooltip)",
            "Component Sliders"
        )

    @classmethod
    def get_page_description(self):
        return C_(
            "color sliders panel: tab description (in tooltip)",
            "Adjust individual components of the color.",
        )

    def get_page_widget(self):
        return self._page_widget

    def set_color_manager(self, manager):
        ColorAdjuster.set_color_manager(self, manager)
        for slider in self._sliders:
            slider.set_color_manager(manager)

    def render_as_icon(self, cr, size):
        """Renders as an icon into a Cairo context.
        """
        # Strategy: construct tmp R,G,B sliders with a color that shows off
        # their primary a bit. Render carefully (might need special handling
        # for the 16px size).
        from adjbases import ColorManager
        mgr = ColorManager(prefs={}, datapath=".")
        mgr.set_color(RGBColor(0.3, 0.3, 0.4))
        adjs = [RGBRedSlider(), RGBGreenSlider(), RGBBlueSlider()]
        for adj in adjs:
            adj.set_color_manager(mgr)
        if size <= 16:
            cr.save()
            for adj in adjs:
                adj.BORDER_WIDTH = 1
                adj.render_background_cb(cr, wd=16, ht=5)
                cr.translate(0, 5)
            cr.restore()
        else:
            cr.save()
            bar_ht = int(size // 3)
            offset = int((size - bar_ht*3) // 2)
            cr.translate(0, offset)
            for adj in adjs:
                adj.BORDER_WIDTH = max(2, int(size // 16))
                adj.render_background_cb(cr, wd=size, ht=bar_ht)
                cr.translate(0, bar_ht)
            cr.restore()
        for adj in adjs:
            adj.set_color_manager(None)


class RGBRedSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "RGB Red")

    def get_background_validity(self):
        col = self.get_managed_color()
        r, g, b = col.get_rgb()
        return g, b

    def get_color_for_bar_amount(self, amt):
        col = RGBColor(color=self.get_managed_color())
        col.r = amt
        return col

    def get_bar_amount_for_color(self, col):
        return col.r


class RGBGreenSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "RGB Green")

    def get_background_validity(self):
        col = self.get_managed_color()
        r, g, b = col.get_rgb()
        return r, b

    def get_color_for_bar_amount(self, amt):
        col = RGBColor(color=self.get_managed_color())
        col.g = amt
        return col

    def get_bar_amount_for_color(self, col):
        return col.g


class RGBBlueSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "RGB Blue")

    def get_background_validity(self):
        col = self.get_managed_color()
        r, g, b = col.get_rgb()
        return r, g

    def get_color_for_bar_amount(self, amt):
        col = RGBColor(color=self.get_managed_color())
        col.b = amt
        return col

    def get_bar_amount_for_color(self, col):
        return col.b


class HSVHueSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "HSV Hue")
    samples = 4

    def get_color_for_bar_amount(self, amt):
        col = HSVColor(color=self.get_managed_color())
        col.h = amt
        return col

    def get_bar_amount_for_color(self, col):
        return col.h


class HSVSaturationSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_(
        "color component slider: tooltip",
        u"HSV Saturation",
    )

    def get_color_for_bar_amount(self, amt):
        col = HSVColor(color=self.get_managed_color())
        col.s = amt
        return col

    def get_bar_amount_for_color(self, col):
        return col.s


class HSVValueSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "HSV Value")

    def get_color_for_bar_amount(self, amt):
        col = HSVColor(color=self.get_managed_color())
        col.v = amt
        return col

    def get_bar_amount_for_color(self, col):
        return col.v


class HCYHueSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "HCY Hue")
    samples = 4

    def get_color_for_bar_amount(self, amt):
        col = HCYColor(color=self.get_managed_color())
        col.h = amt
        return col

    def get_bar_amount_for_color(self, col):
        col = HCYColor(color=col)
        return col.h


class HCYChromaSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "HCY Chroma")

    def get_color_for_bar_amount(self, amt):
        col = HCYColor(color=self.get_managed_color())
        col.c = amt
        return col

    def get_bar_amount_for_color(self, col):
        col = HCYColor(color=col)
        return col.c


class HCYLumaSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_(
        "color component slider: tooltip",
        u"HCY Luma (Y')",
    )

    @property
    def samples(self):
        alloc = self.get_allocation()
        len = self.vertical and alloc.height or alloc.width
        len -= self.BORDER_WIDTH * 2
        return min(int(len // 3), 64)

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
    import os
    import sys
    from adjbases import ColorManager
    mgr = ColorManager(prefs={}, datapath=".")
    cs_adj = ComponentSlidersAdjusterPage()
    cs_adj.set_color_manager(mgr)
    cs_adj.set_managed_color(RGBColor(0.3, 0.6, 0.7))
    if len(sys.argv) > 1:
        icon_name = cs_adj.get_page_icon_name()
        for dir_name in sys.argv[1:]:
            cs_adj.save_icon_tree(dir_name, icon_name)
    else:
        # Interactive test
        window = Gtk.Window()
        window.add(cs_adj.get_page_widget())
        window.set_title(os.path.basename(sys.argv[0]))
        window.connect("destroy", lambda *a: Gtk.main_quit())
        window.show_all()
        Gtk.main()
