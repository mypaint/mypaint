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

from gi.repository import Gtk
import colour
import numpy as np

from lib.color import RGBColor
from lib.color import HSVColor
from lib.color import HCYColor
from lib.color import CAM16Color, CCT_to_RGB, RGB_to_CCT
from .bases import IconRenderable
from .adjbases import ColorAdjuster
from .adjbases import SliderColorAdjuster
from .combined import CombinedAdjusterPage

from lib.gettext import C_


class ComponentSlidersAdjusterPage (CombinedAdjusterPage, IconRenderable):
    """Component sliders for precise adjustment: page for `CombinedAdjuster`.
    """

    def __init__(self):
        from gui.application import get_app
        self.app = get_app()
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
                'RGBRedSlider',
                0,
            ), (
                C_("color sliders panel: red/green/blue: slider label", "G"),
                'RGBGreenSlider',
                0,
            ), (
                C_("color sliders panel: red/green/blue: slider label", "B"),
                'RGBBlueSlider',
                0,
            ), (
                C_("color sliders panel: hue/saturation/value: slider label",
                   "H"),
                'HSVHueSlider',
                12,
            ), (
                C_("color sliders panel: hue/saturation/value: slider label",
                   "S"),
                'HSVSaturationSlider',
                0,
            ), (
                C_("color sliders panel: hue/saturation/value: slider label",
                   "V"),
                'HSVValueSlider',
                0,
            ), (
                C_("color sliders panel: hue/chroma/luma: slider label", "H"),
                'HCYHueSlider',
                12,
            ), (
                C_("color sliders panel: hue/chroma/luma: slider label", "C"),
                'HCYChromaSlider',
                0,
            ), (
                C_("color sliders panel: hue/chroma/luma: slider label", "Y'"),
                'HCYLumaSlider',
                0,
            ), (
                C_("color sliders panel: hue/chroma/luma: slider label", "hue"),
                'CAM16HueSlider',
                12,
            ), (
                C_("color sliders panel: hue/chroma/luma: slider label", "chroma"),
                'CAM16ChromaSlider',
                0,
            ), (
                C_("color sliders panel: hue/chroma/luma: slider label", "value"),
                'CAM16LumaSlider',
                0,
            ), (
                C_("color sliders panel: Temperature: slider label", "warm/cool"),
                'CAM16TempSlider',
                12,
            ), (
                C_("color sliders panel: Limit Purity: slider label", "chroma limit"),
                'CAM16LimitChromaSlider',
                0,
            ), (
                C_("color sliders panel: hue/chroma/luma: slider label",
                   "reset"),
                'CAM16HueNormSlider',
                0,
            ),
        ]
        row = 0
        for row_def in row_defs:
            label_text, adj_class, margin_top = row_def
            active = self.app.preferences["ui.sliders_enabled"].get(
                adj_class, True)
            if active:
                self.app.preferences["ui.sliders_enabled"][adj_class] = True
                adj_class = globals()[adj_class]
                label = Gtk.Label()
                label.set_text(label_text)
                label.set_tooltip_text(adj_class.STATIC_TOOLTIP_TEXT)
                label.set_vexpand(True)
                label.set_hexpand(False)
                label.set_valign(0.0)
                label.set_margin_top(margin_top)
                label.set_margin_left(3)
                label.set_margin_right(3)
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
        from .adjbases import ColorManager
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
        return max(0.0, col.r)


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
        return max(0.0, col.g)


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
        return max(0.0, col.b)


class HSVHueSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "HSV Hue")
    samples = 4

    def get_color_for_bar_amount(self, amt):
        col = HSVColor(color=self.get_managed_color())
        col.h = amt
        return col

    def get_bar_amount_for_color(self, col):
        col = HSVColor(color=col)
        return max(0.0, col.h)


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
        col = HSVColor(color=col)
        return max(0.0, col.s)


class HSVValueSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "HSV Value")

    def get_color_for_bar_amount(self, amt):
        col = HSVColor(color=self.get_managed_color())
        col.v = amt
        return col

    def get_bar_amount_for_color(self, col):
        col = HSVColor(color=col)
        return max(0.0, col.v)


class HCYHueSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "HCY Hue")
    samples = 4

    def get_color_for_bar_amount(self, amt):
        col = HCYColor(color=self.get_managed_color())
        col.h = amt
        return col

    def get_bar_amount_for_color(self, col):
        col = HCYColor(color=col)
        return max(0.0, col.h)


class HCYChromaSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "HCY Chroma")

    def get_color_for_bar_amount(self, amt):
        col = HCYColor(color=self.get_managed_color())
        col.c = amt
        return col

    def get_bar_amount_for_color(self, col):
        col = HCYColor(color=col)
        return max(0.0, col.c)


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
        return min(int(len // 10), 5)

    def get_color_for_bar_amount(self, amt):
        col = HCYColor(color=self.get_managed_color())
        col.y = amt
        return col

    def get_bar_amount_for_color(self, col):
        col = HCYColor(color=col)
        return max(0.0, col.y)

    def get_background_validity(self):
        col = HCYColor(color=self.get_managed_color())
        return int(col.h * 1000), int(col.c * 1000)


class CAM16HueNormSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip",
                             "CAM16 Hue @ s=35, v=50, D65")

    @property
    def samples(self):
        alloc = self.get_allocation()
        len = self.vertical and alloc.height or alloc.width
        len -= self.BORDER_WIDTH * 2
        return min(int(len // 10), 8)

    def get_color_for_bar_amount(self, amt):
        col = self._get_app_brush_color()
        col.limit_purity = None
        col.illuminant = None
        col.cachedrgb = None
        col.h = max(0.0, amt) * 360
        col.v = 50
        col.s = 35
        return col

    def get_bar_amount_for_color(self, col):
        col = self._get_app_brush_color()
        return max(0.0, col.h) / 360

    def get_background_validity(self):
        # This bg should never change
        return True


class CAM16HueSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip", "CAM16 Hue")
    draw_background = True

    @property
    def samples(self):
        alloc = self.get_allocation()
        len = self.vertical and alloc.height or alloc.width
        len -= self.BORDER_WIDTH * 2
        return min(int(len // 10), 16)

    def get_color_for_bar_amount(self, amt):
        col = self._get_app_brush_color()
        col.h = max(0.0, amt) * 360
        col.cachedrgb = None
        col.gamutmapping = "highlightH"
        return col

    def get_bar_amount_for_color(self, col):
        col = self._get_app_brush_color()
        return max(0.0, col.h) / 360

    def get_background_validity(self):
        col = self._get_app_brush_color()
        illuminant = (col.illuminant[0], col.illuminant[1], col.illuminant[2])
        return col.v, col.s, col.h, illuminant, col.limit_purity


class CAM16ChromaSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip",
                             "CAM16 Colorfulness/Chroma/Saturation")
    draw_background = True

    @property
    def samples(self):
        alloc = self.get_allocation()
        len = self.vertical and alloc.height or alloc.width
        len -= self.BORDER_WIDTH * 2
        return min(int(len // 10), 16)

    def get_color_for_bar_amount(self, amt):
        col = self._get_app_brush_color()
        col.s = max(0.0, amt) * 120
        col.gamutmapping = "highlightC"
        col.cachedrgb = None
        return col

    def get_bar_amount_for_color(self, col):
        col = self._get_app_brush_color()
        if col.limit_purity is not None:
            return min(col.s, col.limit_purity) / 120
        else:
            return max(0.0, col.s) / 120

    def get_background_validity(self):
        from gui.application import get_app
        app = get_app()
        cm = self.get_color_manager()
        prefs = cm.get_prefs()
        try:
            if app.brush.get_setting('cie_v') == '':
                return True
            limit_purity = prefs['color.limit_purity']
            vsh = (
                int(app.brush.get_setting('cie_v') * 100),
                int(app.brush.get_setting('cie_s') * 100),
                int(app.brush.get_setting('cie_h') * 100))

            cieaxes = app.brush.get_setting('cieaxes'),
            illuminant = (
                app.brush.get_setting('illuminant_X'),
                app.brush.get_setting('illuminant_Y'),
                app.brush.get_setting('illuminant_Z'))
        except KeyError:
            return True
        return vsh, cieaxes, illuminant, limit_purity


class CAM16LimitChromaSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip",
                             "CAM16 Limit Purity/Chroma")
    draw_background = True

    @property
    def samples(self):
        alloc = self.get_allocation()
        len = self.vertical and alloc.height or alloc.width
        len -= self.BORDER_WIDTH * 2
        return min(int(len // 20), 3)

    def get_color_for_bar_amount(self, amt):
        col = self._get_app_brush_color()
        if amt == 1.0:
            col.limit_purity = None
        else:
            amt = max(0.0, amt)
            col.s = amt * 120
            col.limit_purity = amt * 120
        col.gamutmapping = "highlight"
        col.cachedrgb = None
        return col

    def get_bar_amount_for_color(self, col):
        # pull in color purity preference
        cm = self.get_color_manager()
        prefs = cm.get_prefs()
        limit_purity = prefs['color.limit_purity']
        if limit_purity >= 0.0:
            return max(0.0, limit_purity) / 120
        else:
            return 1.0

    def get_background_validity(self):
        from gui.application import get_app
        app = get_app()
        cm = self.get_color_manager()
        prefs = cm.get_prefs()
        try:
            if app.brush.get_setting('cie_v') == '':
                return True
            limit_purity = prefs['color.limit_purity']
            vsh = (
                int(app.brush.get_setting('cie_v') * 100),
                int(app.brush.get_setting('cie_s') * 100),
                int(app.brush.get_setting('cie_h') * 100))

            cieaxes = app.brush.get_setting('cieaxes'),
            illuminant = (
                app.brush.get_setting('illuminant_X'),
                app.brush.get_setting('illuminant_Y'),
                app.brush.get_setting('illuminant_Z'))
        except KeyError:
            return True
        return vsh, cieaxes, illuminant, limit_purity


class CAM16LumaSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip",
                             "CAM16 Lightness/Brightness")
    draw_background = True

    @property
    def samples(self):
        alloc = self.get_allocation()
        len = self.vertical and alloc.height or alloc.width
        len -= self.BORDER_WIDTH * 2
        return min(int(len // 3), 8)

    def get_color_for_bar_amount(self, amt):
        col = self._get_app_brush_color()
        col.v = max(0.0, amt) * 100
        col.cachedrgb = None
        col.gamutmapping = "highlightL"
        return col

    def get_bar_amount_for_color(self, col):
        col = self._get_app_brush_color()
        return max(0.0, col.v) / 100

    def get_background_validity(self):
        from gui.application import get_app
        app = get_app()
        cm = self.get_color_manager()
        prefs = cm.get_prefs()
        try:
            if app.brush.get_setting('cie_v') == '':
                return True
            limit_purity = prefs['color.limit_purity']
            vsh = (
                int(app.brush.get_setting('cie_v') * 100),
                int(app.brush.get_setting('cie_s') * 100),
                int(app.brush.get_setting('cie_h') * 100))

            cieaxes = app.brush.get_setting('cieaxes'),
            illuminant = (
                app.brush.get_setting('illuminant_X'),
                app.brush.get_setting('illuminant_Y'),
                app.brush.get_setting('illuminant_Z'))
        except KeyError:
            return True
        return vsh, cieaxes, illuminant, limit_purity


class CAM16TempSlider (SliderColorAdjuster):
    STATIC_TOOLTIP_TEXT = C_("color component slider: tooltip",
                             "CAM16 Color Temperature")

    @property
    def samples(self):
        alloc = self.get_allocation()
        len = self.vertical and alloc.height or alloc.width
        len -= self.BORDER_WIDTH * 2
        return min(int(len // 20), 2)

    def get_color_for_bar_amount(self, amt):
        # CCT range from 2500-20000
        # below 1904 is out of sRGB gamut
        # power function to put 6500k near middle
        cct = amt**2. * 20000. + 2500.
        rgb = CCT_to_RGB(cct)
        col = RGBColor(rgb=rgb)
        return col

    def get_bar_amount_for_color(self, col):
        col = self._get_app_brush_color()
        # return CCT in domain of 0-1
        xy = colour.XYZ_to_xy(np.array(col.illuminant))
        cct = colour.temperature.xy_to_CCT_Hernandez1999(xy)
        amt = ((cct - 2500.) / 20000.)**(1./2.)
        return max(0.0, amt)

    def get_background_validity(self):
        # This bg should never change
        return True


if __name__ == '__main__':
    import os
    import sys
    from .adjbases import ColorManager
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
