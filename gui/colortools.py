# This file is part of MyPaint.
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Dockable Workspace tools for color adjusters."""

import gi
from gi.repository import Gtk

from gettext import gettext as _

import workspace

# Old-style "Page" classes to be adapted (refactor these one day)
from colors.hcywheel import HCYAdjusterPage
from colors.hsvwheel import HSVAdjusterPage
from colors.paletteview import PalettePage
from colors.hsvtriangle import HSVTrianglePage
from colors.hsvcube import HSVCubePage
from colors.sliders import ComponentSlidersAdjusterPage

from colors import ColorAdjuster
from colors import ColorPickerButton
from colors import PreviousCurrentColorAdjuster
from colors.uimisc import borderless_button


class _PageToolAdapter (Gtk.VBox, ColorAdjuster):
    """Adapts the CombinedAdjusterPage interface to a workspace tool widget"""
 
    #: The CombinedAdjusterPage class to adapt.
    #: Subclasses must override this and set a __gtype_name__.
    PAGE_CLASS = None

    def __init__(self):
        """Construct a tool widget with subwidgets from self.PAGE_CLASS."""
        # Superclass setup
        Gtk.VBox.__init__(self)
        self.set_spacing(3)
        self.set_border_width(3)
        # Fields for Workspace's use
        self.tool_widget_icon_name = self.PAGE_CLASS.get_page_icon_name()
        self.tool_widget_title = self.PAGE_CLASS.get_page_title()
        self.tool_widget_description = self.PAGE_CLASS.get_page_description()
        # Main adjuster widget
        page = self.PAGE_CLASS()
        page_widget = page.get_page_widget()
        self.pack_start(page_widget, True, True)
        self._adjusters = []
        self._adjusters.append(page)
        # Properties button
        properties_desc = self.PAGE_CLASS.get_properties_description()
        if properties_desc is not None:
            show_props = lambda *a: page.show_properties()
            self.tool_widget_properties = show_props
        # Adjuster setup
        from application import get_app
        self._app = get_app()
        self.set_color_manager(self._app.brush_color_manager)
        # Sizing.
        size = workspace.TOOL_WIDGET_MIN_WIDTH
        self.set_size_request(size, size*0.9)


    def _bookmark_button_clicked_cb(self, button):
        # Same as the PaletteAddCurrentColor
        mgr = self.get_color_manager()
        col = mgr.get_color()
        mgr.palette.append(col, name=None, unique=True, match=True)


    def set_color_manager(self, manager):
        ColorAdjuster.set_color_manager(self, manager)
        for adj in self._adjusters:
            adj.set_color_manager(manager)


class HCYWheelTool (_PageToolAdapter):
    __gtype_name__ = 'MyPaintHCYWheelTool'
    PAGE_CLASS = HCYAdjusterPage


class HSVWheelTool (_PageToolAdapter):
    __gtype_name__ = 'MyPaintHSVWheelTool'
    PAGE_CLASS = HSVAdjusterPage


class PaletteTool (_PageToolAdapter):
    __gtype_name__ = 'MyPaintPaletteTool'
    PAGE_CLASS = PalettePage


class HSVTriangleTool (_PageToolAdapter):
    __gtype_name__ = 'MyPaintHSVTriangleTool'
    PAGE_CLASS = HSVTrianglePage


class HSVCubeTool (_PageToolAdapter):
    __gtype_name__ = 'MyPaintHSVCubeTool'
    PAGE_CLASS = HSVCubePage


class ComponentSlidersTool (_PageToolAdapter):
    __gtype_name__ = 'MyPaintComponentSlidersTool'
    PAGE_CLASS = ComponentSlidersAdjusterPage


