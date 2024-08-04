# This file is part of MyPaint.
# Copyright (C) 2013-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Dockable Workspace tools for color adjusters."""

from __future__ import division, print_function

from lib.gibindings import Gtk

from lib.gettext import C_

from . import widgets
from .toolstack import TOOL_WIDGET_MIN_WIDTH
from gui.colors.hcywheel import HCYAdjusterPage
from gui.colors.hsvwheel import HSVAdjusterPage
from gui.colors.paletteview import PalettePage
from gui.colors.hsvcube import HSVCubePage
from gui.colors.hsvsquare import HSVSquarePage
from gui.colors.sliders import ComponentSlidersAdjusterPage
import gui.colors.changers
from gui.colors import ColorAdjuster


## Adapter classes for old-style "Page" ColorAdjuster classes


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
        self.pack_start(page_widget, True, True, 0)
        self._adjusters = []
        self._adjusters.append(page)
        # Properties button
        properties_desc = self.PAGE_CLASS.get_properties_description()
        if properties_desc is not None:
            show_props = lambda *a: page.show_properties()  # noqa: E731
            self.tool_widget_properties = show_props
        # Adjuster setup
        from gui.application import get_app
        self._app = get_app()
        self.set_color_manager(self._app.brush_color_manager)
        # Sizing.
        size = TOOL_WIDGET_MIN_WIDTH
        self.set_size_request(size, size*0.9)

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


class HSVCubeTool (_PageToolAdapter):
    __gtype_name__ = 'MyPaintHSVCubeTool'
    PAGE_CLASS = HSVCubePage


class HSVSquareTool (_PageToolAdapter):
    __gtype_name__ = 'MyPaintHSVSquareTool'
    PAGE_CLASS = HSVSquarePage


class ComponentSlidersTool (_PageToolAdapter):
    __gtype_name__ = 'MyPaintComponentSlidersTool'
    PAGE_CLASS = ComponentSlidersAdjusterPage


## Adapters for newer ColorAdjusters

class _SimpleAdjusterAdapter (Gtk.VBox):
    """Adapts simple ColorAdjusters to a workspace tool widget.

    Subclasses must provide the following fields:

    * __gtype_name__ (ending with "Tool")
    * tool_widget_icon_name
    * tool_widget_title
    * tool_widget_description
    * ADJUSTER_CLASS

    """

    ADJUSTER_CLASS = None

    def __init__(self):
        super(_SimpleAdjusterAdapter, self).__init__()
        adjuster = self.ADJUSTER_CLASS()
        from gui.application import get_app
        self._app = get_app()
        adjuster.set_color_manager(self._app.brush_color_manager)
        self.pack_start(adjuster, True, True, 0)
        self._adjuster = adjuster

    def set_color_manager(self, manager):
        self._adjuster.set_color_manager(manager)


class WashColorChangerTool (_SimpleAdjusterAdapter):
    __gtype_name__ = "MyPaintWashColorChangerTool"
    ADJUSTER_CLASS = gui.colors.changers.Wash
    tool_widget_icon_name = "mypaint-tool-wash-color-changer"
    tool_widget_title = C_(
        "color changer dock panels: tab tooltip title",
        "Liquid Wash",
    )
    tool_widget_description = C_(
        "color changer dock panels: tab tooltip description",
        "Change color using a liquid-like wash of nearby colors.",
    )


class RingsColorChangerTool (_SimpleAdjusterAdapter):
    __gtype_name__ = "MyPaintRingsColorChangerTool"
    ADJUSTER_CLASS = gui.colors.changers.Rings
    tool_widget_icon_name = "mypaint-tool-rings-color-changer"
    tool_widget_title = C_(
        "color changer dock panels: tab tooltip title",
        "Concentric Rings",
    )
    tool_widget_description = C_(
        "color changer dock panels: tab tooltip description",
        "Change color using concentric HSV rings.",
    )


class CrossedBowlColorChangerTool (_SimpleAdjusterAdapter):
    __gtype_name__ = "MyPaintCrossedBowlColorChangerTool"
    ADJUSTER_CLASS = gui.colors.changers.CrossedBowl
    tool_widget_icon_name = "mypaint-tool-crossed-bowl-color-changer"
    tool_widget_title = C_(
        "color changer dock panels: tab tooltip title",
        "Crossed Bowl",
    )
    tool_widget_description = C_(
        "color changer dock panels: tab tooltip description",
        "Change color with HSV ramps crossing a radial bowl of color.",
    )


def _new_color_adjusters_menu():
    from gui.application import get_app
    app = get_app()
    menu = Gtk.Menu()
    action_names = [
        "HCYWheelTool",
        "HSVWheelTool",
        "PaletteTool",
        "HSVSquareTool",
        "HSVCubeTool",
        "ComponentSlidersTool",
        None,
        "CrossedBowlColorChangerTool",
        "WashColorChangerTool",
        "RingsColorChangerTool",
    ]
    for an in action_names:
        if an is None:
            item = Gtk.SeparatorMenuItem()
        else:
            action = app.find_action(an)
            item = Gtk.MenuItem()
            item.set_use_action_appearance(True)
            item.set_related_action(action)
        menu.append(item)
    return menu


class ColorAdjustersToolItem (widgets.MenuButtonToolItem):
    """Toolbar item for launching any of the available color adjusters

    This is instantiated by the app's UIManager using a FactoryAction which
    must be named "ColorAdjusters" (see factoryaction.py).
    """

    __gtype_name__ = 'MyPaintColorAdjustersToolItem'

    def __init__(self):
        widgets.MenuButtonToolItem.__init__(self)
        self.menu = _new_color_adjusters_menu()


class ColorAdjustersMenuItem (Gtk.MenuItem):
    """Menu item with a static submenu of available color adjusters

    This is instantiated by the app's UIManager using a FactoryAction
    which must be named "ColorAdjusters" (see factoryaction.py).

    """

    __gtype_name__ = "MyPaintColorAdjustersMenuItem"

    def __init__(self):
        Gtk.MenuItem.__init__(self)
        self._submenu = _new_color_adjusters_menu()
        self.set_submenu(self._submenu)
        self._submenu.show_all()
