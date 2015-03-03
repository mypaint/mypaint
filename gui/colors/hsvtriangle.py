# This file is part of MyPaint.
# Copyright (C) 2011,2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""The regular GTK adjuster, wrapped with a ColorAdjuster interface"""

from warnings import warn

from gi.repository import Gtk
from gettext import gettext as _

from util import clamp
from lib.color import HSVColor
from adjbases import ColorAdjuster, PreviousCurrentColorAdjuster
from combined import CombinedAdjusterPage

# This code will grow the sidebars' widths unacceptably if the triangle
# is just allowed to grow. "metrics" != an understandable size, and the
# relationship is somewhat unclear. And the widget is deprecated anyway.
from gui.workspace import TOOL_WIDGET_MIN_WIDTH as MAX_SIZE
from gui.workspace import TOOL_WIDGET_MIN_HEIGHT as MIN_SIZE


class HSVTrianglePage (CombinedAdjusterPage):

    __table = None
    __adj = None

    def __init__(self):
        CombinedAdjusterPage.__init__(self)
        adj = HSVTriangle()
        self.__adj = adj
        self.__table = Gtk.Table(rows=1, columns=1)
        opts = Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND
        self.__table.attach(adj, 0, 1, 0, 1, opts, opts, 3, 3)

    @classmethod
    def get_page_icon_name(class_):
        return "mypaint-tool-color-triangle"

    @classmethod
    def get_page_title(class_):
        return _("HSV Triangle")

    @classmethod
    def get_page_description(class_):
        return _("The standard GTK color selector")

    def get_page_widget(self):
        return self.__table

    def set_color_manager(self, manager):
        ColorAdjuster.set_color_manager(self, manager)
        self.__adj.set_color_manager(manager)


class HSVTriangle (Gtk.VBox, ColorAdjuster):
    """Wrapper around a GtkHSV triangle widget, bound to the app instance"""

    __gtype_name__ = 'HSVTriangle'

    def __init__(self):
        """Initialize"""
        Gtk.VBox.__init__(self)
        ColorAdjuster.__init__(self)
        self._updating = False
        hsv = Gtk.HSV()
        hsv.set_size_request(MIN_SIZE, MIN_SIZE)
        hsv.connect("changed", self._hsv_changed_cb)
        hsv.connect("size-allocate", self._hsv_alloc_cb)
        self.pack_start(hsv, True, True)
        self._hsv_widget = hsv

    def _hsv_alloc_cb(self, hsv, alloc):
        # When extra space is given, grow the HSV wheel.
        old_radius, ring_width = hsv.get_metrics()
        new_radius = min(alloc.width, alloc.height)
        new_radius = clamp(new_radius, MIN_SIZE, MAX_SIZE)
        new_radius -= ring_width
        if new_radius != old_radius:
            hsv.set_metrics(new_radius, ring_width)
            hsv.queue_draw()

    def color_updated(self):
        if self._updating:
            return
        self._updating = True
        color = self.get_managed_color()
        self._hsv_widget.set_color(*color.get_hsv())
        self._updating = False

    def set_current_color(self, color):
        self._hsv_widget.set_color(*color.get_hsv())

    def _hsv_changed_cb(self, hsv):
        if hsv.is_adjusting():
            return
        h, s, v = hsv.get_color()
        color = HSVColor(h, s, v)
        self.set_managed_color(color)


if __name__ == '__main__':
    from adjbases import ColorManager
    mgr = ColorManager(prefs={}, datapath='.')
    win = Gtk.Window()
    win.set_title("hsvtriangle test")
    win.connect("destroy", Gtk.main_quit)
    hsv = HSVTriangle()
    hsv.set_color_manager(mgr)
    win.add(hsv)
    win.show_all()
    Gtk.main()
