# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""The old popup color changers wrapped as new-style Adjusters"""


## Imports
from __future__ import division, print_function

from lib.gibindings import GdkPixbuf
from lib.gibindings import Gdk

import lib.color
import gui.colors
import gui.colors.adjbases
from lib.gettext import C_
from lib import mypaintlib
from lib.helpers import gdkpixbuf2numpy


## Class definitions


class _CColorChanger (gui.colors.adjbases.IconRenderableColorAdjusterWidget):
    """Color changer with a C++ backend in mypaintlib

    These are the old popup colour changers, exposed as sidebar-friendly
    new-style colour adjuster widgets.

    """

    #: Subclasses must provide the C++ backend class here.
    BACKEND_CLASS = None

    def __init__(self):
        super(_CColorChanger, self).__init__()
        self._backend = self.BACKEND_CLASS()
        size = self._backend.get_size()
        self.set_size_request(size, size)
        self._dy = 0
        self._dx = 0
        self._hsv = (0, 0, 0)
        self.connect("map", self._map_cb)
        self.add_events(Gdk.EventMask.STRUCTURE_MASK)

    def color_updated(self):
        self._update_hsv()
        super(_CColorChanger, self).color_updated()

    def render_background_cb(self, cr, wd, ht, icon_border=None):
        self._backend.set_brush_color(*self._hsv)
        size = self._backend.get_size()
        pixbuf = GdkPixbuf.Pixbuf.new(
            GdkPixbuf.Colorspace.RGB, True, 8,
            size, size,
        )
        arr = gdkpixbuf2numpy(pixbuf)
        self._backend.render(arr)
        self._dx = (wd - size) // 2
        self._dy = (ht - size) // 2
        cr.translate(self._dx, self._dy)
        Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
        cr.paint()

    def render_as_icon(self, cr, size):
        backend_size = float(self._backend.get_size())
        scale = size / backend_size
        cr.translate(size // 2, size // 2)
        cr.scale(scale, scale)
        cr.translate(-size // 2, -size // 2)
        super(_CColorChanger, self).render_as_icon(cr, size)

    def get_color_at_position(self, x, y):
        x -= self._dx
        y -= self._dy
        size = self._backend.get_size()
        if (0 <= x < size) and (0 <= y < size):
            hsv = self._backend.pick_color_at(x, y)
            if hsv:
                return lib.color.HSVColor(*hsv)
        return None

    def paint_foreground_cb(self, cr, wd, ht):
        pass

    def set_color_manager(self, manager):
        super(_CColorChanger, self).set_color_manager(manager)
        self._update_hsv()

    def _map_cb(self, widget):
        self._update_hsv()

    def _update_hsv(self):
        col = lib.color.HSVColor(color=self.get_managed_color())
        self._hsv = col.get_hsv()


class CrossedBowl (_CColorChanger):
    """Color changer with HSV ramps crossing a sort of bowl thing."""
    BACKEND_CLASS = mypaintlib.ColorChangerCrossedBowl
    IS_IMMEDIATE = False


class Wash (_CColorChanger):
    """Weird trippy wash of colors."""
    BACKEND_CLASS = mypaintlib.ColorChangerWash
    IS_IMMEDIATE = False


class Rings (_CColorChanger):
    """HSV color rings, nested one inside the other."""
    BACKEND_CLASS = mypaintlib.SCWSColorSelector
    IS_IMMEDIATE = True


## Testing and icon generation


if __name__ == '__main__':
    from lib.gibindings import Gtk
    import os
    import sys
    mgr = gui.colors.ColorManager(prefs={}, datapath='.')
    widget_classes = [CrossedBowl, Wash, Rings]
    widgets = []
    for widget_class in widget_classes:
        widget = widget_class()
        widget.set_color_manager(mgr)
        widgets.append(widget)
    mgr.set_color(lib.color.RGBColor(0.3, 0.6, 0.7))
    if len(sys.argv) > 1:
        for dir_name in sys.argv[1:]:
            for w in widgets:
                icon_name = w.__class__.__name__
                w.save_icon_tree(dir_name, icon_name)
    else:
        # Interactive test
        window = Gtk.Window()
        grid = Gtk.Grid()
        r = 1
        for w in widgets:
            w.set_hexpand(True)
            w.set_vexpand(True)
            grid.attach(w, 0, r, 1, 1)
            r += 1
        window.add(grid)
        window.set_title(os.path.basename(sys.argv[0]))
        window.connect("destroy", lambda *a: Gtk.main_quit())
        window.show_all()
        Gtk.main()
