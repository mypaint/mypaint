# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Color picker functions.
"""

import gui.pygtkcompat

import gtk
from gtk import gdk
import gobject
from gettext import gettext as _

from adjbases import ColorAdjuster
from uicolor import RGBColor
from uimisc import borderless_button


def get_color_at_pointer(display, size=3):
    """Returns the colour at the current pointer position.

    :param display: the gdk.Display holding the pointer to use
    :param size: integer defining a square over which to sample
    :rtype: `uicolor.RGBColor`.

    The colour returned is averaged over a square of `size`x`size` centred at
    the pointer.

    """
    screen, ptr_x_root, ptr_y_root, mods = display.get_pointer()
    win_info = display.get_window_at_pointer()  # FIXME: deprecated (GTK3)
    if win_info[0]:
        # Window is known to GDK, and is a child window of this app for most
        # screen locations. It's most reliable to poll the colour from its
        # toplevel window.
        win = win_info[0].get_toplevel()
        win_x, win_y = win.get_origin()
        ptr_x = ptr_x_root - win_x
        ptr_y = ptr_y_root - win_y
    else:
        # Window is unknown to GDK: foreign, native, or a window manager frame.
        # Use the old method of reading the colour from the root window even
        # though this is probably of diminishing use these days.
        win = screen.get_root_window()
        ptr_x = ptr_x_root
        ptr_y = ptr_y_root
    return get_color_in_window(win, ptr_x, ptr_y, size)


def get_color_in_window(win, x, y, size=3):
    """Attempts to get the color from a position within a GDK window.
    """

    # [GTK3] GDK2 and GDK3 return different tuples: no bitdepth in GDK3
    # We use GTK3 now but for some reason users still get different results,
    # see https://gna.org/bugs/?20791
    geom_tuple = win.get_geometry()
    win_x, win_y, win_w, win_h = geom_tuple[0:4]

    x = int(max(0, x - size/2))
    y = int(max(0, y - size/2))
    w = int(min(size, win_w - x))
    h = int(min(size, win_h - y))
    if w <= 0 or h <= 0:
        return RGBColor(0, 0, 0)
    # The call below can take over 20ms, and it depends on the window size!
    # It must be causing a full-window pixmap copy somewhere.
    pixbuf = gdk.pixbuf_get_from_window(win, x, y, w, h)
    if pixbuf is None:
        errcol = RGBColor(1, 0, 0)
        print "warning: failed to get pixbuf from screen; returning", errcol
        return errcol
    return RGBColor.new_from_pixbuf_average(pixbuf)


class ColorPickerButton (gtk.EventBox, ColorAdjuster):
    """Button for picking a colour from the screen.
    """

    __grab_mask = gdk.BUTTON_RELEASE_MASK \
                | gdk.BUTTON1_MOTION_MASK \
                | gdk.POINTER_MOTION_MASK
    __picking = False

    def __init__(self):
        gtk.EventBox.__init__(self)
        self.connect("button-release-event", self.__button_release_cb)
        self.connect("motion-notify-event", self.__motion_cb)
        button = borderless_button(stock_id=gtk.STOCK_COLOR_PICKER,
                                   tooltip=_("Pick a color from the screen"))
        button.connect("clicked", self.__clicked_cb)
        self.add(button)

    def __clicked_cb(self, widget):
        gobject.idle_add(self.__begin_color_pick)

    def __begin_color_pick(self):
        mgr = self.get_color_manager()
        cursor = mgr.get_picker_cursor()
        window = self.get_window()
        result = gdk.pointer_grab(window, False, self.__grab_mask,
                                  None, cursor, gdk.CURRENT_TIME)
        if result == gdk.GRAB_SUCCESS:
            self.__picking = True

    def __motion_cb(self, widget, event):
        if not self.__picking:
            return
        if event.state & gdk.BUTTON1_MASK:
            # Due to a performance bug, color picking can take more time
            # than we have between two motion events (about 8ms), see above.
            if hasattr(self, 'delayed_color_pick_id'):
                gobject.source_remove(self.delayed_color_pick_id)
            def delayed_color_pick():
                del self.delayed_color_pick_id
                color = get_color_at_pointer(self.get_display())
                self.set_managed_color(color)
            self.delayed_color_pick_id = gobject.idle_add(delayed_color_pick)

    def __button_release_cb(self, widget, event):
        if not self.__picking:
            return False
        if event.state & gdk.BUTTON1_MASK:
            color = get_color_at_pointer(self.get_display())
            self.set_managed_color(color)
            self.__picking = False
            gdk.pointer_ungrab(gdk.CURRENT_TIME)

