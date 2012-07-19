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


def get_color_at_pointer(widget, size=3):
    """Returns the colour at the current pointer position.

    The colour returned is averaged over a square of `size`x`size` centred at
    the pointer.

    """
    # Utility function which might be more usefully migrated to UIColor as (yet
    # another) constructor classmethod.
    display = widget.get_display()
    screen, x_root, y_root, modifiermask = display.get_pointer()
    root_win = screen.get_root_window()
    screen_w, screen_h = screen.get_width(), screen.get_height()
    x = x_root - size/2
    y = y_root - size/2
    if x < 0: x = 0
    if y < 0: y = 0
    if x+size > screen_w: x = screen_w-size
    if y+size > screen_h: y = screen_h-size
    if gui.pygtkcompat.USE_GTK3:
        pixbuf = gdk.pixbuf_get_from_window(root_win, x, y, size, size)
    else:
        colormap = screen.get_system_colormap()
        pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, size, size)
        pixbuf.get_from_drawable(root_win, colormap, x, y, 0, 0, size, size)
    color = RGBColor.new_from_pixbuf_average(pixbuf)
    return color



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
            color = get_color_at_pointer(self)
            self.set_managed_color(color)

    def __button_release_cb(self, widget, event):
        if not self.__picking:
            return False
        if event.state & gdk.BUTTON1_MASK:
            color = get_color_at_pointer(self)
            self.set_managed_color(color)
            self.__picking = False
            gdk.pointer_ungrab(gdk.CURRENT_TIME)

