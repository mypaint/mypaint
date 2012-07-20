# This file is part of MyPaint.
# Copyright (C) 2007-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Various popup colour selectors.
"""

import pygtkcompat
import gtk
from gtk import gdk

import windowing
from lib import mypaintlib
from lib.helpers import gdkpixbuf2numpy


class ColorSelectorPopup(windowing.PopupWindow):
    backend_class = None
    closes_on_picking = True
    def __init__(self, app):
        windowing.PopupWindow.__init__(self, app)

        self.backend = self.backend_class()

        self.image = image = gtk.Image()
        self.add(image)

        self.set_events(gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gtk.gdk.POINTER_MOTION_MASK |
                        gdk.ENTER_NOTIFY |
                        gdk.LEAVE_NOTIFY
                        )
        self.connect("button-release-event", self.button_release_cb)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)

        self.button_pressed = False

    def enter(self):
        self.update_image()
        self.show_all()
        self.get_window().set_cursor(gdk.Cursor(gdk.CROSSHAIR))
        self.mouse_pos = None

    def leave(self, reason):
        # TODO: make a generic "popupmenu" class with this code?
        if reason == 'keyup':
            if self.mouse_pos:
                self.pick_color(*self.mouse_pos)
        self.hide()

    def update_image(self):
        size = self.backend.get_size()
        pixbuf = pygtkcompat.gdk.pixbuf.new(gdk.COLORSPACE_RGB, True, 8,
                                            size, size)
        arr = gdkpixbuf2numpy(pixbuf)
        self.backend.set_brush_color(*self.app.brush.get_color_hsv())
        self.backend.render(arr)
        if pygtkcompat.USE_GTK3:
            # This seems to work, but is pointless until we can get shaped
            # windows working. Might need a NumPy RGBA => ARGB oneliner.
            #
            #   rowstride = arr.strides[0]
            #   surf = cairo.ImageSurface.create_for_data(arr.data,
            #                                             cairo.FORMAT_ARGB32,
            #                                             size, size, rowstride)

            # TODO:FUTURE: Want to be able to do:
            #
            #   region = gdk.cairo_region_create_from_surface(surf)
            #
            # and then use self.shape_combine_region(), but some Cairo structs
            # aren't wrapped: http://stackoverflow.com/questions/6133622 is
            # happening as of python-gi-cairo 3.2.2-1~precise and 
            pass
        else:
            pixmap, mask = pixbuf.render_pixmap_and_mask()
            self.image.set_from_pixmap(pixmap, mask)
            self.shape_combine_mask(mask,0,0)
        self.image.set_from_pixbuf(pixbuf)

    def pick_color(self,x,y):
        hsv = self.backend.pick_color_at(x, y)
        if hsv:
            self.app.brush.set_color_hsv(hsv)
        else:
            self.hide()

    def motion_notify_cb(self, widget, event):
        self.mouse_pos = event.x, event.y

    def button_press_cb(self, widget, event):
        if event.button == 1:
            self.pick_color(event.x,event.y)
        self.button_pressed = True

    def button_release_cb(self, widget, event):
        if self.button_pressed:
            if event.button == 1:
                self.pick_color(event.x,event.y)
                if self.closes_on_picking:
                    # FIXME: hacky?
                    self.popup_state.leave()
                else:
                    self.update_image()



class ColorChangerWashPopup(ColorSelectorPopup):
    backend_class = mypaintlib.ColorChangerWash
    outside_popup_timeout = 0.050

class ColorChangerCrossedBowlPopup(ColorSelectorPopup):
    backend_class = mypaintlib.ColorChangerCrossedBowl
    outside_popup_timeout = 0.050

class ColorRingPopup(ColorSelectorPopup):
    backend_class = mypaintlib.SCWSColorSelector
    closes_on_picking = False
    outside_popup_timeout = 0.050

