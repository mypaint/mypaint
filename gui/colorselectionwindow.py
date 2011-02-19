# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"select color window (GTK and an own window)"
from gettext import gettext as _

import gtk
gdk = gtk.gdk

import windowing
from lib import helpers, mypaintlib


class ToolWidget (gtk.AspectFrame):
    """Tool widget with the standard GTK color selector (triangle)."""

    tool_widget_title = _('Color')

    def __init__(self, app):
        gtk.AspectFrame.__init__(self, None, 0.5, 0.5, 1.2)
        self.set_shadow_type(gtk.SHADOW_NONE)
        self.app = app
        self.app.brush.settings_observers.append(self.brush_modified_cb)

        self.cs = gtk.ColorSelection()
        self.cs.connect('realize', self.on_cs_realize)
        self.cs.connect('color-changed', self.color_changed_cb)
        self.add(self.cs)

        self.cs.set_size_request(200, 240)

        self.in_callback = False


    def on_cs_realize(self, *ignore):
        # Remove unwanted widgets
        # FIXME: really we should be using a gtk.HSV instead of all this crap.
        hbox= self.cs.get_children()[0]
        hbox.set_no_show_all(True)
        l,r = hbox.get_children()
        r.hide()
        return True

    def color_changed_cb(self, cs):
        if self.in_callback:
            return
        b = self.app.brush
        b.set_color_hsv(self.get_color_hsv())

    def brush_modified_cb(self):
        brush_color = self.app.brush.get_color_hsv()
        if brush_color != self.get_color_hsv():
            self.in_callback = True
            self.set_color_hsv(brush_color)
            self.in_callback = False

    def get_color_hsv(self):
        c = self.cs.get_current_color()
        r = float(c.red  ) / 65535
        g = float(c.green) / 65535
        b = float(c.blue ) / 65535
        assert r >= 0.0
        assert g >= 0.0
        assert b >= 0.0
        assert r <= 1.0
        assert g <= 1.0
        assert b <= 1.0
        h, s, v = helpers.rgb_to_hsv(r, g, b)
        return (h, s, v)

    def set_color_hsv(self, hsv):
        h, s, v = hsv
        while h > 1.0: h -= 1.0
        while h < 0.0: h += 1.0
        if s > 1.0: s = 1.0
        if s < 0.0: s = 0.0
        if v > 1.0: v = 1.0
        if v < 0.0: v = 0.0
        rgb  = helpers.hsv_to_rgb(h, s, v)
        self.set_color_rgb(rgb)

    def set_color_rgb(self,rgb):
        r,g,b = rgb
        c = gdk.Color(int(r*65535+0.5), int(g*65535+0.5), int(b*65535+0.5))
        current_color = self.cs.get_current_color()
        self.cs.set_previous_color(current_color)
        self.cs.set_current_color(c)

# own color selector
# see also colorchanger.hpp
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
        self.window.set_cursor(gdk.Cursor(gdk.CROSSHAIR))
        self.mouse_pos = None

    def leave(self, reason):
        # TODO: make a generic "popupmenu" class with this code?
        if reason == 'keyup':
            if self.mouse_pos:
                self.pick_color(*self.mouse_pos)
        self.hide()

    def update_image(self):
        size = self.backend.get_size()
        pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, True, 8, size, size)
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        self.backend.set_brush_color(*self.app.brush.get_color_hsv())
        self.backend.render(arr)
        pixmap, mask = pixbuf.render_pixmap_and_mask()
        self.image.set_from_pixmap(pixmap, mask)
        self.shape_combine_mask(mask,0,0)

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



class ColorChangerPopup(ColorSelectorPopup):
    backend_class = mypaintlib.ColorChanger
    outside_popup_timeout = 0.050

class ColorRingPopup(ColorSelectorPopup):
    backend_class = mypaintlib.SCWSColorSelector
    closes_on_picking = False
    outside_popup_timeout = 0.050

