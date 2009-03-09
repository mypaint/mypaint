# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"select color window (GTK and an own window)"
import gtk, gobject
import colorsys
from lib import helpers, mypaintlib
gdk = gtk.gdk

# GTK selector
class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.add_accel_group(self.app.accel_group)
        self.app.brush.settings_observers.append(self.brush_modified_cb)

        self.set_title('Color')
        self.connect('delete-event', self.app.hide_window_cb)
        def set_hint(widget):
            self.window.set_type_hint(gdk.WINDOW_TYPE_HINT_UTILITY)
        self.connect("realize", set_hint)

        vbox = gtk.VBox()
        self.add(vbox)

        self.cs = gtk.ColorSelection()
        self.cs.connect('color-changed', self.color_changed_cb)
        vbox.pack_start(self.cs)

        self.last_known_color_hsv = (None, None, None)
        self.change_notification = True

    def color_changed_cb(self, cs):
        if not self.change_notification:
            return
        b = self.app.brush
        b.set_color_hsv(self.get_color_hsv())

    def brush_modified_cb(self):
        if not self.change_notification:
            return
        self.change_notification = False
        self.set_color_hsv(self.app.brush.get_color_hsv())
        self.change_notification = True

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
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return (h, s, v)

    def set_color_hsv(self, hsv):
        if not self.change_notification:
            return
        if hsv == self.last_known_color_hsv:
            return
        self.last_known_color_hsv = hsv
        h, s, v = hsv
        while h > 1.0: h -= 1.0
        while h < 0.0: h += 1.0
        if s > 1.0: s = 1.0
        if s < 0.0: s = 0.0
        if v > 1.0: v = 1.0
        if v < 0.0: v = 0.0
        r, g, b  = colorsys.hsv_to_rgb(h, s, v)
        c = gdk.Color(int(r*65535+0.5), int(g*65535+0.5), int(b*65535+0.5))
        # only emit color_changed events if the user directly interacts with the window
        self.change_notification = False
        self.cs.set_current_color(c)
        self.change_notification = True

    def pick_color_at_pointer(self, size=3):
        # grab screen color at cursor (average of size x size rectangle)
        # inspired by gtkcolorsel.c function grab_color_at_mouse()
        screen = self.get_screen()
        colormap = screen.get_system_colormap()
        root = screen.get_root_window()
        screen_w, screen_h = screen.get_width(), screen.get_height()
        display = self.get_display()
        screen_trash, x_root, y_root, modifiermask_trash = display.get_pointer()
        image = None
        x = x_root-size/2
        y = y_root-size/2
        if x < 0: x = 0
        if y < 0: y = 0
        if x+size > screen_w: x = screen_w-size
        if y+size > screen_h: y = screen_h-size
        image = root.get_image(x, y, size, size)
        color_total = (0, 0, 0)
        for x, y in helpers.iter_rect(0, 0, size, size):
            pixel = image.get_pixel(x, y)
            color = colormap.query_color(pixel)
            color = [color.red, color.green, color.blue]
            color_total = (color_total[0]+color[0], color_total[1]+color[1], color_total[2]+color[2])
        N = size*size
        color_total = (color_total[0]/N, color_total[1]/N, color_total[2]/N)
        self.cs.set_current_color(gdk.Color(*color_total))


# own color selector
# see also colorselector.hpp
class ColorSelectorPopup(gtk.Window):
    backend_class = None
    closes_on_picking = True
    def __init__(self, app):
        gtk.Window.__init__(self, gtk.WINDOW_POPUP)
        self.set_gravity(gdk.GRAVITY_CENTER)
        self.set_position(gtk.WIN_POS_MOUSE)

        self.backend = self.backend_class()
        
        self.app = app
        self.add_accel_group(self.app.accel_group)

        #self.set_title('Color')
        self.connect('delete-event', self.app.hide_window_cb)

        self.image = image = gtk.Image()
        self.add(image)
        
	self.set_events(gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.ENTER_NOTIFY |
                        gdk.LEAVE_NOTIFY
                        )
        self.connect("button-release-event", self.button_release_cb)
        self.connect("button-press-event", self.button_press_cb)

        self.button_pressed = False

    def enter(self):
        self.update_image()
        self.show_all()
        self.window.set_cursor(gdk.Cursor(gdk.CROSSHAIR))
    
    def leave(self):
        self.hide()

    def update_image(self):
        size = self.backend.size
        pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, True, 8, size, size)
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        self.backend.set_brush_color(*self.app.brush.get_color_hsv())
        self.backend.render(arr)
        pixmap, mask = pixbuf.render_pixmap_and_mask()
        self.image.set_from_pixmap(pixmap, mask)
        self.shape_combine_mask(mask,0,0)
        
    def pick_color(self,x,y):
        hsv = self.backend.pick_color_at(x, y)
        self.app.brush.set_color_hsv(hsv)
    
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



class ChangeColorPopup(ColorSelectorPopup):
    backend_class = mypaintlib.ColorChanger 
    outside_popup_timeout = 0.050

class ColorWheelPopup(ColorSelectorPopup):
    backend_class = mypaintlib.SCWSColorSelector 
    closes_on_picking = False
    outside_popup_timeout = 0.200

