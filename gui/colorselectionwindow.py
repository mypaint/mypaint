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


class ToolWidget (gtk.VBox):
    """Tool widget with the standard GTK color selector (triangle)."""

    tool_widget_title = _('Color')

    def __init__(self, app):
        gtk.VBox.__init__(self)
        self.app = app
        self.app.brush.observers.append(self.brush_modified_cb)
        self.hsvwidget = hsvwidget = gtk.HSV()
        hsvwidget.set_size_request(150, 150)
        hsvwidget.connect("size-allocate", self.on_hsvwidget_size_allocate)
        hsvwidget.connect("changed", self.on_hsvwidget_changed)
        self.pack_start(hsvwidget, True, True)
        self.col_picker = ColorPickerButton(self)
        self.next_col = ColorPreview()
        self.next_col.set_tooltip_text(_("Currently chosen color"))
        self.current_col = ColorPreview()
        self.current_col.set_tooltip_text(_("Last color painted with"))
        hbox = gtk.HBox()
        frame = gtk.Frame()
        frame.add(hbox)
        hbox.pack_start(self.current_col, True, True)
        hbox.pack_start(self.next_col, True, True)
        hbox.pack_start(self.col_picker, False, False)
        self.pack_start(frame, False, False)
        self.picking = False  # updated by col_picker
        self.in_brush_modified_cb = False
        self.set_color_hsv(self.app.brush.get_color_hsv())
        app.ch.color_pushed_observers.append(self.color_pushed_cb)

    def color_pushed_cb(self, pushed_color):
        rgb = self.app.ch.last_color
        hsv = helpers.rgb_to_hsv(*rgb)
        self.current_col.set_color_hsv(*hsv)

    def brush_modified_cb(self, settings):
        if not settings.intersection(('color_h', 'color_s', 'color_v')):
            return
        brush_color = self.app.brush.get_color_hsv()
        if brush_color != self.get_color_hsv():
            self.in_brush_modified_cb = True  # do we still need this?
            self.set_color_hsv(brush_color)
            self.in_brush_modified_cb = False

    def get_color_hsv(self):
        return self.hsvwidget.get_color()

    def set_color_hsv(self, hsv):
        h, s, v = hsv
        while h > 1.0: h -= 1.0
        while h < 0.0: h += 1.0
        if s > 1.0: s = 1.0
        if s < 0.0: s = 0.0
        if v > 1.0: v = 1.0
        if v < 0.0: v = 0.0
        self.hsvwidget.set_color(h, s, v)

    def on_hsvwidget_size_allocate(self, hsvwidget, alloc):
        radius = min(alloc.width, alloc.height) - 4
        hsvwidget.set_metrics(radius, max(12, int(radius/20)))
        hsvwidget.queue_draw()

    def on_hsvwidget_changed(self, hsvwidget):
        hsv = hsvwidget.get_color()
        self.next_col.set_color_hsv(*hsv)
        color_finalized = not (hsvwidget.is_adjusting() or self.picking)
        if color_finalized:
            if not self.in_brush_modified_cb:
                b = self.app.brush
                b.set_color_hsv(self.get_color_hsv())


class ColorPreview (gtk.DrawingArea):

    def __init__(self):
        gtk.DrawingArea.__init__(self)
        self.rgb = 0, 0, 0
        self.connect('expose-event', self.on_expose)
        self.set_size_request(20, 20)

    def set_color_hsv(self, h, s, v):
        self.rgb = helpers.hsv_to_rgb(h, s, v)
        self.queue_draw()

    def on_expose(self, widget, event):
        cr = self.window.cairo_create()
        cr.set_source_rgb(*self.rgb)
        cr.paint()


class ColorPickerButton (gtk.EventBox):
    """Button for picking a colour from the screen.
    """

    def __init__(self, selector):
        gtk.EventBox.__init__(self)
        self.set_tooltip_text(_("Pick Colour"))
        self.button = gtk.Button()
        self.image = gtk.Image()
        self.image.set_from_stock(gtk.STOCK_COLOR_PICKER, gtk.ICON_SIZE_BUTTON)
        self.add(self.button)
        self.button.add(self.image)
        # Clicking the button initiates a grab:
        self.button.connect("clicked", self.on_clicked)
        self.grabbed = False
        # While grabbed, the eventbox part listens for these:
        self.grab_mask = gdk.BUTTON_RELEASE_MASK | gdk.BUTTON1_MOTION_MASK
        self.connect("motion-notify-event", self.on_motion_notify_event)
        self.connect("button-release-event", self.on_button_release_event)
        # Newly picked colours are advertised to the associated selector:
        self.selector = selector

    def pick_color_at_pointer(self, widget, size=3):
        screen = widget.get_screen()
        colormap = screen.get_system_colormap()
        root = screen.get_root_window()
        screen_w, screen_h = screen.get_width(), screen.get_height()
        display = widget.get_display()
        screen_trash, x_root, y_root, mods = display.get_pointer()
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
        color_rgb = [ch/65535. for ch in color_total]
        return color_rgb

    def on_clicked(self, widget):
        cursor = self.selector.app.cursor_color_picker
        result = gdk.pointer_grab(self.window, False, self.grab_mask, None, cursor)
        if result == gdk.GRAB_SUCCESS:
            self.grabbed = True

    def on_motion_notify_event(self, widget, event):
        if not event.state & gdk.BUTTON1_MASK:
            return
        rgb = self.pick_color_at_pointer(widget)
        hsv = helpers.rgb_to_hsv(*rgb)
        self.selector.picking = True
        self.selector.set_color_hsv(hsv)

    def on_button_release_event(self, widget, event):
        if not self.grabbed:
            return False
        gdk.pointer_ungrab()
        rgb = self.pick_color_at_pointer(self)
        hsv = helpers.rgb_to_hsv(*rgb)
        self.selector.picking = False
        self.selector.set_color_hsv(hsv)


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

