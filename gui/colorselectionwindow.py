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
import stock
from lib import mypaintlib
from lib.helpers import clamp,gdkpixbuf2numpy


DEBUG_HSV_WIDGET_NOT_WRAPPED = False


class ToolWidget (gtk.VBox):
    """Tool widget with the standard GTK color selector (triangle)."""

    stock_id = stock.TOOL_COLOR_SELECTOR

    def __init__(self, app):
        gtk.VBox.__init__(self)
        self.app = app
        self.app.brush.observers.append(self.brush_modified_cb)
        self.color_sel = gtk.ColorSelection()
        self.hsv_widget = None
        self.hsv_container = self.extract_hsv_container(self.color_sel)
        self.add_details_dialogs(self.hsv_container)
        self.color_sel.connect("color-changed", self.on_color_changed)
        self.pack_start(self.hsv_container, True, True)
        self.in_brush_modified_cb = False
        self.set_color_hsv(self.app.brush.get_color_hsv())
        app.ch.color_pushed_observers.append(self.color_pushed_cb)


    def color_pushed_cb(self, pushed_color):
        rgb = self.app.ch.last_color
        color = gdk.Color(*[int(c*65535) for c in rgb])
        self.color_sel.set_previous_color(color)


    def brush_modified_cb(self, settings):
        if not settings.intersection(('color_h', 'color_s', 'color_v')):
            return
        brush_color = self.app.brush.get_color_hsv()
        if brush_color != self.get_color_hsv():
            self.in_brush_modified_cb = True  # do we still need this?
            self.set_color_hsv(brush_color)
            self.in_brush_modified_cb = False


    def get_color_hsv(self):
        if self.hsv_widget is not None:
            # if we can, it's better to go to the HSV widget direct
            return self.hsv_widget.get_color()
        else:
            # not as good, loses hue information if saturation == 0
            color = self.color_sel.get_current_color()
            return color.hue, color.saturation, color.value


    def set_color_hsv(self, hsv):
        h, s, v = hsv
        while h > 1.0: h -= 1.0
        while h < 0.0: h += 1.0
        s = clamp(s, 0.0, 1.0)
        v = clamp(v, 0.0, 1.0)
        if self.hsv_widget is not None:
            self.hsv_widget.set_color(h, s, v)
        else:
            color = gdk.color_from_hsv(h, s, v)
            self.color_sel.set_current_color(color)


    def on_color_changed(self, color_sel):
        h, s, v = self.get_color_hsv()
        color_finalized = not color_sel.is_adjusting()
        if color_finalized and not self.in_brush_modified_cb:
            b = self.app.brush
            b.set_color_hsv((h, s, v))


    def on_dialog_ok_clicked(self, button, dialog):
        color = dialog.colorsel.get_current_color()
        self.color_sel.set_current_color(color)
        dialog.destroy()


    def on_dialog_cancel_clicked(self, button, dialog):
        dialog.destroy()


    def on_color_swatch_button_press(self, swatch, event):
        if event.type != gdk._2BUTTON_PRESS:
            return False
        dialog = gtk.ColorSelectionDialog(_("Color details"))
        dialog.set_position(gtk.WIN_POS_MOUSE)
        dialog.colorsel.set_current_color(self.color_sel.get_current_color())
        dialog.colorsel.set_previous_color(self.color_sel.get_previous_color())

        dialog.ok_button.connect("clicked", self.on_dialog_ok_clicked, dialog)
        dialog.cancel_button.connect("clicked", self.on_dialog_cancel_clicked,
                                     dialog)
        dialog.run()
        return True


    def add_details_dialogs(self, hsv_container):
        prev, current = self.find_widgets(hsv_container,
            lambda w: w.get_name() == 'GtkDrawingArea')
        current.connect("button-press-event", self.on_color_swatch_button_press)
        prev.connect("button-press-event", self.on_color_swatch_button_press)


    def on_hsvwidget_size_allocate(self, hsvwidget, alloc):
        # We can only control the GtkHSV's radius if PyGTK exposes it to us
        # as the undocumented gtk.HSV.
        if not hasattr(hsvwidget, "set_metrics"):
            return
        padding = 5
        radius = min(alloc.width, alloc.height) - (2 * padding)
        ring_width = max(12, int(radius/16))
        hsvwidget.set_metrics(radius, ring_width)
        hsvwidget.queue_draw()


    def extract_hsv_container(self, cs):
        """Extract the HSV wheel and nearby buttons from a ColorSelector

        This is ugly, but it's useful to support pre-2.18 versions of (py)gtk.
        """
        hsv, = self.find_widgets(cs, lambda w: w.get_name() == 'GtkHSV')
        hsv.unset_flags(gtk.CAN_FOCUS)
        hsv.unset_flags(gtk.CAN_DEFAULT)
        hsv.set_size_request(150, 150)
        container = hsv.parent
        container.parent.remove(container)
        # Make the packing box give extra space to the HSV widget
        container.set_child_packing(hsv, True, True, 0, gtk.PACK_START)
        container.set_spacing(0)
        # When extra space is given, grow the HSV wheel.
        # We can only control the GtkHSV's radius if PyGTK exposes it to us
        # as the undocumented gtk.HSV.
        if hasattr(hsv, "set_metrics"):
            hsv.connect("size-allocate", self.on_hsvwidget_size_allocate)
            if DEBUG_HSV_WIDGET_NOT_WRAPPED:
                print "DEBUG: emulating old pygtk where gtk.HSV is not wrapped"
            else:
                self.hsv_widget = hsv
        return container


    def find_widgets(self, widget, predicate):
        queue = [widget]
        found = []
        while len(queue) > 0:
            w = queue.pop(0)
            if predicate(w):
                found.append(w)
            if hasattr(w, "get_children"):
                for w2 in w.get_children():
                    queue.append(w2)
        return found



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
        arr = gdkpixbuf2numpy(pixbuf)
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

