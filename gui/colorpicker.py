# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import pygtkcompat
import gtk, gobject
gdk = gtk.gdk
import windowing
import cairo

popup_width = 80
popup_height = 80


class ColorPicker (windowing.PopupWindow):

    outside_popup_timeout = 0

    def __init__(self, app, doc):
        windowing.PopupWindow.__init__(self, app)
        self.set_can_focus(False)
        # TODO: put the mouse position onto the selected color

        self.event_mask = gdk.BUTTON_PRESS_MASK | gdk.BUTTON_RELEASE_MASK
        self.add_events(self.event_mask)

        if pygtkcompat.USE_GTK3:
            self.connect("draw", self.draw_cb)
        else:
            self.connect("expose-event", self.expose_cb)
        self.connect("show", self.show_cb)
        self.connect("hide", self.hide_cb)

        self.connect("motion-notify-event", self.motion_notify_cb)

        self.set_size_request(popup_width, popup_height)

        self.doc = doc

        self.idle_handler = None
        self.initial_modifiers = 0


    def pick(self):
        # fixed size is prefered to brush radius: https://gna.org/bugs/?14794
        size = 6
        self.app.pick_color_at_pointer(self, size)


    def enter(self):
        self.pick()
        # popup placement
        x, y = self.get_position()
        self.move(x, y + popup_height)
        self.initial_modifiers = 0
        self.show_all()


    def show_cb(self, widget):

        # Grab the pointer
        # This is required for the color picker to pick continuously, and for
        # the color picker window to disappear after being launched with a
        # mouse button binding. https://gna.org/bugs/?19710
        result = gdk.pointer_grab(self.get_window(), False,
                event_mask=self.event_mask | gdk.POINTER_MOTION_MASK,
                cursor=self.app.cursor_color_picker)
        if result != gdk.GRAB_SUCCESS:
            print 'Warning: pointer grab failed with result', result
            self.leave(reason=None)
            return

        # A GTK-level grab helps for some reason. Without it, the first time
        # the window is shown, it doesn't receive motion-notify events as
        # described in https://gna.org/bugs/?20358
        self.grab_add()

        self.popup_state.register_mouse_grab(self)

        # Record the modifiers which were held when the window was shown.
        display = gdk.display_get_default()
        screen, x, y, modifiers = display.get_pointer()
        modifiers &= gtk.accelerator_get_default_mod_mask()
        self.initial_modifiers = modifiers


    def leave(self, reason):
        self.hide()


    def hide_cb(self, window):
        # Undo open grabs.
        self.grab_remove()
        gdk.pointer_ungrab()

        # Disconnect update idler
        if self.idle_handler:
            gobject.source_remove(self.idle_handler)
            self.idle_handler = None


    def motion_notify_cb(self, widget, event):
        if self.initial_modifiers:
            modifiers = event.state & gtk.accelerator_get_default_mod_mask()
            if modifiers & self.initial_modifiers == 0:
                # stop picking when the user releases CTRL (or ALT)
                self.leave("initial modifiers no longer held")
                self.initial_modifiers = 0
                return
        if not self.idle_handler:
            self.idle_handler = gobject.idle_add(self.motion_update_idler)


    def motion_update_idler(self):
        self.pick()
        self.queue_draw()
        self.idle_handler = None
        return False


    def expose_cb(self, widget, event):
        cr = self.window.cairo_create()
        return self.draw_cb(widget, cr)


    def draw_cb(self, widget, cr):
        #cr.set_source_rgb (1.0, 1.0, 1.0)
        cr.set_source_rgba (1.0, 1.0, 1.0, 0.0) # transparent
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        color = self.app.brush.get_color_rgb()

        line_width = 3.0
        distance = 2*line_width
        rect = [0, 0, popup_height, popup_width]
        rect[0] += distance/2.0
        rect[1] += distance/2.0
        rect[2] -= distance
        rect[3] -= distance
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.rectangle(*rect)
        cr.set_source_rgb(*color)
        cr.fill_preserve()
        cr.set_line_width(line_width)
        cr.set_source_rgb(0, 0, 0)
        cr.stroke()

        return True
