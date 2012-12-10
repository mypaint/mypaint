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


class ColorPicker:

    outside_popup_timeout = 0


    def __init__(self, app, doc):
        self.app = app
        self.doc = doc
        self.idle_handler = None
        self.initial_modifiers = 0
        # On-demand popup window instance. Destroyed completely then the state
        # leaves. Freeing it fixes an otherwise intractable problem with stylus
        # grabs: https://gna.org/bugs/?20358
        self._win = None
        self._popup_callbacks = []


    def connect(self, event_name, cb):
        # Keep stategroup.State() happy.
        # All it tries to connect are some enter/leave notify callbacks. Not
        # helpful for this window.
        self._popup_callbacks.append((event_name, cb))


    @property
    def win(self):
        # Access, creating if needed, the private popup window instance.
        if self._win is not None:
            return self._win
        print "DEBUG: creating picker popup window"
        self._win = windowing.PopupWindow(self.app)
        self._win.set_can_focus(False)
        self.event_mask = gdk.BUTTON_PRESS_MASK | gdk.BUTTON_RELEASE_MASK
        self._win.add_events(self.event_mask)
        if pygtkcompat.USE_GTK3:
            self._win.connect("draw", self.draw_cb)
        else:
            self._win.connect("expose-event", self.expose_cb)
        self._win.connect("show", self.show_cb)
        self._win.connect("hide", self.hide_cb)
        self._win.connect("motion-notify-event", self.motion_notify_cb)
        self._win.set_size_request(popup_width, popup_height)
        for event_name, cb in self._popup_callbacks:
            self._win.connect(event_name, cb)
        return self._win


    def pick(self):
        # fixed size is prefered to brush radius: https://gna.org/bugs/?14794
        size = 6
        self.app.pick_color_at_pointer(self.win, size)


    def enter(self):
        self.pick()
        # popup placement
        x, y = self.win.get_position()
        self.win.move(x, y + popup_height)
        self.win.initial_modifiers = 0
        self.win.show_all()


    def leave(self, reason):
        self.win.hide()


    def show_cb(self, widget):
        assert self._win is not None

        # Grab the pointer
        # This is required for the color picker to pick continuously, and for
        # the color picker window to disappear after being launched with a
        # mouse button binding. https://gna.org/bugs/?19710
        result = gdk.pointer_grab(self._win.get_window(), False,
                event_mask=self.event_mask | gdk.POINTER_MOTION_MASK,
                cursor=self.app.cursor_color_picker)
        if result != gdk.GRAB_SUCCESS:
            print 'Warning: pointer grab failed with result', result
            gdk.pointer_ungrab()
            self.leave(reason=None)
            return

        # A GTK-level grab helps for some reason. Without it, the first time
        # the window is shown, it doesn't receive motion-notify events as
        # described in https://gna.org/bugs/?20358
        self._win.grab_add()

        self.popup_state.register_mouse_grab(self._win)

        # Record the modifiers which were held when the window was shown.
        display = gdk.display_get_default()
        screen, x, y, modifiers = display.get_pointer()
        modifiers &= gtk.accelerator_get_default_mod_mask()
        self.initial_modifiers = modifiers


    def hide_cb(self, window):
        assert self._win is not None

        # Being polite: undo open grabs.
        self._win.grab_remove()
        gdk.pointer_ungrab()

        # Disconnect update idler
        if self.idle_handler:
            gobject.source_remove(self.idle_handler)
            self.idle_handler = None

        # Being assertive: destroy the hidden window.
        # Seems to result in better results after picking with a stylus button:
        # see https://gna.org/bugs/?20358
        print "DEBUG: destroying picker popup window"
        self._win.destroy()
        self._win = None


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
        self.win.queue_draw()
        self.idle_handler = None
        return False


    def expose_cb(self, widget, event):
        # PyGTK draw handler.
        cr = widget.get_window().cairo_create()
        return self.draw_cb(widget, cr)


    def draw_cb(self, widget, cr):
        # GTK3+PyGI drawing.
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
