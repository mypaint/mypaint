# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk, gobject
gdk = gtk.gdk
import windowing
import cairo

popup_width = 80
popup_height = 80

class ColorPicker(windowing.PopupWindow):
    outside_popup_timeout = 0
    def __init__(self, app, doc):
        windowing.PopupWindow.__init__(self, app)
        # TODO: put the mouse position onto the selected color

        self.add_events(gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK
                        )
        self.connect("expose_event", self.expose_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)

        self.set_size_request(popup_width, popup_height)

        self.doc = doc

        self.idle_handler = None
        self.require_ctrl = False

    def pick(self):
        # fixed size is prefered to brush radius, see https://gna.org/bugs/?14794
        #size = int(self.app.brush.get_actual_radius() * math.sqrt(math.pi))
        #if size < 6: size = 6
        size = 6
        self.app.pick_color_at_pointer(self, size)

    def enter(self):
        self.pick()

        # popup placement
        x, y = self.get_position()
        self.move(x, y + popup_height)
        self.show_all()

        # Using a GTK grab rather than a gdk pointer grab seems to
        # fix https://gna.org/bugs/?17940
        self.grab_add()
        self.app.doc.tdw.set_override_cursor(self.app.cursor_color_picker)

        self.popup_state.register_mouse_grab(self)

        self.require_ctrl = False
    
    def leave(self, reason):
        self.grab_remove()
        self.app.doc.tdw.set_override_cursor(None)

        if self.idle_handler:
            gobject.source_remove(self.idle_handler)
            self.idle_handler = None
        self.hide()

    def motion_notify_cb(self, widget, event):
        if event.state & gdk.CONTROL_MASK or event.state & gdk.MOD1_MASK:
            self.require_ctrl = True
        elif self.require_ctrl:
            # stop picking when the user releases CTRL (or ALT)
            return

        def update():
            self.idle_handler = None
            self.pick()
            self.queue_draw()

        if not self.idle_handler:
            self.idle_handler = gobject.idle_add(update)

    def expose_cb(self, widget, event):
        cr = self.window.cairo_create()

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
