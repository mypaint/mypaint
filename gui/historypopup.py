# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
gdk = gtk.gdk

import random
import numpy, cairo
from lib import helpers
import windowing

"""
Worklist (planning/prototyping)
- short keypress switches between two colors (==> still possible to do black/white the old way)
- long keypress allows to move the mouse over another color
  - the mouseover color is selected when the key is released
- (unsure) pressing the key twice (without painting in-between) cycles through the colors
  - the problem is that you don't see the colors
  ==> different concept:
    - short keypress opens the color ring with cursor centered on hole (ring stays open)
    - the ring disappears as soon as you touch it (you can just continue painting)
    - pressing the key again cycles the ring
- recent colors should be saved with painting

Observation:
- it seems quite unnatural that you can have a /shorter/ popup duration by pressing the key /longer/
  ==> you rather want a /minimum/ duration
"""

popup_height = 60
bigcolor_width   = popup_height
smallcolor_width = popup_height/2

class HistoryPopup(windowing.PopupWindow):
    outside_popup_timeout = 0
    def __init__(self, app, doc):
        windowing.PopupWindow.__init__(self, app)
        # TODO: put the mouse position onto the selected color
        self.set_position(gtk.WIN_POS_MOUSE)

        self.app = app
        self.app.kbm.add_window(self)

        self.popup_width = bigcolor_width + (self.app.ch.num_colors-1)*smallcolor_width

        self.set_events(gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.ENTER_NOTIFY |
                        gdk.LEAVE_NOTIFY
                        )
        self.connect("button-release-event", self.button_release_cb)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("expose_event", self.expose_cb)

        self.set_size_request(self.popup_width, popup_height)

        self.selection = None

        self.doc = doc
        guidoc = app.doc
        guidoc.input_stroke_ended_observers.append(self.input_stroke_ended_cb)
        self.is_shown = False

    def enter(self):
        # finish pending stroke, if any (causes stroke_finished_cb to get called)
        self.doc.split_stroke()
        ch = self.app.ch
        if self.selection is None:
            self.selection = ch.num_colors - 1
            color = self.app.brush.get_color_hsv()
            if ch.hsv_equal(ch.colors[self.selection], color):
                self.selection -= 1
        else:
            self.selection = (self.selection - 1) % ch.num_colors

        ch.atomic = True
        self.app.brush.set_color_hsv(ch.colors[self.selection])
        ch.atomic = False

        # popup placement
        x, y = self.get_position()
        bigcolor_center_x = self.selection * smallcolor_width + bigcolor_width/2
        self.move(x + self.popup_width/2 - bigcolor_center_x, y + bigcolor_width)
        self.show_all()
        self.is_shown = True

        self.window.set_cursor(gdk.Cursor(gdk.CROSSHAIR))
    
    def leave(self, reason):
        self.hide()
        self.is_shown = False

    def button_press_cb(self, widget, event):
        pass

    def button_release_cb(self, widget, event):
        pass

    def input_stroke_ended_cb(self, event):
        if self.is_shown: return
        brush = self.app.brush
        self.selection = None
        if not brush.is_eraser():
            color = brush.get_color_hsv()
            self.app.ch.push_color(color)

    def expose_cb(self, widget, event):
        cr = self.window.cairo_create()

        cr.set_source_rgb(0.9, 0.9, 0.9)
        cr.paint()

        cr.set_line_join(cairo.LINE_JOIN_ROUND)

        cr.translate(0.0, popup_height/2.0)

        for i, c in enumerate(self.app.ch.colors):
            if i != self.selection:
                cr.scale(0.5, 0.5)

            line_width = 3.0
            distance = 2*line_width
            rect = [0, -popup_height/2.0, popup_height, popup_height]
            rect[0] += distance/2.0
            rect[1] += distance/2.0
            rect[2] -= distance
            rect[3] -= distance
            cr.rectangle(*rect)
            cr.set_source_rgb(*helpers.hsv_to_rgb(*c))
            cr.fill_preserve()
            cr.set_line_width(line_width)
            cr.set_source_rgb(0, 0, 0)
            cr.stroke()
            cr.translate(popup_height, 0)

            if i != self.selection:
                cr.scale(2.0, 2.0)

        return True
