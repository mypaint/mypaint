# This file is part of MyPaint.
# Copyright (C) 2009-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2010-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function

from lib.gibindings import Gtk
from lib.gibindings import Gdk
import cairo

from . import windowing


"""Color history popup."""


## Module constants

popup_height = 60
bigcolor_width = popup_height
smallcolor_width = popup_height // 2


## Class definitions

class HistoryPopup (windowing.PopupWindow):
    """History popup window.

    This window is normally bound to the "X" key, and has the following
    behavior:

    1. x, x, x, draw: cycles 3 colours back;
    2. x, draw, x, draw, x, draw, ...: flips between two most recent
       colours at the time this sequence started.

    It's managed from a StateGroup: see gui.stategroup. See also
    https://github.com/mypaint/mypaint/issues/93 for discussion.

    """

    outside_popup_timeout = 0

    def __init__(self, app, model):
        super(HistoryPopup, self).__init__(app)
        # TODO: put the mouse position onto the selected color
        # FIXME: This duplicates stuff from the PopupWindow
        self.set_position(Gtk.WindowPosition.MOUSE)
        self.app = app
        self.app.kbm.add_window(self)

        hist_len = len(self.app.brush_color_manager.get_history())
        self.popup_width = bigcolor_width + (hist_len-1)*smallcolor_width

        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.BUTTON_RELEASE_MASK |
                        Gdk.EventMask.ENTER_NOTIFY_MASK |
                        Gdk.EventMask.LEAVE_NOTIFY_MASK
                        )
        self.connect("button-release-event", self.button_release_cb)
        self.connect("button-press-event", self.button_press_cb)

        self.connect("draw", self.draw_cb)

        self.set_size_request(self.popup_width, popup_height)

        # Selection index. Each enter() (i.e. keypress) advances this.
        self.selection = None

        # Reset the selection when something is drawn,
        # or when the colour history changes.
        mgr = self.app.brush_color_manager
        mgr.color_history_updated += self._reset_selection_cb
        model.canvas_area_modified += self._reset_selection_cb

    def enter(self):
        mgr = self.app.brush_color_manager
        hist = mgr.get_history()
        if self.selection is None:
            self.selection = len(hist) - 1
            color = mgr.get_color()
            if hist[self.selection] == color:
                self.selection -= 1
        else:
            self.selection = (self.selection - 1) % len(hist)

        mgr.set_color(hist[self.selection])

        # popup placement
        x, y = self.get_position()
        bigcolor_center_x = (self.selection * smallcolor_width +
                             bigcolor_width // 2)
        self.move(x + self.popup_width // 2 - bigcolor_center_x,
                  y + bigcolor_width)
        self.show_all()

        window = self.get_window()
        cursor = Gdk.Cursor.new_for_display(
            window.get_display(), Gdk.CursorType.CROSSHAIR)
        window.set_cursor(cursor)

    def leave(self, reason):
        self.hide()

    def button_press_cb(self, widget, event):
        pass

    def button_release_cb(self, widget, event):
        pass

    def _reset_selection_cb(self, *_args, **_kwargs):
        self.selection = None

    def draw_cb(self, widget, cr):
        cr.set_source_rgb(0.9, 0.9, 0.9)
        cr.paint()

        cr.set_line_join(cairo.LINE_JOIN_ROUND)

        cr.translate(0.0, popup_height/2.0)

        hist = self.app.brush_color_manager.get_history()
        for i, c in enumerate(hist):
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
            cr.set_source_rgb(*c.get_rgb())
            cr.fill_preserve()
            cr.set_line_width(line_width)
            cr.set_source_rgb(0, 0, 0)
            cr.stroke()
            cr.translate(popup_height, 0)

            if i != self.selection:
                cr.scale(2.0, 2.0)

        return True
