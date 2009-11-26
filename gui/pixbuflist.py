# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
gdk = gtk.gdk
from math import sqrt
from lib import helpers

DRAG_ITEM_NAME = 103

MOTION_NONE = 1
MOTION_BUTTON_PRESSED = 2

def rho(x1,y1, x2,y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)

class PixbufList(gtk.DrawingArea):
    # interface to be implemented by children
    def on_select(self, item):
        pass
    def get_tooltip(self, item):
        return self.namefunc(item)

    def __init__(self, itemlist, item_w, item_h, namefunc=None, pixbuffunc=lambda x: x):
        gtk.DrawingArea.__init__(self)
        #if not disable_dragging:
        if True:
            self.drag_dest_set(gtk.DEST_DEFAULT_ALL,
                    [('LIST_ITEM', gtk.TARGET_SAME_APP, DRAG_ITEM_NAME)],
                    gdk.ACTION_MOVE | gdk.ACTION_COPY)
            self.connect('drag-data-get', self.drag_data_get)
            self.connect('motion-notify-event', self.motion_notify)
            self.connect('button-press-event', self.on_button_press)
            self.connect('button-release-event', self.on_button_release)
        self.itemlist = itemlist
        self.motion_mode = MOTION_NONE
        self.press_x = 0
        self.press_y = 0
        self.pixbuffunc = pixbuffunc
        self.namefunc = namefunc
        self.pixbuf = None

        self.spacing_outside = 0
        self.border_visible = 2
        self.spacing_inside = 0
        self.set_size(item_w, item_h)

        self.selected = None

        self.connect("expose-event", self.expose_cb)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("configure-event", self.configure_event_cb)
        self.set_events(gdk.EXPOSURE_MASK |
                        gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.POINTER_MOTION_MASK)
        self.update()

    def set_size(self, item_w, item_h):
        self.item_w = item_w
        self.item_h = item_h
        self.thumbnails = {}

    def on_button_press(self, widget, event):
        self.control_pressed = bool( event.state & gdk.CONTROL_MASK )
        if event.button == 1:
            self.press_x = event.x
            self.press_y = event.y
            self.motion_mode = MOTION_BUTTON_PRESSED

    def motion_notify(self, widget, event):
        if self.motion_mode != MOTION_BUTTON_PRESSED:
            return
        if rho(event.x, event.y, self.press_x, self.press_y) > max(self.item_w, self.item_h):
            if self.control_pressed:
                action = gdk.ACTION_COPY
            else:
                action = gdk.ACTION_MOVE
            self.drag_begin([('LIST_ITEM', gtk.TARGET_SAME_APP, DRAG_ITEM_NAME)],
                            action, 1, event)

    def on_button_release(self, widget, event):
        self.motion_mode = MOTION_NONE

    def drag_data_get(self, widget, context, selection, targetType, time):
        item = self.selected
        assert item in self.itemlist
        assert targetType == DRAG_ITEM_NAME
        name = self.namefunc(item)
        selection.set(selection.target, 8, name)

    def drag_data_received(self, widget, context, x,y, selection, targetType, time):
        item_name = selection.data
        target_item_idx = self.index(x, y)
        if target_item_idx > len(self.itemlist):
            return
        w = context.get_source_widget()
        copy = context.action==gdk.ACTION_COPY
        success = self.on_drag_data(copy, w, item_name, target_item_idx)
        context.finish(success, False, time)

    def update(self, width = None, height = None):
        """
        Redraws the widget from scratch.
        """
        self.total_border = self.border_visible + self.spacing_inside + self.spacing_outside
        self.total_w = self.item_w + 2*self.total_border
        self.total_h = self.item_h + 2*self.total_border

        if width is None:
            if not self.pixbuf: return
            width = self.pixbuf.get_width()
            height = self.pixbuf.get_height()
        width = max(width, self.total_w)
        self.tiles_w = width / self.total_w
        self.tiles_h = len(self.itemlist)/self.tiles_w + 1
        height = self.tiles_h * self.total_h
        self.set_size_request(self.total_w, height)
        self.pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, width, height)
        self.pixbuf.fill(0xffffffff) # white
        for i, item in enumerate(self.itemlist):
            x = (i % self.tiles_w) * self.total_w
            y = (i / self.tiles_w) * self.total_h
            x += self.total_border
            y += self.total_border

            pixbuf = self.pixbuffunc(item)
            if pixbuf not in self.thumbnails:
                self.thumbnails[pixbuf] = helpers.pixbuf_thumbnail(pixbuf, self.item_w, self.item_h)
            pixbuf = self.thumbnails[pixbuf]
            pixbuf.copy_area(0, 0, self.item_w, self.item_h, self.pixbuf, x, y)
        self.queue_draw()

    def set_selected(self, item):
        self.selected = item
        self.queue_draw()

    def index(self, x,y):
        x, y = int(x), int(y)
        i = x / self.total_w
        if i >= self.tiles_w: i = self.tiles_w - 1
        if i < 0: i = 0
        i = i + self.tiles_w * (y / self.total_h)
        if i < 0: i = 0
        return i

    def button_press_cb(self, widget, event):
        self.control_pressed = bool( event.state & gdk.CONTROL_MASK )
        if event.button == 1:
            self.press_x = event.x
            self.press_y = event.y
            self.motion_mode = MOTION_BUTTON_PRESSED
        i = self.index(event.x, event.y)
        if i >= len(self.itemlist): return
        item = self.itemlist[i]
        self.set_selected(item)
        self.on_select(item)

    def configure_event_cb(self, widget, size):
        if self.pixbuf and self.pixbuf.get_width() == size.width:
            if self.pixbuf.get_height() == size.height:
                return
        self.update(size.width, size.height)

    def expose_cb(self, widget, event):
        rowstride = self.pixbuf.get_rowstride()
        pixels = self.pixbuf.get_pixels()
        
        # cut to maximal size
        p_w, p_h = self.pixbuf.get_width(), self.pixbuf.get_height()

        widget.window.draw_rgb_image(
            widget.style.black_gc,
            0, 0, p_w, p_h,
            'normal',
            pixels, rowstride)

        # draw borders
        i = 0
        for b in self.itemlist:
            if b is self.selected:
                gc = widget.style.black_gc
            else:
                gc = widget.style.white_gc
            x = (i % self.tiles_w) * self.total_w
            y = (i / self.tiles_w) * self.total_h
            w = self.total_w
            h = self.total_h
            def shrink(pixels, x, y, w, h):
                x += pixels
                y += pixels
                w -= 2*pixels
                h -= 2*pixels
                return (x, y, w, h)
            x, y, w, h = shrink(self.spacing_outside, x, y, w, h)
            for j in range(self.border_visible):
                widget.window.draw_rectangle(gc, False, x, y, w-1, h-1)
                x, y, w, h = shrink(1, x, y, w, h)
            i += 1

        return True
