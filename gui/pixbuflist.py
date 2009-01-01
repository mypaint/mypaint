# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
gdk = gtk.gdk

class PixbufList(gtk.DrawingArea):
    """
    This widget presents a list of items to the user. Each item must
    have a pixbuf. The user can select a single item. Items can be
    dragged around, causing the list to be reordered immediately.

    update() must be called when the list or any pixbufs have changed
    """

    # interface to be implemented by children
    def on_order_change(self):
        pass
    def on_select(self, item):
        pass

    def __init__(self, itemlist, item_w, item_h, pixbuffunc=lambda x: x):
        gtk.DrawingArea.__init__(self)
        self.itemlist = itemlist
        self.pixbuffunc = pixbuffunc
        self.pixbuf = None

        self.item_w = item_w
        self.item_h = item_h
        self.spacing_outside = 0
        self.border_visible = 1
        self.spacing_inside = 1

        self.selected = None
        self.dragging_allowed = True
        self.grabbed = None
        self.dragging = False

        self.connect("expose-event", self.expose_cb)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("button-release-event", self.button_release_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)
        self.connect("configure-event", self.configure_event_cb)
        self.set_events(gdk.EXPOSURE_MASK |
                        gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.POINTER_MOTION_MASK)
        self.update()

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
        self.tiles_w = (width / self.total_w) or 1
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
            pixbuf.copy_area(0, 0, self.item_w, self.item_h, self.pixbuf, x, y)
        self.queue_draw()

    def set_selected(self, item):
        self.selected = item
        self.queue_draw()

    def index(self, event):
        x, y = int(event.x), int(event.y)
        i = x / self.total_w
        if i >= self.tiles_w: i = self.tiles_w - 1
        if i < 0: i = 0
        i = i + self.tiles_w * (y / self.total_h)
        if i < 0: i = 0
        return i

    def button_press_cb(self, widget, event):
        i = self.index(event)
        if i >= len(self.itemlist): return
        item = self.itemlist[i]
        self.set_selected(item)
        self.on_select(item)
        if self.dragging_allowed:
            self.grabbed = item

    def button_release_cb(self, widget, event):
        self.grabbed = None
        if self.dragging:
            self.on_order_change()
            self.dragging = False

    def motion_notify_cb(self, widget, event):
        if not self.grabbed: return
        i = self.index(event)
        if i >= len(self.itemlist): return
        if self.itemlist[i] is not self.grabbed:
            self.itemlist.remove(self.grabbed)
            self.itemlist.insert(i, self.grabbed)
            self.dragging = True
            self.update()

    #def size_request_cb(self, widget, size):
    def configure_event_cb(self, widget, size):
        if self.pixbuf and self.pixbuf.get_width() == size.width:
            if self.pixbuf.get_height() == size.height:
                return
        self.update(size.width, size.height)

    def expose_cb(self, widget, event):
        rowstride = self.pixbuf.get_rowstride()
        pixels = self.pixbuf.get_pixels()
        
        # cut to maximal size
        e_x, e_y = event.area.x, event.area.y
        e_w, e_h = event.area.width, event.area.height
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
