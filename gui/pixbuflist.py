# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
gdk = gtk.gdk
from lib import helpers
from math import ceil

DRAG_ITEM_NAME = 103

class PixbufList(gtk.DrawingArea):
    # interface to be implemented by children
    def on_select(self, item):
        pass
    def on_drag_data(self, copy, source_widget, brush_name, target_idx):
        return False
    def drag_begin_cb(self, widget, context):
        widget.drag_insertion_index = None
    def drag_end_cb(self, widget, context):
        widget.drag_insertion_index = None


    def __init__(self, itemlist, item_w, item_h, namefunc=None, pixbuffunc=lambda x: x):
        gtk.DrawingArea.__init__(self)
        self.itemlist = itemlist
        self.pixbuffunc = pixbuffunc
        self.namefunc = namefunc
        self.dragging_allowed = True

        self.pixbuf = None
        self.spacing_outside = 0
        self.border_visible = 2
        self.border_visible_outside_cell = 1
        self.spacing_inside = 0
        self.set_size(item_w, item_h)

        self.selected = None
        self.tooltip_text = None

        self.connect("expose-event", self.expose_cb)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("configure-event", self.configure_event_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)
        self.set_events(gdk.EXPOSURE_MASK |
                        gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.POINTER_MOTION_MASK)
        
        self.realized_once = False
        self.connect("realize", self.on_realize)
        
        self.drag_highlighted = False
        self.drag_insertion_index = None
        self.update()
    
    def on_realize(self, widget):
        if self.realized_once:
            return
        self.realized_once = True
        if self.dragging_allowed:
            # DnD setup.
            self.connect('drag-data-get', self.drag_data_get_cb)
            self.connect('drag-motion', self.drag_motion_cb)
            self.connect('drag-leave', self.drag_leave_cb)
            self.connect('drag-begin', self.drag_begin_cb)
            self.connect('drag-end', self.drag_end_cb)
            self.connect('drag-data-received', self.drag_data_received_cb)
            # Users can drag pixbufs *to* anywhere on a pixbuflist at all times.
            self.drag_dest_set(gtk.DEST_DEFAULT_ALL,
                    [('LIST_ITEM', gtk.TARGET_SAME_APP, DRAG_ITEM_NAME)],
                    gdk.ACTION_MOVE | gdk.ACTION_COPY)
            # Dragging *from* a list can only happen over a pixbuf: see motion_notify_cb
            self.drag_source_sensitive = False

    def set_size(self, item_w, item_h):
        self.item_w = item_w
        self.item_h = item_h
        self.thumbnails = {}

    def motion_notify_cb(self, widget, event):
        i = self.index(event.x, event.y) 
        over_item = i < len(self.itemlist)
        if over_item:
            if self.namefunc is not None:
                item = self.itemlist[i]
                item_name = self.namefunc(item)
                # Tooltip changing has to happen over two motion-notifys
                # because we want to force the tooltip box to move with the
                # mouse pointer.
                if self.tooltip_text != item_name:
                    self.tooltip_text = item_name
                    self.set_has_tooltip(False)
                    # pop down on the 1st event with this name
                else:
                    self.set_tooltip_text(item_name)
                    # pop up on the 2nd
            if self.dragging_allowed:
                if not self.drag_source_sensitive:
                    self.drag_source_set(gtk.gdk.BUTTON1_MASK,
                        [('LIST_ITEM', gtk.TARGET_SAME_APP, DRAG_ITEM_NAME)],
                        gdk.ACTION_COPY|gdk.ACTION_MOVE)
                    self.drag_source_sensitive = True
        else:
            if self.tooltip_text is not None:
                self.set_has_tooltip(False)
                self.tooltip_text = None
            if self.dragging_allowed:
                if self.drag_source_sensitive:
                    self.drag_source_unset()
                    self.drag_source_sensitive = False

    def drag_motion_cb(self, widget, context, x, y, time):
        if not self.dragging_allowed:
            return False
        action = None
        source_widget = context.get_source_widget()
        if self is source_widget:
            # Only moves are possible
            action = gdk.ACTION_MOVE
        else:
            # Dragging from another widget, default action is copy
            action = gdk.ACTION_COPY
            # However, if the item already exists here, it's a move
            sel = source_widget.selected
            if sel in self.itemlist:
                action = gdk.ACTION_MOVE
            else:
                # the user can force a move by pressing shift
                px, py, kbmods = self.get_window().get_pointer()
                if kbmods & gdk.SHIFT_MASK:
                    action = gdk.ACTION_MOVE
        context.drag_status(action, time)
        if not self.drag_highlighted:
            #self.drag_highlight()   # XXX nonfunctional
            self.drag_highlighted = True
            self.queue_draw()
        if self.drag_highlighted:
            i = self.index(x, y)
            if i != self.drag_insertion_index:
                self.queue_draw()
                self.drag_insertion_index = i

    def drag_leave_cb(self, widget, context, time):
        if widget.drag_highlighted:
            #widget.drag_unhighlight()   # XXX nonfunctional
            widget.drag_highlighted = False
            widget.drag_insertion_index = None
            widget.queue_draw()

    def drag_data_get_cb(self, widget, context, selection, targetType, time):
        item = self.selected
        assert item in self.itemlist
        assert targetType == DRAG_ITEM_NAME
        name = self.namefunc(item)
        selection.set(selection.target, 8, name)

    def drag_data_received_cb(self, widget, context, x,y, selection, targetType, time):
        item_name = selection.data
        target_item_idx = self.index(x, y) # idx always valid, we reject drops at invalid idx
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
        self.tiles_w = max(1, int( width / self.total_w ))
        self.tiles_h = max(1, int( ceil( float(len(self.itemlist)) / self.tiles_w ) ))

        height = self.tiles_h * self.total_h
        self.set_size_request(self.total_w, height)

        self.pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, True, 8, width, height)
        self.pixbuf.fill(0xffffff00) # transparent
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
        # cut to maximal size
        p_w, p_h = self.pixbuf.get_width(), self.pixbuf.get_height()

        self.window.draw_rectangle(widget.style.base_gc[gtk.STATE_NORMAL],
                                   True, 0, 0, p_w, p_h)

        if self.drag_highlighted:
            self.window.draw_rectangle(widget.style.black_gc, False, 0, 0, p_w-1, p_h-1)

        widget.window.draw_pixbuf(widget.style.black_gc,
                                  self.pixbuf,
                                  0, 0, 0, 0) 

        # draw borders
        i = 0
        last_i = len(self.itemlist) - 1
        for b in self.itemlist:
            rect_gc = None
            if b is self.selected:
                rect_gc = widget.style.bg_gc[gtk.STATE_SELECTED]
            elif  i == self.drag_insertion_index \
              or (i == last_i and self.drag_insertion_index > i):
                rect_gc = widget.style.fg_gc[gtk.STATE_NORMAL]
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
            for j in range(self.border_visible_outside_cell):
                x, y, w, h = shrink(-1, x, y, w, h)
            for j in range(self.border_visible + self.border_visible_outside_cell):
                if rect_gc:
                    widget.window.draw_rectangle(rect_gc, False, x, y, w-1, h-1)
                x, y, w, h = shrink(1, x, y, w, h)
            i += 1

        return True
