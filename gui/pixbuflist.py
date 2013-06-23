# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from warnings import warn
from math import ceil

import gtk2compat
import gobject
import gtk
from gtk import gdk

from lib import helpers
from colors import RGBColor


DRAG_ITEM_NAME = 103
ITEM_SIZE_DEFAULT = 48


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

    # GType naming, for GtkBuilder
    __gtype_name__ = 'PixbufList'

    def __init__(self, itemlist=None,
                 item_w=ITEM_SIZE_DEFAULT,
                 item_h=ITEM_SIZE_DEFAULT,
                 namefunc=None, pixbuffunc=lambda x: x):
        gtk.DrawingArea.__init__(self)

        if itemlist is not None:
            self.itemlist = itemlist
        else:
            self.itemlist = []
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
        self.in_potential_drag = False

        self.connect("draw", self.draw_cb)

        self.connect("button-press-event", self.button_press_cb)
        self.connect("button-release-event", self.button_release_cb)
        self.connect("configure-event", self.configure_event_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)
        self.set_events(gdk.EXPOSURE_MASK |
                        gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.POINTER_MOTION_MASK |
                        # Allow switching between mouse and pen inside the widget
                        gdk.PROXIMITY_OUT_MASK |
                        gdk.PROXIMITY_IN_MASK)

        self.get_settings().set_property("gtk-dnd-drag-threshold",
            int(min(item_w, item_h) * 0.75))

        self.realized_once = False
        self.connect("realize", self.on_realize)

        self.drag_highlighted = False
        self.drag_insertion_index = None

        # Fake a nicer style
        style_context = self.get_style_context()
        style_context.add_class(gtk.STYLE_CLASS_VIEW)

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
            targets_list = [('LIST_ITEM', gtk.TARGET_SAME_APP, DRAG_ITEM_NAME)]
            targets_list = [gtk.TargetEntry.new(*e) for e in targets_list]
            self.drag_dest_set(gtk.DEST_DEFAULT_ALL, targets_list,
                               gdk.ACTION_MOVE | gdk.ACTION_COPY)
            # Dragging *from* a list can only happen over a pixbuf: see motion_notify_cb
            self.drag_source_sensitive = False

    def set_itemlist(self, items):
        self.itemlist = items
        self.update()

    def set_size(self, item_w, item_h):
        self.item_w = item_w
        self.item_h = item_h
        self.thumbnails = {}

    def motion_notify_cb(self, widget, event):
        over_item = False
        if self.point_is_inside(event.x, event.y):
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
                    targets_list = [ ('LIST_ITEM', gtk.TARGET_SAME_APP,
                                      DRAG_ITEM_NAME) ]
                    targets_list = [ gtk.TargetEntry.new(*e)
                                     for e in targets_list  ]
                    self.drag_source_set(gtk.gdk.BUTTON1_MASK, targets_list,
                                         gdk.ACTION_COPY|gdk.ACTION_MOVE)
                    self.drag_source_sensitive = True
        else:
            if self.tooltip_text is not None:
                self.set_has_tooltip(False)
                self.tooltip_text = None
            if self.dragging_allowed and self.drag_source_sensitive:
                if not self.in_potential_drag:
                    # If we haven't crossed the drag threshold yet, don't kill
                    # the potential drag before it starts.
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
        #self.set_size_request(-1, -1)
        gobject.idle_add(self.set_size_request, self.total_w, height)

        self.pixbuf = gtk2compat.gdk.pixbuf.new(gdk.COLORSPACE_RGB, True,
                                                 8, width, height)
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
            pixbuf.composite(self.pixbuf, x, y, self.item_w, self.item_h, x, y, 1, 1, gdk.INTERP_BILINEAR, 255)

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

    def point_is_inside(self, x, y):
        alloc = self.get_allocation()
        w = alloc.width
        h = alloc.height
        return x >= 0 and y >= 0 and x < w and y < h

    def button_press_cb(self, widget, event):
        ex, ey = int(event.x), int(event.y)
        if not self.point_is_inside(ex, ey):
            return False
        i = self.index(ex, ey)
        if i >= len(self.itemlist): return
        item = self.itemlist[i]
        self.set_selected(item)
        self.on_select(item)
        if self.selected is not None:
            # early exception if drag&drop would break
            assert self.selected in self.itemlist, 'selection failed: the user selected %r by pointing at it, but after calling on_select() %r is active instead!' % (item, self.selected)
        self.in_potential_drag = True

    def button_release_cb(self, widget, event):
        self.in_potential_drag = False

    def configure_event_cb(self, widget, size):
        if self.pixbuf and self.pixbuf.get_width() == size.width:
            if self.pixbuf.get_height() == size.height:
                return
        self.update(size.width, size.height)


    def draw_cb(self, widget, cr):
        # Paint the base colour, and the list's pixbuf.
        state_flags = widget.get_state_flags()
        style_context = widget.get_style_context()
        bg_color_gdk = style_context.get_background_color(state_flags)
        bg_color = RGBColor.new_from_gdk_rgba(bg_color_gdk)
        cr.set_source_rgb(*bg_color.get_rgb())
        cr.paint()
        gdk.cairo_set_source_pixbuf(cr, self.pixbuf, 0, 0)
        cr.paint()
        # border colors
        gdkrgba = style_context.get_background_color(
                    state_flags|gtk.StateFlags.SELECTED)
        selected_color = RGBColor.new_from_gdk_rgba(gdkrgba)
        gdkrgba = style_context.get_background_color(
                    state_flags|gtk.StateFlags.NORMAL)
        insertion_color = RGBColor.new_from_gdk_rgba(gdkrgba)
        # Draw borders
        last_i = len(self.itemlist) - 1
        for i, b in enumerate(self.itemlist):
            rect_color = None
            if b is self.selected:
                rect_color = selected_color
            elif i == self.drag_insertion_index \
                    or (i == last_i and self.drag_insertion_index > i):
                rect_color = insertion_color
            if rect_color is None:
                continue
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
            for j in xrange(self.border_visible + self.border_visible_outside_cell):
                cr.set_source_rgb(*rect_color.get_rgb())
                cr.rectangle(x, y, w-1, h-1)   # FIXME: check pixel alignment
                cr.stroke()
                x, y, w, h = shrink(1, x, y, w, h)
        return True


if __name__ == '__main__':
    win = gtk.Window()
    win.set_title("pixbuflist test")
    test_list = PixbufList()
    win.add(test_list)
    test_list.set_size_request(256, 128)
    win.connect("destroy", lambda *a: gtk.main_quit())
    win.show_all()
    gtk.main()
