# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

"extending the C myDrawWidget a bit, eg with infinite canvas"
import gtk, gc
from mydrawwidget import MyDrawWidget
from helpers import Rect

class InfiniteMyDrawWidget(MyDrawWidget):
    def __init__(self):
        MyDrawWidget.__init__(self)
        self.init_canvas()
        MyDrawWidget.clear(self)
        self.connect("size-allocate", self.size_allocate_event_cb)
        self.connect("dragging-finished", self.dragging_finished_cb)
        self.connect("proximity-in-event", self.proximity_cb)
        self.connect("proximity-out-event", self.proximity_cb)
        self.toolchange_observers = []

    def init_canvas(self):
        self.canvas_w = 1
        self.canvas_h = 1
        self.viewport_x = 0.0
        self.viewport_y = 0.0
        self.original_canvas_x0 = 0
        self.original_canvas_y0 = 0

    def clear_internal(self):
        self.discard_and_resize(1, 1)
        self.init_canvas()
        MyDrawWidget.clear(self)

    def clear(self):
        self.clear_internal()
        if self.window: 
            self.resize_if_needed()

    def allow_dragging(self, allow=True):
        if allow:
            MyDrawWidget.allow_dragging(self, 1)
        else:
            MyDrawWidget.allow_dragging(self, 0)

    def save(self, filename):
        pixbuf = self.get_nonwhite_as_pixbuf()
        pixbuf.save(filename, 'png')

    def load(self, filename_or_pixbuf):
        if isinstance(filename_or_pixbuf, str):
            pixbuf = gtk.gdk.pixbuf_new_from_file(filename)
        else:
            pixbuf = filename_or_pixbuf

        self.clear_internal()

        if pixbuf.get_has_alpha():
            print 'Loaded file has an alpha channel. Rendering it on white instead.'
            print 'NOT IMPLEMENTED'
            class FormatError(Exception):
                pass
            raise FormatError, 'PNG files with alpha channel are not supported!'
            #TODO
            #w, h = pixbuf.get_width(), pixbuf.get_height()
            #new_pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, w, h)
            #new_pixbuf.fill(0xffffffff) # white
            #pixbuf = new_pixbuf
        self.resize_if_needed(old_pixbuf = pixbuf)

    def save_snapshot(self):
        return self.get_as_pixbuf(), self.original_canvas_x0, self.original_canvas_y0

    def load_snapshot(self, data):
        pixbuf, original_canvas_x0, original_canvas_y0 = data

        x = self.original_canvas_x0 - original_canvas_x0
        y = self.original_canvas_y0 - original_canvas_y0
        self.resize_if_needed(old_pixbuf=pixbuf, old_pixbuf_pos=(x, y))

    def proximity_cb(self, widget, something):
        for f in self.toolchange_observers:
            f()

    def dragging_finished_cb(self, widget):
        self.viewport_x = self.get_viewport_x()
        self.viewport_y = self.get_viewport_y()
        self.resize_if_needed()
        
    def size_allocate_event_cb(self, widget, allocation):
        size = (allocation.width, allocation.height)
        self.resize_if_needed(size=size)

    def zoom(self, new_zoom):
        vp_w, vp_h = self.window.get_size()
        old_zoom = self.get_zoom()

        center_x = self.viewport_x + vp_w / old_zoom / 2.0
        center_y = self.viewport_y + vp_h / old_zoom / 2.0

        self.set_zoom(new_zoom)

        self.viewport_x = center_x - vp_w / new_zoom / 2.0
        self.viewport_y = center_y - vp_h / new_zoom / 2.0

        self.set_viewport(self.viewport_x, self.viewport_y)
        self.resize_if_needed()


    def set_viewport(self, x, y):
        MyDrawWidget.set_viewport(self, x, y)
        self.viewport_x = x
        self.viewport_y = y

    # orig = coordinates relative to the original (0, 0), in contrast
    #        to the pixbuf coordinates (which shift at every resize)
    def get_viewport_orig(self):
        return (self.viewport_x - self.original_canvas_x0, self.viewport_y - self.original_canvas_y0) 
    def set_viewport_orig(self, x, y):
        self.set_viewport(x + self.original_canvas_x0, y + self.original_canvas_y0)

    def scroll(self, dx, dy):
        zoom = self.get_zoom()
        self.set_viewport(self.viewport_x + dx/zoom, self.viewport_y + dy/zoom)
        self.resize_if_needed()

    def resize_if_needed(self, old_pixbuf = None, old_pixbuf_pos = (0, 0), size = None, also_include_rect = None):
        viewport_old_orig = self.get_viewport_orig()

        vp_w, vp_h = size or self.window.get_size()
        zoom = self.get_zoom()
        vp_w = int(vp_w/zoom)
        vp_h = int(vp_h/zoom)

        # calculation are done in oldCanvas coordinates
        oldCanvas = Rect(0, 0, self.canvas_w, self.canvas_h)
        viewport  = Rect(int(self.viewport_x+0.5), int(self.viewport_y+0.5), vp_w, vp_h)
        
        # add space; needed to draw into the non-visible part at the border
        border = max(30, min(vp_w/4, vp_h/4)) # quite arbitrary
        viewport.expand(border)
        newCanvas = viewport.copy()

        if also_include_rect:
            # relative to (0, 0) of the current pixbuf
            newCanvas.expandToIncludeRect(Rect(*also_include_rect))

        if old_pixbuf:
            x, y = old_pixbuf_pos
            old_pixbuf_rect = Rect(x, y, old_pixbuf.get_width(), old_pixbuf.get_height())
        else:
            if newCanvas in oldCanvas:
                # big enough already
                return
            old_pixbuf = self.get_as_pixbuf()
            old_pixbuf_rect = Rect(0, 0, old_pixbuf.get_width(), old_pixbuf.get_height())

            # again, avoid too frequent resizing
            viewport.expand(border)
            newCanvas.expandToIncludeRect(viewport)

        newCanvas.expandToIncludeRect(old_pixbuf_rect)


        # now, combine the (possibly already painted) rect with the (visible) viewport
        self.canvas_w = newCanvas.w
        self.canvas_h = newCanvas.h
        self.discard_and_resize(self.canvas_w, self.canvas_h)
        translate_x = oldCanvas.x - newCanvas.x
        translate_y = oldCanvas.y - newCanvas.y

        # paste old image back
        w, h = newCanvas.w, newCanvas.h
        #print 'Resizing canvas to %dx%d = %.3fMB' % (w, h, w*h*3/1024.0/1024.0)
        new_pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, w, h)
        new_pixbuf.fill(0xffffffff) # white
        old_pixbuf.copy_area(src_x=0, src_y=0,
                             width=old_pixbuf.get_width(), height=old_pixbuf.get_height(),
                             dest_pixbuf=new_pixbuf,
                             dest_x=old_pixbuf_rect.x+translate_x, dest_y=old_pixbuf_rect.y+translate_y)
        self.set_from_pixbuf(new_pixbuf)

        self.original_canvas_x0 += translate_x
        self.original_canvas_y0 += translate_y
        self.set_viewport_orig(*viewport_old_orig)

        # free that memory now
        del new_pixbuf, old_pixbuf
        gc.collect()
