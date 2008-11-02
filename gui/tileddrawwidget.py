# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

from lib import helpers
import gtk, numpy, cairo
gdk = gtk.gdk
from math import floor, ceil
import time

import random
from lib import mypaintlib, tiledsurface # FIXME: should not have to import those

class TiledDrawWidget(gtk.DrawingArea):

    def __init__(self, document):
        gtk.DrawingArea.__init__(self)
        #self.connect("proximity-in-event", self.proximity_cb)
        #self.connect("proximity-out-event", self.proximity_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)
        self.connect("button-press-event", self.button_updown_cb)
        self.connect("button-release-event", self.button_updown_cb)
        self.connect("expose-event", self.expose_cb)
        self.connect("enter-notify-event", self.enter_notify_cb)
        self.connect("leave-notify-event", self.leave_notify_cb)

        self.set_events(gdk.EXPOSURE_MASK
                        | gdk.ENTER_NOTIFY_MASK
                        | gdk.LEAVE_NOTIFY_MASK
                        | gdk.BUTTON_PRESS_MASK
                        | gdk.BUTTON_RELEASE_MASK
                        | gdk.POINTER_MOTION_MASK
                        | gdk.PROXIMITY_IN_MASK
                        | gdk.PROXIMITY_OUT_MASK
                        )

        self.set_extension_events (gdk.EXTENSION_EVENTS_ALL)

        self.doc = document
        self.doc.canvas_observers.append(self.canvas_modified_cb)
        self.doc.brush.observers.append(self.brush_modified_cb)

        self.cursor_size = None

        self.last_event_time = None
        self.last_event_x = None
        self.last_event_y = None

        self.visualize_rendering = False

        self.translation_x = 0.0
        self.translation_y = 0.0
        self.scale = 1.0
        self.rotation = 0.0
        self.viewport_locked = False

        self.has_pointer = False
        self.dragfunc = None

        # gets overwritten for the main window
        self.zoom_max = 5.0
        self.zoom_min = 1/5.0

        self.show_layers_above = True
        self.doc.layer_observers.append(self.layer_selected_cb)


    def button_updown_cb(self, widget, event):
        d = event.device
        if False:
            print 'button_updown_cb', repr(d.name), event.button, event.type
            print '  has_cursor', d.has_cursor
            print '  mode', d.mode
            print '  axes', d.num_axes, [event.get_axis(i) for i in range(d.num_axes+1)]
            print '  keys', d.num_keys
            print '  source', d.source
            #print '  state', d.get_state()

    def enter_notify_cb(self, widget, event):
        self.has_pointer = True
    def leave_notify_cb(self, widget, event):
        self.has_pointer = False

    def motion_notify_cb(self, widget, event):
        if self.last_event_time:
            dtime = (event.time - self.last_event_time)/1000.0
            dx = event.x - self.last_event_x
            dy = event.y - self.last_event_y
        else:
            dtime = None
        self.last_event_time = event.time
        self.last_event_x = event.x
        self.last_event_y = event.y
        if dtime is None:
            return

        if self.dragfunc:
            self.dragfunc(dx, dy)
            return

        cr = self.get_model_coordinates_cairo_context()
        x, y = cr.device_to_user(event.x, event.y)
        
        pressure = event.get_axis(gdk.AXIS_PRESSURE)
        if pressure is None:
            if event.state & gdk.BUTTON1_MASK:
                pressure = 0.5
            else:
                pressure = 0.0

        if not self.doc.brush:
            print 'no brush!'
            return

        self.doc.stroke_to(dtime, x, y, pressure)

    def canvas_modified_cb(self, x1, y1, w, h):
        # create an expose event with the event bbox rotated/zoomed
        # OPTIMIZE: estimated to cause at least twice more rendering work than neccessary
        x2 = x1 + w - 1
        y2 = y1 + h - 1
        # transform 4 bbox corners to screen coordinates
        corners = [(x1, y1), (x1+w-1, y1), (x1, y1+h-1), (x1+w-1, y1+h-1)]
        cr = self.get_model_coordinates_cairo_context() # OPTIMIZE: profile how much time does this takes; we could easily get rid of it
        corners = [cr.user_to_device(x, y) for (x, y) in corners]
        self.queue_draw_area(*helpers.rotated_rectangle_bbox(corners))

    def expose_cb(self, widget, event):
        self.update_cursor() # hack to get the initial cursor right

        t = time.time()
        if hasattr(self, 'last_expose_time'):
            # just for basic performance comparisons... but we could sleep if we make >50fps
            #print '%d fps' % int(1.0/(t-self.last_expose_time))
            pass
        self.last_expose_time = t
        #print 'expose', tuple(event.area)

        self.repaint(event.area)

    def get_model_coordinates_cairo_context(self, cr=None):
        if cr is None:
            cr = self.window.cairo_create()
        cr.translate(self.translation_x, self.translation_y)
        cr.rotate(self.rotation)
        cr.scale(self.scale, self.scale)
        # does not seem to make a difference:
        #cr.set_antialias(cairo.ANTIALIAS_SUBPIXEL)
        # this one neither:
        #cr.set_antialias(cairo.ANTIALIAS_NONE)
        # looks like we always get nearest-neighbour downsampling
        return cr

    def repaint(self, device_bbox=None, model_bbox=None):
        # FIXME: ...we do not fill tile-free background white in this function...

        cr = self.window.cairo_create()

        if device_bbox is None:
            w, h = self.window.get_size()
            device_bbox = (0, 0, w, h)

        #print 'device bbox', tuple(device_bbox)

        # actually this is only neccessary if we are not answering an expose event
        cr.rectangle(*device_bbox)
        cr.clip()

        # fill it all white, though not required in the most common case
        if self.visualize_rendering:
            # grey
            tmp = random.random()
            cr.set_source_rgb(tmp, tmp, tmp)
        else:
            cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.paint()

        # bye bye device coordinates
        self.get_model_coordinates_cairo_context(cr)

        if model_bbox is not None:
            cr.rectangle(*model_bbox)
            cr.clip()

        # do not attempt to render all the possible white space outside the image (can be huge if zoomed out)
        cr.rectangle(*self.doc.get_bbox())
        cr.clip()

        # calculate the final model bbox with all the clipping above
        x1, y1, x2, y2 = cr.clip_extents()
        x1, y1 = int(floor(x1)), int(floor(y1))
        x2, y2 = int(ceil (x2)), int(ceil (y2))

        from lib import tiledsurface
        N = tiledsurface.N
        # FIXME: remove this limitation?
        # code here should not need to know about tiles?
        x1 = x1/N*N
        y1 = y1/N*N
        x2 = (x2/N*N) + N
        y2 = (y2/N*N) + N
        w, h = x2-x1+1, y2-y1+1
        model_bbox = x1, y1, w, h
        assert w >= 0 and h >= 0

        #print 'model bbox', model_bbox

        # not sure if it is a good idea to clip so tightly
        # has no effect right now because device_bbox is always smaller
        cr.rectangle(*model_bbox)
        cr.clip()
        
        #print 'rendering pixbuf', w, h

        pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, w, h)
        if self.visualize_rendering:
            # green
            #pixbuf.fill((int(random.random()*0xff)<<8)&0xff000000)
            pixbuf.fill(0xffffffff)
        else:
            pixbuf.fill(0xffffffff)
        arr = pixbuf.get_pixels_array()
        arr = mypaintlib.gdkpixbuf2numpy(arr)

        layers = None
        if self.show_layers_above:
            layers = self.doc.layers[0:self.doc.layer_idx+1]
        self.doc.render(arr, -x1, -y1, layers)

        #widget.window.draw_pixbuf(None, pixbuf, 0, 0, 0, 0)

        cr.set_source_pixbuf(pixbuf, x1, y1)
        cr.paint()

        if self.visualize_rendering:
            # visualize painted bboxes (blue)
            cr.set_source_rgba(0, 0, random.random(), 0.4)
            cr.paint()


    def lock_viewport(self, lock=True):
        self.viewport_locked = lock

    def scroll(self, dx, dy):
        if self.viewport_locked:
            return
        self.translation_x -= dx
        self.translation_y -= dy
        #OPTIMIZE: fast scrolling without so much rerendering
        # but not if combined with rotation/zoom change
        self.queue_draw()

    def rotozoom_with_center(self, function):
        if self.viewport_locked:
            return
        if self.has_pointer and self.last_event_x is not None:
            cx, cy = self.last_event_x, self.last_event_y
        else:
            w, h = self.window.get_size()
            cx, cy = w/2.0, h/2.0
        cr = self.get_model_coordinates_cairo_context()
        cx_device, cy_device = cr.device_to_user(cx, cy)
        function()
        self.scale = helpers.clamp(self.scale, self.zoom_min, self.zoom_max)
        cr = self.get_model_coordinates_cairo_context()
        cx_new, cy_new = cr.user_to_device(cx_device, cy_device)
        self.translation_x += cx - cx_new
        self.translation_y += cy - cy_new
        self.queue_draw()

    def zoom(self, zoom_step):
        def f(): self.scale *= zoom_step
        self.rotozoom_with_center(f)

    def set_zoom(self, zoom):
        def f(): self.scale = zoom
        self.rotozoom_with_center(f)

    def rotate(self, angle_step):
        def f(): self.rotation += angle_step
        self.rotozoom_with_center(f)

    def set_rotation(self, angle):
        def f(): self.rotation = angle
        self.rotozoom_with_center(f)


    def start_drag(self, dragfunc):
        self.dragfunc = dragfunc
    def stop_drag(self, dragfunc):
        if self.dragfunc == dragfunc:
            self.dragfunc = None


    def brush_modified_cb(self):
        self.update_cursor()

    def update_cursor(self):
        #return
        # OPTIMIZE: looks like this can be a major slowdown with X11
        if not self.window: return
        d = int(self.doc.brush.get_actual_radius())*2

        if d < 6: d = 6
        if d > 500: d = 500 # hm, better ask display for max cursor size? also, 500 is pretty slow
        if self.cursor_size == d:
            return
        self.cursor_size = d

        cursor = gdk.Pixmap(None, d+1, d+1,1)
        mask   = gdk.Pixmap(None, d+1, d+1,1)
        colormap = gdk.colormap_get_system()
        black = colormap.alloc_color('black')
        white = colormap.alloc_color('white')

        bgc = cursor.new_gc(foreground=black)
        wgc = cursor.new_gc(foreground=white)
        cursor.draw_rectangle(wgc, True, 0, 0, d+1, d+1)
        cursor.draw_arc(bgc,False, 0, 0, d, d, 0, 360*64)

        bgc = mask.new_gc(foreground=black)
        wgc = mask.new_gc(foreground=white)
        mask.draw_rectangle(bgc, True, 0, 0, d+1, d+1)
        mask.draw_arc(wgc, False, 0, 0, d, d, 0, 360*64)
        mask.draw_arc(wgc, False, 1, 1, d-2, d-2, 0, 360*64)

        self.window.set_cursor(gdk.Cursor(cursor,mask,gdk.color_parse('black'), gdk.color_parse('white'),(d+1)/2,(d+1)/2))

    def layer_selected_cb(self):
        self.queue_draw() # OPTIMIZE

    def toggle_show_layers_above(self):
        self.show_layers_above = not self.show_layers_above
        self.layer_selected_cb()
