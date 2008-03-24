# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

# First widget that allows drawing on a tiled layer.
# TODO:
# - dragging, zooming
# - allow more than one layer (external document object?)

import gtk, numpy
gdk = gtk.gdk
import mypaintlib, tilelib, brush

class TiledDrawWidget(gtk.DrawingArea):
    def __init__(self):
        gtk.DrawingArea.__init__(self)
        #self.connect("dragging-finished", self.dragging_finished_cb)
        self.connect("proximity-in-event", self.proximity_cb)
        self.connect("proximity-out-event", self.proximity_cb)
        self.toolchange_observers = []

        self.connect("motion_notify_event", self.motion_notify_cb)
        #self.connect("button_press_event", self.button_press_cb)
        self.connect("expose_event", self.expose_cb)

        self.set_events(gdk.EXPOSURE_MASK
                        | gdk.LEAVE_NOTIFY_MASK
                        | gdk.BUTTON_PRESS_MASK
                        | gdk.BUTTON_RELEASE_MASK
                        | gdk.POINTER_MOTION_MASK
                        | gdk.PROXIMITY_IN_MASK
                        | gdk.PROXIMITY_OUT_MASK
                        )

        self.set_extension_events (gdk.EXTENSION_EVENTS_ALL)

        self.brush = None
        self.layer = tilelib.TiledLayer()

        self.last_event_time = None

        self.recording = None


    def proximity_cb(self, widget, something):
        for f in self.toolchange_observers:
            f()

    def motion_notify_cb(self, widget, event):
        if not self.brush:
            print 'no brush!'
            return

        if not self.last_event_time:
            self.last_event_time = event.time
            return
        dtime = (event.time - self.last_event_time)/1000.0
        self.last_event_time = event.time

        x = event.x
        y = event.y
        pressure = event.get_axis(gdk.AXIS_PRESSURE)
        if pressure is None:
            if event.state & gtk.gdk.BUTTON1_MASK:
                pressure = 0.5
            else:
                pressure = 0.0

        # OPTIMIZE: move those into the C brush code
        assert pressure >= 0.0 and pressure <= 1.0
        assert x < 1e8 and y < 1e8 and x > -1e8 and y > -1e8

        if self.recording is not None:
            self.recording.append((dtime, x, y, pressure))
        bbox = self.brush.tiled_surface_stroke_to (self.layer, x, y, pressure, dtime)
        if bbox:
            self.queue_draw_area(*bbox)

    def expose_cb(self, widget, event):

        w, h = self.window.get_size()
        pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, w, h)
        gpb =    gdk.Pixbuf(gdk.COLORSPACE_RGB, True, 8, w, h)  

        pixbuf.fill(0xffffffff)
        #arr = gpb.get_pixels_array()
        #print repr(arr)[-50:]
        arr = pixbuf.get_pixels_array()
        #print repr(arr)[-50:]
        self.layer.compositeOverRGB8(arr)
        widget.window.draw_pixbuf(None, pixbuf, 0, 0, 0, 0)

    def clear(self):
        print 'TODO: clear'

    def allow_dragging(self):
        print 'TODO: allow dragging'

    def set_brush(self, b):
        self.brush = b


    def start_recording(self):
        assert self.recording is None
        self.recording = []

    def stop_recording(self):
        # OPTIMIZE 
        # - for space: just gzip? use integer datatypes?
        # - for time: maybe already use array storage while recording?
        data = numpy.array(self.recording, dtype='float64').tostring()
        version = '2'
        self.recording = None
        return version + data

    def playback(self, data):
        version, data = data[0], data[1:]
        assert version == '2'
        for dtime, x, y, pressure in numpy.fromstring(data, dtype='float64'):
            self.brush.tiled_surface_stroke_to (self.layer, x, y, pressure, dtime)
