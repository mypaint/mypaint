# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gobject
import cairo
import gtk
from gtk import gdk
import pygtkcompat

import os
import random
from math import floor, ceil, log
from numpy import isfinite
from warnings import warn

from lib import helpers, tiledsurface, pixbufsurface
import cursor

def _make_testbed_model():
    warn("Creating standalone model for testing", RuntimeWarning, 2)
    import lib.brush, lib.document
    brush = lib.brush.BrushInfo()
    brush.load_defaults()
    return lib.document.Document(brush)


class Overlay:
    """Base class/interface for objects which paint things over a TDW.
    """

    def paint(self, cr):
        """Paint information onto a TiledDrawWidget.

        The drawing interface is very simple. `cr` is a Cairo context in either
        display coordinates or model coordinates: which one you get depends on
        which list the Overlay is appended to on its tdw.
        """
        pass


class CanvasEventBox(gtk.EventBox):
    """Handle events on the canvas."""
    
    def __init__(self, application=None, document=None, drag_handler=None):
        gtk.EventBox.__init__(self)

        self.doc = document
        self.app = application

        self.drag_handler_update = drag_handler

        # workaround for https://gna.org/bugs/?14372 ([Windows] crash when moving the pen during startup)
        def at_application_start(*junk):
            self.connect("motion-notify-event", self.motion_notify_cb)
            self.connect("button-press-event", self.button_press_cb)
            self.connect("button-release-event", self.button_release_cb)
        gobject.idle_add(at_application_start)

        self.add_events(gdk.POINTER_MOTION_MASK
                        # Workaround for https://gna.org/bugs/index.php?16253
                        # Mypaint doesn't use proximity-*-event for anything
                        # yet, but this seems to be needed for scrollwheels
                        # etc. to keep working.
                        | gdk.PROXIMITY_OUT_MASK
                        | gdk.PROXIMITY_IN_MASK
                        # for some reason we also need to specify events handled in drawwindow.py:
                        | gdk.BUTTON_PRESS_MASK
                        | gdk.BUTTON_RELEASE_MASK
                        )

        self.set_extension_events (gdk.EXTENSION_EVENTS_ALL)

        self.last_event_time = None
        self.last_event_x = None
        self.last_event_y = None
        self.last_event_device = None
        self.last_event_had_pressure_info = False
        self.last_painting_pos = None
        self.device_observers = [] #: Notified during drawing when input devices change

        #self.scroll_at_edges = False
        self.pressure_mapping = None

        self.bad_devices = []
        self.motions = []

        self._input_stroke_started_observers = []
        self._input_stroke_ended_observers = [] #: Access via gui.document
        
    def device_used(self, device):
        """Tell the TDW about a device being used."""
        if device == self.last_event_device:
            return True
        for func in self.device_observers:
            func(self.last_event_device, device)
        self.last_event_device = device
        return False

        # Do not interpolate between motion events from different
        # devices.  If the final pressure value from the previous
        # device was not 0.0, the motion event of the new device could
        # cause a visible stroke, even if pressure is 0.0.
        self.doc.brush.reset()


    def motion_notify_cb(self, widget, event, button1_pressed=None):
        if not self.is_sensitive:
            return

        if self.last_event_time:
            dtime = (event.time - self.last_event_time)/1000.0
            dx = event.x - self.last_event_x
            dy = event.y - self.last_event_y
        else:
            dtime = None
        self.last_event_x = event.x
        self.last_event_y = event.y
        self.last_event_time = event.time
        if dtime is None:
            return

        same_device = self.device_used(event.device)

        if self.drag_handler_update is not None:
            in_drag_mode = self.drag_handler_update(dx, dy, event.x, event.y)
            if in_drag_mode:
                # Don't draw in drag mode
                return

        # Refuse drawing if the layer is locked or hidden
        if self.doc.layer.locked or not self.doc.layer.visible:
            return
            # TODO: some feedback, maybe

        x, y = self.display_to_model(event.x, event.y)

        pressure = event.get_axis(gdk.AXIS_PRESSURE)

        if pressure is not None and (pressure > 1.0 or pressure < 0.0 or not isfinite(pressure)):
            if event.device.name not in self.bad_devices:
                print 'WARNING: device "%s" is reporting bad pressure %+f' % (event.device.name, pressure)
                self.bad_devices.append(event.device.name)
            if not isfinite(pressure):
                # infinity/nan: use button state (instead of clamping in brush.hpp)
                # https://gna.org/bugs/?14709
                pressure = None

        if pressure is None:
            self.last_event_had_pressure_info = False
            if button1_pressed is None:
                button1_pressed = event.state & gdk.BUTTON1_MASK
            if button1_pressed:
                pressure = 0.5
            else:
                pressure = 0.0
        else:
            self.last_event_had_pressure_info = True

        xtilt = event.get_axis(gdk.AXIS_XTILT)
        ytilt = event.get_axis(gdk.AXIS_YTILT)
        # Check whether tilt is present.  For some tablets without
        # tilt support GTK reports a tilt axis with value nan, instead
        # of None.  https://gna.org/bugs/?17084
        if xtilt is None or ytilt is None or not isfinite(xtilt+ytilt):
            xtilt = 0.0
            ytilt = 0.0

        if event.state & gdk.CONTROL_MASK or event.state & gdk.MOD1_MASK:
            # color picking, do not paint
            # Don't simply return; this is a workaround for unwanted lines in https://gna.org/bugs/?16169
            pressure = 0.0

        ### CSS experimental - scroll when touching the edge of the screen in fullscreen mode
        #
        # Disabled for the following reasons:
        # - causes irritation when doing fast strokes near the edge
        # - scrolling speed depends on the number of events received (can be a huge difference between tablets/mouse)
        # - also, mouse button scrolling is usually enough
        #
        #if self.scroll_at_edges and pressure <= 0.0:
        #  screen_w = gdk.screen_width()
        #  screen_h = gdk.screen_height()
        #  trigger_area = 10
        #  if (event.x <= trigger_area):
        #    self.scroll(-10,0)
        #  if (event.x >= (screen_w-1)-trigger_area):
        #    self.scroll(10,0)
        #  if (event.y <= trigger_area):
        #    self.scroll(0,-10)
        #  if (event.y >= (screen_h-1)-trigger_area):
        #    self.scroll(0,10)

        if self.pressure_mapping:
            pressure = self.pressure_mapping(pressure)
        if event.state & gdk.SHIFT_MASK:
            pressure = 0.0

        if pressure:
            self.last_painting_pos = x, y

        # If the device has changed and the last pressure value from the previous device
        # is not equal to 0.0, this can leave a visible stroke on the layer even if the 'new'
        # device is not pressed on the tablet and has a pressure axis == 0.0.
        # Reseting the brush when the device changes fixes this issue, but there may be a
        # much more elegant solution that only resets the brush on this edge-case.
        if not same_device:
            self.doc.brush.reset()

        # On Windows, GTK timestamps have a resolution around
        # 15ms, but tablet events arrive every 8ms.
        # https://gna.org/bugs/index.php?16569
        # TODO: proper fix in the brush engine, using only smooth,
        #       filtered speed inputs, will make this unneccessary
        if dtime < 0.0:
            print 'Time is running backwards, dtime=%f' % dtime
            dtime = 0.0
        data = (x, y, pressure, xtilt, ytilt)
        if dtime == 0.0:
            self.motions.append(data)
        elif dtime > 0.0:
            if self.motions:
                # replay previous events that had identical timestamp
                if dtime > 0.1:
                    # really old events, don't associate them with the new one
                    step = 0.1
                else:
                    step = dtime
                step /= len(self.motions)+1
                for data_old in self.motions:
                    self.doc.stroke_to(step, *data_old)
                    dtime -= step
                self.motions = []
            self.doc.stroke_to(dtime, *data)

    def button_press_cb(self, win, event):
        if event.type != gdk.BUTTON_PRESS:
            # ignore the extra double-click event
            return

        if event.button == 1:
            for func in self._input_stroke_started_observers:
                func(event)

            # mouse button pressed (while painting without pressure information)
            if not self.last_event_had_pressure_info:
                # For the mouse we don't get a motion event for "pressure"
                # changes, so we simulate it. (Note: we can't use the
                # event's button state because it carries the old state.)
                self.motion_notify_cb(win, event, button1_pressed=True)

    def button_release_cb(self, win, event):
        # (see comment above in button_press_cb)
        if event.button == 1:
            if not self.last_event_had_pressure_info:
                self.motion_notify_cb(win, event, button1_pressed=False)
            # Outsiders can access this via gui.document
            for func in self._input_stroke_ended_observers:
                func(event)

class DragHandler(object):
    """Handle drag logic and cursor updates."""
    
    CANNOT_DRAW_CURSOR = gdk.Cursor(gdk.CIRCLE)

    
    def __init__(self, doc, widget):
        
        self.doc = doc
        self.widget = widget
        self.drag_op = None
        self.override_cursor = None

    @property
    def window(self):

        if pygtkcompat.USE_GTK3:
            # FIXME: no code can rely on .window in GTK+3
            return None

        return self.widget.window

    def update(self, dx, dy, x, y):
        if self.drag_op:
            self.drag_op.on_update(dx, dy, x, y)
            return True
            
        return False

    def start_drag(self, drag_op, modifier):
        if self.drag_op is not None:
            self.stop_drag()
        assert self.drag_op is None
        self.drag_op = drag_op
        self.drag_op.on_start(modifier)
        c = gdk.Cursor(drag_op.cursor)
        self.set_override_cursor(c)

    def stop_drag(self):
        if self.drag_op is not None:
            self.set_override_cursor(None)
            self.drag_op.on_stop()
            self.drag_op = None

    def update_cursor(self):        
        if not self.window:
            return

        if self.override_cursor is not None:
            c = self.override_cursor
        elif not self.widget.is_sensitive:
            c = None
        elif self.doc.layer.locked or not self.doc.layer.visible:
            c = self.CANNOT_DRAW_CURSOR
        else:
            b = self.doc.brush.brushinfo
            radius = b.get_effective_radius()*self.widget.scale
            c = cursor.get_brush_cursor(radius, b.is_eraser(), b.get_base_value('lock_alpha') > 0.9)
        self.window.set_cursor(c)

    def set_override_cursor(self, cursor):
        """Set a cursor which will always be used.

        Used by the colour picker. The override cursor will be used regardless
        of the criteria update_cursor() normally uses. Pass None to let it
        choose normally again.
        """
        self.override_cursor = cursor
        self.update_cursor()

    def brush_modified_cb(self, settings):
        self.update_cursor()
    

class TiledDrawWidget(gtk.VBox):
    """Widget for showing a lib.document.Document 
    
    Most aspects are delegated to dedicated classes. See CanvasEventBox,
    DragHandler and CanvasRenderer."""

    # Register a GType name for Glade, GtkBuilder etc.
    __gtype_name__ = "TiledDrawWidget"

    def __init__(self, app=None, document=None):
        gtk.VBox.__init__(self)

        self.app = app
        if document is None:
            document = _make_testbed_model()
        self.doc = document

        self.renderer = CanvasRenderer(self.app, self.doc)
        self.drag_handler = DragHandler(self.doc, self.renderer)
        self.event_box = CanvasEventBox(self.app, self.doc, self.drag_handler.update)

        # HACK
        self.event_box.display_to_model = self.renderer.display_to_model
        
        self.event_box.add(self.renderer)
        self.pack_start(self.event_box)

        self.doc.canvas_observers.append(self.renderer.canvas_modified_cb)
        self.doc.brush.brushinfo.observers.append(self.drag_handler.brush_modified_cb)

        self.drag_handler.update_cursor() # hack to get the initial cursor right

        self.has_pointer = False

        def track_pointer(widget):
            """Track pointer enter/leave on widget."""
            widget.add_events(gdk.ENTER_NOTIFY_MASK | gdk.LEAVE_NOTIFY_MASK)
            widget.connect("enter-notify-event", self.enter_notify_cb)
            widget.connect("leave-notify-event", self.leave_notify_cb)

        # We track both renderer and event_box because when pressing and releasing
        # mouse button the event_box gets leave event, but the renderer gets the enter event
        track_pointer(self.event_box)
        track_pointer(self.renderer)

    def enter_notify_cb(self, widget, event):
        self.has_pointer = True
    def leave_notify_cb(self, widget, event):
        self.has_pointer = False

    # Forward public API to delegates
    # TODO: attempt to reduce this interface
    @property
    def start_drag(self):
        return self.drag_handler.start_drag
 
    @property
    def stop_drag(self):
        return self.drag_handler.stop_drag

    @property
    def drag_op(self):
        return self.drag_handler.drag_op

    @property
    def scale(self):
        return self.renderer.scale

    @scale.setter
    def scale(self, n):
        self.renderer.scale = n

    @property
    def zoom_min(self):
        return self.renderer.zoom_min

    @zoom_min.setter
    def zoom_min(self, n):
        self.renderer.zoom_min = n

    @property
    def zoom_max(self):
        return self.renderer.zoom_max

    @zoom_max.setter
    def zoom_max(self, n):
        self.renderer.zoom_max = n

    @property
    def rotation(self):
        return self.renderer.rotation

    @property
    def mirrored(self):
        return self.renderer.mirrored

    @property
    def display_overlays(self):
        return self.renderer.display_overlays

    @property
    def model_overlays(self):
        return self.renderer.model_overlays

    @property
    def overlay_layer(self):
        return self.renderer.overlay_layer

    @overlay_layer.setter
    def overlay_layer(self, l):
        self.renderer.overlay_layer = l

    @property
    def device_observers(self):
        return self.event_box.device_observers

    @property
    def _input_stroke_started_observers(self):
        return self.event_box._input_stroke_started_observers

    @property
    def _input_stroke_ended_observers(self):
        return self.event_box._input_stroke_ended_observers

    @property
    def recenter_document(self):
        return self.renderer.recenter_document

    @property
    def display_to_model(self):
        return self.renderer.display_to_model

    @property
    def model_to_display(self):
        return self.renderer.model_to_display

    @property
    def set_override_cursor(self):
        return self.drag_handler.set_override_cursor

    @property
    def device_used(self):
        return self.event_box.device_used

    def get_last_painting_pos(self):
        return self.event_box.last_painting_pos
    def set_last_painting_pos(self, value):
        self.event_box.last_painting_pos = value
    last_painting_pos = property(get_last_painting_pos, set_last_painting_pos)

    @property
    def get_cursor_in_model_coordinates(self):
        return self.renderer.get_cursor_in_model_coordinates

    @property
    def scroll(self):
        return self.renderer.scroll

    @property
    def get_center(self):
        return self.renderer.get_center

    @property
    def toggle_show_layers_above(self):
        return self.renderer.toggle_show_layers_above

    @property
    def queue_draw_area(self):
        return self.renderer.queue_draw_area

    def get_current_layer_solo(self):
        return self.renderer.current_layer_solo
    def set_current_layer_solo(self, enabled):
        self.renderer.current_layer_solo = enabled
    current_layer_solo = property(get_current_layer_solo, set_current_layer_solo)

    def get_neutral_background_pixbuf(self):
        return self.renderer.neutral_background_pixbuf
    def set_neutral_background_pixbuf(self, pixbuf):
        self.renderer.neutral_background_pixbuf = pixbuf
    neutral_background_pixbuf = property(get_neutral_background_pixbuf, set_neutral_background_pixbuf)

    # Transform logic
    def rotozoom_with_center(self, function, at_pointer=False):
        if at_pointer and self.has_pointer and self.event_box.last_event_x is not None:
            cx, cy = self.event_box.last_event_x, self.event_box.last_event_y
        else:
            allocation = self.get_allocation()
            w, h = allocation.width, allocation.height
            cx, cy = self.renderer.get_center()
        cx_model, cy_model = self.renderer.display_to_model(cx, cy)
        function()
        self.renderer.scale = helpers.clamp(self.renderer.scale, self.renderer.zoom_min, self.renderer.zoom_max)
        cx_new, cy_new = self.renderer.model_to_display(cx_model, cy_model)
        self.renderer.translation_x += cx - cx_new
        self.renderer.translation_y += cy - cy_new
        self.renderer.queue_draw()

    def zoom(self, zoom_step):
        def f(): self.renderer.scale *= zoom_step
        self.rotozoom_with_center(f, at_pointer=True)

    def set_zoom(self, zoom, at_pointer=True):
        def f(): self.renderer.scale = zoom
        self.rotozoom_with_center(f, at_pointer)
        self.drag_handler.update_cursor()

    def rotate(self, angle_step):
        if self.renderer.mirrored: angle_step = -angle_step
        def f(): self.renderer.rotation += angle_step
        self.rotozoom_with_center(f)

    def set_rotation(self, angle):
        if self.renderer.mirrored: angle = -angle
        def f(): self.renderer.rotation = angle
        self.rotozoom_with_center(f)

    def mirror(self):
        def f(): self.renderer.mirrored = not self.renderer.mirrored
        self.rotozoom_with_center(f)

    def set_mirrored(self, mirrored):
        def f(): self.renderer.mirrored = mirrored
        self.rotozoom_with_center(f)

class CanvasRenderer(gtk.DrawingArea):
    """Render the document model to screen.
    
    Can render the document in a transformed way, including translation,
    scaling and rotation."""

    def __init__(self, app=None, document=None):
        gtk.DrawingArea.__init__(self)

        if pygtkcompat.USE_GTK3:
            pass #FIXME: implement
        else:
            self.connect("expose-event", self.expose_cb)

        self.connect("size-allocate", self.size_allocate_cb)
        self.connect("state-changed", self.state_changed_cb)

        self.app = app
        self.doc = document
        
        self.cursor_info = None
        self.visualize_rendering = False

        self.translation_x = 0.0
        self.translation_y = 0.0
        self.scale = 1.0
        self.rotation = 0.0
        self.mirrored = False

        self.current_layer_solo = False
        self.show_layers_above = True

        self.overlay_layer = None

        # gets overwritten for the main window
        self.zoom_max = 5.0
        self.zoom_min = 1/5.0

        # Sensitivity; we draw via a cached snapshot while the widget is
        # insensitive. tdws are generally only insensitive during loading and
        # saving, and because we now process the GTK main loop during loading
        # and saving, we need to avoid drawing partially-loaded files.

        self.is_sensitive = True    # just mirrors gtk.STATE_INSENSITIVE
        self.snapshot_pixmap = None

        # Overlays
        self.model_overlays = []
        self.display_overlays = []

    def state_changed_cb(self, widget, oldstate):
        # Keeps track of the sensitivity state, and regenerates
        # the snapshot pixbuf on entering it.
        sensitive = self.get_state() != gtk.STATE_INSENSITIVE
        if sensitive:
            self.snapshot_pixmap = None
        else:
            if self.snapshot_pixmap is None:
                self.snapshot_pixmap = self.get_snapshot()
        self.is_sensitive = sensitive

    def size_allocate_cb(self, widget, allocation):
        old_alloc = getattr(self, 'stored_allocation', allocation)
        if old_alloc != allocation:
            dx = allocation.x - old_alloc.x
            dy = allocation.y - old_alloc.y
            self.scroll(dx, dy)
        self.stored_allocation = allocation

    def canvas_modified_cb(self, x1, y1, w, h):
        if not pygtkcompat.USE_GTK3 and not self.window:
            return

        if w == 0 and h == 0:
            # Full redraw (used when background has changed).
            #print 'full redraw'
            self.queue_draw()
            return

        if self.is_translation_only():
            x, y = self.model_to_display(x1, y1)
            self.queue_draw_area(int(x), int(y), w, h)
        else:
            # Create an expose event with the event bbox rotated/zoomed.
            # OPTIMIZE: This is estimated to cause at least twice as much
            #           rendering work as neccessary.
            # Transform 4 bbox corners to screen coordinates.
            corners = [(x1, y1), (x1+w-1, y1), (x1, y1+h-1), (x1+w-1, y1+h-1)]
            corners = [self.model_to_display(x, y) for (x, y) in corners]
            self.queue_draw_area(*helpers.rotated_rectangle_bbox(corners))

    def expose_cb(self, widget, event):
        
        if self.snapshot_pixmap:
            gc = self.get_style().fg_gc[self.get_state()]
            area = event.area
            x,y,w,h = area.x, area.y, area.width, area.height
            self.window.draw_drawable(gc, self.snapshot_pixmap, x,y, x,y, w,h)
        else:
            self.repaint(event.area)
        return True


    def display_to_model(self, disp_x, disp_y):
        """Converts display coordinates to model coordinates.
        """
        cr = self.__get_model_cairo_context()
        return cr.device_to_user(disp_x, disp_y)


    def model_to_display(self, model_x, model_y):
        """Converts model coordinates to display coordinates.
        """
        cr = self.__get_model_cairo_context()
        return cr.user_to_device(model_x, model_y)


    def __get_model_cairo_context(self, cr=None):
        # OPTIMIZE: Check whether this is a bottleneck during
        #           painting (many motion events) - if yes, use cache.
        if cr is None:
            cr = self.window.cairo_create()

        scale = self.scale
        # check if scale is almost a power of two
        scale_log2 = log(scale, 2)
        scale_log2_rounded = round(scale_log2)
        if abs(scale_log2-scale_log2_rounded) < 0.01:
            scale = 2.0**scale_log2_rounded

        rotation = self.rotation
        # maybe we should check if rotation is almost a multiple of 90 degrees?

        cr.translate(self.translation_x, self.translation_y)
        cr.rotate(rotation)
        cr.scale(scale, scale)

        # Align the translation such that (0,0) maps to an integer
        # screen pixel, to keep image rendering fast and sharp.
        x, y = cr.user_to_device(0, 0)
        x, y = cr.device_to_user(round(x), round(y))
        cr.translate(x, y)

        if self.mirrored:
            m = list(cr.get_matrix())
            m[0] = -m[0]
            m[2] = -m[2]
            cr.set_matrix(cairo.Matrix(*m))
        return cr

    def is_translation_only(self):
        return self.rotation == 0.0 and self.scale == 1.0 and not self.mirrored

    def get_cursor_in_model_coordinates(self):
        x, y, modifiers = self.window.get_pointer()
        return self.display_to_model(x, y)

    def get_visible_layers(self):
        # FIXME: tileddrawwidget should not need to know whether the document has layers
        layers = self.doc.layers
        if not self.show_layers_above:
            layers = self.doc.layers[0:self.doc.layer_idx+1]
        layers = [l for l in layers if l.visible]
        return layers

    def repaint(self, device_bbox=None):
        cr, surface, sparse, mipmap_level, gdk_clip_region = self.render_prepare(device_bbox)
        self.render_execute(cr, surface, sparse, mipmap_level, gdk_clip_region)
        # Model coordinate space:
        cr.restore()  # CONTEXT2<<<
        for overlay in self.model_overlays:
            cr.save()
            overlay.paint(cr)
            cr.restore()
        # Back to device coordinate space
        cr.restore()  # CONTEXT1<<<
        for overlay in self.display_overlays:
            cr.save()
            overlay.paint(cr)
            cr.restore()

    def render_prepare(self, device_bbox):
        if device_bbox is None:
            allocation = self.get_allocation()
            w, h = allocation.width, allocation.height
            device_bbox = (0, 0, w, h)
        #print 'device bbox', tuple(device_bbox)

        gdk_clip_region = self.window.get_clip_region()
        x, y, w, h = device_bbox
        sparse = not gdk_clip_region.point_in(x+w/2, y+h/2)

        cr = self.window.cairo_create()

        # actually this is only neccessary if we are not answering an expose event
        cr.rectangle(*device_bbox)
        cr.clip()

        # fill it all white, though not required in the most common case
        if self.visualize_rendering:
            # grey
            tmp = random.random()
            cr.set_source_rgb(tmp, tmp, tmp)
            cr.paint()

        # bye bye device coordinates
        cr.save()   # >>>CONTEXT1
        self.__get_model_cairo_context(cr)
        cr.save()   # >>>CONTEXT2

        # choose best mipmap
        hq_zoom = False
        if self.app and self.app.preferences['view.high_quality_zoom']:
            hq_zoom = True
        if hq_zoom:
            # can cause a very clear slowdown on some hardware
            # (we probably could avoid this by doing rendering differently)
            mipmap_level = max(0, int(floor(log(1.0/self.scale,2))))
        else:
            mipmap_level = max(0, int(ceil(log(1/self.scale,2))))
        # OPTIMIZE: if we would render tile scanlines, we could probably use the better one above...
        mipmap_level = min(mipmap_level, tiledsurface.MAX_MIPMAP_LEVEL)
        cr.scale(2**mipmap_level, 2**mipmap_level)

        translation_only = self.is_translation_only()

        # calculate the final model bbox with all the clipping above
        x1, y1, x2, y2 = cr.clip_extents()
        if not translation_only:
            # Looks like cairo needs one extra pixel rendered for interpolation at the border.
            # If we don't do this, we get dark stripe artefacts when panning while zoomed.
            x1 -= 1
            y1 -= 1
            x2 += 1
            y2 += 1
        x1, y1 = int(floor(x1)), int(floor(y1))
        x2, y2 = int(ceil (x2)), int(ceil (y2))

        # We render with alpha just to get hardware acceleration, we
        # don't actually use the alpha channel. Speedup factor 3 for
        # ATI/Radeon Xorg driver (and hopefully others).
        # https://bugs.freedesktop.org/show_bug.cgi?id=28670
        surface = pixbufsurface.Surface(x1, y1, x2-x1+1, y2-y1+1)

        del x1, y1, x2, y2, w, h

        return cr, surface, sparse, mipmap_level, gdk_clip_region

    def render_execute(self, cr, surface, sparse, mipmap_level, gdk_clip_region):
        translation_only = self.is_translation_only()
        model_bbox = surface.x, surface.y, surface.w, surface.h

        #print 'model bbox', model_bbox

        # not sure if it is a good idea to clip so tightly
        # has no effect right now because device_bbox is always smaller
        cr.rectangle(*model_bbox)
        cr.clip()

        layers = self.get_visible_layers()

        if self.visualize_rendering:
            surface.pixbuf.fill((int(random.random()*0xff)<<16)+0x00000000)

        tiles = surface.get_tiles()

        background = None
        if self.current_layer_solo:
            background = self.neutral_background_pixbuf
            layers = [self.doc.layer]
            # this is for hiding instead
            #layers.pop(self.doc.layer_idx)
        if self.overlay_layer:
            idx = layers.index(self.doc.layer)
            layers.insert(idx+1, self.overlay_layer)

        for tx, ty in tiles:
            if sparse:
                # it is worth checking whether this tile really will be visible
                # (to speed up the L-shaped expose event during scrolling)
                # (speedup clearly visible; slowdown measurable when always executing this code)
                N = tiledsurface.N
                if translation_only:
                    x, y = cr.user_to_device(tx*N, ty*N)
                    bbox = (int(x), int(y), N, N)
                else:
                    #corners = [(tx*N, ty*N), ((tx+1)*N-1, ty*N), (tx*N, (ty+1)*N-1), ((tx+1)*N-1, (ty+1)*N-1)]
                    # same problem as above: cairo needs to know one extra pixel for interpolation
                    corners = [(tx*N-1, ty*N-1), ((tx+1)*N, ty*N-1), (tx*N-1, (ty+1)*N), ((tx+1)*N, (ty+1)*N)]
                    corners = [cr.user_to_device(x_, y_) for (x_, y_) in corners]
                    bbox = gdk.Rectangle(*helpers.rotated_rectangle_bbox(corners))

                if gdk_clip_region.rect_in(bbox) == gdk.OVERLAP_RECTANGLE_OUT:
                    continue


            dst = surface.get_tile_memory(tx, ty)
            self.doc.blit_tile_into(dst, False, tx, ty, mipmap_level, layers, background)

        if translation_only:
            # not sure why, but using gdk directly is notably faster than the same via cairo
            x, y = cr.user_to_device(surface.x, surface.y)
            self.window.draw_pixbuf(None, surface.pixbuf, 0, 0, int(x), int(y), dither=gdk.RGB_DITHER_MAX)
        else:
            #print 'Position (screen coordinates):', cr.user_to_device(surface.x, surface.y)
            cr.set_source_pixbuf(surface.pixbuf, round(surface.x), round(surface.y))
            pattern = cr.get_source()

            # We could set interpolation mode here (eg nearest neighbour)
            #pattern.set_filter(cairo.FILTER_NEAREST)  # 1.6s
            #pattern.set_filter(cairo.FILTER_FAST)     # 2.0s
            #pattern.set_filter(cairo.FILTER_GOOD)     # 3.1s
            #pattern.set_filter(cairo.FILTER_BEST)     # 3.1s
            #pattern.set_filter(cairo.FILTER_BILINEAR) # 3.1s

            if self.scale > 3.0:
                # pixelize at high zoom-in levels
                pattern.set_filter(cairo.FILTER_NEAREST)

            cr.paint()

        if self.doc.frame_enabled:
            # Draw a overlay for all the area outside the "document area"
            cr.save()
            cr.set_source_rgba(0.25, 0.25, 0.25, 1.0)
            cr.set_operator(cairo.OPERATOR_OVER)
            mipmap_factor = 2**mipmap_level
            frame = self.doc.get_frame()
            cr.rectangle(frame[0]/mipmap_factor, frame[1]/mipmap_factor,
                            frame[2]/mipmap_factor, frame[3]/mipmap_factor)
            cr.rectangle(*model_bbox)
            cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            cr.fill()
            cr.restore()

        if self.visualize_rendering:
            # visualize painted bboxes (blue)
            cr.set_source_rgba(0, 0, random.random(), 0.4)
            cr.paint()

    def scroll(self, dx, dy):
        self.translation_x -= dx
        self.translation_y -= dy
        if False:
            # This speeds things up nicely when scrolling is already
            # fast, but produces temporary artefacts and an
            # annoyingliy non-constant framerate otherwise.
            #
            # It might be worth it if it was done only once per
            # redraw, instead of once per motion event. Maybe try to
            # implement something like "queue_scroll" with priority
            # similar to redraw?
            self.window.scroll(int(-dx), int(-dy))
        else:
            self.queue_draw()

    def get_center(self):
        """Return the centre position in display coordinates.
        """
        alloc = self.get_allocation()
        return alloc.width/2.0, alloc.height/2.0

    def recenter_document(self):
        """Recentres the view onto the document's centre.
        """
        x, y, w, h = self.doc.get_effective_bbox()
        desired_cx_model = x+w/2.0
        desired_cy_model = y+h/2.0
        cx, cy = self.get_center()
        cx_model, cy_model = self.display_to_model(cx, cy)
        self.translation_x += (cx_model - desired_cx_model)*self.scale
        self.translation_y += (cy_model - desired_cy_model)*self.scale
        self.queue_draw()

    def toggle_show_layers_above(self):
        self.show_layers_above = not self.show_layers_above
        self.queue_draw()


if __name__ == '__main__':
    tdw = TiledDrawWidget()
    tdw.set_size_request(640, 480)
    win = gtk.Window()
    win.set_title("tdw test")
    win.connect("destroy", lambda *a: gtk.main_quit())
    win.add(tdw)
    win.show_all()
    gtk.main()
