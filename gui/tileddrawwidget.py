# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk2compat
import gobject
import cairo
import gtk
from gtk import gdk

import os
import random
from math import floor, ceil, log, exp
from numpy import isfinite
from warnings import warn
import weakref
import logging
logger = logging.getLogger(__name__)

from lib import helpers, tiledsurface, pixbufsurface
from lib.observable import event
import cursor


class TiledDrawWidget (gtk.EventBox):
    """Widget for showing a lib.document.Document

    Rendering is delegated to a dedicated class see `CanvasRenderer`.

    """

    ## Register a GType name for Glade, GtkBuilder etc.
    __gtype_name__ = "TiledDrawWidget"


    # List of weakrefs to all known TDWs.
    __tdw_refs = []


    @classmethod
    def get_active_tdw(kin):
        """Returns the most recently created or entered TDW.
        """
        # Find and return the first visible, mapped etc. TDW in the list
        invis_refs = []
        active_tdw = None
        while len(kin.__tdw_refs) > 0:
            tdw_ref = kin.__tdw_refs[0]
            tdw = tdw_ref()
            if tdw is not None:
                if tdw.get_window() is not None and tdw.get_visible():
                    active_tdw = tdw
                    break
                else:
                    invis_refs.append(tdw_ref)
            kin.__tdw_refs.pop(0)
        kin.__tdw_refs.extend(invis_refs)
        assert active_tdw is not None
        return active_tdw


    def __init__(self):
        """Instantiate a TiledDrawWidget.

        """
        gtk.EventBox.__init__(self)

        if __name__ == '__main__':
            app = None
        else:
            import application
            app = application.get_app()
        self.app = app
        self.doc = None

        self.add_events(gdk.POINTER_MOTION_MASK
            # Workaround for https://gna.org/bugs/index.php?16253
            # Mypaint doesn't use proximity-*-event for anything
            # yet, but this seems to be needed for scrollwheels
            # etc. to keep working.
            | gdk.PROXIMITY_OUT_MASK
            | gdk.PROXIMITY_IN_MASK
            # For some reason we also need to specify events
            # handled in drawwindow.py:
            | gdk.BUTTON_PRESS_MASK
            | gdk.BUTTON_RELEASE_MASK)
        self.last_painting_pos = None

        self.renderer = CanvasRenderer(self)
        self.add(self.renderer)
        self.renderer.update_cursor() # get the initial cursor right

        self.add_events(gdk.ENTER_NOTIFY_MASK)
        self.connect("enter-notify-event", self.enter_notify_cb)
        self.__tdw_refs.insert(0, weakref.ref(self))

        self._last_alloc_pos = (0, 0)
        self.connect("size-allocate", self._size_allocate_cb)

        #: Scroll to match appearing/disappearing sidebars and toolbars.
        self.scroll_on_allocate = True

        forwarder = lambda *a: self.transformation_updated()
        self.renderer.transformation_updated += forwarder

    @event
    def transformation_updated(self):
        """Forwarded event: transformation was updated"""


    def _size_allocate_cb(self, widget, alloc):
        """Allow for allocation changes under certain circumstances

        We need to allow for changes like toolbars or sidebars appearing or
        disappearing on the top or the left.  The canvas position should remain
        stationary on the screen in these cases.  This size-allocate handler
        deals with that by issuing appropriate scroll() events.

        See also `scroll_on_allocate`.

        """
        # Capture the last allocated position in toplevel coords
        toplevel = self.get_toplevel()
        new_pos = self.translate_coordinates(toplevel, alloc.x, alloc.y)
        old_pos = self._last_alloc_pos
        self._last_alloc_pos = new_pos
        # When things change measurably, scroll to make up the difference
        if not self.scroll_on_allocate:
            return
        if None in (old_pos, new_pos):
            return
        if old_pos != new_pos:
            dx = new_pos[0] - old_pos[0]
            dy = new_pos[1] - old_pos[1]
            self.renderer.scroll(dx, dy)


    def set_model(self, model):
        assert self.doc is None
        renderer = self.renderer
        model.canvas_observers.append(renderer.canvas_modified_cb)
        model.doc_observers.append(renderer.model_structure_changed_cb)
        model.brush.brushinfo.observers.append(renderer.brush_modified_cb)
        self.doc = model
        self.renderer.queue_draw()


    def enter_notify_cb(self, widget, event):
        # Track the active TDW
        self_ref = weakref.ref(self)
        self.__tdw_refs.remove(self_ref)
        self.__tdw_refs.insert(0, self_ref)
        # Ensure the cursor reflects layer-locking changes etc.
        self.renderer.update_cursor()


    # Forward public API to delegates

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
    def pixelize_threshold(self):
        return self.renderer.pixelize_threshold


    @pixelize_threshold.setter
    def pixelize_threshold(self, n):
        self.renderer.pixelize_threshold = n


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
        return self.renderer.set_override_cursor

    def get_last_painting_pos(self):
        return self.last_painting_pos

    def set_last_painting_pos(self, value):
        self.last_painting_pos = value

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
    def get_center_model_coords(self):
        return self.renderer.get_center_model_coords

    @property
    def recenter_on_model_coords(self):
        return self.renderer.recenter_on_model_coords

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
    def rotozoom_with_center(self, function, center=None):
        cx = None
        cy = None
        if center is not None:
            cx, cy = center
        if cx is None or cy is None:
            cx, cy = self.renderer.get_center()
        cx_model, cy_model = self.renderer.display_to_model(cx, cy)
        function()
        self.renderer.scale = helpers.clamp(self.renderer.scale,
                                            self.renderer.zoom_min,
                                            self.renderer.zoom_max)
        cx_new, cy_new = self.renderer.model_to_display(cx_model, cy_model)
        self.renderer.translation_x += cx - cx_new
        self.renderer.translation_y += cy - cy_new
        self.renderer.queue_draw()

    def zoom(self, zoom_step, center=None):
        def f(): self.renderer.scale *= zoom_step
        self.rotozoom_with_center(f, center)

    def set_zoom(self, zoom, center=None):
        def f(): self.renderer.scale = zoom
        self.rotozoom_with_center(f, center)
        self.renderer.update_cursor()

    def rotate(self, angle_step, center=None):
        if self.renderer.mirrored: angle_step = -angle_step
        def f(): self.renderer.rotation += angle_step
        self.rotozoom_with_center(f, center)

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


    def get_transformation(self):
        """Returns a snapshot/memento/record of the current transformation.

        :rtype: a CanvasTransformation initialized with a copy of the current
          transformation variables.

        """
        tr = CanvasTransformation()
        tr.translation_x = self.renderer.translation_x
        tr.translation_y = self.renderer.translation_y
        tr.scale = self.renderer.scale
        tr.rotation = self.renderer.rotation
        tr.mirrored = self.renderer.mirrored
        return tr


    def set_transformation(self, transformation):
        """Sets the current transformation, and redraws.

        :param transformation: a CanvasTransformation object.

        """
        self.renderer.translation_x = transformation.translation_x
        self.renderer.translation_y = transformation.translation_y
        self.renderer.scale = transformation.scale
        self.renderer.rotation = transformation.rotation
        self.renderer.mirrored = transformation.mirrored
        self.renderer.queue_draw()
        self.renderer.update_cursor()


class CanvasTransformation (object):
    """Record of a TiledDrawWidget's canvas (view) transformation.
    """

    translation_x = 0.0
    translation_y = 0.0
    scale = 1.0
    rotation = 0.0
    mirrored = False

    def __repr__(self):
        return "<%s dx=%0.3f dy=%0.3f scale=%0.3f rot=%0.3f%s>" % (
                    self.__class__.__name__,
                    self.translation_x, self.translation_y,
                    self.scale, self.rotation,
                    (self.mirrored and " mirrored" or ""))


class DrawCursorMixin(object):
    """Mixin for renderer widgets needing a managed drawing cursor.

    Required members: self.doc, self.scale, gtk.Widget stuff.

    """


    def init_draw_cursor(self):
        """Initialize internal fields for DrawCursorMixin"""
        self._override_cursor = None
        self._first_map_cb_id = self.connect("map", self._first_map_cb)

    def _first_map_cb(self, widget, *a):
        """Updates the cursor on the first map"""
        assert self.get_window() is not None
        assert self.get_mapped()
        self.disconnect(self._first_map_cb_id)
        self._first_map_cb_id = None
        self.update_cursor()


    def update_cursor(self):
        # Callback for updating the cursor
        if not self.get_mapped():
            return
        window = self.get_window()
        app = self.app
        if window is None:
            logger.error("update_cursor: no window")
            return
        override_cursor = self._override_cursor
        if override_cursor is not None:
            c = override_cursor
        elif self.get_state() == gtk.STATE_INSENSITIVE:
            c = None
        elif self.doc is None:
            logger.error("update_cursor: no document")
            return
        elif self.doc.layer.locked or not self.doc.layer.visible:
            # Cursor to represent that one cannot draw.
            # Often a red circle with a diagonal bar through it.
            c = gdk.Cursor(gdk.CIRCLE)
        elif app is None:
            logger.error("update_cursor: no app")
            return
        # Last two cases only pertain to FreehandOnlyMode cursors.
        # XXX refactor: bad for separation of responsibilities, put the
        # special cases in the mode class.
        elif app.preferences.get("cursor.freehand.style",None) == 'crosshair':
            c = app.cursors.get_freehand_cursor()
        else:
            radius, style = self._get_cursor_info()
            c = cursor.get_brush_cursor(radius, style, self.app.preferences)
        window.set_cursor(c)


    def set_override_cursor(self, cursor):
        """Set a cursor which will always be used.

        Used by the colour picker. The override cursor will be used regardless
        of the criteria update_cursor() normally uses. Pass None to let it
        choose normally again.
        """
        self._override_cursor = cursor
        gobject.idle_add(self.update_cursor)


    def _get_cursor_info(self):
        """Return factors determining the cursor size and shape.
        """
        b = self.doc.brush.brushinfo
        base_radius = exp(b.get_base_value('radius_logarithmic'))
        r = base_radius
        r += 2 * base_radius * b.get_base_value('offset_by_random')
        r *= self.scale
        r += 0.5
        if b.is_eraser():
            style = cursor.BRUSH_CURSOR_STYLE_ERASER
        elif b.is_alpha_locked():
            style = cursor.BRUSH_CURSOR_STYLE_LOCK_ALPHA
        elif b.is_colorize():
            style = cursor.BRUSH_CURSOR_STYLE_COLORIZE
        else:
            style = cursor.BRUSH_CURSOR_STYLE_NORMAL
        return (r, style)


    def brush_modified_cb(self, settings):
        """Handles brush modifications: set up by the main TDW.
        """
        if settings & set(['radius_logarithmic', 'offset_by_random',
                           'eraser', 'lock_alpha', 'colorize']):
            # Reducing the number of updates is probably a good idea
            self.update_cursor()


def calculate_transformation_matrix(scale, rotation, translation_x, translation_y, mirrored):

    scale = scale
    # check if scale is almost a power of two
    scale_log2 = log(scale, 2)
    scale_log2_rounded = round(scale_log2)
    if abs(scale_log2-scale_log2_rounded) < 0.01:
        scale = 2.0**scale_log2_rounded

    # maybe we should check if rotation is almost a multiple of 90 degrees?

    matrix = cairo.Matrix()
    matrix.translate(translation_x, translation_y)
    matrix.rotate(rotation)
    matrix.scale(scale, scale)

    # Align the translation such that (0,0) maps to an integer
    # screen pixel, to keep image rendering fast and sharp.
    x, y = matrix.transform_point(0, 0)
    inverse = cairo.Matrix(*list(matrix))
    assert not inverse.invert()
    x, y = inverse.transform_point(round(x), round(y))
    matrix.translate(x, y)

    if mirrored:
        m = list(matrix)
        m[0] = -m[0]
        m[2] = -m[2]
        matrix = cairo.Matrix(*m)

    return matrix

class CanvasRenderer(gtk.DrawingArea, DrawCursorMixin):
    """Render the document model to screen.

    Can render the document in a transformed way, including translation,
    scaling and rotation."""

    def __init__(self, tdw):
        gtk.DrawingArea.__init__(self)
        self.init_draw_cursor()

        self.connect("draw", self.draw_cb)

        self.connect("state-changed", self.state_changed_cb)

        self._tdw = tdw

        self.visualize_rendering = False

        self.translation_x = 0.0
        self.translation_y = 0.0
        self.scale = 1.0
        self.rotation = 0.0
        self.mirrored = False
        self.cached_transformation_matrix = None

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
        self.snapshot_pixmap = None # FIXME: not used, see draw_cb()

        # Overlays
        self.model_overlays = []
        self.display_overlays = []

        # Pizelize at high zoom-ins.
        # The icon editor needs to be able to adjust this.
        self.pixelize_threshold = 2.8

    @property
    def app(self):
        return self._tdw.app


    @property
    def doc(self):
        return self._tdw.doc


    @event
    def transformation_updated(self):
        """Event: transformation was updated"""


    def _invalidate_cached_transform_matrix(self):
        self.cached_transformation_matrix = None

    def _get_x(self):
        return self._translation_x
    def _set_x(self, val):
        self._translation_x = val
        self._invalidate_cached_transform_matrix()
    translation_x = property(_get_x, _set_x)

    def _get_y(self):
        return self._translation_y
    def _set_y(self, val):
        self._translation_y = val
        self._invalidate_cached_transform_matrix()
    translation_y = property(_get_y, _set_y)

    def _get_scale(self):
        return self._scale
    def _set_scale(self, val):
        self._scale = val
        self._invalidate_cached_transform_matrix()
    scale = property(_get_scale, _set_scale)

    def _get_rotation(self):
        return self._rotation
    def _set_rotation(self, val):
        self._rotation = val
        self._invalidate_cached_transform_matrix()
    rotation = property(_get_rotation, _set_rotation)

    def _get_mirrored(self):
        return self._mirrored
    def _set_mirrored(self, val):
        self._mirrored = val
        self._invalidate_cached_transform_matrix()
    mirrored = property(_get_mirrored, _set_mirrored)


    def state_changed_cb(self, widget, oldstate):
        # Keeps track of the sensitivity state, and regenerates
        # the snapshot pixbuf on entering it.
        sensitive = self.get_state() != gtk.STATE_INSENSITIVE
        if sensitive:
            self.snapshot_pixmap = None
        else:
            if self.snapshot_pixmap is None:
                logger.debug("TODO: generate a static snapshot pixmap")
        self.is_sensitive = sensitive


    def canvas_modified_cb(self, x, y, w, h):
        if not self.get_window():
            return

        if w == 0 and h == 0:
            # Full redraw (used when background has changed).
            #logger.debug('Full redraw')
            self.queue_draw()
            return

        # Create an expose event with the event bbox rotated/zoomed.
        corners = [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]
        corners = [self.model_to_display(x, y) for (x, y) in corners]
        self.queue_draw_area(*helpers.rotated_rectangle_bbox(corners))

    def model_structure_changed_cb(self, doc):
        # Reflect layer locked and visible flag changes
        self.update_cursor()

    def draw_cb(self, widget, cr):
        #TODO: (GTK3 migration fallout)
        #  ...should display snapshot instead of normal content, I think
        #  (if it's only during loading, we could also just render blank instead?)
        if self.snapshot_pixmap:
            logger.debug("TODO: paint static snapshot pixmap")
        self.repaint(cr, None)
        return True

    def display_to_model(self, disp_x, disp_y):
        """Converts display coordinates to model coordinates.
        """
        matrix = cairo.Matrix(*self._get_model_view_transformation())
        assert not matrix.invert()
        view_model = matrix
        return view_model.transform_point(disp_x, disp_y)


    def model_to_display(self, model_x, model_y):
        """Converts model coordinates to display coordinates.
        """
        model_view = self._get_model_view_transformation()
        return model_view.transform_point(model_x, model_y)


    def _get_model_view_transformation(self):
        if self.cached_transformation_matrix is None:
            matrix = calculate_transformation_matrix(
                        self.scale, self.rotation,
                        self.translation_x, self.translation_y,
                        self.mirrored)
            self.cached_transformation_matrix = matrix
            self.transformation_updated()
        return self.cached_transformation_matrix


    def is_translation_only(self):
        return self.rotation == 0.0 and self.scale == 1.0 and not self.mirrored


    def get_cursor_in_model_coordinates(self):
        x, y = self.get_pointer()   # FIXME: deprecated in GTK3
        return self.display_to_model(x, y)


    def get_visible_layers(self):
        # FIXME: tileddrawwidget should not need to know whether the
        # model has layers
        if not self.doc:
            return []
        layers = self.doc.layers
        if not self.show_layers_above:
            layers = self.doc.layers[0:self.doc.layer_idx+1]
        layers = [l for l in layers if l.visible]
        return layers


    def repaint(self, cr, device_bbox=None):
        if not self.doc:
            cr.set_source_rgb(0, 0, 0)
            cr.paint()
            return
        transformation, surface, sparse, mipmap_level, clip_region = self.render_prepare(cr, device_bbox)
        self.render_execute(cr, transformation, surface, sparse, mipmap_level, clip_region)
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


    def render_get_clip_region(self, cr, device_bbox):
        # Could this be an alternative?
        #x0, y0, x1, y1 = cr.clip_extents()
        #sparse = True
        #clip_region = (x0, y0, x1-x0, y1-y0)
        #return clip_region, sparse

        # Get the area which needs to be updated, in device coordinates, and
        # determine whether the render is "sparse", [TODO: define what this
        # means]
        x, y, w, h = device_bbox
        cx, cy = x+w/2, y+h/2

        # As of 2012-07-08, Ubuntu Precise (LTS, unfortunately) and Debian
        # unstable(!) use python-cairo 1.8.8, which is too old to support
        # the cairo.Region return from Gdk.Window.get_clip_region() we
        # really need. They'll be important to support for a while, so we
        # have to use an inefficient workaround using the complete clip
        # rectangle for the update.
        #
        # http://stackoverflow.com/questions/6133622/
        # http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=653588
        # http://packages.ubuntu.com/python-cairo
        # http://packages.debian.org/python-cairo

        clip_exists, rect = gdk.cairo_get_clip_rectangle(cr)
        if clip_exists:
            # It's a wrapped cairo_rectangle_int_t, CairoRectangleInt
            area = (rect.x, rect.y, rect.width, rect.height)
            clip_region = area
            sparse = (cx < rect.x or cx > rect.x+rect.width
                      or cy < rect.y or cy > rect.y+rect.height)
        else:
            clip_region = None
            sparse = False

        return clip_region, sparse

    def tile_is_visible(self, tx, ty, transformation, clip_region, sparse, translation_only):
        if not sparse:
            return True

        # it is worth checking whether this tile really will be visible
        # (to speed up the L-shaped expose event during scrolling)
        # (speedup clearly visible; slowdown measurable when always executing this code)
        N = tiledsurface.N
        if translation_only:
            x, y = transformation.transform_point(tx*N, ty*N)
            bbox = (int(x), int(y), N, N)
        else:
            corners = [(tx*N, ty*N), ((tx+1)*N, ty*N), (tx*N, (ty+1)*N), ((tx+1)*N, (ty+1)*N)]
            corners = [transformation.transform_point(x_, y_) for (x_, y_) in corners]
            bbox = helpers.rotated_rectangle_bbox(corners)

        c_r = gdk.Rectangle()
        c_r.x, c_r.y, c_r.width, c_r.height = clip_region
        bb_r = gdk.Rectangle()
        bb_r.x, bb_r.y, bb_r.width, bb_r.height = bbox
        intersects, isect_r = gdk.rectangle_intersect(bb_r, c_r)
        return intersects


    def render_prepare(self, cr, device_bbox):
        if device_bbox is None:
            allocation = self.get_allocation()
            w, h = allocation.width, allocation.height
            device_bbox = (0, 0, w, h)
        # logger.debug('device bbox: %r', tuple(device_bbox))

        clip_region, sparse = self.render_get_clip_region(cr, device_bbox)
        x, y, w, h = device_bbox

        # fill it all white, though not required in the most common case
        if self.visualize_rendering:
            # grey
            tmp = random.random()
            cr.set_source_rgb(tmp, tmp, tmp)
            cr.paint()

        transformation = cairo.Matrix(*self._get_model_view_transformation())

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
        transformation.scale(2**mipmap_level, 2**mipmap_level)

        # bye bye device coordinates
        cr.save()   # >>>CONTEXT1
        cr.transform(transformation)
        cr.save()   # >>>CONTEXT2

        # calculate the final model bbox with all the clipping above
        x1, y1, x2, y2 = cr.clip_extents()
        if not self.is_translation_only():
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

        return transformation, surface, sparse, mipmap_level, clip_region

    def render_execute(self, cr, transformation, surface, sparse, mipmap_level, clip_region):
        translation_only = self.is_translation_only()
        model_bbox = surface.x, surface.y, surface.w, surface.h

        #logger.debug('model bbox: %r', model_bbox)

        # not sure if it is a good idea to clip so tightly
        # has no effect right now because device_bbox is always smaller
        cr.rectangle(*model_bbox)
        cr.clip()

        layers = self.get_visible_layers()

        if self.visualize_rendering:
            surface.pixbuf.fill((int(random.random()*0xff)<<16)+0x00000000)

        background = None
        if self.current_layer_solo:
            background = self.neutral_background_pixbuf
            layers = [self.doc.layer]
            # this is for hiding instead
            #layers.pop(self.doc.layer_idx)
        if self.overlay_layer:
            idx = layers.index(self.doc.layer)
            layers.insert(idx+1, self.overlay_layer)

        # Composite
        tiles = []
        for tx, ty in surface.get_tiles():
            if self.tile_is_visible(tx, ty, transformation, clip_region, sparse, translation_only):
                tiles.append((tx, ty))
        self.doc.render_into(surface, tiles, mipmap_level, layers, background)

        # The speedup below worked for GTK2, is there is an equivalent for GTK3?
        #if translation_only:
        #    # not sure why, but using gdk directly is notably faster than the same via cairo
        #    x, y = self.model_to_display(surface.x, surface.y)
        #    self.window.draw_pixbuf(None, surface.pixbuf, 0, 0, int(x), int(y),
        #                            dither=gdk.RGB_DITHER_MAX)

        #logger.debug('Position (screen coordinates): %r', cr.model_to_display(surface.x, surface.y))
        gdk.cairo_set_source_pixbuf(cr, surface.pixbuf,
                                    round(surface.x), round(surface.y))
        pattern = cr.get_source()

        # We could set interpolation mode here (eg nearest neighbour)
        #pattern.set_filter(cairo.FILTER_NEAREST)  # 1.6s
        #pattern.set_filter(cairo.FILTER_FAST)     # 2.0s
        #pattern.set_filter(cairo.FILTER_GOOD)     # 3.1s
        #pattern.set_filter(cairo.FILTER_BEST)     # 3.1s
        #pattern.set_filter(cairo.FILTER_BILINEAR) # 3.1s

        # Pixelize at high zoom-in levels
        if self.scale > self.pixelize_threshold:
            pattern.set_filter(cairo.FILTER_NEAREST)

        cr.paint()

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
            # annoyingly non-constant framerate otherwise.
            #
            # It might be worth it if it was done only once per
            # redraw, instead of once per motion event. Maybe try to
            # implement something like "queue_scroll" with priority
            # similar to redraw? (The GTK commit responsible for bug
            # http://bugzilla.gnome.org/show_bug.cgi?id=702392 might
            # solve this problem, I think.)
            self.window.scroll(int(-dx), int(-dy))
        else:
            self.queue_draw()


    def get_center(self):
        """Return the center position in display coordinates.
        """
        alloc = self.get_allocation()
        return alloc.width/2.0, alloc.height/2.0


    def get_center_model_coords(self):
        """Return the center position in model coordinates.
        """
        center = self.get_center()
        return self.display_to_model(*center)


    def recenter_document(self):
        """Recentres the view onto the document's centre.
        """
        x, y, w, h = self.doc.get_effective_bbox()
        cx = x+w/2.0
        cy = y+h/2.0
        self.recenter_on_model_coords(cx, cy)


    def recenter_on_model_coords(self, cx, cy):
        """Recentres the view to a specified point, in model coordinates.
        """
        dcx, dcy = self.model_to_display(cx, cy)
        self.recenter_on_display_coords(dcx, dcy)


    def recenter_on_display_coords(self, cx, cy):
        current_cx, current_cy = self.get_center()
        self.translation_x += current_cx - cx
        self.translation_y += current_cy - cy
        self.queue_draw()


    def toggle_show_layers_above(self):
        self.show_layers_above = not self.show_layers_above
        self.queue_draw()


def _make_testbed_model():
    import lib.brush, lib.document
    brush = lib.brush.BrushInfo()
    brush.load_defaults()
    return lib.document.Document(brush)


def _test():
    from document import CanvasController
    from canvasevent import FreehandOnlyMode
    model = _make_testbed_model()
    tdw = TiledDrawWidget()
    tdw.set_model(model)
    tdw.set_size_request(640, 480)
    tdw.renderer.visualize_rendering = True
    ctrlr = CanvasController(tdw)
    ctrlr.init_pointer_events()
    ctrlr.modes.default_mode_class = FreehandOnlyMode
    win = gtk.Window()
    win.set_title("tdw test")
    win.connect("destroy", lambda *a: gtk.main_quit())
    win.add(tdw)
    win.show_all()
    gtk.main()


if __name__ == '__main__':
    _test()
