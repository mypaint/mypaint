# This file is part of MyPaint.
# Copyright (C) 2008-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2008-2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


## Imports

from __future__ import division, print_function
import random
from math import floor, ceil, log, exp
import math
import weakref
import contextlib
import logging

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib
import cairo
import numpy as np

from lib import helpers, tiledsurface, pixbufsurface
from lib.observable import event
import lib.layer
from . import cursor
from .drawutils import render_checks
from .windowing import clear_focus
import gui.style
import lib.color
import lib.alg
from lib.pycompat import xrange

logger = logging.getLogger(__name__)

## Class definitions


class TiledDrawWidget (Gtk.EventBox):
    """Widget for showing a lib.document.Document

    Rendering is delegated to a dedicated class: see `CanvasRenderer`.

    """

    ## Register a GType name for Glade, GtkBuilder etc.
    __gtype_name__ = "TiledDrawWidget"

    # List of weakrefs to all known TDWs.
    __tdw_refs = []

    @classmethod
    def get_active_tdw(kin):  # noqa: N804
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

    @classmethod
    def get_visible_tdws(cls):
        """Iterates across all mapped and visible TDWs"""
        for tdw_ref in cls.__tdw_refs:
            tdw = tdw_ref()
            if tdw and tdw.get_window() and tdw.get_visible():
                yield tdw

    @classmethod
    def get_tdw_under_device(cls, device):
        """Get the TDW directly under a device's pointer

        :param Gdk.Device device: the device to look up
        :rtype: tuple
        :returns: (tdw, x, y)

        This classmethod returns
        the TDW directly under the master pointer for a given device,
        and the pointer's coordinates
        relative to the top-left of that TDW's window.
        It is intended for use in picker code,
        and within the kinds of device-specific pointer grabs
        which the lib.picker presenters establish.
        Most pointer event handling code doesn't need to call this.

        If no known tdw is under the device's position,
        the tuple `(None, -1, -1)` is returned.

        """
        # get_last_event_window() does not work in pointer grabs.
        #  dev_win = device.get_last_event_window()
        # But we want *under* the pointer semantics anyway, and
        # get_window_at_position() behaves itself in grabs. Hopefully.
        dev_win, win_x, win_y = device.get_window_at_position()
        if dev_win is None:
            return (None, -1, -1)
        # The window returned is that of the tdw's renderer, i.e. that
        # of the tdw itself under the current working assumptions.
        for tdw in cls.get_visible_tdws():
            tdw_win = tdw.get_window()
            assert tdw_win is tdw.renderer.get_window(), (
                "Picking will break in this packing configuration "
                "because the tdw's event window is not the same as "
                "its descendent renderer's window."
            )
            if tdw_win is dev_win:
                return (tdw, win_x, win_y)
        return (None, -1, -1)

    def __init__(self, idle_redraw_priority=None):
        """Instantiate a TiledDrawWidget.

        """
        super(TiledDrawWidget, self).__init__()

        if __name__ == '__main__':
            app = None
        else:
            from . import application
            app = application.get_app()
        self.app = app
        self.doc = None
        self.last_tdw_event_info = None

        self.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK
        )
        # Support smooth scrolling unless configured not to
        if app and app.preferences.get("ui.support_smooth_scrolling", True):
            self.add_events(Gdk.EventMask.SMOOTH_SCROLL_MASK)

        self.last_painting_pos = None

        self.renderer = CanvasRenderer(
            self,
            idle_redraw_priority = idle_redraw_priority,
        )
        self.add(self.renderer)
        self.renderer.update_cursor()  # get the initial cursor right

        self.add_events(Gdk.EventMask.ENTER_NOTIFY_MASK)
        self.connect("enter-notify-event", self.enter_notify_cb)
        self.__tdw_refs.insert(0, weakref.ref(self))

        self._last_alloc_pos = (0, 0)
        self.connect("size-allocate", self._size_allocate_cb)
        self.connect("realize", self._realize_cb)

        #: Scroll to match appearing/disappearing sidebars and toolbars.
        self.scroll_on_allocate = True

        forwarder = self._announce_transformation_updated
        self.renderer.transformation_updated += forwarder

    def _announce_transformation_updated(self, *args):
        self.transformation_updated()

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
            self.renderer.scroll(dx, dy, ongoing=False)

    def _realize_cb(self, widget):
        logger.debug("Turning off event compression for %r's window", widget)
        win = widget.get_window()
        win.set_event_compression(False)

    def set_model(self, model):
        assert self.doc is None
        renderer = self.renderer
        model.canvas_area_modified += renderer.canvas_modified_cb
        root = model.layer_stack
        root.current_path_updated += renderer.current_layer_changed_cb
        root.layer_properties_changed += renderer.layer_props_changed_cb
        model.brush.brushinfo.observers.append(renderer.brush_modified_cb)
        model.frame_enabled_changed += renderer.frame_enabled_changed_cb
        model.frame_updated += renderer.frame_updated_cb
        self.doc = model
        self.renderer.queue_draw()

    def enter_notify_cb(self, widget, event):
        clear_focus(widget.get_toplevel())
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

    def get_pointer_in_model_coordinates(self):
        """Returns the pointer/cursor location in model coords.

        :returns: core pointer's position, as a model-relative (x, y)
        :rtype: tuple

        This should only be used in action callbacks, never for events.

        """
        win = self.get_window()
        display = self.get_display()
        devmgr = display and display.get_device_manager() or None
        coredev = devmgr and devmgr.get_client_pointer() or None
        if coredev and win:
            win_, x, y, mods = win.get_device_position_double(coredev)
            return self.display_to_model(x, y)
        else:
            return (0., 0.)

    @property
    def scroll(self):
        return self.renderer.scroll

    @property
    def get_center(self):
        return self.renderer.get_center

    @property
    def get_center_model_coords(self):
        return self.renderer.get_center_model_coords

    def get_corners_model_coords(self):
        """Returns the viewport corners in model coordinates.

        :returns: Corners [TL, TR, BR, BL] as (x, y) pairs of floats.
        :rtype: list

        See also lib.helpers.rotated_rectangle_bbox() if you need to
        turn this into a bounding box in model-space.

        """
        alloc = self.get_allocation()
        x = alloc.x
        y = alloc.y
        w = alloc.width
        h = alloc.height
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        corners = [self.display_to_model(*p) for p in corners]
        return corners

    @property
    def recenter_on_model_coords(self):
        return self.renderer.recenter_on_model_coords

    @property
    def pick_color(self):
        return self.renderer.pick_color

    @property
    def queue_draw_area(self):
        return self.renderer.queue_draw_area

    # Transform logic

    @contextlib.contextmanager
    def _fixed_center(self, center=None, ongoing=True):
        """Keep a fixed center when zoom or rotation changes

        :param tuple center: Center of the rotation, display (X, Y)
        :param bool ongoing: Hint that this is in an ongoing change

        This context manager's cleanup phase applies a corrective
        transform which keeps the specified center in the same position
        on the screen. If the center isn't specified, the center pixel
        of the widget itself is used.

        It also queues a redraw.

        """
        # Determine the requested (or default) center in model space
        cx = None
        cy = None
        if center is not None:
            cx, cy = center
        if cx is None or cy is None:
            cx, cy = self.renderer.get_center()
        cx_model, cy_model = self.renderer.display_to_model(cx, cy)
        # Execute the changes in the body of the with statement
        yield
        # Corrective transform (& limits)
        self.renderer.scale = helpers.clamp(self.renderer.scale,
                                            self.renderer.zoom_min,
                                            self.renderer.zoom_max)
        cx_new, cy_new = self.renderer.model_to_display(cx_model, cy_model)
        self.renderer.translation_x += cx - cx_new
        self.renderer.translation_y += cy - cy_new
        # Redraw handling
        if ongoing:
            self.renderer.defer_hq_rendering()
        self.renderer.queue_draw()

    def zoom(self, zoom_step, center=None, ongoing=True):
        """Multiply the current scale factor"""
        with self._fixed_center(center, ongoing):
            self.renderer.scale *= zoom_step

    def set_zoom(self, zoom, center=None, ongoing=False):
        """Set the zoom to an exact value"""
        with self._fixed_center(center, ongoing):
            self.renderer.scale = zoom
        self.renderer.update_cursor()

    def rotate(self, angle_step, center=None, ongoing=True):
        """Rotate the view by a step"""
        if self.renderer.mirrored:
            angle_step = -angle_step
        with self._fixed_center(center, ongoing):
            self.renderer.rotation += angle_step

    def set_rotation(self, angle, ongoing=False):
        """Set the rotation to an exact value"""
        if self.renderer.mirrored:
            angle = -angle
        with self._fixed_center(None, ongoing):
            self.renderer.rotation = angle

    def mirror(self):
        with self._fixed_center(None, False):
            self.renderer.mirrored = not self.renderer.mirrored

    def set_mirrored(self, mirrored):
        """Set mirroring to a discrete state"""
        with self._fixed_center(None, False):
            self.renderer.mirrored = mirrored

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

    def get_move_cursor_name_for_edge(self, cursor_pos, edge_p1, edge_p2,
                                      tolerance=5, finite=True):
        """Get move cursor & detect hits on a line between two points

        :param tuple cursor_pos: cursor position, as display (x, y)
        :param tuple edge_p1: point on the edge, as model (x, y)
        :param tuple edge_p2: point on the edge, as model (x, y)
        :param int tolerance: slack for cursor pos., in display pixels
        :param bool finite: if false, the edge extends beyond p1, p2
        :returns: move direction cursor string, or None
        :rtype: str

        This can be used by special input modes when resizing objects on
        screen, for example frame edges and the symmetry axis.

        If the returned cursor isn't None, its value is the name of the
        most appropriate move cursor to use during the move. This method
        does not return a normal vector in model space; you'll have to
        calculate that for yourself.

        See also:
        * gui.cursor.Name (naming consts for cursors)

        """
        if finite:
            nearest_point = lib.alg.nearest_point_on_segment
        else:
            nearest_point = lib.alg.nearest_point_on_line
        x0, y0 = cursor_pos
        p1 = self.model_to_display(*edge_p1)
        p2 = self.model_to_display(*edge_p2)
        closest = nearest_point(p1, p2, cursor_pos)
        if closest:
            x1, y1 = closest
            if (x0 - x1)**2 + (y0 - y1)**2 > tolerance**2:
                return None
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            # Cursor name by angle - closest to being perpendicular to edge.
            edge_angle_perp = math.atan2(-dy, dx) + math.pi / 2
            return cursor.get_move_cursor_name_for_angle(edge_angle_perp)


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
            (self.mirrored and " mirrored" or "")
        )


class DrawCursorMixin(object):
    """Mixin for renderer widgets needing a managed drawing cursor.

    Required members: self.doc, self.scale, Gtk.Widget stuff.

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
        layer = self.doc._layers.current
        if self._insensitive_state_content:
            c = None
        elif override_cursor is not None:
            c = override_cursor
        elif self.doc is None:
            logger.error("update_cursor: no document")
            return
        elif not layer.get_paintable():
            # Cursor to represent that one cannot draw.
            # Often a red circle with a diagonal bar through it.
            c = Gdk.Cursor.new_for_display(
                window.get_display(), Gdk.CursorType.CIRCLE)
        elif app is None:
            logger.error("update_cursor: no app")
            return
        # Last two cases only pertain to FreehandMode cursors.
        # XXX refactor: bad for separation of responsibilities, put the
        # special cases in the mode class.
        elif app.preferences.get("cursor.freehand.style", None) == 'crosshair':
            c = app.cursors.get_freehand_cursor()
        else:
            radius, style = self._get_cursor_info()
            c = cursor.get_brush_cursor(radius, style, self.app.preferences)
        window.set_cursor(c)

    def set_override_cursor(self, cursor):
        """Set a cursor which will always be used.

        Used by the color picker. The override cursor will be used regardless
        of the criteria update_cursor() normally uses. Pass None to let it
        choose normally again.
        """
        self._override_cursor = cursor
        GLib.idle_add(self.update_cursor)

    def _get_cursor_info(self):
        """Return factors determining the cursor size and shape.
        """
        b = self.doc.brush.brushinfo
        r = b.get_visual_radius() * self.scale + 0.5
        if b.is_eraser():
            style = cursor.BRUSH_CURSOR_STYLE_ERASER
        elif b.is_alpha_locked():
            style = cursor.BRUSH_CURSOR_STYLE_LOCK_ALPHA
        elif b.is_colorize():
            style = cursor.BRUSH_CURSOR_STYLE_COLORIZE
        else:
            style = cursor.BRUSH_CURSOR_STYLE_NORMAL
        return r, style

    def brush_modified_cb(self, settings):
        """Handles brush modifications: set up by the main TDW.
        """
        if settings & set(['radius_logarithmic', 'offset_by_random',
                           'eraser', 'lock_alpha', 'colorize']):
            # Reducing the number of updates is probably a good idea
            self.update_cursor()


def calculate_transformation_matrix(scale, rotation,
                                    translation_x, translation_y,
                                    mirrored):
    scale = scale
    # check if scale is almost a power of two
    scale_log2 = log(scale, 2)
    scale_log2_rounded = round(scale_log2)
    if abs(scale_log2 - scale_log2_rounded) < 0.01:
        scale = 2.0 ** scale_log2_rounded

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


class CanvasRenderer (Gtk.DrawingArea, DrawCursorMixin):
    """Render the document model to screen.

    Can render the document in a transformed way, including translation,
    scaling and rotation.

    """

    ## Method defs

    def __init__(self, tdw, idle_redraw_priority=None):
        super(CanvasRenderer, self).__init__()
        self.init_draw_cursor()

        self.connect("draw", self._draw_cb)
        self._idle_redraw_priority = idle_redraw_priority
        self._idle_redraw_queue = []
        self._idle_redraw_src_id = None

        self.connect("state-changed", self._state_changed_cb)

        self._tdw = tdw

        # Currently we don't need a window while we're packed into the
        # TiledDrawWidget (a GtkEventBox subclass).
        # Capturing events via a separate EventBox separates concerns
        # and theoretically makes this the renderer component pluggable.
        # GtkDrawingArea's PyGI implementation still uses no_window,
        # unlike other GI flavours, and unlike C-style gtk. We want the
        # standard behaviour though.
        self.set_has_window(False)

        self.visualize_rendering = False

        self.translation_x = 0.0
        self.translation_y = 0.0
        self.scale = 1.0
        self.rotation = 0.0
        self.mirrored = False
        self.cached_transformation_matrix = None
        self.display_filter = None

        self.overlay_layer = None

        # gets overwritten for the main window
        self.zoom_max = 5.0
        self.zoom_min = 1 / 5.0

        # Sensitivity; we draw via a cached snapshot while the widget is
        # insensitive. tdws are generally only insensitive during loading and
        # saving, and because we now process the GTK main loop during loading
        # and saving, we need to avoid drawing partially-loaded files.
        self._insensitive_state_content = None

        # Overlays
        self.model_overlays = []
        self.display_overlays = []

        # Pixelize at high zoom-ins.
        # The icon editor needs to be able to adjust this.
        self.pixelize_threshold = 2.8

        # Backgroundless rendering
        self._real_alpha_check_pattern = None
        self._fake_alpha_check_tile = None
        self._init_alpha_checks()

        # Higher-quality mipmap choice
        # Turn off for a speedup during dragging or scrolling
        self._hq_rendering = True
        self._restore_hq_rendering_timeout_id = None

        self.connect("configure-event", self._configure_event_cb)

    def _init_alpha_checks(self):
        """Initialize the alpha check backgrounds"""
        # Real: checkerboard pattern, rendered via Cairo
        assert tiledsurface.N % gui.style.ALPHA_CHECK_SIZE == 0
        n = tiledsurface.N
        size = gui.style.ALPHA_CHECK_SIZE
        nchecks = int(n // size)
        cairo_surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, n, n)
        cr = cairo.Context(cairo_surf)
        render_checks(cr, size, nchecks)
        cairo_surf.flush()
        # Real: MyPaint background surface for layers-but-no-bg rendering
        pattern = cairo.SurfacePattern(cairo_surf)
        pattern.set_extend(cairo.EXTEND_REPEAT)
        self._real_alpha_check_pattern = pattern
        # Fake: faster rendering, but ugly
        tile = np.empty((n, n, 4), dtype='uint16')
        f = 1 << 15
        col1 = [int(f * c) for c in gui.style.ALPHA_CHECK_COLOR_1] + [f]
        col2 = [int(f * c) for c in gui.style.ALPHA_CHECK_COLOR_2] + [f]
        tile[:] = col1
        for i in xrange(nchecks):
            for j in xrange(nchecks):
                if (i + j) % 2 == 0:
                    continue
                ia, ib = (i * size), ((i + 1) * size)
                ja, jb = (j * size), ((j + 1) * size)
                tile[ia:ib, ja:jb] = col2
        self._fake_alpha_check_tile = tile

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

    def _state_changed_cb(self, widget, oldstate):
        """Handle the sensitivity state changing

        Saving and loading images toggles the sensitivity state on all
        toplevel windows. This causes a state shift on the TDW too.
        While the TDW is insensitive, its cursor is updated to respect
        the toplevel's cursor (typically a watch or an hourglass or
        something).

        """
        insensitive = self.get_state_flags() & Gtk.StateFlags.INSENSITIVE
        if insensitive and (not self._insensitive_state_content):
            alloc = widget.get_allocation()
            w = alloc.width
            h = alloc.height
            surface = self._new_image_surface_from_visible_area(0, 0, w, h)
            self._insensitive_state_content = surface
        elif (not insensitive) and self._insensitive_state_content:
            self._insensitive_state_content = None
        self.update_cursor()

    ## Redrawing

    def canvas_modified_cb(self, model, x, y, w, h):
        """Handles area redraw notifications from the underlying model"""

        if self._insensitive_state_content:
            return False

        if not self.get_window():
            return

        if w == 0 and h == 0:
            # Full redraw (used when background has changed).
            # logger.debug('Full redraw')
            self.queue_draw()
            return

        # Create an expose event with the event bbox rotated/zoomed.
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        corners = [self.model_to_display(x, y) for (x, y) in corners]
        bbox = helpers.rotated_rectangle_bbox(corners)
        self.queue_draw_area(*bbox)

    def queue_draw(self):
        if self._idle_redraw_priority is None:
            super(CanvasRenderer, self).queue_draw()
            return
        self._queue_idle_redraw(None)

    def queue_draw_area(self, x, y, w, h):
        if self._idle_redraw_priority is None:
            super(CanvasRenderer, self).queue_draw_area(x, y, w, h)
            return
        bbox = helpers.Rect(x, y, w, h)
        self._queue_idle_redraw(bbox)

    def _queue_idle_redraw(self, bbox):
        queue = self._idle_redraw_queue
        if bbox is None:
            queue[:] = []
        elif None in queue:
            return
        else:
            queue[:] = [b for b in queue if not bbox.contains(b)]
            for b in queue:
                if b.contains(bbox):
                    return
        queue.append(bbox)
        if self._idle_redraw_src_id is not None:
            return
        src_id = GLib.idle_add(
            self._idle_redraw_cb,
            priority = self._idle_redraw_priority,
        )
        self._idle_redraw_src_id = src_id

    def _idle_redraw_cb(self):
        assert self._idle_redraw_src_id is not None
        queue = self._idle_redraw_queue
        if len(queue) > 0:
            bbox = queue.pop(0)
            if bbox is None:
                super(CanvasRenderer, self).queue_draw()
            else:
                super(CanvasRenderer, self).queue_draw_area(*bbox)
        if len(queue) == 0:
            self._idle_redraw_src_id = None
            return False
        return True

    ## Redraw events

    def current_layer_changed_cb(self, rootstack, path):
        self.update_cursor()

    def layer_props_changed_cb(self, rootstack, path, layer, changed):
        self.update_cursor()

    def frame_enabled_changed_cb(self, model, enabled):
        self.queue_draw()

    def frame_updated_cb(self, model, old_frame, new_frame):
        pass
        # self.queue_draw()

    ## Transformations and coords

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
            scale_factor = self.get_scale_factor()
            # HiDPI: logical "device" (widget) pixels to screen pixels
            matrix = calculate_transformation_matrix(
                self.scale / scale_factor, self.rotation,
                self.translation_x, self.translation_y,
                self.mirrored
            )
            self.cached_transformation_matrix = matrix
            self.transformation_updated()
        return self.cached_transformation_matrix

    def _configure_event_cb(self, widget, event):
        # The docs say to handle this on the toplevel to be notified of
        # HiDPI scale_factor changes. Not sure yet whether this will
        # work down at this level - I've no fancy HiDPI hardware to test
        # with. For now, just try this.
        logger.debug("configure-event received. Invalidating transform")
        self._invalidate_cached_transform_matrix()
        self.queue_draw()

    def is_translation_only(self):
        return self.rotation == 0.0 and self.scale == 1.0 and not self.mirrored

    def pick_color(self, x, y, size=3):
        """Picks the rendered colour at a particular point.

        :param int x: X coord of pixel to pick (widget/device coords)
        :param int y: Y coord of pixel to pick (widget/device coords)
        :param int size: Size of the sampling square.
        :returns: The colour sampled.
        :rtype: lib.color.UIColor

        This method operates by rendering part of the document using the
        current settings, then averaging the colour values of the pixels
        within the sampling square.

        """
        # TODO: the ability to turn *off* this kind of "sample merged".
        # Ref: https://github.com/mypaint/mypaint/issues/333

        # Make a square surface for the sample.
        size = max(1, int(size))

        # Extract a square Cairo surface containing the area to sample
        r = int(size // 2)
        x = int(x) - r
        y = int(y) - r
        surf = self._new_image_surface_from_visible_area(
            x, y,
            size, size,
            use_filter = False,
        )

        # Extract a pixbuf, then an average color.
        # surf.write_to_png("/tmp/grab.png")
        pixbuf = Gdk.pixbuf_get_from_surface(surf, 0, 0, size, size)
        color = lib.color.UIColor.new_from_pixbuf_average(pixbuf)
        return color

    def _new_image_surface_from_visible_area(self, x, y, w, h,
                                             use_filter=True):
        """Render part of the doc to a new cairo image surface, as seen.

        :param int x: Rectangle left edge (widget/device coords)
        :param int y: Rectangle top edge (widget/device coords)
        :param int w: Rectangle width (widget/device coords)
        :param int h: Rectangle height (widget/device coords)
        :param bool use_filter: Apply display filters to rendering.
        :rtype: cairo.ImageSurface
        :returns: A rendered of the document, as seen on screen

        Creates and returns a new cairo.ImageSurface of the given size,
        containing an image of the document as it would appears on
        screen within the given rectangle. The area to extract is a
        rectangle in display (widget) coordinates, but it doesn't
        actually have to be within the visible area. Used for
        snapshotting and sampling colours.

        """

        # Start with a clean black slate.
        x = int(x)
        y = int(y)
        w = max(1, int(w))
        h = max(1, int(h))
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        cr = cairo.Context(surf)
        cr.set_source_rgb(0, 0, 0)
        cr.paint()

        # The rendering routines used are those used by the normal draw
        # handler, so we need an offset before calling them.
        cr.translate(-x, -y)

        # Paint checkerboard if we won't be rendering an opaque background
        model = self.doc
        render_is_opaque = model and model.layer_stack.get_render_is_opaque()
        if (not render_is_opaque) and self._draw_real_alpha_checks:
            cr.set_source(self._real_alpha_check_pattern)
            cr.paint()
        if not model:
            return surf

        # Render just what we need.
        transformation, surface, sparse, mipmap_level, clip_rect = \
            self._render_prepare(cr)
        display_filter = None
        if use_filter:
            display_filter = self.display_filter
        self._render_execute(
            cr,
            transformation,
            surface,
            sparse,
            mipmap_level,
            clip_rect,
            filter = display_filter,
        )
        surf.flush()
        return surf

    @property
    def _draw_real_alpha_checks(self):
        if not self.app:
            return True
        return self.app.preferences["view.real_alpha_checks"]

    def _draw_cb(self, widget, cr):
        """Draw handler"""

        # Don't render any partial views of the document if the widget
        # isn't sensitive to user input. If we don't do this, loading a
        # big doc might show each layer individually as it loads.
        if self._insensitive_state_content:
            cr.set_source_surface(self._insensitive_state_content, 0, 0)
            cr.paint()
            return True

        # Paint checkerboard if we won't be rendering an opaque background
        model = self.doc
        render_is_opaque = model and model.layer_stack.get_render_is_opaque()
        if (not render_is_opaque) and self._draw_real_alpha_checks:
            cr.set_source(self._real_alpha_check_pattern)
            cr.paint()
        if not model:
            return True

        # Paint a random grey behind what we're about to render
        # if visualization is needed.
        if self.visualize_rendering:
            tmp = random.random()
            cr.set_source_rgb(tmp, tmp, tmp)
            cr.paint()

        # Prep a pixbuf-surface aligned to the model to render into.
        # This also applies the transformation.
        transformation, surface, sparse, mipmap_level, clip_rect = \
            self._render_prepare(cr)

        # not sure if it is a good idea to clip so tightly
        # has no effect right now because device_bbox is always smaller
        model_bbox = surface.x, surface.y, surface.w, surface.h
        cr.rectangle(*model_bbox)
        cr.clip()

        # Clear the pixbuf to be rendered with a random red,
        # to make it apparent if something is not being painted.
        if self.visualize_rendering:
            surface.pixbuf.fill(int(random.random() * 0xff) << 16)

        # Render to the pixbuf, then paint it.
        self._render_execute(
            cr,
            transformation,
            surface,
            sparse,
            mipmap_level,
            clip_rect,
            filter = self.display_filter,
        )

        # Using different random blues helps make one rendered bbox
        # distinct from the next when the user is painting.
        if self.visualize_rendering:
            cr.set_source_rgba(0, 0, random.random(), 0.4)
            cr.paint()

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

        return True

    def _render_get_clip_region(self, cr, device_bbox):
        """Get the area that needs to be updated, in device coords.

        Called when handling "draw" events.  This uses Cairo's clip
        region, which ultimately derives from the areas sent by
        lib.document.Document.canvas_area_modified().  These can be
        small or large if the update comes form the user drawing
        something. When panning or zooming the canvas, it's a full
        redraw, and the area corresponds to the entire display.

        :param cairo.Context cr: as passed to the "draw" event handler.
        :param tuple device_bbox: (x,y,w,h) widget extents
        :returns: (clipregion, sparse)
        :rtype: tuple

        The clip region return value is a lib.helpers.Rect containing
        the area to redraw in display coordinates, or None.

        This also determines whether the redraw is "sparse", meaning
        that the clip region returned does not contain the centre of the
        device bbox.

        """

        # Could this be an alternative?
        # x0, y0, x1, y1 = cr.clip_extents()
        # sparse = True
        # clip_region = (x0, y0, x1-x0, y1-y0)
        # return clip_region, sparse

        # Get the area which needs to be updated, in device coordinates, and
        # determine whether the render is "sparse", [TODO: define what this
        # means]
        x, y, w, h = device_bbox
        cx, cy = x + w // 2, y + h // 2

        # As of 2012-07-08, Ubuntu Precise (LTS, unfortunately) and Debian
        # unstable(!) use python-cairo 1.8.8, which does not support
        # the cairo.Region return from Gdk.Window.get_clip_region() we
        # really need. They'll be important to support for a while, so we
        # have to use an inefficient workaround using the complete clip
        # rectangle for the update.
        #
        # http://stackoverflow.com/questions/6133622/
        # http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=653588
        # http://packages.ubuntu.com/python-cairo
        # http://packages.debian.org/python-cairo
        #
        # Update 2015-08-14: actually, on Debian-derived systems you
        # need gir1.2-freedesktop 1.44.0, which contains a wrapping for
        # some cairoish return values from GTK/GDK, including
        # CairoRegion: cairo-1.0.typelib. On MSYS2, ensure you have
        # mingw-w64-i686-gobject-introspection-runtime or its x86_64
        # equivalent installed, also at version 1.44.

        clip_exists, rect = Gdk.cairo_get_clip_rectangle(cr)
        if clip_exists:
            # It's a wrapped cairo_rectangle_int_t, CairoRectangleInt
            # Convert to a better representation for our purposes,
            # noting https://github.com/mypaint/mypaint/issues/433
            rect = helpers.Rect(rect.x, rect.y, rect.width, rect.height)
            sparse = (
                cx < rect.x
                or cx > (rect.x + rect.w)
                or cy < rect.y
                or cy > (rect.y + rect.h)
            )
        else:
            rect = None
            sparse = False

        return rect, sparse

    def _tile_is_visible(self, tx, ty, transformation, clip_rect,
                         translation_only):
        """Tests whether an individual tile is visible.

        This is sometimes worth doing during rendering, but not always.
        Currently we use _render_get_clip_region()'s sparse flag to
        determine whether this is necessary. The original logic for
        this function was documented as...

        > it is worth checking whether this tile really will be visible
        > (to speed up the L-shaped expose event during scrolling)
        > (speedup clearly visible; slowdown measurable when always
        > executing this code)

        I'm not 100% certain that GTK3 does panning redraws this way,
        so perhaps this method is uneccessary for those?
        However this method is always used when rendering during
        painting, or other activities that send partial updates.

        """
        n = tiledsurface.N
        if translation_only:
            x, y = transformation.transform_point(tx * n, ty * n)
            bbox = (int(x), int(y), n, n)
        else:
            corners = [
                (tx * n, ty * n),
                ((tx + 1) * n, ty * n),
                (tx * n, (ty + 1) * n),
                ((tx + 1) * n, (ty + 1) * n),
            ]
            corners = [
                transformation.transform_point(x_, y_)
                for (x_, y_) in corners
            ]
            bbox = helpers.rotated_rectangle_bbox(corners)
        tile_rect = helpers.Rect(*bbox)
        return clip_rect.overlaps(tile_rect)

    def _render_prepare(self, cr):
        """Prepares a blank pixbuf & other details for later rendering.

        Called when handling "draw" events. The size and shape of the
        returned pixbuf (wrapped in a tile-accessible and read/write
        lib.pixbufsurface.Surface) is determined by the Cairo clipping
        region that expresses what we've been asked to redraw, and by
        the TDW's own view transformation of the document.

        """
        # Determine what to draw, and the nature of the reveal.
        allocation = self.get_allocation()
        w, h = allocation.width, allocation.height
        device_bbox = (0, 0, w, h)
        clip_rect, sparse = self._render_get_clip_region(cr, device_bbox)
        x, y, w, h = device_bbox

        # Use a copy of the cached translation matrix for this
        # rendering. It'll need scaling if using a mipmap level
        # greater than zero.
        transformation = cairo.Matrix(*self._get_model_view_transformation())

        # HQ rendering causes a very clear slowdown on some hardware.
        # Probably could avoid this entirely by rendering differently,
        # but for now, if the canvas is being panned around,
        # just render more simply.
        if self._hq_rendering:
            mipmap_level = max(0, int(floor(log(1 / self.scale, 2))))
        else:
            mipmap_level = max(0, int(ceil(log(1 / self.scale, 2))))

        # OPTIMIZE: If we would render tile scanlines,
        # OPTIMIZE:  we could probably use the better one above...
        mipmap_level = min(mipmap_level, tiledsurface.MAX_MIPMAP_LEVEL)
        transformation.scale(2**mipmap_level, 2**mipmap_level)

        # bye bye device coordinates
        cr.save()   # >>>CONTEXT1
        cr.transform(transformation)
        cr.save()   # >>>CONTEXT2

        # calculate the final model bbox with all the clipping above
        x1, y1, x2, y2 = cr.clip_extents()
        if not self.is_translation_only():
            # Looks like cairo needs one extra pixel rendered for
            # interpolation at the border. If we don't do this, we get dark
            # stripe artifacts when panning while zoomed.
            x1 -= 1
            y1 -= 1
            x2 += 1
            y2 += 1
        x1, y1 = int(floor(x1)), int(floor(y1))
        x2, y2 = int(ceil(x2)), int(ceil(y2))

        # We always render with alpha to get hardware acceleration,
        # even when we could avoid using the alpha channel. Speedup
        # factor 3 for ATI/Radeon Xorg driver (and hopefully others).
        # https://bugs.freedesktop.org/show_bug.cgi?id=28670

        surface = pixbufsurface.Surface(x1, y1, x2 - x1 + 1, y2 - y1 + 1)
        return transformation, surface, sparse, mipmap_level, clip_rect

    def _render_execute(self, cr, transformation, surface, sparse,
                        mipmap_level, clip_rect, filter=None):
        """Renders tiles into a prepared pixbufsurface, then blits it.


        """
        translation_only = self.is_translation_only()

        if self.visualize_rendering:
            surface.pixbuf.fill(int(random.random() * 0xff) << 16)

        fake_alpha_check_tile = None
        if not self._draw_real_alpha_checks:
            fake_alpha_check_tile = self._fake_alpha_check_tile

        # Determine which tiles to render.
        tiles = list(surface.get_tiles())
        if sparse:
            tiles = [
                (tx, ty) for (tx, ty) in tiles
                if self._tile_is_visible(
                    tx,
                    ty,
                    transformation,
                    clip_rect,
                    translation_only,
                )
            ]

        # Composite each stack of tiles in the exposed area
        # into the pixbufsurface.
        self.doc._layers.render(
            surface,
            tiles,
            mipmap_level,
            overlay = self.overlay_layer,
            opaque_base_tile = fake_alpha_check_tile,
            filter = filter,
        )

        # Set the surface's underlying pixbuf as the source, then paint
        # it with Cairo. We don't care if it's pixelized at high zoom-in
        # levels: in fact, it'll look sharper and better.
        Gdk.cairo_set_source_pixbuf(
            cr, surface.pixbuf,
            round(surface.x), round(surface.y)
        )
        if self.scale > self.pixelize_threshold:
            pattern = cr.get_source()
            pattern.set_filter(cairo.FILTER_NEAREST)
        cr.paint()

    def scroll(self, dx, dy, ongoing=True):
        self.translation_x -= dx
        self.translation_y -= dy
        if ongoing:
            self.defer_hq_rendering()
        self.queue_draw()

        # This speeds things up nicely when scrolling is already
        # fast, but produces temporary artifacts and an
        # annoyingly non-constant framerate otherwise.
        #
        # self.window.scroll(int(-dx), int(-dy))
        #
        # It might be worth it if it was done only once per
        # redraw, instead of once per motion event. Maybe try to
        # implement something like "queue_scroll" with priority
        # similar to redraw? (The GTK commit responsible for bug
        # http://bugzilla.gnome.org/show_bug.cgi?id=702392 might
        # solve this problem, I think.)

    def get_center(self):
        """Return the center position in display coordinates.
        """
        alloc = self.get_allocation()
        return (alloc.width / 2.0, alloc.height / 2.0)

    def get_center_model_coords(self):
        """Return the center position in model coordinates.
        """
        center = self.get_center()
        return self.display_to_model(*center)

    def recenter_document(self):
        """Recentres the view onto the document's centre.
        """
        x, y, w, h = self.doc.get_effective_bbox()
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
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

    def defer_hq_rendering(self, t=1.0 / 8):
        """Use faster but lower-quality rendering for a brief period

        :param float t: The time to defer for, in seconds

        This method is intended to be called repeatedly
        from scroll or drag event handlers,
        or other times when the entire display
        may need to be redrawn repeatedly in short order.
        It turns off normal rendering, and updates the future time
        at which normal rendering will be automatically resumed.
        Resumption of normal service entails a full redraw,
        so choose `t` appropriately.

        Normal rendering looks better (it uses a better mipmap),
        and it's OK for most screen updates.
        However it's slow enough to make rendering
        lag appreciably when scrolling.

        """
        if self._restore_hq_rendering_timeout_id:
            GLib.source_remove(self._restore_hq_rendering_timeout_id)
            self._restore_hq_rendering_timeout_id = None
        else:
            logger.debug("hq_rendering: deferring for %0.3fs...", t)
            self._hq_rendering = False
        self._restore_hq_rendering_timeout_id = GLib.timeout_add(
            interval = int(t * 1000),
            function = self._resume_hq_rendering_timeout_cb,
        )

    def _resume_hq_rendering_timeout_cb(self):
        self._hq_rendering = True
        self.queue_draw()
        self._restore_hq_rendering_timeout_id = None
        logger.debug("hq_rendering: resumed")
        return False


## Testing


def _make_testbed_model():
    import lib.brush
    import lib.document
    brush = lib.brush.BrushInfo()
    brush.load_defaults()
    return lib.document.Document(brush)


def _test():
    from document import CanvasController
    from freehand import FreehandMode
    model = _make_testbed_model()
    tdw = TiledDrawWidget()
    tdw.set_model(model)
    tdw.set_size_request(640, 480)
    tdw.renderer.visualize_rendering = True
    ctrlr = CanvasController(tdw)
    ctrlr.init_pointer_events()
    ctrlr.modes.default_mode_class = FreehandMode
    win = Gtk.Window()
    win.set_title("tdw test")
    win.connect("destroy", lambda *a: Gtk.main_quit())
    win.add(tdw)
    win.show_all()
    Gtk.main()
    model.cleanup()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _test()
