# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team
# Copyright (C) 2012-2013 by Ali Lown <ali@lown.me.uk>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

from __future__ import division, print_function
import math
import bisect

from gettext import gettext as _

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib
import cairo

import gui.mode
from . import overlays
from . import tileddrawwidget
from .toolstack import SizedVBoxToolWidget, TOOL_WIDGET_NATURAL_HEIGHT_SHORT
import lib.alg as geom
import gui.cursor
import gui.drawutils
import gui.style


## Module consts

PHI = (1.+math.sqrt(2))/2.
REDRAW_PRIORITY = GLib.PRIORITY_LOW


## Helper funcs

def _points_to_enclosing_rect(points):
    """Convert a list of (x, y) points to their encompassing rect.
    """
    points = list(points)
    x, y = points.pop(0)
    xmin = xmax = x
    ymin = ymax = y
    for x, y in points:
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
    return xmin, ymin, xmax-xmin, ymax-ymin


## Class defs

class VisibleAreaOverlay (overlays.Overlay):
    """Overlay for the preview TDW which shows the main TDW's area"""

    ## Class vars

    ## Method defs

    def __init__(self, preview):
        overlays.Overlay.__init__(self)
        self._preview = preview
        self._paint_rect = None  #: Last-painted region, display coords
        self._paint_shapes = None  #: Preview box, display coords
        self._paint_topleft = None

    def paint(self, cr):
        """Paint a viewfinder box showing the main TDW's viewport"""
        if not self._paint_shapes:
            return
        cr.set_line_join(cairo.LINE_JOIN_MITER)
        cr.set_line_cap(cairo.LINE_CAP_SQUARE)
        pixel_centered = (not self._preview.viewport_is_rotated)
        line_color = gui.style.EDITABLE_ITEM_COLOR
        if self._preview.zone == _EditZone.INSIDE:
            line_color = gui.style.ACTIVE_ITEM_COLOR

        line_width = gui.style.DRAGGABLE_EDGE_WIDTH
        pixel_centring_offset = 0.5 if (line_width % 2) else 0.0
        if self._paint_topleft:
            tlx, tly = self._paint_topleft
            tlx = int(tlx) + pixel_centring_offset   # always centred
            tly = int(tly) + pixel_centring_offset
            cr.rectangle(tlx, tly, 1, 1)
            gui.drawutils.render_drop_shadow(
                cr = cr,
                z = 1,
                line_width = line_width,
            )
            cr.set_line_width(line_width)
            cr.set_source_rgb(*line_color.get_rgb())
            cr.fill_preserve()
            cr.stroke()
        for shape in self._paint_shapes:
            points = list(shape)
            if not points:
                continue
            x, y = points.pop(0)
            if pixel_centered:
                x = int(x) + pixel_centring_offset
                y = int(y) + pixel_centring_offset
            cr.move_to(x, y)
            for x, y in points:
                if pixel_centered:
                    x = int(x) + pixel_centring_offset
                    y = int(y) + pixel_centring_offset
                cr.line_to(x, y)
            cr.set_line_width(line_width)
            cr.set_source_rgb(*line_color.get_rgb())
            gui.drawutils.render_drop_shadow(
                cr = cr,
                z = 1,
                line_width = line_width,
            )
            cr.stroke()

    def update_location(self):
        """Queues redraws for the preview when the main view changes"""
        # Last location paint()ed: box might not be there any more.
        if self._paint_rect:
            self._preview.tdw.queue_draw_area(*self._paint_rect)
        # Calculate a shape to paint in preview TDW display coords.
        viewport_overlay_shapes = None
        if self._preview.show_viewfinder:
            viewport_overlay_shapes = self._preview.viewport_overlay_shapes
        if not viewport_overlay_shapes:
            self._paint_shapes = None
            self._paint_rect = None
            return
        paint_shapes = []
        shape_points = []
        alloc = self._preview.tdw.get_allocation()
        for shape in viewport_overlay_shapes:
            points = []
            for x, y in shape:
                x, y = self._preview.tdw.model_to_display(x, y)
                points.append((x, y))
            paint_shapes.append(points)
            shape_points.extend(points)
        # Top-left (or right) dot
        topleft = self._preview.viewport_overlay_topleft
        if topleft:
            topleft = self._preview.tdw.model_to_display(*topleft)
        # Invalidation rectangle.
        alloc = self._preview.tdw.get_allocation()
        x, y, w, h = _points_to_enclosing_rect(shape_points)
        # Accommodate a later render_drop_shadow()
        offsets = gui.drawutils.get_drop_shadow_offsets(
            z = 1,
            line_width = gui.style.DRAGGABLE_EDGE_WIDTH,
        )
        (offs_left, offs_top, offs_right, offs_bottom) = offsets
        x -= offs_left
        y -= offs_top
        w += offs_right + offs_left
        h += offs_bottom + offs_top
        outside = ((x > alloc.width) or (y > alloc.height) or
                   (x+w < 0) or (y+h < 0))
        if outside:
            self._paint_rect = None
            self._paint_shapes = None
            self._paint_topleft = None
        else:
            self._preview.tdw.queue_draw_area(x, y, w, h)
            self._paint_rect = x, y, w, h
            self._paint_shapes = paint_shapes
            self._paint_topleft = topleft


class _EditZone:
    OUTSIDE = 0
    INSIDE = 1


class PreviewTool (SizedVBoxToolWidget):
    """Tool widget for previewing the whole canvas.

    We overlay a preview rectangle showing where the main document view is
    pointing. The zoom and centering of the preview widget encompasses the
    document's bounding box,
    TODO: and also the viewing rectangle.

    """

    ## Class vars

    SIZED_VBOX_NATURAL_HEIGHT = TOOL_WIDGET_NATURAL_HEIGHT_SHORT

    tool_widget_icon_name = "mypaint-view-symbolic"

    # TRANSLATORS: title of panel showing an overview of the whole canvas
    tool_widget_title = _("Preview")

    tool_widget_description = _("Show preview of the whole drawing area")

    __gtype_name__ = 'MyPaintPreviewTool'

    #: Zoom the preview only to a limited number of zoom levels - reduces
    #: the frequency of zooming, at the expence of a close match.
    SUPPORTED_ZOOMLEVELS_ONLY = False

    #: The preview's zoom can be fitted to include the whole of the viewport
    #: highlight rectangle, even if it lies outside the document area.
    ZOOM_INCLUDES_VIEWPORT_RECT = False

    #: Prefs key for the flag controlling whether the viewport highlight
    #: rectangle is visible.
    SHOW_VIEWFINDER_PREFS_KEY = "preview.show_viewfinder"

    ## Method defs

    def __init__(self):
        super(SizedVBoxToolWidget, self).__init__()
        from gui.application import get_app
        app = get_app()
        self.app = app
        self._main_tdw = app.doc.tdw
        self._model = app.doc.model
        self.tdw = tileddrawwidget.TiledDrawWidget(
            idle_redraw_priority = REDRAW_PRIORITY,
        )
        self.tdw.set_model(self._model)
        self.tdw.zoom_min = 1/50.0
        self.tdw.set_size_request(64, 64)
        self.pack_start(self.tdw, True, True, 0)
        self._cursor = None

        # Cursors for states
        self._cursor_move_here = app.cursors.get_icon_cursor(
            "mypaint-view-zoom-symbolic",
            cursor_name=gui.cursor.Name.ARROW,
        )
        self._cursor_drag_ready = app.cursors.get_icon_cursor(
            "mypaint-view-pan-symbolic",
            cursor_name=gui.cursor.Name.HAND_OPEN,
        )
        self._cursor_drag_active = app.cursors.get_icon_cursor(
            "mypaint-view-pan-symbolic",
            cursor_name=gui.cursor.Name.HAND_CLOSED,
        )
        self._cursor_no_op = app.cursors.get_icon_cursor(
            None,
            cursor_name=gui.cursor.Name.ARROW,
        )

        # Overlay shapes (used by the overlay)
        self.viewport_overlay_shapes = []
        self.viewport_overlay_topleft = None
        self.viewport_is_rotated = False
        self.viewport_is_mirrored = False

        # TRANSLATORS: The preview panel shows where the "camera" of the
        # TRANSLATORS: main view is pointing.
        checkbtn = Gtk.CheckButton(label=_("Show Viewfinder"))
        checkbtn.set_active(self.show_viewfinder)
        self.pack_start(checkbtn, False, False, 0)
        checkbtn.connect("toggled", self._show_viewfinder_toggled_cb)

        self._overlay = VisibleAreaOverlay(self)
        self.tdw.display_overlays.append(self._overlay)

        if self.SUPPORTED_ZOOMLEVELS_ONLY:
            self._zoomlevel_values = [
                1.0/128, 1.5/128,
                1.0/64, 1.5/64,
                1.0/32, 1.5/32,
                1.0/16, 1.0/8, 2.0/11, 0.25, 1.0/3, 0.50, 2.0/3,
                1.0
            ]

        self.tdw.zoom_min = 1.0 / 128
        self.tdw.zoom_max = float(app.preferences.get('view.default_zoom', 1))

        # Used for detection of potential effective bbox changes during
        # canvas modify events
        self._x_min = self._x_max = None
        self._y_min = self._y_max = None

        # Used for determining if a potential bbox change is a real one
        self._last_bbox = None

        # Watch various objects for updates
        docmodel = self._model
        layerstack = docmodel.layer_stack
        observed_events = {
            self._canvas_area_modified_cb: [
                docmodel.canvas_area_modified,
            ],
            self._recreate_preview_transformation: [
                layerstack.layer_inserted,
                layerstack.layer_deleted,
            ],
        }
        for observer_method, events in observed_events.items():
            for event in events:
                event += observer_method

        # Watch the main model's frame for scale and zoom
        docmodel.frame_updated += self._frame_modified_cb
        docmodel.frame_enabled_changed += self._frame_modified_cb

        # Main controller observers, for updating our overlay
        self.app.doc.view_changed_observers.append(self._main_view_changed_cb)

        # Click and drag tracking
        self._zone = _EditZone.OUTSIDE
        self._drag_start = None
        self._button_pressed = None

        # Events for the preview widget
        self.tdw.add_events(
            Gdk.EventMask.BUTTON1_MOTION_MASK | Gdk.EventMask.SCROLL_MASK)
        preview_tdw_events = {
            # Clicks and drags
            "button-press-event": self._button_press_cb,
            "button-release-event": self._button_release_cb,
            "motion-notify-event": self._motion_notify_cb,
            "scroll-event": self._scroll_event_cb,
            "leave-notify-event": self._leave_notify_cb,
            # Handle resizes
            "size-allocate": self._recreate_preview_transformation,
        }
        for signal, callback in preview_tdw_events.items():
            self.tdw.connect(signal, callback)

    ## Show Viewfinder toggle

    def _show_viewfinder_toggled_cb(self, checkbtn):
        """Show Viewfinder action callback"""
        self.show_viewfinder = checkbtn.get_active()
        self._set_cursor(self._cursor_no_op)

    @property
    def show_viewfinder(self):
        """Show Viewfinder property: stored directly in app preferences"""
        return self.app.preferences.get(self.SHOW_VIEWFINDER_PREFS_KEY, True)

    @show_viewfinder.setter
    def show_viewfinder(self, value):
        old_value = self.show_viewfinder
        self.app.preferences[self.SHOW_VIEWFINDER_PREFS_KEY] = bool(value)
        if old_value != value:
            self._update_preview_transformation(force=True)

    ## Cursor for the preview TDW

    def _set_cursor(self, value):
        """Sets the preview TDW cursor"""
        if value == self._cursor:
            return
        self._cursor = value
        self.tdw.set_override_cursor(value)

    ## Preview TDW event handlers

    def _leave_notify_cb(self, widget, event):
        if self._drag_start:
            return
        if event.mode != Gdk.CrossingMode.NORMAL:
            return
        self.set_zone(_EditZone.OUTSIDE, update_ui=True)

    def _scroll_event_cb(self, widget, event):
        """Scroll events on the preview manipulate the main view"""
        if not self.show_viewfinder:
            return False
        # Zoom or rotate the main document's view via its controller
        doc = self.app.doc
        # Centre of rotation for main tdw
        mx, my = self.tdw.display_to_model(event.x, event.y)
        cx, cy = doc.tdw.model_to_display(mx, my)
        # Handle like ScrollableModeMixin, but affect a different doc.
        d = event.direction
        if d == Gdk.ScrollDirection.UP:
            if event.state & Gdk.ModifierType.SHIFT_MASK:
                doc.rotate(doc.ROTATE_CLOCKWISE, center=(cx, cy))
            else:
                doc.zoom(doc.ZOOM_INWARDS, center=(cx, cy))
        elif d == Gdk.ScrollDirection.DOWN:
            if event.state & Gdk.ModifierType.SHIFT_MASK:
                doc.rotate(doc.ROTATE_ANTICLOCKWISE, center=(cx, cy))
            else:
                doc.zoom(doc.ZOOM_OUTWARDS, center=(cx, cy))
        elif d == Gdk.ScrollDirection.RIGHT:
            doc.rotate(doc.ROTATE_ANTICLOCKWISE, center=(cx, cy))
        elif d == Gdk.ScrollDirection.LEFT:
            doc.rotate(doc.ROTATE_CLOCKWISE, center=(cx, cy))
        return True

    def _button_press_cb(self, widget, event):
        if not self.show_viewfinder:
            return False
        if not self._drag_start and event.button == 1:
            zone = self._get_zone_at_pos(event.x, event.y)
            self.set_zone(zone, update_ui=False)
            if zone == _EditZone.INSIDE:
                pmx, pmy = self.tdw.display_to_model(event.x, event.y)
                cmx, cmy = self._main_tdw.get_center_model_coords()
                self._drag_start = (cmx, cmy, pmx, pmy)
                self._set_cursor(self._cursor_drag_active)
        self._button_pressed = event.button
        return True

    def _button_release_cb(self, widget, event):
        if not self.show_viewfinder:
            return False
        if event.button == self._button_pressed:
            if self._drag_start:
                self._drag_start = None
            elif event.button == 1:
                mx, my = self.tdw.display_to_model(event.x, event.y)
                self._main_tdw.recenter_on_model_coords(mx, my)
                self.app.doc.notify_view_changed()
            # Cursor is now directly over the overlay
            zone = _EditZone.INSIDE
            self.set_zone(zone, update_ui=True)
        self._button_pressed = None
        return True

    def _motion_notify_cb(self, widget, event):
        if not self.show_viewfinder:
            return False
        if self._drag_start:
            pmx, pmy = self.tdw.display_to_model(event.x, event.y)
            cmx0, cmy0, pmx0, pmy0 = self._drag_start
            dmx, dmy = pmx-pmx0, pmy-pmy0
            self._main_tdw.recenter_on_model_coords(cmx0+dmx, cmy0+dmy)
            self.app.doc.notify_view_changed(prioritize=True)
        else:
            zone = self._get_zone_at_pos(event.x, event.y)
            self.set_zone(zone)
        return True

    def _get_zone_at_pos(self, x, y):
        """Gets the hover zone for a pointer position"""
        if self._drag_start is not None:
            return _EditZone.OUTSIDE
        elif self.viewport_overlay_shapes:
            points = []
            for shape in self.viewport_overlay_shapes:
                points.extend(shape)
            model_xy = self.tdw.display_to_model(x, y)
            viewport_shape = geom.convex_hull(points)
            if geom.point_in_convex_poly(model_xy, viewport_shape):
                return _EditZone.INSIDE
        return _EditZone.OUTSIDE

    def get_zone(self):
        """Getter (property: `zone`)"""
        return self._zone

    def set_zone(self, value, update_ui=None):
        """Setter (property: `zone`), with redraw and cursor setting"""
        value_changed = (value != self._zone)
        self._zone = value
        if update_ui is None:  # use the default unless explicitly overridden
            update_ui = value_changed
        if not update_ui:
            return
        cursor = {
            _EditZone.INSIDE: self._cursor_drag_ready,
            _EditZone.OUTSIDE: self._cursor_move_here,
        }.get(value)
        self._set_cursor(cursor)
        self.tdw.queue_draw()   # rendering is sensitive to zone

    zone = property(get_zone, set_zone)

    def _main_view_changed_cb(self, doc):
        """Callback: viewport changed on the main drawing canvas"""
        self._update_viewport_overlay()

    def _update_viewport_overlay(self):
        """Updates the viewport overlay's position"""
        alloc = self._main_tdw.get_allocation()
        x, y = 0., 0.
        w, h = alloc.width, alloc.height
        # List of viewport corners
        nw = w/4*PHI
        nh = h/4*PHI
        overlay_shapes_disp = [
            [(x, y+nh), (x, y), (x+nw, y)],
            [(x, h-nh), (x, h), (x+nw, h)],
            [(w-nw, y), (w, y), (w, x+nh)],
            [(w-nw, h), (w, h), (w, h-nh)],
        ]
        # To model coords
        overlay_shapes_model = []
        for shape in overlay_shapes_disp:
            shape = [self._main_tdw.display_to_model(*pos)
                     for pos in shape]
            overlay_shapes_model.append(shape)
        self.viewport_overlay_shapes = overlay_shapes_model
        # Top left/right dot, for displaying the orientation
        self.viewport_is_mirrored = (self._main_tdw.mirrored)
        self.viewport_is_rotated = (self._main_tdw.rotation != 0)
        if not (self.viewport_is_mirrored or self.viewport_is_rotated):
            self.viewport_overlay_topleft = None
        else:
            k = nh/2
            j = nw/2
            direction = self._main_tdw.get_direction()
            if direction == Gtk.TextDirection.RTL:
                j = w-j
            topleft = j, k
            topleft = self._main_tdw.display_to_model(*topleft)
            self.viewport_overlay_topleft = topleft
        if self._drag_start:
            # Too distracting to change the preview transform
            self._overlay.update_location()
        else:
            # User might have moved the view outside the existing bbox.
            updated = self._update_preview_transformation()
            if not updated:
                self._overlay.update_location()

    def _limit_scale(self, scale):
        """Limits a calculated scale to the permitted ones"""
        scale = min(scale, self.tdw.zoom_max)
        scale = max(scale, self.tdw.zoom_min)
        if self.SUPPORTED_ZOOMLEVELS_ONLY:
            # Limit to a supported zoom level
            scale_i = bisect.bisect_left(self._zoomlevel_values, scale)
            if scale_i >= len(self._zoomlevel_values):
                scale_i = len(self._zoomlevel_values) - 1
            scale = self._zoomlevel_values[max(0, scale_i-1)]
        return scale

    def _frame_modified_cb(self, *_ignored):
        # Effective bbox change due to frame adjustment or toggle. The
        # only reason to do this separately is to support
        # ZOOM_INCLUDES_VIEWPORT_RECT.
        updated = self._update_preview_transformation()
        if not updated:
            self.tdw.queue_draw()

    def _recreate_preview_transformation(self, *_ignored):
        """Update the preview transformation fully: no optimizations

        Handler for the layer stack being restructured, or the preview
        panel being resized. Both need a full transformation update and
        redraw.
        """
        self._update_preview_transformation(force=True)

    def _canvas_area_modified_cb(self, main_model, x, y, w, h):
        """Callback: layer contents have changed on the main canvas.

        Called when layer contents change and a redraw is
        required.  This tries to avoid unnecessary updates to the
        projection, for example when the the drawing happens draws
        inside the previously known area.

        """
        outside_existing = False
        if x == 0 and y == 0 and w == 0 and h == 0:
            # This is a redraw-all notification. Don't track the zeros.
            outside_existing = True
        else:
            # Real update rectangle: track size.
            if self._x_min is None or x < self._x_min:
                self._x_min = x
                outside_existing = True
            if self._x_max is None or x+w > self._x_max:
                self._x_max = x+w
                outside_existing = True
            if self._y_min is None or y < self._y_min:
                self._y_min = y
                outside_existing = True
            if self._y_max is None or y+h > self._y_max:
                self._y_max = y+h
                outside_existing = True
        # Update if the user went outside the existing area.
        if outside_existing:
            self._update_preview_transformation()

    def _update_preview_transformation(self, force=False):
        """Update preview's scale and centering, if needed.

        This only updates the preview transformation when needed, to
        avoid unncecessary redraws: if the transformation is updated, a
        full redraw is performed.

        :param force: Always update scale and centering.
        :return: True if an update was performed.

        """

        # Clear tracking variables, if update forced
        if force:
            self._x_min = None
            self._x_max = None
            self._y_min = None
            self._y_max = None
            self._last_bbox = None

        # Preview TDW's size, into which everything must be fitted
        alloc = self.tdw.get_allocation()

        # A list of points in model coords which we want to be all inside
        if self.ZOOM_INCLUDES_VIEWPORT_RECT:
            defining_points = list(self.viewport_overlay_shapes)
        else:
            defining_points = []
        model_bbox = tuple(self._model.get_effective_bbox())  # Axis aligned...
        x, y, w, h = model_bbox
        defining_points.extend([(x, y), (x+w, y+h)])        # ...so two suffice

        # Convert to an axis-aligned bounding box.
        # Don't resize unless this has actually changed.
        # Avoids juddering.
        bbox = _points_to_enclosing_rect(defining_points)
        if not force and bbox == self._last_bbox:
            return False
        self._last_bbox = bbox
        x, y, w, h = bbox

        # Avoid a division by zero
        if w == 0:
            w = 64
        if h == 0:
            h = 64

        # Tracking vars may have been reset.
        # The bbox is a pretty good seed value for them...
        if (self._x_min is None) or (x < self._x_min):
            self._x_min = x
        if (self._x_max is None) or (x+w > self._x_max):
            self._x_max = x+w
        if (self._y_min is None) or (y < self._y_min):
            self._y_min = y
        if (self._y_max is None) or (y+h > self._y_max):
            self._y_max = y+h

        # Scale to fit within a rectangle slightly smaller than the widget.
        # Slight borders are nice.
        border = 12
        zoom_x = (alloc.width - border) / w
        zoom_y = (alloc.height - border) / h

        # Set the preview canvas's size and scale
        scale = self._limit_scale(min(zoom_x, zoom_y))
        self.tdw.scale = scale
        cx = x + w/2.
        cy = y + h/2.
        self.tdw.recenter_on_model_coords(cx, cy)

        # Update the overlay, since the transformation has changed
        self._overlay.update_location()
        return True
