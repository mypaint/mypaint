# This file is part of MyPaint.
# Copyright (C) 2012 by Ali Lown <ali@lown.me.uk>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import math
import bisect

from gettext import gettext as _
import gobject
import gtk
from gtk import gdk
import pango
import cairo

import canvasevent
import document
import dialogs
import overlays
import tileddrawwidget

from gui.colors import geom   # XXX refactoring needed! But from here for now.


PHI = (1.+math.sqrt(2))/2.


def _points_to_enclosing_rect(points):
    """Convert a list of (x, y) points to their encompassing rect.
    """
    points = list(points)
    x, y = points.pop(0)
    xmin = xmax = x
    ymin = ymax = y
    for x, y in points:
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
    return xmin, ymin, xmax-xmin, ymax-ymin


class VisibleOverlay (overlays.Overlay):
    """Overlay for the preview TDW which shows the extent of the main TDW.
    """

    INNER_LINE_WIDTH = 1.5
    INNER_LINE_RGBA = 0.832, 1.000, 0.090, 1.0

    OUTER_LINE_WIDTH = 3
    OUTER_LINE_RGBA = 0.451/2, 0.823/2, 0.086/2, 0.5

    def __init__(self, app, preview):

        self.app = app
        self.main_model = app.doc.model
        self.main_doc = app.doc
        self.main_tdw = app.doc.tdw
        self.preview = preview

        # Painting instructions
        self.paint_rect = None  #: Last-painted region, display coords
        self.paint_shapes = None  #: Preview box, display coords
        self.paint_topleft = None


    def paint(self, cr):
        if not self.paint_shapes:
            return
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        if self.paint_topleft:
            tlx, tly = self.paint_topleft
            r1 = self.OUTER_LINE_WIDTH / 2.
            r2 = self.INNER_LINE_WIDTH / 2.
            cr.set_line_width(self.OUTER_LINE_WIDTH)
            cr.set_source_rgba(*self.OUTER_LINE_RGBA)
            cr.rectangle(tlx, tly, 1, 1)
            cr.fill_preserve()
            cr.stroke_preserve()
            cr.set_line_width(self.INNER_LINE_WIDTH)
            cr.set_source_rgba(*self.INNER_LINE_RGBA)
            cr.fill_preserve()
            cr.stroke()
        for shape in self.paint_shapes:
            points = list(shape)
            if not points:
                continue
            x, y = points.pop(0)
            cr.move_to(x, y)
            for x, y in points:
                cr.line_to(x, y)
            cr.set_line_width(self.OUTER_LINE_WIDTH)
            cr.set_source_rgba(*self.OUTER_LINE_RGBA)
            cr.stroke_preserve()
            cr.set_line_width(self.INNER_LINE_WIDTH)
            cr.set_source_rgba(*self.INNER_LINE_RGBA)
            cr.stroke()


    def update_location(self):
        """Queues redraws for the preview when the main view changes.
        """

        # Last location paint()ed: box might not be there any more.
        if self.paint_rect:
            self.preview.tdw.queue_draw_area(*self.paint_rect)

        # Calculate a shape to paint in preview TDW display coords.
        viewport_overlay_shapes = None
        if self.preview.show_viewfinder:
            viewport_overlay_shapes = self.preview.viewport_overlay_shapes
        if not viewport_overlay_shapes:
            self.paint_shapes = None
            self.paint_rect = None
            return
        paint_shapes = []
        shape_points = []
        alloc = self.preview.tdw.get_allocation()
        for shape in viewport_overlay_shapes:
            points = []
            for x, y in shape:
                x, y = self.preview.tdw.model_to_display(x, y)
                points.append((x, y))
            paint_shapes.append(points)
            shape_points.extend(points)

        # Top-left (or right) dot
        topleft = self.preview.viewport_overlay_topleft
        if topleft:
            topleft = self.preview.tdw.model_to_display(*topleft)

        # Invalidation rectangle.
        alloc = self.preview.tdw.get_allocation()
        x, y, w, h = _points_to_enclosing_rect(shape_points)
        lw = self.OUTER_LINE_WIDTH
        x = int(x - lw/2) - 2
        y = int(y - lw/2) - 2
        w = int(w + lw) + 4
        h = int(h + lw) + 4
        outside = ((x > alloc.width) or (y > alloc.height) or
                   (x+w < 0) or (y+h < 0))
        if outside:
            self.paint_rect = None
            self.paint_shapes = None
            self.paint_topleft = None
        else:
            self.preview.tdw.queue_draw_area(x, y, w, h)
            self.paint_rect = x, y, w, h
            self.paint_shapes = paint_shapes
            self.paint_topleft = topleft


class PreviewTool (gtk.VBox):
    """Tool widget for previewing the whole canvas.

    We overlay a preview rectangle showing where the main document view is
    pointing. The zoom and centering of the preview widget encompasses the
    document's bounding box,
    TODO: and also the viewing rectangle.

    """

    tool_widget_icon_name = "mypaint-tool-preview-window"

    #TRANSLATORS: title of dockable widget showing an overview of the whole canvas
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


    def __init__(self):
        gtk.VBox.__init__(self)
        from application import get_app
        app = get_app()
        self.app = app
        self.set_size_request(250, 250)
        self.main_tdw = app.doc.tdw

        self.model = app.doc.model
        self.tdw = tileddrawwidget.TiledDrawWidget()
        self.tdw.set_model(self.model)
        self.tdw.zoom_min = 1/50.0
        self.tdw.set_size_request(250, 250)
        self.pack_start(self.tdw, True, True)

        # Cursor property
        self._cursor = None
        self.cursor = None

        # Cursors for states
        cursor_icon = "mypaint-view-pan"
        self.cursor_move_here = app.cursors.get_icon_cursor(cursor_icon,
                                 cursor_name="cursor_arrow")
        self.cursor_drag_ready = app.cursors.get_icon_cursor(cursor_icon,
                                  cursor_name="cursor_hand_open")
        self.cursor_drag_active = app.cursors.get_icon_cursor(cursor_icon,
                                   cursor_name="cursor_hand_closed")
        self.cursor_no_op = app.cursors.get_icon_cursor(None,
                             cursor_name="cursor_arrow")

        #TRANSLATORS: shows a zoomed view inside the overview, like a camera
        checkbtn = gtk.CheckButton(_("Show Viewfinder"))
        checkbtn.set_active(self.show_viewfinder)
        self.show_viewport_checkbutton = checkbtn
        self.pack_start(checkbtn, False, False)
        checkbtn.connect("toggled", self.show_viewfinder_toggled_cb)

        self.visible_overlay = VisibleOverlay(app, self)
        self.tdw.display_overlays.append(self.visible_overlay)

        if self.SUPPORTED_ZOOMLEVELS_ONLY:
            self._zoomlevel_values = [
                1.0/128, 1.5/128,
                1.0/64, 1.5/64,
                1.0/32, 1.5/32,
                1.0/16, 1.0/8, 2.0/11, 0.25, 1.0/3, 0.50, 2.0/3,
                1.0 ]

        self.tdw.zoom_min = 1.0 / 128
        self.tdw.zoom_max = float(app.preferences.get('view.default_zoom', 1))

        # Used for detection of potential effective bbox changes during
        # canvas modify events
        self.x_min = self.x_max = None
        self.y_min = self.y_max = None

        # Used for determining if a potential bbox change is a real one
        self._last_bbox = None

        # Model observers for scale and zoom
        self.model.canvas_observers.append(self.canvas_modified_cb)
        self.model.doc_observers.append(self.doc_structure_modified_cb)
        self.model.frame_observers.append(self.frame_modified_cb)
        self.tdw.connect("size-allocate", self.size_alloc_cb)

        # Main controller observers, for updating our overlay
        self.viewport_overlay_shapes = []
        self.viewport_overlay_topleft = []
        self.app.doc.view_changed_observers.append(self.main_view_changed_cb)

        # Handle clicks and drags
        self._drag_start = None
        self._button_pressed = None
        self.tdw.add_events(gdk.BUTTON1_MOTION_MASK | gdk.SCROLL_MASK)
        self.tdw.connect("button-press-event", self.button_press_cb)
        self.tdw.connect("button-release-event", self.button_release_cb)
        self.tdw.connect("motion-notify-event", self.motion_notify_cb)
        self.tdw.connect("scroll-event", self.scroll_event_cb)


    @property
    def show_viewfinder(self):
        return self.app.preferences.get(self.SHOW_VIEWFINDER_PREFS_KEY, True)

    @show_viewfinder.setter
    def show_viewfinder(self, value):
        old_value = self.show_viewfinder
        self.app.preferences[self.SHOW_VIEWFINDER_PREFS_KEY] = bool(value)
        if old_value != value:
            self.update_preview_transformation(force=True)


    @property
    def cursor(self):
        """The current cursor - settable by property.
        """
        return self._cursor

    @cursor.setter
    def cursor(self, value):
        if value == self._cursor:
            return
        self._cursor = value
        self.tdw.set_override_cursor(value)


    def show_viewfinder_toggled_cb(self, checkbtn):
        self.show_viewfinder = checkbtn.get_active()
        self.cursor = self.cursor_no_op


    def scroll_event_cb(self, widget, event):
        """Handle scroll events on the preview: manipulates the main view.
        """

        if not self.show_viewfinder:
            return False

        # Zoom or rotate the main document.
        doc = self.app.doc

        # Centre of rotation for main tdw
        mx, my = self.tdw.display_to_model(event.x, event.y)
        cx, cy = doc.tdw.model_to_display(mx, my)

        # Handle like ScrollableModeMixin, but affect a different doc.
        d = event.direction
        if d == gdk.SCROLL_UP:
            if event.state & gdk.SHIFT_MASK:
                doc.rotate(doc.ROTATE_CLOCKWISE, center=(cx, cy))
            else:
                doc.zoom(doc.ZOOM_INWARDS, center=(cx, cy))
        elif d == gdk.SCROLL_DOWN:
            if event.state & gdk.SHIFT_MASK:
                doc.rotate(doc.ROTATE_ANTICLOCKWISE, center=(cx, cy))
            else:
                doc.zoom(doc.ZOOM_OUTWARDS, center=(cx, cy))
        elif d == gdk.SCROLL_RIGHT:
            doc.rotate(doc.ROTATE_ANTICLOCKWISE, center=(cx, cy))
        elif d == gdk.SCROLL_LEFT:
            doc.rotate(doc.ROTATE_CLOCKWISE, center=(cx, cy))

        return True


    def button_press_cb(self, widget, event):
        if not self.show_viewfinder:
            return False
        if not self._drag_start and event.button == 1:
            if self.viewport_overlay_shapes:
                points = []
                for shape in self.viewport_overlay_shapes:
                    points.extend(shape)
                pmx, pmy = self.tdw.display_to_model(event.x, event.y)
                cmx, cmy = self.main_tdw.get_center_model_coords()
                inside = False
                shape = geom.convex_hull(points)
                if geom.point_in_convex_poly((pmx, pmy), shape):
                    self._drag_start = (cmx, cmy, pmx, pmy)
                    self.cursor = self.cursor_drag_active
        self._button_pressed = event.button
        return True


    def button_release_cb(self, widget, event):
        if not self.show_viewfinder:
            return False
        if event.button == self._button_pressed:
            if self._drag_start:
                self._drag_start = None
            elif event.button == 1:
                mx, my = self.tdw.display_to_model(event.x, event.y)
                self.main_tdw.recenter_on_model_coords(mx, my)
                self.app.doc.notify_view_changed()
            # Cursor is now directly over the overlay
            self.cursor = self.cursor_drag_ready
        self._button_pressed = None
        return True


    def motion_notify_cb(self, widget, event):
        if not self.show_viewfinder:
            return False
        pmx, pmy = self.tdw.display_to_model(event.x, event.y)
        if self._drag_start:
            cmx0, cmy0, pmx0, pmy0 = self._drag_start
            dmx, dmy = pmx-pmx0, pmy-pmy0
            self.main_tdw.recenter_on_model_coords(cmx0+dmx, cmy0+dmy)
            self.app.doc.notify_view_changed(prioritize=True)
        else:
            cursor = None
            if self.viewport_overlay_shapes:
                points = []
                for shape in self.viewport_overlay_shapes:
                    points.extend(shape)
                shape = geom.convex_hull(points)
                icon_name = "mypaint-view-pan"
                if geom.point_in_convex_poly((pmx, pmy), shape):
                    cursor = self.cursor_drag_ready
                else:
                    cursor = self.cursor_move_here
            self.cursor = cursor
        return True


    def main_view_changed_cb(self, doc):
        """Callback: viewport changed on the main drawing canvas.
        """
        self._update_viewport_overlay()


    def _update_viewport_overlay(self):
        """Updates the viewport overlay's position.
        """
        alloc = self.main_tdw.get_allocation()
        x, y = 0., 0.
        w, h = float(alloc.width), float(alloc.height)

        # Start with a list of "viewport corners"
        n = min(w, h)/4*PHI
        overlay_shapes_disp = [ [(x, y+n), (x, y), (x+n, y)],
                                [(x, h-n), (x, h), (x+n, h)],
                                [(w-n, y), (w, y), (w, x+n)],
                                [(w-n, h), (w, h), (w, h-n)], ]

        # For very long viewports, add another pair of lines along the
        # long axis. Not too long, and not too short hopefully.
        if w > h*PHI:
            m = h/2*PHI
            overlay_shapes_disp.extend([
                    [(x+m, y), (w-m, y)],
                    [(x+m, h), (w-m, h)],
                ])
        elif h > w*PHI:
            m = h/2*PHI
            overlay_shapes_disp.extend([
                    [(x, y+m), (x, h-m)],
                    [(w, y+m), (w, h-m)],
                ])

        # To model coords
        overlay_shapes_model = []
        for shape in overlay_shapes_disp:
            shape = [self.main_tdw.display_to_model(*pos)
                     for pos in shape]
            overlay_shapes_model.append(shape)
        self.viewport_overlay_shapes = overlay_shapes_model

        # Top left/right dot, for displaying the orientation
        if self.main_tdw.rotation == 0.0 and not self.main_tdw.mirrored:
            self.viewport_overlay_topleft = None
        else:
            k = n - n/PHI
            j = n/PHI
            direction = self.main_tdw.get_direction()
            if direction == gtk.TEXT_DIR_RTL:
                j = w-j
            topleft = j, k
            topleft = self.main_tdw.display_to_model(*topleft)
            self.viewport_overlay_topleft = topleft

        if self._drag_start:
            # Too distracting to change the preview transform
            self.visible_overlay.update_location()
        else:
            # User might have moved the view outside the existing bbox.
            updated = self.update_preview_transformation()
            if not updated:
                self.visible_overlay.update_location()


    def limit_scale(self, scale):
        """Limits a calculated scale to the permitted ones.
        """
        scale = min(scale, self.tdw.zoom_max)
        scale = max(scale, self.tdw.zoom_min)
        if self.SUPPORTED_ZOOMLEVELS_ONLY:
            # Limit to a supported zoom level
            scale_i = bisect.bisect_left(self._zoomlevel_values, scale)
            if scale_i >= len(self._zoomlevel_values):
                scale_i = len(self._zoomlevel_values) - 1
            scale = self._zoomlevel_values[max(0, scale_i-1)]
        return scale


    def size_alloc_cb(self, widget, alloc):
        """Callback: a preview widget has been resized.
        """
        # Reqires a full transformation update.
        self.update_preview_transformation(force=True)


    def frame_modified_cb(self, *args):
        # Effective bbox change due to frame adjustment or toggle.
        updated = self.update_preview_transformation()
        if not updated:
            self.tdw.queue_draw()


    def doc_structure_modified_cb(self, *args):
        # Potentially a layer clear or document-new, which could
        # affect the bbox.
        self.update_preview_transformation(force=True)


    def canvas_modified_cb(self, x, y, w, h):
        """Callback: layer contents have changed on the main canvas.

        E.g. drawing. Called when layer contents change and redraw is required.
        Try to avoid unnecessary updates, e.g. drawing inside the previously
        known area.

        """

        outside_existing = False
        if x == 0 and y == 0 and w == 0 and h == 0:
            # This is a redraw-all notification. Don't track the zeros.
            outside_existing = True
        else:
            # Real update rectangle: track size.
            if self.x_min is None or x < self.x_min:
                self.x_min = x
                outside_existing = True
            if self.x_max is None or x+w > self.x_max:
                self.x_max = x+w
                outside_existing = True
            if self.y_min is None or y < self.y_min:
                self.y_min = y
                outside_existing = True
            if self.y_max is None or y+h > self.y_max:
                self.y_max = y+h
                outside_existing = True

        # Update if the user went outside the existing area.
        if outside_existing:
            self.update_preview_transformation()


    def update_preview_transformation(self, force=False):
        """Update preview's scale and centering, if needed.

        This only updates the preview transformation when needed, to avoid
        unncecessary redraws: if the transformation is updated, a full redraw
        is performed.

        :param force: Always update scale and centering.
        :return: True if an update was performed.

        """

        # Clear tracking variables, if update forced
        if force:
            self.x_min = None
            self.x_max = None
            self.y_min = None
            self.y_max = None
            self._last_bbox = None

        # Preview TDW's size, into which everything must be fitted
        alloc = self.tdw.get_allocation()

        # A list of points in model coords which we want to be all inside
        if self.ZOOM_INCLUDES_VIEWPORT_RECT:
            defining_points = list(self.viewport_overlay_shapes)
        else:
            defining_points = []
        model_bbox = tuple(self.model.get_effective_bbox()) # Axis aligned...
        x, y, w, h = model_bbox
        defining_points.extend([(x, y), (x+w, y+h)])      #... so two suffice

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
        if x < self.x_min:
            self.x_min = x
        if x+w > self.x_max:
            self.x_max = x+w
        if y < self.y_min:
            self.y_min = y
        if y+h > self.y_max:
            self.y_max = y+h

        # Scale to fit within a rectangle slightly smaller than the widget.
        # Slight borders are nice.
        border = 12
        zoom_x = (float(alloc.width) - border) / w
        zoom_y = (float(alloc.height) - border) / h

        # Set the preview canvas's size and scale
        scale = self.limit_scale(min(zoom_x, zoom_y))
        self.tdw.scale = scale
        cx = x + w/2.
        cy = y + h/2.
        self.tdw.recenter_on_model_coords(cx, cy)

        # Update the overlay, since the transfrmation has changed
        self.visible_overlay.update_location()
        return True
