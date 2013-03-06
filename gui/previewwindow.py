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
import tileddrawwidget


def _points_to_enclosing_rect(corners):
    """Convert a list of (x, y) points to their encompassing rect.
    """
    corners = list(corners)
    x, y = corners.pop(0)
    xmin = xmax = x
    ymin = ymax = y
    for x, y in corners:
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
    return xmin, ymin, xmax-xmin, ymax-ymin


class VisibleOverlay (tileddrawwidget.Overlay):
    """Overlay for the preview TDW which shows the extent of the main TDW.
    """

    OUTER_LINE_WIDTH = 5
    OUTER_LINE_RGBA = 0.0, 0.5, 0.5, 0.2
    INNER_LINE_WIDTH = 1
    INNER_LINE_RGBA = 1.0, 1.0, 1.0, 1.0


    def __init__(self, app, preview):

        self.app = app
        self.main_model = app.doc.model
        self.main_doc = app.doc
        self.main_tdw = app.doc.tdw
        self.preview = preview

        # Painting instructions
        self.paint_rect = None  #: Last-painted region, display coords
        self.paint_corners = None  #: Preview box, display coords


    def paint(self, cr):
        if not self.paint_corners:
            return
        cr.set_line_width(self.OUTER_LINE_WIDTH)
        cr.set_source_rgba(*self.OUTER_LINE_RGBA)
        corners = list(self.paint_corners)
        x, y = corners.pop(0)
        cr.move_to(x, y)
        for x, y in corners:
            cr.line_to(x, y)
        cr.close_path()
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

        # Preview rectangle corners, in preview TDW display coords
        main_view_corners = self.preview._main_view_corners
        if not main_view_corners:
            return
        paint_corners = [self.preview.tdw.model_to_display(*c)
                         for c in main_view_corners]

        # Pixel edge alignment (for the inner line)
        d = 0.5 * self.INNER_LINE_WIDTH
        paint_corners = [(int(x)+d, int(y)+d) for x, y in paint_corners]

        # Drawing area
        alloc = self.preview.tdw.get_allocation()
        x, y, w, h = _points_to_enclosing_rect(paint_corners)
        lw = int(self.OUTER_LINE_WIDTH / 2) + 1
        x = int(x) - lw
        y = int(y) - lw
        w = int(w) + 1 + 2*lw
        h = int(h) + 1 + 2*lw
        outside = (x > alloc.width or y > alloc.height or
                   x+w < alloc.x or y+h < alloc.y)
        if outside:
            # No need to redraw
            self.paint_rect = None
            self.paint_corners = None
        else:
            self.preview.tdw.queue_draw_area(x, y, w, h)
            self.paint_rect = x, y, w, h
            self.paint_corners = paint_corners


class ToolWidget (gtk.EventBox):
    """Tool widget for previewing the whole canvas.

    We overlay a preview rectangle showing where the main document view is
    pointing. The zoom and centering of the preview widget encompasses the
    document's bounding box,
    TODO: and also the viewing rectangle.

    """

    stock_id = "mypaint-tool-preview-window"
    tool_widget_title = _("Preview")

    SUPPORTED_ZOOMLEVELS_ONLY = True   # Not sure if this looks better


    def __init__(self, app):
        gtk.EventBox.__init__(self)
        self.app = app
        self.set_size_request(250, 250)
        self.main_tdw = app.doc.tdw

        self.model = app.doc.model
        self.tdw = tileddrawwidget.TiledDrawWidget(app, self.model)
        self.tdw.zoom_min = 1/50.0
        self.tdw.set_size_request(250, 250)
        self.tdw.set_override_cursor(gdk.Cursor(gdk.LEFT_PTR))
        self.add(self.tdw)

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
        self.tdw.zoom_max = 1.0 / 2
        self.tdw.scale = self.app.preferences['view.default_zoom'] / 2

        # Used for detection of potential effective bbox changes during
        # canvas modify events
        self.x_min = self.x_max = None
        self.y_min = self.y_max = None

        # Used for determining if a potential bbox change is a real one
        self._last_preview_bbox = None

        # Model observers for scale and zoom
        self.model.canvas_observers.append(self.canvas_modified_cb)
        self.model.doc_observers.append(self.doc_structure_modified_cb)
        self.model.frame_observers.append(self.frame_modified_cb)
        self.connect("size-allocate", self.size_alloc_cb)

        # Main controller observers, for updating our overlay
        self._main_view_corners = []
        self.app.doc.view_changed_observers.append(self.main_view_changed_cb)

        # Handle clicks and drags
        self._drag_start = None
        self.add_events(gdk.BUTTON1_MOTION_MASK | gdk.SCROLL_MASK)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("button-release-event", self.button_release_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)
        self.connect("scroll-event", self.scroll_event_cb)


    def scroll_event_cb(self, widget, event):
        """Handle scroll events on the preview: manipulates the main view.
        """

        # Zoom or rotate the main document.
        doc = self.app.doc

        # Recenter main doc
        mx, my = self.tdw.display_to_model(event.x, event.y)
        doc.tdw.recenter_on_model_coords(mx, my) # notify deferred (see below)

        # Handle like ScrollableModeMixin, but affect a different doc.
        d = event.direction
        if d == gdk.SCROLL_UP:
            if event.state & gdk.SHIFT_MASK:
                doc.rotate('RotateLeft', at_pointer=False)
            else:
                doc.zoom('ZoomIn', at_pointer=False)
        elif d == gdk.SCROLL_DOWN:
            if event.state & gdk.SHIFT_MASK:
                doc.rotate('RotateRight', at_pointer=False)
            else:
                doc.zoom('ZoomOut', at_pointer=False)
        elif d == gdk.SCROLL_RIGHT:
            doc.rotate('RotateRight', at_pointer=False)
        elif d == gdk.SCROLL_LEFT:
            doc.rotate('RotateLeft', at_pointer=False)
        else:
            # Deferred call required by recentering block above.
            # Above calls it, but be ready for future scroll directions.
            doc.notify_view_changed()

        return True


    def button_press_cb(self, widget, event):
        if not self._drag_start:
            if event.button == 1:
                mx, my = self.tdw.display_to_model(event.x, event.y)
                self._drag_start = mx, my
                self.main_tdw.recenter_on_model_coords(mx, my)
                self.app.doc.notify_view_changed()
        return True


    def button_release_cb(self, widget, event):
        if self._drag_start:
            if event.button == 1:
                self._drag_start = None
                self.app.doc.notify_view_changed()
        return True


    def motion_notify_cb(self, widget, event):
        if self._drag_start:
            self.main_tdw = self.app.doc.tdw
            mx0, my0 = self._drag_start
            mx, my = self.tdw.display_to_model(event.x, event.y)
            self.main_tdw.recenter_on_model_coords(mx, my)
            # Upping the priority here keeps it feeling more direct.
            self.app.doc.notify_view_changed(prioritize=True)
        return True


    def main_view_changed_cb(self, doc):
        """Callback: viewport changed on the main drawing canvas.
        """

        alloc = self.main_tdw.get_allocation()
        main_x0, main_y0 = 0., 0.
        main_x1, main_y1 = main_x0+alloc.width, main_y0+alloc.height

        main_corners = [ (main_x0, main_y0), (main_x0, main_y1),
                         (main_x1, main_y1), (main_x1, main_y0), ]
        main_corners = [self.main_tdw.display_to_model(*c) for c in main_corners]
        self._main_view_corners = main_corners

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
        """Callback: preview widget has been resized.
        """
        # Reqires a full transformation update.
        self.update_preview_transformation(force=True)


    def frame_modified_cb(self, *args):
        # Effective bbox change due to frame adjustment or toggle.
        updated = self.update_preview_transformation()
        if not updated:
            self.tdw.queue_draw()


    def doc_structure_modified_cb(self, *args):
        # Potentially a layer clear, which could affect the bbox.
        updated = self.update_preview_transformation()
        if not updated:
            self.tdw.queue_draw()


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
            self._last_preview_bbox = None

        # Preview TDW's size, into which everything must be fitted
        alloc = self.tdw.get_allocation()

        # A list of points in model coords which we want to be all inside
        defining_points = list(self._main_view_corners)
        model_bbox = tuple(self.model.get_effective_bbox()) # Axis aligned...
        x, y, w, h = model_bbox
        defining_points.extend([(x, y), (x+w, y+h)])      #... so two suffice

        # Convert to an axis-aligned bounding box.
        # Don't resize unless this has actually changed.
        # Avoids juddering.
        bbox = _points_to_enclosing_rect(defining_points)
        if not force and bbox == self._last_preview_bbox:
            return False
        self._last_preview_bbox = bbox
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
