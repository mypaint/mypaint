# This file is part of MyPaint.
# Copyright (C) 2011-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Color and brush history view widgets"""


## Imports

from __future__ import division, print_function

from lib.gibindings import Gtk
from lib.gibindings import GLib
from lib.gibindings import GdkPixbuf

from lib.color import RGBColor
from .colors import ColorAdjuster
from lib.observable import event
from . import widgets


## Module constants

HISTORY_PREVIEW_SIZE = 32


## Class definitions


class BrushHistoryView (Gtk.HBox):
    """A set of clickable images showing the brush usage history"""

    def __init__(self, app):
        Gtk.HBox.__init__(self)
        self._app = app
        bm = app.brushmanager
        self._history_images = []
        s = HISTORY_PREVIEW_SIZE
        self.set_border_width(widgets.SPACING)
        for i, brush in enumerate(bm.history):
            image = ManagedBrushPreview()
            image.set_size_request(s, s)
            self._history_images.append(image)
            button = widgets.borderless_button()
            button.add(image)
            button.connect("clicked", self._history_button_clicked_cb, i)
            self.pack_end(button, True, False, 0)
        app.doc.input_stroke_ended += self._stroke_ended_cb
        self._update_history_images()

    def _stroke_ended_cb(self, doc, event):
        GLib.idle_add(self._update_history_images)

    def _update_history_images(self):
        bm = self._app.brushmanager
        assert self._history_images is not None
        assert len(bm.history) == len(self._history_images)
        for image, brush in zip(self._history_images, bm.history):
            image.set_from_managed_brush(brush)

    def _history_button_clicked_cb(self, button, i):
        bm = self._app.brushmanager
        try:
            brush = bm.history[i]
        except IndexError:
            pass
        else:
            bm.select_brush(brush)
        self.button_clicked()

    @event
    def button_clicked(self):
        """Event: a color history button was clicked"""


class ManagedBrushPreview (Gtk.Image):
    """Updateable widget displaying a brushmanager.ManagedBrush's preview"""

    ICON_SIZE = HISTORY_PREVIEW_SIZE
    TOOLTIP_ICON_SIZE = 48

    def __init__(self, brush=None):
        Gtk.Image.__init__(self)
        self.pixbuf = None
        self.image_size = None
        self.brush_name = None
        self.set_from_managed_brush(brush)
        s = self.ICON_SIZE
        self.set_size_request(s, s)
        self.connect("size-allocate", self.on_size_allocate)
        self.connect("query-tooltip", self.on_query_tooltip)
        self.set_property("has-tooltip", True)

    def set_from_managed_brush(self, brush):
        if brush is None:
            return
        self.pixbuf = brush.preview.copy()
        self.brush_name = brush.get_display_name()
        self._update()

    def on_size_allocate(self, widget, alloc):
        # if new_size != self.image_size:
        if self.image_size is None:
            # XXX dubious fix: what if the preview receives a new size in the
            # middle of its lifetime? Under GTK3 however, permitting this makes
            # the preview keep growing by about 4px each penstroke or brush
            # selection. Not sure why.
            self.image_size = alloc.width, alloc.height
            self._update()

    def _get_scaled_pixbuf(self, size):
        if self.pixbuf is None:
            pixbuf = GdkPixbuf.Pixbuf.new(
                GdkPixbuf.Colorspace.RGB,
                False, 8, size, size,
            )
            pixbuf.fill(0xffffffff)
            return pixbuf
        else:
            interp = GdkPixbuf.InterpType.BILINEAR
            return self.pixbuf.scale_simple(size, size, interp)

    def on_query_tooltip(self, widget, x, y, keyboard_mode, tooltip):
        s = self.TOOLTIP_ICON_SIZE
        scaled_pixbuf = self._get_scaled_pixbuf(s)
        tooltip.set_icon(scaled_pixbuf)
        tooltip.set_text(self.brush_name)
        # TODO: use markup, and summarize changes (i18n fun)
        return True

    def _update(self):
        if not self.image_size:
            return
        w, h = self.image_size
        s = min(w, h)
        scaled_pixbuf = self._get_scaled_pixbuf(s)
        self.set_from_pixbuf(scaled_pixbuf)


class ColorHistoryView (Gtk.HBox, ColorAdjuster):
    """A set of clickable ColorPreviews showing the usage history"""

    def __init__(self, app):
        Gtk.HBox.__init__(self)
        self._app = app
        self._history = []
        self.set_border_width(widgets.SPACING)
        s = HISTORY_PREVIEW_SIZE
        mgr = app.brush_color_manager
        for i, color in enumerate(mgr.get_history()):
            button = widgets.borderless_button()
            preview = ColorPreview(color)
            preview.set_size_request(s, s)
            button.add(preview)
            button.connect("clicked", self._button_clicked_cb, i)
            self.pack_end(button, True, False, 0)
            self._history.append(preview)
        self.set_color_manager(mgr)

    def color_history_updated(self):
        """Callback: history got updated via the ColorManager"""
        mgr = self.get_color_manager()
        for preview, color in zip(self._history, mgr.get_history()):
            preview.color = color

    def _button_clicked_cb(self, button, i):
        """Internal: on history button clicks, set the current color"""
        mgr = self.get_color_manager()
        history = mgr.get_history()
        try:
            color = history[i]
        except IndexError:
            pass
        else:
            mgr.set_color(color)
        self.button_clicked()

    @event
    def button_clicked(self):
        """Event: a color history button was clicked"""


class ColorPreview (Gtk.AspectFrame):
    """Updatable widget displaying a single color"""

    def __init__(self, color=None):
        """Initialize with a color (default is black"""
        Gtk.AspectFrame.__init__(self, xalign=0.5, yalign=0.5,
                                 ratio=1.0, obey_child=False)
        self.set_shadow_type(Gtk.ShadowType.IN)
        self.drawingarea = Gtk.DrawingArea()
        self.add(self.drawingarea)
        if color is None:
            color = RGBColor(0, 0, 0)
        self._color = color
        self.drawingarea.set_size_request(8, 8)
        self.drawingarea.connect("draw", self._draw_cb)

    def set_color(self, color):
        self._color = color
        self.drawingarea.queue_draw()

    def get_color(self):
        return self._color

    color = property(get_color, set_color)

    def _draw_cb(self, widget, cr):
        cr.set_source_rgb(*self._color.get_rgb())
        cr.paint()


class HistoryPanel (Gtk.VBox):

    __gtype_name__ = "MyPaintHistoryPanel"

    tool_widget_icon_name = "mypaint-history-symbolic"
    tool_widget_title = "Recent Brushes & Colors"
    tool_widget_description = ("The most recently used brush\n"
                               "presets and painting colors")

    def __init__(self):
        Gtk.VBox.__init__(self)
        from gui.application import get_app
        app = get_app()
        color_hist_view = ColorHistoryView(app)
        self.pack_start(color_hist_view, True, False, 0)
        brush_hist_view = BrushHistoryView(app)
        self.pack_start(brush_hist_view, True, False, 0)
