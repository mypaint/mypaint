# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Footer widget behaviour."""


## Imports
from __future__ import division, print_function

import math
import logging

import cairo

from lib.gibindings import Gdk
from lib.gibindings import GdkPixbuf

import gui.brushmanager
from gui.quickchoice import BrushChooserPopup  # noqa

import lib.xml
from lib.gettext import C_
from gettext import gettext as _

logger = logging.getLogger(__name__)


## Class definitions

class BrushIndicatorPresenter (object):
    """Behaviour for a clickable footer brush indicator

    This presenter's view is a DrawingArea instance
    which is used to display the current brush's preview image.
    Its model is the BrushManager instance belonging to the main app.
    Both the view and the model
    must be set after construction
    and before or during realization.

    When the view DrawingArea is clicked,
    a QuickBrushChooser is popped up near it,
    allowing the user to change the current brush.

    The code assumes that
    a single instance of the view DrawingArea
    is packed into the lower right corner
    (lower left, for rtl locales)
    of the main drawing window's footer bar.

    48px is a good width for the view widget.
    If the preview is too tall to display fully,
    it is drawn truncated with a cute gradient effect.
    The user can hover the pointer to show a tooltip
    with the full preview image.

    """

    _TOOLTIP_ICON_SIZE = 48
    _EDGE_HIGHLIGHT_RGBA = (1, 1, 1, 0.25)
    _OUTLINE_RGBA = (0, 0, 0, 0.4)
    _DEFAULT_BRUSH_DISPLAY_NAME = _("Unknown Brush")
    # FIXME: Use brushmanager.py's source string while we are in string
    # FIXME: freeze.

    ## Initialization

    def __init__(self):
        """Basic initialization"""
        super(BrushIndicatorPresenter, self).__init__()
        self._brush_preview = None
        self._brush_name = self._DEFAULT_BRUSH_DISPLAY_NAME
        self._brush_desc = None
        self._drawing_area = None
        self._brush_manager = None
        self._chooser = None
        self._click_button = None

    def set_drawing_area(self, da):
        """Set the view DrawingArea.

        :param Gtk.DrawingArea da: the drawing area

        The view should be set before or during its realization.

        """
        self._drawing_area = da
        da.set_has_window(True)
        da.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK
        )
        da.connect("draw", self._draw_cb)
        da.connect("query-tooltip", self._query_tooltip_cb)
        da.set_property("has-tooltip", True)
        da.connect("button-press-event", self._button_press_cb)
        da.connect("button-release-event", self._button_release_cb)

    def set_brush_manager(self, bm):
        """Set the model BrushManager.

        :param gui.brushmanager.BrushManager bm: the model BrushManager

        """
        self._brush_manager = bm
        bm.brush_selected += self._brush_selected_cb

    def set_chooser(self, chooser):
        """Set an optional popup, to be shown when clicked.

        :param BrushChooserPopup chooser: popup to show

        """
        self._chooser = chooser

    ## View event handlers

    def _draw_cb(self, da, cr):
        """Paint a preview of the current brush to the view."""
        if not self._brush_preview:
            cr.set_source_rgb(1, 0, 1)
            cr.paint()
            return
        aw = da.get_allocated_width()
        ah = da.get_allocated_height()

        # Work in a temporary group so that
        # the result can be masked with a gradient later.
        cr.push_group()

        # Paint a shadow line around the edge of
        # where the the brush preview will go.
        # There's an additional top border of one pixel
        # for alignment with the color preview widget
        # in the other corner.
        cr.rectangle(1.5, 2.5, aw-3, ah)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.set_source_rgba(*self._OUTLINE_RGBA)
        cr.set_line_width(3)
        cr.stroke()

        # Scale and align the brush preview in its own saved context
        cr.save()
        # Clip rectangle for the bit in the middle of the shadow.
        # Note that the bottom edge isn't shadowed.
        cr.rectangle(1, 2, aw-2, ah)
        cr.clip()
        # Scale and align the preview to the top of that clip rect.
        preview = self._brush_preview
        pw = preview.get_width()
        ph = preview.get_height()
        area_size = float(max(aw, ah)) - 2
        preview_size = float(max(pw, ph))
        x = math.floor(-pw/2.0)
        y = 0
        cr.translate(aw/2.0, 2)
        scale = area_size / preview_size
        cr.scale(scale, scale)
        Gdk.cairo_set_source_pixbuf(cr, preview, x, y)
        cr.paint()
        cr.restore()

        # Finally a highlight around the edge in the house style
        # Note that the bottom edge isn't highlighted.
        cr.rectangle(1.5, 2.5, aw-3, ah)
        cr.set_line_width(1)
        cr.set_source_rgba(*self._EDGE_HIGHLIGHT_RGBA)
        cr.stroke()

        # Paint the group within a gradient mask
        cr.pop_group_to_source()
        mask = cairo.LinearGradient(0, 0, 0, ah)
        mask.add_color_stop_rgba(0.0, 1, 1, 1, 1.0)
        mask.add_color_stop_rgba(0.8, 1, 1, 1, 1.0)
        mask.add_color_stop_rgba(0.95, 1, 1, 1, 0.5)
        mask.add_color_stop_rgba(1.0, 1, 1, 1, 0.1)
        cr.mask(mask)

    def _query_tooltip_cb(self, da, x, y, keyboard_mode, tooltip):
        s = self._TOOLTIP_ICON_SIZE
        scaled_pixbuf = self._get_scaled_pixbuf(s)
        tooltip.set_icon(scaled_pixbuf)
        brush_name = self._brush_name
        if not brush_name:
            brush_name = self._DEFAULT_BRUSH_DISPLAY_NAME
            # Rare cases, see https://github.com/mypaint/mypaint/issues/402.
            # Probably just after init.
        template_params = {"brush_name": lib.xml.escape(brush_name)}
        markup_template = C_(
            "current brush indicator: tooltip (no-description case)",
            u"<b>{brush_name}</b>",
        )
        if self._brush_desc:
            markup_template = C_(
                "current brush indicator: tooltip (description case)",
                u"<b>{brush_name}</b>\n{brush_desc}",
            )
            template_params["brush_desc"] = lib.xml.escape(self._brush_desc)
        markup = markup_template.format(**template_params)
        tooltip.set_markup(markup)
        # TODO: summarize changes?
        return True

    def _button_press_cb(self, widget, event):
        if not self._chooser:
            return False
        if event.button != 1:
            return False
        if event.type != Gdk.EventType.BUTTON_PRESS:
            return False
        self._click_button = event.button
        return True

    def _button_release_cb(self, widget, event):
        if event.button != self._click_button:
            return False
        self._click_button = None
        chooser = self._chooser
        if not chooser:
            return
        if chooser.get_visible():
            chooser.hide()
        else:
            chooser.popup(
                widget = self._drawing_area,
                above = True,
                textwards = False,
                event = event,
            )
        return True

    ## Model event handlers

    def _brush_selected_cb(self, bm, brush, brushinfo):
        if brush is None:
            return
        self._brush_preview = brush.preview.copy()
        self._brush_name = brush.get_display_name()
        self._brush_desc = brush.description
        self._drawing_area.queue_draw()

    ## Utility methods

    def _get_scaled_pixbuf(self, size):
        if self._brush_preview is None:
            pixbuf = GdkPixbuf.Pixbuf.new(
                GdkPixbuf.Colorspace.RGB,
                False, 8, size, size,
            )
            pixbuf.fill(0x00ff00ff)
            return pixbuf
        else:
            interp = GdkPixbuf.InterpType.BILINEAR
            return self._brush_preview.scale_simple(size, size, interp)
