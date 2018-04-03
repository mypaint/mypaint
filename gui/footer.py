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
logger = logging.getLogger(__name__)

import cairo
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GdkPixbuf

from . import pixbuflist  # noqa: E402
from . import brushmanager  # noqa: E402

import lib.xml
from lib.gettext import C_
from gettext import gettext as _

## Class definitions

class BrushIndicatorPresenter (object):
    """Behaviour for a clickable footer brush indicator

    This presenter's view is a BrushWidget instance
    which is used to display the current brush's preview image.
    Its model is the BrushManager instance belonging to the main app.
    Both the view and the model
    must be set after construction
    and before or during realization.

    When the view BrushWidget is clicked,
    a QuickBrushChooser is popped up near it,
    allowing the user to change the current brush.

    The code assumes that
    a single instance of the view BrushWidget
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
        self._brush_widget = None
        self._brush_manager = None
        self._chooser = None
        self._click_button = None

    def set_brush_widget(self, brush_widget):
        """Set the view BrushWidget.

        :param gui.BrushWidget brush_widget: the model BrushManager

        The view should be set before or during its realization.

        """
        self._brush_widget = brush_widget
        brush_widget.set_has_window(True)
        brush_widget.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK
        )
        brush_widget.connect("draw", self._draw_cb)
        brush_widget.connect("query-tooltip", self._query_tooltip_cb)
        brush_widget.set_property("has-tooltip", True)
        brush_widget.connect("button-press-event", self._button_press_cb)
        brush_widget.connect("button-release-event", self._button_release_cb)
        brush_widget.connect("drag-begin", self._drag_begin_cb)
        brush_widget.connect("drag-data-get", self._drag_data_get_cb)
        brush_widget.drag_source_set(
            Gdk.ModifierType.BUTTON1_MASK,
            [Gtk.TargetEntry.new(*e) for e in pixbuflist.DRAG_TARGETS],
            Gdk.DragAction.COPY
        )

    def set_brush_manager(self, bm):
        """Set the model BrushManager.

        :param gui.brushmanager.BrushManager bm: the model BrushManager

        """
        self._brush_manager = bm
        bm.brush_selected += self._brush_selected_cb

    def set_chooser(self, chooser):
        """Set an optional popup, to be shown when clicked.

        :param gui.quickchoice.BrushChooserPopup chooser: popup to show

        """
        self._chooser = chooser

    ## View event handlers

    def _drag_begin_cb(self, widget, context):
        preview = self._brush_manager.selected_brush.preview
        preview = preview.scale_simple(
            preview.get_width() // 2,
            preview.get_height() // 2,
            GdkPixbuf.InterpType.BILINEAR,
        )
        Gtk.drag_set_icon_pixbuf(context, preview, 0, 0)

    # Not sure if needed?
    def _drag_data_get_cb(self, widget, context, selection, info, time):
        if info != pixbuflist.DRAG_ITEM_ID:
            return False
        current_brush = self._brush_manager.selected_brush
        dragid = str(id(current_brush))
        selection.set_text(dragid, -1)
        return True

    def _draw_cb(self, brush_widget, cr):
        """Paint a preview of the current brush to the view."""
        if not self._brush_preview:
            cr.set_source_rgb(1, 0, 1)
            cr.paint()
            return
        aw = brush_widget.get_allocated_width()
        ah = brush_widget.get_allocated_height()

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

    def _query_tooltip_cb(self, brush_widget, x, y, keyboard_mode, tooltip):
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
        if event.button != 1 and event.button != 3:  # left or right click
            return False
        if event.type != Gdk.EventType.BUTTON_PRESS:
            return False
        self._click_button = event.button
        return False  # don't consume in case we want to drag

    def _button_release_cb(self, widget, event):
        if event.button != self._click_button:
            return False
        self._click_button = None
        if event.button == 1:
            chooser = self._chooser
            if not chooser:
                return
            if chooser.get_visible():
                chooser.hide()
            else:
                chooser.popup(
                    widget=self._brush_widget,
                    above=True,
                    textwards=False,
                    event=event,
                )
        elif event.button == 3:
            self._show_context_menu()
        return True

    def _show_context_menu(self):
        menu = Gtk.Menu()
        current_brush = self._brush_manager.selected_brush
        faves = self._brush_manager.get_group_brushes(
            brushmanager.FAVORITES_BRUSH_GROUP
        )
        in_faves = current_brush in faves
        fave_item_message = (
            C_(
                "brush group: context menu for a single brush",
                "Add from Favorites"
            ) if not in_faves
            else C_(
                "brush group: context menu for a single brush",
                "Remove from Favorites"
            )
        )
        favesItem = Gtk.MenuItem(fave_item_message)
        fav_cb = self._favorite_cb if not in_faves else self._unfavorite_cb
        favesItem.connect("activate", fav_cb)
        menu.append(favesItem)

        settingsItem = Gtk.MenuItem(C_(
            "brush group: context menu for a single brush",
            "Edit Brush Settings",
        ))
        settingsItem.connect("activate", self._brush_settings_cb)
        menu.append(settingsItem)

        time = Gtk.get_current_event_time()
        menu.show_all()
        menu.popup(
            parent_menu_shell=None,
            parent_menu_item=None,
            func=None,
            button=3,
            activate_time=time,
            data=None,
        )

    def _favorite_cb(self, menuitem):
        self._brush_manager.favorite_brush(self._brush_manager.selected_brush)

    def _unfavorite_cb(self, menuitem):
        self._brush_manager.unfavorite_brush(
            self._brush_manager.selected_brush
        )

    def _brush_settings_cb(self, menuitem):
        from gui import application
        brush_editor = application.get_app().brush_settings_window
        brush_editor.show()

    ## Model event handlers

    def _brush_selected_cb(self, bm, brush, brushinfo):
        if brush is None:
            return
        self._brush_preview = brush.preview.copy()
        self._brush_name = brush.get_display_name()
        self._brush_desc = brush.description
        self._brush_widget.queue_draw()
        self._brush_widget.brush = self._brush_manager.selected_brush

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
