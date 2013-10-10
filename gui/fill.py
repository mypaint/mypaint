# This file is part of MyPaint.
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Flood fill tool"""

## Imports

import gi
from gi.repository import Gtk
from gi.repository import Gdk
from gettext import gettext as _

import canvasevent

from application import get_app

## Class defs

class FloodFillMode (canvasevent.SwitchableModeMixin,
                     canvasevent.ScrollableModeMixin,
                     canvasevent.SingleClickMode):
    """Mode for flood-filling with the current brush color"""

    ## Class constants

    __action_name__ = "FloodFillMode"
    permitted_switch_actions = set([
        'RotateViewMode', 'ZoomViewMode', 'PanViewMode',
        'ColorPickMode', 'ShowPopupMenu',
        ])

    _OPTIONS_WIDGET = None
    _CURSOR_FILL_PERMITTED = "cursor_crosshair_precise_open"
    _CURSOR_FILL_FORBIDDEN = "cursor_arrow_forbidden"

    ## Instance vars (and defaults)

    _current_cursor = _CURSOR_FILL_PERMITTED
    _tdws = None
    _fill_permitted = True
    _x = None
    _y = None

    @property
    def cursor(self):
        name = self._current_cursor
        app = get_app()
        return app.cursors.get_action_cursor(self.__action_name__, name)

    ## Method defs

    def enter(self, **kwds):
        super(FloodFillMode, self).enter(**kwds)
        self._tdws = set([self.doc.tdw])
        self._update_ui()

    @classmethod
    def get_name(cls):
        return _(u'Flood Fill')

    def get_usage(self):
        return _(u"Fill areas with color")

    def __init__(self, ignore_modifiers=False, **kwds):
        super(FloodFillMode, self).__init__(**kwds)
    
    def clicked_cb(self, tdw, event):
        """Update cursor and maybe fill with settings"""
        x, y = tdw.display_to_model(event.x, event.y)
        self._x = x
        self._y = y
        self._tdws.add(tdw)
        self._update_ui()
        color = self.doc.app.brush_color_manager.get_color()
        opts = self.get_options_widget()
        self.doc.model.flood_fill(x, y, color.get_rgb(),
                                  tolerance=opts.tolerance,
                                  sample_merged=opts.sample_merged,
                                  make_new_layer=opts.make_new_layer)
        opts.make_new_layer = False
        return False

    def motion_notify_cb(self, tdw, event):
        """Track position, and update cursor"""
        x, y = tdw.display_to_model(event.x, event.y)
        self._x = x
        self._y = y
        self._tdws.add(tdw)
        self._update_ui()
        return super(FloodFillMode, self).motion_notify_cb(tdw, event)

    def model_structure_changed_cb(self, doc):
        super(FloodFillMode, self).model_structure_changed_cb(doc)
        self._update_ui()

    def _update_ui(self):
        x, y = self._x, self._y
        if None in (x, y):
            x, y = self.current_position()
        model = self.doc.model

        # Determine which layer will receive the fill based on the options
        opts = self.get_options_widget()
        target_layer = model.layer
        if opts.make_new_layer:
            target_layer = None

        # Determine whether the target layer can be filled
        permitted = True
        if target_layer is not None:
            permitted = target_layer.visible and not target_layer.locked 
        if model.frame_enabled:
            fx1, fy1, fw, fh = model.get_frame()
            fx2, fy2 = fx1+fw, fy1+fh
            permitted &= x >= fx1 and y >= fy1 and x < fx2 and y < fy2
        self._fill_permitted = permitted

        # Update cursor of any TDWs we've crossed
        if self._fill_permitted:
            cursor = self._CURSOR_FILL_PERMITTED
        else:
            cursor = self._CURSOR_FILL_FORBIDDEN

        if cursor != self._current_cursor:
            self._current_cursor = cursor
            for tdw in self._tdws:
                tdw.set_override_cursor(self.cursor)


    ## Mode options

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = FloodFillOptionsWidget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET


class FloodFillOptionsWidget (Gtk.Grid):
    """Configuration widget for the flood fill tool"""

    TOLERANCE_PREF = 'flood_fill.tolerance'
    SAMPLE_MERGED_PREF = 'flood_fill.sample_merged'
    # "make new layer" is a temportary toggle, and is not saved to prefs

    DEFAULT_TOLERANCE = 0.05
    DEFAULT_SAMPLE_MERGED = False
    DEFAULT_MAKE_NEW_LAYER = False

    def __init__(self):
        Gtk.Grid.__init__(self)

        self.set_row_spacing(6)
        self.set_column_spacing(6)
        from application import get_app
        self.app = get_app()
        prefs = self.app.preferences

        row = 0
        label = Gtk.Label()
        label.set_markup(_("Tolerance:"))
        label.set_tooltip_text(
            _("How much pixel colors are allowed to vary from the start\n"
              "before Flood Fill will refuse to fill them"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.attach(label, 0, row, 1, 1)
        value = float(prefs.get(self.TOLERANCE_PREF, self.DEFAULT_TOLERANCE))
        adj = Gtk.Adjustment(value, 0.0, 1.0, 0.05, 0.05, 0)
        adj.connect("value-changed", self._tolerance_changed_cb)
        self._tolerance_adj = adj
        scale = Gtk.Scale()
        scale.set_hexpand(True)
        scale.set_adjustment(adj)
        scale.set_draw_value(False)
        self.attach(scale, 1, row, 1, 1)

        row += 1
        label = Gtk.Label()
        label.set_markup(_("Source:"))
        label.set_tooltip_text(_("Which visible layers should be filled"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.attach(label, 0, row, 1, 1)

        text = _("Sample Merged")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("When considering which area to fill, use a\n"
              "temporary merge of all the visible layers."))
        self.attach(checkbut, 1, row, 1, 1)
        active = bool(prefs.get(self.SAMPLE_MERGED_PREF,
                                self.DEFAULT_SAMPLE_MERGED))
        checkbut.set_active(active)
        checkbut.connect("toggled", self._sample_merged_toggled_cb)
        self._sample_merged_toggle = checkbut

        row += 1
        label = Gtk.Label()
        label.set_markup(_("Target:"))
        label.set_tooltip_text(_("Where the output should go"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.attach(label, 0, row, 1, 1)

        text = _("New Layer (once)")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Create a new layer with the results of the fill.\n"
              "This is turned off automatically after use."))
        self.attach(checkbut, 1, row, 1, 1)
        active = self.DEFAULT_MAKE_NEW_LAYER
        checkbut.set_active(active)
        self._make_new_layer_toggle = checkbut

        row += 1
        align = Gtk.Alignment(0.5, 1.0, 1.0, 0.0)
        align.set_vexpand(True)
        button = Gtk.Button(_("Reset"))
        button.connect("clicked", self._reset_clicked_cb)
        button.set_tooltip_text(_("Reset options to their defaults"))
        align.add(button)
        self.attach(align, 0, row, 2, 1)

    @property
    def tolerance(self):
        return float(self._tolerance_adj.get_value())

    @property
    def make_new_layer(self):
        return bool(self._make_new_layer_toggle.get_active())

    @make_new_layer.setter
    def make_new_layer(self, value):
        self._make_new_layer_toggle.set_active(bool(value))

    @property
    def sample_merged(self):
        return bool(self._sample_merged_toggle.get_active())

    def _tolerance_changed_cb(self, adj):
        self.app.preferences[self.TOLERANCE_PREF] = self.tolerance

    def _sample_merged_toggled_cb(self, checkbut):
        self.app.preferences[self.SAMPLE_MERGED_PREF] = self.sample_merged

    def _reset_clicked_cb(self, button):
        self._tolerance_adj.set_value(self.DEFAULT_TOLERANCE)
        self._make_new_layer_toggle.set_active(self.DEFAULT_MAKE_NEW_LAYER)
        self._sample_merged_toggle.set_active(self.DEFAULT_SAMPLE_MERGED)
