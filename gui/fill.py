# This file is part of MyPaint.
# Copyright (C) 2018 by the Mypaint Development Team
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Flood fill tool"""

## Imports
from __future__ import division, print_function

import gi
from gi.repository import Gtk
from gi.repository import Gdk
from gettext import gettext as _

import gui.mode
import gui.cursor

import lib.floodfill


## Class defs

class FloodFillMode (gui.mode.ScrollableModeMixin,
                     gui.mode.SingleClickMode):
    """Mode for flood-filling with the current brush color"""

    ## Class constants

    ACTION_NAME = "FloodFillMode"
    GC_ACTION_NAME = "FloodFillGCMode"

    permitted_switch_actions = set([
        'RotateViewMode', 'ZoomViewMode', 'PanViewMode',
        'ColorPickMode', 'ShowPopupMenu',
        ])

    _OPTIONS_WIDGET = None
    _CURSOR_FILL_PERMITTED = gui.cursor.Name.CROSSHAIR_OPEN_PRECISE
    _CURSOR_FILL_FORBIDDEN = gui.cursor.Name.ARROW_FORBIDDEN

    ## Instance vars (and defaults)

    pointer_behavior = gui.mode.Behavior.PAINT_NOBRUSH
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    _current_cursor = (False, _CURSOR_FILL_PERMITTED)
    _tdws = None
    _fill_permitted = True
    _x = None
    _y = None

    @property
    def cursor(self):
        gc_on, name = self._current_cursor
        from gui.application import get_app
        app = get_app()
        action_name = self.GC_ACTION_NAME if gc_on else self.ACTION_NAME
        return app.cursors.get_action_cursor(action_name, name)

    ## Method defs

    def enter(self, doc, **kwds):
        super(FloodFillMode, self).enter(doc, **kwds)
        self._tdws = set([self.doc.tdw])
        rootstack = self.doc.model.layer_stack
        rootstack.current_path_updated += self._update_ui
        rootstack.layer_properties_changed += self._update_ui
        self._update_ui()

    def leave(self, **kwds):
        rootstack = self.doc.model.layer_stack
        rootstack.current_path_updated -= self._update_ui
        rootstack.layer_properties_changed -= self._update_ui
        return super(FloodFillMode, self).leave(**kwds)

    @classmethod
    def get_name(cls):
        return _(u'Flood Fill')

    def get_usage(self):
        return _(u"Fill areas with color")

    def __init__(self, ignore_modifiers=False, **kwds):
        super(FloodFillMode, self).__init__(**kwds)
        opts = self.get_options_widget()
        self._current_cursor = (opts.gap_closing, self._CURSOR_FILL_PERMITTED)

    def clicked_cb(self, tdw, event):
        """Flood-fill with the current settings where clicked

        If the current layer is not fillable, a new layer will always be
        created for the fill.
        """
        from gui.application import get_app
        self.app = get_app()
        try:
            self.EOTF = self.app.preferences['display.colorspace_EOTF']
        except: 
            self.EOTF = 2.2
        x, y = tdw.display_to_model(event.x, event.y)
        self._x = x
        self._y = y
        self._tdws.add(tdw)
        self._update_ui()
        color = self.doc.app.brush_color_manager.get_color()
        opts = self.get_options_widget()
        make_new_layer = opts.make_new_layer
        rootstack = tdw.doc.layer_stack
        if not rootstack.current.get_fillable():
            make_new_layer = True
        rgb = color.get_rgb()
        rgb = (rgb[0]**self.EOTF, rgb[1]**self.EOTF, rgb[2]**self.EOTF)
        tdw.doc.flood_fill(x, y, rgb,
                           tolerance=opts.tolerance,
                           offset=opts.offset, feather=opts.feather,
                           gap_closing_options=opts.gap_closing_options,
                           sample_merged=opts.sample_merged,
                           make_new_layer=make_new_layer)
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

    def _update_ui(self, *_ignored):
        """Updates the UI from the model"""
        x, y = self._x, self._y
        if None in (x, y):
            x, y = self.current_position()
        model = self.doc.model

        # Determine which layer will receive the fill based on the options
        opts = self.get_options_widget()
        target_layer = model.layer_stack.current
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
            cursor = (opts.gap_closing, self._CURSOR_FILL_PERMITTED)
        else:
            cursor = (opts.gap_closing, self._CURSOR_FILL_FORBIDDEN)

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
    OFFSET_PREF = 'flood_fill.offset'
    FEATHER_PREF = 'flood_fill.feather'

    # Gap closing related parameters
    GAP_CLOSING_PREF = 'flood_fill.gap_closing'
    GAP_SIZE_PREF = 'flood_fill.gap_size'
    RETRACT_SEEPS_PREF = 'flood_fill.retract_seeps'
    # "make new layer" is a temportary toggle, and is not saved to prefs

    DEFAULT_TOLERANCE = 0.05
    DEFAULT_SAMPLE_MERGED = False
    DEFAULT_MAKE_NEW_LAYER = False
    DEFAULT_OFFSET = 0
    DEFAULT_FEATHER = 0

    # Gap closing related defaults
    DEFAULT_GAP_CLOSING = False
    DEFAULT_GAP_SIZE = 5
    DEFAULT_RETRACT_SEEPS = True

    def __init__(self):
        Gtk.Grid.__init__(self)

        self.set_row_spacing(6)
        self.set_column_spacing(6)
        from gui.application import get_app
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
        value = prefs.get(self.TOLERANCE_PREF, self.DEFAULT_TOLERANCE)
        value = float(value)
        adj = Gtk.Adjustment(value=value, lower=0.0, upper=1.0,
                             step_increment=0.05, page_increment=0.05,
                             page_size=0)
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
              "temporary merge of all the visible layers\n"
              "underneath the current layer"))
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
        self.attach(Gtk.Separator(), 0, row, 2, 1)

        row += 1
        label = Gtk.Label()
        label.set_markup(_("Offset:"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.attach(label, 0, row, 1, 1)

        TILE_SIZE = lib.floodfill.TILE_SIZE
        value = prefs.get(self.OFFSET_PREF, self.DEFAULT_OFFSET)
        adj = Gtk.Adjustment(value=value,
                             lower=-TILE_SIZE, upper=TILE_SIZE,
                             step_increment=1, page_increment=4)
        adj.connect("value-changed", self._offset_changed_cb)
        self._offset_adj = adj
        spinbut = Gtk.SpinButton()
        spinbut.set_tooltip_text(
            _("The distance in pixels to grow/shrink the fill"))
        spinbut.set_hexpand(True)
        spinbut.set_adjustment(adj)
        spinbut.set_numeric(True)
        self.attach(spinbut, 1, row, 1, 1)

        row += 1
        label = Gtk.Label()
        label.set_markup(_("Feather:"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.attach(label, 0, row, 1, 1)

        value = prefs.get(self.FEATHER_PREF, self.DEFAULT_FEATHER)
        adj = Gtk.Adjustment(value=value,
                             lower=0, upper=TILE_SIZE,
                             step_increment=1, page_increment=4)
        adj.connect("value-changed", self._feather_changed_cb)
        self._feather_adj = adj
        spinbut = Gtk.SpinButton()
        spinbut.set_tooltip_text(
            _("The amount of blur to apply to the fill before painting it in"))
        spinbut.set_hexpand(True)
        spinbut.set_adjustment(adj)
        spinbut.set_numeric(True)
        self.attach(spinbut, 1, row, 1, 1)

        row += 1
        self.attach(Gtk.Separator(), 0, row, 2, 1)

        row += 1
        gap_closing_params = Gtk.Grid()
        self._gap_closing_grid = gap_closing_params

        text = _("Use gap closing")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Try to detect gaps and not fill past them"))
        self._gap_closing_toggle = checkbut
        checkbut.connect("toggled", self._gap_closing_toggled_cb)
        active = prefs.get(self.GAP_CLOSING_PREF, self.DEFAULT_GAP_CLOSING)
        checkbut.set_active(active)
        gap_closing_params.set_sensitive(active)
        self.attach(checkbut, 0, row, 2, 1)

        row += 1
        self.attach(gap_closing_params, 0, row, 2, 1)

        gcp_row = 0
        label = Gtk.Label()
        label.set_markup(_("Max gap size:"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        gap_closing_params.attach(label, 0, gcp_row, 1, 1)

        value = prefs.get(self.GAP_SIZE_PREF, self.DEFAULT_GAP_SIZE)
        adj = Gtk.Adjustment(value=value,
                             lower=1, upper=int(TILE_SIZE/2),
                             step_increment=1, page_increment=4)
        adj.connect("value-changed", self._max_gap_size_changed_cb)
        self._max_gap_adj = adj
        spinbut = Gtk.SpinButton()
        spinbut.set_tooltip_text(
            _("The size of the largest gaps that can be detected"))
        spinbut.set_hexpand(True)
        spinbut.set_adjustment(adj)
        spinbut.set_numeric(True)
        gap_closing_params.attach(spinbut, 1, gcp_row, 1, 1)

        gcp_row += 1
        text = _("Retract seeps")
        checkbut = Gtk.CheckButton.new_with_label(text)
        active = prefs.get(self.RETRACT_SEEPS_PREF, self.DEFAULT_RETRACT_SEEPS)
        checkbut.set_active(active)
        checkbut.set_tooltip_text(
            _("Try to pull back the fill from out of the gaps"))
        checkbut.connect("toggled", self._retract_seeps_toggled_cb)
        self._retract_seeps_toggle = checkbut
        gap_closing_params.attach(checkbut, 1, gcp_row, 1, 1)

        row += 1
        align = Gtk.Alignment.new(0.5, 1.0, 1.0, 0.0)
        align.set_vexpand(True)
        button = Gtk.Button(label=_("Reset"))
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

    @property
    def offset(self):
        return int(self._offset_adj.get_value())

    @property
    def feather(self):
        return int(self._feather_adj.get_value())

    @property
    def gap_closing(self):
        return bool(self._gap_closing_toggle.get_active())

    @property
    def max_gap_size(self):
        return int(self._max_gap_adj.get_value())

    @property
    def retract_seeps(self):
        return bool(self._retract_seeps_toggle.get_active())

    @property
    def gap_closing_options(self):
        if self.gap_closing:
            return lib.floodfill.GapClosingOptions(
                self.max_gap_size, self.retract_seeps)
        else:
            return None

    def _tolerance_changed_cb(self, adj):
        self.app.preferences[self.TOLERANCE_PREF] = self.tolerance

    def _sample_merged_toggled_cb(self, checkbut):
        self.app.preferences[self.SAMPLE_MERGED_PREF] = self.sample_merged

    def _offset_changed_cb(self, adj):
        self.app.preferences[self.OFFSET_PREF] = self.offset

    def _feather_changed_cb(self, adj):
        self.app.preferences[self.FEATHER_PREF] = self.feather

    def _gap_closing_toggled_cb(self, adj):
        self._gap_closing_grid.set_sensitive(self.gap_closing)
        self.app.preferences[self.GAP_CLOSING_PREF] = self.gap_closing

    def _max_gap_size_changed_cb(self, adj):
        self.app.preferences[self.GAP_SIZE_PREF] = self.max_gap_size

    def _retract_seeps_toggled_cb(self, adj):
        self.app.preferences[self.RETRACT_SEEPS_PREF] = self.retract_seeps

    def _reset_clicked_cb(self, button):
        self._tolerance_adj.set_value(self.DEFAULT_TOLERANCE)
        self._make_new_layer_toggle.set_active(self.DEFAULT_MAKE_NEW_LAYER)
        self._sample_merged_toggle.set_active(self.DEFAULT_SAMPLE_MERGED)
        self._offset_adj.set_value(self.DEFAULT_OFFSET)
        self._feather_adj.set_value(self.DEFAULT_FEATHER)
        # Gap closing params
        self._max_gap_adj.set_value(self.DEFAULT_GAP_SIZE)
        self._retract_seeps_toggle.set_active(self.DEFAULT_RETRACT_SEEPS)
        self._gap_closing_toggle.set_active(self.DEFAULT_GAP_CLOSING)
