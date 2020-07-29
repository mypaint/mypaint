# This file is part of MyPaint.
# Copyright (C) 2018 by the Mypaint Development Team
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Flood fill tool"""

# Imports
from __future__ import division, print_function

import weakref
from lib.gibindings import Gtk
from lib.gibindings import Pango
from lib.gibindings import GLib
from gettext import gettext as _
from lib.gettext import C_

import cairo

import gui.mode
import gui.cursor
from gui.blendmodehandler import BlendModes
import gui.layers
import gui.overlays
from gui.sliderwidget import InputSlider

import lib.eotf
import lib.floodfill
import lib.helpers
import lib.mypaintlib
import lib.layer
import lib.modes


# Class defs

class FloodFillMode (
        gui.mode.ScrollableModeMixin, gui.mode.DragMode):
    """Mode for flood-filling with the current brush color"""

    # Class constants

    ACTION_NAME = "FloodFillMode"
    GC_ACTION_NAME = "FloodFillGCMode"

    SPRING_LOADED = False

    permitted_switch_actions = set([
        'RotateViewMode', 'ZoomViewMode', 'PanViewMode',
        'ColorPickMode', 'ShowPopupMenu',
        ])

    _OPTIONS_WIDGET = None
    _CURSOR_FILL_NORMAL = gui.cursor.Name.CROSSHAIR_OPEN_PRECISE
    _CURSOR_FILL_ERASER = gui.cursor.Name.ERASER
    _CURSOR_FILL_ALPHA_LOCKED = gui.cursor.Name.ALPHA_LOCK
    _CURSOR_FILL_COLORIZE = gui.cursor.Name.COLORIZE
    _CURSOR_FILL_FORBIDDEN = gui.cursor.Name.ARROW_FORBIDDEN

    _MODE_CURSORS = [
        _CURSOR_FILL_NORMAL,
        _CURSOR_FILL_ERASER,
        _CURSOR_FILL_ALPHA_LOCKED,
        _CURSOR_FILL_COLORIZE,
    ]

    # Instance vars (and defaults)

    pointer_behavior = gui.mode.Behavior.PAINT_NOBRUSH
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    _current_cursor = (False, _CURSOR_FILL_NORMAL)
    _tdws = None
    _fill_permitted = True
    _x = None
    _y = None

    @property
    def active_cursor(self):
        return self.cursor

    @property
    def inactive_cursor(self):
        return self.cursor

    @property
    def cursor(self):
        gc_on, name = self._current_cursor
        action_name = self.GC_ACTION_NAME if gc_on else self.ACTION_NAME
        return self.app.cursors.get_action_cursor(action_name, name)

    def get_current_cursor(self):
        return self._MODE_CURSORS[self.bm.active_mode.mode_type]

    # Method defs

    def enter(self, doc, **kwds):
        super(FloodFillMode, self).enter(doc, **kwds)
        self._tdws = set([self.doc.tdw])
        self.app.blendmodemanager.register(self.bm)
        rootstack = self.doc.model.layer_stack
        rootstack.current_path_updated += self._update_ui
        rootstack.layer_properties_changed += self._update_ui
        self._update_ui()

    def leave(self, **kwds):
        self.app.blendmodemanager.deregister(self.bm)
        rootstack = self.doc.model.layer_stack
        rootstack.current_path_updated -= self._update_ui
        rootstack.layer_properties_changed -= self._update_ui
        return super(FloodFillMode, self).leave(**kwds)

    @classmethod
    def get_name(cls):
        return C_(
            "Name of the fill mode\n"
            "In other software, 'Flood Fill' is also known as "
            "'Bucket Fill' or just 'Fill'",
            u'Flood Fill'
        )

    def get_usage(self):
        return C_(
            "Usage description of the Flood Fill mode",
            u"Fill areas with color"
        )

    def __init__(self, ignore_modifiers=False, **kwds):
        super(FloodFillMode, self).__init__(**kwds)
        opts = self.get_options_widget()
        self._current_cursor = (opts.gap_closing, self._CURSOR_FILL_NORMAL)
        from gui.application import get_app
        self.app = get_app()
        self.bm = self.get_blend_modes()
        self.bm.mode_changed += self.update_blend_mode
        self._prev_release = (0, None)
        self._seed_pixels = set()
        self._overlay = None
        self._target_pos = None

    def get_blend_modes(self):
        return self.get_options_widget().get_blend_modes()

    def update_blend_mode(self, mode_manager, old_mode, new_mode):
        if old_mode is not new_mode:
            self._update_cursor(self.get_options_widget())

    def drag_update_cb(self, tdw, event, ev_x, ev_y, dx, dy):
        """Add pixel coordinate to seed set (if not there already)"""
        x, y = tdw.display_to_model(ev_x, ev_y)
        seed_candidate = (int(x), int(y))
        self._overlay.add_point((ev_x, ev_y))
        if self._fill_permitted and seed_candidate not in self._seed_pixels:
            self._seed_pixels.add(seed_candidate)

    def drag_start_cb(self, tdw, event):
        """Create overlay and set initial target pixel & seed"""
        self._seed_pixels = set()
        x, y = tdw.display_to_model(self.start_x, self.start_y)
        # Permit setting the target color (the one pixels are tested against)
        # from outside a frame, but no corresponding seed is added
        self._target_pos = (x, y)
        self._overlay = FloodFillOverlay(tdw)
        self._overlay.add_point((self.start_x, self.start_y))
        if self._fill_permitted:
            self._seed_pixels.add((int(x), int(y)))

    def drag_stop_cb(self, tdw):
        """Remove overlay and run the fill if valid seeds were marked"""
        self._overlay.cleanup()
        self._overlay = None
        if self._seed_pixels:
            self.fill(tdw)

    def _blend_parameters(self, comp_mode):
        """Get lock_alpha flag and compositing mode"""
        lock_alpha = False

        blend_mode = self.bm.active_mode.mode_type
        # alpha locking - may require two compositing steps per tile
        if blend_mode == BlendModes.LOCK_ALPHA:
            # with CombineNormal, compositing can be done in a single step
            if comp_mode == lib.mypaintlib.CombineNormal:
                comp_mode = lib.mypaintlib.CombineSourceAtop
            lock_alpha = True

        # erasing - overrides other compositing modes when enabled
        elif blend_mode == BlendModes.ERASE:
            comp_mode = lib.mypaintlib.CombineDestinationOut

        # colorize - non-spectral (for now) color blend + alpha locking
        elif blend_mode == BlendModes.COLORIZE:
            comp_mode = lib.mypaintlib.CombineColor
            lock_alpha = True

        return lock_alpha, comp_mode

    def fill(self, tdw):
        """Flood-fill with the current settings and marked seeds

        If the current layer is not fillable, a new layer will always be
        created for the fill.
        """
        self._tdws.add(tdw)
        self._update_ui()
        color = self.doc.app.brush_color_manager.get_color()
        opts = self.get_options_widget()
        make_new_layer = opts.make_new_layer
        rootstack = tdw.doc.layer_stack
        if not rootstack.current.get_fillable():
            make_new_layer = True
        eotf = lib.eotf.eotf()
        rgb = color.get_rgb()
        if eotf != 1.0:
            rgb = (rgb[0]**eotf, rgb[1]**eotf, rgb[2]**eotf)
        view_bbox = None
        if opts.limit_to_view:
            corners = tdw.get_corners_model_coords()
            view_bbox = lib.helpers.rotated_rectangle_bbox(corners)
        seeds = self._seed_pixels
        target_pos = self._target_pos

        lock_alpha, comp_mode = self._blend_parameters(opts.blend_mode)

        fill_args = lib.floodfill.FloodFillArguments(
            target_pos=target_pos,
            seeds=seeds,
            color=rgb,
            tolerance=opts.tolerance,
            offset=opts.offset,
            feather=opts.feather,
            gap_closing_options=opts.gap_closing_options,
            mode=comp_mode,
            lock_alpha=lock_alpha,
            opacity=opts.opacity,
            # Below are set in lib.document
            framed=False,
            bbox=None
        )

        tdw.doc.flood_fill(
            fill_args=fill_args,
            view_bbox=view_bbox,
            sample_merged=opts.sample_merged,
            src_path=opts.src_path,
            make_new_layer=make_new_layer,
            status_cb=status_callback
        )
        opts.make_new_layer = False

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
        if permitted and model.frame_enabled:
            fx1, fy1, fw, fh = model.get_frame()
            fx2, fy2 = fx1+fw, fy1+fh
            permitted = fx1 <= x < fx2 and fy1 <= y < fy2
        self._fill_permitted = permitted
        self._update_cursor(opts)

    def _update_cursor(self, opts):
        """Update the cursor
        The cursor used is based on fill options, blend mode,
        and if a fill can be run for the current position and layer.
        """
        # Update cursor of any TDWs we've crossed
        if self._fill_permitted:
            cursor = (opts.gap_closing, self.get_current_cursor())
        else:
            cursor = (opts.gap_closing, self._CURSOR_FILL_FORBIDDEN)

        if cursor != self._current_cursor:
            self._current_cursor = cursor
            for tdw in self._tdws:
                tdw.set_override_cursor(self.cursor)

    # Mode options

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = FloodFillOptionsWidget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET


def status_callback(handler):
    """
    Set up and run fill info/cancellation dialog
    :param handler: handler for fill info/cancellation
    :return: False if the fill is cancelled, None otherwise
    """
    app = gui.application.get_app()

    # Prevent escape release (always) triggering mode stack popping
    app.kbm.enabled = False

    # Create new dialog for each occurrence, hopefully
    # occurrences are rare enough for it not to matter very much.
    status_dialog = Gtk.MessageDialog(
        parent=app.drawWindow, buttons=Gtk.ButtonsType.CANCEL)

    curr_stage = [None]

    # Status update ticker callback - also handles dialog destruction
    def status_update():
        if handler.running():
            # Only update the header when the stage has changed
            if curr_stage[0] != handler.stage:
                curr_stage[0] = handler.stage
                status_dialog.set_property("text", handler.stage_string)
            status_dialog.set_property(
                "secondary-text", handler.progress_string)
            return True
        else:
            # Destroy dialog when fill is done, whether cancelled or not
            status_dialog.response(Gtk.ResponseType.OK)
            return False

    # Update the status message 20 times/s
    GLib.timeout_add(50, status_update)
    result = status_dialog.run()
    if result != Gtk.ResponseType.OK:
        handler.cancel()
    status_dialog.hide()
    handler.wait()
    status_dialog.destroy()
    app.kbm.enabled = True
    return result == Gtk.ResponseType.OK


class FloodFillOverlay (gui.overlays.Overlay):
    """
    Overlay indicating pixels that constitute the fill seeds

    The line width is zoom independent, so the visualization
    does not actually correspond 1:1 to the seed pixels.
    Marking pixel locations directly however, would make
    the markings virtually invisible when zoomed out.
    """

    DASH_LENGTH = 9
    # Clean out (SKIP-1)/SKIP points for smoother lines
    SKIP = 6

    def __init__(self, tdw):
        self._tdw = tdw
        self._line_points = []
        self._skip_index = 0
        tdw.display_overlays.append(self)

    def cleanup(self):
        self._line_points = []
        self._tdw.display_overlays.remove(self)
        self._skip_index = 0
        bounds = self._full_bounds()
        if bounds:
            self._tdw.queue_draw_area(*bounds)

    def add_point(self, point):
        x, y = point
        # self._full_bounds = self._update_bounds(self._full_bounds, x, y)
        # Clean up lines by only storing every nth point (n = SKIP)
        if len(self._line_points) > 1 and self._skip_index % self.SKIP != 0:
            self._line_points[-1] = point
        else:
            self._line_points.append(point)
        if len(self._line_points) > 2:
            x0, y0 = self._line_points[-2]
            # margin
            xmin, ymin = min(x0, x), min(y0, y)
            w, h = abs(x0 - x), abs(y0 - y)
            m = 2  # margin to account for line width
            self._tdw.queue_draw_area(xmin - m, ymin - m, w + 2 * m, h + 2 * m)
        self._skip_index += 1

    def _full_bounds(self):
        bounds = lib.helpers.coordinate_bounds(self._line_points)
        if bounds:
            x0, y0, x1, y1 = bounds
            w, h = x1 - x0, y1 - y0
            m = 2
            return x0 - m, y0 - m, w + 2*m, h + 2*m

    def paint(self, cr):
        """
        Paint the pixels of the marked set (approximately) as a stroke
        with alternating white and black dashes, for visibility
        """
        d_len = self.DASH_LENGTH
        if len(self._line_points) > 1:
            cr.set_line_join(cairo.LINE_JOIN_ROUND)
            cr.set_source_rgba(0.0, 0.0, 0.0, 0.8)
            cr.set_dash([d_len])
            cr.move_to(*self._line_points[0])
            for (x, y) in self._line_points[1:]:
                cr.line_to(x, y)
            cr.stroke_preserve()
            cr.set_source_rgba(1.0, 1.0, 1.0, 0.8)
            cr.set_dash([d_len], d_len)
            cr.stroke()


class FloodFillOptionsWidget (Gtk.Grid):
    """Configuration widget for the flood fill tool"""

    TOLERANCE_PREF = 'flood_fill.tolerance'
    LIM_TO_VIEW_PREF = 'flood_fill.limit_to_view'
    SAMPLE_MERGED_PREF = 'flood_fill.sample_merged'
    OFFSET_PREF = 'flood_fill.offset'
    FEATHER_PREF = 'flood_fill.feather'
    OPACITY_PREF = 'flood_fill.opacity'
    BLEND_MODE_PREF = 'flood_fill.blend_mode'

    # Gap closing related parameters
    GAP_CLOSING_PREF = 'flood_fill.gap_closing'
    GAP_SIZE_PREF = 'flood_fill.gap_size'
    RETRACT_SEEPS_PREF = 'flood_fill.retract_seeps'
    # "make new layer" is a temporary toggle, and is not saved to prefs

    DEFAULT_TOLERANCE = 0.05
    DEFAULT_LIM_TO_VIEW = False
    DEFAULT_SAMPLE_MERGED = False
    DEFAULT_MAKE_NEW_LAYER = False
    DEFAULT_BLEND_MODE = 0
    DEFAULT_OPACITY = 1.0
    DEFAULT_OFFSET = 0
    DEFAULT_FEATHER = 0

    # Gap closing related defaults
    DEFAULT_GAP_CLOSING = False
    DEFAULT_GAP_SIZE = 5
    DEFAULT_RETRACT_SEEPS = True

    # The state of the blend mode modifiers
    _BLEND_MODES = None

    def update_blend_mode(self, manager, old_mode, new_mode):
        """ Enable/disable mode selection combo box"""
        if old_mode == new_mode:
            return
        if new_mode.mode_type in [BlendModes.ERASE, BlendModes.COLORIZE]:
            self._blend_mode_combo.set_sensitive(False)
        else:
            self._blend_mode_combo.set_sensitive(True)
        self._update_blend_mode_warning(new_mode)

    def _update_blend_mode_warning(self, mode):
        """Show/hide warning label"""
        no_op_combination = (
            mode.mode_type == BlendModes.LOCK_ALPHA and
            self.blend_mode in lib.modes.MODES_DECREASING_BACKDROP_ALPHA
        )
        if no_op_combination and not self._warning_shown:
            self.attach(*self._bm_warning_label)
            self._bm_warning_label[0].show()
            self._warning_shown = True
        elif not no_op_combination and self._warning_shown:
            self.remove(self._bm_warning_label[0])
            self._warning_shown = False
        self.show()

    def __init__(self):
        Gtk.Grid.__init__(self)

        self.set_row_spacing(6)
        self.set_column_spacing(6)
        from gui.application import get_app
        self.app = get_app()
        prefs = self.app.preferences

        row = 0
        label = Gtk.Label()
        label.set_markup(C_(
            "fill options: numeric value that determines whether tested pixels"
            " will be included in the fill, based on color difference",
            u"Tolerance:"))
        label.set_tooltip_text(C_(
            "fill options: Tolerance (tooltip) "
            "Note: 'the start' refers to the color of "
            "the starting point (pixel) of the fill",
            u"How much pixel colors are allowed to vary from the start\n"
            u"before Flood Fill will refuse to fill them"))
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
        scale = InputSlider(adj)
        scale.set_hexpand(True)
        scale.set_draw_value(False)
        self.attach(scale, 1, row, 1, 1)

        row += 1
        label = Gtk.Label()
        label.set_markup(C_(
            "fill options: option category (label) "
            "Options under this category relate to what the fill is"
            "based on, not where the actual fill ends up.",
            u"Source:"))
        label.set_tooltip_text(C_(
            "fill options: 'Source:' category (tooltip)",
            u"The input that the fill will be based on"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.attach(label, 0, row, 1, 1)

        # Selection independent fill-basis

        root = self.app.doc.model.layer_stack
        src_list = FlatLayerList(root)
        self.src_list = src_list
        combo = Gtk.ComboBox.new_with_model(src_list)
        cell = Gtk.CellRendererText()
        cell.set_property("ellipsize", Pango.EllipsizeMode.END)
        combo.pack_start(cell, True)

        def layer_name_render(_, name_cell, model, it):
            """
            Display layer groups in italics and child layers
            indented with two spaces per level
            """
            name, path, layer = model[it][:3]
            if name is None:
                name = "Layer"
            if layer is None:
                name_cell.set_property(
                    "markup", "( <i>{text}</i> )".format(text=name)
                )
                return
            indented = "  " * (len(path) - 1) + name
            if isinstance(layer, lib.layer.LayerStack):
                name_cell.set_property(
                    "markup", "<i>{text}</i>".format(text=indented)
                )
            else:
                name_cell.set_property("text", indented)

        def sep_func(model, it):
            return model[it][0] is None

        combo.set_row_separator_func(sep_func)
        combo.set_cell_data_func(cell, layer_name_render)
        combo.set_tooltip_text(
            C_(
                "fill options: 'Source' category: Layer dropdown (tooltip)",
                u"Select a specific layer you want the fill to be based on"
            )
        )
        combo.set_active(0)
        self._prev_src_layer = None
        root.layer_inserted += self._layer_inserted_cb
        self._src_combo_cb_id = combo.connect(
            "changed", self._src_combo_changed_cb
        )
        self._src_combo = combo
        self.attach(combo, 1, row, 2, 1)

        row += 1

        text = C_(
            "fill options: 'Source:' category: toggle (label)\n"
            "When this option is enabled, the fill is based\n"
            "on the combination of all visible layers",
            u"Sample Merged")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            C_("fill options: Sample Merged (tooltip)",
               u"When considering which area to fill, use a\n"
               u"temporary merge of all the visible layers\n"
               u"underneath the current layer")
        )
        self.attach(checkbut, 1, row, 1, 1)
        active = bool(prefs.get(self.SAMPLE_MERGED_PREF,
                                self.DEFAULT_SAMPLE_MERGED))
        checkbut.set_active(active)
        checkbut.connect("toggled", self._sample_merged_toggled_cb)
        self._sample_merged_toggle = checkbut
        self._src_combo.set_sensitive(not active)

        row += 1

        text = C_("fill options: toggle whether the fill will be limited "
                  "by the viewport",
                  u"Limit to View")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            C_("fill options: Limit to View (tooltip)\n"
               "Note: 'that can fit the view' is equivalent to: "
               "'in which the entire visible part of the canvas can fit",
               u"Limit the area that can be filled, based on the viewport.\n"
               u"If the view is rotated, the fill will be limited to the\n"
               u"smallest canvas-aligned rectangle that can fit the view."))
        self.attach(checkbut, 1, row, 1, 1)
        active = bool(prefs.get(self.LIM_TO_VIEW_PREF,
                                self.DEFAULT_LIM_TO_VIEW))
        checkbut.set_active(active)
        checkbut.connect("toggled", self._limit_to_view_toggled_cb)
        self._limit_to_view_toggle = checkbut

        row += 1
        label = Gtk.Label()
        label.set_markup(C_(
            "fill options: option category (label)\n"
            "Options under this category relate to where the fill "
            "will end up (default: the active layer) and how it "
            "will be combined with that layer.",
            u"Target:"))
        label.set_tooltip_text(C_(
            "fill options: 'Target:' category (tooltip)",
            u"Where the output should go"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.attach(label, 0, row, 1, 1)

        text = C_(
            "fill options: Target | toggle (label)\n"
            "When this option is enabled, the fill will be placed on a new\n"
            "layer above the active layer. Option resets after each fill.",
            u"New Layer (once)"
        )
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            C_("fill options: Target | New Layer (tooltip)",
               u"Create a new layer with the results of the fill.\n"
               u"This is turned off automatically after use."))
        self.attach(checkbut, 1, row, 1, 1)
        active = self.DEFAULT_MAKE_NEW_LAYER
        checkbut.set_active(active)
        self._make_new_layer_toggle = checkbut

        row += 1
        label = Gtk.Label()
        label.set_markup(u"<b>\u26a0</b>")  # unicode warning sign
        label.set_tooltip_text(C_(
            "fill options: Target | Blend Mode dropdown - warning text",
            u"This mode does nothing when alpha locking is enabled!")
        )
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self._warning_shown = False
        # Not attached here, warning label is only shown for no-op combinations
        self._bm_warning_label = (label, 0, row, 1, 1)

        modes = list(lib.modes.STANDARD_MODES)
        modes.remove(lib.mypaintlib.CombineSpectralWGM)
        modes.insert(0, lib.mypaintlib.CombineSpectralWGM)
        combo = gui.layers.new_blend_mode_combo(modes, lib.modes.MODE_STRINGS)
        combo.set_tooltip_text(C_(
            "fill options: Target | Blend Mode dropdown (tooltip)",
            u"Blend mode used when filling"))
        # Reinstate the last _mode id_ independent of mode-list order
        mode_type = prefs.get(self.BLEND_MODE_PREF, self.DEFAULT_BLEND_MODE)
        mode_dict = {mode: index for index, mode, in enumerate(modes)}
        # Fallback is only necessary for compat. if a mode is ever removed
        active = mode_dict.get(int(mode_type), self.DEFAULT_BLEND_MODE)
        combo.set_active(active)
        combo.connect(
            "changed", self._bm_combo_changed_cb
        )
        self._blend_mode_combo = combo
        self.attach(combo, 1, row, 2, 1)

        row += 1
        label = Gtk.Label()
        label.set_markup(_(u"Opacity:"))
        label.set_tooltip_text(C_(
            "fill options: Opacity slider (tooltip)",
            u"Opacity of the fill"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.attach(label, 0, row, 1, 1)
        value = prefs.get(self.OPACITY_PREF, self.DEFAULT_OPACITY)
        adj = Gtk.Adjustment(value=value, lower=0.0, upper=1.0,
                             step_increment=0.05, page_increment=0.05,
                             page_size=0)
        adj.connect("value-changed", self._opacity_changed_cb)
        self._opacity_adj = adj
        scale = InputSlider()
        scale.set_hexpand(True)
        scale.set_adjustment(adj)
        scale.set_draw_value(False)
        self.attach(scale, 1, row, 1, 1)

        row += 1
        self.attach(Gtk.Separator(), 0, row, 2, 1)

        row += 1
        label = Gtk.Label()
        label.set_markup(C_(
            "fill options: numeric option - grow/shrink fill (label)",
            u"Offset:"
        ))
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
        spinbut.set_tooltip_text(C_(
            "fill options: Offset (tooltip)",
            u"The distance in pixels to grow or shrink the fill"
        ))
        spinbut.set_hexpand(True)
        spinbut.set_adjustment(adj)
        spinbut.set_numeric(True)
        self.attach(spinbut, 1, row, 1, 1)

        row += 1
        label = Gtk.Label()
        label.set_markup(C_(
            "fill options: numeric option for blurring fill (label)",
            u"Feather:"
        ))
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
        spinbut.set_tooltip_text(C_(
            "fill options: Feather (tooltip)",
            u"The amount of blur to apply to the fill"
        ))
        spinbut.set_hexpand(True)
        spinbut.set_adjustment(adj)
        spinbut.set_numeric(True)
        self.attach(spinbut, 1, row, 1, 1)

        row += 1
        self.attach(Gtk.Separator(), 0, row, 2, 1)

        row += 1
        gap_closing_params = Gtk.Grid()
        self._gap_closing_grid = gap_closing_params

        text = C_(
            "fill options: gap detection toggle (label)",
            u'Use Gap Detection'
        )
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(C_(
            "fill options: Use Gap Detection (tooltip)",
            u"Try to detect gaps and not fill past them.\n"
            u"Note: This can be a lot slower than the regular fill, "
            u"only enable when you need it."
        ))
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
        label.set_markup(C_(
            "fill options: gap-detection sub-option, numeric setting (label)",
            u"Max Gap Size:"
        ))
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
        spinbut.set_tooltip_text(C_(
            "fill options: Max Gap Size (tooltip)",
            u"The size of the largest gaps that can be detected.\n"
            u"Using large values can make the fill run a lot slower."
        ))
        spinbut.set_hexpand(True)
        spinbut.set_adjustment(adj)
        spinbut.set_numeric(True)
        gap_closing_params.attach(spinbut, 1, gcp_row, 1, 1)

        gcp_row += 1
        text = C_(
            "fill options: on/off sub-option, numeric (label)\n"
            "When enabled, if the fill stops after going past a detected gap, "
            "it 'pulls' the fill back out of the gap to the other side of it.",
            u"Prevent seeping"
        )
        checkbut = Gtk.CheckButton.new_with_label(text)
        active = prefs.get(self.RETRACT_SEEPS_PREF, self.DEFAULT_RETRACT_SEEPS)
        checkbut.set_active(active)
        checkbut.set_tooltip_text(C_(
            "fill options: Prevent seeping (tooltip)",
            u"Try to prevent the fill from seeping into the gaps.\n"
            u"If a fill starts in a detected gap, this option will do nothing."
        ))
        checkbut.connect("toggled", self._retract_seeps_toggled_cb)
        self._retract_seeps_toggle = checkbut
        gap_closing_params.attach(checkbut, 1, gcp_row, 1, 1)

        row += 1
        align = Gtk.Alignment.new(0.5, 1.0, 1.0, 0.0)
        align.set_vexpand(True)
        button = Gtk.Button(label=_("Reset"))
        button.connect("clicked", self._reset_clicked_cb)
        button.set_tooltip_text(C_(
            "fill options: Reset button (tooltip)",
            "Reset options to their defaults"))
        align.add(button)
        self.attach(align, 0, row, 2, 1)

        # Set up blend modifier callbacks
        self.bm = self.get_blend_modes()
        self.bm.mode_changed += self.update_blend_mode

    # Fill blend modes
    def get_blend_modes(self):
        """Get the (class singleton) blend modes manager"""
        cls = self.__class__
        if cls._BLEND_MODES is None:
            cls._BLEND_MODES = BlendModes()
        return cls._BLEND_MODES

    @property
    def tolerance(self):
        return float(self._tolerance_adj.get_value())

    @property
    def opacity(self):
        return float(self._opacity_adj.get_value())

    @property
    def make_new_layer(self):
        return bool(self._make_new_layer_toggle.get_active())

    @make_new_layer.setter
    def make_new_layer(self, value):
        self._make_new_layer_toggle.set_active(bool(value))

    @property
    def blend_mode(self):
        active = self._blend_mode_combo.get_active()
        active = 0 if active == -1 else active
        return self._blend_mode_combo.get_model()[active][0]

    @property
    def sample_merged(self):
        return bool(self._sample_merged_toggle.get_active())

    @property
    def limit_to_view(self):
        return bool(self._limit_to_view_toggle.get_active())

    @property
    def src_path(self):
        row = self._src_combo.get_active_iter()
        if row is not None:
            return self._src_combo.get_model()[row][1]
        else:
            return None

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

    def _limit_to_view_toggled_cb(self, checkbut):
        self.app.preferences[self.LIM_TO_VIEW_PREF] = self.limit_to_view

    def _sample_merged_toggled_cb(self, checkbut):
        self._src_combo.set_sensitive(not self.sample_merged)
        self.app.preferences[self.SAMPLE_MERGED_PREF] = self.sample_merged

    def _opacity_changed_cb(self, adj):
        self.app.preferences[self.OPACITY_PREF] = self.opacity

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

    def _bm_combo_changed_cb(self, combo):
        self.app.preferences[self.BLEND_MODE_PREF] = self.blend_mode
        self._update_blend_mode_warning(self.get_blend_modes().active_mode)

    def _layer_inserted_cb(self, root, path):
        """Check if the newly inserted layer was the last actively
        selected fill src layer, and reinstate the selection if so.
        """
        layer = root.deepget(path)
        if layer and self._prev_src_layer and self._prev_src_layer() is layer:
            # Restore previous selection layer
            combo = self._src_combo
            for entry in combo.get_model():
                if entry[2] is layer:
                    # Don't trigger callback
                    with combo.handler_block(self._src_combo_cb_id):
                        combo.set_active_iter(entry.iter)
                    return

    def _src_combo_changed_cb(self, combo):
        """Track the last selected choice of layer to maintain
        selection between layer moves that use intermediate
        layer deletion (as well undoing layer deletions)
        """
        row = combo.get_active_iter()
        if row is not None:
            layer = combo.get_model()[row][2]
            if layer is None:
                self._prev_src_layer = None
            else:
                self._prev_src_layer = weakref.ref(layer)
        else:
            # Option unset by layer deletion, set to default
            # without triggering callback again
            with combo.handler_block(self._src_combo_cb_id):
                combo.set_active(0)

    def _reset_clicked_cb(self, button):
        self._tolerance_adj.set_value(self.DEFAULT_TOLERANCE)
        self._make_new_layer_toggle.set_active(self.DEFAULT_MAKE_NEW_LAYER)
        self._src_combo.set_active(0)
        self._limit_to_view_toggle.set_active(self.DEFAULT_LIM_TO_VIEW)
        self._sample_merged_toggle.set_active(self.DEFAULT_SAMPLE_MERGED)
        self._opacity_adj.set_value(self.DEFAULT_OPACITY)
        self._offset_adj.set_value(self.DEFAULT_OFFSET)
        self._feather_adj.set_value(self.DEFAULT_FEATHER)
        # Gap closing params
        self._max_gap_adj.set_value(self.DEFAULT_GAP_SIZE)
        self._retract_seeps_toggle.set_active(self.DEFAULT_RETRACT_SEEPS)
        self._gap_closing_toggle.set_active(self.DEFAULT_GAP_CLOSING)


class FlatLayerList(Gtk.ListStore):
    """Stores a flattened copy of the layer tree"""

    def __init__(self, root_stack):
        super(FlatLayerList, self).__init__()

        root_stack.layer_properties_changed += self._layer_props_changed_cb
        root_stack.layer_inserted += self._layer_inserted_cb
        root_stack.layer_deleted += self._layer_deleted_cb

        self.root = root_stack
        # Column data : name, layer_path, layer
        self.set_column_types((str, object, object))
        default_selection = C_(
            "fill option: default option in the Source Layer dropdown",
            u"Selected Layer"
        )
        # Add default option and separator
        self.append((default_selection, None, None))
        self.append((None, None, None))
        # Flatten layer tree into rows
        for layer in root_stack:
            self._initalize(layer)

    def _layer_props_changed_cb(self, root, layerpath, layer, changed):
        """Update copies of layer names when changed"""
        if 'name' in changed:
            for item in self:
                if item[1] == layerpath:
                    item[0] = layer.name
                    return

    def _layer_inserted_cb(self, root, path):
        """Create a row for the inserted layer and update
        the paths of existing layers if necessary
        """
        layer = root.deepget(path)
        new_row = (layer.name, path, layer)
        for item in self:
            item_path = item[1]
            if item_path and path <= item_path:
                row_iter = item.iter
                self.insert_before(row_iter, new_row)
                self._update_paths(row_iter)
                return
        # If layer added to bottom, no other updates necessary
        self.append(new_row)

    def _layer_deleted_cb(self, root, path):
        """Remove the row for the deleted layer, and also any
        rows for layers that were children of the deleted layer
        """
        def is_child(p):
            return lib.layer.path_startswith(p, path)

        for item in self:
            if item[1] == path:
                row = item.iter
                # Remove rows for all children
                while self.remove(row) and is_child(self[row][1]):
                    pass
                # Update rows (if any) below last deleted
                if self.iter_is_valid(row):
                    self._update_paths(row)
                return

    def _update_paths(self, row_iter):
        """Update the paths for existing layers
        at or below the point of the given iterator
        """
        while row_iter:
            item = self[row_iter]
            path = self.root.deepindex(item[2])
            if path is not None:
                item[1] = path
            row_iter = self.iter_next(row_iter)

    def _initalize(self, layer):
        """Add a new row for the layer (unless it is the root)
        Subtrees are added recursively, depth-first traversal.
        """
        # if layer is not self.root:
        name = layer.name
        path = self.root.deepindex(layer)
        self.append((name, path, layer))
        if isinstance(layer, lib.layer.LayerStack):
            for child in layer:
                self._initalize(child)
