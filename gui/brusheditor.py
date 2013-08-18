# This file is part of MyPaint.
# Copyright (C) 2007-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Brush editor"""


## Imports

import os
import logging
logger = logging.getLogger(__name__)

from gettext import gettext as _
import gi
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import Pango
from gi.repository import GObject
from gi.repository import GdkPixbuf

import lib.brush
import curve
import dialogs
import brushmanager
from brushlib import brushsettings
from builderhacks import add_objects_from_template_string
from windowing import SubWindow


## Brush name mangling functions


def brush_name_to_display(name):
    return name.replace("_", " ")


def brush_display_to_name(name):
    return name.replace(" ", "_")


## Class definitions


class BrushEditorWindow (SubWindow):
    """Window containing the brush settings editor"""

    ## Class constants

    _UI_DEFINITION_FILE = "brusheditor.glade"
    _LISTVIEW_CNAME_COLUMN = 0
    _LISTVIEW_DISPLAYNAME_COLUMN = 1
    _LISTVIEW_IS_SELECTABLE_COLUMN = 2
    _LISTVIEW_FONT_WEIGHT_COLUMN = 3


    ## Construction

    def __init__(self):
        app = None
        if __name__ != '__main__':
            from application import get_app
            app = get_app()
            self._brush = app.brush
            bm = app.brushmanager
            bm.brush_selected += self.brush_selected_cb
        else:
            self._brush = lib.brush.BrushInfo()
            self._brush.load_defaults()
        SubWindow.__init__(self, app, key_input=False)
        # Adjusters: may be shared those of the app
        self._base_adj = {}  #: setting cname => base value adj
        self._input_y_adj = {}  #: input name => scale y range (+-) adj
        self._input_xmin_adj = {}  #: input name => scale x min adj
        self._input_xmax_adj = {}  #: input name => scale x min adj
        self._disable_input_adj_changed_cb = False
        self._init_adjustments()
        self.set_title("Brush Settings Editor")
        self._setting = None
        self._builder = Gtk.Builder()
        self._build_ui()
        self.connect_after("show", self._post_show_cb)
        editor = self._builder.get_object("brush_editor")
        self.add(editor)
        self._brush.observers.append(self.brush_modified_cb)
        self._live_update_idle_cb_id = None
        GObject.idle_add(self._update_brush_header)


    def _init_adjustments(self):
        """Initializes adjustments for the scales used internally

        When running as part of the MyPaint app, the brush setting ones are
        shared with it.
        """
        # Brush setting base values
        if self.app:
            for s in brushsettings.settings_visible:
                adj = self.app.brush_adjustment[s.cname]
                self._base_adj[s.cname] = adj
            # The application instance manages value-changed callbacks itself.
        else:
            for s in brushsettings.settings_visible:
                adj = Gtk.Adjustment(value=s.default,
                                     lower=s.min, upper=s.max,
                                     step_incr=0.01, page_incr=0.1)
                self._base_adj[s.cname] = adj
            changed_cb = self._testmode_base_value_adj_changed_cb
            for cname, adj in self._base_adj.iteritems():
                adj.connect('value-changed', changed_cb, cname)
        # Per-input scale maxima and minima
        for inp in brushsettings.inputs:
            name = inp.name
            adj = Gtk.Adjustment(value=1.0/4.0, lower=-1.0, upper=1.0,
                                 step_incr=0.01, page_incr=0.1)
            adj.connect("value-changed", self.input_adj_changed_cb, inp)
            self._input_y_adj[name] = adj
            lower = -20.0
            upper = +20.0
            if inp.hard_min is not None:
                lower = inp.hard_min
            if inp.hard_max is not None:
                upper = inp.hard_max            
            adj = Gtk.Adjustment(value=inp.soft_min,
                                 lower=lower, upper=upper-0.1,
                                 step_incr=0.01, page_incr=0.1)
            adj.connect("value-changed", self.input_adj_changed_cb, inp)
            self._input_xmin_adj[name] = adj
            adj = Gtk.Adjustment(value=inp.soft_max,
                                 lower=lower+0.1, upper=upper,
                                 step_incr=0.01, page_incr=0.1)
            adj.connect("value-changed", self.input_adj_changed_cb, inp)
            self._input_xmax_adj[name] = adj


    def _build_ui(self):
        """Builds the UI from ``brusheditor.glade``"""
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        ui_path = os.path.join(ui_dir, self._UI_DEFINITION_FILE)
        ui_fp = open(ui_path, 'r')
        ui_xml = ui_fp.read()
        ui_fp.close()
        self._builder.add_from_string(ui_xml)
        self._populate_inputs(ui_xml)
        self._populate_settings()
        self._builder.connect_signals(self)
        for inp in brushsettings.inputs:
            grid = self._builder.get_object("by%s_curve_grid" % inp.name)
            GObject.idle_add(grid.hide)
            curve = self._builder.get_object("by%s_curve" % inp.name)
            def _curve_changed_cb(curve, i=inp):
                self._update_brush_from_input_widgets(i)
            curve.changed_cb = _curve_changed_cb
            btn = self._builder.get_object("by%s_reset_button" % inp.name)
            btn.connect("clicked",self.input_adj_reset_button_clicked_cb,inp)
        # Certain actions must be coordinated via a real app instance
        if not self.app:
            action_buttons = ["clone_button", "rename_button",
                              "edit_icon_button", "delete_button",
                             "live_update_checkbutton", "save_button"]
            for b_name in action_buttons:
                w = self._builder.get_object(b_name)
                w.set_sensitive(False)


    def _populate_inputs(self, ui_xml):
        # Grid repacking magic for the templated stuff

        # XXX FRAGILE
        # It would be nice to use the template's own left-attach, top-attach,
        # width and height child properties to replicate its layout as
        # specified, but unfortunately the current versions of PyGI are all
        # broken in one way or another with respect to getting container child
        # properties. So we fudge it until upstream fix
        # https://bugzilla.gnome.org/show_bug.cgi?id=685076
        group_start_row = 4
        tmpl_layouts = [
            ("by{name}_label",           0, 0, 1, 1),
            ("by{name}_expander_button", 1, 0, 1, 1),
            ("by{name}_scale",           2, 0, 1, 1),
            ("by{name}_reset_button",    3, 0, 1, 1),
            ("by{name}_curve_grid",      2, 2, 2, 1),  ]
        grid_pos = {}
        grid = self._builder.get_object("setting_editor_grid")
        # Extract the relative layout and pattern of by-input widgets
        group_step = 0
        tmpl_objs = []
        for tmpl_id, x, y, w, h in tmpl_layouts:
            tmpl_obj = self._builder.get_object(tmpl_id)
            group_step = max(group_step, y+h)
            tmpl_objs.append(tmpl_obj)
        # Remove the "originals": they're not useful in themselves
        for tmpl_obj in tmpl_objs:
            grid.remove(tmpl_obj)
        # Generate lots of clones, one group per brush input
        for i in brushsettings.inputs:
            params = dict(dname=i.dname, name=i.name, tooltip=i.tooltip)
            object_ids = [layout[0] for layout in tmpl_layouts]
            widgets = add_objects_from_template_string(self._builder, ui_xml,
                                                       object_ids, params)
            for layout, widget in zip(tmpl_layouts, widgets):
                tmpl_id, x, y, w, h = layout
                y += group_start_row
                grid.attach(widget, x, y, w, h)
                if tmpl_id == "by{name}_curve_grid":
                    widget.hide()
            group_start_row += group_step
            label = self._builder.get_object("by%s_label" % i.name)
            label.set_tooltip_text(i.tooltip)
            btn = self._builder.get_object("by%s_expander_button" % i.name)
            btn.__input = i
            # Hook up curve max and min adjustments
            cb = self._update_axis_label
            fmt = "%+.2f"
            # Y axis
            scale = self._builder.get_object("by%s_scale" % i.name)
            scale_adj = self._input_y_adj[i.name]
            scale_lbl = self._builder.get_object("by%s_ymax_label" % i.name)
            scale_adj.connect("value-changed", cb, scale_lbl, fmt, False)
            cb(scale_adj, scale_lbl, fmt, False)
            scale_lbl = self._builder.get_object("by%s_ymin_label" % i.name)
            scale_adj.connect("value-changed", cb, scale_lbl, fmt, True)
            cb(scale_adj, scale_lbl, fmt, True)
            scale.set_adjustment(scale_adj)
            # X axis: min
            sbut = self._builder.get_object("by%s_xmin_scalebutton" % i.name)
            sbut_lbl = self._builder.get_object("by%s_xmin_label" % i.name)
            sbut_adj = self._input_xmin_adj[i.name]
            sbut_adj.connect("value-changed", cb, sbut_lbl, fmt, False)
            cb(sbut_adj, sbut_lbl, fmt, False)
            sbut.set_adjustment(sbut_adj)
            # X axis: max
            sbut = self._builder.get_object("by%s_xmax_scalebutton" % i.name)
            sbut_lbl = self._builder.get_object("by%s_xmax_label" % i.name)
            sbut_adj = self._input_xmax_adj[i.name]
            sbut_adj.connect("value-changed", cb, sbut_lbl, fmt, False)
            cb(sbut_adj, sbut_lbl, fmt, False)
            sbut.set_adjustment(sbut_adj)


    def _update_axis_label(self, adj, label, strfmt, negate):
        """Updates a label widget with an adjustment value when it changes"""
        value = adj.get_value()
        if negate:
            value *= -1
        label.set_text(strfmt % (value,))


    def _populate_settings(self):
        groups = [{
                'id': 'experimental',
                'title': _('Experimental'),
                'settings' : [],
            }, {
                'id': 'basic',
                'title': _('Basic'),
                'settings': [
                    'radius_logarithmic', 'radius_by_random',
                    'hardness', 'snap_to_pixel', 'anti_aliasing', 'eraser',
                    'offset_by_random', 'elliptical_dab_angle',
                    'elliptical_dab_ratio', 'direction_filter',
                ],
            }, {
                'id': 'opacity',
                'title': _('Opacity'),
                'settings': [
                    'opaque', 'opaque_multiply', 'opaque_linearize',
                    'lock_alpha',
                ],
            }, {
                'id': 'dabs',
                'title': _('Dabs'),
                'settings': [
                    'dabs_per_basic_radius', 'dabs_per_actual_radius',
                    'dabs_per_second',
                ],
            }, {
                'id': 'smudge',
                'title': _('Smudge'),
                'settings': ['smudge', 'smudge_length', 'smudge_radius_log'],
            }, {
                'id': 'speed',
                'title': _('Speed'),
                'settings': [
                    'speed1_slowness', 'speed2_slowness', 'speed1_gamma',
                    'speed2_gamma', 'offset_by_speed',
                    'offset_by_speed_slowness',
                ],
            }, {
                'id': 'tracking',
                'title': _('Tracking'),
                'settings': [
                    'slow_tracking', 'slow_tracking_per_dab', 'tracking_noise'
                ],
            }, {
                'id': 'stroke',
                'title': _('Stroke'),
                'settings': [
                    'stroke_threshold', 'stroke_duration_logarithmic',
                    'stroke_holdtime',
                ],
            }, {
                'id': 'color',
                'title': _('Color'),
                'settings': [
                    'change_color_h', 'change_color_l',
                    'change_color_hsl_s', 'change_color_v',
                    'change_color_hsv_s', 'restore_color',
                    'colorize',
                ],
            }, {
                'id': 'custom',
                'title': _('Custom'),
                'settings': [ 'custom_input', 'custom_input_slowness' ],
            }]
        hidden_settings = ['color_h', 'color_s', 'color_v']

        # Add new settings to the "experimental" group
        grouped_settings = set(hidden_settings)
        for g in groups:
            grouped_settings.update(g['settings'])
        for s in brushsettings.settings:
            n = s.cname
            if n not in grouped_settings:
                groups[0]['settings'].append(n)
                logger.warning('Setting %r should be added to a group', n)
        # Hide experimental group if empty
        if not groups[0]['settings']:
            groups.pop(0)
        # Groups to open by default
        open_paths = []
        open_ids = set(["experimental", "basic"])
        # Populate the treestore
        store = self._builder.get_object("settings_treestore")
        root_iter = store.get_iter_first()
        group_num = 0        
        for group in groups:
            # Columns: [cname, displayname, is_selectable, font_weight]
            row_data = [None, group["title"], False, Pango.Weight.NORMAL]
            group_iter = store.append(root_iter, row_data)
            for i, cname in enumerate(group['settings']):
                s = brushsettings.settings_dict[cname]
                row_data = [cname, s.name, True, Pango.Weight.NORMAL]
                store.append(group_iter, row_data)
            if group['id'] in open_ids:
                open_paths.append(group_num)
            group_num += 1
        # Connect signals and handler functions
        v = self._builder.get_object("settings_treeview")
        sel = v.get_selection()
        sel.set_select_function(self._settings_treeview_selectfunc, None)
        # Process the paths-to-open
        if open_paths:
            for path in open_paths:
                v.expand_to_path(Gtk.TreePath(path))


    def _post_show_cb(self, widget):
        self._current_setting_changed()


    ## Main action buttons

    def save_button_clicked_cb(self, button):
        """Save the current brush settings (overwrites)"""
        bm = self.app.brushmanager
        b = bm.selected_brush
        if not b.name:
            msg = _('No brush selected, please use "Add As New" instead.')
            dialogs.error(self, msg)
            return
        b.brushinfo = self.app.brush.clone()
        b.save()


    def live_update_checkbutton_toggled_cb(self, checkbutton):
        """Realtime update of last stroke with the current brush settings"""
        self._queue_live_update()

    def edit_icon_button_clicked_cb(self, button):
        logger.info("Editing icon for current brush")
        action = self.app.find_action("BrushIconEditorWindow")
        action.activate()


    def rename_button_clicked_cb(self, button):
        """Rename the current brush; user is prompted for a new name"""
        bm = self.app.brushmanager
        src_brush = bm.selected_brush
        if not src_brush.name:
            dialogs.error(self, _('No brush selected!'))
            return

        src_name_pp = src_brush.name.replace('_', ' ')
        dst_name = dialogs.ask_for_name(self, _("Rename Brush"), src_name_pp)
        if not dst_name:
            return
        dst_name = dst_name.replace(' ', '_')
        # ensure we don't overwrite an existing brush by accident
        dst_deleted = None
        for group, brushes in bm.groups.iteritems():
            for b2 in brushes:
                if b2.name == dst_name:
                    if group == brushmanager.DELETED_BRUSH_GROUP:
                        dst_deleted = b2
                    else:
                        msg = _('A brush with this name already exists!')
                        dialogs.error(self, msg)
                        return

        logger.info("Renaming brush %r -> %r", src_brush.name, dst_name)
        if dst_deleted:
            deleted_group = brushmanager.DELETED_BRUSH_GROUP
            deleted_brushes = bm.get_group_brushes(deleted_group)
            deleted_brushes.remove(dst_deleted)
            bm.brushes_changed(deleted_brushes)

        # save src as dst
        src_name = src_brush.name
        src_brush.name = dst_name
        src_brush.save()
        src_brush.name = src_name
        # load dst
        dst_brush = brushmanager.ManagedBrush(bm, dst_name, persistent=True)
        dst_brush.load()

        # Replace src with dst, but keep src in the deleted list if it
        # is a stock brush
        self._delete_brush(src_brush, replacement=dst_brush)

        bm.select_brush(dst_brush)


    def delete_button_clicked_cb(self, button):
        """Deletes the current brush, with a confirmation dialog"""
        bm = self.app.brushmanager
        b = bm.selected_brush
        if not b.name:
            dialogs.error(self, _('No brush selected!'))
            return
        if not dialogs.confirm(self, _("Really delete brush from disk?")):
            return
        bm.select_brush(None)
        self._delete_brush(b, replacement=None)


    def clone_button_clicked_cb(self, button):
        """Create and save a new brush based on the current working brush"""
        bm = self.app.brushmanager
        # First, clone and save to disk. Set a null name to avoid a
        # mis-highlight in brush selectors.
        b = bm.selected_brush.clone(name=None)    # ManagedBrush with preview
        b.brushinfo = self._brush.clone()  # current unsaved settings
        b.brushinfo.set_string_property("parent_brush_name", None)
        b.save()
        # Put in the "New" group, and notify that the group has changed
        group = brushmanager.NEW_BRUSH_GROUP
        brushes = bm.get_group_brushes(group)
        brushes.insert(0, b)
        b.persistent = True   # Brush was saved
        bm.brushes_changed(brushes)
        # Highlight the new brush
        bm.select_brush(b)
        # Pretend that the active painting brush is a child of the new clone,
        # for the sake of the strokemap and strokes drawn immediately after
        # cloning.
        self.app.brush.set_string_property("parent_brush_name", b.name)
        # Reveal the added group if it's hidden
        ws = self.app.workspace
        ws.show_tool_widget("MyPaintBrushGroupTool", (group,))


    ## Utility functions for managing curves

    def _get_brushpoints_from_curvewidget(self, inp):
        scale_y_adj = self._input_y_adj[inp.name]
        curve_widget = self._builder.get_object("by%s_curve" % inp.name)
        scale_y = scale_y_adj.get_value()
        if not scale_y:
            return []
        brush_points = [self._point_widget2real(p, inp)
                        for p in curve_widget.points]
        nonzero = [True for x, y in brush_points if y != 0]
        if not nonzero:
            return []
        return brush_points

    
    def _point_widget2real(self, p, inp):
        x, y = p
        scale_y_adj = self._input_y_adj[inp.name]
        xmax_adj = self._input_xmax_adj[inp.name]
        xmin_adj = self._input_xmin_adj[inp.name]
        scale_y = scale_y_adj.get_value()
        xmax = xmax_adj.get_value()
        xmin = xmin_adj.get_value()
        scale_x = xmax - xmin
        x = xmin + (x * scale_x)
        y = (0.5-y) * 2.0 * scale_y
        return (x, y)


    def _point_real2widget(self, p, inp):
        x, y = p
        scale_y_adj = self._input_y_adj[inp.name]
        xmax_adj = self._input_xmax_adj[inp.name]
        xmin_adj = self._input_xmin_adj[inp.name]
        scale_y = scale_y_adj.get_value()
        xmax = xmax_adj.get_value()
        xmin = xmin_adj.get_value()
        scale_x = xmax - xmin
        assert scale_x
        if scale_y == 0:
            y = None
        else:
            y = -(y/scale_y/2.0)+0.5 
        x = (x-xmin)/scale_x
        return (x, y)


    def _get_x_normal(self, inp):
        """Returns the x coordinate of the 'normal' value of the input"""
        return self._point_real2widget((inp.normal, 0.0), inp)[0]


    def _update_graypoint(self, inp):
        curve_widget = self._builder.get_object("by%s_curve" % inp.name)
        curve_widget.graypoint = (self._get_x_normal(inp), 0.5)
        curve_widget.queue_draw()


    @staticmethod
    def _points_equal(points_a, points_b):
        if len(points_a) != len(points_b):
            return False
        for a, b in zip(points_a, points_b):
            for v1, v2 in zip(a, b):
                if abs(v1-v2) > 0.0001:
                    return False
        return True


    ## Brush event handling

    def brush_selected_cb(self, bm, managed_brush, brushinfo):
        """Update GUI when a new brush is selected via the brush manager"""
        self._update_brush_header()
        self._update_setting_ui(expanders=True)


    def _update_brush_header(self):
        """Updates the header strip with the current brush's icon and name"""
        mb = None
        if self.app:
            mb = self.app.brushmanager.selected_brush
        # Brush name label
        if mb:
            name = brush_name_to_display(mb.name)
        else:
            name = "(Not running as part of MyPaint)"
        label = self._builder.get_object("brush_name_label")
        label.set_text(name)
        # Brush icon
        image = self._builder.get_object("brush_preview_image")
        w = image.get_allocated_width()
        h = image.get_allocated_height()
        if mb:
            pixbuf = mb.preview
        else:
            pixbuf = None
        if pixbuf:
            pixbuf = pixbuf.scale_simple(w, h, GdkPixbuf.InterpType.BILINEAR)
        if not pixbuf:
            pixbuf = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB,
                                          True, 8, w, h)
        image.set_from_pixbuf(pixbuf)
        


    def brush_modified_cb(self, settings, expanders=False):
        """Update gui when the brush has been modified"""
        if self._setting is None or self._setting.cname not in settings:
            return
        self._update_setting_ui(expanders=expanders)
        # Live update
        self._queue_live_update()


    ## GUI updating from the brush


    def _update_setting_ui(self, expanders=False):
        """Updates all the UI elements for the current setting"""
        # Update base value adjuster and slider
        if self._setting is None:
            return
        base_adj = self._base_adj[self._setting.cname]
        # Update its value if running in test mode
        # Normally the app will do this itself
        if not self.app:
            newvalue = self._brush.get_base_value(self._setting.cname)
            base_adj.set_value(newvalue)
        # Associate the base value scale with the right adjustment
        scale = self._builder.get_object("base_value_scale")
        if scale.get_adjustment() is not base_adj:
            scale.set_adjustment(base_adj)
        # Update brush dynamics curves and sliders
        for inp in brushsettings.inputs:
            # check whether we really need to update anything
            #points_old = self._get_brushpoints_from_curvewidget(inp)
            #points_new = self._brush.get_points(self._setting.cname, inp.name)
            # try not to destroy changes that the user made to the widget
            # (not all gui states are stored inside the brush)
            #if not self._points_equal(points_old, points_new):
            self._update_input_curve(inp, expander=expanders)
            self._update_graypoint(inp)


    def _update_input_curve(self, inp, expander=False):
        """Update curve scale adjustments to fit the curve into view"""
        scale_y_adj = self._input_y_adj[inp.name]
        xmax_adj = self._input_xmax_adj[inp.name]
        xmin_adj = self._input_xmin_adj[inp.name]
        curve_widget = self._builder.get_object("by%s_curve" % inp.name)

        brush_points = self._brush.get_points(self._setting.cname, inp.name)
        brush_points_zero = [(inp.soft_min, 0.0), (inp.soft_max, 0.0)]
        if not brush_points:
            brush_points = brush_points_zero

        # 1. update the scale

        xmin, xmax = brush_points[0][0], brush_points[-1][0]
        assert xmax > xmin
        assert max([x for x, y in brush_points]) == xmax
        assert min([x for x, y in brush_points]) == xmin

        y_min = min([y for x, y in brush_points])
        y_max = max([y for x, y in brush_points])
        scale_y = max(abs(y_min), abs(y_max))

        # choose between scale_y and -scale_y (arbitrary)
        if brush_points[0][1] > brush_points[-1][1]:
            scale_y = -scale_y

        if not scale_y:
            brush_points = brush_points_zero
            # if xmin/xmax were non-default, reset them
            xmin = inp.soft_min
            xmax = inp.soft_max

        self._disable_input_adj_changed_cb = True
        # set adjustment limits imposed by brush setting
        diff = self._setting.max - self._setting.min
        scale_y_adj.set_upper(+diff)
        scale_y_adj.set_lower(-diff)
        scale_y_adj.set_value(scale_y)
        # set adjustment values such that all points are visible
        xmax_adj.set_value(xmax)
        xmin_adj.set_value(xmin)
        self._disable_input_adj_changed_cb = False

        # 2. calculate the default curve (the one we display if there is no curve)
        curve_points_zero = [self._point_real2widget(p, inp)
                              for p in brush_points_zero]
        # widget x coordinate of the "normal" input value
        x_normal = self._get_x_normal(inp)

        y0 = -1.0
        y1 = +1.0
        # the default _scroll_setting_editorline should go through zero at x_normal
        # change one of the border points to achieve this
        if x_normal >= 0.0 and x_normal <= 1.0:
            if x_normal < 0.5:
                y0 = -0.5/(x_normal-1.0)
                y1 = 0.0
            else:
                y0 = 1.0
                y1 = -0.5/x_normal + 1.0

        (x0, junk0), (x1, junk1) = curve_points_zero
        curve_points_zero = [(x0, y0), (x1, y1)]

        # 3. display the curve

        if scale_y:
            curve_points = [self._point_real2widget(p, inp)
                             for p in brush_points]
        else:
            curve_points = curve_points_zero

        assert len(curve_points) >= 2
        curve_widget.points = curve_points
        curve_widget.queue_draw()
        self._update_graypoint(inp)

        # 4. reset the expander
        if not expander:
            return
        interesting = not self._points_equal(curve_points, curve_points_zero)
        self._set_input_expanded(inp, interesting, scroll=False)


    ## Settings treeview management and change callbacks


    def _settings_treeview_selectfunc(self, seln, model, path, is_seld, data):
        """Determines whether settings listview rows can be selected"""
        i = model.get_iter(path)
        is_leaf = model.get_value(i, self._LISTVIEW_IS_SELECTABLE_COLUMN)
        return is_leaf


    def settings_treeview_row_activated_cb(self, view, path, column):
        """Double clicking opens expander rows"""
        model = view.get_model()
        i = model.get_iter(path)
        is_leaf = model.get_value(i, self._LISTVIEW_IS_SELECTABLE_COLUMN)
        if is_leaf or not view.get_visible():
            return
        if view.row_expanded(path):
            view.collapse_row(path)
        else:
            view.expand_row(path, True)


    def settings_treeview_cursor_changed_cb(self, view):
        """User has chosen a different setting using the treeview"""
        sel = view.get_selection()
        if sel is None:
            return
        model, i = sel.get_selected()
        setting = self._setting
        if i is None:
            setting = None
        else:
            cname = model.get_value(i, self._LISTVIEW_CNAME_COLUMN)
            setting = brushsettings.settings_dict[cname]
        if setting is not self._setting:
            self._setting = setting
            self._current_setting_changed()


    def _current_setting_changed(self):
        """Update the UI to reflect a change of currently brush setting"""
        no_setting_label = self._builder.get_object("no_setting_label")
        setting_editor_grid = self._builder.get_object("setting_editor_grid")
        if self._setting is None:
            no_setting_label.show_all()
            setting_editor_grid.hide()
            return
        no_setting_label.hide()
        setting_editor_grid.show()
        # Update setting name label
        label = self._builder.get_object("setting_name_label")
        label.set_label(self._setting.name)
        # Update setting description
        label.set_tooltip_text(self._setting.tooltip)
        label = self._builder.get_object("setting_name_field_label")
        label.set_tooltip_text(self._setting.tooltip)
        # Simulate the brush changing for just this setting to update
        # the base value sliders and input curve stuff.
        self.brush_modified_cb(self._setting.cname, expanders=True)
        # Scroll the setting editor to the top
        self._scroll_setting_editor(widget=None)



    ## Adjuster change callbacks

    def _testmode_base_value_adj_changed_cb(self, adj, cname):
        """User adjusted the setting's base value using the scale (test only)
        """
        value = adj.get_value()
        self._brush.set_base_value(cname, value)


    def base_value_reset_button_clicked_cb(self, button):
        """User reset the setting's base value using the button"""
        # Bound by Gtk.Builder.connect_signals()
        s = self._setting
        adj = self._base_adj[s.cname]
        adj.set_value(s.default)


    def input_adj_changed_cb(self, adj, inp):
        """User adjusted one of the curve extent scales or scalebuttons"""
        if self._disable_input_adj_changed_cb:
            return
        assert self._setting is not None
        assert self._brush is not None
        # 1. verify and constrain the adjustment changes
        scale_y_adj = self._input_y_adj[inp.name]
        xmax_adj = self._input_xmax_adj[inp.name]
        xmin_adj = self._input_xmin_adj[inp.name]
        scale_y = scale_y_adj.get_value()
        xmax = xmax_adj.get_value()
        xmin = xmin_adj.get_value()
        if xmax <= xmin:
            # change the other one
            if adj is xmax_adj:
                xmin_adj.set_value(xmax - 0.1)
            elif adj is xmin_adj:
                xmax_adj.set_value(xmin + 0.1)
            else:
                assert False
            return # the adjustment change causes another call of this function
        assert xmax > xmin
        # 2. interpret the points displayed in the curvewidget
        #    according to the new scale (update the brush)
        self._update_brush_from_input_widgets(inp)


    def input_adj_reset_button_clicked_cb(self, btn, inp):
        """User reset an input mapping by clicking its reset button"""
        assert self._setting is not None
        assert self._brush is not None
        scale_y_adj = self._input_y_adj[inp.name]
        scale_y_adj.set_value(0.0)
        #self._update_brush_from_input_widgets(inp)
        points_zero = [(inp.soft_min, 0.0), (inp.soft_max, 0.0)]
        self._brush.set_points(self._setting.cname, inp.name, points_zero)


    ## Brush updating

 
    def _update_brush_from_input_widgets(self, inp):
        # update the brush dynamics with the points from the curve_widget
        points = self._get_brushpoints_from_curvewidget(inp)
        self._brush.set_points(self._setting.cname, inp.name, points)


    def _delete_brush(self, b, replacement=None):
        bm = self.app.brushmanager
        for brushes in bm.groups.itervalues():
            if b in brushes:
                idx = brushes.index(b)
                if replacement:
                    brushes[idx] = replacement
                else:
                    del brushes[idx]
                bm.brushes_changed(brushes)
                assert b not in brushes, \
                        'Brush exists multiple times in the same group!'
        if not b.delete_from_disk():
            # stock brush can't be deleted
            deleted_group = brushmanager.DELETED_BRUSH_GROUP
            deleted_brushes = bm.get_group_brushes(deleted_group)
            deleted_brushes.insert(0, b)
            bm.brushes_changed(deleted_brushes)


    ## Live update


    @property
    def _live_update_enabled(self):
        """Whether Live Update is active"""
        cb = self._builder.get_object("live_update_checkbutton")
        return cb.get_active() and self.get_visible()


    def _queue_live_update(self):
        """Queues a single live update of the most recent brushstroke"""
        # Not if already queued, or if disabled
        if self._live_update_idle_cb_id:
            return
        if not self._live_update_enabled:
            return
        # Only in live-updatable modes.
        # Currently: only FreehandMode supports this. It could potentially
        # work with other modes: please test!
        if not getattr(self.app.doc.modes.top, "is_live_updateable", False):
            return
        cbid = GObject.idle_add(self._live_update_idle_cb)
        self._live_update_idle_cb_id = cbid


    def _live_update_idle_cb(self):
        """Live update idle routine"""
        doc = self.app.doc.model
        doc.redo_last_stroke_with_different_brush(self._brush)
        self._live_update_idle_cb_id = None
        return False


    ## Expander button callbacks


    def byname_expander_button_clicked_cb(self, button):
        inp = button.__input
        arrow = button.get_child()
        grid = self._builder.get_object("by%s_curve_grid" % inp.name)
        self._set_input_expanded(inp, not grid.get_visible())


    ## UI utility functions

    def _set_input_expanded(self, inp, expand, scroll=True):
        getobj = self._builder.get_object
        arrow = getobj("by%s_expander_arrow" % (inp.name,))
        grid = getobj("by%s_curve_grid" % (inp.name,))
        if expand:
            arrow.set_property("arrow-type", Gtk.ArrowType.DOWN)
            grid.show_all()
            if scroll:
                GObject.idle_add(self._scroll_setting_editor, grid)
        else:
            arrow.set_property("arrow-type", Gtk.ArrowType.RIGHT)
            grid.hide()
       

    def _scroll_setting_editor(self, widget=None):
        scrolls = self._builder.get_object("setting_editor_scrolls")
        adj = scrolls.get_vadjustment()
        if widget is None:
            adj.set_value(0.0)
        else:
            val = adj.get_value()
            page = adj.get_page_size()
            upper = adj.get_upper()
            maxval = upper - page
            alloc = widget.get_allocation()
            x, y, w, h = (alloc.x, alloc.y, alloc.width, alloc.height)
            bottom = y + h
            if bottom > val + page:
                newval = min(maxval, bottom - page)
                adj.set_value(newval)
        return False



def _test():
    """Run interactive tests, outside the application."""
    logging.basicConfig()
    win = BrushEditorWindow()
    win.connect("delete-event", lambda *a: Gtk.main_quit())
    win.show_all()
    Gtk.main()



if __name__ == '__main__':
    _test()


