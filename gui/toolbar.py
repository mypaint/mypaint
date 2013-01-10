# This file is part of MyPaint.
# Copyright (C) 2011 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""The application toolbar, and its specialised widgets.
"""

import os

import pygtkcompat
import gtk
from gtk import gdk
import gobject
from gettext import gettext as _
import pango

import dialogs
from brushlib import brushsettings
import dropdownpanel
import widgets
from colors import RGBColor, ColorAdjuster, HSVTriangle
from colors import PreviousCurrentColorAdjuster, ColorPickerButton
from lib.helpers import escape

FRAMEWORK_XML = 'gui/toolbar.xml'
MERGEABLE_XML = [
    ("toolbar1_file", 'gui/toolbar-file.xml', _("File handling")),
    ("toolbar1_scrap", 'gui/toolbar-scrap.xml', _("Scraps switcher")),
    ("toolbar1_edit", 'gui/toolbar-edit.xml', _("Undo and Redo")),
    ("toolbar1_blendmodes", 'gui/toolbar-blendmodes.xml', _("Blend Modes")),
    ("toolbar1_linemodes", 'gui/toolbar-linemodes.xml', _("Line Mode")), #FIXME: make plural (string freeze hack)
    ("toolbar1_view_modes", 'gui/toolbar-view-modes.xml', _("View (Main)")),
    ("toolbar1_view_manips", 'gui/toolbar-view-manips.xml', _("View (Alternative/Secondary)")),
    ("toolbar1_view_resets", 'gui/toolbar-view-resets.xml', _("View (Resetting)")),
    ("toolbar1_subwindows", 'gui/toolbar-subwindows.xml', _("Subwindows")),
    ]
HISTORY_PREVIEW_SIZE = 48


class ToolbarManager:
    """Manager for toolbars, currently just the main one.

    The main toolbar, /toolbar1, contains a menu button and quick
    access to the painting tools.
    """

    # icon_size = gtk.icon_size_register("MYPAINT_TOOLBAR_ICON_SIZE", 32, 32)
    icon_size = gtk.ICON_SIZE_LARGE_TOOLBAR

    def __init__(self, draw_window):
        self.draw_window = draw_window
        self.app = draw_window.app
        self.toolbar1_ui_loaded = {}  # {name: merge_id, ...}
        self.init_actions()
        toolbarpath = os.path.join(self.app.datapath, FRAMEWORK_XML)
        self.app.ui_manager.add_ui_from_file(toolbarpath)
        self.toolbar1 = self.app.ui_manager.get_widget('/toolbar1')
        self.toolbar1.set_style(gtk.TOOLBAR_ICONS)
        self.toolbar1.set_icon_size(self.icon_size)
        self.toolbar1.set_border_width(0)
        self.toolbar1.connect("popup-context-menu",
            self.on_toolbar1_popup_context_menu)
        self.toolbar1_popup = self.app.ui_manager\
            .get_widget('/toolbar1-settings-menu')
        self.menu_button = MainMenuButton(_("MyPaint"), draw_window.popupmenu)
        self.menu_button.set_border_width(0)
        menu_toolitem = gtk.ToolItem()
        menu_toolitem.add(self.menu_button)
        for item in self.toolbar1:
            if isinstance(item, gtk.SeparatorToolItem):
                item.set_draw(False)
        self.init_proxies()
        self.toolbar1.insert(menu_toolitem, 0)

    def init_actions(self):
        ag = self.draw_window.action_group
        actions = []
        self.item_actions = [
                ColorDropdownToolAction("ColorDropdown", None,
                    _("Current Color"), None),
                BrushDropdownToolAction("BrushDropdown", None,
                    _("Current Brush"), None),
                BrushSettingsDropdownToolAction("BrushSettingsDropdown", None,
                    _("Brush Settings"), None),
                LineDropdownToolAction("LineDropdown", None,
                    _("Line Mode"), None),
                ]
        actions += self.item_actions

        self.settings_actions = []
        for name, ui_xml, label in MERGEABLE_XML:
            action = gtk.ToggleAction(name, label, None, None)
            action.connect("toggled", self.on_settings_toggle, ui_xml)
            self.settings_actions.append(action)
        actions += self.settings_actions

        for action in actions:
            ag.add_action(action)

    def init_proxies(self):
        for action in self.item_actions:
            for p in action.get_proxies():
                if hasattr(p, "set_app"):
                    p.set_app(self.app)
        # Merge in UI pieces based on the user's saved preferences
        for action in self.settings_actions:
            name = action.get_property("name")
            active = self.app.preferences["ui.toolbar_items"].get(name, False)
            action.set_active(active)
            action.toggled()

    def on_toolbar1_popup_context_menu(self, toolbar, x, y, button):
        menu = self.toolbar1_popup
        def _posfunc(*a):
            return x, y, True
        time = gdk.CURRENT_TIME
        data = None
        # GTK3: arguments have a different order, and "data" is required.
        # GTK3: Use keyword arguments for max compatibility.
        menu.popup(parent_menu_shell=None, parent_menu_item=None,
                   func=_posfunc, button=button, activate_time=time,
                   data=None)

    def on_settings_toggle(self, toggleaction, ui_xml_file):
        name = toggleaction.get_property("name")
        merge_id = self.toolbar1_ui_loaded.get(name, None)
        if toggleaction.get_active():
            self.app.preferences["ui.toolbar_items"][name] = True
            if merge_id is not None:
                return
            ui_xml_path = os.path.join(self.app.datapath, ui_xml_file)
            merge_id = self.app.ui_manager.add_ui_from_file(ui_xml_path)
            self.toolbar1_ui_loaded[name] = merge_id
        else:
            self.app.preferences["ui.toolbar_items"][name] = False
            if merge_id is None:
                return
            self.app.ui_manager.remove_ui(merge_id)
            self.toolbar1_ui_loaded.pop(name)



from curve import CurveWidget

class LineDropdownToolItem (gtk.ToolItem):
    """Dropdown panel on the toolbar for changing line mode.
    """

    __gtype_name__ = "LineDropdownToolItem"
    settings_coordinate = [('entry_pressure', (0,1)),
                           ('midpoint_pressure', (1,1)),
                           ('exit_pressure', (3,1)),
                           ('line_head', (1,0)),
                           ('line_tail', (2,0))]
    action_names = ["SwitchableFreehandMode", "StraightMode",
                    "SequenceMode", "EllipseMode"]


    def __init__(self):
        gtk.ToolItem.__init__(self)
        self.set_homogeneous(False)
        self.button_image = gtk.Image()
        self.button_image.set_from_icon_name('mypaint-line-mode', ToolbarManager.icon_size)
        self.line_mode_panel = dropdownpanel.DropdownPanelButton(self.button_image)
        self.vbox = gtk.VBox()
        self.vbox.set_border_width(widgets.SPACING_TIGHT)
        self.vbox.set_spacing(widgets.SPACING_TIGHT)
        frame = gtk.Frame()
        frame.add(self.vbox)
        frame.set_shadow_type(gtk.SHADOW_OUT)
        self.line_mode_panel.set_property("panel-widget", frame)
        self.add(self.line_mode_panel)
        self.connect("create-menu-proxy", lambda *a: True)


    def set_app(self, app):
        """Deferred initialization: provides the main app instance.
        """

        self.app = app

        # Action switcher buttons
        bbox = gtk.HButtonBox()
        frame = widgets.section_frame(_("Line Mode"))
        frame.add(bbox)
        bbox.set_border_width(widgets.SPACING)
        for action_name in self.action_names:
            action = app.find_action(action_name)
            if action.get_active():
                self.update_icon_from_action(action)
            action.connect("changed", self.linemode_action_changed_cb)
            button = gtk.ToggleButton()
            button.set_related_action(action)
            button.connect("clicked", self.linemode_button_clicked_cb)
            button.set_can_focus(False)
            button.set_can_default(False)
            button.set_image_position(gtk.POS_TOP)
            button.set_relief(gtk.RELIEF_HALF)
            image = action.create_icon(gtk.ICON_SIZE_LARGE_TOOLBAR)
            image.set_padding(widgets.SPACING_TIGHT, widgets.SPACING_TIGHT)
            button.set_image(image)
            bbox.pack_start(button)
        self.vbox.pack_start(frame, False, False)
        bbox.show()

        # Pressure settings.
        frame = widgets.section_frame(_("Line Pressure"))
        self.vbox.pack_start(frame, True, True)
        curve = CurveWidget(npoints=4,
                            ylockgroups=((1,2),),
                            changed_cb=self.curve_changed_cb)
        curve_align = gtk.Alignment(0, 0, 1, 1)
        curve_align.add(curve)
        curve_align.set_padding(widgets.SPACING, widgets.SPACING,
                                widgets.SPACING, widgets.SPACING)
        frame.add(curve_align)
        curve_align.show()
        curve.points = [(0.0,0.2), (0.33,.5),(0.66, .5), (1.0,.33)]
        for setting, coord_pair in self.settings_coordinate:
            adj = app.line_mode_settings.adjustments[setting]
            value = adj.get_value()
            index, subindex = coord_pair
            if not setting.startswith ('line'):
                value = 1.0 - value
            coord = None
            if subindex == 0:
                coord = (value, curve.points[index][1])
            else:
                coord = (curve.points[index][0], value )
            curve.set_point(index, coord)
        self.curve_changed_cb (curve)


    def update_icon_from_action(self, action):
        """Updates the icon based on an action's icon.
        """
        icon_name = action.get_icon_name()
        self.button_image.set_from_icon_name(icon_name, ToolbarManager.icon_size)


    def linemode_action_changed_cb(self, action, current_action):
        """Updates the dropdown button when the line mode changes."""
        if action is current_action:
            self.update_icon_from_action(action)


    def linemode_button_clicked_cb(self, widget):
        """Dismisses the dropdown panel when a linemode button is clicked.
        """
        gobject.idle_add(self.line_mode_panel.panel_hide)


    def curve_changed_cb(self, curve):
        """Updates the linemode pressure settings when the curve is altered.
        """
        for setting, coord_pair in self.settings_coordinate:
            index, subindex = coord_pair
            points = curve.points
            value = curve.points[index][subindex]
            if not setting.startswith('line'):
                value = 1.0 - value
            value = max(0.0001, value)
            adj = self.app.line_mode_settings.adjustments[setting]
            adj.set_value(value)


class ColorDropdownToolItem (gtk.ToolItem):
    """Toolbar colour indicator, history access, and changer.
    """

    __gtype_name__ = "ColorDropdownToolItem"

    def __init__(self, *a, **kw):
        gtk.ToolItem.__init__(self)
        self.main_blob = ColorBlob()
        self.dropdown_button = dropdownpanel.DropdownPanelButton(self.main_blob)
        self.app = None
        self.blob_size = ToolbarManager.icon_size
        self.connect("toolbar-reconfigured", self.on_toolbar_reconf)
        self.connect("create-menu-proxy", lambda *a: True)
        self.set_tooltip_text(_("Color History and other tools"))
        self.add(self.dropdown_button)

    def set_app(self, app):
        self.app = app
        self.app.brush.observers.append(self.on_brush_settings_changed)
        self.main_blob.color = self.app.brush_color_manager.get_color()

        panel_frame = gtk.Frame()
        panel_frame.set_shadow_type(gtk.SHADOW_OUT)
        self.dropdown_button.set_property("panel-widget", panel_frame)
        panel_vbox = gtk.VBox()
        panel_vbox.set_spacing(widgets.SPACING_TIGHT)
        panel_vbox.set_border_width(widgets.SPACING)
        panel_frame.add(panel_vbox)

        def hide_panel_cb(*a):
            self.dropdown_button.panel_hide()

        def hide_panel_idle_cb(*a):
            gobject.idle_add(self.dropdown_button.panel_hide)

        # Colour changing
        section_frame = widgets.section_frame(_("Change Color"))
        panel_vbox.pack_start(section_frame, True, True)

        section_table = gtk.Table()
        section_table.set_col_spacings(widgets.SPACING)
        section_table.set_border_width(widgets.SPACING)
        section_frame.add(section_table)

        hsv_widget = HSVTriangle()
        hsv_widget.set_size_request(175, 175)
        hsv_widget.set_color_manager(app.brush_color_manager)
        section_table.attach(hsv_widget, 0, 1, 0, 1)

        preview_hbox = gtk.HBox()
        color_picker = ColorPickerButton()
        preview_adj = PreviousCurrentColorAdjuster()
        preview_adj.set_color_manager(app.brush_color_manager)
        color_picker.set_color_manager(app.brush_color_manager)
        preview_hbox.pack_start(color_picker, False, False)
        preview_hbox.pack_start(preview_adj, True, True)

        side_vbox = gtk.VBox()
        side_vbox.set_spacing(widgets.SPACING_TIGHT)
        section_table.attach(side_vbox, 1, 2, 0, 1)

        def init_proxy(widget, action_name):
            action = self.app.find_action(action_name)
            widget.set_related_action(action)
            widget.connect("clicked", hide_panel_cb)
            return widget

        button = init_proxy(gtk.Button(), "ColorDetailsDialog")
        side_vbox.pack_end(button, False, False)
        side_vbox.pack_end(preview_hbox, False, False)

        side_vbox.pack_end(gtk.Alignment(), True, True)
        button = init_proxy(gtk.ToggleButton(), "ColorWindow")
        button.set_label(_("Color Window"))
        side_vbox.pack_end(button, False, False)

        # History
        section_frame = widgets.section_frame(_("Recently Used"))
        panel_vbox.pack_start(section_frame, True, True)

        history_view = ColorHistoryView(self)
        section_frame.add(history_view)

    def on_history_button_clicked(self):
        self.dropdown_button.panel_hide()

    def on_toolbar_reconf(self, toolitem):
        toolbar = self.get_parent()
        lookup_ret = gtk.icon_size_lookup(self.get_icon_size())
        if pygtkcompat.USE_GTK3:
            lookup_succeeded, iw, ih = lookup_ret
            assert lookup_succeeded
        else:
            iw, ih = lookup_ret
        self.blob_size = max(iw, ih)
        self.main_blob.set_size_request(iw, ih)

    def on_brush_settings_changed(self, changes):
        if not changes.intersection(set(['color_h', 'color_s', 'color_v'])):
            return
        mgr = self.app.brush_color_manager
        self.main_blob.color = mgr.get_color()



class BrushDropdownToolItem (gtk.ToolItem):
    """Toolbar brush indicator, history access, and changer.
    """

    __gtype_name__ = "BrushDropdownToolItem"

    def __init__(self):
        gtk.ToolItem.__init__(self)
        self.history_images = []
        self.main_image = ManagedBrushPreview()
        self.dropdown_button = dropdownpanel.DropdownPanelButton(self.main_image)
        self.app = None
        self.image_size = ToolbarManager.icon_size
        self.connect("toolbar-reconfigured", self.on_toolbar_reconf)
        self.connect("create-menu-proxy", lambda *a: True)
        self.set_tooltip_text(_("Brush history etc."))
        self.add(self.dropdown_button)

    def set_app(self, app):
        self.app = app
        bm = self.app.brushmanager
        bm.selected_brush_observers.append(self.on_selected_brush)
        self.app.doc.input_stroke_ended_observers\
            .append(self.doc_input_stroke_ended_cb)
        self.update_history_images()

        panel_frame = gtk.Frame()
        panel_frame.set_shadow_type(gtk.SHADOW_OUT)
        self.dropdown_button.set_property("panel-widget", panel_frame)
        panel_vbox = gtk.VBox()
        panel_vbox.set_spacing(widgets.SPACING_TIGHT)
        panel_vbox.set_border_width(widgets.SPACING)
        panel_frame.add(panel_vbox)

        # Quick brush changer
        section_frame = widgets.section_frame(_("Change Brush"))
        panel_vbox.pack_start(section_frame, True, True)

        section_vbox = gtk.VBox()
        section_vbox.set_border_width(widgets.SPACING)
        section_vbox.set_spacing(widgets.SPACING_TIGHT)
        section_frame.add(section_vbox)

        quick_changer = dialogs.QuickBrushChooser(app, self.on_quick_change_select)
        evbox = gtk.EventBox()
        evbox.add(quick_changer)
        section_vbox.pack_start(evbox, True, True)

        # List editor button
        list_editor_button = gtk.ToggleButton()
        list_editor_action = self.app.find_action("BrushSelectionWindow")
        list_editor_button.set_related_action(list_editor_action)
        close_panel_cb = lambda *a: self.dropdown_button.panel_hide()
        list_editor_button.connect("clicked", close_panel_cb)
        section_vbox.pack_start(list_editor_button, False, False)

        # Brush history
        section_frame = widgets.section_frame(_("Recently Used"))
        panel_vbox.pack_start(section_frame, True, True)

        history_hbox = gtk.HBox()
        history_hbox.set_border_width(widgets.SPACING)
        section_frame.add(history_hbox)
        for i, image in enumerate(self.history_images):
            button = widgets.borderless_button()
            button.add(image)
            button.connect("clicked", self.on_history_button_clicked, i)
            history_hbox.pack_end(button, True, True)


    def doc_input_stroke_ended_cb(self, event):
        gobject.idle_add(self.update_history_images)


    def update_history_images(self):
        bm = self.app.brushmanager
        if not self.history_images:
            s = HISTORY_PREVIEW_SIZE
            for brush in bm.history:
                image = ManagedBrushPreview()
                image.set_size_request(s, s)
                self.history_images.append(image)
        for i, brush in enumerate(bm.history):
            image = self.history_images[i]
            image.set_from_managed_brush(brush)


    def on_toolbar_reconf(self, toolitem):
        toolbar = self.get_parent()
        lookup_ret = gtk.icon_size_lookup(self.get_icon_size())
        if pygtkcompat.USE_GTK3:
            lookup_succeeded, iw, ih = lookup_ret
            assert lookup_succeeded
        else:
            iw, ih = lookup_ret
        self.image_size = max(iw, ih)
        self.main_image.set_size_request(iw, ih)


    def on_selected_brush(self, brush, brushinfo):
        self.main_image.set_from_managed_brush(brush)


    def on_history_button_clicked(self, button, i):
        bm = self.app.brushmanager
        brush = bm.history[i]
        bm.select_brush(brush)
        self.dropdown_button.panel_hide()


    def on_quick_change_select(self, brush):
        self.dropdown_button.panel_hide(immediate=False)
        self.app.brushmanager.select_brush(brush)



class BrushSettingsDropdownToolItem (gtk.ToolItem):
    __gtype_name__ = "BrushSettingsDropdownToolItem"

    active_stock_id = 'mypaint-brush-mods-active'
    inactive_stock_id = 'mypaint-brush-mods-inactive'

    setting_cnames = ["radius_logarithmic", "slow_tracking", "opaque", "pressure_gain_log"]

    blend_modes_table = [
        (0, 1, 0, 1, "BlendModeNormal"),
        (0, 1, 1, 2, "BlendModeEraser"),
        (1, 2, 0, 1, "BlendModeLockAlpha"),
        (1, 2, 1, 2, "BlendModeColorize"),  ]

    def __init__(self):
        gtk.ToolItem.__init__(self)
        self.set_homogeneous(False)
        self.button_image = gtk.Image()
        self.button_image.set_from_stock(self.inactive_stock_id,
                                         ToolbarManager.icon_size)
        self.button_shows_modified = False
        self.button = dropdownpanel.DropdownPanelButton(self.button_image)
        self.vbox = gtk.VBox()
        frame = gtk.Frame()
        frame.add(self.vbox)
        frame.set_shadow_type(gtk.SHADOW_OUT)
        self.button.set_property("panel-widget", frame)
        self.add(self.button)
        self.connect("create-menu-proxy", lambda *a: True)

    def set_app(self, app):
        self.app = app

        # A limited subset of the available brush settings.

        frame = widgets.section_frame(_("Quick Brush Settings"))
        table = gtk.Table()
        table.set_homogeneous(False)
        table.set_row_spacings(widgets.SPACING_TIGHT)
        table.set_col_spacings(widgets.SPACING)
        table.set_border_width(widgets.SPACING)
        frame.add(table)
        self.vbox.pack_start(frame, True, True)

        sg_row_height = gtk.SizeGroup(gtk.SIZE_GROUP_VERTICAL)
        sg_slider_width = gtk.SizeGroup(gtk.SIZE_GROUP_HORIZONTAL)
        row = 0
        for setting_cname in self.setting_cnames:
            scale = gtk.HScale()
            scale.set_size_request(128, -1)
            scale.set_draw_value(False)
            scale.set_can_focus(False)
            scale.set_can_default(False)
            s = brushsettings.settings_dict[setting_cname]
            adj = app.brush_adjustment[setting_cname]
            scale.set_adjustment(adj)
            scale.set_tooltip_text(s.tooltip)
            #scale.set_update_policy(gtk.UPDATE_DISCONTINUOUS)
            sg_row_height.add_widget(scale)
            sg_slider_width.add_widget(scale)

            label = gtk.Label(_("%s:") % s.name)
            label.set_alignment(0.0, 0.5)
            label.set_tooltip_text(s.tooltip)
            sg_row_height.add_widget(label)

            reset_button = widgets.borderless_button(
                stock_id=gtk.STOCK_CLEAR,
                tooltip=_("Restore '%s' to this brush's saved value") % s.name)
            reset_button.connect("clicked", self.reset_button_clicked_cb,
                                 adj, setting_cname)
            sg_row_height.add_widget(reset_button)

            adj.connect("value-changed", self.adjustment_changed_cb,
                reset_button, setting_cname)
            adj.value_changed()

            table.attach(label, 0, 1, row, row+1, gtk.FILL)
            table.attach(scale, 1, 2, row, row+1, gtk.FILL|gtk.EXPAND)
            table.attach(reset_button, 2, 3, row, row+1, gtk.FILL)
            row += 1
        table.set_col_spacing(1, widgets.SPACING_TIGHT)

        # Access to the brush settings window, and a big reset-all button
        # aligned with the settings above.

        frame = widgets.section_frame(_("Detailed Brush Settings"))
        hbox = gtk.HBox()
        hbox.set_spacing(widgets.SPACING)
        hbox.set_border_width(widgets.SPACING)
        frame.add(hbox)
        self.vbox.pack_start(frame, True, True)

        widget = gtk.ToggleButton()
        action = self.app.find_action("BrushSettingsWindow")
        widget.set_related_action(action)
        #widget.set_label(_("Edit All Settings"))
        hbox.pack_start(widget, True, True)
        widget.connect("toggled", lambda a: self.button.panel_hide())
        sg_slider_width.add_widget(widget)

        widget = gtk.Button(_("Restore Saved Settings"))
        widget.connect("clicked", self.reset_all_clicked_cb)
        widget.set_tooltip_text(_("Reset all brush settings to the current brush's saved values"))
        hbox.pack_start(widget, True, True)
        sg_slider_width.add_widget(widget)
        self.reset_all_button = widget

        # Brush blend modes

        vbox = gtk.VBox()
        vbox.set_border_width(widgets.SPACING)
        vbox.set_spacing(widgets.SPACING_TIGHT)
        frame = widgets.section_frame(_("Brush Blend Mode"))
        frame.add(vbox)
        self.vbox.pack_start(frame, True, True)

        table = gtk.Table(2, 2, homogeneous=True)
        topts = gtk.EXPAND|gtk.FILL
        tpad = 0
        table.set_row_spacings(widgets.SPACING_TIGHT)
        table.set_col_spacings(widgets.SPACING)
        for la, ra, ta, ba, action_name in self.blend_modes_table:
            action = self.app.find_action(action_name)
            button = gtk.ToggleButton()
            button.set_related_action(action)
            button.set_can_focus(False)
            button.set_can_default(False)
            button.set_image_position(gtk.POS_LEFT)
            button.set_alignment(0.0, 0.5)
            button.set_relief(gtk.RELIEF_HALF)
            image = action.create_icon(gtk.ICON_SIZE_BUTTON)
            image.set_padding(widgets.SPACING_TIGHT, widgets.SPACING_TIGHT)
            button.set_image(image)
            table.attach(button, la, ra, ta, ba, topts, topts, tpad, tpad)
            button.connect("clicked", self.blendmode_button_clicked_cb)
        vbox.pack_start(table, False, False)

        self.vbox.set_border_width(widgets.SPACING)
        self.vbox.set_spacing(widgets.SPACING)

        self.app.brush.observers.append(self.brush_settings_changed_cb)


    def blendmode_button_clicked_cb(self, widget):
        self.button.panel_hide()


    def reset_button_clicked_cb(self, widget, adj, setting_cname):
        default = self._get_current_brush_default(setting_cname)
        adj.set_value(default)


    def reset_all_clicked_cb(self, widget):
        parent_brush = self.app.brushmanager.get_parent_brush(brushinfo=self.app.brush)
        self.app.brushmanager.select_brush(parent_brush)
        self.app.brushmodifier.normal_mode.activate()
        self.button.panel_hide()


    def adjustment_changed_cb(self, widget, button, cname):
        default = self._get_current_brush_default(cname)
        button.set_sensitive(widget.get_value() != default)
        self.button.panel_hide(immediate=False, release=False, leave=True)


    def _get_current_brush_default(self, cname):
        bm = self.app.brushmanager
        parent = bm.get_parent_brush(brushinfo=self.app.brush)
        if parent is None:
            return brushsettings.settings_dict[cname].default
        else:
            return parent.brushinfo.get_base_value(cname)


    def brush_settings_changed_cb(self, *a):
        stock_id = None
        if self._current_brush_is_modified():
            if not self.button_shows_modified:
                stock_id = self.active_stock_id
                self.button_shows_modified = True
            if not self.reset_all_button.get_sensitive():
                self.reset_all_button.set_sensitive(True)
        else:
            if self.button_shows_modified:
                stock_id = self.inactive_stock_id
                self.button_shows_modified = False
            if self.reset_all_button.get_sensitive():
                self.reset_all_button.set_sensitive(False)
        if stock_id is not None:
            self.button_image.set_from_stock(stock_id, ToolbarManager.icon_size)


    def _current_brush_is_modified(self):
        current_bi = self.app.brush
        parent_b = self.app.brushmanager.get_parent_brush(brushinfo=current_bi)
        if parent_b is None:
            return True
        return not parent_b.brushinfo.matches(current_bi)


class ColorDropdownToolAction (gtk.Action):
    __gtype_name__ = "ColorDropdownToolAction"
    def do_create_tool_item(self):
        return ColorDropdownToolItem()


class LineDropdownToolAction (gtk.Action):
    __gtype_name__ = "LineDropdownToolAction"
    def do_create_tool_item(self):
        return LineDropdownToolItem()


class BrushDropdownToolAction (gtk.Action):
    __gtype_name__ = "BrushDropdownToolAction"
    def do_create_tool_item(self):
        return BrushDropdownToolItem()


class BrushSettingsDropdownToolAction (gtk.Action):
    __gtype_name__ = "BrushSettingsDropdownToolAction"
    def do_create_tool_item(self):
        return BrushSettingsDropdownToolItem()


class ManagedBrushPreview (gtk.Image):
    """Updateable widget displaying a brushmanager.ManagedBrush`'s preview.
    """

    ICON_SIZE = 32
    TOOLTIP_ICON_SIZE = 48

    def __init__(self, brush=None):
        gtk.Image.__init__(self)
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
        new_size = alloc.width, alloc.height
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
            theme = gtk.icon_theme_get_default()
            return theme.load_icon(gtk.STOCK_MISSING_IMAGE, size, 0)
        else:
            return self.pixbuf.scale_simple(size, size, gdk.INTERP_BILINEAR)

    def on_query_tooltip(self, widget, x, y, keyboard_mode, tooltip):
        s = self.TOOLTIP_ICON_SIZE
        scaled_pixbuf = self._get_scaled_pixbuf(s)
        tooltip.set_icon(scaled_pixbuf)
        tooltip.set_text(self.brush_name)  # TODO: use markup, and summarize changes (i18n fun)
        return True

    def _update(self):
        if not self.image_size:
            return
        w, h = self.image_size
        s = min(w, h)
        scaled_pixbuf = self._get_scaled_pixbuf(s)
        self.set_from_pixbuf(scaled_pixbuf)


class ColorHistoryView (gtk.HBox, ColorAdjuster):
    """A set of ColorBlobs showing the usage history.
    """

    def __init__(self, toolitem):
        gtk.HBox.__init__(self)
        self.__history_blobs = []
        self.__app = toolitem.app
        self.__toolitem = toolitem
        self.set_border_width(widgets.SPACING)
        s = HISTORY_PREVIEW_SIZE
        mgr = self.__app.brush_color_manager
        for color in mgr.get_history():
            button = widgets.borderless_button()
            blob = ColorBlob(color)
            blob.set_size_request(s, s)
            button.add(blob)
            button.connect("clicked", self.__button_clicked_cb)
            self.pack_end(button, True, True)
            self.__history_blobs.append(blob)
        self.set_color_manager(mgr)

    def update_cb(self):
        mgr = self.get_color_manager()
        for blob, color in zip(self.__history_blobs, mgr.get_history()):
            blob.color = color

    def __button_clicked_cb(self, button):
        blob = button.get_child()
        mgr = self.get_color_manager()
        mgr.set_color(blob.color)
        self.__toolitem.on_history_button_clicked()


class ColorBlob (gtk.AspectFrame):
    """Updatable widget displaying a single colour.
    """

    def __init__(self, color=None):
        gtk.AspectFrame.__init__(self, xalign=0.5, yalign=0.5, ratio=1.0, obey_child=False)
        self.set_name("thinborder-color-blob-%d" % id(self))
        self.set_shadow_type(gtk.SHADOW_IN)
        self.drawingarea = gtk.DrawingArea()
        self.add(self.drawingarea)
        if color is None:
            color = RGBColor(0, 0, 0)
        self._color = color
        self.drawingarea.set_size_request(1, 1)
        if pygtkcompat.USE_GTK3:
            self.drawingarea.connect("draw", self.on_draw)
        else:
            self.drawingarea.connect("expose-event", self.on_expose)

    def set_color(self, color):
        self._color = color
        self.drawingarea.queue_draw()

    def get_color(self):
        return self._color

    color = property(get_color, set_color)

    def on_expose(self, widget, event):
        cr = widget.get_window().cairo_create()
        self.on_draw(widget, cr)

    def on_draw(self, widget, cr):
        cr.set_source_rgb(*self._color.get_rgb())
        cr.paint()


class MainMenuButton (gtk.ToggleButton):
    """Launches the popup menu when clicked.

    This sits inside the main toolbar when the main menu bar is hidden. In
    addition to providing access to the app's menu associated with the main
    view, this is a little more compliant with Fitts's Law than a normal
    `gtk.MenuBar`: our local style modifications mean that for most styles,
    when the window is fullscreened with only the "toolbar" present the
    ``(0,0)`` screen pixel hits this button.

    Support note: Compiz edge bindings sometimes get in the way of this, so
    turn those off if you want Fitts's compliance.
    """

    def __init__(self, text, menu):
        gtk.Button.__init__(self)
        self.menu = menu
        hbox1 = gtk.HBox()
        hbox2 = gtk.HBox()
        label = gtk.Label()
        hbox1.pack_start(label, True, True)
        arrow = gtk.Arrow(gtk.ARROW_DOWN, gtk.SHADOW_IN)
        hbox1.pack_start(arrow, False, False)
        hbox2.pack_start(hbox1, True, True, widgets.SPACING_TIGHT)

        # Text settings
        text = unicode(text)
        markup = "<b>%s</b>" % escape(text)
        label.set_markup(markup)

        self.add(hbox2)
        self.set_relief(gtk.RELIEF_NONE)
        self.connect("button-press-event", self.on_button_press)

        # No keynav.
        #DISABLED: self.connect("toggled", self.on_toggled)
        self.set_can_focus(False)
        self.set_can_default(False)

        for sig in "selection-done", "deactivate", "cancel":
            menu.connect(sig, self.on_menu_dismiss)


    def on_enter(self, widget, event):
        # Not this set_state(). That one.
        #self.set_state(gtk.STATE_PRELIGHT)
        gtk.Widget.set_state(self, gtk.STATE_PRELIGHT)


    def on_leave(self, widget, event):
        #self.set_state(gtk.STATE_NORMAL)
        gtk.Widget.set_state(self, gtk.STATE_NORMAL)


    def on_button_press(self, widget, event):
        # Post the menu. Menu operation is much more convincing if we call
        # popup() with event details here rather than leaving it to the toggled
        # handler.
        self._show_menu(event)
        self.set_active(True)
        return True


    ## Key nav only. We don't support it right now, so don't compile.
    #def on_toggled(self, togglebutton):
    #    # Post the menu from a keypress. Dismiss handler untoggles it.
    #    if togglebutton.get_active():
    #        if not self.menu.get_property("visible"):
    #            self._show_menu()


    def _show_menu(self, event=None):
        button = 1
        time = 0
        if event is not None:
            button = event.button
            time = event.time
        pos_func = self._get_popup_menu_position
        # GTK3: arguments have a different order, and "data" is required.
        # GTK3: Use keyword arguments for max compatibility.
        self.menu.popup(parent_menu_shell=None, parent_menu_item=None,
                        func=pos_func, button=button,
                        activate_time=time, data=None)


    def on_menu_dismiss(self, *a, **kw):
        # Reset the button state when the user's finished, and
        # park focus back on the menu button.
        self.set_state(gtk.STATE_NORMAL)
        self.set_active(False)
        self.grab_focus()


    def _get_popup_menu_position(self, menu, *junk):
        # Underneath the button, at the same x position.
        x, y = self.get_window().get_origin()
        y += self.get_allocation().height
        return x, y, True
