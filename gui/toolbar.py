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
from math import pi

import gtk
from gtk import gdk
import gobject
from gettext import gettext as _
import pango

from lib.helpers import hsv_to_rgb, clamp
import dialogs
from brushlib import brushsettings
import stock
import dropdownpanel
import widgets
from hsvcompat import ColorChangerHSV
import linemode

FRAMEWORK_XML = 'gui/toolbar.xml'
MERGEABLE_XML = [
    ("toolbar1_file", 'gui/toolbar-file.xml', _("File handling")),
    ("toolbar1_scrap", 'gui/toolbar-scrap.xml', _("Scraps switcher")),
    ("toolbar1_edit", 'gui/toolbar-edit.xml', _("Undo and Redo")),
    ("toolbar1_blendmodes", 'gui/toolbar-blendmodes.xml', _("Blend Modes")),
    ("toolbar1_editmodes", 'gui/toolbar-editmodes.xml', _("Editing Modes")),
    ("toolbar1_view", 'gui/toolbar-view.xml', _("View")),
    ("toolbar1_subwindows", 'gui/toolbar-subwindows.xml', _("Subwindows")),
    ]


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
                p.set_app(self.app)
        # Merge in UI pieces based on the user's saved preferences
        for action in self.settings_actions:
            name = action.get_property("name")
            active = self.app.preferences["ui.toolbar_items"].get(name, False)
            action.set_active(active)
            action.toggled()

    def on_toolbar1_popup_context_menu(self, toolbar, x, y, button):
        menu = self.toolbar1_popup
        def posfunc(m):
            return x, y, True
        menu.popup(None, None, posfunc, button, 0)

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


class LineDropdownToolItem (gtk.ToolItem):

    __gtype_name__ = "LineDropdownToolItem"
    linemode_settings = ['FreehandMode', 'StraightMode', 'SequenceMode', 'EllipseMode']
    pressure_settings = ["entry_pressure", "midpoint_pressure", "exit_pressure"]
    head_tail_settings = ["line_head", "line_tail"]
    shape_settings = ["entry_pressure", "midpoint_pressure", "exit_pressure", "line_head", "line_tail"]
    settings_coordinate = {'entry_pressure': (0,1),
                           'midpoint_pressure': (1,1),
                           'exit_pressure': (3,1),
                           'line_head': (1,0),
                           'line_tail': (2,0),
                           }



    def __init__(self):
        gtk.ToolItem.__init__(self)
        self.set_homogeneous(False)
        self.button_image = gtk.Image()
        self.button_image.set_from_stock(stock.LINE_MODES, ToolbarManager.icon_size)
        self.line_mode_panel = dropdownpanel.DropdownPanelButton(self.button_image)
        self.vbox = gtk.VBox()
        frame = gtk.Frame()
        frame.add(self.vbox)
        frame.set_shadow_type(gtk.SHADOW_OUT)
        self.line_mode_panel.set_property("panel-widget", frame)
        self.add(self.line_mode_panel)


    def set_app(self, app):
        self.app = app

        # Line modes.

        vbox = gtk.VBox()
        vbox.set_border_width(widgets.SPACING_TIGHT)
        vbox.set_spacing(widgets.SPACING_TIGHT)
        frame = widgets.section_frame(_("Line Mode"))
        frame.add(vbox)
        self.vbox.pack_start(frame, True, True)
        self.vbox.set_border_width(widgets.SPACING_TIGHT)
        self.vbox.set_spacing(widgets.SPACING_TIGHT)

        # Line Mode Icons

        for mode in self.linemode_settings:
            action = self.app.find_action(mode)
            hbox = gtk.HBox()
            icon = gtk.Image()
            icon.set_from_stock(action.stock_id, ToolbarManager.icon_size)
            hbox.pack_start(icon, False, True)
            label = gtk.ToggleButton()
            label.set_relief(gtk.RELIEF_NONE)
            label.set_tooltip_text(action.tooltip)
            action.connect_proxy(label)
            label.connect("toggled", lambda m: self.line_mode_panel.panel_hide())
            hbox.pack_start(label, False, True)
            vbox.pack_start(hbox, False, False)

        # Pressure settings.

        def settings_frame():
            self.vbox.pack_start(frame, True, True)
            from curve import FixedCurveWidget
            curve = FixedCurveWidget(npoints = 4,
                                     ylockgroups = ((1,2),),
                                     changed_cb = self.curve_changed_cb)
            frame.add(curve)
            curve.set_tooltip_text('Curve defining the amount of pressure applied at different points in the line.\n'
                                   '\nX position = distance along the line;\n'
                                   '    minimum X = start of line;\n'
                                   '    maximum X = end of line\n'
                                   '    (only the central two points can be adjusted in X axis)\n'
                                   'Y position = amount of pressure\n'
                                   '    The Y position of the central two points is locked together.\n')
            #vbox.pack_start(w, True, True)
            curve.show()
            curve.points = [(0.0,0.2), (0.33,.5),(0.66, .5), (1.0,.33)]
            for setting in (self.shape_settings):
                value = app.line_mode_adjustment[setting].get_value()
                index, subindex = self.settings_coordinate[setting]
                if not setting.startswith ('line'):#if setting != 'line_head
                    value = 1.0 - value
                coord = None
                if subindex == 0:
                    coord = (value, curve.points[index][1])
                else:
                    coord = (curve.points[index][0], value )
                curve.set_point(index, coord)
            self.curve_changed_cb (curve)

        frame = widgets.section_frame(_("Line Shape"))
        settings_frame()

    def curve_changed_cb(self, curve):
        for setting in self.shape_settings:
            coord = self.settings_coordinate [setting]
            points = curve.points
            value = curve.points[coord[0]][coord[1]]
            if not setting.startswith('line'):
                value = 1.0 - value
            value = max(0.0001, value)
            #print ('setting %r (%s) to %f' % (coord, setting, value))
            #if setting.startswith('line_'):
            #    setting = {'line_tail':'line_head', 'line_head':'line_tail'}[setting]
            self.app.linemode.change_line_setting(setting, value)
        #print (curve.points)


class ColorDropdownToolItem (gtk.ToolItem):
    """Toolbar colour indicator, history access, and changer.
    """

    __gtype_name__ = "ColorDropdownToolItem"

    HISTORY_PREVIEW_SIZE = 48

    def __init__(self, *a, **kw):
        gtk.ToolItem.__init__(self)
        self.history_blobs = []
        self.main_blob = ColorBlob()
        self.dropdown_button = dropdownpanel.DropdownPanelButton(self.main_blob)
        self.app = None
        self.blob_size = ToolbarManager.icon_size
        self.connect("toolbar-reconfigured", self.on_toolbar_reconf)
        self.connect("create-menu-proxy", self.on_create_menu_proxy)
        self.set_tooltip_text(_("Color History and other tools"))
        self.add(self.dropdown_button)

    def set_app(self, app):
        self.app = app
        self.app.brush.observers.append(self.on_brush_settings_changed)
        self.main_blob.hsv = self.app.brush.get_color_hsv()

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

        hsv_widget = ColorChangerHSV(app, details=False)
        hsv_widget.set_size_request(175, 175)
        section_table.attach(hsv_widget, 0, 1, 0, 1)

        def is_preview_hbox(w):
            return isinstance(w, gtk.HBox) and isinstance(w.parent, gtk.VBox)
        preview_hbox, = widgets.find_widgets(hsv_widget, is_preview_hbox)
        preview_hbox.parent.remove(preview_hbox)

        def is_color_picker(w):
            return isinstance(w, gtk.Button)
        color_picker, = widgets.find_widgets(preview_hbox, is_color_picker)
        color_picker.connect("clicked", hide_panel_idle_cb)

        side_vbox = gtk.VBox()
        side_vbox.set_spacing(widgets.SPACING_TIGHT)
        section_table.attach(side_vbox, 1, 2, 0, 1)

        def init_proxy(widget, action_name):
            action = self.app.find_action(action_name)
            action.connect_proxy(widget)
            widget.connect("clicked", hide_panel_cb)
            return widget

        button = init_proxy(gtk.Button(), "ColorDetailsDialog")
        side_vbox.pack_end(button, False, False)
        side_vbox.pack_end(preview_hbox, False, False)

        side_vbox.pack_end(gtk.Alignment(), True, True)
        button = init_proxy(gtk.ToggleButton(), "ColorSamplerWindow")
        side_vbox.pack_end(button, False, False)
        button = init_proxy(gtk.ToggleButton(), "ColorSelectionWindow")
        side_vbox.pack_end(button, False, False)

        # History
        section_frame = widgets.section_frame(_("Recently Used"))
        panel_vbox.pack_start(section_frame, True, True)

        self.history_blobs = []
        history_hbox = gtk.HBox()
        history_hbox.set_border_width(widgets.SPACING)
        s = self.HISTORY_PREVIEW_SIZE
        for hsv in app.ch.colors:
            button = widgets.borderless_button()
            blob = ColorBlob(hsv)
            blob.set_size_request(s, s)
            button.add(blob)
            button.connect("clicked", self.on_history_button_clicked, blob)
            history_hbox.pack_end(button, True, True)
            self.history_blobs.append(blob)
        app.ch.color_pushed_observers.append(self.color_pushed_cb)
        section_frame.add(history_hbox)

    def color_pushed_cb(self, color):
        for blob, hsv in zip(self.history_blobs, self.app.ch.colors):
            blob.hsv = hsv

    def on_history_button_clicked(self, widget, blob):
        self.app.brush.set_color_hsv(blob.hsv)
        self.dropdown_button.panel_hide()

    def on_create_menu_proxy(self, toolitem):
        # Do not appear on the overflow menu.
        # Though possibly just duplicating the custom items into a submenu
        # would work here.
        self.set_proxy_menu_item("", None)
        return True

    def on_toolbar_reconf(self, toolitem):
        toolbar = self.parent
        iw, ih = gtk.icon_size_lookup(self.get_icon_size())
        self.blob_size = max(iw, ih)
        self.main_blob.set_size_request(iw, ih)

    def on_brush_settings_changed(self, changes):
        if not changes.intersection(set(['color_h', 'color_s', 'color_v'])):
            return
        self.main_blob.hsv = self.app.brush.get_color_hsv()



class BrushDropdownToolItem (gtk.ToolItem):
    """Toolbar brush indicator, history access, and changer.
    """

    __gtype_name__ = "BrushDropdownToolItem"

    HISTORY_PREVIEW_SIZE = 48

    def __init__(self):
        gtk.ToolItem.__init__(self)
        self.history_images = []
        self.main_image = ManagedBrushPreview()
        self.dropdown_button = dropdownpanel.DropdownPanelButton(self.main_image)
        self.app = None
        self.image_size = ToolbarManager.icon_size
        self.connect("toolbar-reconfigured", self.on_toolbar_reconf)
        self.connect("create-menu-proxy", self.on_create_menu_proxy)
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
        list_editor_action.connect_proxy(list_editor_button)
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


    def on_create_menu_proxy(self, toolitem):
        self.set_proxy_menu_item("", None)
        return True


    def doc_input_stroke_ended_cb(self, event):
        gobject.idle_add(self.update_history_images)


    def update_history_images(self):
        bm = self.app.brushmanager
        if not self.history_images:
            s = self.HISTORY_PREVIEW_SIZE
            for brush in bm.history:
                image = ManagedBrushPreview()
                image.set_size_request(s, s)
                self.history_images.append(image)
        for i, brush in enumerate(bm.history):
            image = self.history_images[i]
            image.set_from_managed_brush(brush)


    def on_toolbar_reconf(self, toolitem):
        toolbar = self.parent
        iw, ih = gtk.icon_size_lookup(self.get_icon_size())
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

    setting_cnames = ["radius_logarithmic", "slow_tracking", "opaque", "hardness"]

    def __init__(self):
        gtk.ToolItem.__init__(self)
        self.set_homogeneous(False)
        self.button_image = gtk.Image()
        self.button_image.set_from_stock(stock.BRUSH_MODIFIERS_INACTIVE,
                                         ToolbarManager.icon_size)
        self.button_shows_modified = False
        self.button = dropdownpanel.DropdownPanelButton(self.button_image)
        self.vbox = gtk.VBox()
        frame = gtk.Frame()
        frame.add(self.vbox)
        frame.set_shadow_type(gtk.SHADOW_OUT)
        self.button.set_property("panel-widget", frame)
        self.add(self.button)

    def set_app(self, app):
        self.app = app

        # A limited subset of the abailable brush settings.

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
        action.connect_proxy(widget)
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

        for a in ["BlendModeNormal", "BlendModeEraser",
                  "BlendModeLockAlpha", "BlendModeColorize"]:
            action = self.app.find_action(a)
            cb = gtk.CheckButton()
            action.connect_proxy(cb)
            vbox.pack_start(cb, False, False)
            cb.connect("clicked", lambda a: self.button.panel_hide())

        self.vbox.set_border_width(widgets.SPACING)
        self.vbox.set_spacing(widgets.SPACING)

        self.app.brush.observers.append(self.brush_settings_changed_cb)


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
                stock_id = stock.BRUSH_MODIFIERS_ACTIVE
                self.button_shows_modified = True
            if not self.reset_all_button.get_sensitive():
                self.reset_all_button.set_sensitive(True)
        else:
            if self.button_shows_modified:
                stock_id = stock.BRUSH_MODIFIERS_INACTIVE
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
ColorDropdownToolAction.set_tool_item_type(ColorDropdownToolItem)


class LineDropdownToolAction (gtk.Action):
    __gtype_name__ = "LineDropdownToolAction"
LineDropdownToolAction.set_tool_item_type(LineDropdownToolItem)


class BrushDropdownToolAction (gtk.Action):
    __gtype_name__ = "BrushDropdownToolAction"
BrushDropdownToolAction.set_tool_item_type(BrushDropdownToolItem)


class BrushSettingsDropdownToolAction (gtk.Action):
    __gtype_name__ = "BrushSettingsDropdownToolAction"
BrushSettingsDropdownToolAction\
        .set_tool_item_type(BrushSettingsDropdownToolItem)


class ManagedBrushPreview (gtk.Image):
    """Updateable widget displaying a brushmanager.ManagedBrush`'s preview.
    """

    TOOLTIP_ICON_SIZE = 48

    def __init__(self, brush=None):
        gtk.Image.__init__(self)
        self.pixbuf = None
        self.image_size = None
        self.brush_name = None
        self.set_from_managed_brush(brush)
        self.set_size_request(32, 32)
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
        if new_size != self.image_size:
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
        tooltip.set_text(self.brush_name)  # XXX markup and summary of changes
        return True

    def _update(self):
        if not self.image_size:
            return
        w, h = self.image_size
        s = min(w, h)
        scaled_pixbuf = self._get_scaled_pixbuf(s)
        self.set_from_pixbuf(scaled_pixbuf)



class ColorBlob (gtk.AspectFrame):
    """Updatable widget displaying a single colour.
    """

    def __init__(self, hsv=None):
        gtk.AspectFrame.__init__(self, xalign=0.5, yalign=0.5, ratio=1.0, obey_child=False)
        self.set_name("thinborder-color-blob-%d" % id(self))
        self.set_shadow_type(gtk.SHADOW_IN)
        self.drawingarea = gtk.DrawingArea()
        self.add(self.drawingarea)
        if hsv is None:
            hsv = 0.0, 0.0, 0.0
        self._hsv = hsv
        self.drawingarea.set_size_request(1, 1)
        self.drawingarea.connect("expose-event", self.on_expose)

    def set_hsv(self, hsv):
        self._hsv = hsv
        self.drawingarea.queue_draw()

    def get_hsv(self):
        return self._hsv

    hsv = property(get_hsv, set_hsv)

    def on_expose(self, widget, event):
        cr = widget.window.cairo_create()
        cr.set_source_rgb(*hsv_to_rgb(*self._hsv))
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
        label = gtk.Label(text)
        hbox1.pack_start(label, True, True)
        arrow = gtk.Arrow(gtk.ARROW_DOWN, gtk.SHADOW_IN)
        hbox1.pack_start(arrow, False, False)
        hbox2.pack_start(hbox1, True, True, widgets.SPACING_TIGHT)

        # Text settings
        attrs = pango.AttrList()
        attrs.change(pango.AttrWeight(pango.WEIGHT_SEMIBOLD, 0, -1))
        label.set_attributes(attrs)

        self.add(hbox2)
        self.set_relief(gtk.RELIEF_NONE)
        self.set_can_focus(True)
        self.set_can_default(False)
        self.connect("toggled", self.on_toggled)

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
        pos_func = self._get_popup_menu_position
        self.menu.popup(None, None, pos_func, event.button, event.time)
        self.set_active(True)


    def on_toggled(self, togglebutton):
        # Post the menu from a keypress. Dismiss handler untoggles it.
        if togglebutton.get_active():
            if not self.menu.get_property("visible"):
                pos_func = self._get_popup_menu_position
                self.menu.popup(None, None, pos_func, 1, 0)


    def on_menu_dismiss(self, *a, **kw):
        # Reset the button state when the user's finished, and
        # park focus back on the menu button.
        self.set_state(gtk.STATE_NORMAL)
        self.set_active(False)
        self.grab_focus()


    def _get_popup_menu_position(self, menu, *junk):
        # Underneath the button, at the same x position.
        x, y = self.window.get_origin()
        y += self.allocation.height
        return x, y, True
