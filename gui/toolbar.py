# This file is part of MyPaint.
# Copyright (C) 2011-2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""The application toolbar, and its specialised widgets"""


## Imports

import os

import gtk2compat
import gtk
from gtk import gdk
import gobject
from gettext import gettext as _
import pango
from libmypaint import brushsettings

import factoryaction
import dialogs
import dropdownpanel
import widgets
from colors import RGBColor, ColorAdjuster, HSVTriangle
from colors import PreviousCurrentColorAdjuster, ColorPickerButton
from history import ColorHistoryView, BrushHistoryView
from history import ManagedBrushPreview, ColorPreview
from lib.helpers import escape
from linemode import LineModeCurveWidget


## Module constants


FRAMEWORK_XML = 'gui/toolbar.xml'
MERGEABLE_XML = [
    ("toolbar1_file", 'gui/toolbar-file.xml', _("File handling")),
    ("toolbar1_scrap", 'gui/toolbar-scrap.xml', _("Scraps switcher")),
    ("toolbar1_edit", 'gui/toolbar-edit.xml', _("Undo and Redo")),
    ("toolbar1_blendmodes", 'gui/toolbar-blendmodes.xml', _("Blend Modes")),
    ("toolbar1_linemodes", 'gui/toolbar-linemodes.xml', _("Line Modes")),
    ("toolbar1_view_modes", 'gui/toolbar-view-modes.xml', _("View (Main)")),
    ("toolbar1_view_manips", 'gui/toolbar-view-manips.xml', _("View (Alternative/Secondary)")),
    ("toolbar1_view_resets", 'gui/toolbar-view-resets.xml', _("View (Resetting)")),
    ("toolbar1_subwindows", 'gui/toolbar-subwindows.xml', _("Subwindows")),
    ]


## Class definitions

class ToolbarManager (object):
    """Manager for toolbars, currently just the main one.

    The main toolbar, /toolbar1, contains a menu button and quick
    access to the painting tools.
    """

    def __init__(self, draw_window):
        super(ToolbarManager, self).__init__()
        self.draw_window = draw_window
        self.app = draw_window.app
        self.toolbar1_ui_loaded = {}  # {name: merge_id, ...}
        self.init_actions()
        toolbarpath = os.path.join(self.app.datapath, FRAMEWORK_XML)
        self.app.ui_manager.add_ui_from_file(toolbarpath)
        self.toolbar1 = self.app.ui_manager.get_widget('/toolbar1')
        self.toolbar1.set_style(gtk.TOOLBAR_ICONS)
        self.toolbar1.set_icon_size(_get_icon_size())
        self.toolbar1.set_border_width(0)
        self.toolbar1.connect("popup-context-menu",
            self.on_toolbar1_popup_context_menu)
        self.toolbar1_popup = self.app.ui_manager\
            .get_widget('/toolbar1-settings-menu')
        for item in self.toolbar1:
            if isinstance(item, gtk.SeparatorToolItem):
                item.set_draw(False)
        styles = self.toolbar1.get_style_context()
        styles.add_class(gtk.STYLE_CLASS_PRIMARY_TOOLBAR)

        # Merge in UI pieces based on the user's saved preferences
        for action in self.settings_actions:
            name = action.get_property("name")
            active = self.app.preferences["ui.toolbar_items"].get(name, False)
            action.set_active(active)
            action.toggled()


    def init_actions(self):
        ag = self.draw_window.action_group
        actions = []

        self.settings_actions = []
        for name, ui_xml, label in MERGEABLE_XML:
            action = gtk.ToggleAction(name, label, None, None)
            action.connect("toggled", self.on_settings_toggle, ui_xml)
            self.settings_actions.append(action)
        actions += self.settings_actions

        for action in actions:
            ag.add_action(action)


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


def _get_icon_size():
    from application import get_app
    app = get_app()
    size = str(app.preferences.get("ui.toolbar_icon_size", "large")).lower()
    if size == 'small':
        return widgets.ICON_SIZE_SMALL
    else:
        return widgets.ICON_SIZE_LARGE


class ColorDropdownToolItem (gtk.ToolItem):
    """Toolbar colour indicator, history access, and changer"""

    __gtype_name__ = "MyPaintColorDropdownToolItem"

    def __init__(self):
        gtk.ToolItem.__init__(self)
        preview = ColorPreview()
        self.dropdown_button = dropdownpanel.DropdownPanelButton(preview)
        self.preview_size = _get_icon_size()
        self.connect("toolbar-reconfigured", self._toolbar_reconf_cb)
        self.connect("create-menu-proxy", lambda *a: True)
        self.set_tooltip_text(_("Color History and other tools"))
        self.add(self.dropdown_button)

        from application import get_app
        app = get_app()
        app.brush.observers.append(self._brush_settings_changed_cb)
        preview.color = app.brush_color_manager.get_color()

        self._app = app
        self._main_preview = preview

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
            action = app.find_action(action_name)
            assert action is not None, \
                    "Must be able to find action %s" % (action_name,)
            widget.set_related_action(action)
            widget.connect("clicked", hide_panel_cb)
            return widget

        button = init_proxy(gtk.Button(), "ColorDetailsDialog")
        side_vbox.pack_end(button, False, False)
        side_vbox.pack_end(preview_hbox, False, False)

        side_vbox.pack_end(gtk.Alignment(), True, True)
        button = init_proxy(gtk.ToggleButton(), "HCYWheelTool")
        button.set_label(_("HCY Wheel"))
        side_vbox.pack_end(button, False, False)
        button = init_proxy(gtk.ToggleButton(), "PaletteTool")
        button.set_label(_("Color Palette"))
        side_vbox.pack_end(button, False, False)

        # History
        section_frame = widgets.section_frame(_("Recently Used"))
        panel_vbox.pack_start(section_frame, True, True)

        history = ColorHistoryView(app)
        history.button_clicked += self._history_button_clicked
        section_frame.add(history)

    def _history_button_clicked(self, view):
        self.dropdown_button.panel_hide()

    def _toolbar_reconf_cb(self, toolitem):
        toolbar = self.get_parent()
        lookup_ret = gtk.icon_size_lookup(self.get_icon_size())
        lookup_succeeded, iw, ih = lookup_ret
        assert lookup_succeeded
        size = max(iw, ih)
        self._main_preview.set_size_request(size, size)

    def _brush_settings_changed_cb(self, changes):
        if not changes.intersection(set(['color_h', 'color_s', 'color_v'])):
            return
        mgr = self._app.brush_color_manager
        color = mgr.get_color()
        self._main_preview.set_color(color)



class BrushDropdownToolItem (gtk.ToolItem):
    """Toolbar brush indicator, history access, and changer.
    """

    __gtype_name__ = "MyPaintBrushDropdownToolItem"

    def __init__(self):
        gtk.ToolItem.__init__(self)
        self.history_images = []
        self.main_image = ManagedBrushPreview()
        self.dropdown_button = dropdownpanel.DropdownPanelButton(self.main_image)
        self.app = None
        self.image_size = _get_icon_size()
        self.connect("toolbar-reconfigured", self._toolbar_reconf_cb)
        self.connect("create-menu-proxy", lambda *a: True)
        self.set_tooltip_text(_("Brush history etc."))
        self.add(self.dropdown_button)

        from application import get_app
        app = get_app()
        self.app = app
        bm = self.app.brushmanager
        bm.brush_selected += self._brush_selected_cb

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

        quick_changer = dialogs.QuickBrushChooser(app, self._quick_change_select_cb)
        evbox = gtk.EventBox()
        evbox.add(quick_changer)
        section_vbox.pack_start(evbox, True, True)

        # List editor button
        # FIXME: perhaps list out the brush groups now?

        # Brush history
        section_frame = widgets.section_frame(_("Recently Used"))
        panel_vbox.pack_start(section_frame, True, True)

        history = BrushHistoryView(app)
        history.set_border_width(widgets.SPACING)
        history.button_clicked += self._history_button_clicked_cb
        section_frame.add(history)


    def _toolbar_reconf_cb(self, toolitem):
        toolbar = self.get_parent()
        lookup_ret = gtk.icon_size_lookup(self.get_icon_size())
        lookup_succeeded, iw, ih = lookup_ret
        assert lookup_succeeded
        self.image_size = max(iw, ih)
        self.main_image.set_size_request(iw, ih)

    def _brush_selected_cb(self, bm, brush, brushinfo):
        self.main_image.set_from_managed_brush(brush)

    def _history_button_clicked_cb(self, view):
        self.dropdown_button.panel_hide()

    def _quick_change_select_cb(self, brush):
        self.dropdown_button.panel_hide(immediate=False)
        self.app.brushmanager.select_brush(brush)


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
