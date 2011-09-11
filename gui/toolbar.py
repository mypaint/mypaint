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


class MainToolbar (gtk.HBox):
    """The main 'toolbar': menu button and quick access to painting tools.
    """

    icon_size = gtk.icon_size_register("MYPAINT_TOOLBAR_ICON_SIZE", 32, 32)

    def __init__(self, draw_window):
        gtk.HBox.__init__(self)
        self.draw_window = draw_window
        self.app = draw_window.app
        self.init_actions()
        toolbarpath = os.path.join(self.app.datapath, 'gui/toolbar.xml')
        toolbarbar_xml = open(toolbarpath).read()
        self.app.ui_manager.add_ui_from_string(toolbarbar_xml)
        self.toolbar1 = self.app.ui_manager.get_widget('/toolbar1')
        self.toolbar1.set_style(gtk.TOOLBAR_ICONS)
        self.toolbar1.set_icon_size(self.icon_size)
        self.toolbar1.set_border_width(0)
        self.toolbar1.connect("style-set", self.on_toolbar1_style_set)
        self.menu_button = FakeMenuButton(_("MyPaint"), draw_window.popupmenu)
        self.menu_button.set_border_width(0)
        self.pack_start(self.menu_button, False, False)
        self.pack_start(self.toolbar1, True, True)
        self.menu_button.set_flags(gtk.CAN_DEFAULT)
        draw_window.set_default(self.menu_button)
        self.init_proxies()

    def init_actions(self):
        ag = self.draw_window.action_group
        self.actions = [
                ColorMenuToolAction("ColorMenuToolButton", None,
                    _("Current Color"), None),
                BrushMenuToolAction("BrushMenuToolButton", None,
                    _("Current Brush"), None),
                BrushSettingsDropdownToolAction("BrushSettingsDropdown", None,
                    _("Brush Settings"), None),
                ]
        for toolaction in self.actions:
            ag.add_action(toolaction)

    def init_proxies(self):
        for action in self.actions:
            for p in action.get_proxies():
                p.set_app(self.app)

    def on_toolbar1_style_set(self, widget, oldstyle):
        style = widget.style.copy()
        self.menu_button.set_style(style)
        style = widget.style.copy()
        self.set_style(style)


class ColorMenuToolButton (gtk.MenuToolButton):
    """Toolbar colour indicator, history access, and changer.

    The button part shows the current colour, and allows it to be changed
    in detail when clicked. The menu contains the colour history, and
    a selection of other ways of changing the colour.
    """

    __gtype_name__ = "ColorMenuToolButton"

    def __init__(self, *a, **kw):
        self.main_blob = ColorBlob()
        gtk.MenuToolButton.__init__(self, self.main_blob, None)
        self.app = None
        self.connect("toolbar-reconfigured", self.on_toolbar_reconf)
        self.connect("show-menu", self.on_show_menu)
        self.menu_blobs = []
        menu = gtk.Menu()
        self.set_menu(menu)
        self.blob_size = 1
        self.connect("clicked", self.on_clicked)
        self.connect("create-menu-proxy", self.on_create_menu_proxy)
        self.set_arrow_tooltip_text(_("Color History and other tools"))

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

    def set_app(self, app):
        self.app = app
        self.app.brush.observers.append(self.on_brush_settings_changed)
        self.main_blob.hsv = self.app.brush.get_color_hsv()

    def on_brush_settings_changed(self, changes):
        if not changes.intersection(set(['color_h', 'color_s', 'color_v'])):
            return
        self.main_blob.hsv = self.app.brush.get_color_hsv()

    def on_show_menu(self, menutoolbutton):
        if self.app is None:
            return
        init = not self.menu_blobs
        s = self.blob_size
        menu = self.get_menu()
        for i, hsv in enumerate(self.app.ch.colors):
            if init:
                blob = ColorBlob(hsv)
                self.menu_blobs.append(blob)
                blob_menuitem = gtk.MenuItem()
                blob_menuitem.add(blob)
                menu.prepend(blob_menuitem)
                blob_menuitem.show_all()
                blob_menuitem.connect("activate", self.on_menuitem_activate, i)
            else:
                blob = self.menu_blobs[i]
            blob.hsv = hsv
            blob.set_size_request(s, s)
        if init:
            for name in ["ColorRingPopup", "ColorChangerPopup",
                         "ColorSelectionWindow"]:
                action = self.app.drawWindow.action_group.get_action(name)
                item = action.create_menu_item()
                menu.append(item)

    def on_menuitem_activate(self, menuitem, i):
        hsv = self.app.ch.colors[i]
        self.app.brush.set_color_hsv(hsv)

    def on_clicked(self, toolbutton):
        dialogs.change_current_color_detailed(self.app)


class BrushMenuToolButton (gtk.MenuToolButton):
    """Toolbar brush indicator, history access, and changer.
    """

    __gtype_name__ = "BrushMenuToolButton"

    def __init__(self, *a, **kw):
        self.main_image = ManagedBrushPreview()
        gtk.MenuToolButton.__init__(self, self.main_image, None)
        self.app = None
        self.image_size = 1
        self.connect("toolbar-reconfigured", self.on_toolbar_reconf)
        self.connect("show-menu", self.on_show_menu)
        self.menu_images = []
        self.menu_labels = []
        menu = gtk.Menu()
        self.set_menu(menu)
        self.connect("clicked", self.on_clicked)
        self.connect("create-menu-proxy", self.on_create_menu_proxy)
        self.set_arrow_tooltip_text(_("Brush history etc."))

    def on_create_menu_proxy(self, toolitem):
        self.set_proxy_menu_item("", None)
        return True

    def on_toolbar_reconf(self, toolitem):
        toolbar = self.parent
        iw, ih = gtk.icon_size_lookup(self.get_icon_size())
        self.image_size = max(iw, ih)
        self.main_image.set_size_request(iw, ih)

    def set_app(self, app):
        self.app = app
        bm = self.app.brushmanager
        bm.selected_brush_observers.append(self.on_selected_brush)

    def on_selected_brush(self, brush, brushinfo):
        self.main_image.set_from_managed_brush(brush)

    def on_show_menu(self, menutoolbutton):
        if self.app is None:
            return
        bm = self.app.brushmanager
        init = not self.menu_images
        s = self.image_size
        menu = self.get_menu()
        for i, brush in enumerate(bm.history):
            name = brush.brushinfo.get_string_property("parent_brush_name")
            if init:
                hbox = gtk.HBox()
                label = gtk.Label()
                label.set_alignment(0, 0.5)
                label.set_padding(8, 0)
                image = ManagedBrushPreview()
                self.menu_images.append(image)
                self.menu_labels.append(label)
                hbox.pack_start(image, False, False)
                hbox.pack_start(label, True, True)
                menuitem = gtk.MenuItem()
                menuitem.add(hbox)
                menu.prepend(menuitem)
                menuitem.show_all()
                menuitem.connect("activate", self.on_menuitem_activate, i)
            else:
                image = self.menu_images[i]
                label = self.menu_labels[i]
            image.set_from_managed_brush(brush)
            image.set_size_request(s, s)
            label.set_text(name)
        if init:
            ag = self.app.drawWindow.action_group
            brush_window_action = ag.get_action("BrushSelectionWindow")
            item = brush_window_action.create_menu_item()
            menu.append(item)
            sep = gtk.SeparatorMenuItem()
            menu.append(sep)
            sep.show()
            action_names = ["BlendModeNormal", "BlendModeEraser",
                            "BlendModeLockAlpha"]
            ag = self.app.brushmodifier.action_group
            for action_name in action_names:
                action = ag.get_action(action_name)
                item = action.create_menu_item()
                menu.append(item)



    def on_menuitem_activate(self, menuitem, i):
        bm = self.app.brushmanager
        brush = bm.history[i]
        bm.select_brush(brush)

    def on_clicked(self, toolbutton):
        dialogs.change_current_brush_quick(self.app)



class BrushSettingsDropdownToolItem (gtk.ToolItem):
    __gtype_name__ = "BrushSettingsDropdownToolItem"

    setting_cnames = ["radius_logarithmic", "slow_tracking", "opaque", "hardness"]

    def __init__(self):
        gtk.ToolItem.__init__(self)
        self.set_homogeneous(False)
        self.button_image = gtk.Image()
        self.button_image.set_from_stock(stock.BRUSH_MODIFIERS_INACTIVE,
                                         MainToolbar.icon_size)
        self.button_shows_modified = False
        self.button = dropdownpanel.DropdownPanelButton(self.button_image)
        self.button.set_name(widgets.BORDERLESS_BUTTON_NAME)
        self.button.set_relief(gtk.RELIEF_NONE)
        self.button.set_can_default(False)
        self.button.set_can_focus(False)
        self.vbox = gtk.VBox()
        frame = gtk.Frame()
        frame.add(self.vbox)
        frame.set_shadow_type(gtk.SHADOW_OUT)
        self.button.set_panel_widget(frame)
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

            reset_button = widgets.borderless_image_button(
                stock_id=gtk.STOCK_CLEAR,
                tooltip=_("Reset '%s'") % s.name)
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

        widget = gtk.Button(_("Reset All"))
        widget.connect("clicked", self.reset_all_clicked_cb)
        widget.set_tooltip_text(_("Reset the brush's settings"))
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

        for a in ["BlendModeNormal", "BlendModeEraser", "BlendModeLockAlpha"]:
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
            self.button_image.set_from_stock(stock_id, MainToolbar.icon_size)


    def _current_brush_is_modified(self):
        current_bi = self.app.brush
        parent_b = self.app.brushmanager.get_parent_brush(brushinfo=current_bi)
        if parent_b is None:
            return True
        return not parent_b.brushinfo.matches(current_bi)


class ColorMenuToolAction (gtk.Action):
    """Allows `ColorMenuToolButton`s to be added by `gtk.UIManager`.
    """
    __gtype_name__ = "ColorMenuToolAction"

ColorMenuToolAction.set_tool_item_type(ColorMenuToolButton)


class BrushMenuToolAction (gtk.Action):
    """Allows `BrushMenuToolButton`s to be added by `gtk.UIManager`.
    """
    __gtype_name__ = "BrushMenuToolAction"

BrushMenuToolAction.set_tool_item_type(BrushMenuToolButton)



class BrushSettingsDropdownToolAction (gtk.Action):
    __gtype_name__ = "BrushSettingsDropdownToolAction"
BrushSettingsDropdownToolAction\
        .set_tool_item_type(BrushSettingsDropdownToolItem)



class ManagedBrushPreview (gtk.Image):
    """Updateable widget displaying a brushmanager.ManagedBrush`'s preview.
    """

    def __init__(self, brush=None):
        gtk.Image.__init__(self)
        self.pixbuf = None
        self.image_size = None
        self.set_from_managed_brush(brush)
        self.set_size_request(32, 32)
        self.connect("size-allocate", self.on_size_allocate)

    def set_from_managed_brush(self, brush):
        if brush is None:
            return
        if not brush.preview:
            brush.load_preview()
        self.pixbuf = brush.preview.copy()
        self._update()

    def on_size_allocate(self, widget, alloc):
        new_size = alloc.width, alloc.height
        if new_size != self.image_size:
            self.image_size = alloc.width, alloc.height
            self._update()

    def _update(self):
        if not (self.pixbuf and self.image_size):
            return
        w, h = self.image_size
        s = min(w, h)
        scaled_pixbuf = self.pixbuf.scale_simple(s, s, gdk.INTERP_BILINEAR)
        self.set_from_pixbuf(scaled_pixbuf)


class ColorBlob (gtk.DrawingArea):
    """Updatable widget displaying a single colour.
    """

    def __init__(self, hsv=None):
        gtk.DrawingArea.__init__(self)
        if hsv is None:
            hsv = 0.0, 0.0, 0.0
        self._hsv = hsv
        self.set_size_request(1, 1)
        self.connect("expose-event", self.on_expose)

    def set_hsv(self, hsv):
        self._hsv = hsv
        self.queue_draw()

    def get_hsv(self):
        return self._hsv

    hsv = property(get_hsv, set_hsv)

    def on_expose(self, widget, event):
        cr = self.window.cairo_create()
        cr.set_source_rgb(*hsv_to_rgb(*self._hsv))
        cr.paint()


class FakeMenuButton(gtk.EventBox):
    """Launches the popup menu when clicked.

    One of these sits to the left of the real toolbar when the main menu bar is
    hidden. In addition to providing access to a popup menu associated with the
    main view, this is a little more compliant with Fitts's Law than a normal
    `gtk.MenuBar`: when the window is fullscreened with only the "toolbar"
    present the ``(0, 0)`` screen pixel hits this button. Support note: Compiz
    edge bindings sometimes get in the way of this, so turn those off if you
    want Fitts's compliance.
    """

    def __init__(self, text, menu):
        gtk.EventBox.__init__(self)
        self.menu = menu
        self.label = gtk.Label(text)
        self.label.set_padding(8, 0)

        # Text settings
        #self.label.set_angle(5)
        attrs = pango.AttrList()
        attrs.change(pango.AttrWeight(pango.WEIGHT_HEAVY, 0, -1))
        self.label.set_attributes(attrs)

        # Intercept mouse clicks and use them for activating the togglebutton
        # even if they're in its border, or (0, 0). Fitts would approve.
        invis = self.invis_window = gtk.EventBox()
        invis.set_visible_window(False)
        invis.set_above_child(True)
        invis.connect("button-press-event", self.on_button_press)
        invis.connect("enter-notify-event", self.on_enter)
        invis.connect("leave-notify-event", self.on_leave)

        # The underlying togglebutton can default and focus. Might as well make
        # the Return key do something useful rather than invoking the 1st
        # toolbar item.
        self.togglebutton = gtk.ToggleButton()
        self.togglebutton.add(self.label)
        self.togglebutton.set_relief(gtk.RELIEF_HALF)
        self.togglebutton.set_flags(gtk.CAN_FOCUS)
        self.togglebutton.set_flags(gtk.CAN_DEFAULT)
        self.togglebutton.connect("toggled", self.on_togglebutton_toggled)

        invis.add(self.togglebutton)
        self.add(invis)
        for sig in "selection-done", "deactivate", "cancel":
            menu.connect(sig, self.on_menu_dismiss)


    def on_enter(self, widget, event):
        # Not this set_state(). That one.
        #self.togglebutton.set_state(gtk.STATE_PRELIGHT)
        gtk.Widget.set_state(self.togglebutton, gtk.STATE_PRELIGHT)


    def on_leave(self, widget, event):
        #self.togglebutton.set_state(gtk.STATE_NORMAL)
        gtk.Widget.set_state(self.togglebutton, gtk.STATE_NORMAL)


    def on_button_press(self, widget, event):
        # Post the menu. Menu operation is much more convincing if we call
        # popup() with event details here rather than leaving it to the toggled
        # handler.
        pos_func = self._get_popup_menu_position
        self.menu.popup(None, None, pos_func, event.button, event.time)
        self.togglebutton.set_active(True)


    def on_togglebutton_toggled(self, togglebutton):
        # Post the menu from a keypress. Dismiss handler untoggles it.
        if togglebutton.get_active():
            if not self.menu.get_property("visible"):
                pos_func = self._get_popup_menu_position
                self.menu.popup(None, None, pos_func, 1, 0)


    def on_menu_dismiss(self, *a, **kw):
        # Reset the button state when the user's finished, and
        # park focus back on the menu button.
        self.set_state(gtk.STATE_NORMAL)
        self.togglebutton.set_active(False)
        self.togglebutton.grab_focus()


    def _get_popup_menu_position(self, menu, *junk):
        # Underneath the button, at the same x position.
        x, y = self.window.get_origin()
        y += self.allocation.height
        return x, y, True


    def set_style(self, style):
        # Propagate style changes to all children as well. Since this button is
        # stored on the toolbar, the main window makes it share a style with
        # it. Looks prettier.
        gtk.EventBox.set_style(self, style)
        style = style.copy()
        widget = self.togglebutton
        widget.set_style(style)
        style = style.copy()
        widget = widget.get_child()
        widget.set_style(style)
