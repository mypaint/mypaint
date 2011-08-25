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


class MainToolbar (gtk.HBox):
    """The main 'toolbar': menu button and quick access to painting tools.
    """

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
        color_toolaction = ColorMenuToolAction("ColorMenuToolButton",
            None, _("Current Color"), None)
        ag.add_action(color_toolaction)

    def init_proxies(self):
        ag = self.draw_window.action_group
        for p in ag.get_action("ColorMenuToolButton").get_proxies():
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
        menu.set_reserve_toggle_size(False)
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


class ColorMenuToolAction (gtk.Action):
    """Allows `ColorMenuToolButton`s to be added by `gtk.UIManager`.
    """
    __gtype_name__ = "ColorMenuToolAction"

ColorMenuToolAction.set_tool_item_type(ColorMenuToolButton)


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
