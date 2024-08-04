# This file is part of MyPaint.
# Copyright (C) 2013-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Combined menubar and toolbars."""


## Imports

from __future__ import division, print_function
import logging

from lib.gibindings import GObject
from lib.gibindings import Gtk
from lib.gibindings import Gdk
from gettext import gettext as _

logger = logging.getLogger(__name__)


## Class definitions

class TopBar (Gtk.Grid):
    """Combined menubar and toolbars which compacts when fullscreened.

    This is a container widget for two horizontal toolbars and a menubar
    with specialized behaviour when its parent window is fullscreened:
    the menubar is repacked into the toolbar, and temporary CSS styles
    are applied in order to attempt greater Fitts's Law compliance (and
    a nicer look).

    The toolbars and menubar are presented as properties for greater
    flexibility in construction. All of these properties must be set up
    at the time the widget is realized.

    """

    ## Class constants

    __gtype_name__ = 'MyPaintTopBar'
    ICON_NAMES = {
        "FileMenu": "mypaint-file-symbolic",
        "EditMenu": "mypaint-edit-symbolic",
        "ViewMenu": "mypaint-view-symbolic",
        "BrushMenu": "mypaint-brush-symbolic",
        "ColorMenu": "mypaint-colors-symbolic",
        "LayerMenu": "mypaint-layers-symbolic",
        "ScratchMenu": "mypaint-scratchpad-symbolic",
        "HelpMenu": "mypaint-help-symbolic",
    }

    ## GObject properties, for Builder-style construction

    #: The toolbar to present in position 1.
    toolbar1 = GObject.Property(
        type=Gtk.Toolbar,
        flags=GObject.ParamFlags.READABLE | GObject.ParamFlags.WRITABLE,
        nick='Toolbar-1 widget',
        blurb="First GtkToolbar to show. This must be set at realize time."
    )

    #: The toolbar to present in position 2.
    toolbar1 = GObject.Property(
        type=Gtk.Toolbar,
        flags=GObject.ParamFlags.READABLE | GObject.ParamFlags.WRITABLE,
        nick='Toolbar-2 widget',
        blurb="Second GtkToolbar to show. This must be set at realize time."
    )

    #: The menubar to present.
    menubar = GObject.Property(
        type=Gtk.MenuBar,
        flags=GObject.ParamFlags.READABLE | GObject.ParamFlags.WRITABLE,
        nick='Menu Bar widget',
        blurb="The GtkMenuBar to show. This must be set at realize time."
    )

    ## Construction & initialization

    def __init__(self):
        """Initialize"""
        Gtk.Grid.__init__(self)
        self.connect("realize", self._realize_cb)
        # Widgets used in fullscreen mode
        fs_menu = Gtk.Menu()
        self._fs_menu = fs_menu
        self._fs_menubutton = FakeMenuButton(_("<b>MyPaint</b>"), fs_menu)
        self._fs_toolitem = Gtk.ToolItem()

    def _realize_cb(self, widget):
        """Assorted setup when the widget is realized"""
        assert self.menubar is not None
        assert self.toolbar1 is not None
        assert self.toolbar2 is not None
        # Packing details for Grid
        self.menubar.set_hexpand(True)
        self.toolbar1.set_hexpand(True)
        self.toolbar2.set_hexpand(False)
        self._fs_menubutton.set_hexpand(False)
        # Specialized styles
        prov = Gtk.CssProvider()
        prov.load_from_data(b"""
                .topbar {
                    padding: 0px; /* required by toolbars */
                    margin: 0px;  /* required by menubar */
                }
            """)
        bars = [self.toolbar1, self.toolbar2, self.menubar]
        for b in bars:
            style = b.get_style_context()
            style.add_provider(prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
            style.add_class("topbar")
        # Initial packing; assume a non-fullscreened state
        self.attach(self.menubar, 0, 0, 2, 1)
        self.attach(self.toolbar1, 0, 1, 1, 1)
        self.attach(self.toolbar2, 1, 1, 1, 1)
        # Track state transitions of the window's toplevel
        toplevel = self.get_toplevel()
        assert toplevel is not None
        toplevel.connect("window-state-event", self._toplevel_state_event_cb)

    ## Event handling

    def _toplevel_state_event_cb(self, toplevel, event):
        """Repacks widgets when the toplevel changes fullsceen state"""
        if not event.changed_mask & Gdk.WindowState.FULLSCREEN:
            return
        menubar = self.menubar
        toolbar1 = self.toolbar1
        toolbar2 = self.toolbar2
        assert self is toolbar1.get_parent()
        assert self is toolbar2.get_parent()
        if event.new_window_state & Gdk.WindowState.FULLSCREEN:
            # Remove menubar, use menu button on the 1st toolbar instead
            assert menubar.get_parent() is self
            assert self._fs_menubutton.get_parent() is None
            menubar.hide()
            self.remove(menubar)
            for menuitem in list(menubar):
                menubar.remove(menuitem)
                self._fs_menu.append(menuitem)
                item_name = menuitem.get_name()
                icon_name = self.ICON_NAMES.get(item_name, None)
                if hasattr(menuitem, "set_image"):
                    if icon_name:
                        icon_image = Gtk.Image()
                        icon_image.set_from_icon_name(
                            icon_name,
                            Gtk.IconSize.MENU,
                        )
                        menuitem.set_image(icon_image)
                    else:
                        logger.warning(
                            "No icon for %r in the fullscreen state",
                            item_name,
                        )
                menuitem.show_all()
            toolbar1.hide()
            self.remove(toolbar1)
            self.remove(toolbar2)
            self.attach(toolbar1, 0, 0, 1, 1)
            self.attach(toolbar2, 1, 0, 1, 1)
            toolbar1.insert(self._fs_toolitem, 0)
            self._fs_toolitem.add(self._fs_menubutton)
            self._fs_toolitem.show_all()
        else:
            # Windowed mode: use a regular menu bar above the toolbar
            assert menubar.get_parent() is None
            assert self._fs_menubutton.get_parent() is self._fs_toolitem
            toolbar1.remove(self._fs_toolitem)
            self._fs_toolitem.remove(self._fs_menubutton)
            for menuitem in list(self._fs_menu):
                self._fs_menu.remove(menuitem)
                menubar.append(menuitem)
                if hasattr(menuitem, "set_image"):
                    menuitem.set_image(None)
            toolbar1.hide()
            toolbar2.hide()
            self.remove(toolbar1)
            self.remove(toolbar2)
            self.attach(menubar, 0, 0, 2, 1)
            self.attach(toolbar1, 0, 1, 1, 1)
            self.attach(toolbar2, 1, 1, 1, 1)
            menubar.show()
        toolbar1.show_all()
        toolbar2.show_all()


class FakeMenuButton (Gtk.EventBox):
    """Button-styled widget that launches a dropdown menu when clicked"""

    def __init__(self, markup, menu):
        """Initialize

        :param markup: Markup to display in the button.
        :param menu: The menu to present when clicked.
        """
        Gtk.EventBox.__init__(self)
        self.menu = menu
        self.label = Gtk.Label()
        self.label.set_markup(markup)
        self.label.set_padding(8, 0)
        # Intercept mouse clicks and use them for activating the togglebutton
        # even if they're in its border, or (0, 0). Fitts would approve.
        invis = Gtk.EventBox()
        invis.set_visible_window(False)
        invis.set_above_child(True)
        invis.connect("button-press-event", self._button_press_cb)
        invis.connect("enter-notify-event", self._enter_cb)
        invis.connect("leave-notify-event", self._leave_cb)
        self.invis_window = invis
        # Toggle button, for the look of the thing only
        self.togglebutton = Gtk.ToggleButton()
        self.togglebutton.add(self.label)
        self.togglebutton.set_relief(Gtk.ReliefStyle.NONE)
        self.togglebutton.connect("toggled", self._togglebutton_toggled_cb)
        # The underlying togglebutton can default and focus. Might as well make
        # the Return key do something useful rather than invoking the 1st
        # toolbar item.
        self.togglebutton.set_can_default(True)
        self.togglebutton.set_can_focus(True)
        # Packing
        invis.add(self.togglebutton)
        self.add(invis)
        # Menu signals
        for sig in "selection-done", "deactivate", "cancel":
            menu.connect(sig, self._menu_dismiss_cb)

    def _enter_cb(self, widget, event):
        """Prelight the button when hovered"""
        self.togglebutton.set_state_flags(Gtk.StateFlags.PRELIGHT, False)

    def _leave_cb(self, widget, event):
        """Un-prelight the button when the pointer leaves"""
        self.togglebutton.unset_state_flags(Gtk.StateFlags.PRELIGHT)

    def _button_press_cb(self, widget, event):
        """Post the menmu when clicked

        Menu operation is much more convincing if we call popup() with event
        details here rather than leaving it to the child button's "toggled"
        event handler.
        """
        pos_func = self._get_popup_menu_position
        self.menu.popup(parent_menu_shell=None, parent_menu_item=None,
                        func=pos_func, data=None, button=event.button,
                        activate_time=event.time)
        self.togglebutton.set_active(True)

    def _togglebutton_toggled_cb(self, togglebutton):
        """Post the menu from a keypress activating the toggle

        The menu dismiss handler untoggles it."""
        if togglebutton.get_active():
            if not self.menu.get_property("visible"):
                pos_func = self._get_popup_menu_position
                self.menu.popup(
                    parent_menu_shell=None,
                    parent_menu_item=None,
                    func=pos_func,
                    data=None,
                    button=1,
                    activate_time=Gdk.CURRENT_TIME,
                )

    def _menu_dismiss_cb(self, *a, **kw):
        """Reset the button state when the user's finished

        Also transfer focus back to the menu button."""
        self.unset_state_flags(Gtk.StateFlags.PRELIGHT)
        self.togglebutton.set_active(False)
        self.togglebutton.grab_focus()

    def _get_popup_menu_position(self, menu, *junk):
        """Position function for menu popup

        This places the menu underneath the button, at the same x position.
        """
        win = self.get_window()
        x, y = win.get_origin()[1:]
        y += self.get_allocated_height()
        return x, y, True


## Testing

def _test():
    """Run an interactive test"""
    toplevel = Gtk.Window()
    toplevel.set_title("topbar test")
    toplevel.connect("destroy", lambda *a: Gtk.main_quit())
    mainbox = Gtk.VBox()
    topbar = TopBar()
    canvas = Gtk.DrawingArea()
    toplevel.set_size_request(500, 300)

    # Fullscreen action
    fs_act = Gtk.ToggleAction.new("Fullscreen", "Fullscreen",
                                  "Enter fullscreen mode",
                                  Gtk.STOCK_FULLSCREEN)

    def _fullscreen_cb(action, toplevel):
        if action.get_active():
            toplevel.fullscreen()
        else:
            toplevel.unfullscreen()
    fs_act.connect("toggled", _fullscreen_cb, toplevel)

    # One normally constructed menubar
    menu1 = Gtk.Menu()
    menuitem1 = Gtk.MenuItem.new_with_label("Demo")
    menuitem2 = Gtk.CheckMenuItem()
    menuitem3 = Gtk.MenuItem.new_with_label("Quit")
    menuitem1.set_submenu(menu1)
    menuitem2.set_related_action(fs_act)
    menuitem3.connect("activate", lambda *a: Gtk.main_quit())
    menu1.append(menuitem2)
    menu1.append(menuitem3)
    menuitem1.show()
    menuitem2.show()
    menuitem3.show()
    menubar = Gtk.MenuBar()
    menubar.append(menuitem1)

    # We need a pair of toolbars too.
    toolbar1 = Gtk.Toolbar()
    toolbar2 = Gtk.Toolbar()
    toolitem1 = Gtk.ToggleToolButton()
    toolitem1.set_related_action(fs_act)
    toolbar2.insert(toolitem1, -1)
    toolitem2 = Gtk.SeparatorToolItem()
    toolitem2.set_draw(False)
    toolitem2.set_expand(True)
    toolbar1.insert(toolitem2, 0)
    # Some junk items, to verify appearance in various GTK3 themes
    toolitem3 = Gtk.ToolButton.new_from_stock(Gtk.STOCK_ZOOM_100)
    toolitem3.set_is_important(True)
    toolitem4 = Gtk.ToolButton.new_from_stock(Gtk.STOCK_ZOOM_100)
    toolitem4.set_is_important(True)
    toolitem4.set_sensitive(False)
    toolitem5 = Gtk.ToolButton.new_from_stock(Gtk.STOCK_ZOOM_100)
    toolitem6 = Gtk.ToolButton.new_from_stock(Gtk.STOCK_ZOOM_100)
    toolitem6.set_sensitive(False)
    toolbar1.insert(toolitem6, 0)
    toolbar1.insert(toolitem5, 0)
    toolbar1.insert(toolitem4, 0)
    toolbar2.insert(toolitem3, 0)

    # Assign topbar's properties
    topbar.toolbar1 = toolbar1
    topbar.toolbar2 = toolbar2
    topbar.menubar = menubar

    # Pack main UI, and start demo
    mainbox.pack_start(topbar, False, False, 0)
    mainbox.pack_start(canvas, True, True, 0)
    toplevel.add(mainbox)
    toplevel.show_all()
    topbar.show_all()
    Gtk.main()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _test()
