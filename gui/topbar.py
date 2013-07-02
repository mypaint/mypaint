# This file is part of MyPaint.
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Combined menubar and toolbar."""


## Imports

import os
import math
from logging import getLogger
logger = getLogger(__name__)

from gi.repository import GObject
from gi.repository import Gtk
from gi.repository import Gdk
import cairo
from gettext import gettext as _

from workspace import Workspace

from random import sample, choice


## Class definitions

class TopBar (Gtk.VBox):
    """Combined menubar and toolbar which compacts when fullscreened.

    This is a container widget for a horizontal toolbar and a menubar with
    specialized behaviour when its parent window is fullscreened: the menubar
    is repacked into the toolbar, and temporary CSS styles are applied in order
    to attempt greater Fitts's Law compliance (and a nicer look).

    The toolbar and menubar are presented as properties for greater flexibility
    in construction.

    """

    ## Class constants

    __gtype_name__ = 'MyPaintTopBar'
    ICON_NAMES = {
        "FileMenu": "gtk-file",
        "EditMenu": "gtk-edit",
        "ViewMenu": "mypaint-view-zoom",
        "BrushMenu": "mypaint-tool-brush",
        "ColorMenu": "mypaint-tool-paint-color",
        "LayerMenu": "mypaint-tool-layers",
        "ScratchMenu": "mypaint-tool-scratchpad",
        "HelpMenu": "gtk-help",
        }


    ## GObject properties, for Builder-style construction

    #: The toolbar to present.
    toolbar = GObject.property(
            type=Gtk.Toolbar, flags=GObject.PARAM_READWRITE,
            nick='Toolbar widget',
            blurb="The GtkToolbar to show. This must be set at realize time.")

    #: The menubar to present.
    menubar = GObject.property(
            type=Gtk.MenuBar, flags=GObject.PARAM_READWRITE,
            nick='Menu Bar widget',
            blurb="The GtkMenuBar to show. This must be set at realize time.")


    ## Construction & initialization

    def __init__(self):
        Gtk.VBox.__init__(self)
        self.connect("realize", self._realize_cb)
        fs_menuitem = Gtk.MenuItem()
        fs_menuitem.set_label(_("MyPaint"))
        fs_menu = Gtk.Menu()
        fs_menuitem.set_submenu(fs_menu)
        self._fs_menuitem = fs_menuitem
        self._fs_menu = fs_menu


    def _realize_cb(self, widget):
        assert self.menubar is not None
        assert self.toolbar is not None

        # Specialized styles
        prov = Gtk.CssProvider()
        prov.load_from_data("""
                .topbar {
                    padding: 0px; /* required by toolbar */
                    margin: 0px;  /* required by menubar */
                    -GtkMenuBar-internal-padding: 0px;
                    -GtkToolBar-internal-padding: 0px;
                }
                .topbar .topbar {
                    background: rgba(0,0,0,0);
                    border-color: rgba(0,0,0,0);
                    border-width: 0px;
                    border-style: none;
                }
                .topitem {
                    border-color: rgba(0,0,0,0);
                    border-width: 0px;
                    border-style: solid;
                    margin: 0px;
                }
            """)

        bars = [self.toolbar, self.menubar]
        for b in bars:
            style = b.get_style_context()
            style.add_provider(prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
            style.add_class("topbar")

        # Initial packing; assume a non-fullscreened state
        self.pack_start(self.menubar, False, False, 0)
        self.pack_start(self.toolbar, False, False, 0)

        toplevel = self.get_toplevel()
        assert toplevel is not None
        toplevel.connect("window-state-event", self._toplevel_state_event_cb)



    ## Event handling


    def _toplevel_state_event_cb(self, toplevel, event):
        if not event.changed_mask & Gdk.WindowState.FULLSCREEN:
            return
        menubar = self.menubar
        toolbar = self.toolbar
        assert self is toolbar.get_parent()

        if event.new_window_state & Gdk.WindowState.FULLSCREEN:
            assert self is menubar.get_parent()
            menubar.hide()
            self.remove(menubar)
            ti = Gtk.ToolItem()
            for menuitem in list(menubar):
                menubar.remove(menuitem)
                self._fs_menu.append(menuitem)
                item_name = menuitem.get_name()
                icon_name = self.ICON_NAMES.get(item_name, None)
                if icon_name:
                    icon_image = Gtk.Image()
                    icon_image.set_from_icon_name(icon_name, Gtk.IconSize.MENU)
                    menuitem.set_image(icon_image)
                else:
                    logger.warning("No icon for %r in the fullscreen state",
                                   item_name)
                menuitem.show_all()
            menubar.append(self._fs_menuitem)
            ti.add(menubar)
            toolbar.get_style_context().add_class(Gtk.STYLE_CLASS_MENUBAR)
            toolbar.hide()
            self.remove(toolbar)
            self.pack_start(toolbar, True, True, 0)
            toolbar.insert(ti, 0)
            ti.show_all()
            self._fs_menuitem.show_all()
        else:
            assert self is not menubar.get_parent()
            menubar.remove(self._fs_menuitem)
            ti = menubar.get_parent()
            ti.remove(menubar)
            for menuitem in list(self._fs_menu):
                self._fs_menu.remove(menuitem)
                menubar.append(menuitem)
                menuitem.set_image(None)
            toolbar.remove(ti)
            del ti
            toolbar.get_style_context().remove_class(Gtk.STYLE_CLASS_MENUBAR)
            toolbar.hide()
            self.remove(toolbar)
            self.pack_start(menubar, False, False, 0)
            self.pack_start(toolbar, True, True, 0)
            menubar.show()
        toolbar.show_all()

