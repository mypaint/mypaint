
"""Combined menubar and toolbar.
"""

import os
import math

from gi.repository import GObject
from gi.repository import Gtk
from gi.repository import Gdk
import cairo
from gettext import gettext as _

from workspace import Workspace

from random import sample, choice



class TopBar (Gtk.VBox):
    """Combined menubar and toolbar which compacts when fullscreened.

    This is a container widget for a horizontal toolbar and a menubar with
    specialized behaviour when its parent window is fullscreened: the menubar
    is repacked into the toolbar, and temporary CSS styles are applied in order
    to attempt greater Fitts's Law compliance (and a nicer look).

    The toolbar and menubar are presented as properties for greater flexibility
    in construction.

    """

    __gtype_name__ = 'MyPaintTopBar'

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


    def __init__(self):
        Gtk.VBox.__init__(self)
        self.connect("realize", self._realize_cb)


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

        #children = [m for m in self.menubar] + [t for t in self.toolbar]
        #for c in children:
        #    style = c.get_style_context()
        #    style.add_provider(prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        #    style.add_class("topitem")
        #    #print ">>>", style.get_path().to_string()

        # Initial packing; assume a non-fullscreened state
        self.pack_start(self.menubar, False, False, 0)
        self.pack_start(self.toolbar, False, False, 0)

        toplevel = self.get_toplevel()
        assert toplevel is not None
        toplevel.connect("window-state-event", self._toplevel_state_event_cb)


    def _toplevel_state_event_cb(self, toplevel, event):
        if not event.changed_mask & Gdk.WindowState.FULLSCREEN:
            return
        menubar = self.menubar
        toolbar = self.toolbar
        assert self is toolbar.get_parent()

        if event.new_window_state & Gdk.WindowState.FULLSCREEN:
            assert self is menubar.get_parent()
            self.remove(menubar)
            ti = Gtk.ToolItem()
            ti.add(menubar)
            toolbar.get_style_context().add_class(Gtk.STYLE_CLASS_MENUBAR)
            toolbar.hide()
            self.remove(toolbar)
            self.pack_start(toolbar, True, True, 0)
            toolbar.insert(ti, 0)
            ti.show_all()
        else:
            assert self is not menubar.get_parent()
            ti = menubar.get_parent()
            ti.remove(menubar)
            toolbar.remove(ti)
            del ti
            toolbar.get_style_context().remove_class(Gtk.STYLE_CLASS_MENUBAR)
            toolbar.hide()
            self.remove(toolbar)
            self.pack_start(menubar, False, False, 0)
            self.pack_start(toolbar, True, True, 0)
            menubar.show_all()
        toolbar.show_all()

