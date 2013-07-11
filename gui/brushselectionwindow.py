# This file is part of MyPaint.
# Copyright (C) 2007-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Brush selection/organizer windows

These can be used for selecting brushes, and can be docked into the sidebar.
They are responsible for ordering, loading and saving brush lists.

"""

## Imports

import platform
import logging
logger = logging.getLogger(__name__)

import gtk2compat

from gettext import gettext as _
from gettext import ngettext
if gtk2compat.USE_GTK3:
    import gi
    from gi.repository import PangoCairo
import pango
import gtk
from gtk import gdk

import pixbuflist
import dialogs
import brushmanager
from brushlib import brushsettings
from lib.helpers import escape
from colors import RGBColor
from workspace import SizedVBoxToolWidget
import widgets


## FIXME: unused widgets
# FIXME: move "common settings" somewhere, perhaps a dockable panel


def get_common_settings_widget(app):
    """Return a widget with controls for manipulating common settings"""

    cmn = ['radius_logarithmic', 'opaque', 'hardness']
    common_settings = [s for s in brushsettings.settings_visible if s.cname in cmn]
    settings_box = gtk.VBox()

    def value_changed_cb(adj, cname, app):
        app.brush.set_base_value(cname, adj.get_value())

    def get_setting_widget(setting):
        """Return a widget to control a single setting"""
        adj = app.brush_adjustment[s.cname]
        adj.connect('value-changed', value_changed_cb, s.cname, app)

        l = gtk.Label(s.name)
        l.set_alignment(0, 0.5)

        h = gtk.HScale(adj)
        h.set_digits(2)
        h.set_draw_value(True)
        h.set_value_pos(gtk.POS_LEFT)

        box = gtk.HBox()
        box.pack_start(l)
        box.pack_start(h)
        return box

    for s in common_settings:
        settings_box.pack_start(get_setting_widget(s))

    return settings_box


## Class definitions


class BrushList (pixbuflist.PixbufList):
    """Flowed grid of brush icons showing a group, click to set the brush"""


    ICON_SIZE = 48
    MIN_WIDTH_NICONS = 1
    NATURAL_WIDTH_NICONS = 4
    MIN_HEIGHT_NICONS = 1

    def __init__(self, app, group):
        self.app = app
        self.bm = app.brushmanager
        self.brushes = self.bm.groups[group]
        self.group = group
        s = self.ICON_SIZE
        pixbuflist.PixbufList.__init__(self, self.brushes, s, s,
                                       namefunc = lambda x: x.name,
                                       pixbuffunc = lambda x: x.preview)
        # Support device changing with the same event as that used
        # for brush choice:
        if not gtk2compat.USE_GTK3:
            self.set_extension_events(gdk.EXTENSION_EVENTS_ALL)

        self.set_selected(self.bm.selected_brush)
        self.bm.brushes_changed += self.brushes_modified_cb
        self.bm.brush_selected += self.brush_selected_cb


    def do_get_request_mode(self):
        return gtk.SizeRequestMode.HEIGHT_FOR_WIDTH


    def do_get_preferred_width(self):
        return (self.MIN_WIDTH_NICONS * self.ICON_SIZE,
                self.NATURAL_WIDTH_NICONS * self.ICON_SIZE)


    def do_get_preferred_height_for_width(self, width):
        icons_wide = max(1, int(width / self.ICON_SIZE))
        num_brushes = len(self.brushes)
        icons_tall = max(int(num_brushes / icons_wide),
                         max(self.MIN_HEIGHT_NICONS, 1))
        if icons_tall * icons_wide  < num_brushes:
            icons_tall += 1
        return (icons_tall * self.ICON_SIZE,
                icons_tall * self.ICON_SIZE)


    def brushes_modified_cb(self, bm, brushes):
        if brushes is self.brushes:
            self.update()

    def brush_selected_cb(self, bm, managed_brush, brushinfo):
        self.set_selected(managed_brush)

    def remove_brush(self, brush):
        self.brushes.remove(brush)
        self.bm.brushes_changed(self.brushes)

    def insert_brush(self, idx, brush):
        self.brushes.insert(idx, brush)
        self.bm.brushes_changed(self.brushes)

    def button_press_cb(self, widget, event):
        if gtk2compat.USE_GTK3:
            device = event.get_source_device()
        else:
            device = event.device
        self.app.device_monitor.device_used(device)
        return pixbuflist.PixbufList.button_press_cb(self, widget, event)

    def drag_begin_cb(self, widget, context):
        preview = self.bm.selected_brush.preview
        preview = preview.scale_simple(preview.get_width()//2, preview.get_height()//2, gtk.gdk.INTERP_BILINEAR)
        self.drag_source_set_icon_pixbuf(preview)
        pixbuflist.PixbufList.drag_begin_cb(self, widget, context)

    #def drag_end_cb(self, widget, context):
    #    pixbuflist.PixbufList.drag_end_cb(self, widget, context)

    def on_drag_data(self, copy, source_widget, brush_name, target_idx):
        assert source_widget, 'cannot handle drag data from another app'
        brush = None
        for b in source_widget.brushes:
            if b.name == brush_name:
                brush = b
                break
        if brush is None:
            logger.error("No brush named %r in drag source widget", brush_name)
            return False
        if source_widget is self:
            copy = False
        else:
            if brush in self.brushes:
                source_widget.remove_brush(brush)
                self.remove_brush(brush)
                self.insert_brush(target_idx, brush)
                return True
        if not copy:
            source_widget.remove_brush(brush)
        self.insert_brush(target_idx, brush)
        return True

    def on_select(self, brush):
        # brush changed on harddisk?
        if brush.reload_if_changed():
            for brushes in self.bm.groups.itervalues():
                self.bm.brushes_changed(brushes)
        self.bm.select_brush(brush)


class BrushGroupTool (SizedVBoxToolWidget):
    """Dockable tool widget showing just one BrushGroup"""

    __gtype_name__ = "MyPaintBrushGroupTool"


    ## Construction and updating

    def __init__(self, group):
        """Construct, to show a named group"""
        SizedVBoxToolWidget.__init__(self)
        self._group = group
        self._scrolls = gtk.ScrolledWindow()
        self._dialog = None
        self._brush_list = None
        from application import get_app
        self._app = get_app()
        if group not in self._app.brushmanager.groups:
            raise ValueError, "No group named %r" % (group,)
        self.pack_start(self._scrolls)
        self._update_brush_list()


    def _update_brush_list(self):
        """Updates the brush list to match the group name"""
        if self._brush_list:
            self._brush_list.destroy()
        viewport = self._scrolls.get_child()
        if viewport:
            self._scrolls.remove(viewport)
            viewport.destroy()
        self._brush_list = BrushList(self._app, self._group)
        self._scrolls.add_with_viewport(self._brush_list)
        self._brush_list.show_all()


    ## Tool widget properties and methods


    @property
    def tool_widget_title(self):
        return brushmanager.translate_group_name(self._group)

    @property
    def tool_widget_description(self):
        if not self._group in self._app.brushmanager.groups:
            return None
        nbrushes = len(self._app.brushmanager.groups[self._group])
        #TRANSLATORS: number of brushes in a brush group, for tooltips
        return ngettext("%d brush", "%d brushes", nbrushes) % (nbrushes,)

    @property
    def tool_widget_icon_name(self):
        return "mypaint-tool-brush"  # fallback only

    def tool_widget_get_icon_pixbuf(self, size):
        if not self._group in self._app.brushmanager.groups:
            return None
        brushes = self._app.brushmanager.groups[self._group]
        if not brushes:
            return None
        icon = brushes[0].preview
        if size == icon.get_width() and size == icon.get_height():
            return icon.copy()
        else:
            return icon.scale_simple(size, size, gdk.INTERP_BILINEAR)


    def tool_widget_properties(self):
        """Run the properties dialog"""
        toplevel = self.get_toplevel()
        if not self._dialog:
            #TRANSLATORS: brush group properties dialog title
            flags = gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT
            buttons = (gtk.STOCK_CANCEL, gtk.RESPONSE_REJECT)
            dia = gtk.Dialog(title=_("Group %s") % (self._group,),
                             flags=flags, buttons=buttons)
            dia.set_position(gtk.WIN_POS_MOUSE)
            btn = gtk.Button(_("Rename Group"))
            btn.connect("clicked", self._rename_cb)
            dia.vbox.pack_start(btn, False, False)
            btn = gtk.Button(_("Export as Zipped Brushset"))
            btn.connect("clicked", self._export_cb)
            dia.vbox.pack_start(btn, False, False)
            btn = gtk.Button(_("Delete Group"))
            btn.connect("clicked", self._delete_cb)
            dia.vbox.pack_start(btn, False, False)
            dia.vbox.show_all()
            self._dialog = dia
        self._dialog.set_transient_for(toplevel)
        self._dialog.run()
        self._dialog.hide()


    ## Properties dialog action callbacks

    def _rename_cb(self, widget):
        """Properties dialog rename callback"""
        # XXX Because of the way this works, groups can only be renamed from
        # XXX    the widget's properties dialog at present. Maybe that's OK.
        self._dialog.hide()
        old_group = self._group
        new_group = dialogs.ask_for_name(self, _('Rename Group'), old_group)
        if not new_group:
            return
        if old_group not in self._app.brushmanager.groups:
            return
        if new_group not in self._app.brushmanager.groups:
            self._app.brushmanager.rename_group(old_group, new_group)
            self._group = new_group
            workspace = self._app.workspace
            gtype_name = self.__gtype_name__
            workspace.update_tool_widget_params(gtype_name,
                                                (old_group,), (new_group,))
            self._update_brush_list()
        else:
            dialogs.error(self, _('A group with this name already exists!'))


    def _delete_cb(self, widget):
        """Properties dialog delete callback"""
        self._dialog.hide()
        name = brushmanager.translate_group_name(self._group)
        msg = _('Really delete group "%s"?') % (name,)
        bm = self._app.brushmanager
        if not dialogs.confirm(self, msg):
            return
        bm.delete_group(self._group)
        if self._group not in bm.groups:
            self._app.workspace.hide_tool_widget(self.__gtype_name__,
                                                 (self._group,))
            return
        msg = _('Group "%s" cannot be deleted. Try emptying it first.')
        dialogs.error(self, msg % (name,))


    def _export_cb(self, widget):
        """Properties dialog export callback"""
        self._dialog.hide()
        format_id, filename = dialogs.save_dialog(
                _("Export Brushes"), None,
                [(_("MyPaint brush package (*.zip)"), "*.zip")],
                default_format = (0, ".zip"))
        if filename is not None:
            self._app.brushmanager.export_group(self._group, filename)



class BrushGroupsMenu (gtk.Menu):
    """Dynamic menu containing all the brush groups"""

    def __init__(self):
        gtk.Menu.__init__(self)
        from application import get_app
        self.app = get_app()
        # Static items
        item = gtk.SeparatorMenuItem()
        self.append(item)
        item = gtk.MenuItem(_("New Group..."))
        item.connect("activate", self._new_brush_group_cb)
        self.append(item)
        item = gtk.MenuItem(_("Import Brushes..."))
        item.connect("activate", self.app.drawWindow.import_brush_pack_cb)
        self.append(item)
        item = gtk.MenuItem(_("Get More Brushes..."))
        item.connect("activate", self.app.drawWindow.download_brush_pack_cb)
        self.append(item)
        # Dynamic items
        bm = self.app.brushmanager
        self._items = {}
        self._update(bm)
        bm.groups_changed += self._update


    def _new_brush_group_cb(self, widget):
        # XXX should be moved somewhere more sensible than this
        toplevel = self.app.drawWindow
        name = dialogs.ask_for_name(toplevel, _('Create Group'), '')
        if name:
            bm = self.app.brushmanager
            bm.create_group(name)


    def _update(self, bm):
        """Update dynamic items in response to the groups list changing"""
        for item in self._items.itervalues():
            if item not in self:
                continue
            self.remove(item)
        activate_cb = self._brush_group_item_activate_cb
        for name in reversed(sorted(bm.groups)):
            if name in self._items:
                item = self._items[name]
            else:
                item = gtk.ImageMenuItem()
                label = brushmanager.translate_group_name(name)
                item.set_label(label)
                item.connect("activate", activate_cb, name)
                self._items[name] = item
            self.prepend(item)
        for name, item in list(self._items.iteritems()):
            if item in self:
                continue
            self._items.pop(name)
        self.show_all()

    def _brush_group_item_activate_cb(self, menuitem, group_name):
        workspace = self.app.workspace
        gtype_name = BrushGroupTool.__gtype_name__
        params = (group_name,)
        workspace.show_tool_widget(gtype_name, params)


class BrushGroupsToolItem (widgets.MenuOnlyToolButton):
    """Toolbar item showing a dynamic dropdown BrushGroupsMenu

    This is instantiated by the app's UIManager using a FactoryAction which
    must be named "BrushGroups" (see factoryaction.py).
    """

    __gtype_name__ = "MyPaintBrushGroupsToolItem"

    def __init__(self):
        widgets.MenuOnlyToolButton.__init__(self)
        self._menu = BrushGroupsMenu()
        self.set_menu(self._menu)
        #self._menu.show_all()


class BrushGroupsMenuItem (gtk.MenuItem):
    """Brush list menu item with a dynamic BrushGroupsMenu as its submenu

    This is instantiated by the app's UIManager using a FactoryAction which
    must be named "BrushGroups" (see factoryaction.py).
    """

    __gtype_name__ = "MyPaintBrushGroupsMenuItem"

    def __init__(self):
        gtk.MenuItem.__init__(self)
        self._submenu = BrushGroupsMenu()
        self.set_submenu(self._submenu)
        self._submenu.show_all()

