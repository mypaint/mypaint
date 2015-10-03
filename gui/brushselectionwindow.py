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

import logging
logger = logging.getLogger(__name__)

import gtk2compat

from lib.gettext import C_
from lib.gettext import ngettext
if gtk2compat.USE_GTK3:
    import gi
    from gi.repository import PangoCairo
import pango
import gtk
from gtk import gdk
import glib
from libmypaint import brushsettings

import pixbuflist
import dialogs
import brushmanager
from brusheditor import BrushEditorWindow
from workspace import SizedVBoxToolWidget
import widgets


## Helper functions

def _managedbrush_idfunc(managedbrush):
    return managedbrush.name

def _managedbrush_namefunc(managedbrush):
    template = "{name}"
    if managedbrush.description:
        template = "{name}\n{description}"
    return template.format(
        name = managedbrush.get_display_name(),
        description = managedbrush.description,
    )

def _managedbrush_pixbuffunc(managedbrush):
    return managedbrush.preview


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
        pixbuflist.PixbufList.__init__(
            self, self.brushes, s, s,
            namefunc=_managedbrush_namefunc,
            pixbuffunc=_managedbrush_pixbuffunc,
            idfunc = _managedbrush_idfunc,
        )
        # Support device changing with the same event as that used
        # for brush choice:
        if not gtk2compat.USE_GTK3:
            self.set_extension_events(gdk.EXTENSION_EVENTS_ALL)

        self.set_selected(self.bm.selected_brush)
        self.bm.brushes_changed += self.brushes_modified_cb
        self.bm.brush_selected += self.brush_selected_cb
        self.item_selected += self._item_selected_cb
        self.item_popup += self._item_popup_cb

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
        if icons_tall * icons_wide < num_brushes:
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

    def on_drag_data(self, copy, source_widget, brush_dragid, target_idx):
        assert source_widget, 'cannot handle drag data from another app'
        brush = None
        for b in source_widget.brushes:
            b_dragid = source_widget.idfunc(b)
            if b_dragid == brush_dragid:
                brush = b
                break
        if brush is None:
            logger.error(
                "No brush with dragid=%r in drag source widget",
                brush_dragid,
            )
            return False
        if source_widget is self:
            copy = False
        elif brush in self.brushes:
            return True
        if not copy:
            source_widget.remove_brush(brush)
        self.insert_brush(target_idx, brush)
        return True

    def _item_selected_cb(self, self_, brush):
        # brush changed on harddisk?
        if brush.reload_if_changed():
            for brushes in self.bm.groups.itervalues():
                self.bm.brushes_changed(brushes)
        self.bm.select_brush(brush)

    def _item_popup_cb(self, self_, brush):
        time = gtk.get_current_event_time()
        self.menu = BrushPopupMenu(self, brush)
        self.menu.show_all()
        self.menu.popup(parent_menu_shell=None, parent_menu_item=None,
            func=None, button=3, activate_time=time, data=None)


class BrushPopupMenu(gtk.Menu):
    def __init__(self, bl, brush):
        super(BrushPopupMenu, self).__init__()
        faves = bl.bm.groups[brushmanager.FAVORITES_BRUSH_GROUP]
        if brush not in faves:
            item = gtk.MenuItem(C_("brush list context menu", "Add to favorites"))
            item.connect("activate", BrushPopupMenu.favorite_cb, bl, brush)
            self.append(item)
        else:
            item = gtk.MenuItem(C_("brush list context menu", "Remove from favorites"))
            item.connect("activate", BrushPopupMenu.unfavorite_cb, bl, brush)
            self.append(item)

        if bl.group != brushmanager.FAVORITES_BRUSH_GROUP:
            item = gtk.MenuItem(C_("brush list context menu", "Clone"))
            item.connect("activate", BrushPopupMenu.clone_cb, bl, brush)
            self.append(item)

        item = gtk.MenuItem(C_("brush list context menu", "Edit brush settings"))
        item.connect("activate", BrushPopupMenu.edit_cb, bl, brush)
        self.append(item)

        if bl.group != brushmanager.FAVORITES_BRUSH_GROUP:
            item = gtk.MenuItem(C_("brush list context menu", "Delete"))
            item.connect("activate", BrushPopupMenu.delete_cb, bl, brush, self)
            self.append(item)

    @staticmethod
    def favorite_cb(menuitem, bl, brush):
        faves = bl.bm.groups[brushmanager.FAVORITES_BRUSH_GROUP]
        if brush not in faves:
            faves.append(brush)
        bl.bm.brushes_changed(faves)
        bl.bm.save_brushorder()

    @staticmethod
    def unfavorite_cb(menuitem, bl, brush):
        faves = bl.bm.groups[brushmanager.FAVORITES_BRUSH_GROUP]
        try:
            faves.remove(brush)
        except ValueError:
            return
        bl.bm.brushes_changed(faves)
        bl.bm.save_brushorder()

    @staticmethod
    def clone_cb(menuitem, bl, brush):
        new_name = brush.name + "copy"
        brush_copy = brush.clone(new_name)
        index = bl.brushes.index(brush) + 1
        bl.insert_brush(index, brush_copy)
        brush_copy.save()
        bl.bm.save_brushorder()

    @staticmethod
    def edit_cb(menuitem, bl, brush):
        bl.bm.select_brush(brush)
        BrushEditorWindow().show()

    @staticmethod
    def delete_cb(menuitem, bl, brush, menu):
        msg = C_(
            "brush list commands",
            "Really delete brush from disk?",
        )
        if not dialogs.confirm(menu, msg):
            return
        bl.remove_brush(brush)
        faves = bl.bm.groups[brushmanager.FAVORITES_BRUSH_GROUP]
        bl.bm.brushes_changed(faves)


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
            raise ValueError("No group named %r" % group)
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
        if self._group not in self._app.brushmanager.groups:
            return None
        nbrushes = len(self._app.brushmanager.groups[self._group])
        #TRANSLATORS: number of brushes in a brush group, for tooltips
        return ngettext("%d brush", "%d brushes", nbrushes) % (nbrushes,)

    @property
    def tool_widget_icon_name(self):
        return "mypaint-brushes-symbolic"  # fallback only

    def tool_widget_get_icon_pixbuf(self, size):
        if self._group not in self._app.brushmanager.groups:
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
            #TRANSLATORS: properties dialog for the current brush group
            flags = gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT
            buttons = (gtk.STOCK_CANCEL, gtk.RESPONSE_REJECT)
            dia = gtk.Dialog(
                title=C_(
                        "brush group properties dialog: title",
                        u"Group \u201C{group_name}\u201D",
                    ).format(
                        group_name = self._group,
                    ),
                flags=flags,
                buttons=buttons)
            dia.set_position(gtk.WIN_POS_MOUSE)
            btn = gtk.Button(C_(
                "brush group properties dialog: action buttons",
                "Rename Group",
            ))
            btn.connect("clicked", self._rename_cb)
            dia.vbox.pack_start(btn, False, False)
            btn = gtk.Button(C_(
                "brush group properties dialog: action buttons",
                "Export as Zipped Brushset",
            ))
            btn.connect("clicked", self._export_cb)
            dia.vbox.pack_start(btn, False, False)
            btn = gtk.Button(C_(
                "brush group properties dialog: action buttons",
                "Delete Group",
            ))
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
        new_group = dialogs.ask_for_name(
            self,
            C_("brush group rename dialog: title", "Rename Group"),
            old_group,
        )
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
            dialogs.error(self, C_(
                'brush group rename',
                'A group with this name already exists!',
            ))

    def _delete_cb(self, widget):
        """Properties dialog delete callback"""
        self._dialog.hide()
        name = brushmanager.translate_group_name(self._group)
        msg = C_(
            "brush group delete",
            u"Really delete group \u201C{group_name}\u201D?",
        ).format(
            group_name = name,
        )
        bm = self._app.brushmanager
        if not dialogs.confirm(self, msg):
            return
        bm.delete_group(self._group)
        if self._group not in bm.groups:
            remover = lambda t, q: (
                self._app.workspace.remove_tool_widget(t, q) or False
            )
            glib.idle_add(remover, self.__gtype_name__, (self._group,))
            return
        # Special groups like "Deleted" cannot be deleted,
        # but the error message is very confusing in that case...
        msg = C_(
            "brush group delete",
            u"Could not delete group \u201C{group_name}\u201D.\n"
            u"Some special groups cannot be deleted.",
        ).format(
            group_name = name,
        )
        dialogs.error(self, msg)

    def _export_cb(self, widget):
        """Properties dialog export callback"""
        self._dialog.hide()
        format_id, filename = dialogs.save_dialog(
            C_("brush group export dialog: title", "Export Brushes"),
            None,
            [(
                C_(
                    "brush group export dialog",
                    "MyPaint brush package (*.zip)",
                ),
                "*.zip",
            )],
            default_format=(0, ".zip"),
        )
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
        item = gtk.MenuItem(C_("brush groups menu", "New Group..."))
        item.connect("activate", self._new_brush_group_cb)
        self.append(item)
        item = gtk.MenuItem(C_("brush groups menu", "Import Brushes..."))
        item.connect("activate", self.app.drawWindow.import_brush_pack_cb)
        self.append(item)
        item = gtk.MenuItem(C_("brush groups menu", "Get More Brushes..."))
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
        name = dialogs.ask_for_name(
            toplevel,
            C_("new brush group dialog: title", 'Create Group'),
            '',
        )
        if name is None:
            return
        name = name.strip()
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
        workspace.add_tool_widget(gtype_name, params)


class BrushGroupsToolItem (widgets.MenuButtonToolItem):
    """Toolbar item showing a dynamic dropdown BrushGroupsMenu

    This is instantiated by the app's UIManager using a FactoryAction which
    must be named "BrushGroups" (see factoryaction.py).
    """

    __gtype_name__ = "MyPaintBrushGroupsToolItem"

    def __init__(self):
        widgets.MenuButtonToolItem.__init__(self)
        self.menu = BrushGroupsMenu()


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
