# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2007-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2009-2019 by the MyPaint Development Team.
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

from __future__ import division, print_function
import logging

from lib.gibindings import Gtk
from lib.gibindings import GdkPixbuf
from lib.gibindings import GLib

from lib.gettext import C_
from lib.gettext import ngettext

from . import pixbuflist
from . import dialogs
from . import brushmanager
from .toolstack import SizedVBoxToolWidget
from . import widgets

logger = logging.getLogger(__name__)


## Helper functions

def managedbrush_idfunc(managedbrush):
    """Returns the id of a ManagedBrush."""
    return managedbrush.name


def managedbrush_namefunc(managedbrush):
    """Returns tooltip of a ManagedBrush."""
    template = "{name}"
    if managedbrush.description:
        template = "{name}\n{description}"
    return template.format(
        name = managedbrush.get_display_name(),
        description = managedbrush.description,
    )


def managedbrush_pixbuffunc(managedbrush):
    """Returns pixbuf preview of a ManagedBrush."""
    return managedbrush.preview


## Class definitions


class BrushList (pixbuflist.PixbufList):
    """Flowed grid of brush icons showing a group, click to set the brush"""

    ICON_SIZE = 48
    MIN_WIDTH_NICONS = 1
    NATURAL_WIDTH_NICONS = 4
    MIN_HEIGHT_NICONS = 1

    def __init__(self, app, group):
        """Construct, for a named group."""
        self.app = app
        self.bm = app.brushmanager
        self.group = group
        s = self.ICON_SIZE
        super(BrushList, self).__init__(
            self.brushes, s, s,
            namefunc=managedbrush_namefunc,
            pixbuffunc=managedbrush_pixbuffunc,
            idfunc=managedbrush_idfunc,
        )
        self.set_selected(self.bm.selected_brush)
        self.bm.groups_changed += self._groups_changed_cb
        self.bm.brushes_changed += self._brushes_changed_cb
        self.bm.brush_selected += self._brush_selected_cb
        self.item_selected += self._item_selected_cb
        self.item_popup += self._item_popup_cb

    @property
    def brushes(self):
        """The list of brushes being shown.

        The returned list belongs to the main app's BrushManager,
        and is created on demand if there is no such group.
        If you reorder it or remove brushes,
        you must call the BrushManager's brushes_changed() method too.

        """
        group_name = self.group
        return self.bm.get_group_brushes(group_name)

    def do_get_request_mode(self):
        return Gtk.SizeRequestMode.HEIGHT_FOR_WIDTH

    def do_get_preferred_width(self):
        return (self.MIN_WIDTH_NICONS * self.ICON_SIZE,
                self.NATURAL_WIDTH_NICONS * self.ICON_SIZE)

    def do_get_preferred_height_for_width(self, width):
        icons_wide = max(1, int(width // self.ICON_SIZE))
        num_brushes = len(self.brushes)
        icons_tall = max(int(num_brushes // icons_wide),
                         max(self.MIN_HEIGHT_NICONS, 1))
        if icons_tall * icons_wide < num_brushes:
            icons_tall += 1
        return (icons_tall * self.ICON_SIZE,
                icons_tall * self.ICON_SIZE)

    def _groups_changed_cb(self, bm):
        # In case the group has been deleted and recreated, we do this
        group_name = self.group
        group_brushes = self.bm.groups.get(group_name, [])
        self.itemlist = group_brushes
        self.update()
        # See https://github.com/mypaint/mypaint/issues/654

    def _brushes_changed_cb(self, bm, brushes):
        # CARE: this might be called in response to the group being deleted.
        # Don't recreate it by accident.
        group_name = self.group
        group_brushes = self.bm.groups.get(group_name)
        if brushes is group_brushes:
            self.update()

    def _brush_selected_cb(self, bm, managed_brush, brushinfo):
        self.set_selected(managed_brush)

    def remove_brush(self, brush):
        self.brushes.remove(brush)
        self.bm.brushes_changed(self.brushes)

    def insert_brush(self, idx, brush):
        self.brushes.insert(idx, brush)
        self.bm.brushes_changed(self.brushes)

    def button_press_cb(self, widget, event):
        device = event.get_source_device()
        self.app.device_monitor.device_used(device)
        return super(BrushList, self).button_press_cb(widget, event)

    def drag_begin_cb(self, widget, context):
        preview = self.bm.selected_brush.preview
        preview = preview.scale_simple(
            preview.get_width() // 2,
            preview.get_height() // 2,
            GdkPixbuf.InterpType.BILINEAR,
        )
        Gtk.drag_set_icon_pixbuf(context, preview, 0, 0)
        super(BrushList, self).drag_begin_cb(widget, context)

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
            for brushes in self.bm.groups.values():
                self.bm.brushes_changed(brushes)
        self.bm.select_brush(brush)

    def _item_popup_cb(self, self_, brush):
        time = Gtk.get_current_event_time()
        menu = BrushPopupMenu(self, brush)
        menu.show_all()
        menu.popup(
            parent_menu_shell = None,
            parent_menu_item = None,
            func = None,
            button = 3,
            activate_time = time,
            data = None,
        )


class BrushPopupMenu (Gtk.Menu):
    """Pop-up menu for brush actions."""

    def __init__(self, bl, brush):
        super(BrushPopupMenu, self).__init__()
        self._brush = brush
        self._brushlist = bl
        self._app = bl.app
        faves = bl.bm.get_group_brushes(brushmanager.FAVORITES_BRUSH_GROUP)
        if brush not in faves:
            item = Gtk.MenuItem(label=C_(
                "brush group: context menu for a single brush",
                "Add to Favorites",
            ))
            item.connect("activate", self._favorite_cb)
            self.append(item)
        else:
            item = Gtk.MenuItem(label=C_(
                "brush group: context menu for a single brush",
                "Remove from Favorites",
            ))
            item.connect("activate", self._unfavorite_cb)
            self.append(item)

        if bl.group != brushmanager.FAVORITES_BRUSH_GROUP:
            item = Gtk.MenuItem(label=C_(
                "brush group: context menu for a single brush",
                "Clone",
            ))
            item.connect("activate", self._clone_cb)
            self.append(item)

        item = Gtk.MenuItem(label=C_(
            "brush group: context menu for a single brush",
            "Edit Brush Settings",
        ))
        item.connect("activate", self._edit_cb)
        self.append(item)

        if bl.group != brushmanager.FAVORITES_BRUSH_GROUP:
            item = Gtk.MenuItem(label=C_(
                "brush group: context menu for a single brush",
                "Remove from Group",
            ))
            item.connect("activate", self._remove_cb)
            self.append(item)

    def _favorite_cb(self, menuitem):
        bl = self._brushlist
        brush = self._brush
        # Update the faves group if the brush isn't already there.
        faves_group_name = brushmanager.FAVORITES_BRUSH_GROUP
        faves = bl.bm.get_group_brushes(faves_group_name)
        if brush not in faves:
            faves.append(brush)
            bl.bm.brushes_changed(faves)
            bl.bm.save_brushorder()
        # Show the faves group
        workspace = self._app.workspace
        gtype_name = BrushGroupTool.__gtype_name__
        params = (faves_group_name,)
        workspace.reveal_tool_widget(gtype_name, params)
        # Highlight the (possibly copied) brush
        bl.bm.select_brush(brush)

    def _unfavorite_cb(self, menuitem):
        bl = self._brushlist
        brush = self._brush
        faves = bl.bm.get_group_brushes(brushmanager.FAVORITES_BRUSH_GROUP)
        try:
            faves.remove(brush)
        except ValueError:
            return
        bl.bm.brushes_changed(faves)
        bl.bm.save_brushorder()

    def _clone_cb(self, menuitem):
        bl = self._brushlist
        brush = self._brush
        # Pick a nice unique name
        new_name = C_(
            "brush group: context menu: unique names for cloned brushes",
            u"{original_name} copy"
        ).format(
            original_name = brush.name,
        )
        uniquifier = 0
        while bl.bm.get_brush_by_name(new_name):
            uniquifier += 1
            new_name = C_(
                "brush group: context menu: unique names for cloned brushes",
                u"{original_name} copy {n}"
            ).format(
                name = brush.name,
                n = uniquifier,
            )
        # Make a copy and insert it near the original
        brush_copy = brush.clone(new_name)
        index = bl.brushes.index(brush) + 1
        bl.insert_brush(index, brush_copy)
        brush_copy.save()
        bl.bm.save_brushorder()
        # Select the copy, for highlighting
        bl.bm.select_brush(brush_copy)

    def _edit_cb(self, menuitem):
        bl = self._brushlist
        brush = self._brush
        bl.bm.select_brush(brush)
        brush_editor = self._app.brush_settings_window
        brush_editor.show_all()

    def _remove_cb(self, menuitem):
        bl = self._brushlist
        brush = self._brush
        msg = C_(
            "brush group: context menu: remove from group",
            u"Really remove brush “{brush_name}” "
            u"from group “{group_name}”?"
        ).format(
            brush_name = brush.name,
            group_name = bl.group,
        )
        if not dialogs.confirm(bl, msg):
            return
        bl.remove_brush(brush)


class BrushGroupTool (SizedVBoxToolWidget):
    """Dockable tool widget showing just one BrushGroup"""

    __gtype_name__ = "MyPaintBrushGroupTool"

    ## Construction and updating

    def __init__(self, group):
        """Construct, to show a named group"""
        SizedVBoxToolWidget.__init__(self)
        self._group = group
        self._scrolls = Gtk.ScrolledWindow()
        self._dialog = None
        self._brush_list = None
        from gui.application import get_app
        self._app = get_app()
        if group not in self._app.brushmanager.groups:
            raise ValueError("No group named %r" % group)
        self.pack_start(self._scrolls, True, True, 0)
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
        self._scrolls.add(self._brush_list)
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
        # TRANSLATORS: number of brushes in a brush group, for tooltips
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
            return icon.scale_simple(size, size, GdkPixbuf.InterpType.BILINEAR)

    def tool_widget_properties(self):
        """Run the properties dialog"""
        if not self._dialog:
            title = C_(
                "brush group properties dialog: title",
                # TRANSLATORS: properties dialog for the current brush group
                u"Group \u201C{group_name}\u201D",
            ).format(
                group_name = self._group,
            )
            dia = Gtk.Dialog(
                title=title,
                modal=True,
                destroy_with_parent=True,
                window_position=Gtk.WindowPosition.MOUSE,
            )
            dia.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.REJECT)
            btn = Gtk.Button(label=C_(
                "brush group properties dialog: action buttons",
                "Rename Group",
            ))
            btn.connect("clicked", self._rename_cb)
            dia.vbox.pack_start(btn, False, False, 0)
            btn = Gtk.Button(label=C_(
                "brush group properties dialog: action buttons",
                "Export as Zipped Brushset",
            ))
            btn.connect("clicked", self._export_cb)
            dia.vbox.pack_start(btn, False, False, 0)
            btn = Gtk.Button(label=C_(
                "brush group properties dialog: action buttons",
                "Delete Group",
            ))
            btn.connect("clicked", self._delete_cb)
            dia.vbox.pack_start(btn, False, False, 0)
            dia.vbox.show_all()
            self._dialog = dia
        self._dialog.set_transient_for(self.get_toplevel())
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
            GLib.idle_add(
                self._remove_panel_idle_cb,
                self.__gtype_name__, (self._group,),
            )
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

    def _remove_panel_idle_cb(self, typespec, paramspec):
        self._app.workspace.remove_tool_widget(typespec, paramspec)
        return False

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


class BrushGroupsMenu (Gtk.Menu):
    """Dynamic menu containing all the brush groups"""

    def __init__(self):
        super(BrushGroupsMenu, self).__init__()
        from gui.application import get_app
        self.app = get_app()
        # Static items
        item = Gtk.SeparatorMenuItem()
        self.append(item)
        item = Gtk.MenuItem(label=C_("brush groups menu", u"New Group…"))
        item.connect("activate", self._new_brush_group_cb)
        self.append(item)
        item = Gtk.MenuItem(label=C_("brush groups menu", u"Import Brushes…"))
        item.connect("activate", self.app.drawWindow.import_brush_pack_cb)
        self.append(item)
        item = Gtk.MenuItem(label=C_(
            "brush groups menu", u"Get More Brushes…")
        )
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
        for item in self._items.values():
            if item not in self:
                continue
            self.remove(item)
        activate_cb = self._brush_group_item_activate_cb
        for name in reversed(sorted(bm.groups)):
            if name in self._items:
                item = self._items[name]
            else:
                item = Gtk.ImageMenuItem()
                label = brushmanager.translate_group_name(name)
                item.set_label(label)
                item.connect("activate", activate_cb, name)
                self._items[name] = item
            self.prepend(item)
        for name, item in list(self._items.items()):
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
        super(BrushGroupsToolItem, self).__init__()
        self.menu = BrushGroupsMenu()


class BrushGroupsMenuItem (Gtk.MenuItem):
    """Brush list menu item with a dynamic BrushGroupsMenu as its submenu

    This is instantiated by the app's UIManager using a FactoryAction which
    must be named "BrushGroups" (see factoryaction.py).
    """

    __gtype_name__ = "MyPaintBrushGroupsMenuItem"

    def __init__(self):
        super(BrushGroupsMenuItem, self).__init__()
        self._submenu = BrushGroupsMenu()
        self.set_submenu(self._submenu)
        self._submenu.show_all()
