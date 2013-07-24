# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os
import logging
logger = logging.getLogger(__name__)

import gtk
from gtk import gdk
from gettext import gettext as _

import lib.document
import tileddrawwidget, brushmanager, dialogs


def startfile(path):
    import os
    import platform
    if platform.system == 'Windows':
        os.startfile(path)
    else:
        os.system("xdg-open " + path)


class BrushManipulationWidget (gtk.VBox):
    """Widget for manipulating a brush"""

    def __init__(self, app):
        gtk.VBox.__init__(self)
        self.app = app
        self.bm = app.brushmanager

        self.init_widgets()

        self.bm.brush_selected += self.brush_selected_cb

    def init_widgets(self):
        l = self.brush_name_label = gtk.Label()
        l.set_text(_('(unnamed brush)'))
        l.set_alignment(0.0, 0.0)
        self.pack_start(l, expand=False)

        hbox = gtk.HBox()
        self.pack_start(hbox, expand=False, padding=2)

        b = gtk.Button(_('Save Settings'))
        b.connect('clicked', self.update_settings_cb)
        hbox.pack_start(b, expand=False)

        b = gtk.Button(_('Add as New'))
        b.connect('clicked', self.create_brush_cb)
        hbox.pack_start(b, expand=False)

        b = gtk.ToggleButton()
        a = self.app.find_action("BrushIconEditorWindow")
        b.set_related_action(a)
        hbox.pack_start(b, expand=False)

        b = gtk.Button(_('Rename...'))
        b.connect('clicked', self.rename_brush_cb)
        hbox.pack_start(b, expand=False)

        b = gtk.Button(_('Remove...'))
        b.connect('clicked', self.delete_brush_cb)
        hbox.pack_start(b, expand=False)


    def brush_selected_cb(self, bm, managed_brush, brushinfo):
        name = managed_brush.name
        if name is None:
            name = _('(unnamed brush)')
        else:
            name = name.replace('_', ' ')   # XXX safename/unsafename utils?
        self.brush_name_label.set_text(name)


    def create_brush_cb(self, window):
        """Create and save a new brush based on the current working brush."""
        b = self.bm.selected_brush.clone(name=None)
        b.brushinfo.set_string_property("parent_brush_name", None) #avoid mis-hilight
        b.save()

        group = brushmanager.NEW_BRUSH_GROUP
        brushes = self.bm.get_group_brushes(group)
        brushes.insert(0, b)
        b.persistent = True   # Brush was saved
        self.bm.brushes_changed(brushes)

        self.bm.select_brush(b)

        # Pretend that the active app.brush is a child of the new one, for the
        # sake of the strokemap and strokes drawn immediately after.
        self.app.brush.set_string_property("parent_brush_name", b.name)

        # Reveal the added group if it's hidden
        ws = self.app.workspace
        ws.show_tool_widget("MyPaintBrushGroupTool", (group,))


    def rename_brush_cb(self, window):
        src_brush = self.bm.selected_brush
        if not src_brush.name:
            dialogs.error(self, _('No brush selected!'))
            return

        dst_name = dialogs.ask_for_name(self, _("Rename Brush"), src_brush.name.replace('_', ' '))
        if not dst_name:
            return
        dst_name = dst_name.replace(' ', '_')
        # ensure we don't overwrite an existing brush by accident
        dst_deleted = None
        for group, brushes in self.bm.groups.iteritems():
            for b2 in brushes:
                if b2.name == dst_name:
                    if group == brushmanager.DELETED_BRUSH_GROUP:
                        dst_deleted = b2
                    else:
                        dialogs.error(self, _('A brush with this name already exists!'))
                        return

        logger.info("Renaming brush %r --> %r", src_brush.name, dst_name)
        if dst_deleted:
            deleted_brushes = self.bm.get_group_brushes(brushmanager.DELETED_BRUSH_GROUP)
            deleted_brushes.remove(dst_deleted)
            self.bm.brushes_changed(deleted_brushes)

        # save src as dst
        src_name = src_brush.name
        src_brush.name = dst_name
        src_brush.save()
        src_brush.name = src_name
        # load dst
        dst_brush = brushmanager.ManagedBrush(self.bm, dst_name, persistent=True)
        dst_brush.load()

        # replace src with dst (but keep src in the deleted list if it is a stock brush)
        self.delete_brush_internal(src_brush, replacement=dst_brush)

        self.bm.select_brush(dst_brush)

    def update_settings_cb(self, window):
        b = self.bm.selected_brush
        if not b.name:
            dialogs.error(self, _('No brush selected, please use "Add As New" instead.'))
            return
        b.brushinfo = self.app.brush.clone()
        b.save()

    def delete_brush_cb(self, window):
        b = self.bm.selected_brush
        if not b.name:
            dialogs.error(self, _('No brush selected!'))
            return
        if not dialogs.confirm(self, _("Really delete brush from disk?")):
            return
        self.bm.select_brush(None)
        self.delete_brush_internal(b)

    def delete_brush_internal(self, b, replacement=None):
        for brushes in self.bm.groups.itervalues():
            if b in brushes:
                idx = brushes.index(b)
                if replacement:
                    brushes[idx] = replacement
                else:
                    del brushes[idx]
                self.bm.brushes_changed(brushes)
                assert b not in brushes, \
                        'Brush exists multiple times in the same group!'

        if not b.delete_from_disk():
            # stock brush can't be deleted
            deleted_brushes = self.bm.get_group_brushes(brushmanager.DELETED_BRUSH_GROUP)
            deleted_brushes.insert(0, b)
            self.bm.brushes_changed(deleted_brushes)


