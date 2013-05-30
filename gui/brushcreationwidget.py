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
from document import CanvasController
from canvasevent import FreehandOnlyMode


def startfile(path):
    import os
    import platform
    if platform.system == 'Windows':
        os.startfile(path)
    else:
        os.system("xdg-open " + path)


class BrushManipulationWidget(gtk.VBox):
    """ """
    def __init__(self, app, brushicon_editor):
        gtk.VBox.__init__(self)
        self.app = app
        self.bm = app.brushmanager
        self.brushicon_editor = brushicon_editor

        self.init_widgets()

        self.bm.selected_brush_observers.append(self.brush_selected_cb)

    def init_widgets(self):
        l = self.brush_name_label = gtk.Label()
        l.set_text(_('(unnamed brush)'))
        l.set_alignment(0.0, 0.0)
        self.pack_start(l, expand=False)

        hbox = gtk.HBox()
        self.pack_start(hbox, expand=False, padding=2)

        buttons = [
        (self.update_settings_cb, _('Save Settings')),
        (self.create_brush_cb, _('Add As New')),
        (self.edit_brush_cb, _('Edit Brush Icon')),
        (self.rename_brush_cb, _('Rename...')),
        (self.delete_brush_cb, _('Remove...')),
        ]

        for clicked_cb, tooltip in buttons:
            b = gtk.Button(tooltip)
            b.connect('clicked', clicked_cb)
            hbox.pack_start(b, expand=False)

    def brush_selected_cb(self, managed_brush, brushinfo):
        name = managed_brush.name
        if name is None:
            name = _('(unnamed brush)')
        else:
            name = name.replace('_', ' ')   # XXX safename/unsafename utils?
        self.brush_name_label.set_text(name)

    def edit_brush_cb(self, window):
        self.edit_brush_properties_cb()

    def create_brush_cb(self, window):
        """Create and save a new brush based on the current working brush."""
        b = brushmanager.ManagedBrush(self.bm)
        b.brushinfo = self.app.brush.clone()
        b.brushinfo.set_string_property("parent_brush_name", None) #avoid mis-hilight
        b.preview = self.brushicon_editor.get_preview_pixbuf()
        b.save()

        if self.bm.active_groups:
            group = self.bm.active_groups[0]
        else:
            group = brushmanager.DEFAULT_BRUSH_GROUP

        brushes = self.bm.get_group_brushes(group, make_active=True)
        brushes.insert(0, b)
        b.persistent = True   # Brush was saved
        self.bm.notify_brushes_observers(brushes)

        self.bm.select_brush(b)

        # Pretend that the active app.brush is a child of the new one, for the
        # sake of the strokemap and strokes drawn immediately after.
        self.app.brush.set_string_property("parent_brush_name", b.name)

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
            self.bm.notify_brush_observers(deleted_brushes)

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
                self.bm.notify_brush_observers(brushes)
                assert b not in brushes, 'Brush exists multiple times in the same group!'

        if not b.delete_from_disk():
            # stock brush can't be deleted
            deleted_brushes = self.bm.get_group_brushes(brushmanager.DELETED_BRUSH_GROUP)
            deleted_brushes.insert(0, b)
            self.bm.notify_brush_observers(deleted_brushes)


class BrushIconEditorWidget(gtk.VBox):

    def __init__(self, app):
        gtk.VBox.__init__(self)
        self.app = app
        self.bm = app.brushmanager

        self.set_border_width(8)

        self.init_widgets()

        self.bm.selected_brush_observers.append(self.brush_selected_cb)

        self.set_brush_preview_edit_mode(False)

    def init_widgets(self):
        button_box = gtk.HBox()

        model = lib.document.Document(self.app.brush)
        self.tdw = tileddrawwidget.TiledDrawWidget()
        self.tdw.set_model(model)
        self.tdw.set_size_request(brushmanager.PREVIEW_W*2,
                                  brushmanager.PREVIEW_H*2)
        self.tdw.scale = 2.0

        tdw_box = gtk.HBox()
        tdw_box.pack_start(self.tdw, expand=False, fill=False)
        tdw_box.pack_start(gtk.Label(), expand=True)

        self.pack_start(tdw_box, expand=False, fill=False, padding=3)
        self.pack_start(button_box, expand=False, fill=False, padding=3)

        ctrlr = CanvasController(self.tdw)
        ctrlr.init_pointer_events()
        ctrlr.modes.default_mode_class = FreehandOnlyMode

        self.brush_preview_edit_mode_button = b = gtk.CheckButton(_('Edit'))
        b.connect('toggled', self.brush_preview_edit_mode_cb)
        button_box.pack_start(b, expand=False, padding=3)

        self.brush_preview_clear_button = b = gtk.Button(_('Clear'))
        b.connect('clicked', self.clear_cb)
        button_box.pack_start(b, expand=False, padding=3)

        self.brush_preview_save_button = b = gtk.Button(_('Save'))
        b.connect('clicked', self.update_preview_cb)
        button_box.pack_start(b, expand=False, padding=3)

    def clear_cb(self, window):
        self.tdw.doc.clear_layer()

    def brush_preview_edit_mode_cb(self, button):
        self.set_brush_preview_edit_mode(button.get_active())

    def set_brush_preview_edit_mode(self, edit_mode):
        self.brush_preview_edit_mode = edit_mode

        self.brush_preview_edit_mode_button.set_active(edit_mode)
        self.brush_preview_save_button.set_sensitive(edit_mode)
        self.brush_preview_clear_button.set_sensitive(edit_mode)
        self.tdw.set_sensitive(edit_mode)

    def set_preview_pixbuf(self, pixbuf):
        if pixbuf is None:
            self.tdw.doc.clear()
        else:
            self.tdw.doc.load_from_pixbuf(pixbuf)

    def get_preview_pixbuf(self):
        w, h = brushmanager.PREVIEW_W, brushmanager.PREVIEW_H
        return self.tdw.doc.render_as_pixbuf(0, 0, w, h, alpha=False)

    def update_preview_cb(self, window):
        pixbuf = self.get_preview_pixbuf()
        b = self.bm.selected_brush
        if not b.name:
            dialogs.error(self, _('No brush selected, please use "Add As New" instead.'))
            return
        b.preview = pixbuf
        b.save()
        for brushes in self.bm.groups.itervalues():
            if b in brushes:
                self.bm.notify_brush_observers(brushes)

    def brush_selected_cb(self, managed_brush, brushinfo):
        # Update brush icon preview if it is not in edit mode
        if not self.brush_preview_edit_mode:
            self.set_preview_pixbuf(managed_brush.preview)

