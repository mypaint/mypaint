# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os
import gtk
gdk = gtk.gdk
from lib import document
import tileddrawwidget, brushmanager, dialogs
from gettext import gettext as _

def startfile(path):
    import os
    import platform
    if platform.system == 'Windows':
        os.startfile(path)
    else:
        os.system("xdg-open " + path)

class Widget(gtk.HBox):
    def __init__(self, app):
        gtk.HBox.__init__(self)
        self.app = app
        self.bm = app.brushmanager

        self.set_border_width(8)

        left_vbox = gtk.VBox()
        right_vbox = gtk.VBox()
        self.pack_start(left_vbox, expand=False, fill=False)
        self.pack_end(right_vbox, expand=False, fill=False)

        #expanded part, left side
        doc = document.Document()
        self.tdw = tileddrawwidget.TiledDrawWidget(doc)
        self.tdw.set_size_request(brushmanager.preview_w, brushmanager.preview_h)
        left_vbox.pack_start(self.tdw, expand=False, fill=False)

        b = gtk.Button(_('Clear'))
        def clear_cb(window):
            self.tdw.doc.clear_layer()
        b.connect('clicked', clear_cb)
        left_vbox.pack_start(b, expand=False, padding=5)

        #expanded part, right side
        l = self.brush_name_label = gtk.Label()
        l.set_justify(gtk.JUSTIFY_LEFT)
        l.set_text(_('(no name)'))
        right_vbox.pack_start(l, expand=False)

        right_vbox_buttons = [
        (_('Add As New'), self.create_brush_cb),
        (_('Rename...'), self.rename_brush_cb),
        (_('Remove...'), self.delete_brush_cb),
        (_('Settings...'), self.brush_settings_cb),
        (_('Save Settings'), self.update_settings_cb),
        (_('Save Preview'), self.update_preview_cb),
        (_('About brush'),  self.show_about_cb),
        ]

        for title, clicked_cb in right_vbox_buttons:
            b = gtk.Button(title)
            b.connect('clicked', clicked_cb)
            right_vbox.pack_start(b, expand=False)

        self.last_selected_brush = self.bm.selected_brush
        self.bm.selected_brush_observers.append(self.brush_selected_cb)
        self.app.brush.settings_observers.append(self.brush_modified_cb)


    def set_preview_pixbuf(self, pixbuf):
        if pixbuf is None:
            self.tdw.doc.clear()
        else:
            self.tdw.doc.load_from_pixbuf(pixbuf)

    def get_preview_pixbuf(self):
        pixbuf = self.tdw.doc.render_as_pixbuf(0, 0, brushmanager.preview_w, brushmanager.preview_h)
        return pixbuf

    def show_about_cb(self, window):
        b = self.bm.selected_brush
        path = b.get_fileprefix()
        dir = os.path.dirname(path)
        found = False
        for name in ["README", "LICENSE", "LEGAL", "COPYRIGHT"]:
            for another_name in [name, name + '.txt', name + '.TXT', name.lower(), name.lower() + '.txt']:
                filename = os.path.join(dir, another_name)
                if os.path.isfile(filename):
                    startfile(filename)
                    found = True
                    break
        if not found:
            dialogs.error(self, _('No README file for this brush!'))

    def brush_settings_cb(self, window):
        w = self.app.brushSettingsWindow
        w.show_all() # might be for the first time
        w.present()

    def create_brush_cb(self, window):
        b = brushmanager.ManagedBrush(self.bm)
        b.settings_str = self.app.brush.save_to_string()
        b.preview = self.get_preview_pixbuf()
        b.save()

        if self.bm.active_groups:
            group = self.bm.active_groups[0]
        else:
            group = brushmanager.DEFAULT_BRUSH_GROUP

        brushes = self.bm.get_group_brushes(group, make_active=True)
        brushes.insert(0, b)
        for f in self.bm.brushes_observers: f(brushes)

        self.bm.select_brush(b)

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

        print 'renaming brush', repr(src_brush.name), '-->', repr(dst_name)
        if dst_deleted:
            deleted_brushes = self.bm.get_group_brushes(brushmanager.DELETED_BRUSH_GROUP)
            deleted_brushes.remove(dst_deleted)
            for f in self.bm.brushes_observers: f(deleted_brushes)

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

    def update_preview_cb(self, window):
        pixbuf = self.get_preview_pixbuf()
        b = self.bm.selected_brush
        if not b.name:
            dialogs.error(self, _('No brush selected, please use "add as new" instead.'))
            return
        b.preview = pixbuf
        b.save()
        for brushes in self.bm.groups.itervalues():
            if b in brushes:
                for f in self.bm.brushes_observers: f(brushes)

    def update_settings_cb(self, window):
        b = self.bm.selected_brush
        if not b.name:
            dialogs.error(self, _('No brush selected, please use "add as new" instead.'))
            return
        b.settings_str = self.app.brush.save_to_string()
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
                for f in self.bm.brushes_observers: f(brushes)
                assert b not in brushes, 'Brush exists multiple times in the same group!'

        if not b.delete_from_disk():
            # stock brush can't be deleted
            deleted_brushes = self.bm.get_group_brushes(brushmanager.DELETED_BRUSH_GROUP)
            deleted_brushes.insert(0, b)
            for f in self.bm.brushes_observers: f(deleted_brushes)

    def brush_selected_cb(self, b):
        name = b.name
        if name is None:
            name = _('(no name)')
        else:
            name = name.replace('_', ' ')
        self.brush_name_label.set_text(name)

        # selecting twice loads preview
        if b == self.last_selected_brush:
            self.set_preview_pixbuf(b.preview)
        self.last_selected_brush = self.bm.selected_brush

    def brush_modified_cb(self):
        self.tdw.doc.set_brush(self.app.brush)

