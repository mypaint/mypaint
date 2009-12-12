# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
gdk = gtk.gdk
from lib import document
import tileddrawwidget, brushmanager, dialogs
from gettext import gettext as _

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
        (_('add as new'), self.create_brush_cb),
        (_('rename...'), self.rename_brush_cb),
        (_('remove...'), self.delete_brush_cb),
        (_('settings...'), self.brush_settings_cb),
        (_('save settings'), self.update_settings_cb),
        (_('save preview'), self.update_preview_cb),
        ]

        for title, clicked_cb in right_vbox_buttons:
            b = gtk.Button(title)
            b.connect('clicked', clicked_cb)
            right_vbox.pack_start(b, expand=False)

        self.app.brushmanager.selected_brush_observers.append(self.brush_selected_cb)
        self.app.brush.settings_observers.append(self.brush_modified_cb)


    def set_preview_pixbuf(self, pixbuf):
        if pixbuf is None:
            self.tdw.doc.clear()
        else:
            self.tdw.doc.load_from_pixbuf(pixbuf)

    def get_preview_pixbuf(self):
        pixbuf = self.tdw.doc.render_as_pixbuf(0, 0, brushmanager.preview_w, brushmanager.preview_h)
        return pixbuf

    def brush_settings_cb(self, window):
        w = self.app.brushSettingsWindow
        w.show_all() # might be for the first time
        w.present()

    def create_brush_cb(self, window):
        b = brushmanager.ManagedBrush(self.bm)
        b.copy_settings_from(self.app.brush)
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
        b = self.bm.selected_brush
        if not b.name:
            display = gdk.display_get_default()
            display.beep()
            return

        name = dialogs.ask_for_name(self, _("Rename Brush"), b.name.replace('_', ' '))
        if not name:
            return
        name = name.replace(' ', '_')
        print 'renaming brush', repr(b.name), '-->', repr(name)
        # ensure we don't overwrite an existing brush by accident
        for group, brushes in self.bm.groups.iteritems():
            if group == brushmanager.DELETED_BRUSH_GROUP:
                continue
            for b2 in brushes:
                if b2.name == name:
                    dialogs.error(self, _('A brush with this name already exists!'))
                    return
        success = b.delete_from_disk()
        old_name = b.name
        b.name = name
        b.save()
        if not success:
            # we are renaming a stock brush
            # we can't delete the original; instead we put it away so it doesn't reappear
            old_brush = brushmanager.ManagedBrush(self.bm)
            old_brush.load(old_name)
            deleted_brushes = self.bm.get_group_brushes(brushmanager.DELETED_BRUSH_GROUP)
            deleted_brushes.insert(0, old_brush)
            for f in self.bm.brushes_observers: f(deleted_brushes)

        self.bm.select_brush(b)

    def update_preview_cb(self, window):
        pixbuf = self.get_preview_pixbuf()
        b = self.bm.selected_brush
        if not b.name:
            # no brush selected
            display = gdk.display_get_default()
            display.beep()
            return
        b.preview = pixbuf
        b.save()
        for brushes in self.bm.groups.itervalues():
            if b in brushes:
                for f in self.bm.brushes_observers: f(brushes)

    def update_settings_cb(self, window):
        b = self.bm.selected_brush
        if not b.name:
            # no brush selected
            display = gdk.display_get_default()
            display.beep()
            return
        b.copy_settings_from(self.app.brush)
        b.save()

    def delete_brush_cb(self, window):
        b = self.bm.selected_brush
        if not b.name:
            display = gdk.display_get_default()
            display.beep()
            return
        if not dialogs.confirm(self, _("Really delete brush from disk?")):
            return

        self.bm.select_brush(None)

        for brushes in self.bm.groups.itervalues():
            if b in brushes:
                brushes.remove(b)
                for f in self.bm.brushes_observers: f(brushes)

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

    def brush_modified_cb(self):
        self.tdw.doc.set_brush(self.app.brush)

