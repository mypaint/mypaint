# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

"select brush window"
import gtk
from lib import brush, document
import tileddrawwidget, pixbuflist

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.app.brush_selected_callbacks.insert(0, self.brush_selected_cb)
        self.app.brush.settings_observers.append(self.brush_modified_cb)
        self.add_accel_group(self.app.accel_group)

        self.set_title('Brush selection')
        self.connect('delete-event', self.app.hide_window_cb)

        vbox = gtk.VBox()
        self.add(vbox)

        self.brushlist = BrushList(self.app)
        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        scroll.add_with_viewport(self.brushlist)
        vbox.pack_start(scroll)

        vbox.pack_start(gtk.HSeparator(), expand=False)

        expander = self.expander = gtk.Expander(label='Edit')
        expander.set_expanded(False)
        vbox.pack_start(expander, expand=False, fill=False)

        hbox = gtk.HBox()
        hbox.set_border_width(8)
        expander.add(hbox)
        self.tdw_doc = document.Document()
        self.tdw = tileddrawwidget.TiledDrawWidget(self.tdw_doc)
        self.tdw.lock_viewport()
        self.tdw.set_size_request(brush.preview_w, brush.preview_h)
        hbox.pack_start(self.tdw, expand=False, fill=False)

        vbox2 = gtk.VBox()
        hbox.pack_end(vbox2, expand=False, fill=False)
        #hbox.properties.padding = 10
        #hbox.set_spacing(10)

        b = gtk.Button('Clear')
        def clear_cb(window):
            self.tdw_doc.clear_layer()
        b.connect('clicked', clear_cb)
        vbox2.pack_start(b, expand=False)

        b = gtk.Button('add as new')
        b.connect('clicked', self.add_as_new_cb)
        vbox2.pack_start(b, expand=False)

        b = gtk.Button('save preview')
        b.connect('clicked', self.update_preview_cb)
        vbox2.pack_start(b, expand=False)

        b = gtk.Button('save settings')
        b.connect('clicked', self.update_settings_cb)
        vbox2.pack_start(b, expand=False)

        b = gtk.Button('delete selected')
        b.connect('clicked', self.delete_selected_cb)
        vbox2.pack_start(b, expand=False)

    def set_preview_pixbuf(self, pixbuf):
        if pixbuf is None:
            self.tdw_doc.clear()
        else:
            self.tdw.doc.load_from_pixbuf(pixbuf)

    def get_preview_pixbuf(self):
        pixbuf = self.tdw.doc.render_as_pixbuf(0, 0, brush.preview_w, brush.preview_h)
        return pixbuf

    def add_as_new_cb(self, window):
        b = brush.Brush(self.app)
        b.copy_settings_from(self.app.brush)
        b.update_preview(self.get_preview_pixbuf())
        self.app.brushes.insert(0, b)
        self.brushlist.update()
        self.app.select_brush(b)
        b.save()
        self.app.save_brushorder()

    def update_preview_cb(self, window):
        pixbuf = self.get_preview_pixbuf()
        b = self.app.selected_brush
        if b is None:
            # no brush selected
            display = gtk.gdk.display_get_default()
            display.beep()
            return
        b.update_preview(pixbuf)
        b.save()
        self.brushlist.update()

    def update_settings_cb(self, window):
        b = self.app.selected_brush
        if b is None:
            # no brush selected
            display = gtk.gdk.display_get_default()
            display.beep()
            return
        b.copy_settings_from(self.app.brush)
        b.save()

    def delete_selected_cb(self, window):
        b = self.app.selected_brush
        if b is None: return

        d = gtk.Dialog("Really delete this brush?",
             self,
             gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
             (gtk.STOCK_YES, gtk.RESPONSE_ACCEPT,
              gtk.STOCK_NO, gtk.RESPONSE_REJECT))
        response = d.run()
        d.destroy()
        if response != gtk.RESPONSE_ACCEPT: return

        self.app.select_brush(None)
        self.app.brushes.remove(b)
        b.delete_from_disk()
        self.brushlist.update()

    def brush_selected_cb(self, brush):
        if brush is None: return
        if brush is self.app.selected_brush:
            # selected same brush twice: load pixmap
            self.set_preview_pixbuf(brush.preview)

    def brush_modified_cb(self):
        self.tdw_doc.set_brush(self.app.brush)


class BrushList(pixbuflist.PixbufList):
    "choose a brush by preview"
    def __init__(self, app):
        self.app = app
        pixbuflist.PixbufList.__init__(self, self.app.brushes, brush.thumb_w, brush.thumb_h, lambda x: x.preview_thumb)
        self.app.brush_selected_callbacks.append(self.brush_selected_cb)

    def on_select(self, brush):
        # keep the color setting
        color = self.app.brush.get_color_hsv()
        brush.set_color_hsv(color)

        # brush changed on harddisk?
        changed = brush.reload_if_changed()
        if changed:
            self.update()

        self.app.select_brush(brush)

    def on_order_change(self):
        self.app.save_brushorder()

    def brush_selected_cb(self, brush):
        self.set_selected(brush)
