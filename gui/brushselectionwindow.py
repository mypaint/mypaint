# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"select brush window"
import gtk
gdk = gtk.gdk
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
        def set_hint(widget):
            self.window.set_type_hint(gdk.WINDOW_TYPE_HINT_UTILITY)
        self.connect("realize", set_hint)

        # TODO: evaluate glade/gazpacho, the code below is getting scary

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

        vbox2 = gtk.VBox()
        hbox.pack_start(vbox2, expand=False, fill=False)

        doc = document.Document()
        self.tdw = tileddrawwidget.TiledDrawWidget(doc)
        self.tdw.set_size_request(brush.preview_w, brush.preview_h)
        vbox2.pack_start(self.tdw, expand=False, fill=False)

        b = gtk.Button('Clear')
        def clear_cb(window):
            self.tdw.doc.clear_layer()
        b.connect('clicked', clear_cb)
        vbox2.pack_start(b, expand=False, padding=5)


        #vbox2a = gtk.VBox()
        #hbox.pack_end(vbox2a, expand=True, fill=True, padding=5)
        #l = self.brush_name_label = gtk.Label()
        #l.set_justify(gtk.JUSTIFY_LEFT)
        #vbox2a.pack_start(l, expand=False)
        #tv = self.brush_info_textview = gtk.TextView()
        #vbox2a.pack_start(tv, expand=True)

        vbox2b = gtk.VBox()
        hbox.pack_end(vbox2b, expand=False, fill=False)

        l = self.brush_name_label = gtk.Label()
        l.set_justify(gtk.JUSTIFY_LEFT)
        l.set_text('(no name)')
        vbox2b.pack_start(l, expand=False)

        b = gtk.Button('add as new')
        b.connect('clicked', self.add_as_new_cb)
        vbox2b.pack_start(b, expand=False)

        b = gtk.Button('rename...')
        b.connect('clicked', self.rename_cb)
        vbox2b.pack_start(b, expand=False)

        b = gtk.Button('save preview')
        b.connect('clicked', self.update_preview_cb)
        vbox2b.pack_start(b, expand=False)

        b = gtk.Button('save settings')
        b.connect('clicked', self.update_settings_cb)
        vbox2b.pack_start(b, expand=False)

        b = gtk.Button('delete selected')
        b.connect('clicked', self.delete_selected_cb)
        vbox2b.pack_start(b, expand=False)

    def set_preview_pixbuf(self, pixbuf):
        if pixbuf is None:
            self.tdw.doc.clear()
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

    def rename_cb(self, window):
        b = self.app.selected_brush
        if b is None or not b.name:
            display = gtk.gdk.display_get_default()
            display.beep()
            return

        d = gtk.Dialog("Rename Brush",
                       self,
                       gtk.DIALOG_MODAL,
                       (gtk.STOCK_CANCEL, gtk.RESPONSE_REJECT,
                        gtk.STOCK_OK, gtk.RESPONSE_ACCEPT))

        hbox = gtk.HBox()
        d.vbox.pack_start(hbox)
        hbox.pack_start(gtk.Label('Name'))

        e = gtk.Entry()
        e.set_text(b.name)
        e.select_region(0, len(b.name))
        def responseToDialog(entry, dialog, response):  
            dialog.response(response)  
        e.connect("activate", responseToDialog, d, gtk.RESPONSE_ACCEPT)  

        hbox.pack_start(e)
        d.vbox.show_all()
        if d.run() == gtk.RESPONSE_ACCEPT:
            new_name = e.get_text()
            print 'renaming brush', repr(b.name), '-->', repr(new_name)
            if [True for x in self.app.brushes if x.name == new_name]:
                print 'Target already exists!'
                display = gtk.gdk.display_get_default()
                display.beep()
                d.destroy()
                return
            b.delete_from_disk()
            b.name = new_name
            b.save()
            self.app.select_brush(b)
            self.app.save_brushorder()
        d.destroy()

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
        name = brush.name
        if name is None:
            name = '(no name)'
        #else:
        #    name += '.myb'
        self.brush_name_label.set_text(name)
        if brush is self.app.selected_brush:
            # selected same brush twice: load pixmap
            self.set_preview_pixbuf(brush.preview)

    def brush_modified_cb(self):
        self.tdw.doc.set_brush(self.app.brush)


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
