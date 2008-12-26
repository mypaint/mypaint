# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

import gtk, os
gdk = gtk.gdk
import tileddrawwidget, pixbuflist
from lib import tiledsurface, document

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.add_accel_group(self.app.accel_group)

        self.set_title('Background')
        self.connect('delete-event', self.app.hide_window_cb)

        vbox = gtk.VBox()
        self.add(vbox)

        nb = gtk.Notebook()
        vbox.pack_start(nb)

        self.bgl = BackgroundList(self.app)
        nb.append_page(self.bgl, gtk.Label('Pattern'))

        self.cs = gtk.ColorSelection()
        self.cs.connect('color-changed', self.color_changed_cb)
        nb.append_page(self.cs, gtk.Label('Color'))

        hbox = gtk.HBox()
        vbox.pack_start(hbox, expand=False)

        b = gtk.Button('save as default')
        b.connect('clicked', self.save_as_default_cb)
        hbox.pack_start(b)

    def color_changed_cb(self, widget):
        rgb = self.cs.get_current_color()
        rgb = rgb.red, rgb.green, rgb.blue
        rgb = [int(x / 65535.0 * 255.0) for x in rgb] 
        doc = self.app.drawWindow.doc
        doc.set_background(rgb)

    def save_as_default_cb(self, widget):
        print 'TODO'


class BackgroundObject:
    pass


class BackgroundList(pixbuflist.PixbufList):
    def __init__(self, app):
        self.app = app

        stock_path = os.path.join(self.app.datapath, 'backgrounds')
        user_path  = os.path.join(self.app.confpath, 'backgrounds')
        if not os.path.isdir(user_path):
            os.mkdir(user_path)
        self.backgrounds = []

        def listdir(path):
            l = [os.path.join(path, filename) for filename in os.listdir(path)]
            l.sort()
            return l

        for filename in listdir(user_path) + listdir(stock_path):
            if not filename.lower().endswith('.png'):
                continue
            obj = BackgroundObject()
            obj.pixbuf = gdk.pixbuf_new_from_file(filename)
            self.backgrounds.append(obj)

        pixbuflist.PixbufList.__init__(self, self.backgrounds, tiledsurface.N, tiledsurface.N)

    def on_select(self, bg):
        doc = self.app.drawWindow.doc
        doc.set_background(bg.pixbuf)
