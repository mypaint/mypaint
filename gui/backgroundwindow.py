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

        #vbox.pack_start(gtk.Label('Choose a background pattern:'), expand=False)

        self.bgl = BackgroundList(self.app)
        vbox.pack_start(self.bgl, padding=5)

    def brush_modified_cb(self):
        self.tdw.doc.set_brush(self.app.brush)

class BackgroundObject:
    pass


class BackgroundList(pixbuflist.PixbufList):
    "choose a brush by preview"
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
        #doc.set_background((123, 0, 50))
        doc.set_background(bg.pixbuf)
