# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gettext import gettext as _
import gtk, os
gdk = gtk.gdk
from glob import glob
import pixbuflist
import windowing
from lib import tiledsurface, helpers
N = tiledsurface.N

RESPONSE_SAVE_AS_DEFAULT = 1

class Window(windowing.Dialog):
    def __init__(self, app):
        flags = gtk.DIALOG_DESTROY_WITH_PARENT
        buttons = (_('save as default'), RESPONSE_SAVE_AS_DEFAULT,
                   gtk.STOCK_OK, gtk.RESPONSE_ACCEPT)
        windowing.Dialog.__init__(self, app=app, title=_('Background'),
                                  parent=app.drawWindow, flags=flags,
                                  buttons=buttons)

        #set up window
        self.connect('response', self.on_response)

        notebook = self.nb = gtk.Notebook()
        self.vbox.pack_start(notebook)

        #set up patterns tab
        patterns_scroll = gtk.ScrolledWindow()
        patterns_scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        notebook.append_page(patterns_scroll, gtk.Label(_('Pattern')))

        self.bgl = BackgroundList(self)
        patterns_scroll.add_with_viewport(self.bgl)

        #set up colors tab
        color_vbox = gtk.VBox()
        notebook.append_page(color_vbox, gtk.Label(_('Color')))

        self.cs = gtk.ColorSelection()
        self.cs.connect('color-changed', self.color_changed_cb)
        color_vbox.pack_start(self.cs, expand=True)

        b = gtk.Button(_('add color to patterns'))
        b.connect('clicked', self.add_color_to_patterns_cb)
        color_vbox.pack_start(b, expand=False)

    def on_response(self, dialog, response, *args):
        if response == RESPONSE_SAVE_AS_DEFAULT:
             self.save_as_default_cb()
        elif response == gtk.RESPONSE_ACCEPT:
            self.hide()

    def color_changed_cb(self, widget):
        rgb = self.cs.get_current_color()
        rgb = rgb.red, rgb.green, rgb.blue
        rgb = [int(x / 65535.0 * 255.0) for x in rgb] 
        pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, N, N)
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        arr[:,:] = rgb
        self.set_background(pixbuf)

    def save_as_default_cb(self):
        pixbuf = self.current_background_pixbuf
        pixbuf.save(os.path.join(self.app.confpath, 'backgrounds', 'default.png'), 'png')
        self.hide()

    def set_background(self, pixbuf):
        doc = self.app.doc.model
        doc.set_background(pixbuf)
        self.current_background_pixbuf = pixbuf

    def add_color_to_patterns_cb(self, widget):
        pixbuf = self.current_background_pixbuf
        i = 1
        while 1:
            filename = os.path.join(self.app.confpath, 'backgrounds', 'color%02d.png' % i)
            if not os.path.exists(filename):
                break
            i += 1
        pixbuf.save(filename, 'png')
        self.bgl.backgrounds.append(pixbuf)
        self.bgl.update()
        self.bgl.set_selected(pixbuf)
        self.nb.set_current_page(0)

class BackgroundList(pixbuflist.PixbufList):
    def __init__(self, win):
        self.app = win.app
        self.win = win

        stock_path = os.path.join(self.app.datapath, 'backgrounds')
        user_path  = os.path.join(self.app.confpath, 'backgrounds')
        if not os.path.isdir(user_path):
            os.mkdir(user_path)
        self.backgrounds = []

        def listdir(path):
            l = glob(os.path.join(path, '*.png')) + glob(os.path.join(path, '*/*.png'))
            l.sort(key=os.path.getmtime)
            return l

        files = listdir(stock_path)
        files.sort()
        files += listdir(user_path)

        for filename in files:
            if not filename.lower().endswith('.png'):
                continue
            pixbuf = gdk.pixbuf_new_from_file(filename)

            # error checking
            def error(msg):
                self.app.message_dialog(msg, type = gtk.MESSAGE_WARNING, flags = gtk.DIALOG_MODAL)
            if pixbuf.get_has_alpha():
                error(_('The background %s was ignored because it has an alpha channel. Please remove it.') % filename)
                continue
            w, h = pixbuf.get_width(), pixbuf.get_height()
            if w % N != 0 or h % N != 0 or w == 0 or h == 0:
                error(_('The background %s was ignored because it has the wrong size. Only (N*%d)x(M*%d) is supported.') % (filename, N, N))
                continue

            if os.path.basename(filename).lower() == 'default.png':
                self.win.set_background(pixbuf)
                continue

            self.backgrounds.append(pixbuf)

        pixbuflist.PixbufList.__init__(self, self.backgrounds, N, N, pixbuffunc=self.pixbuf_scaler)
        self.dragging_allowed = False

        self.pixbufs_scaled = {}

    def pixbuf_scaler(self, pixbuf):
        w, h = pixbuf.get_width(), pixbuf.get_height()
        if w == N and h == N:
            return pixbuf
        if pixbuf not in self.pixbufs_scaled:
            scaled = helpers.pixbuf_thumbnail(pixbuf, N, N)
            # add plus sign
            self.app.pixmaps.plus.composite(scaled, 0, 0, N, N, 0, 0, 1.0, 1.0, gdk.INTERP_BILINEAR, 255)

            self.pixbufs_scaled[pixbuf] = scaled

        return self.pixbufs_scaled[pixbuf]

    def on_select(self, pixbuf):
        self.win.set_background(pixbuf)
