# This file is part of MyPaint.
# Copyright (C) 2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os
from glob import glob

import gui.pygtkcompat
from gettext import gettext as _
import gtk
from gtk import gdk

import pixbuflist
import windowing
from lib import tiledsurface
from lib import helpers

N = tiledsurface.N

RESPONSE_SAVE_AS_DEFAULT = 1

class Window(windowing.Dialog):
    def __init__(self, app):
        flags = gtk.DIALOG_DESTROY_WITH_PARENT
        buttons = (_('Save as Default'), RESPONSE_SAVE_AS_DEFAULT,
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

        def lazy_init(*ignored):
            if not self.bgl.initialized:
                self.bgl.initialize()
        self.connect("realize", lazy_init)

        #set up colors tab
        color_vbox = gtk.VBox()
        notebook.append_page(color_vbox, gtk.Label(_('Color')))

        self.cs = gtk.ColorSelection()
        self.cs.connect('color-changed', self.color_changed_cb)
        color_vbox.pack_start(self.cs, expand=True)

        b = gtk.Button(_('Add color to Patterns'))
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
        pixbuf = gui.pygtkcompat.gdk.pixbuf.new(gdk.COLORSPACE_RGB, False,
                                                8, N, N)
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        arr[:,:] = rgb
        self.set_background(pixbuf)

    def save_as_default_cb(self):
        pixbuf = self.current_background_pixbuf
        path = os.path.join(self.app.user_datapath,
                            'backgrounds', 'default.png')
        gui.pygtkcompat.gdk.pixbuf.save(pixbuf, path, 'png')
        self.hide()

    def set_background(self, pixbuf):
        doc = self.app.doc.model
        doc.set_background(pixbuf, make_default=True)
        self.current_background_pixbuf = pixbuf

    def add_color_to_patterns_cb(self, widget):
        pixbuf = self.current_background_pixbuf
        i = 1
        while 1:
            filename = os.path.join(self.app.user_datapath,
                                    'backgrounds', 'color%02d.png' % i)
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
        pixbuflist.PixbufList.__init__(self, None, N, N, pixbuffunc=self.pixbuf_scaler)
        self.app = win.app
        self.win = win

        self.dragging_allowed = False

        stock_path = os.path.join(self.app.datapath, 'backgrounds')
        user_path  = os.path.join(self.app.user_datapath, 'backgrounds')
        if not os.path.isdir(user_path):
            os.mkdir(user_path)

        def listdir(path):
            l = glob(os.path.join(path, '*.png')) + glob(os.path.join(path, '*/*.png'))
            l.sort(key=os.path.getmtime)
            return l

        self.background_files = listdir(stock_path)
        self.background_files.sort()
        self.background_files += listdir(user_path)

        # Load default background
        defaults = []
        for filename in reversed(self.background_files):
            if os.path.basename(filename).lower() == 'default.png':
                defaults.append(filename)
                self.background_files.remove(filename)
        pixbuf = self.load_pixbufs(defaults)[0]
        self.win.set_background(pixbuf)

        self.pixbufs_scaled = {}
        # Lazily loaded by self.initialize()
        self.backgrounds = []

    @property
    def initialized(self):
        return len(self.backgrounds) != 0

    def initialize(self):
        self.backgrounds = self.load_pixbufs(self.background_files)
        self.set_itemlist(self.backgrounds)

    def load_pixbufs(self, files):

        pixbufs = []
        load_errors = []
        for filename in files:
            if not filename.lower().endswith('.png'):
                continue
            try:
                pixbuf = gdk.pixbuf_new_from_file(filename)
            except Exception, ex:
                print ex
                load_errors.append(
                    _('Gdk-Pixbuf couldn\'t load "{filename}", and reported "{error}"').format(
                    filename=filename, error=repr(ex)))
                continue
            supported = True
            if pixbuf.get_has_alpha():
                load_errors.append(
                    _('"%s" has an alpha channel. Background images with '
                      'transparency are not supported.')
                    % filename)
                supported = False
            w, h = pixbuf.get_width(), pixbuf.get_height()
            if w % N != 0 or h % N != 0 or w == 0 or h == 0:
                load_errors.append(
                    _('{filename} has an unsupported size. Background images '
                      'must have widths and heights which are multiples '
                      'of {number} pixels.').format(filename=filename, number=N))
                supported = False
            if not supported:
                continue

            pixbufs.append(pixbuf)

        if load_errors:
            msg = "\n\n".join(load_errors)
            self.app.message_dialog(
                text=_("One or more backgrounds could not be loaded"),
                title=_("Error loading backgrounds"),
                secondary_text=_("Please remove the unloadable files, or "
                                 "check your libgdkpixbuf installation."),
                long_text=msg,
                type=gtk.MESSAGE_WARNING,
                flags=gtk.DIALOG_MODAL)

        return pixbufs

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
