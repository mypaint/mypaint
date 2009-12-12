# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os
from os.path import join
import gtk, gobject
gdk = gtk.gdk
from lib import brush
import filehandling, keyboard, brushmanager

class Application: # singleton
    """
    This class serves as a global container for everything that needs
    to be shared in the GUI. Its constructor is the last part of the
    initialization, called by main.py or by the testing scripts.
    """
    def __init__(self, datapath, confpath, filenames):
        self.confpath = confpath
        self.datapath = datapath

        if not os.path.isdir(self.confpath):
            os.mkdir(self.confpath)
            print 'Created', self.confpath

        self.ui_manager = gtk.UIManager()

        # if we are not installed, use the the icons from the source
        theme = gtk.icon_theme_get_default()
        themedir_src = join(self.datapath, 'desktop/icons')
        theme.prepend_search_path(themedir_src)
        if not theme.has_icon('mypaint'):
            print 'Warning: Where have all my icons gone?'
            print 'Theme search path:', theme.get_search_path()
        gtk.window_set_default_icon_name('mypaint')

        gdk.set_program_class('MyPaint')

        self.pixmaps = PixbufDirectory(join(self.datapath, 'pixmaps'))
        self.cursor_color_picker = gdk.Cursor(gdk.display_get_default(), self.pixmaps.cursor_color_picker, 1, 30)

        # unmanaged main brush; always the same instance (we can attach settings_observers)
        # this brush is where temporary changes (color, size...) happen
        self.brush = brush.Brush()

        self.brushmanager = brushmanager.BrushManager(join(datapath, 'brushes'), join(confpath, 'brushes'))
        self.kbm = keyboard.KeyboardManager()
        self.filehandler = filehandling.FileHandler(self)

        self.brushmanager.selected_brush_observers.append(self.brush_selected_cb)

        self.window_names = '''
        drawWindow
        brushSettingsWindow
        brushSelectionWindow
        colorSelectionWindow
        colorSamplerWindow
        settingsWindow
        backgroundWindow
        '''.split()
        for name in self.window_names:
            module = __import__(name.lower(), globals(), locals(), [])
            window = self.__dict__[name] = module.Window(self)
            if name != 'drawWindow':
                def set_hint(widget):
                    widget.window.set_type_hint(gdk.WINDOW_TYPE_HINT_UTILITY)
                window.connect("realize", set_hint)
            self.load_window_position(name, window)

        self.kbm.start_listening()
        self.filehandler.doc = self.drawWindow.doc
        self.filehandler.filename = None
        gtk.accel_map_load(join(self.confpath, 'accelmap.conf'))

        def at_application_start(*trash):
            if filenames:
                #open the first file, no matter how many that has been specified
                fn = filenames[0].replace('file:///', '/') # some filebrowsers do this (should only happen with outdated mypaint.desktop)
                self.filehandler.open_file(fn)

        gobject.idle_add(at_application_start)

    def brush_selected_cb(self, b):
        assert b is not self.brush
        if b:
            self.brush.copy_settings_from(b)

    def hide_window_cb(self, window, event):
        # used by some of the windows
        window.hide()
        return True

    def save_gui_config(self):
        gtk.accel_map_save(join(self.confpath, 'accelmap.conf'))
        self.save_window_positions()
        
    def save_window_positions(self):
        f = open(join(self.confpath, 'windowpos.conf'), 'w')
        f.write('# name visible x y width height\n')
        for name in self.window_names:
            window = self.__dict__[name]
            x, y = window.get_position()
            w, h = window.get_size()
            if hasattr(window, 'geometry_before_fullscreen'):
                x, y, w, h = window.geometry_before_fullscreen
            visible = window.get_property('visible')
            f.write('%s %s %d %d %d %d\n' % (name, visible, x, y, w, h))

    def load_window_position(self, name, window):
        try:
            for line in open(join(self.confpath, 'windowpos.conf')):
                if line.startswith(name):
                    parts = line.split()
                    visible = parts[1] == 'True'
                    x, y, w, h = [int(i) for i in parts[2:2+4]]
                    window.parse_geometry('%dx%d+%d+%d' % (w, h, x, y))
                    if visible or name == 'drawWindow':
                        window.show_all()
                    return
        except IOError:
            pass

        if name == 'brushSelectionWindow':
            window.parse_geometry('300x500')

        # default visibility setting
        if name in 'drawWindow brushSelectionWindow colorSelectionWindow'.split():
            window.show_all()

    def message_dialog(self, text, type=gtk.MESSAGE_INFO, flags=0):
        """utility function to show a message/information dialog"""
        d = gtk.MessageDialog(self.drawWindow, flags=flags, buttons=gtk.BUTTONS_OK, type=type)
        d.set_markup(text)
        d.run()
        d.destroy()

class PixbufDirectory:
    def __init__(self, dirname):
        self.dirname = dirname
        self.cache = {}

    def __getattr__(self, name):
        if name not in self.cache:
            pixbuf = gdk.pixbuf_new_from_file(join(self.dirname, name + '.png'))
            self.cache[name] = pixbuf
        return self.cache[name]
