# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os
import gtk, gobject
gdk = gtk.gdk
from lib import brush
import filehandling, keyboard
from brushselectionwindow import DEFAULT_BRUSH_GROUP

class Application: # singleton
    """
    This class serves as a global container for everything that needs
    to be shared in the GUI. Its constructor is the last part of the
    initialization, called by main.py or by the testing scripts.
    """
    def __init__(self, datapath, confpath, filenames):
        self.confpath = confpath
        self.datapath = datapath

        self.ui_manager = gtk.UIManager()

        # if we are not installed, use the the icons from the source
        theme = gtk.icon_theme_get_default()
        themedir_src = os.path.join(self.datapath, 'desktop/icons')
        theme.prepend_search_path(themedir_src)
        if not theme.has_icon('mypaint'):
            print 'Warning: Where have all my icons gone?'
            print 'Theme search path:', theme.get_search_path()
        gtk.window_set_default_icon_name('mypaint')

        gdk.set_program_class('MyPaint')

        self.pixmaps = PixbufDirectory(os.path.join(self.datapath, 'pixmaps'))
        self.cursor_color_picker = gdk.Cursor(gdk.display_get_default(), self.pixmaps.cursor_color_picker, 1, 30)

        self.user_brushpath = os.path.join(self.confpath, 'brushes')
        self.stock_brushpath = os.path.join(self.datapath, 'brushes')

        if not os.path.isdir(self.confpath):
            os.mkdir(self.confpath)
            print 'Created', self.confpath
        if not os.path.isdir(self.user_brushpath):
            os.mkdir(self.user_brushpath)

        self.init_brushes()
        self.kbm = keyboard.KeyboardManager()
        self.filehandler = filehandling.FileHandler(self)

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

        self.brushSelectionWindow.disable_selection_callback = False # FIXME: huh?
        self.kbm.start_listening()
        self.filehandler.doc = self.drawWindow.doc
        self.filehandler.filename = None
        gtk.accel_map_load(os.path.join(self.confpath, 'accelmap.conf'))

        # TODO: remember last selected brush, and select one at frist startup
        #if self.brushes:
        #    self.select_brush(self.brushes[0])
        self.brush.set_color_hsv((0, 0, 0))

        def at_application_start(*trash):
            if filenames:
                #open the first file, no matter how many that has been specified
                fn = filenames[0].replace('file:///', '/') # some filebrowsers do this (should only happen with outdated mypaint.desktop)
                self.filehandler.open_file(fn)

        gobject.idle_add(at_application_start)

    def init_brushes(self):
        self.brush = brush.Brush(self)
        self.selected_brush = self.brush
        self.selected_brush_observers = []
        self.contexts = []
        for i in range(10):
            c = brush.Brush(self)
            c.name = 'context%02d' % i
            self.contexts.append(c)
        self.selected_context = None

        brush_by_name = {}
        def get_brush(name):
            if name not in brush_by_name:
                b = brush.Brush(self)
                b.load(name)
                brush_by_name[name] = b
            return brush_by_name[name]

        # maybe this should be save/loaded too?
        self.brush_by_device = {}

        def read_groups(filename):
            groups = {}
            if os.path.exists(filename):
                curr_group = DEFAULT_BRUSH_GROUP
                groups[curr_group] = []
                for line in open(filename):
                    name = line.strip()
                    if name.startswith('#'):
                        continue
                    if name.startswith('Group: '):
                        curr_group = unicode(name[7:], 'utf-8')
                        if curr_group not in groups:
                            groups[curr_group] = []
                        continue
                    try:
                        b = get_brush(name)
                    except IOError, e:
                        print e, '(removed from group)'
                        continue
                    if b in groups[curr_group]:
                        print filename + ': Warning: brush appears twice in the same group, ignored'
                        continue
                    groups[curr_group].append(b)
            return groups

        # tree-way-merge of brush groups (for upgrading)
        base  = read_groups(os.path.join(self.user_brushpath,  'order_default.conf'))
        our   = read_groups(os.path.join(self.user_brushpath,  'order.conf'))
        their = read_groups(os.path.join(self.stock_brushpath, 'order.conf'))

        if base == their:
            self.brushgroups = our
        else:
            print 'Merging upstream brush changes into your collection.'
            groups = set(base).union(our).union(their)
            for group in groups:
                # treat the non-existing groups as if empty
                base_brushes = base.setdefault(group, [])
                our_brushes = our.setdefault(group, [])
                their_brushes = their.setdefault(group, [])
                # add new brushes
                insert_index = 0
                for b in their_brushes:
                    if b in our_brushes:
                        insert_index = our_brushes.index(b) + 1
                    else:
                        if b in their_brushes:
                            our_brushes.insert(insert_index, b)
                            insert_index += 1
                # remove deleted brushes
                for b in base_brushes:
                    if b not in their_brushes and b in our_brushes:
                        our_brushes.remove(b)
                # remove empty groups
                if not our_brushes:
                    del our[group]
            # finish
            self.brushgroups = our
            self.save_brushorder()
            data = open(os.path.join(self.stock_brushpath, 'order.conf')).read()
            open(os.path.join(self.user_brushpath,  'order_default.conf'), 'w').write(data)
                
        # handle brushes that are in the brush directory, but not in any group
        def listbrushes(path):
            return [filename[:-4] for filename in os.listdir(path) if filename.endswith('.myb')]
        for name in listbrushes(self.stock_brushpath) + listbrushes(self.user_brushpath):
            b = get_brush(name)
            if name.startswith('context'):
                i = int(name[-2:])
                self.contexts[i] = b
                continue
            if not [True for group in our.itervalues() if b in group]:
                self.brushgroups.setdefault(DEFAULT_BRUSH_GROUP, [])
                self.brushgroups[DEFAULT_BRUSH_GROUP].insert(0, b)

        # clean up legacy stuff
        fn = os.path.join(self.user_brushpath, 'deleted.conf')
        if os.path.exists(fn):
            os.path.remove(fn)

    def save_brushorder(self):
        f = open(os.path.join(self.user_brushpath, 'order.conf'), 'w')
        f.write('# this file saves brush groups and order\n')
        for group, brushes in self.brushgroups.iteritems():
            f.write('Group: %s\n' % group.encode('utf-8'))
            for b in brushes:
                f.write(b.name + '\n')
        f.close()

    def select_brush(self, brush):
        assert brush is not self.brush # self.brush never gets exchanged
        self.selected_brush = brush
        if brush is not None:
            self.brush.copy_settings_from(brush)

        for callback in self.selected_brush_observers:
            callback(brush)

    def hide_window_cb(self, window, event):
        # used by some of the windows
        window.hide()
        return True

    def save_gui_config(self):
        gtk.accel_map_save(os.path.join(self.confpath, 'accelmap.conf'))
        self.save_window_positions()
        
    def save_window_positions(self):
        f = open(os.path.join(self.confpath, 'windowpos.conf'), 'w')
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
            for line in open(os.path.join(self.confpath, 'windowpos.conf')):
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
            pixbuf = gdk.pixbuf_new_from_file(os.path.join(self.dirname, name + '.png'))
            self.cache[name] = pixbuf
        return self.cache[name]
