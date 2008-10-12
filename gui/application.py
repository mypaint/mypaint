# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

import gtk, os
from lib import brush

class Application: # singleton
    def __init__(self, share, confpath, loadimage):
        self.confpath = confpath

        datapaths = [share]

        self.datapath = None

        for p in datapaths:
            if os.path.isdir(os.path.join(p, 'brushes')):
                self.datapath = p
                break
        if not self.datapath:
            print 'Default brush collection not found! Searched:'
            print ' '.join(datapaths)
            raise SystemExit

        self.user_brushpath = os.path.join(self.confpath, 'brushes')
        self.stock_brushpath = os.path.join(self.datapath, 'brushes')

        if not os.path.isdir(self.confpath):
            os.mkdir(self.confpath)
            print 'Created', self.confpath
        if not os.path.isdir(self.user_brushpath):
            os.mkdir(self.user_brushpath)

        self.init_brushes()

        self.window_names = 'drawWindow brushSettingsWindow brushSelectionWindow colorSelectionWindow settingsWindow'.split()
        for name in self.window_names:
            module = __import__(name.lower(), globals(), locals(), [])
            window = self.__dict__[name] = module.Window(self)
            self.load_window_position(name, window)

        gtk.accel_map_load(os.path.join(self.confpath, 'accelmap.conf'))

        if loadimage:
            self.drawWindow.open_file(loadimage)

    def init_brushes(self):
        self.brush = brush.Brush(self)
        self.brushes = []
        self.selected_brush = None
        self.brush_selected_callbacks = [self.brush_selected_cb]
        self.contexts = []
        for i in range(10):
            c = brush.Brush(self)
            c.name = 'context%02d' % i
            self.contexts.append(c)
        self.selected_context = None

        # find all brush names to load
        deleted = []
        filename = os.path.join(self.user_brushpath, 'deleted.conf')
        if os.path.exists(filename): 
            for line in open(filename):
                deleted.append(line.strip())
        def listbrushes(path):
            return [filename[:-4] for filename in os.listdir(path) if filename.endswith('.myb')]
        stock_names = listbrushes(self.stock_brushpath)
        user_names =  listbrushes(self.user_brushpath)
        stock_names = [name for name in stock_names if name not in deleted and name not in user_names]
        loadnames_unsorted = user_names + stock_names
        loadnames_sorted = []

        # sort them
        for path in [self.user_brushpath, self.stock_brushpath]:
            filename = os.path.join(path, 'order.conf')
            if not os.path.exists(filename): continue
            for line in open(filename):
                name = line.strip()
                if name in loadnames_sorted: continue
                if name not in loadnames_unsorted: continue
                loadnames_unsorted.remove(name)
                loadnames_sorted.append(name)
        if len(loadnames_unsorted) > 3: 
            # many new brushes, do not disturb user's order
            loadnames = loadnames_sorted + loadnames_unsorted
        else:
            loadnames = loadnames_unsorted + loadnames_sorted

        for name in loadnames:
            # load brushes from disk
            b = brush.Brush(self)
            b.load(name)
            if name.startswith('context'):
                i = int(name[-2:])
                assert i >= 0 and i < 10 # 10 for now...
                self.contexts[i] = b
            else:
                self.brushes.append(b)

        if self.brushes:
            self.select_brush(self.brushes[0])

        self.brush.set_color_hsv((0, 0, 0))

    def save_brushorder(self):
        f = open(os.path.join(self.user_brushpath, 'order.conf'), 'w')
        f.write('# this file saves brushorder\n')
        f.write('# the first one (upper left) will be selected at startup\n')
        for b in self.brushes:
            f.write(b.name + '\n')
        f.close()

    def brush_selected_cb(self, brush):
        "actually set the new brush"
        assert brush is not self.brush # self.brush never gets exchanged
        if brush in self.brushes:
            self.selected_brush = brush
        else:
            #print 'Warning, you have selected a brush not in the list.'
            # TODO: maybe find out parent and set this as selected_brush
            self.selected_brush = None
        if brush is not None:
            self.brush.copy_settings_from(brush)

    def select_brush(self, brush):
        for callback in self.brush_selected_callbacks:
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




