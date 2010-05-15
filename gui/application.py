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
import filehandling, keyboard, brushmanager, windowing, document

## TODO: move all window sizing stuff to windowing module
##       while the keys below refer to fields in the Application, this will
##       be ugly, however.

# The window geometry that will be assumed at save time.
SAVE_POS_GRAVITY = gdk.GRAVITY_NORTH_WEST

# Window names and their default positions and visibilities. They interrelate,
# so define them together.
WINDOW_DEFAULTS = {
    'drawWindow':           (None, True), # initial geom overridden, but set vis
    # Colour choosers go in the top-right quadrant by default
    'colorSamplerWindow':   ("280x350-50+75",  True ),
    'colorSelectionWindow': ("-145+100",        False), # positions strangely
    # Brush-related dialogs go in the bottom right by default
    'brushSelectionWindow': ("280x350-50-50",  True ),
    'brushSettingsWindow':  ("400x250-345-50",        False),
    # Preferences go "inside" the main window (TODO: just center-on-parent?)
    'preferencesWindow':    ("+200+100",       False),
    # Layer details in the bottom-left quadrant.
    'layersWindow':         ("200x300+50-50",  False),
    'backgroundWindow':     ("500x400+265-50", False),
}

# The main drawWindow gets centre stage, sized around the default set of
# visible utility windows. Sizing constraints for this:
CENTRE_STAGE_CONSTRAINTS = (\
    166, # left: leave some space for icons (but: OSX?)
    50,  # top: avoid panel
    75,  # bottom: avoid panel
    10,  # right (small screens): ensure close button not covered at least
    375, # right (big screens): don't cover brushes or colour
    # "big screens" are anything 3 times wider than the big-right margin:
    ## > Size. The Center Stage content should be at least twice as wide as
    ## > whatever is in its side margins, and twice as tall as its top and
    ## > bottom margins. (The user may change its size, but this is how it
    ## > should be when the user first sees it.)
    ## >     -- http://designinginterfaces.com/Center_Stage
)


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

        self.preferences = {}
        self.brushmanager = brushmanager.BrushManager(join(datapath, 'brushes'), join(confpath, 'brushes'))
        self.kbm = keyboard.KeyboardManager()
        self.filehandler = filehandling.FileHandler(self)
        self.doc = document.Document(self)

        self.brush.copy_settings_from(self.brushmanager.selected_brush)
        self.brush.set_color_hsv((0, 0, 0))
        self.brushmanager.selected_brush_observers.append(self.brush_selected_cb)

        self.user_subwindows = windowing.UserSubWindows(self)
        self.window_names = ['drawWindow'] \
          + [n for n in WINDOW_DEFAULTS.keys() if n != 'drawWindow']
        for name in self.window_names:
            module = __import__(name.lower(), globals(), locals(), [])
            window = self.__dict__[name] = module.Window(self)
            using_default = self.load_window_position(name, window)
            if using_default:
                if name == 'drawWindow':
                    def on_map_event(win, ev, dw):
                        windowing.centre_stage(win, *CENTRE_STAGE_CONSTRAINTS)
                else:
                    def on_map_event(win, ev, dw):
                        windowing.move_to_monitor_of(win, dw)
                window.connect('map-event', on_map_event, self.drawWindow)

        self.kbm.start_listening()
        self.filehandler.doc = self.doc
        self.filehandler.filename = None
        gtk.accel_map_load(join(self.confpath, 'accelmap.conf'))

        def at_application_start(*trash):
            if filenames:
                # Open only the first file, no matter how many has been specified
                # If the file does not exist just set it as the file to save to
                fn = filenames[0].replace('file:///', '/') # some filebrowsers do this (should only happen with outdated mypaint.desktop)
                if not os.path.exists(fn):
                    self.filehandler.filename = fn
                else:
                    self.filehandler.open_file(fn)

        gobject.idle_add(at_application_start)

    def save_settings(self):
        """Saves the current settings to persistent storage."""
        def save_config():
            f = open(join(self.confpath, 'settings.conf'), 'w')
            p = self.preferences
            print >>f, 'global_pressure_mapping =', p['input.global_pressure_mapping']
            print >>f, 'save_scrap_prefix =', repr(p['saving.scrap_prefix'])
            print >>f, 'input_devices_mode =', repr(p['input.device_mode'])
            f.close()
        save_config()

    def load_settings(self):
        '''Loads the settings from peristent storage. Uses defaults if
        not explicitly configured'''
        def get_config():
            dummyobj = {}
            tmpdict = {}
            settingspath = join(self.confpath, 'settings.conf')
            if os.path.exists(settingspath):
                exec open(settingspath) in dummyobj
                tmpdict['saving.scrap_prefix'] = dummyobj['save_scrap_prefix']
                tmpdict['input.device_mode'] = dummyobj['input_devices_mode']
                tmpdict['input.global_pressure_mapping'] = dummyobj['global_pressure_mapping']
            return tmpdict

        DEFAULT_CONFIG = {
            'saving.scrap_prefix': 'scrap',
            'input.device_mode': 'screen',
            'input.global_pressure_mapping': [(0.0, 1.0), (1.0, 0.0)],
        }
        self.preferences = DEFAULT_CONFIG
        self.preferences.update(get_config())

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
            gravity = window.get_gravity()
            if gravity != SAVE_POS_GRAVITY:
                # Then it was never mapped, and must still be using the
                # defaults. Don't save position (which'd be wrong anyway).
                continue
            if hasattr(window, 'geometry_before_fullscreen'):
                x, y, w, h = window.geometry_before_fullscreen
            visible = window in self.user_subwindows.windows \
                or window.get_property('visible') 
            f.write('%s %s %d %d %d %d\n' % (name, visible, x, y, w, h))

    def load_window_position(self, name, window):
        geometry, visible = WINDOW_DEFAULTS.get(name, (None, False))
        using_default = True
        try:
            for line in open(join(self.confpath, 'windowpos.conf')):
                if line.startswith(name):
                    parts = line.split()
                    visible = parts[1] == 'True'
                    x, y, w, h = [int(i) for i in parts[2:2+4]]
                    geometry = '%dx%d+%d+%d' % (w, h, x, y)
                    using_default = False
                    break
        except IOError:
            pass
        # Initial gravities can be all over the place. Fix aberrant ones up
        # when the windows are safely on-screen so their position can be
        # saved sanely. Doing this only when the window's mapped means that the
        # window position stays where we specified in the defaults, and the
        # window manager (should) compensate for us without the position
        # changing.
        if geometry is not None:
            window.parse_geometry(geometry)
            if using_default:
                initial_gravity = window.get_gravity()
                if initial_gravity != SAVE_POS_GRAVITY:
                    def fix_gravity(w, event):
                        if w.get_gravity() != SAVE_POS_GRAVITY:
                            w.set_gravity(SAVE_POS_GRAVITY)
                    window.connect("map-event", fix_gravity)
        if visible:
            window.show_all()
        return using_default

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
            try:
                pixbuf = gdk.pixbuf_new_from_file(join(self.dirname, name + '.png'))
            except gobject.GError, e:
                raise AttributeError, str(e)
            self.cache[name] = pixbuf
        return self.cache[name]
