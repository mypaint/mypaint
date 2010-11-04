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
from lib import brush, helpers, mypaintlib
import filehandling, keyboard, brushmanager, windowing, document
import colorhistory

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
    'colorSamplerWindow':   ("220x220-50+100",  False),
    'colorSelectionWindow': ("-250+150",        True), # positions strangely
    # Brush-related dialogs go in the bottom right by default
    'brushSelectionWindow': ("240x350-50-50",   True ),
    'brushSettingsWindow':  ("400x400-325-50",  False),
    # Preferences go "inside" the main window (TODO: just center-on-parent?)
    'preferencesWindow':    ("+200+100",        False),
    # Layer details in the bottom-left quadrant.
    'layersWindow':         ("200x300+50-50",   False),
    'backgroundWindow':     ("500x400-275+75", False),
    # Debug menu
    'inputTestWindow':      (None, False),
}

# The main drawWindow gets centre stage, sized around the default set of
# visible utility windows. Sizing constraints for this:
CENTRE_STAGE_CONSTRAINTS = (\
    166, # left: leave some space for icons (but: OSX?)
    50,  # top: avoid panel
    75,  # bottom: avoid panel
    10,  # right (small screens): ensure close button not covered at least
    220, # right (big screens): don't overlap brushes or colour (much)
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

        # if we are not installed, use the icons from the source
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
        self.load_settings()

        self.brushmanager = brushmanager.BrushManager(join(datapath, 'brushes'), join(confpath, 'brushes'), self)
        self.kbm = keyboard.KeyboardManager()
        self.filehandler = filehandling.FileHandler(self)
        self.doc = document.Document(self)

        self.set_current_brush(self.brushmanager.selected_brush)
        self.brush.set_color_hsv((0, 0, 0))
        self.brushmanager.selected_brush_observers.append(self.brush_selected_cb)
        self.init_brush_adjustments()

        self.ch = colorhistory.ColorHistory(self)

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

            self.apply_settings()
            if not self.pressure_devices:
                print 'No pressure sensitive devices found.'
            self.drawWindow.present()

        gobject.idle_add(at_application_start)

    def save_settings(self):
        """Saves the current settings to persistent storage."""
        def save_config():
            settingspath = join(self.confpath, 'settings.json')
            jsonstr = helpers.json_dumps(self.preferences)
            f = open(settingspath, 'w')
            f.write(jsonstr)
            f.close()
        self.brushmanager.save_brushes_for_devices()
        save_config()

    def apply_settings(self):
        """Applies the current settings."""
        self.update_input_mapping()
        self.update_input_devices()
        try:
            self.preferencesWindow.update_ui()
        except AttributeError:
            pass

    def load_settings(self):
        '''Loads the settings from persistent storage. Uses defaults if
        not explicitly configured'''
        def get_legacy_config():
            dummyobj = {}
            tmpdict = {}
            settingspath = join(self.confpath, 'settings.conf')
            if os.path.exists(settingspath):
                exec open(settingspath) in dummyobj
                tmpdict['saving.scrap_prefix'] = dummyobj['save_scrap_prefix']
                tmpdict['input.device_mode'] = dummyobj['input_devices_mode']
                tmpdict['input.global_pressure_mapping'] = dummyobj['global_pressure_mapping']
            return tmpdict
        def get_json_config():
            settingspath = join(self.confpath, 'settings.json')
            jsonstr = open(settingspath).read()
            return helpers.json_loads(jsonstr)

        DEFAULT_CONFIG = {
            'saving.scrap_prefix': '~/MyPaint/scrap',
            'input.device_mode': 'screen',
            'input.global_pressure_mapping': [(0.0, 1.0), (1.0, 0.0)],
            'input.enable_history_popup': True,
            'view.default_zoom': 1.0,
            'saving.default_format': 'openraster',
            'brushmanager.selected_brush' : None,
            'brushmanager.selected_groups' : [],
        }
        self.preferences = DEFAULT_CONFIG
        try: 
            user_config = get_json_config()
        except IOError:
            user_config = get_legacy_config()
        self.preferences.update(user_config)

    def init_brush_adjustments(self, ):
        """Initializes all the brush adjustments for the current brush"""
        self.brush_adjustment = {}
        from brushlib import brushsettings
        for i, s in enumerate(brushsettings.settings_visible):
            adj = gtk.Adjustment(value=s.default, lower=s.min, upper=s.max, step_incr=0.01, page_incr=0.1)
            self.brush_adjustment[s.cname] = adj

    def update_input_mapping(self):
        p = self.preferences['input.global_pressure_mapping']
        if len(p) == 2 and abs(p[0][1]-1.0)+abs(p[1][1]-0.0) < 0.0001:
            # 1:1 mapping (mapping disabled)
            self.doc.tdw.pressure_mapping = None
        else:
            # TODO: maybe replace this stupid mapping by a hard<-->soft slider?
            m = mypaintlib.Mapping(1)
            m.set_n(0, len(p))
            for i, (x, y) in enumerate(p):
                m.set_point(0, i, x, 1.0-y)

            def mapping(pressure):
                return m.calculate_single_input(pressure)
            self.doc.tdw.pressure_mapping = mapping

    def update_input_devices(self):
        # init extended input devices
        self.pressure_devices = []
        for device in gdk.devices_list():
            #print device.name, device.source

            #if device.source in [gdk.SOURCE_PEN, gdk.SOURCE_ERASER]:
            # The above contition is True sometimes for a normal USB
            # Mouse. https://gna.org/bugs/?11215
            # In fact, GTK also just guesses this value from device.name.

            last_word = device.name.split()[-1].lower()
            if last_word == 'pad':
                # Setting the intuos3 pad into "screen mode" causes
                # glitches when you press a pad-button in mid-stroke,
                # and it's not a pointer device anyway. But it reports
                # axes almost identical to the pen and eraser.
                #
                # device.name is usually something like "wacom intuos3 6x8 pad" or just "pad"
                print 'Ignoring "%s" (probably wacom keypad device)' % device.name
                continue
            if last_word == 'cursor':
                # this is a "normal" mouse and does not work in screen mode
                print 'Ignoring "%s" (probably wacom mouse device)' % device.name
                continue

            for use, val_min, val_max in device.axes:
                # Some mice have a third "pressure" axis, but without
                # minimum or maximum. https://gna.org/bugs/?14029
                if use == gdk.AXIS_PRESSURE and val_min != val_max:
                    if 'mouse' in device.name.lower():
                        # Real fix for the above bug https://gna.org/bugs/?14029
                        print 'Ignoring "%s" (probably a mouse, but it reports extra axes)' % device.name
                        continue

                    self.pressure_devices.append(device.name)
                    modesetting = self.preferences['input.device_mode']
                    mode = getattr(gdk, 'MODE_' + modesetting.upper())
                    if device.mode != mode:
                        print 'Setting %s mode for "%s"' % (modesetting, device.name)
                        device.set_mode(mode)
                    break

    def set_current_brush(self, managed_brush):
        """
        Copies a ManagedBrush's settings into the brush settings currently used
        for painting. Sets the parent brush name to something long-lasting too,
        specifically the nearest persistent brush in managed_brush's ancestry.
        """
        if managed_brush is None:
            return
        self.brush.load_from_brushinfo(managed_brush.brushinfo)

        # If the user just picked a brush from the brush selection window,
        # it's likely to have no parent.
        if not managed_brush.brushinfo.has_key("parent_brush_name"):
            parent_mb = self.brushmanager.find_nearest_persistent_brush(managed_brush)
            parent_mb_name = parent_mb is not None and parent_mb.name or None
            self.brush.brushinfo["parent_brush_name"] = parent_mb_name

    def brush_selected_cb(self, brush):
        assert brush is not self.brush
        self.set_current_brush(brush)

    def hide_window_cb(self, window, event):
        # used by some of the windows
        window.hide()
        return True

    def save_gui_config(self):
        gtk.accel_map_save(join(self.confpath, 'accelmap.conf'))
        self.save_window_positions()
        self.save_settings()

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

    def pick_color_at_pointer(self, widget, size=3):
        '''Grab screen color at cursor (average of size x size rectangle)'''
        # inspired by gtkcolorsel.c function grab_color_at_mouse()
        screen = widget.get_screen()
        colormap = screen.get_system_colormap()
        root = screen.get_root_window()
        screen_w, screen_h = screen.get_width(), screen.get_height()
        display = widget.get_display()
        screen_trash, x_root, y_root, modifiermask_trash = display.get_pointer()
        image = None
        x = x_root-size/2
        y = y_root-size/2
        if x < 0: x = 0
        if y < 0: y = 0
        if x+size > screen_w: x = screen_w-size
        if y+size > screen_h: y = screen_h-size
        image = root.get_image(x, y, size, size)
        color_total = (0, 0, 0)
        for x, y in helpers.iter_rect(0, 0, size, size):
            pixel = image.get_pixel(x, y)
            color = colormap.query_color(pixel)
            color = [color.red, color.green, color.blue]
            color_total = (color_total[0]+color[0], color_total[1]+color[1], color_total[2]+color[2])
        N = size*size
        color_total = (color_total[0]/N, color_total[1]/N, color_total[2]/N)
        color_rgb = [ch/65535. for ch in color_total]
        self.brush.set_color_rgb(color_rgb)

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
