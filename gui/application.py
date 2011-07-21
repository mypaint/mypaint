# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os, sys
from os.path import join
import gtk, gobject
gdk = gtk.gdk
from lib import brush, helpers, mypaintlib
import filehandling, keyboard, brushmanager, windowing, document, layout
import colorhistory, brushmodifier

class Application: # singleton
    """
    This class serves as a global container for everything that needs
    to be shared in the GUI. Its constructor is the last part of the
    initialization, called by main.py or by the testing scripts.
    """
    def __init__(self, datapath, confpath, filenames):
        self.confpath = confpath
        self.datapath = datapath

        # create config directory, and subdirs where the user might drop files
        # TODO make scratchpad dir something pulled from preferences #PALETTE1
        for d in ['', 'backgrounds', 'brushes', 'scratchpads']:
            d = os.path.join(self.confpath, d)
            if not os.path.isdir(d):
                os.mkdir(d)
                print 'Created', d

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
        self.brush = brush.BrushInfo()
        self.brush.load_defaults()

        self.preferences = {}
        self.load_settings()

        self.brushmanager = brushmanager.BrushManager(join(datapath, 'brushes'), join(confpath, 'brushes'), self)
        self.kbm = keyboard.KeyboardManager()
        self.filehandler = filehandling.FileHandler(self)
        self.brushmodifier = brushmodifier.BrushModifier(self)
        self.doc = document.Document(self)
        
        self.filehandler.scratchpad_filename = ""
        self.filehandler.scratchpad_doc = document.Scratchpad(self)
        
        if not self.preferences.get("scratchpad.last_opened_scratchpad", None):
            self.preferences["scratchpad.last_opened_scratchpad"] = self.filehandler.get_scratchpad_autosave()
        self.filehandler.scratchpad_filename = self.preferences["scratchpad.last_opened_scratchpad"]

        self.brush.set_color_hsv((0, 0, 0))
        self.init_brush_adjustments()

        self.ch = colorhistory.ColorHistory(self)

        self.layout_manager = layout.LayoutManager(
            app=self,
            prefs=self.preferences["layout.window_positions"],
            factory=windowing.window_factory,
            factory_opts=[self]  )
        self.drawWindow = self.layout_manager.get_widget_by_role("main-window")
        self.layout_manager.show_all()

        self.kbm.start_listening()
        self.filehandler.doc = self.doc
        self.filehandler.filename = None
        gtk.accel_map_load(join(self.confpath, 'accelmap.conf'))
        
        # Load the background settings window.
        # FIXME: this line shouldn't be needed, but we need to load this up
        # front to get any non-default background that the user has configured
        # from the preferences.
        self.layout_manager.get_subwindow_by_role("backgroundWindow")

        # And the brush settings window, or things like eraser mode will break.
        # FIXME: brush_adjustments should not be dependent on this
        self.layout_manager.get_subwindow_by_role("brushSettingsWindow")

        def at_application_start(*trash):
            self.brushmanager.select_initial_brush()
            if filenames:
                # Open only the first file, no matter how many has been specified
                # If the file does not exist just set it as the file to save to
                fn = filenames[0].replace('file:///', '/') # some filebrowsers do this (should only happen with outdated mypaint.desktop)
                if not os.path.exists(fn):
                    self.filehandler.filename = fn
                else:
                    self.filehandler.open_file(fn)

            # Load last scratchpad
            if not self.preferences["scratchpad.last_opened_scratchpad"]:
                self.preferences["scratchpad.last_opened_scratchpad"] = self.filehandler.get_scratchpad_autosave()
                self.filehandler.scratchpad_filename = self.preferences["scratchpad.last_opened_scratchpad"]
            if os.path.isfile(self.filehandler.scratchpad_filename):
                self.filehandler.open_scratchpad(self.filehandler.scratchpad_filename)

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
        self.filehandler.save_scratchpad(self.filehandler.scratchpad_filename)
        save_config()

    def apply_settings(self):
        """Applies the current settings."""
        self.update_input_mapping()
        self.update_input_devices()
        prefs_win = self.layout_manager.get_widget_by_role('preferencesWindow')
        prefs_win.update_ui()

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
            try:
                return helpers.json_loads(jsonstr)
            except Exception, e:
                print "settings.json: %s" % (str(e),)
                print "warning: failed to load settings.json, using defaults"
                return {}
        if sys.platform == 'win32':
            import glib
            scrappre = join(glib.get_user_special_dir(glib.USER_DIRECTORY_DOCUMENTS).decode('utf-8'),'MyPaint','scrap')
        else:
            scrappre = '~/MyPaint/scrap'
        DEFAULT_CONFIG = {
            'saving.scrap_prefix': scrappre,
            'input.device_mode': 'screen',
            'input.global_pressure_mapping': [(0.0, 1.0), (1.0, 0.0)],
            'view.default_zoom': 1.0,
            'view.high_quality_zoom': True,
            'saving.default_format': 'openraster',
            'brushmanager.selected_brush' : None,
            'brushmanager.selected_groups' : [],
            "input.button1_shift_action": 'straight_line',
            "input.button1_ctrl_action":  'ColorPickerPopup',
            "input.button2_action":       'pan_canvas',
            "input.button2_shift_action": 'rotate_canvas',
            "input.button2_ctrl_action":  'zoom_canvas',
            "input.button3_action":       'ColorHistoryPopup',
            "input.button3_shift_action": 'no_action',
            "input.button3_ctrl_action":  'no_action',

            "scratchpad.last_opened_scratchpad": "",

            # Default window positions.
            # See gui.layout.set_window_initial_position for the meanings
            # of the common x, y, w, and h settings
            "layout.window_positions": {

                # Main window default size. Sidebar width is saved here
                'main-window': dict(sbwidth=270, x=64, y=32, w=-74, h=-96),

                # Tool windows. These can be undocked (floating=True) or set
                # initially hidden (hidden=True), or be given an initial sidebar
                # index (sbindex=<int>) or height in the sidebar (sbheight=<int>)
                # Non-hidden entries determine the default set of tools.
                'colorSamplerWindow': dict(sbindex=1, floating=False, hidden=False,
                                           x=-200, y=128,
                                           w=200, h=300, sbheight=300),
                'colorSelectionWindow': dict(sbindex=0, floating=True, hidden=True,
                                             x=-128, y=64,
                                             w=200, h=250, sbheight=250),
                'brushSelectionWindow': dict(sbindex=2, floating=True,
                                             x=-128, y=-128,
                                             w=250, h=350, sbheight=350),
                'layersWindow': dict(sbindex=3, floating=True,
                                     x=128, y=-128,
                                     w=200, h=200, sbheight=200),

                'scratchWindow': dict(sbindex=4, floating=True,
                                     x=128, y=-128,
                                     w=200, h=200, sbheight=200),

                # Non-tool subwindows. These cannot be docked, and are all
                # intially hidden.
                'brushSettingsWindow': dict(x=-460, y=-128, w=300, h=300),
                'backgroundWindow': dict(),
                'inputTestWindow': dict(),
                'frameWindow': dict(),
                'preferencesWindow': dict(),
            },
        }
        window_pos = DEFAULT_CONFIG["layout.window_positions"]
        self.window_names = window_pos.keys()
        self.preferences = DEFAULT_CONFIG
        try: 
            user_config = get_json_config()
        except IOError:
            user_config = get_legacy_config()
        user_window_pos = user_config.get("layout.window_positions", {})
        # note: .update() replaces the window position dict, but we want to update it
        self.preferences.update(user_config)
        # update window_pos, and drop window names that don't exist any more
        # (we need to drop them because otherwise we will try to show a non-existing window)
        for role in self.window_names:
            if role in user_window_pos:
                window_pos[role] = user_window_pos[role]
        self.preferences["layout.window_positions"] = window_pos

    def init_brush_adjustments(self):
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

    def save_gui_config(self):
        gtk.accel_map_save(join(self.confpath, 'accelmap.conf'))
        self.save_settings()

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
