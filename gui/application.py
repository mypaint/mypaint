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
import colorhistory, brushmodifier, linemode
import stock
from overlays import LastPaintPosOverlay, ScaleOverlay


class Application: # singleton
    """
    This class serves as a global container for everything that needs
    to be shared in the GUI. Its constructor is the last part of the
    initialization, called by main.py or by the testing scripts.
    """
    def __init__(self, datapath, extradata, confpath, filenames):
        """Construct, but do not run.

        :`datapath`:
            Usually ``$PREFIX/share/mypaint``. Where MyPaint should find its
            app-specific read-only data, e.g. UI definition XML, backgrounds
            and brush defintions.
        :`extradata`:
            Where to find the defaults for MyPaint's themeable UI icons. This
            will be effectively used in addition to ``$XDG_DATA_DIRS`` for the
            purposes of icon lookup. Normally it's ``$PREFIX/share``, to support
            unusual installations outside the usual locations. It should contain
            an ``icons/`` subdirectory.
        :`confpath`:
            Where the user's configuration is stored. ``$HOME/.mypaint`` is
            typical on Unix-like OSes.
        """
        self.confpath = confpath
        self.datapath = datapath

        # create config directory, and subdirs where the user might drop files
        # TODO make scratchpad dir something pulled from preferences #PALETTE1
        for d in ['', 'backgrounds', 'brushes', 'scratchpads']:
            d = os.path.join(self.confpath, d)
            if not os.path.isdir(d):
                os.mkdir(d)
                print 'Created', d

        # Default location for our icons. The user's theme can override these.
        icon_theme = gtk.icon_theme_get_default()
        icon_theme.append_search_path(join(extradata, "icons"))

        # Icon sanity check
        if not icon_theme.has_icon('mypaint') \
                or not icon_theme.has_icon('mypaint-tool-brush'):
            print 'Error: Where have my icons gone?'
            print 'Icon search path:', icon_theme.get_search_path()
            print "Mypaint can't run sensibly without its icons; " \
                + "please check your installation."
            print 'see https://gna.org/bugs/?18460 for possible solutions'
            sys.exit(1)
        gtk.window_set_default_icon_name('mypaint')

        stock.init_custom_stock_items()

        self.ui_manager = gtk.UIManager()

        gdk.set_program_class('MyPaint')

        self.pixmaps = PixbufDirectory(join(self.datapath, 'pixmaps'))
        self.cursor_color_picker = gdk.Cursor(gdk.display_get_default(), self.pixmaps.cursor_color_picker, 1, 30)

        # unmanaged main brush; always the same instance (we can attach settings_observers)
        # this brush is where temporary changes (color, size...) happen
        self.brush = brush.BrushInfo()
        self.brush.load_defaults()

        self.preferences = {}
        self.load_settings()

        self.scratchpad_filename = ""
        self.kbm = keyboard.KeyboardManager()
        self.doc = document.Document(self)
        self.scratchpad_doc = document.Document(self, leader=self.doc)
        self.brushmanager = brushmanager.BrushManager(join(datapath, 'brushes'), join(confpath, 'brushes'), self)
        self.filehandler = filehandling.FileHandler(self)
        self.brushmodifier = brushmodifier.BrushModifier(self)
        self.linemode = linemode.LineMode(self)

        if not self.preferences.get("scratchpad.last_opened_scratchpad", None):
            self.preferences["scratchpad.last_opened_scratchpad"] = self.filehandler.get_scratchpad_autosave()
        self.scratchpad_filename = self.preferences["scratchpad.last_opened_scratchpad"]

        self.brush.set_color_hsv((0, 0, 0))
        self.init_brush_adjustments()
        self.init_line_mode_adjustments()

        self.ch = colorhistory.ColorHistory(self)

        self.layout_manager = layout.LayoutManager(
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

        def at_application_start(*junk):
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
                self.scratchpad_filename = self.preferences["scratchpad.last_opened_scratchpad"]
            if os.path.isfile(self.scratchpad_filename):
                try:
                    self.filehandler.open_scratchpad(self.scratchpad_filename)
                except AttributeError, e:
                    print "Scratchpad widget isn't initialised yet, so cannot centre"


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
        self.brushmanager.save_brush_history()
        self.filehandler.save_scratchpad(self.scratchpad_filename)
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
            'ui.hide_menubar_in_fullscreen': True,
            'ui.hide_toolbar_in_fullscreen': True,
            'ui.hide_subwindows_in_fullscreen': True,
            'ui.parts': dict(main_toolbar=True, menubar=False),
            'ui.feedback.scale': True,
            'ui.feedback.last_pos': False,
            'ui.toolbar_items': dict(
                toolbar1_file=False,
                toolbar1_scrap=False,
                toolbar1_edit=True,
                toolbar1_editmodes=True,
                toolbar1_blendmodes=False,
                toolbar1_view=False,
                toolbar1_subwindows=True,
            ),
            'saving.default_format': 'openraster',
            'brushmanager.selected_brush' : None,
            'brushmanager.selected_groups' : [],

            "scratchpad.last_opened_scratchpad": "",

            # Default window positions.
            # See gui.layout.set_window_initial_position for the meanings
            # of the common x, y, w, and h settings
            "layout.window_positions": {

                # Main window default size. Sidebar width is saved here
                'main-window': dict(sbwidth=250, x=50, y=32, w=-50, h=-100),

                # Tool windows. These can be undocked (floating=True) or set
                # initially hidden (hidden=True), or be given an initial sidebar
                # index (sbindex=<int>) or height in the sidebar (sbheight=<int>)
                # Non-hidden entries determine the default set of tools.
                'colorSelectionWindow': dict(
                        sbindex=0, floating=True, hidden=True,
                        x=-100, y=125, w=160, h=200, sbheight=200),
                'colorSamplerWindow': dict(
                        sbindex=1, floating=True, hidden=True,
                        x=-270, y=125, w=200, h=275, sbheight=275),
                'brushSelectionWindow': dict(
                        sbindex=2, floating=True, hidden=True,
                        x=-100, y=-150, w=250, h=350, sbheight=350),
                'layersWindow': dict(
                        sbindex=3, floating=True, hidden=True,
                        x=-460, y=-150, w=200, h=200, sbheight=200),
                'scratchWindow': dict(
                        sbindex=4, floating=True, hidden=True,
                        x=-555, y=125, w=300, h=250, sbheight=250),

                # Non-tool subwindows. These cannot be docked, and are all
                # intially hidden.
                'brushSettingsWindow': dict(x=-460, y=-128, w=300, h=300),
                'backgroundWindow': dict(),
                'inputTestWindow': dict(),
                'frameWindow': dict(),
                'preferencesWindow': dict(),
            },
        }
        if sys.platform == 'win32':
            # The Linux wacom driver inverts the button numbers of the
            # pen flip button, because middle-click is the more useful
            # action on Linux. However one of the two buttons is often
            # accidentally hit with the thumb while painting. We want
            # to assign panning to this button by default.
            DEFAULT_CONFIG.update({
                "input.button1_shift_action": 'straight_line',
                "input.button1_ctrl_action":  'ColorPickerPopup',
                "input.button3_action":       'pan_canvas',
                "input.button3_shift_action": 'rotate_canvas',
                "input.button3_ctrl_action":  'zoom_canvas',
                "input.button2_action":       'ColorHistoryPopup',
                "input.button2_shift_action": 'no_action',
                "input.button2_ctrl_action":  'no_action',
                })
        else:
            DEFAULT_CONFIG.update({
                "input.button1_shift_action": 'straight_line',
                "input.button1_ctrl_action":  'ColorPickerPopup',
                "input.button2_action":       'pan_canvas',
                "input.button2_shift_action": 'rotate_canvas',
                "input.button2_ctrl_action":  'zoom_canvas',
                "input.button3_action":       'ColorHistoryPopup',
                "input.button3_shift_action": 'no_action',
                "input.button3_ctrl_action":  'no_action',
                })

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

    def add_action_group(self, ag):
        self.ui_manager.insert_action_group(ag, -1)

    def find_action(self, name):
        for ag in self.ui_manager.get_action_groups():
            result = ag.get_action(name)
            if result is not None:
                return result

    def init_brush_adjustments(self):
        """Initializes all the brush adjustments for the current brush"""
        self.brush_adjustment = {}
        from brushlib import brushsettings
        for i, s in enumerate(brushsettings.settings_visible):
            adj = gtk.Adjustment(value=s.default, lower=s.min, upper=s.max, step_incr=0.01, page_incr=0.1)
            self.brush_adjustment[s.cname] = adj

    def init_line_mode_adjustments(self):
        self.line_mode_adjustment = {}
        for i, s in enumerate(linemode.line_mode_settings):
            adj = gtk.Adjustment(value=s.default, lower=s.min, upper=s.max, step_incr=0.01, page_incr=0.1)
            self.line_mode_adjustment[s.cname] = adj

    def update_input_mapping(self):
        p = self.preferences['input.global_pressure_mapping']
        if len(p) == 2 and abs(p[0][1]-1.0)+abs(p[1][1]-0.0) < 0.0001:
            # 1:1 mapping (mapping disabled)
            self.doc.tdw.pressure_mapping = None
        else:
            # TODO: maybe replace this stupid mapping by a hard<-->soft slider?
            m = mypaintlib.MappingWrapper(1)
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

            name = device.name.lower()
            last_word = name.split()[-1]
            if last_word == 'pad':
                # Setting the intuos3 pad into "screen mode" causes
                # glitches when you press a pad-button in mid-stroke,
                # and it's not a pointer device anyway. But it reports
                # axes almost identical to the pen and eraser.
                #
                # device.name is usually something like "wacom intuos3 6x8 pad" or just "pad"
                print 'Ignoring "%s" (probably wacom keypad device)' % device.name
                continue
            if last_word == 'touchpad':
                # eg. "SynPS/2 Synaptics TouchPad"
                # Cannot paint at all, cannot select brushes, if we enable this one.
                print 'Ignoring "%s" (probably laptop touchpad which screws up gtk+ if enabled)' % device.name
                continue
            if last_word == 'cursor':
                # this is a "normal" mouse and does not work in screen mode
                print 'Ignoring "%s" (probably wacom mouse device)' % device.name
                continue
            if 'transceiver' in name:
                # eg. "Microsoft Microsoft 2.4GHz Transceiver V1.0"
                # Cannot paint after moving outside of painting area.
                print 'Ignoring "%s" (a transceiver is probably not a pressure sensitive tablet, known to screw up gtk+ when enabled)' % device.name
                continue
            if 'glidepoint' in name:
                # AlpsPS/2 ALPS GlidePoint
                # https://gna.org/bugs/?19790
                print 'Ignoring "%s" (probably a mouse-like device reporting extra axes)' % device.name
                continue
            if 'keyboard' in name:
                # e.g "Plus More Entertainment LTD. USB-compliant keyboard."
                # has a scroll wheel, breaks drawing, color picking and choosing the brush
                print 'Ignoring "%s" (probably a multifunc keyboard)' % device.name
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
        screen_junk, x_root, y_root, modifiermask_trash = display.get_pointer()
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
