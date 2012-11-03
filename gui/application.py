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
import brushmodifier, linemode
import colors
from colorwindow import BrushColorManager
from overlays import LastPaintPosOverlay, ScaleOverlay

import pygtkcompat

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

        if pygtkcompat.USE_GTK3:
            gtk.Window.set_default_icon_name('mypaint')
        else:
            gtk.window_set_default_icon_name('mypaint')

        # Stock items, core actions, and menu structure
        builder_xml = join(datapath, "gui", "mypaint.xml")
        self.builder = gtk.Builder()
        self.builder.add_from_file(builder_xml)
        factory = self.builder.get_object("stock_icon_factory")
        factory.add_default()

        self.ui_manager = self.builder.get_object("app_ui_manager")
        signal_callback_objs = []

        gdk.set_program_class('MyPaint')

        self.pixmaps = PixbufDirectory(join(self.datapath, 'pixmaps'))
        self.cursor_color_picker = gdk.Cursor(
                  pygtkcompat.gdk.display_get_default(),
                  self.pixmaps.cursor_color_picker,
                  1, 30)

        # unmanaged main brush; always the same instance (we can attach settings_observers)
        # this brush is where temporary changes (color, size...) happen
        self.brush = brush.BrushInfo()
        self.brush.load_defaults()

        # Global pressure mapping function, ignored unless set
        self.pressure_mapping = None

        self.preferences = {}
        self.load_settings()

        self.scratchpad_filename = ""
        self.kbm = keyboard.KeyboardManager(self)
        self.doc = document.Document(self)
        signal_callback_objs.append(self.doc)
        signal_callback_objs.append(self.doc.modes)
        self.scratchpad_doc = document.Document(self, leader=self.doc)
        self.brushmanager = brushmanager.BrushManager(join(datapath, 'brushes'), join(confpath, 'brushes'), self)
        self.filehandler = filehandling.FileHandler(self)
        signal_callback_objs.append(self.filehandler)
        self.brushmodifier = brushmodifier.BrushModifier(self)
        self.line_mode_settings = linemode.LineModeSettings(self)

        # Monitors changes of input device & saves device-specific brushes
        self.device_monitor = DeviceUseMonitor(self)

        if not self.preferences.get("scratchpad.last_opened_scratchpad", None):
            self.preferences["scratchpad.last_opened_scratchpad"] = self.filehandler.get_scratchpad_autosave()
        self.scratchpad_filename = self.preferences["scratchpad.last_opened_scratchpad"]

        self.brush_color_manager = BrushColorManager(self)
        self.brush_color_manager.set_picker_cursor(self.cursor_color_picker)
        self.brush_color_manager.set_data_path(datapath)

        self.init_brush_adjustments()

        self.layout_manager = layout.LayoutManager(
            prefs=self.preferences["layout.window_positions"],
            factory=windowing.window_factory,
            factory_opts=[self]  )
        self.drawWindow = self.layout_manager.get_widget_by_role("main-window")
        self.layout_manager.show_all()

        signal_callback_objs.append(self.drawWindow)

        # Connect signals defined in mypaint.xml
        callback_finder = CallbackFinder(signal_callback_objs)
        self.builder.connect_signals(callback_finder)

        self.kbm.start_listening()
        self.filehandler.doc = self.doc
        self.filehandler.filename = None
        pygtkcompat.gtk.accel_map_load(join(self.confpath, 'accelmap.conf'))

        # Load the background settings window.
        # FIXME: this line shouldn't be needed, but we need to load this up
        # front to get any non-default background that the user has configured
        # from the preferences.
        self.layout_manager.get_subwindow_by_role("backgroundWindow")

        # And the brush settings window, or things like eraser mode will break.
        # FIXME: brush_adjustments should not be dependent on this
        self.layout_manager.get_subwindow_by_role("brushSettingsWindow")

        def at_application_start(*junk):
            col = self.brush_color_manager.get_color()
            self.brushmanager.select_initial_brush()
            self.brush_color_manager.set_color(col)
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
            'frame.color_rgba': (0.12, 0.12, 0.12, 0.92),
            'misc.context_restores_color': True,

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
                'brushSelectionWindow': dict(
                        sbindex=2, floating=True, hidden=True,
                        x=-100, y=-150, w=250, h=350, sbheight=350),
                'layersWindow': dict(
                        sbindex=3, floating=True, hidden=True,
                        x=-460, y=-150, w=200, h=200, sbheight=200),
                'scratchWindow': dict(
                        sbindex=4, floating=True, hidden=True,
                        x=-555, y=125, w=300, h=250, sbheight=250),
                'colorWindow': dict(
                        sbindex=0, floating=True, hidden=True,
                        x=-100, y=125, w=250, h=300, sbheight=300),

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

    def update_input_mapping(self):
        p = self.preferences['input.global_pressure_mapping']
        if len(p) == 2 and abs(p[0][1]-1.0)+abs(p[1][1]-0.0) < 0.0001:
            # 1:1 mapping (mapping disabled)
            self.pressure_mapping = None
        else:
            # TODO: maybe replace this stupid mapping by a hard<-->soft slider?
            #       But then we would also need a "minimum pressure" setting,
            #       or else this often used workaround is no longer possible:
            #       http://wiki.mypaint.info/File:Pressure_workaround.png
            m = mypaintlib.MappingWrapper(1)
            m.set_n(0, len(p))
            for i, (x, y) in enumerate(p):
                m.set_point(0, i, x, 1.0-y)

            def mapping(pressure):
                return m.calculate_single_input(pressure)
            self.pressure_mapping = mapping

    def update_input_devices(self):
        # avoid doing this 5 times at startup
        modesetting = self.preferences['input.device_mode']
        if getattr(self, 'last_modesetting', None) == modesetting:
            return
        self.last_modesetting = modesetting

        # init extended input devices
        self.pressure_devices = []

        if pygtkcompat.USE_GTK3:
            display = pygtkcompat.gdk.display_get_default()
            device_mgr = display.get_device_manager()
            for device in device_mgr.list_devices(gdk.DeviceType.SLAVE):
                if device.get_source() == gdk.InputSource.KEYBOARD:
                    continue
                name = device.get_name().lower()
                n_axes = device.get_n_axes()
                if n_axes <= 0:
                    continue
                # TODO: may need exception voodoo, min/max checking etc. here
                #       like the GTK2 code below.
                for i in xrange(n_axes):
                    use = device.get_axis_use(i)
                    if use != gdk.AxisUse.PRESSURE:
                        continue
                    # Set preferred device mode
                    mode = getattr(gdk.InputMode, modesetting.upper())
                    if device.get_mode() != mode:
                        print 'Setting %s mode for "%s"' \
                          % (mode, device.get_name())
                        device.set_mode(mode)
                    # Record as a pressure-sensitive device
                    self.pressure_devices.append(name)
                    break
            return

        # GTK2/PyGTK
        print 'Looking for GTK devices with pressure:'
        for device in gdk.devices_list():
            #print device.name, device.source

            #if device.source in [gdk.SOURCE_PEN, gdk.SOURCE_ERASER]:
            # The above contition is True sometimes for a normal USB
            # Mouse. https://gna.org/bugs/?11215
            # In fact, GTK also just guesses this value from device.name.

            #print 'Device "%s" (%s) reports %d axes.' % (device.name, device.source.value_name, len(device.axes))

            pressure = False
            for use, val_min, val_max in device.axes:
                if use == gdk.AXIS_PRESSURE:
                    print 'Device "%s" has a pressure axis' % device.name
                    # Some mice have a third "pressure" axis, but without minimum or maximum.
                    if val_min == val_max:
                        print 'But the pressure range is invalid'
                    else:
                        pressure = True
                    break
            if not pressure:
                #print 'Skipping device "%s" because it has no pressure axis' % device.name
                continue

            name = device.name.lower()
            name = name.replace('-', ' ').replace('_', ' ')
            last_word = name.split()[-1]

            # Step 1: BLACKLIST
            if last_word == 'pad':
                # Setting the intuos3 pad into "screen mode" causes
                # glitches when you press a pad-button in mid-stroke,
                # and it's not a pointer device anyway. But it reports
                # axes almost identical to the pen and eraser.
                #
                # device.name is usually something like "wacom intuos3 6x8 pad" or just "pad"
                print 'Skipping "%s" (probably wacom keypad device)' % device.name
                continue
            if last_word == 'touchpad':
                print 'Skipping "%s" (probably a laptop touchpad without pressure info)' % device.name
                continue
            if last_word == 'cursor':
                # for wacom, this is the "normal" mouse and does not work in screen mode
                print 'Skipping "%s" (probably wacom mouse device)' % device.name
                continue
            if 'keyboard' in name:
                print 'Skipping "%s" (probably a keyboard)' % device.name
                continue
            if 'mouse' in name and 'mousepen' not in name:
                print 'Skipping "%s" (probably a mouse)' % device.name
                continue

            # Step 2: WHITELIST
            #
            # Required now as too many input devices report a pressure
            # axis with recent Xorg versions. Wrongly enabling them
            # breaks keyboard and/or mouse input in random ways.
            #
            # Only whole words are matched.
            tablet_strings  = '''
            tablet pressure graphic art pen stylus eraser pencil brush
            wacom bamboo intuos graphire cintiq
            hanvon rollick graphicpal artmaster sentip
            genius mousepen
            '''
            match = False
            words = name.split()
            for s in tablet_strings.split():
                if s in words:
                    match = True
            if not match:
                print 'Skipping "%s" (not in the list of known tablets)' % device.name
                continue

            self.pressure_devices.append(device.name)
            mode = getattr(gdk, 'MODE_' + modesetting.upper())
            if device.mode != mode:
                print 'Setting %s mode for "%s"' % (modesetting, device.name)
                device.set_mode(mode)
        print ''

    def save_gui_config(self):
        pygtkcompat.gtk.accel_map_save(join(self.confpath, 'accelmap.conf'))
        self.save_settings()

    def message_dialog(self, text, type=gtk.MESSAGE_INFO, flags=0,
                       secondary_text=None, long_text=None, title=None):
        """Utility function to show a message/information dialog.
        """
        d = gtk.MessageDialog(self.drawWindow, flags=flags, type=type,
                              buttons=gtk.BUTTONS_OK)
        d.set_markup(text)
        if title is not None:
            d.set_title(title)
        if secondary_text is not None:
            d.format_secondary_markup(secondary_text)
        if long_text is not None:
            buf = gtk.TextBuffer()
            buf.set_text(long_text)
            tv = gtk.TextView(buf)
            tv.show()
            tv.set_editable(False)
            tv.set_wrap_mode(gtk.WRAP_WORD_CHAR)
            scrolls = gtk.ScrolledWindow()
            scrolls.show()
            scrolls.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_ALWAYS)
            scrolls.add(tv)
            scrolls.set_size_request(-1, 300)
            scrolls.set_shadow_type(gtk.SHADOW_IN)
            d.get_message_area().pack_start(scrolls)
        d.run()
        d.destroy()

    def pick_color_at_pointer(self, widget, size=3):
        """Set the brush colour from the current pointer position on screen.

        This is a wrapper for `gui.colors.get_color_at_pointer()`, and
        additionally sets the current brush colour.

        """
        color = colors.get_color_at_pointer(widget, size)
        self.brush_color_manager.set_color(color)


class DeviceUseMonitor (object):
    """Monitors device uses and detects changes.
    """

    def __init__(self, app):
        """Initialize.

        :param app: the main Application singleton.
        """
        object.__init__(self)
        self.app = app
        self.device_observers = []   #: See `device_used()`.
        self._last_event_device = None
        self._last_pen_device = None
        self.device_observers.append(self.device_changed_cb)


    def device_used(self, device):
        """Notify about a device being used; for use by controllers etc.

        :param device: the device being used

        If the device has changed, this method then notifies the registered
        observers via callbacks in device_observers. Callbacks are invoked as

            callback(old_device, new_device)

        This method returns True if the device was the same as the previous
        device, and False if it has changed.

        """
        if device == self._last_event_device:
            return True
        for func in self.device_observers:
            func(self._last_event_device, device)
        self._last_event_device = device
        return False


    def device_is_eraser(self, device):
        if device is None:
            return False
        return device.source == gdk.SOURCE_ERASER \
                or 'eraser' in device.name.lower()


    def device_changed_cb(self, old_device, new_device):
        # small problem with this code: it doesn't work well with brushes that
        # have (eraser not in [1.0, 0.0])

        if pygtkcompat.USE_GTK3:
            new_device.name = new_device.props.name
            new_device.source = new_device.props.input_source

        print 'device change:', new_device.name, new_device.source

        # When editing brush settings, it is often more convenient to use the
        # mouse. Because of this, we don't restore brushsettings when switching
        # to/from the mouse. We act as if the mouse was identical to the last
        # active pen device.

        if new_device.source == gdk.SOURCE_MOUSE and self._last_pen_device:
            new_device = self._last_pen_device
        if new_device.source == gdk.SOURCE_PEN:
            self._last_pen_device = new_device
        if old_device and old_device.source == gdk.SOURCE_MOUSE \
                    and self._last_pen_device:
            old_device = self._last_pen_device

        bm = self.app.brushmanager
        if old_device:
            # Clone for saving
            old_brush = bm.clone_selected_brush(name=None)
            bm.store_brush_for_device(old_device.name, old_brush)

        if new_device.source == gdk.SOURCE_MOUSE:
            # Avoid fouling up unrelated devbrushes at stroke end
            self.app.preferences.pop('devbrush.last_used', None)
        else:
            # Select the brush and update the UI.
            # Use a sane default if there's nothing associated
            # with the device yet.
            brush = bm.fetch_brush_for_device(new_device.name)
            if brush is None:
                if self.device_is_eraser(new_device):
                    brush = bm.get_default_eraser()
                else:
                    brush = bm.get_default_brush()
            self.app.preferences['devbrush.last_used'] = new_device.name
            bm.select_brush(brush)


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


class CallbackFinder:
    """Finds callbacks amongst a list of objects.

    It's not possible to call `GtkBuilder.connect_signals()` more than once,
    but we use more tnan one backend object. Thus, this little workaround is
    necessary during construction.

    See http://stackoverflow.com/questions/4637792

    """

    def __init__(self, objects):
        self._objs = list(objects)

    def __getitem__(self, name):
        # PyGTK/GTK2 uses getitem
        name = str(name)
        found = [getattr(obj, name) for obj in self._objs
                  if hasattr(obj, name)]
        if len(found) == 1:
            return found[0]
        elif len(found) > 1:
            print "WARNING: ambiguity: %r resolves to %r" % (name, found)
            print "WARNING: using first match only."
            return found[0]
        else:
            raise AttributeError, "No method named %r was defined " \
                "on any of %r" % (name, self._objs)

    # PyGI/GTK3's override uses getattr()
    __getattr__ = __getitem__

