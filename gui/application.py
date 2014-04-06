# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import locale
import gettext
import os
import sys
from os.path import join
import logging
logger = logging.getLogger(__name__)

import gobject
import gtk
from gtk import gdk

import lib.document
from lib import brush
from lib import helpers
from lib import mypaintlib
from libmypaint import brushsettings


def get_app():
    """Returns the `gui.application.Application` singleton object."""
    # Define this up front: gui.* requires the singleton object pretty much
    # everywhere, and the app instance carries them as members.
    return Application._INSTANCE


import gtk2compat
import filehandling
import keyboard
import brushmanager
import windowing
import document
import tileddrawwidget
import workspace
import topbar
import drawwindow
import backgroundwindow
import preferenceswindow
import brusheditor
import layerswindow
import previewwindow
import optionspanel
import framewindow
import scratchwindow
import inputtestwindow
import brushiconeditor
import history
import colortools
import brushmodifier
import toolbar
import linemode
import colors
import colorpreview
import fill
from brushcolor import BrushColorManager
from overlays import LastPaintPosOverlay
from overlays import ScaleOverlay
from buttonmap import ButtonMapping


class Application (object):
    """Main application singleton.

    This class serves as a global container for everything that needs
    to be shared in the GUI. Its constructor is the last part of the
    initialization, called by main.py or by the testing scripts.

    Access via `gui.application.get_app()`.

    """

    #: Singleton instance
    _INSTANCE = None


    def __init__(self, filenames, app_datapath, app_extradatapath,
                 user_datapath, user_confpath, version, fullscreen=False):
        """Construct, but do not run.

        :params filenames: The list of files to load.
          Note: only the first is used.
        :param app_datapath: App-specific read-only data area.
          Path used for UI definition XML, and the default sets of backgrounds,
          palettes, and brush defintions. Often $PREFIX/share/.
        :param app_extradatapath: Extra search path for themeable UI icons.
          This will be used in addition to $XDG_DATA_DIRS for the purposes of
          icon lookup. Normally it's $PREFIX/share, to support unusual
          installations outside the usual locations. It should contain an
          icons/ subdirectory.
        :param user_datapath: Location of the user's app-specific data.
          For MyPaint, this means the user's brushes, backgrounds, and
          scratchpads. Commonly $XDG_DATA_HOME/mypaint, i.e.
          ~/.local/share/mypaint
        :param user_confpath: Location of the user's app-specific config area.
          This is where MyPaint will save user preferences data and the
          keyboard accelerator map. Commonly $XDG_CONFIG_HOME/mypaint, i.e.
          ~/.config/mypaint
        :param version: Version string for the about dialog.
        :param fullscreen: Go fullscreen after starting.

        """
        assert Application._INSTANCE is None
        super(Application, self).__init__()
        Application._INSTANCE = self

        self.user_confpath = user_confpath #: User configs (see __init__)
        self.user_datapath = user_datapath #: User data (see __init__)

        self.datapath = app_datapath

        self.version = version  #: version string for the app.

        # create config directory, and subdirs where the user might drop files
        for basedir in [self.user_confpath, self.user_datapath]:
            if not os.path.isdir(basedir):
                os.mkdir(basedir)
                logger.info('Created basedir %r', basedir)
        for datasubdir in ['backgrounds', 'brushes', 'scratchpads']:
            datadir = os.path.join(self.user_datapath, datasubdir)
            if not os.path.isdir(datadir):
                os.mkdir(datadir)
                logger.info('Created data subdir %r', datadir)


        # Default location for our icons. The user's theme can override these.
        icon_theme = gtk.icon_theme_get_default()
        icon_theme.append_search_path(join(app_extradatapath, "icons"))

        # Icon sanity check
        if not icon_theme.has_icon('mypaint') \
                or not icon_theme.has_icon('mypaint-tool-brush'):
            logger.error('Error: Where have my icons gone?')
            logger.error('Icon search path: %r', icon_theme.get_search_path())
            logger.error("Mypaint can't run sensibly without its icons; "
                         "please check your installation. See "
                         "https://gna.org/bugs/?18460 for possible solutions")
            sys.exit(1)

        gtk.Window.set_default_icon_name('mypaint')

        # Core actions and menu structure
        resources_xml = join(self.datapath, "gui", "resources.xml")
        self.builder = gtk.Builder()
        self.builder.set_translation_domain("mypaint")
        self.builder.add_from_file(resources_xml)

        self.ui_manager = self.builder.get_object("app_ui_manager")
        signal_callback_objs = []

        gdk.set_program_class('MyPaint')

        self.pixmaps = PixbufDirectory(join(self.datapath, 'pixmaps'))
        self.cursor_color_picker = gdk.Cursor(
                  gtk2compat.gdk.display_get_default(),
                  self.pixmaps.cursor_color_picker,
                  1, 30)
        self.cursors = CursorCache(self)

        # unmanaged main brush; always the same instance (we can attach settings_observers)
        # this brush is where temporary changes (color, size...) happen
        self.brush = brush.BrushInfo()
        self.brush.load_defaults()

        # Global pressure mapping function, ignored unless set
        self.pressure_mapping = None

        self.preferences = {}
        self.load_settings()

        # Keyboard manager
        self.kbm = keyboard.KeyboardManager(self)

        # File I/O
        self.filehandler = filehandling.FileHandler(self)

        # Load the main interface
        mypaint_main_xml = join(self.datapath, "gui", "mypaint.glade")
        self.builder.add_from_file(mypaint_main_xml)

        # Main drawing window
        self.drawWindow = self.builder.get_object("drawwindow")
        signal_callback_objs.append(self.drawWindow)

        # Workspace widget. Manages layout of toolwindows, and autohide in
        # fullscreen.
        workspace = self.builder.get_object("app_workspace")
        workspace.build_from_layout(self.preferences["workspace.layout"])
        workspace.floating_window_created += self._floating_window_created_cb
        fs_autohide_action = self.builder.get_object("FullscreenAutohide")
        fs_autohide_action.set_active(workspace.autohide_enabled)
        self.workspace = workspace

        # Working document: viewer widget
        app_canvas = self.builder.get_object("app_canvas")

        # Working document: model and controller
        model = lib.document.Document(self.brush)
        self.doc = document.Document(self, app_canvas, model)
        app_canvas.set_model(model)

        signal_callback_objs.append(self.doc)
        signal_callback_objs.append(self.doc.modes)

        self.scratchpad_filename = ""
        scratchpad_model = lib.document.Document(self.brush, painting_only=True)
        scratchpad_tdw = tileddrawwidget.TiledDrawWidget()
        scratchpad_tdw.set_model(scratchpad_model)
        self.scratchpad_doc = document.Document(self, scratchpad_tdw,
                                                scratchpad_model,
                                                leader=self.doc)
        self.brushmanager = brushmanager.BrushManager(
                join(app_datapath, 'brushes'),
                join(user_datapath, 'brushes'),
                self)
        signal_callback_objs.append(self.filehandler)
        self.brushmodifier = brushmodifier.BrushModifier(self)
        signal_callback_objs.append(self.brushmodifier)
        self.line_mode_settings = linemode.LineModeSettings(self)

        # Button press mapping
        self.button_mapping = ButtonMapping()

        # Monitors changes of input device & saves device-specific brushes
        self.device_monitor = DeviceUseMonitor(self)

        if not self.preferences.get("scratchpad.last_opened_scratchpad", None):
            self.preferences["scratchpad.last_opened_scratchpad"] = self.filehandler.get_scratchpad_autosave()
        self.scratchpad_filename = self.preferences["scratchpad.last_opened_scratchpad"]

        self.brush_color_manager = BrushColorManager(self)
        self.brush_color_manager.set_picker_cursor(self.cursor_color_picker)
        self.brush_color_manager.set_data_path(self.datapath)

        #: Mapping of setting cname to a GtkAdjustment which controls the base
        #: value of that setting for the app's current brush.
        self.brush_adjustment = {}
        self.init_brush_adjustments()

        # Connect signals defined in mypaint.xml
        callback_finder = CallbackFinder(signal_callback_objs)
        self.builder.connect_signals(callback_finder)

        self.kbm.start_listening()
        self.filehandler.doc = self.doc
        self.filehandler.filename = None
        gtk2compat.gtk.accel_map_load(join(self.user_confpath,
                                            'accelmap.conf'))

        # Load the default background image if one exists
        layer_stack = self.doc.model.layer_stack
        inited_background = False
        for datapath in [self.user_datapath, self.datapath]:
            bg_path = join(datapath, backgroundwindow.BACKGROUNDS_SUBDIR,
                           backgroundwindow.DEFAULT_BACKGROUND)
            if not os.path.exists(bg_path):
                continue
            bg, errors = backgroundwindow.load_background(bg_path)
            if bg:
                layer_stack.set_background(bg, make_default=True)
                inited_background = True
                break
            else:
                logger.warning("Failed to load default background image %r",
                               bg_path)
                if errors:
                    for error in errors:
                        logger.warning("warning: %r", error)

        # Otherwise, set a fallback background colour which depends on the UI
        # brightness and isn't too glaringly odd if the user's theme doesn't
        # have dark/light variants.
        if not inited_background:
            if self.preferences["ui.dark_theme_variant"]:
                bg_color = 153, 153, 153
            else:
                bg_color = 204, 204, 204
            layer_stack.set_background(bg_color, make_default=True)
            inited_background = True

        # Non-dockable subwindows
        # Loading is deferred as late as possible
        self._subwindow_classes = {
            # action-name: action-class
            "BackgroundWindow": backgroundwindow.BackgroundWindow,
            "BrushEditorWindow": brusheditor.BrushEditorWindow,
            "PreferencesWindow": preferenceswindow.PreferencesWindow,
            "InputTestWindow": inputtestwindow.InputTestWindow,
            "BrushIconEditorWindow": brushiconeditor.BrushIconEditorWindow,
            }
        self._subwindows = {}

        # Show main UI.
        self.drawWindow.show_all()
        gobject.idle_add(self._at_application_start, filenames, fullscreen)


    def _at_application_start(self, filenames, fullscreen):
        col = self.brush_color_manager.get_color()
        self.brushmanager.select_initial_brush()
        self.brush_color_manager.set_color(col)
        if filenames:
            # Open only the first file, no matter how many has been specified
            # If the file does not exist just set it as the file to save to
            fn = filenames[0].replace('file:///', '/')
            # ^ some filebrowsers do this (should only happen with outdated
            #   mypaint.desktop)
            if not os.path.exists(fn):
                self.filehandler.filename = fn
            else:
                self.filehandler.open_file(fn)

        # Load last scratchpad
        sp_autosave_key = "scratchpad.last_opened_scratchpad"
        autosave_name = self.preferences[sp_autosave_key]
        if not autosave_name:
            autosave_name = self.filehandler.get_scratchpad_autosave()
            self.preferences[sp_autosave_key] = autosave_name
            self.scratchpad_filename = autosave_name
        if os.path.isfile(autosave_name):
            try:
                self.filehandler.open_scratchpad(autosave_name)
            except AttributeError:
                pass

        self.apply_settings()
        if not self.pressure_devices:
            logger.warning('No pressure sensitive devices found.')
        self.drawWindow.present()

        # Handle fullscreen command line option
        if fullscreen:
            self.drawWindow.fullscreen_cb()


    def save_settings(self):
        """Saves the current settings to persistent storage."""
        self.brushmanager.save_brushes_for_devices()
        self.brushmanager.save_brush_history()
        self.filehandler.save_scratchpad(self.scratchpad_filename)
        settingspath = join(self.user_confpath, 'settings.json')
        jsonstr = helpers.json_dumps(self.preferences)
        f = open(settingspath, 'w')
        f.write(jsonstr)
        f.close()


    def apply_settings(self):
        """Applies the current settings.
        """
        self.update_input_mapping()
        self.update_input_devices()
        self.update_button_mapping()
        self.preferences_window.update_ui()


    def load_settings(self):
        """Loads the settings from persistent storage.

        Uses defaults if not explicitly configured.

        """
        def get_json_config():
            settingspath = join(self.user_confpath, 'settings.json')
            jsonstr = open(settingspath).read()
            try:
                return helpers.json_loads(jsonstr)
            except Exception, e:
                logger.warning("settings.json: %s", str(e))
                logger.warning("Failed to load settings: using defaults")
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
            'ui.parts': dict(main_toolbar=True, menubar=True),
            'ui.feedback.scale': False,
            'ui.feedback.last_pos': False,
            'ui.feedback.symmetry': True,
            'ui.toolbar_items': dict(
                toolbar1_file=True,
                toolbar1_scrap=False,
                toolbar1_edit=True,
                toolbar1_blendmodes=False,
                toolbar1_linemodes=True,
                toolbar1_view_modes=True,
                toolbar1_view_manips=False,
                toolbar1_view_resets=True,
                toolbar1_subwindows=True,
            ),
            'ui.toolbar_icon_size': 'large',
            'ui.dark_theme_variant': True,
            'saving.default_format': 'openraster',
            'brushmanager.selected_brush' : None,
            'brushmanager.selected_groups' : [],
            'frame.color_rgba': (0.12, 0.12, 0.12, 0.92),
            'misc.context_restores_color': True,

            "scratchpad.last_opened_scratchpad": "",

            # Initial main window positions
            "workspace.layout": {
                "position": dict(x=50, y=32, w=-50, h=-100),
                "autohide": True,
            },

            # Linux defaults.
            # Alt is the normal window resizing/moving key these days,
            # so provide a Ctrl-based equivalent for all alt actions.
            'input.button_mapping': {
                # Note that space is treated as a fake Button2
                '<Shift>Button1':          'StraightMode',
                '<Control>Button1':        'ColorPickMode',
                '<Alt>Button1':            'ColorPickMode',
                'Button2':                 'PanViewMode',
                '<Shift>Button2':          'RotateViewMode',
                '<Control>Button2':        'ZoomViewMode',
                '<Alt>Button2':            'ZoomViewMode',
                '<Control><Shift>Button2': 'FrameEditMode',
                '<Alt><Shift>Button2':     'FrameEditMode',
                'Button3':                 'ShowPopupMenu',
            },
        }
        if sys.platform == 'win32':
            # The Linux wacom driver inverts the button numbers of the
            # pen flip button, because middle-click is the more useful
            # action on Linux. However one of the two buttons is often
            # accidentally hit with the thumb while painting. We want
            # to assign panning to this button by default.
            linux_mapping = DEFAULT_CONFIG["input.button_mapping"]
            DEFAULT_CONFIG["input.button_mapping"] = {}
            for bp, actname in linux_mapping.iteritems():
                bp = bp.replace("Button2", "ButtonTMP")
                bp = bp.replace("Button3", "Button2")
                bp = bp.replace("ButtonTMP", "Button3")
                DEFAULT_CONFIG["input.button_mapping"][bp] = actname

        self.preferences = DEFAULT_CONFIG.copy()
        try:
            user_config = get_json_config()
        except IOError:
            user_config = {}
        self.preferences.update(user_config)
        if 'ColorPickerPopup' in self.preferences["input.button_mapping"].values():
            # old config file; users who never assigned any buttons would
            # end up with Ctrl-Click color picker broken after upgrade
            self.preferences["input.button_mapping"] = DEFAULT_CONFIG["input.button_mapping"]

    def add_action_group(self, ag):
        self.ui_manager.insert_action_group(ag, -1)

    def find_action(self, name):
        for ag in self.ui_manager.get_action_groups():
            result = ag.get_action(name)
            if result is not None:
                return result


    ## Brush settings: GtkAdjustments for base values

    def init_brush_adjustments(self):
        """Initializes the base value adjustments for all brush settings"""
        assert not self.brush_adjustment
        changed_cb = self._brush_adjustment_value_changed_cb
        for s in brushsettings.settings_visible:
            adj = gtk.Adjustment(value=s.default, lower=s.min, upper=s.max,
                                 step_incr=0.01, page_incr=0.1)
            self.brush_adjustment[s.cname] = adj
            adj.connect("value-changed", changed_cb, s.cname)
        self.brush.observers.append(self._brush_modified_cb)


    def _brush_adjustment_value_changed_cb(self, adj, cname):
        """Updates a brush setting when the user tweaks it using a scale"""
        newvalue = adj.get_value()
        if self.brush.get_base_value(cname) != newvalue:
            self.brush.set_base_value(cname, newvalue)


    def _brush_modified_cb(self, settings):
        """Updates the brush's base setting adjustments on brush changes"""
        for cname in settings:
            adj = self.brush_adjustment.get(cname, None)
            if adj is None:
                continue
            value = self.brush.get_base_value(cname)
            adj.set_value(value)


    ## Button mappings, global pressure curve, input devices...

    def update_button_mapping(self):
        self.button_mapping.update(self.preferences["input.button_mapping"])


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

        logger.info('Looking for GTK devices with pressure')
        display = gtk2compat.gdk.display_get_default()
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
                    logger.info('Setting %s mode for %r',
                                mode.value_name, device.get_name())
                    device.set_mode(mode)
                # Record as a pressure-sensitive device
                self.pressure_devices.append(name)
                break
        return

        # GTK2/PyGTK (unused, but consider porting fully to GTK3)
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
            tablet_strings  = '''
            tablet pressure graphic stylus eraser pencil brush
            wacom bamboo intuos graphire cintiq
            hanvon rollick graphicpal artmaster sentip
            genius mousepen
            aiptek
            touchcontroller
            '''
            match = False
            for s in tablet_strings.split():
                if s in name:
                    match = True

            words = name.split()
            if 'pen' in words or 'art' in words:
                match = True
            if 'uc logic' in name:
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
        gtk2compat.gtk.accel_map_save(join(self.user_confpath, 'accelmap.conf'))
        workspace = self.workspace
        self.preferences["workspace.layout"] = workspace.get_layout()
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
            tv = gtk.TextView.new_with_buffer(buf)
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
        # Due to a performance bug, color picking can take more time
        # than we have between two motion events (about 8ms).
        if hasattr(self, 'delayed_color_pick_id'):
            gobject.source_remove(self.delayed_color_pick_id)

        def delayed_color_pick():
            del self.delayed_color_pick_id
            color = colors.get_color_at_pointer(widget.get_display(), size)
            self.brush_color_manager.set_color(color)

        self.delayed_color_pick_id = gobject.idle_add(delayed_color_pick)


    ## Subwindows

    @property
    def background_window(self):
        """The background switcher subwindow."""
        return self.get_subwindow("BackgroundWindow")


    @property
    def brush_settings_window(self):
        """The brush settings editor subwindow."""
        return self.get_subwindow("BrushSettingsWindow")


    @property
    def brush_icon_editor_window(self):
        """The brush icon editor subwindow."""
        return self.get_subwindow("BrushIconEditorWindow")

    @property
    def brush_icon_editor_window(self):
        """The brush editor subwindow."""
        return self.get_subwindow("BrushEditorWindow")


    @property
    def preferences_window(self):
        """The preferences subwindow."""
        return self.get_subwindow("PreferencesWindow")


    @property
    def input_test_window(self):
        """The input test window."""
        return self.get_subwindow("InputTestWindow")


    def get_subwindow(self, name):
        """Get a subwindow by its name."""
        if name in self._subwindows:
            window = self._subwindows[name]
        elif name in self._subwindow_classes:
            window_class = self._subwindow_classes[name]
            window = window_class()
            window.__toggle_action = self.find_action(name)
            window.connect("hide", self._subwindow_hide_cb)
            self._subwindows[name] = window
        else:
            raise ValueError, "Unkown subwindow %r" % (name,)
        return window


    def has_subwindow(self, name):
        """True if the named subwindow is known."""
        return name in self._subwindow_classes


    def _subwindow_hide_cb(self, subwindow):
        """Toggles off a subwindow's related action when it's hidden."""
        action = subwindow.__toggle_action
        if action and action.get_active():
            action.set_active(False)


    ## Special UI areas

    @property
    def statusbar(self):
        """Returns the application statusbar."""
        return self.builder.get_object("app_statusbar")


    ## Workspace callbacks

    def _floating_window_created_cb(self, workspace, floatwin):
        """Adds newly created `workspace.ToolStackWindow`s to the kbm."""
        self.kbm.add_window(floatwin)



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

        if gtk2compat.USE_GTK3:
            new_device.name = new_device.props.name
            new_device.source = new_device.props.input_source

        logger.debug('Device change: name=%r source=%s',
                     new_device.name, new_device.source.value_name)

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


class CursorCache (object):
    """Cache of custom cursors for actions."""

    # Known cursor names and their hot pixels
    CURSOR_HOTSPOTS = {
        "cursor_arrow": (1, 1),
        "cursor_arrow_move": (1, 1),
        "cursor_pencil": (7, 22),
        "cursor_hand_open": (11, 12),
        "cursor_hand_closed": (11, 12),
        "cursor_crosshair_open": (11, 11),
        "cursor_crosshair_closed": (11, 11),
        "cursor_crosshair_precise_open": (12, 11),
        "cursor_move_n_s": (11, 11),
        "cursor_move_w_e": (11, 11),
        "cursor_move_nw_se": (11, 11),
        "cursor_move_ne_sw": (11, 11),
        "cursor_forbidden_everywhere": (11, 11),
        "cursor_arrow_forbidden": (7, 4),
        "cursor_arrow": (7, 4),
    }

    def __init__(self, app):
        object.__init__(self)
        self.app = app
        self.cache = {}


    def get_overlay_cursor(self, icon_pixbuf, cursor_name="cursor_arrow"):
        """Returns an overlay cursor. Not cached.

        :param icon_pixbuf: a gdk.Pixbuf containing a small (~22px) image,
           or None
        :param cursor_name: name of a pixmaps/ cursor image to use for the
           pointer part, minus the .png

        The overlay icon will be overlaid to the bottom and right of the
        returned cursor image.

        """

        pointer_pixbuf = getattr(self.app.pixmaps, cursor_name)
        pointer_w = pointer_pixbuf.get_width()
        pointer_h = pointer_pixbuf.get_height()
        hot_x, hot_y = self.CURSOR_HOTSPOTS.get(cursor_name, (None, None))
        if hot_x is None:
            hot_x = 1
            hot_y = 1

        cursor_pixbuf = gtk2compat.GdkPixbufCompat.new(gdk.COLORSPACE_RGB,
                                                        True, 8, 32, 32)
        cursor_pixbuf.fill(0x00000000)

        pointer_pixbuf.composite(cursor_pixbuf, 0, 0, pointer_w, pointer_h,
                               0, 0, 1, 1, gdk.INTERP_NEAREST, 255)
        if icon_pixbuf is not None:
            icon_w = icon_pixbuf.get_width()
            icon_h = icon_pixbuf.get_height()
            icon_x = 32 - icon_w
            icon_y = 32 - icon_h
            icon_pixbuf.composite(cursor_pixbuf, icon_x, icon_y, icon_w, icon_h,
                                  icon_x, icon_y, 1, 1, gdk.INTERP_NEAREST, 255)

        display = self.app.drawWindow.get_display()
        cursor = gdk.Cursor(display, cursor_pixbuf, hot_x, hot_y)
        return cursor


    def get_pixmaps_cursor(self, pixmap_name, cursor_name="cursor_arrow"):
        """Returns an overlay cursor for a named PNG in pixmaps/. Cached.

        :param pixmap_name: the name of a file in pixmaps/, minus the .png,
           containing a small (~22px) image, or None
        :param cursor_name: name of a pixmaps/ cursor image to use for the
           pointer part, minus the .png

        """
        # Return from cache, if we have an entry
        cache_key = ("pixmaps", pixmap_name, cursor_name)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Build cursor
        if pixmap_name is Npne:
            pixbuf = None
        else:
            pixbuf = getattr(self.app.pixmaps, pixmap_name)
        cursor = self.get_overlay_cursor(pixbuf, cursor_name)

        # Cache and return
        self.cache[cache_key] = cursor
        return cursor


    def get_freehand_cursor(self, cursor_name="cursor_crosshair_precise_open"):
        """Returns a cursor for the current app.brush. Cached.

        :param cursor_name: name of a pixmaps/ image to use, minus the .png

        An icon for the brush blend mode will be overlaid to the bottom and
        right of the cursor image.

        """
        # Pick an icon
        if self.app.brush.is_eraser():
            icon_name = "mypaint-eraser-symbolic"
        elif self.app.brush.is_alpha_locked():
            icon_name = "mypaint-lock-alpha-symbolic"
        elif self.app.brush.is_colorize():
            icon_name = "mypaint-colorize-symbolic"
        else:
            icon_name = None
        return self.get_icon_cursor(icon_name, cursor_name)


    def get_action_cursor(self, action_name, cursor_name="cursor_arrow"):
        """Returns an overlay cursor for a named action. Cached.

        :param action_name: the name of a GtkAction defined in mypaint.xml
        :param cursor_name: name of a pixmaps/ image to use, minus the .png

        The action's icon will be overlaid at a small size to the bottom and
        right of the cursor image.

        """
        # Find a small action icon for the overlay
        action = self.app.find_action(action_name)
        if action is None:
            return gdk.Cursor(gdk.BOGOSITY)
        icon_name = action.get_icon_name()
        if icon_name is None:
            return gdk.Cursor(gdk.BOGOSITY)
        return self.get_icon_cursor(icon_name, cursor_name)


    def get_icon_cursor(self, icon_name, cursor_name="cursor_arrow"):
        """Returns an overlay cursor for a named icon. Cached.

        :param icon_name: themed icon system name.
        :param cursor_name: name of a pixmaps/ image to use, minus the .png

        The icon will be overlaid at a small size to the bottom and right of
        the cursor image.

        """

        # Return from cache, if we have an entry
        cache_key = ("actions", icon_name, cursor_name)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if icon_name is not None:
            # Look up icon via the user's current theme
            icon_theme = gtk.icon_theme_get_default()
            for icon_size in gtk.ICON_SIZE_SMALL_TOOLBAR, gtk.ICON_SIZE_MENU:
                valid, width, height = gtk.icon_size_lookup(icon_size)
                if not valid:
                    continue
                size = min(width, height)
                if size > 24:
                    continue
                flags = 0
                icon_pixbuf = icon_theme.load_icon(icon_name, size, flags)
                if icon_pixbuf:
                    break
            if not icon_pixbuf:
                logger.warning("Can't find icon %r for cursor: search path=%r",
                               icon_name)
                logger.debug("Search path: %r", icon_theme.get_search_path())
        else:
            icon_pixbuf = None

        # Build cursor
        cursor = self.get_overlay_cursor(icon_pixbuf, cursor_name)

        # Cache and return
        self.cache[cache_key] = cursor
        return cursor



class CallbackFinder:
    """Finds callbacks amongst a list of objects.

    It's not possible to call `GtkBuilder.connect_signals()` more than once,
    but we use more tnan one backend object. Thus, this little workaround is
    necessary during construction.

    See http://stackoverflow.com/questions/4637792

    """

    def __init__(self, objects):
        self._objs = list(objects)

    def __getattr__(self, name):
        name = str(name)
        found = [getattr(obj, name) for obj in self._objs
                  if hasattr(obj, name)]
        if len(found) == 1:
            return found[0]
        elif len(found) > 1:
            logger.warning("ambiguity: %r resolves to %r", name, found)
            logger.warning("using first match only.")
            return found[0]
        else:
            raise AttributeError, \
                ( "No method named %r was defined on any of %r"
                  % (name, self._objs) )

