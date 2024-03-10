# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2019 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from logging import getLogger

from lib.gibindings import Gtk

from . import compatconfig as config

from .compatconfig import C1X, C2X, COMPAT_SETTINGS, DEFAULT_COMPAT

import lib.eotf
from lib.layer.data import BackgroundLayer
from lib.meta import Compatibility, PREREL, MYPAINT_VERSION
from lib.modes import MODE_STRINGS, set_default_mode
from lib.mypaintlib import CombineNormal, CombineSpectralWGM
from lib.mypaintlib import combine_mode_get_info
from lib.gettext import C_

logger = getLogger(__name__)

FILE_WARNINGS = {
    Compatibility.INCOMPATIBLE: 'ui.file_compat_warning_severe',
    Compatibility.PARTIALLY: 'ui.file_compat_warning_mild',
}

_FILE_OPEN_OPTIONS = [
    ('', C_("File Load Compat Options", "Based on file")),
    (C1X, C_("Prefs Dialog|Compatibility", "1.x")),
    (C2X, C_("Prefs Dialog|Compatibility", "2.x")),
]

FILE_WARNING_MSGS = {
    Compatibility.INCOMPATIBLE: C_(
        "file compatibility warning",
        # TRANSLATORS: This is probably a rare warning, and it will not
        # TRANSLATORS: really be shown at all before the release of 3.0
        u"“{filename}” was saved with <b>MyPaint {new_version}</b>."
        " It may be <b>incompatible</b> with <b>MyPaint {current_version}</b>."
        "\n\n"
        "Editing this file with this version of MyPaint is not guaranteed"
        " to work, and may even result in crashes."
        "\n\n"
        "It is <b>strongly recommended</b> to upgrade to <b>MyPaint"
        " {new_version}</b> or newer if you want to edit this file!"),
    Compatibility.PARTIALLY: C_(
        "file compatibility warning",
        u"“{filename}” was saved with <b>MyPaint {new_version}</b>. "
        "It may not be fully compatible with <b>Mypaint {current_version}</b>."
        "\n\n"
        "Saving it with this version of MyPaint may result in data"
        " that is only supported by the newer version being lost."
        "\n\n"
        "To be safe you should upgrade to MyPaint {new_version} or newer."),
}

OPEN_ANYWAY = C_(
    "file compatibility question",
    "Do you want to open this file anyway?"
)

_PIGMENT_OP = combine_mode_get_info(CombineSpectralWGM)['name']


def has_pigment_layers(elem):
    """Check if the layer stack xml contains a pigment layer

    Has to be done before any layers are loaded, since the
    correct eotf value needs to set before loading the tiles.
    """
    # Ignore the composite op of the background.
    # We only need to check for the namespaced attribute, as
    # any file containing the non-namespaced counterpart was
    # created prior (version-wise) to pigment layers.
    bg_attr = BackgroundLayer.ORA_BGTILE_ATTR
    if elem.get(bg_attr, None):
        return False
    op = elem.attrib.get('composite-op', None)
    return op == _PIGMENT_OP or any([has_pigment_layers(c) for c in elem])


def incompatible_ora_cb(app):
    def cb(comp_type, prerel, filename, target_version):
        """ Internal: callback that may show a confirmation/warning dialog

        Unless disabled in settings, when a potentially
        incompatible ora is opened, a warning dialog is
        shown, allowing users to cancel the loading.

        """
        if comp_type == Compatibility.FULLY:
            return True
        logger.warning(
            "Loaded file “{filename}” may be {compat_desc}!\n"
            "App version: {version}, File version: {file_version}".format(
                filename=filename,
                compat_desc=Compatibility.DESC[comp_type],
                version=lib.meta.MYPAINT_VERSION,
                file_version=target_version
            ))
        if prerel and comp_type > Compatibility.INCOMPATIBLE and PREREL != '':
            logger.info("Warning dialog skipped in prereleases.")
            return True
        return incompatible_ora_warning_dialog(
            comp_type, prerel, filename, target_version, app)
    return cb


def incompatible_ora_warning_dialog(
        comp_type, prerel, filename, target_version, app):
    # Skip the dialog if the user has disabled the warning
    # for this level of incompatibility
    warn = app.preferences.get(FILE_WARNINGS[comp_type], True)
    if not warn:
        return True

    # Toggle allowing users to disable future warnings directly
    # in the dialog, this is configurable in the settings too.
    # The checkbutton code is pretty much copied from the filehandling
    # save-to-scrap checkbutton; a lot of duplication.
    skip_warning_text = C_(
        "Version compat warning toggle",
        u"Don't show this warning again"
    )
    skip_warning_button = Gtk.CheckButton.new()
    skip_warning_button.set_label(skip_warning_text)
    skip_warning_button.set_hexpand(False)
    skip_warning_button.set_vexpand(False)
    skip_warning_button.set_halign(Gtk.Align.END)
    skip_warning_button.set_margin_top(12)
    skip_warning_button.set_margin_bottom(12)
    skip_warning_button.set_margin_start(12)
    skip_warning_button.set_margin_end(12)
    skip_warning_button.set_can_focus(False)

    def skip_warning_toggled(checkbut):
        app.preferences[FILE_WARNINGS[comp_type]] = not checkbut.get_active()
        app.preferences_window.compat_preferences.update_ui()
    skip_warning_button.connect("toggled", skip_warning_toggled)

    def_msg = "Invalid key, report this! key={key}".format(key=comp_type)
    msg_markup = FILE_WARNING_MSGS.get(comp_type, def_msg).format(
        filename=filename,
        new_version=target_version,
        current_version=MYPAINT_VERSION
    ) + "\n\n" + OPEN_ANYWAY
    d = Gtk.MessageDialog(
        transient_for=app.drawWindow,
        buttons=Gtk.ButtonsType.NONE,
        modal=True,
        message_type=Gtk.MessageType.WARNING,
    )
    d.set_markup(msg_markup)

    vbox = d.get_content_area()
    vbox.set_spacing(0)
    vbox.set_margin_top(12)
    vbox.pack_start(skip_warning_button, False, True, 0)

    d.add_button(Gtk.STOCK_NO, Gtk.ResponseType.REJECT)
    d.add_button(Gtk.STOCK_YES, Gtk.ResponseType.ACCEPT)
    d.set_default_response(Gtk.ResponseType.REJECT)

    # Without this, the check button takes initial focus
    def show_checkbut(*args):
        skip_warning_button.show()
        skip_warning_button.set_can_focus(True)
    d.connect("show", show_checkbut)

    response = d.run()
    d.destroy()
    return response == Gtk.ResponseType.ACCEPT


class CompatFileBehavior(config.CompatFileBehaviorConfig):
    """ Holds data and functions related to per-file choice of compat mode
    """
    _CFBC = config.CompatFileBehaviorConfig
    _OPTIONS = [
        _CFBC.ALWAYS_1X,
        _CFBC.ALWAYS_2X,
        _CFBC.UNLESS_PIGMENT_LAYER_1X,
    ]
    _LABELS = {
        _CFBC.ALWAYS_1X: (
            C_(
                "Prefs Dialog|Compatibility",
                # TRANSLATORS: One of the options for the
                # TRANSLATORS: "When Not Specified in File"
                # TRANSLATORS: compatibility setting.
                "Always open in 1.x mode"
            )
        ),
        _CFBC.ALWAYS_2X: (
            C_(
                "Prefs Dialog|Compatibility",
                # TRANSLATORS: One of the options for the
                # TRANSLATORS: "When Not Specified in File"
                # TRANSLATORS: compatibility setting.
                "Always open in 2.x mode"
            )
        ),
        _CFBC.UNLESS_PIGMENT_LAYER_1X: (
            C_(
                "Prefs Dialog|Compatibility",
                # TRANSLATORS: One of the options for the
                # TRANSLATORS: "When Not Specified in File"
                # TRANSLATORS: compatibility setting.
                "Open in 1.x mode unless file contains pigment layers"
            )
        ),
    }

    def __init__(self, combobox, prefs):
        self.combo = combobox
        self.prefs = prefs
        options_store = Gtk.ListStore()
        options_store.set_column_types((str, str))
        for option in self._OPTIONS:
            options_store.append((option, self._LABELS[option]))
        combobox.set_model(options_store)

        cell = Gtk.CellRendererText()
        combobox.pack_start(cell, True)
        combobox.add_attribute(cell, 'text', 1)
        self.update_ui()
        combobox.connect('changed', self.changed_cb)

    def update_ui(self):
        self.combo.set_active_id(self.prefs[self.SETTING])

    def changed_cb(self, combo):
        active_id = self.combo.get_active_id()
        self.prefs[self.SETTING] = active_id

    @staticmethod
    def get_compat_mode(setting, root_elem, default):
        """ Get the compat mode to use for a file

        The decision is based on the given file behavior setting
        and the layer stack xml.
        """
        # If more options are added, rewrite to use separate classes.
        if setting == CompatFileBehavior.ALWAYS_1X:
            return C1X
        elif setting == CompatFileBehavior.ALWAYS_2X:
            return C2X
        elif setting == CompatFileBehavior.UNLESS_PIGMENT_LAYER_1X:
            if has_pigment_layers(root_elem):
                logger.info("Pigment layer found!")
                return C2X
            else:
                return C1X
        else:
            msg = "Unknown file compat setting: {setting}, using default mode."
            logger.warning(msg.format(setting=setting))
            return default


class CompatibilityPreferences:
    """ A single instance should be a part of the preference window

    This class handles preferences related to the compatibility modes
    and their settings.
    """

    def __init__(self, app, builder):
        self.app = app
        self._builder = builder
        # Widget references
        getobj = builder.get_object
        # Default compat mode choice radio buttons
        self.default_radio_1_x = getobj('compat_1_x_radiobutton')
        self.default_radio_2_x = getobj('compat_2_x_radiobutton')
        # For each mode, choice for whether pigment is on or off by default
        self.pigment_switch_1_x = getobj('pigment_setting_switch_1_x')
        self.pigment_switch_2_x = getobj('pigment_setting_switch_2_x')
        # For each mode, choice of which layer type is the default
        self.pigment_radio_1_x = getobj('def_new_layer_pigment_1_x')
        self.pigment_radio_2_x = getobj('def_new_layer_pigment_2_x')
        self.normal_radio_1_x = getobj('def_new_layer_normal_1_x')
        self.normal_radio_2_x = getobj('def_new_layer_normal_2_x')

        def file_warning_cb(level):
            def cb(checkbut):
                app.preferences[FILE_WARNINGS[level]] = checkbut.get_active()
            return cb

        self.file_warning_mild = getobj('file_compat_warning_mild')
        self.file_warning_mild.connect(
            "toggled", file_warning_cb(Compatibility.PARTIALLY))

        self.file_warning_severe = getobj('file_compat_warning_severe')
        self.file_warning_severe.connect(
            "toggled", file_warning_cb(Compatibility.INCOMPATIBLE))

        self.compat_file_behavior = CompatFileBehavior(
            getobj('compat_file_behavior_combobox'), self.app.preferences)
        # Initialize widgets and callbacks
        self.setup_layer_type_strings()
        self.setup_widget_callbacks()

    def setup_widget_callbacks(self):
        """ Hook up callbacks for switches and radiobuttons
        """
        # Convenience wrapper - here it is enough to act when toggling on,
        # so ignore callbacks triggered by radio buttons being toggled off.
        def ignore_detoggle(cb_func):
            def cb(btn, *args):
                if btn.get_active():
                    cb_func(btn, *args)
            return cb

        # Connect default layer type toggles
        layer_type_cb = ignore_detoggle(self.set_compat_layer_type_cb)
        self.normal_radio_1_x.connect('toggled', layer_type_cb, C1X, False)
        self.pigment_radio_1_x.connect('toggled', layer_type_cb, C1X, True)
        self.normal_radio_2_x.connect('toggled', layer_type_cb, C2X, False)
        self.pigment_radio_2_x.connect('toggled', layer_type_cb, C2X, True)

        def_compat_cb = ignore_detoggle(self.set_default_compat_mode_cb)
        self.default_radio_1_x.connect('toggled', def_compat_cb, C1X)
        self.default_radio_2_x.connect('toggled', def_compat_cb, C2X)

        pigment_switch_cb = self.default_pigment_changed_cb
        self.pigment_switch_1_x.connect('state-set', pigment_switch_cb, C1X)
        self.pigment_switch_2_x.connect('state-set', pigment_switch_cb, C2X)

    def setup_layer_type_strings(self):
        """ Replace the placeholder labels and add tooltips
        """
        def string_setup(widget, label, tooltip):
            widget.set_label(label)
            widget.set_tooltip_text(tooltip)

        normal_label, normal_tooltip = MODE_STRINGS[CombineNormal]
        string_setup(self.normal_radio_1_x, normal_label, normal_tooltip)
        string_setup(self.normal_radio_2_x, normal_label, normal_tooltip)
        pigment_label, pigment_tooltip = MODE_STRINGS[CombineSpectralWGM]
        string_setup(self.pigment_radio_1_x, pigment_label, pigment_tooltip)
        string_setup(self.pigment_radio_2_x, pigment_label, pigment_tooltip)

    def update_ui(self):
        prefs = self.app.preferences
        # File warnings update (can be changed from confirmation dialogs)
        self.file_warning_mild.set_active(
            prefs.get(FILE_WARNINGS[Compatibility.PARTIALLY], True))
        self.file_warning_severe.set_active(
            prefs.get(FILE_WARNINGS[Compatibility.INCOMPATIBLE], True))

        # Even in a radio button group with 2 widgets, using set_active(False)
        # will not toggle the other button on, hence this ugly pattern.
        if prefs.get(DEFAULT_COMPAT, C2X) == C1X:
            self.default_radio_1_x.set_active(True)
        else:
            self.default_radio_2_x.set_active(True)
        mode_settings = prefs[COMPAT_SETTINGS]
        # 1.x
        self.pigment_switch_1_x.set_active(
            mode_settings[C1X][config.PIGMENT_BY_DEFAULT])
        if mode_settings[C1X][config.PIGMENT_LAYER_BY_DEFAULT]:
            self.pigment_radio_1_x.set_active(True)
        else:
            self.normal_radio_1_x.set_active(True)
        # 2.x
        self.pigment_switch_2_x.set_active(
            mode_settings[C2X][config.PIGMENT_BY_DEFAULT])
        if mode_settings[C2X][config.PIGMENT_LAYER_BY_DEFAULT]:
            self.pigment_radio_2_x.set_active(True)
        else:
            self.normal_radio_2_x.set_active(True)

    def _update_prefs(self, mode, setting, value):
        prefs = self.app.preferences
        prefs[COMPAT_SETTINGS][mode].update({setting: value})

    # Widget callbacks

    def set_default_compat_mode_cb(self, radiobutton, compat_mode):
        self.app.preferences[DEFAULT_COMPAT] = compat_mode

    def set_compat_layer_type_cb(self, btn, mode, use_pigment):
        self._update_prefs(mode, config.PIGMENT_LAYER_BY_DEFAULT, use_pigment)
        update_default_layer_type(self.app)

    def default_pigment_changed_cb(self, switch, use_pigment, mode):
        self._update_prefs(mode, config.PIGMENT_BY_DEFAULT, use_pigment)
        update_default_pigment_setting(self.app)


def ora_compat_handler(app):
    def handler(eotf_value, root_stack_elem):
        default = app.preferences[DEFAULT_COMPAT]
        if eotf_value is not None:
            try:
                eotf_value = float(eotf_value)
                compat = C1X if eotf_value == 1.0 else C2X
            except ValueError:
                msg = "Invalid eotf: {eotf}, using default compat mode!"
                logger.warning(msg.format(eotf=eotf_value))
                eotf_value = None
                compat = default
        else:
            logger.info("No eotf value specified in openraster file")
            # Depending on user settings, decide whether to
            # use the default value for the eotf, or the legacy value of 1.0
            setting = app.preferences[CompatFileBehavior.SETTING]
            compat = CompatFileBehavior.get_compat_mode(
                setting, root_stack_elem, default)
        set_compat_mode(app, compat, custom_eotf=eotf_value)
    return handler


def set_compat_mode(app, compat_mode, custom_eotf=None, update=True):
    """Set compatibility mode

    Set compatibility mode and update associated settings;
    default pigment brush setting and default layer type.
    If the "update" keyword is set to False, the settings
    are not updated.

    If the compatibility mode is changed, the scratchpad is
    saved and reloaded under the new mode settings.
    """
    if compat_mode not in {C1X, C2X}:
        compat_mode = C2X
        msg = "Unknown compatibility mode: '{mode}'! Using 2.x instead."
        logger.warning(msg.format(mode=compat_mode))
    changed = compat_mode != app.compat_mode
    app.compat_mode = compat_mode
    # Save scratchpad (with current eotf)
    if update and changed:
        app.drawWindow.save_current_scratchpad_cb(None)
    # Change eotf and set new compat mode
    if compat_mode == C1X:
        logger.info("Setting mode to 1.x (legacy)")
        lib.eotf.set_eotf(1.0)
    else:
        logger.info("Setting mode to 2.x (standard)")
        lib.eotf.set_eotf(custom_eotf or lib.eotf.base_eotf())
    if update and changed:
        # Reload scratchpad (with new eotf)
        app.drawWindow.revert_current_scratchpad_cb(None)
        for f in app.brush.observers:
            f({'color_h', 'color_s', 'color_v'})
        update_default_layer_type(app)
        update_default_pigment_setting(app)


def update_default_layer_type(app):
    """Update default layer type from settings
    """
    prefs = app.preferences
    mode_settings = prefs[COMPAT_SETTINGS][app.compat_mode]
    if mode_settings[config.PIGMENT_LAYER_BY_DEFAULT]:
        logger.info("Setting default layer type to Pigment")
        set_default_mode(CombineSpectralWGM)
    else:
        logger.info("Setting default layer type to Normal")
        set_default_mode(CombineNormal)


def update_default_pigment_setting(app):
    """Update default pigment brush setting value
    """
    prefs = app.preferences
    mode_settings = prefs[COMPAT_SETTINGS][app.compat_mode]
    app.brushmanager.set_pigment_by_default(
        mode_settings[config.PIGMENT_BY_DEFAULT]
    )


class CompatSelector:
    """ A dropdown menu with file loading compatibility options

    If a file was accidentally set to use the wrong mode, these
    options are used to force opening in a particular mode.
    """

    def __init__(self, app):
        self.app = app
        combo = Gtk.ComboBox()
        store = Gtk.ListStore()
        store.set_column_types((str, str))
        for k, v in _FILE_OPEN_OPTIONS:
            store.append((k, v))
        combo.set_model(store)
        combo.set_active(0)
        cell = Gtk.CellRendererText()
        combo.pack_start(cell, True)
        combo.add_attribute(cell, 'text', 1)
        combo_label = Gtk.Label(
            # TRANSLATORS: This is a label for a dropdown menu in the
            # TRANSLATORS: file chooser dialog when loading .ora files.
            label=C_("File Load Compat Options", "Compatibility mode:")
        )
        hbox = Gtk.HBox()
        hbox.set_spacing(6)
        hbox.pack_start(combo_label, False, False, 0)
        hbox.pack_start(combo, False, False, 0)
        hbox.show_all()
        hbox.set_visible(False)
        self._compat_override = None
        self._combo = combo
        combo.connect('changed', self._combo_changed_cb)
        self._widget = hbox

    def _combo_changed_cb(self, combo):
        idx = combo.get_active()
        if idx >= 0:
            self._compat_override = _FILE_OPEN_OPTIONS[idx][0]
        else:
            self._compat_override = None

    def file_selection_changed_cb(self, chooser):
        """ Show/hide widget and enable/disable override
        """
        fn = chooser.get_filename()
        applicable = fn is not None and fn.endswith('.ora')
        self.widget.set_visible(applicable)
        if not applicable:
            self._compat_override = None
        else:
            self._combo_changed_cb(self._combo)

    @property
    def widget(self):
        return self._widget

    @property
    def compat_function(self):
        """ Returns an overriding compatibility handler or None
        """
        if self._compat_override:
            return lambda *a: set_compat_mode(self.app, self._compat_override)
