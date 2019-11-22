# This file is part of MyPaint.
# Copyright (C) 2019 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from logging import getLogger

from gi.repository import Gtk

import lib.eotf
from lib.layer.data import BackgroundLayer
from lib.modes import MODE_STRINGS, set_default_mode
from lib.mypaintlib import CombineNormal, CombineSpectralWGM
from lib.mypaintlib import combine_mode_get_info
from lib.gettext import C_

logger = getLogger(__name__)

# Keys for settings in the user preferences
DEFAULT_COMPAT = 'default_compatibility_mode'
COMPAT_SETTINGS = 'compability_settings'

# Keys for compat mode sub-options in the user preferences
PIGMENT_BY_DEFAULT = 'pigment_on_by_default'
PIGMENT_LAYER_BY_DEFAULT = 'pigment_layer_is_default'

C1X = '1.x'
C2X = '2.x'


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


class CompatFileBehavior:
    """ Holds data and functions related to per-file choice of compat mode
    """

    # Key for the behavior setting in user preferences
    SETTING = 'compat_behavior_when_unknown'

    # Setting options
    ALWAYS_1X = 'always-1.x'
    ALWAYS_2X = 'always-2.x'
    UNLESS_PIGMENT_LAYER_1X = 'unless-pigment-layer-1.x'

    _OPTIONS = [
        ALWAYS_1X,
        ALWAYS_2X,
        UNLESS_PIGMENT_LAYER_1X,
    ]
    _LABELS = {
        ALWAYS_1X: (
            C_(
                "Prefs Dialog|Compatibility",
                # TRANSLATORS: One of the options for the
                # TRANSLATORS: "When Not Specified in File"
                # TRANSLATORS: compatibility setting.
                "Always open in 1.x mode"
            )
        ),
        ALWAYS_2X: (
            C_(
                "Prefs Dialog|Compatibility",
                # TRANSLATORS: One of the options for the
                # TRANSLATORS: "When Not Specified in File"
                # TRANSLATORS: compatibility setting.
                "Always open in 2.x mode"
            )
        ),
        UNLESS_PIGMENT_LAYER_1X: (
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


# Default compatibility settings
DEFAULT_CONFIG = {
    CompatFileBehavior.SETTING: CompatFileBehavior.UNLESS_PIGMENT_LAYER_1X,
    DEFAULT_COMPAT: C2X,
    COMPAT_SETTINGS: {
        C1X: {
            PIGMENT_BY_DEFAULT: False,
            PIGMENT_LAYER_BY_DEFAULT: False,
        },
        C2X: {
            PIGMENT_BY_DEFAULT: True,
            PIGMENT_LAYER_BY_DEFAULT: True,
        },
    },
}


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
        # Even in a radio button group with 2 widgets, using set_active(False)
        # will not toggle the other button on, hence this ugly pattern.
        if prefs.get(DEFAULT_COMPAT, C2X) == C1X:
            self.default_radio_1_x.set_active(True)
        else:
            self.default_radio_2_x.set_active(True)
        mode_settings = prefs[COMPAT_SETTINGS]
        # 1.x
        self.pigment_switch_1_x.set_active(
            mode_settings[C1X][PIGMENT_BY_DEFAULT])
        if mode_settings[C1X][PIGMENT_LAYER_BY_DEFAULT]:
            self.pigment_radio_1_x.set_active(True)
        else:
            self.normal_radio_1_x.set_active(True)
        # 2.x
        self.pigment_switch_2_x.set_active(
            mode_settings[C2X][PIGMENT_BY_DEFAULT])
        if mode_settings[C2X][PIGMENT_LAYER_BY_DEFAULT]:
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
        self._update_prefs(mode, PIGMENT_LAYER_BY_DEFAULT, use_pigment)
        update_default_layer_type(self.app)

    def default_pigment_changed_cb(self, switch, use_pigment, mode):
        self._update_prefs(mode, PIGMENT_BY_DEFAULT, use_pigment)
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
        update_default_layer_type(app)
        update_default_pigment_setting(app)


def update_default_layer_type(app):
    """Update default layer type from settings
    """
    prefs = app.preferences
    mode_settings = prefs[COMPAT_SETTINGS][app.compat_mode]
    if mode_settings[PIGMENT_LAYER_BY_DEFAULT]:
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
        mode_settings[PIGMENT_BY_DEFAULT]
    )
