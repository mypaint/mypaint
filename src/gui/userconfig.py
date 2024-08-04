# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2007-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2013-2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os
import sys
import json
import logging

from lib.gibindings import GLib

import lib.glib
from lib.eotf import DEFAULT_EOTF
from gui.compatconfig import DEFAULT_CONFIG as COMPAT_CONFIG

logger = logging.getLogger(__name__)


def get_json_config(conf_path):
    """Return user settings read from the settings file
    :param conf_path: The path to the directory containing settings.json
    :type conf_path: str
    :returns: Dict with settings, or the empty dict if the settings file
    cannot be found/read or parsed.
    """
    settingspath = os.path.join(conf_path, u'settings.json')
    logger.debug("Reading app settings from %r", settingspath)
    try:
        # Py3: settings.json has always been UTF-8 even in Py2.
        #
        # The Travis-CI json.loads() from Python 3.4.0 needs
        # unicode strings, always. Later and earlier versions,
        # including Py2 do not need that, if bytes are UTF-8.
        with open(settingspath, "rb") as fp:
            return json.loads(fp.read().decode("utf-8"))
    except IOError:
        logger.warning("Failed to load settings file: %s", settingspath)
    except Exception as e:
        logger.warning("settings.json: %s", str(e))
    logger.warning("Failed to load settings: using defaults")
    return {}


def default_configuration():
    """ Return the default settings
    A subset of these settings are platform dependent.
    """
    if sys.platform == 'win32':
        ud_docs = lib.glib.get_user_special_dir(
            GLib.UserDirectory.DIRECTORY_DOCUMENTS,
        )
        scrappre = os.path.join(ud_docs, u'MyPaint', u'scrap')
    else:
        scrappre = u'~/MyPaint/scrap'
    default_config = {
        'saving.scrap_prefix': scrappre,
        'input.device_mode': 'screen',
        'input.global_pressure_mapping': [(0.0, 1.0), (1.0, 0.0)],
        'input.use_barrel_rotation': True,
        'input.barrel_rotation_subtract_ascension': True,
        'input.barrel_rotation_offset': 0.5,
        'view.default_zoom': 1.0,
        'view.real_alpha_checks': True,
        'ui.hide_menubar_in_fullscreen': True,
        'ui.hide_toolbar_in_fullscreen': True,
        'ui.hide_subwindows_in_fullscreen': True,
        'ui.parts': dict(main_toolbar=True, menubar=True),
        'ui.feedback.scale': False,
        'ui.feedback.last_pos': False,
        'ui.toolbar_items': dict(
            toolbar1_file=True,
            toolbar1_scrap=False,
            toolbar1_edit=True,
            toolbar1_blendmodes=False,
            toolbar1_linemodes=True,
            toolbar1_view_modes=True,
            toolbar1_view_manips=False,
            toolbar1_view_resets=True,
        ),
        'ui.toolbar_icon_size': 'large',
        'ui.dark_theme_variant': True,
        'ui.rendered_tile_cache_size': 16384,
        'saving.default_format': 'openraster',
        'brushmanager.selected_brush': None,
        'brushmanager.selected_groups': [],
        'frame.color_rgba': (0.12, 0.12, 0.12, 0.92),
        'misc.context_restores_color': True,

        'document.autosave_backups': True,
        'document.autosave_interval': 10,

        # configurable EOTF.  Set to 1.0 for legacy non-linear behaviour
        'display.colorspace_EOTF': DEFAULT_EOTF,
        'display.colorspace': "srgb",
        # sRGB is a good default even for OS X since v10.6 / Snow
        # Leopard: http://support.apple.com/en-us/HT3712.
        # Version 10.6 was released in September 2009.

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
            # It is time to free up the modifiers and Button1
            # '<Shift>Button1': 'StraightMode',
            # '<Control>Button1': 'ColorPickMode',
            # '<Alt>Button1': 'ColorPickMode',
            'Button2': 'PanViewMode',
            '<Shift>Button2': 'RotateViewMode',
            '<Control>Button2': 'ZoomViewMode',
            '<Alt>Button2': 'ZoomViewMode',
            '<Control><Shift>Button2': 'FrameEditMode',
            '<Alt><Shift>Button2': 'FrameEditMode',
            'Button3': 'ShowPopupMenu',
        },
    }
    default_config.update(COMPAT_CONFIG)
    if sys.platform == 'win32':
        # The Linux wacom driver inverts the button numbers of the
        # pen flip button, because middle-click is the more useful
        # action on Linux. However one of the two buttons is often
        # accidentally hit with the thumb while painting. We want
        # to assign panning to this button by default.
        linux_mapping = default_config["input.button_mapping"]
        default_config["input.button_mapping"] = {}
        for bp, actname in linux_mapping.items():
            bp = bp.replace("Button2", "ButtonTMP")
            bp = bp.replace("Button3", "Button2")
            bp = bp.replace("ButtonTMP", "Button3")
            default_config["input.button_mapping"][bp] = actname
    return default_config
