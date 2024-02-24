# This file is part of MyPaint.
# Copyright (C) 2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.from lib.gibindings import Gdk

from lib.gibindings import Gdk
from lib.gibindings import Gtk

import logging
logger = logging.getLogger(__name__)


def is_ascii(s):
    return s and all(ord(c) < 128 for c in s)


def translate(hardware_keycode, state, group):
    # We may need to retry several times to deal with garbled text.

    keymap = Gdk.Keymap.get_default()

    # distinct
    it = list(set([group, 0, 1, 2]))

    ok_to_return = False
    keyval = None
    keyval_lower = None

    for g in it:
        res = keymap.translate_keyboard_state(
            hardware_keycode,
            Gdk.ModifierType(0),
            g
        )

        if not res:
            # PyGTK returns None when gdk_keymap_translate_keyboard_state()
            # returns false.  Not sure if this is a bug or a feature - the only
            # time I have seen this happen is when I put my laptop into sleep
            # mode.

            continue

        keyval = res[1]

        # consumed_modifiers = res[4]

        lbl = Gtk.accelerator_get_label(keyval, state)

        if is_ascii(lbl):
            ok_to_return = True
            break

    if not ok_to_return:
        logger.warning(
            'translate_keyboard_state() returned no valid response. '
            'Strange key pressed?')

        return None, None, None, None

    # We want to ignore irrelevant modifiers like ScrollLock.  The stored
    # key binding does not include modifiers that affected its keyval.
    mods = Gdk.ModifierType(
        state
        & Gtk.accelerator_get_default_mod_mask())

    keyval_lower = Gdk.keyval_to_lower(keyval)

    # If lowercasing affects the keysym, then we need to include
    # SHIFT in the modifiers. We re-upper case when we match against
    # the keyval, but display and save in caseless form.
    if keyval != keyval_lower:
        mods |= Gdk.ModifierType.SHIFT_MASK

    # So we get (<Shift>j, Shift+J) but just (plus, +). As I
    # understand it.

    accel_label = Gtk.accelerator_get_label(keyval_lower, mods)

    return keyval, keyval_lower, accel_label, mods
