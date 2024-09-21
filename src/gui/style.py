# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Graphical style constants for on-canvas editable objects

This covers things like the frame outlines, or the axis of symmetry.
Using a consistent set of colors, widths, and sizes helps provide
a unified graphical experience.

See also: gui.drawutils

"""

from __future__ import division, print_function

from lib.color import HCYColor, RGBColor


## Alpha checks (chequerboard pattern)

ALPHA_CHECK_SIZE = 16
ALPHA_CHECK_COLOR_1 = (0.45, 0.45, 0.45)
ALPHA_CHECK_COLOR_2 = (0.50, 0.50, 0.50)


## Floating action buttons (rendered on the canvas)

FLOATING_BUTTON_ICON_SIZE = 16
FLOATING_BUTTON_RADIUS = 16


## Draggable line and handle sizes

DRAGGABLE_POINT_HANDLE_SIZE = 4
DRAGGABLE_EDGE_WIDTH = 2


## Paint-chip style

PAINT_CHIP_HIGHLIGHT_HCY_Y_MULT = 1.1
PAINT_CHIP_HIGHLIGHT_HCY_C_MULT = 1.1
PAINT_CHIP_SHADOW_HCY_Y_MULT = 0.666
PAINT_CHIP_SHADOW_HCY_C_MULT = 0.5


## Drop shadow layout and weight

DROP_SHADOW_ALPHA = 0.5
DROP_SHADOW_BLUR = 2.0
DROP_SHADOW_X_OFFSET = 0.0
DROP_SHADOW_Y_OFFSET = 0.5
# These are only used for otherwise flat editable or draggable objects.


## Colors for additional on-canvas information

# Transient on-canvas information, intended to be read quickly.
# Used for fading textual info or vanishing positional markers.
# Need to be high-contrast, and clear. Black and white is good.

TRANSIENT_INFO_BG_RGBA = (0, 0, 0, 0.666)  #: Transient text bg / outline
TRANSIENT_INFO_RGBA = (1, 1, 1, 1)  #: Transient text / marker


# Editable on-screen items.
# Used for editable handles on things like the document frame,
# when it's being edited.
# It's a good idea to use this and a user-tuneable alpha if the item
# is to be shown on screen permanently, in modes other than the object's
# own edit mode.

EDITABLE_ITEM_COLOR = RGBColor.new_from_hex_str("#ECF0F1")


# Active/dragging state for editable items.

ACTIVE_ITEM_COLOR = RGBColor.new_from_hex_str("#F1C40F")


# Prelight color (for complex modes, when there needs to be a distinction)

PRELIT_ITEM_COLOR = tuple(
        ACTIVE_ITEM_COLOR.interpolate(EDITABLE_ITEM_COLOR, 3)
    )[1]

