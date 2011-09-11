# This file is part of MyPaint.
# Copyright (C) 2011 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
from gtk import gdk
from gettext import gettext as _

# Symbolic names for our custom stock items.  The string values are also the
# names of the icons used, trying to mirror those in standard or widespread
# sets.  Mypaint ships with defaults for most of them using the Tango palette
# under the "hicolor" theme.

TOOL_BRUSH = "mypaint-tool-brush"
TOOL_COLOR_SELECTOR = "mypaint-tool-color-triangle"
TOOL_COLOR_SAMPLER = "mypaint-tool-hue-wheel"
TOOL_SCRATCHPAD = "mypaint-tool-scratchpad"
TOOL_LAYERS = "mypaint-tool-layers"
ROTATE_LEFT = "object-rotate-left"
ROTATE_RIGHT = "object-rotate-right"
MIRROR_HORIZONTAL = "object-flip-horizontal"
MIRROR_VERTICAL = "object-flip-vertical"
BRUSH_BLEND_MODES = "mypaint-brush-blend-modes"
BRUSH_BLEND_MODE_NORMAL = "mypaint-brush-blend-mode-normal"
BRUSH_BLEND_MODE_ERASER = "mypaint-brush-blend-mode-eraser"
BRUSH_BLEND_MODE_ALPHA_LOCK = "mypaint-brush-blend-mode-alpha-lock"
BRUSH_MODIFIERS_ACTIVE = "mypaint-brush-mods-active"
BRUSH_MODIFIERS_INACTIVE = "mypaint-brush-mods-inactive"


_stock_items = [
    # Tool windows. No trailing ellipses on these since none of them require
    # further input before they can do anything: see section 4.3.2.1 of the the
    # GNOME Human Interface Guidelines version 3.0.
    (TOOL_BRUSH, _("Brush List Editor"), gdk.SHIFT_MASK, ord("b"), None),
    (TOOL_COLOR_SELECTOR, _("Color Triangle"), 0, ord("g"), None),
    (TOOL_COLOR_SAMPLER, _("Color Sampler"), 0, ord("t"), None),
    (TOOL_SCRATCHPAD, _("Scratchpad"), gdk.SHIFT_MASK, ord("s"), None),
    (TOOL_LAYERS, _("Layers"), 0, ord("l"), None),
    (ROTATE_LEFT, _("Rotate Counterclockwise"),
        gdk.CONTROL_MASK, gdk.keyval_from_name("Left"), None),
    (ROTATE_RIGHT, _("Rotate Clockwise"),
        gdk.CONTROL_MASK, gdk.keyval_from_name("Right"), None),
    (MIRROR_HORIZONTAL, _("Mirror Horizontal"), 0, ord("i"), None),
    (MIRROR_VERTICAL, _("Mirror Vertical"), 0, ord("u"), None),
    (BRUSH_BLEND_MODES, _("Blend Mode"), 0, 0, None),
    (BRUSH_BLEND_MODE_NORMAL, _("Normal"), 0, ord('n'), None),
    (BRUSH_BLEND_MODE_ERASER, _("Eraser"), 0, ord('e'), None),
    (BRUSH_BLEND_MODE_ALPHA_LOCK, _("Lock Alpha Channel"),
        gdk.SHIFT_MASK, ord('l'), None),
    (BRUSH_MODIFIERS_ACTIVE, _("Brush Modifiers Active"), 0, 0, None),
    (BRUSH_MODIFIERS_INACTIVE, _("Brush Modifiers Inactive"), 0, 0, None),
]

def init_custom_stock_items():
    """Initialise the set of custom stock items defined here.

    Called at application start.
    """
    factory = gtk.IconFactory()
    factory.add_default()
    gtk.stock_add(_stock_items)
    for item_spec in _stock_items:
        stock_id = item_spec[0]
        source = gtk.IconSource()
        source.set_icon_name(stock_id)
        iconset = gtk.IconSet()
        iconset.add_source(source)
        factory.add(stock_id, iconset)
