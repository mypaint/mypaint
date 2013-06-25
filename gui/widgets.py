# This file is part of MyPaint.
# Copyright (C) 2011-2013 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layout constants and constructor functions for common widgets."""

import gi
from gi.repository import Gtk


# Spacing constants

SPACING_CRAMPED = 3    # for use in dockables only
SPACING_TIGHT = 2 * SPACING_CRAMPED
SPACING = 2 * SPACING_TIGHT
SPACING_LOOSE = 3*SPACING_TIGHT


def borderless_button(stock_id=None, size=Gtk.IconSize.BUTTON, tooltip=None):
    button = Gtk.Button()
    if stock_id is not None:
        image = Gtk.Image()
        image.set_from_stock(stock_id, size)
        button.add(image)
    button.set_relief(Gtk.ReliefStyle.NONE)
    button.set_can_default(False)
    button.set_can_focus(False)
    if tooltip is not None:
        button.set_tooltip_text(tooltip)
    cssprov = Gtk.CssProvider()
    cssprov.load_from_data("GtkButton { padding: 0px; }")
    style = button.get_style_context()
    style.add_provider(cssprov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    button.set_has_tooltip(tooltip is not None)
    return button


def section_frame(label_text):
    frame = Gtk.Frame()
    label_markup = "<b>%s</b>" % label_text
    label = Gtk.Label(label_markup)
    label.set_use_markup(True)
    frame.set_label_widget(label)
    frame.set_shadow_type(Gtk.ShadowType.NONE)
    return frame


def find_widgets(widget, predicate):
    """Finds widgets in a container's tree by predicate.
    """
    queue = [widget]
    found = []
    while len(queue) > 0:
        w = queue.pop(0)
        if predicate(w):
            found.append(w)
        if hasattr(w, "get_children"):
            for w2 in w.get_children():
                queue.append(w2)
    return found
