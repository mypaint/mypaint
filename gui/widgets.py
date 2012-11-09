# This file is part of MyPaint.
# Copyright (C) 2011 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layout constants and constructor functions for common widgets.
"""

import gtk
from gtk import gdk


# Spacing constants

SPACING_CRAMPED = 3    # for use in dockables only
SPACING_TIGHT = 2 * SPACING_CRAMPED
SPACING = 2 * SPACING_TIGHT
SPACING_LOOSE = 3*SPACING_TIGHT


# Useful style defaults for named widgets

gtk.rc_parse_string ("""
    style "borderless-style" {
        GtkWidget::focus-padding = 0
        GtkWidget::focus-line-width = 0
        xthickness = 0
        ythickness = 0
    }
    style "thinborder-style" {
        GtkWidget::focus-padding = 0
        GtkWidget::focus-line-width = 0
        xthickness = 1
        ythickness = 1
    }
    style "borderless-toolbar-style" {
        # Confuses some theme engines:
        #GtkToolbar::shadow-type = GTK_SHADOW_NONE
        # Following suffices to comply with Fitts's Law in fullscreen:
        GtkToolbar::internal-padding = 0
        xthickness = 0
        ythickness = 0
    }
    widget "*.borderless*" style "borderless-style"
    widget "*.thinborder*" style "thinborder-style"
    class "GtkToolbar" style "borderless-toolbar-style"
    """)

BORDERLESS_BUTTON_NAME = "borderless-button"


def borderless_button(stock_id=None, size=gtk.ICON_SIZE_BUTTON, tooltip=None):
    button = gtk.Button()
    if stock_id is not None:
        image = gtk.Image()
        image.set_from_stock(stock_id, size)
        button.add(image)
    button.set_name(BORDERLESS_BUTTON_NAME)
    button.set_relief(gtk.RELIEF_NONE)
    button.set_can_default(False)
    button.set_can_focus(False)
    if tooltip is not None:
        button.set_tooltip_text(tooltip)
    button.set_has_tooltip(tooltip is not None)
    return button


def section_frame(label_text):
    frame = gtk.Frame()
    label_markup = "<b>%s</b>" % label_text
    label = gtk.Label(label_markup)
    label.set_use_markup(True)
    frame.set_label_widget(label)
    frame.set_shadow_type(gtk.SHADOW_NONE)
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
