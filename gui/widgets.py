# This file is part of MyPaint.
# Copyright (C) 2011-2013 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layout constants and constructor functions for common widgets."""

from __future__ import print_function

import gi
from gi.repository import Gtk


# Exact icon sizes

ICON_SIZE_LARGE = Gtk.IconSize.LARGE_TOOLBAR   # 24x24, the docs promise
ICON_SIZE_SMALL = Gtk.IconSize.SMALL_TOOLBAR   # 16x16


# Spacing constants

SPACING_CRAMPED = 3    # for use in dockables only
SPACING_TIGHT = 2 * SPACING_CRAMPED
SPACING = 2 * SPACING_TIGHT
SPACING_LOOSE = 3*SPACING_TIGHT


def borderless_button(stock_id=None, icon_name=None, size=ICON_SIZE_SMALL,
                      tooltip=None, action=None):
    button = Gtk.Button()
    if stock_id is not None:
        image = Gtk.Image()
        image.set_from_stock(stock_id, size)
        button.add(image)
    elif icon_name is not None:
        image = Gtk.Image()
        image.set_from_icon_name(icon_name, size)
        button.add(image)
    elif action is not None:
        button.set_related_action(action)
        if button.get_child() is not None:
            button.remove(button.get_child())
        img = action.create_icon(size)
        img.set_padding(4, 4)
        button.add(img)
    button.set_relief(Gtk.ReliefStyle.NONE)
    button.set_can_default(False)
    button.set_can_focus(False)
    if tooltip is not None:
        button.set_tooltip_text(tooltip)
    elif action is not None:
        button.set_tooltip_text(action.get_tooltip())
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


def inline_toolbar(app, tool_defs):
    """Builds a styled inline toolbar"""
    bar = Gtk.Toolbar(
        show_arrow = False,
        icon_size = ICON_SIZE_SMALL,
    )
    bar.set_style(Gtk.ToolbarStyle.ICONS)
    styles = bar.get_style_context()
    styles.add_class(Gtk.STYLE_CLASS_INLINE_TOOLBAR)
    for action_name, override_icon in tool_defs:
        action = app.find_action(action_name)
        toolitem = Gtk.ToolButton()
        toolitem.set_related_action(action)
        if override_icon:
            toolitem.set_icon_name(override_icon)
        bar.insert(toolitem, -1)
        bar.child_set_property(toolitem, "expand", True)
        bar.child_set_property(toolitem, "homogeneous", True)
    bar.set_vexpand(False)
    bar.set_hexpand(True)
    return bar


class MenuButtonToolItem (Gtk.ToolItem):
    """ToolItem which contains a Gtk.MenuButton"""

    def __init__(self):
        Gtk.ToolItem.__init__(self)
        menubtn = Gtk.MenuButton()
        self.add(menubtn)
        self.connect("realize", self._realize_cb)
        menubtn.set_always_show_image(True)
        menubtn.set_relief(Gtk.ReliefStyle.NONE)
        self._menubutton = menubtn
        self.menu = None  #: Populate with the menu to show before realize

    def _realize_cb(self, widget):
        action = self.get_related_action()
        icon_name = action.get_icon_name()
        image = Gtk.Image.new_from_icon_name(
            icon_name,
            get_toolbar_icon_size(),
        )
        self._menubutton.set_image(image)
        if self.menu:
            self._menubutton.set_popup(self.menu)


def get_toolbar_icon_size():
    from application import get_app
    app = get_app()
    size = str(app.preferences.get("ui.toolbar_icon_size", "large"))
    if size.lower() == 'small':
        return ICON_SIZE_SMALL
    else:
        return ICON_SIZE_LARGE

