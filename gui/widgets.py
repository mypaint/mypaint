# This file is part of MyPaint.
# Copyright (C) 2011-2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layout constants and helper functions for common widgets."""

from __future__ import division, print_function
import functools

from lib.gibindings import Gtk
from lib.gibindings import Gdk


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
    """Create a button styled to be borderless.

    >>> borderless_button(icon_name="mypaint")  # doctest: +ELLIPSIS
    <Gtk.Button...>

    """
    button = Gtk.Button()
    if stock_id is not None:
        image = Gtk.Image()
        image.set_from_stock(stock_id, size)
        set_margins(image, 0)
        button.add(image)
    elif icon_name is not None:
        image = Gtk.Image()
        image.set_from_icon_name(icon_name, size)
        set_margins(image, 0)
        button.add(image)
    elif action is not None:
        button.set_related_action(action)
        if button.get_child() is not None:
            button.remove(button.get_child())
        img = action.create_icon(size)
        img.set_padding(4, 4)
        set_margins(img, 0)
        button.add(img)
    button.set_relief(Gtk.ReliefStyle.NONE)
    button.set_can_default(False)
    button.set_can_focus(False)
    set_margins(button, 0)
    if tooltip is not None:
        button.set_tooltip_text(tooltip)
    elif action is not None:
        button.set_tooltip_text(action.get_tooltip())
    cssprov = Gtk.CssProvider()
    cssprov.load_from_data(b"GtkButton { padding: 0px; margin: 0px; }")
    style = button.get_style_context()
    style.add_provider(cssprov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    button.set_has_tooltip(tooltip is not None)
    return button


def set_margins(widget, all_=0, tb=None, lr=None,
                t=None, b=None, l=None, r=None):  # noqa: E741
    """Set margins compatibly on a widget.

    >>> w = Gtk.Label(label="i have wide margins")
    >>> set_margins(w, 42)

    Works around Gtk's deprecation of gtk_widget_set_margin_{left,right}
    in version 3.12.

    """
    top = bot = left = right = 0
    if all_ is not None:
        top = bot = left = right = int(all_)
    if tb is not None:
        top = bot = int(tb)
    if lr is not None:
        left = right = int(lr)
    if t is not None:
        top = int(t)
    if b is not None:
        bot = int(b)
    if l is not None:
        left = int(l)
    if r is not None:
        right = int(r)
    try:
        widget.set_margin_start(left)
        widget.set_margin_end(right)
    except AttributeError:
        widget.set_margin_left(left)
        widget.set_margin_right(right)
    widget.set_margin_top(top)
    widget.set_margin_bottom(bot)


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
    from gui.application import get_app
    app = get_app()
    size = str(app.preferences.get("ui.toolbar_icon_size", "large"))
    if size.lower() == 'small':
        return ICON_SIZE_SMALL
    else:
        return ICON_SIZE_LARGE


def with_wait_cursor(func):
    """python decorator that adds a wait cursor around a function"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        wait_cursor = Gdk.Cursor.new(Gdk.CursorType.WATCH)
        toplevels = Gtk.Window.list_toplevels()
        toplevels = [t for t in toplevels if t.get_window() is not None]
        for toplevel in toplevels:
            toplevel_win = toplevel.get_window()
            if toplevel_win is not None:
                toplevel_win.set_cursor(wait_cursor)
            toplevel.set_sensitive(False)
        try:
            return func(self, *args, **kwargs)
            # gtk main loop may be called in here...
        finally:
            for toplevel in toplevels:
                toplevel.set_sensitive(True)
                # ... which is why we need this check
                toplevel_win = toplevel.get_window()
                if toplevel_win is not None:
                    toplevel_win.set_cursor(None)
    return wrapper
