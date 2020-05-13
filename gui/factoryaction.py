# This file is part of MyPaint.
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Factory for creating custom toolbar and manu items via UIManager"""

from __future__ import division, print_function

from warnings import warn

import gi
from lib.gibindings import Gtk
from lib.gibindings import GObject


class FactoryAction (Gtk.Action):
    """Generic factory action for UI components.

    Define an instance of the factory once for each family of custom proxy
    classes (GtkToolItem, GtkMenuItem etc.) that you need.  Each instance must
    be named after the custom proxy classes the factory is to produce, so the
    definition is best done in a central snippet of GtkBuilder XML.

    For example, if the factory is named ``FooBar``, then its proxy ToolItems
    are expected to have ``__gtype_name__``s of``"MyPaintFooBarToolItem``.

    Creation of proxies can then be done cleanly in the GtkUIManager XML
    definitions by referring to the factory's name as many times as needed.

    """

    #: Predictable name for GtkBuilder XML.
    __gtype_name__ = "MyPaintFactoryAction"

    #: The pattern to use when instantiating a tool item
    TOOL_ITEM_NAME_PATTERN = "MyPaint%sToolItem"

    #: The pattern to use when instantiating a menu item
    MENU_ITEM_NAME_PATTERN = "MyPaint%sMenuItem"

    def __init__(self, *a):
        # GtkAction's own constructor requires params which are all set up by
        # Builder. It warns noisily, so bypass it and invoke its parent
        # class's.
        super(Gtk.Action, self).__init__()

    def do_create_tool_item(self):
        """Returns a new ToolItem

        Invoked by UIManager when it needs a GtkToolItem proxy for a toolbar.

        This method instantiates and returns a new widget from a class named
        after the factory action's own name.  Class lookup is done via GObject:
        see `TOOL_ITEM_NAME_PATTERN` for the ``__gtype_name__`` this method
        will expect.

        """
        gtype_name = self.TOOL_ITEM_NAME_PATTERN % (self.get_name(),)
        tool_item = self._construct(gtype_name)
        tool_item.connect("parent-set", self._tool_item_parent_set)
        return tool_item

    def do_create_menu_item(self):
        """Returns a new MenuItem

        Invoked by UIManager when it needs a MenuItem proxy for a menu.

        This method instantiates and returns a new widget from a class named
        after the factory action's own name.  Class lookup is done via GObject:
        see `TOOL_ITEM_NAME_PATTERN` for the ``__gtype_name__`` this method
        will expect.

        """
        gtype_name = self.MENU_ITEM_NAME_PATTERN % (self.get_name(),)
        menu_item = self._construct(gtype_name)
        #menu_item.connect("parent-set", self._tool_item_parent_set)
        return menu_item

    def _construct(self, gtype_name):
        try:
            gtype = GObject.type_from_name(gtype_name)
        except RuntimeError:
            warn("Cannot construct a new %s: not loaded?" % (gtype_name,),
                 RuntimeWarning)
            return None
        if not gtype.is_a(Gtk.Widget):
            warn("%s is not a Gtk.Widget subclass" % (gtype_name,),
                 RuntimeWarning)
            return None
        widget = gtype.pytype()
        return widget

    def _tool_item_parent_set(self, widget, old_parent):
        parent = widget.get_parent()
        if parent and parent.get_visible():
            widget.show_all()
