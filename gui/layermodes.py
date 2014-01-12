# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layer mode menu"""


## Imports

from gi.repository import Gtk

from lib.layer import COMPOSITE_OPS


## Class definitions

class LayerModeMenuItem (Gtk.ImageMenuItem):
    """Brush list menu item with a dynamic BrushGroupsMenu as its submenu

    This is instantiated by the app's UIManager using a FactoryAction which
    must be named "LayerMode" (see factoryaction.py).
    """

    __gtype_name__ = "MyPaintLayerModeMenuItem"

    def __init__(self):
        """Initialize, called by the LayerMode FactoryAction making a menu"""
        Gtk.ImageMenuItem.__init__(self)
        menu = Gtk.Menu()
        self._menu_items = []
        prev_item = None
        for cname, label, tooltip in COMPOSITE_OPS:
            if prev_item is None:
                item = Gtk.RadioMenuItem()
            else:
                item = Gtk.RadioMenuItem(group=prev_item)
            item.set_label(label)
            item.set_tooltip_text(tooltip)
            item.connect("activate", self._item_activated_cb, cname)
            menu.append(item)
            self._menu_items.append((cname, item))
            prev_item = item
        self._submenu = menu
        self.set_submenu(self._submenu)
        self._submenu.show_all()
        from application import get_app
        app = get_app()
        app.doc.model.doc_observers.append(self._model_updated_cb)
        self._model = app.doc.model
        self._updating = False
        self._model_updated_cb(self._model)

    def _item_activated_cb(self, item, cname):
        """Callback: Update the model when the user selects a menu item"""
        if self._updating:
            return
        self._model.set_layer_compositeop(cname)

    def _model_updated_cb(self, model):
        """Callback: Update the menu when the model's mode changes"""
        if self._updating:
            return
        self._updating = True
        current_mode = model.layer_stack.current.compositeop
        for cname, item in self._menu_items:
            active = bool(cname == current_mode)
            if bool(item.get_active()) != active:
                item.set_active(active)
        self._updating = False

