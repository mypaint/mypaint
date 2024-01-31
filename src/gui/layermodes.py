# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layer mode menu"""


## Imports

from __future__ import division, print_function

from lib.gibindings import Gtk

from lib.modes import STACK_MODES
from lib.modes import STANDARD_MODES
from lib.modes import MODE_STRINGS


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
        spec_separator = (None,)
        modes_menu_spec = (STACK_MODES + spec_separator + STANDARD_MODES)
        for mode in modes_menu_spec:
            if mode is None:
                menu.append(Gtk.SeparatorMenuItem())
                continue
            label, tooltip = MODE_STRINGS.get(mode)
            if prev_item is None:
                item = Gtk.RadioMenuItem()
            else:
                item = Gtk.RadioMenuItem(group=prev_item)
            item.set_label(label)
            item.set_tooltip_text(tooltip)
            item.connect("activate", self._item_activated_cb, mode)
            menu.append(item)
            self._menu_items.append((mode, item))
            prev_item = item
        self._submenu = menu
        self.set_submenu(self._submenu)
        self._submenu.show_all()
        from gui.application import get_app
        app = get_app()
        self._model = app.doc.model
        rootstack = self._model.layer_stack
        rootstack.layer_properties_changed += self._update_actions
        rootstack.current_path_updated += self._update_actions
        self._updating = False
        self._update_actions()

    def _item_activated_cb(self, item, mode):
        """Callback: Update the model when the user selects a menu item"""
        if self._updating:
            return
        self._model.set_current_layer_mode(mode)

    def _update_actions(self, *_ignored):
        """Updates menu actions to reflect the current layer's mode"""
        if self._updating:
            return
        self._updating = True
        rootstack = self._model.layer_stack
        current = rootstack.current
        for mode, item in self._menu_items:
            active = (mode == current.mode)
            if bool(item.get_active()) != active:
                item.set_active(active)
            item.set_sensitive(mode in current.PERMITTED_MODES)
        self._updating = False
