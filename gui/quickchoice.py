# This file is part of MyPaint.
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Widgets and popup dialogs for making quick choices"""

## Imports

import gi
from gi.repository import Gtk
from gi.repository import Gdk

from gettext import gettext as _

from pixbuflist import PixbufList
import brushmanager
import widgets
import spinbox
import windowing
from lib.observable import event


## Class defs

class QuickBrushChooser (Gtk.VBox):
    """A quick chooser widget for brushes"""

    ## Class constants
    PREFS_KEY = 'widgets.brush_chooser.selected_group'
    ICON_SIZE = 48

    ## Method defs
    def __init__(self, app):
        """Initialize"""
        Gtk.VBox.__init__(self)
        self.app = app
        self.bm = app.brushmanager

        active_group_name = app.preferences.get(self.PREFS_KEY, None)

        model = self._make_groups_sb_model()
        self.groups_sb = spinbox.ItemSpinBox(model, self._groups_sb_changed_cb,
                                             active_group_name)
        active_group_name = self.groups_sb.get_value()

        brushes = self.bm.groups[active_group_name][:]

        self.brushlist = PixbufList(brushes, self.ICON_SIZE, self.ICON_SIZE,
                                    namefunc=lambda x: x.name,
                                    pixbuffunc=lambda x: x.preview)
        self.brushlist.dragging_allowed = False
        self.bm.groups_changed += self._update_groups_sb
        self.brushlist.item_selected += self._item_selected_cb

        scrolledwin = Gtk.ScrolledWindow()
        scrolledwin.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.ALWAYS)
        scrolledwin.add_with_viewport(self.brushlist)
        w = int(self.ICON_SIZE * 4.5)
        h = int(self.ICON_SIZE * 5.0)
        scrolledwin.set_min_content_width(w)
        scrolledwin.set_min_content_height(h)
        scrolledwin.get_child().set_size_request(w, h)

        self.pack_start(self.groups_sb, False, False)
        self.pack_start(scrolledwin, True, True)
        self.set_spacing(widgets.SPACING_TIGHT)

    def _item_selected_cb(self, pixbuf_list, brush):
        """Internal: call brush_selected event when an item is chosen"""
        self.brush_selected(brush)

    @event
    def brush_selected(self, brush):
        """Event: a brush was selected

        :param brush: The newly chosen brush
        """

    def _make_groups_sb_model(self):
        """Internal: create the model for the group choice spinbox"""
        group_names = self.bm.groups.keys()
        group_names.sort()
        model = []
        for name in group_names:
            label_text = brushmanager.translate_group_name(name)
            model.append((name, label_text))
        return model

    def _update_groups_sb(self, bm):
        """Internal: update the spinbox model at the top of the widget"""
        model = self._make_groups_sb_model()
        self.groups_sb.set_model(model)

    def _groups_sb_changed_cb(self, group_name):
        """Internal: update the list of brush icons when the group changes"""
        self.app.preferences[self.PREFS_KEY] = group_name
        self.brushlist.itemlist[:] = self.bm.groups[group_name][:]
        self.brushlist.update()


class BrushChooserDialog (windowing.ChooserDialog):
    """Speedy brush chooser dialog"""

    def __init__(self, app):
        """Initialize"""
        windowing.ChooserDialog.__init__(self,
          app=app, title=_("Change Brush"),
          actions=['BrushChooserPopup'],
          config_name="brushchooser")
        self._response_brush = None
        self._chooser = QuickBrushChooser(app)
        self._chooser.brush_selected += self._brush_selected_cb

        bl = self._chooser.brushlist
        bl.connect("button-release-event", self._brushlist_button_release_cb)

        vbox = self.get_content_area()
        vbox.pack_start(self._chooser, True, True)

        self.connect("response", self._response_cb)

    def _brush_selected_cb(self, chooser, brush):
        """Internal: update the response brush when an icon is clicked"""
        self._response_brush = brush

    def _brushlist_button_release_cb(self, *junk):
        """Internal: send an accept response on a button release

        We only send the response (and close the dialog) on button release to
        avoid accidental dabs with the stylus.
        """
        if self._response_brush is not None:
            self.response(Gtk.ResponseType.ACCEPT)

    def _response_cb(self, dialog, response_id):
        """Internal: update the brush on an accept response"""
        if response_id == Gtk.ResponseType.ACCEPT:
            bm = self.app.brushmanager
            bm.select_brush(dialog._response_brush)

