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
import gui.colortools


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


class BrushChooserPopup (windowing.ChooserPopup):
    """Speedy brush chooser popup"""

    def __init__(self, app):
        """Initialize"""
        windowing.ChooserPopup.__init__(self,
           app=app, actions=['ColorChooserPopup', 'BrushChooserPopup'],
           config_name="brushchooser")
        self._chosen_brush = None
        self._chooser = QuickBrushChooser(app)
        self._chooser.brush_selected += self._brush_selected_cb

        bl = self._chooser.brushlist
        bl.connect("button-release-event", self._brushlist_button_release_cb)

        self.add(self._chooser)

    def _brush_selected_cb(self, chooser, brush):
        """Internal: update the response brush when an icon is clicked"""
        self._chosen_brush = brush

    def _brushlist_button_release_cb(self, *junk):
        """Internal: send an accept response on a button release

        We only send the response (and close the dialog) on button release to
        avoid accidental dabs with the stylus.
        """
        if self._chosen_brush is not None:
            bm = self.app.brushmanager
            bm.select_brush(self._chosen_brush)
            self.hide()
            self._chosen_brush = None


class QuickColorChooser (Gtk.VBox):
    """A quick chooser widget for colors"""

    ## Class constants
    _PREFS_KEY = 'widgets.color_chooser.selected_adjuster'
    _ADJUSTER_CLASSES = [
        gui.colortools.PaletteTool,
        gui.colortools.HCYWheelTool,
        gui.colortools.HSVWheelTool,
        gui.colortools.HSVTriangleTool,
        gui.colortools.HSVCubeTool,
        gui.colortools.ComponentSlidersTool,
        gui.colortools.RingsColorChangerTool,
        gui.colortools.WashColorChangerTool,
        gui.colortools.CrossedBowlColorChangerTool,
    ]
    _CHOICE_COMPLETABLE_CLASSES = set([
        gui.colortools.PaletteTool,
        gui.colortools.WashColorChangerTool,
        gui.colortools.RingsColorChangerTool,
        gui.colortools.CrossedBowlColorChangerTool,
    ])

    def __init__(self, app):
        Gtk.VBox.__init__(self)
        self._app = app
        self._spinbox_model = []
        self._adjs = {}
        self._pages = []
        mgr = app.brush_color_manager
        for page_class in self._ADJUSTER_CLASSES:
            name = page_class.__name__
            page = page_class()
            self._pages.append(page)
            self._spinbox_model.append((name, page.tool_widget_title))
            self._adjs[name] = page
            page.set_color_manager(mgr)
            if page_class in self._CHOICE_COMPLETABLE_CLASSES:
                page.connect_after(
                    "button-release-event",
                    self._ccwidget_btn_release_cb,
                )
        active_page = app.preferences.get(self._PREFS_KEY, None)
        sb = spinbox.ItemSpinBox(self._spinbox_model, self._spinbox_changed_cb,
                                 active_page)
        active_page = sb.get_value()
        self._spinbox = sb
        self._active_adj = self._adjs[active_page]
        self.pack_start(sb, False, False, 0)
        self.pack_start(self._active_adj, True, True, 0)
        self.set_spacing(widgets.SPACING_TIGHT)

    def _spinbox_changed_cb(self, page_name):
        self._app.preferences[self._PREFS_KEY] = page_name
        self.remove(self._active_adj)
        new_adj = self._adjs[page_name]
        self._active_adj = new_adj
        self.pack_start(self._active_adj, True, True, 0)
        self._active_adj.show_all()

    def _ccwidget_btn_release_cb(self, ccwidget, event):
        """Internal: fire "choice_completed" after clicking certain widgets"""
        self.choice_completed()
        return False

    @event
    def choice_completed(self):
        """Event: a complete selection was made

        This is emitted by button-release events on certain kinds of colour
        chooser page. Not every page in the chooser emits this event, because
        colour is a three-dimensional quantity: clicking on a two-dimensional
        popup can't make a complete choice of colour with most pages.

        The palette page does emit this event, and it's the default.
        """


class ColorChooserPopup (windowing.ChooserPopup):
    """Speedy color chooser dialog"""

    def __init__(self, app):
        """Initialize"""
        windowing.ChooserPopup.__init__(self, app=app,
          actions=['ColorChooserPopup', 'BrushChooserPopup'],
          config_name="colorchooser")
        self._chooser = QuickColorChooser(app)
        self._chooser.choice_completed += self._choice_completed_cb
        self.add(self._chooser)

    def _choice_completed_cb(self, chooser):
        """Internal: close when a choice is (fully) made

        Close the dialog on button release only to avoid accidental dabs
        with the stylus.
        """
        self.hide()

