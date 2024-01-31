# This file is part of MyPaint.
# Copyright (C) 2013-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Widgets and popup dialogs for making quick choices"""

## Imports

from __future__ import division, print_function
import abc

from lib.gibindings import Gtk

from .pixbuflist import PixbufList
from . import brushmanager
from . import brushselectionwindow
from . import widgets
from . import spinbox
from . import windowing
from lib.observable import event
import gui.colortools
from lib.pycompat import add_metaclass


## Module consts

_DEFAULT_PREFS_ID = u"default"


## Interfaces

@add_metaclass(abc.ABCMeta)
class Advanceable:
    """Interface for choosers which can be advanced by pressing keys.

    Advancing happens if the chooser is already visible and its key is
    pressed again.  This can happen repeatedly.  The actual action
    performed is up to the implementation: advancing some some choosers
    may move them forward through pages of alternatives, while other
    choosers may actually change a brush setting as they advance.

    """

    @abc.abstractmethod
    def advance(self):
        """Advances the chooser to the next page or choice.

        Choosers should remain open when their advance() method is
        invoked. The actual action performed is up to the concrete
        implementation: see the class docs.

        """


## Class defs

class QuickBrushChooser (Gtk.VBox):
    """A quick chooser widget for brushes"""

    ## Class constants

    _PREFS_KEY_TEMPLATE = u"brush_chooser.%s.selected_group"
    ICON_SIZE = 48

    ## Method defs

    def __init__(self, app, prefs_id=_DEFAULT_PREFS_ID):
        """Initialize"""
        Gtk.VBox.__init__(self)
        self.app = app
        self.bm = app.brushmanager

        self._prefs_key = self._PREFS_KEY_TEMPLATE % (prefs_id,)
        active_group_name = app.preferences.get(self._prefs_key, None)

        model = self._make_groups_sb_model()
        self.groups_sb = spinbox.ItemSpinBox(model, self._groups_sb_changed_cb,
                                             active_group_name)
        active_group_name = self.groups_sb.get_value()

        brushes = self.bm.get_group_brushes(active_group_name)

        self.brushlist = PixbufList(
            brushes, self.ICON_SIZE, self.ICON_SIZE,
            namefunc=brushselectionwindow.managedbrush_namefunc,
            pixbuffunc=brushselectionwindow.managedbrush_pixbuffunc,
            idfunc=brushselectionwindow.managedbrush_idfunc
        )
        self.brushlist.dragging_allowed = False
        self.bm.groups_changed += self._groups_changed_cb
        self.bm.brushes_changed += self._brushes_changed_cb
        self.brushlist.item_selected += self._item_selected_cb

        scrolledwin = Gtk.ScrolledWindow()
        scrolledwin.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.ALWAYS)
        scrolledwin.add(self.brushlist)
        w = int(self.ICON_SIZE * 4.5)
        h = int(self.ICON_SIZE * 5.0)
        scrolledwin.set_min_content_width(w)
        scrolledwin.set_min_content_height(h)
        scrolledwin.get_child().set_size_request(w, h)

        self.pack_start(self.groups_sb, False, False, 0)
        self.pack_start(scrolledwin, True, True, 0)
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
        group_names = sorted(self.bm.groups.keys())
        model = []
        for name in group_names:
            label_text = brushmanager.translate_group_name(name)
            model.append((name, label_text))
        return model

    def _groups_changed_cb(self, bm):
        """Internal: update the spinbox model at the top of the widget"""
        model = self._make_groups_sb_model()
        self.groups_sb.set_model(model)
        # In case the group has been deleted and recreated, we do this:
        group_name = self.groups_sb.get_value()
        group_brushes = self.bm.groups.get(group_name, [])
        self.brushlist.itemlist = group_brushes
        self.brushlist.update()
        # See https://github.com/mypaint/mypaint/issues/654

    def _brushes_changed_cb(self, bm, brushes):
        """Internal: update the PixbufList if its group was changed."""
        # CARE: this might be called in response to the group being deleted.
        # Don't recreate it by accident.
        group_name = self.groups_sb.get_value()
        group_brushes = self.bm.groups.get(group_name)
        if brushes is group_brushes:
            self.brushlist.update()

    def _groups_sb_changed_cb(self, group_name):
        """Internal: update the list of brush icons when the group changes"""
        self.app.preferences[self._prefs_key] = group_name
        group_brushes = self.bm.groups.get(group_name, [])
        self.brushlist.itemlist = group_brushes
        self.brushlist.update()

    def advance(self):
        """Advances to the next page of brushes."""
        self.groups_sb.next()


class BrushChooserPopup (windowing.ChooserPopup):
    """Speedy brush chooser popup"""

    def __init__(self, app, prefs_id=_DEFAULT_PREFS_ID):
        """Initialize.

        :param gui.application.Application app: main app instance
        :param unicode prefs_id: prefs identifier for the chooser

        The prefs identifier forms part of preferences key which store
        layout and which page of the chooser is selected. It should
        follow the same syntax rules as Python simple identifiers.

        """
        windowing.ChooserPopup.__init__(
            self,
            app = app,
            actions = [
                'ColorChooserPopup',
                'ColorChooserPopupFastSubset',
                'BrushChooserPopup',
            ],
            config_name = "brush_chooser.%s" % (prefs_id,),
        )
        self._chosen_brush = None
        self._chooser = QuickBrushChooser(app, prefs_id=prefs_id)
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

    def advance(self):
        """Advances to the next page of brushes."""
        self._chooser.advance()


class QuickColorChooser (Gtk.VBox):
    """A quick chooser widget for colors"""

    ## Class constants
    _PREFS_KEY_TEMPLATE = u"color_chooser.%s.selected_adjuster"
    _ALL_ADJUSTER_CLASSES = [
        gui.colortools.HCYWheelTool,
        gui.colortools.HSVWheelTool,
        gui.colortools.PaletteTool,
        gui.colortools.HSVCubeTool,
        gui.colortools.HSVSquareTool,
        gui.colortools.ComponentSlidersTool,
        gui.colortools.RingsColorChangerTool,
        gui.colortools.WashColorChangerTool,
        gui.colortools.CrossedBowlColorChangerTool,
    ]
    _SINGLE_CLICK_ADJUSTER_CLASSES = [
        gui.colortools.PaletteTool,
        gui.colortools.WashColorChangerTool,
        gui.colortools.CrossedBowlColorChangerTool,
    ]

    def __init__(self, app, prefs_id=_DEFAULT_PREFS_ID, single_click=False):
        Gtk.VBox.__init__(self)
        self._app = app
        self._spinbox_model = []
        self._adjs = {}
        self._pages = []
        mgr = app.brush_color_manager
        if single_click:
            adjuster_classes = self._SINGLE_CLICK_ADJUSTER_CLASSES
        else:
            adjuster_classes = self._ALL_ADJUSTER_CLASSES
        for page_class in adjuster_classes:
            name = page_class.__name__
            page = page_class()
            self._pages.append(page)
            self._spinbox_model.append((name, page.tool_widget_title))
            self._adjs[name] = page
            page.set_color_manager(mgr)
            if page_class in self._SINGLE_CLICK_ADJUSTER_CLASSES:
                page.connect_after(
                    "button-release-event",
                    self._ccwidget_btn_release_cb,
                )
        self._prefs_key = self._PREFS_KEY_TEMPLATE % (prefs_id,)
        active_page = app.preferences.get(self._prefs_key, None)
        sb = spinbox.ItemSpinBox(self._spinbox_model, self._spinbox_changed_cb,
                                 active_page)
        active_page = sb.get_value()
        self._spinbox = sb
        self._active_adj = self._adjs[active_page]
        self.pack_start(sb, False, False, 0)
        self.pack_start(self._active_adj, True, True, 0)
        self.set_spacing(widgets.SPACING_TIGHT)

    def _spinbox_changed_cb(self, page_name):
        self._app.preferences[self._prefs_key] = page_name
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

    def advance(self):
        """Advances to the next color selector."""
        self._spinbox.next()


class ColorChooserPopup (windowing.ChooserPopup):
    """Speedy color chooser dialog"""

    def __init__(self, app, prefs_id=_DEFAULT_PREFS_ID, single_click=False):
        """Initialize.

        :param gui.application.Application app: main app instance
        :param unicode prefs_id: prefs identifier for the chooser
        :param bool single_click: limit to just the single-click adjusters

        The prefs identifier forms part of preferences key which store
        layout and which page of the chooser is selected. It should
        follow the same syntax rules as Python simple identifiers.

        """
        windowing.ChooserPopup.__init__(
            self,
            app = app,
            actions = [
                'ColorChooserPopup',
                'ColorChooserPopupFastSubset',
                'BrushChooserPopup',
            ],
            config_name = u"color_chooser.%s" % (prefs_id,),
        )
        self._chooser = QuickColorChooser(
            app,
            prefs_id=prefs_id,
            single_click=single_click,
        )
        self._chooser.choice_completed += self._choice_completed_cb
        self.add(self._chooser)

    def _choice_completed_cb(self, chooser):
        """Internal: close when a choice is (fully) made

        Close the dialog on button release only to avoid accidental dabs
        with the stylus.
        """
        self.hide()

    def advance(self):
        """Advances to the next color selector."""
        self._chooser.advance()


## Classes: interface registration

Advanceable.register(QuickBrushChooser)
Advanceable.register(QuickColorChooser)
Advanceable.register(BrushChooserPopup)
Advanceable.register(ColorChooserPopup)
