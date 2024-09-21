# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2014-2019 by the MyPaint Development Team
# Copyright (C) 2009 by Ilya Portnov <portnov@bk.ru>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layers panel"""


## Imports

from __future__ import division, print_function

from gettext import gettext as _
import os.path
from logging import getLogger

from lib.gibindings import Gtk
from lib.gibindings import GObject

import lib.layer
import lib.xml
from . import widgets
from .widgets import inline_toolbar
from .toolstack import SizedVBoxToolWidget
from . import layers
from lib.modes import STACK_MODES
from lib.modes import STANDARD_MODES
from lib.modes import MODE_STRINGS
from lib.modes import PASS_THROUGH_MODE
import lib.modes
import gui.layervis

logger = getLogger(__name__)

## Module constants

#: UI XML for the current layer's class (framework: ``layerswindow.xml``)
LAYER_CLASS_UI = [
    (lib.layer.SurfaceBackedLayer, """
        <popup name='LayersWindowPopup'>
            <placeholder name="BasicLayerActions">
                <menuitem action='CopyLayer'/>
            </placeholder>
        </popup>
        """),
    (lib.layer.PaintingLayer, """
        <popup name='LayersWindowPopup'>
            <placeholder name="BasicLayerActions">
                <menuitem action='PasteLayer'/>
                <menuitem action='ClearLayer'/>
            </placeholder>
            <placeholder name='AdvancedLayerActions'>
                <menuitem action='TrimLayer'/>
                <separator/>
                <menuitem action='UniqLayerPixels'/>
                <menuitem action='UniqLayerTiles'/>
            </placeholder>
        </popup>
        """),
    (lib.layer.ExternallyEditable, """
        <popup name='LayersWindowPopup'>
            <placeholder name='BasicLayerActions'>
                <separator/>
                <menuitem action='BeginExternalLayerEdit'/>
                <menuitem action='CommitExternalLayerEdit'/>
                <separator/>
            </placeholder>
        </popup>
        """),
    (lib.layer.LayerStack, """
        <popup name='LayersWindowPopup'>
            <placeholder name='AdvancedLayerActions'>
                <menuitem action='RefactorLayerGroupPixels'/>
                <menuitem action='RefactorLayerGroupTiles'/>
            </placeholder>
        </popup>
        """),
]


## Class definitions


class LayersTool (SizedVBoxToolWidget):
    """Panel for arranging layers within a tree structure"""

    ## Class properties

    tool_widget_icon_name = "mypaint-layers-symbolic"
    tool_widget_title = _("Layers")
    tool_widget_description = _("Arrange layers and assign effects")

    LAYER_MODE_TOOLTIP_MARKUP_TEMPLATE = "<b>{name}</b>\n{description}"

    # TRANSLATORS: tooltip for the opacity slider (text)
    # TRANSLATORS: note that "%%" turns into "%"
    OPACITY_SCALE_TOOLTIP_TEXT_TEMPLATE = _("Layer opacity: %d%%")

    # TRANSLATORS: label for the opacity slider (text)
    # TRANSLATORS: note that "%%" turns into "%"
    # TRANSLATORS: most of the time this can just be copied, or left alone
    OPACITY_LABEL_TEXT_TEMPLATE = _(u"%d%%")

    __gtype_name__ = 'MyPaintLayersTool'

    STATUSBAR_CONTEXT = 'layerstool-dnd'

    # TRANSLATORS: status bar messages for drag, without/with modifiers
    STATUSBAR_DRAG_MSG = _(u"Move layer in stack…")
    STATUSBAR_DRAG_INTO_MSG = _("Move layer in stack (dropping into a "
                                "regular layer will create a new group)")

    ## Construction

    def __init__(self):
        GObject.GObject.__init__(self)
        from gui.application import get_app
        app = get_app()
        self.app = app
        self.set_spacing(widgets.SPACING_CRAMPED)
        self.set_border_width(widgets.SPACING_TIGHT)
        # GtkTreeView init
        docmodel = app.doc.model
        view = layers.RootStackTreeView(docmodel)
        self._treemodel = view.get_model()
        self._treeview = view
        # RootStackTreeView events
        view.current_layer_rename_requested += self._layer_properties_cb
        view.current_layer_changed += self._blink_current_layer_cb
        view.current_layer_menu_requested += self._popup_menu_cb
        # Drag and drop
        view.drag_began += self._view_drag_began_cb
        view.drag_ended += self._view_drag_ended_cb
        statusbar_cid = app.statusbar.get_context_id(self.STATUSBAR_CONTEXT)
        self._drag_statusbar_context_id = statusbar_cid
        # View scrolls
        view_scroll = Gtk.ScrolledWindow()
        view_scroll.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
        vscroll_pol = Gtk.PolicyType.ALWAYS
        hscroll_pol = Gtk.PolicyType.AUTOMATIC
        view_scroll.set_policy(hscroll_pol, vscroll_pol)
        view_scroll.add(view)
        view_scroll.set_size_request(-1, 200)
        view_scroll.set_hexpand(True)
        view_scroll.set_vexpand(True)
        # Context menu
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        ui_path = os.path.join(ui_dir, "layerswindow.xml")
        self.app.ui_manager.add_ui_from_file(ui_path)
        menu = self.app.ui_manager.get_widget("/LayersWindowPopup")
        menu.set_title(_("Layer"))
        self.connect("popup-menu", self._popup_menu_cb)
        menu.attach_to_widget(self, None)
        self._menu = menu
        self._layer_specific_ui_mergeids = []
        self._layer_specific_ui_class = None

        # Main layout grid
        grid = Gtk.Grid()
        grid.set_row_spacing(widgets.SPACING_TIGHT)
        grid.set_column_spacing(widgets.SPACING)
        row = -1

        # Visibility set management
        row += 1
        layer_view_ui = gui.layervis.LayerViewUI(docmodel)
        grid.attach(layer_view_ui.widget, 0, row, 6, 1)
        self._layer_view_ui = layer_view_ui

        # Mode dropdown
        row += 1
        # ComboBox w/ list model (mode_num, label, sensitive, scale)
        modes = list(STACK_MODES + STANDARD_MODES)
        modes.remove(lib.mypaintlib.CombineSpectralWGM)
        modes.insert(0, lib.mypaintlib.CombineSpectralWGM)
        combo = layers.new_blend_mode_combo(modes, MODE_STRINGS)
        self._layer_mode_combo = combo
        grid.attach(combo, 0, row, 5, 1)

        # Opacity widgets
        adj = Gtk.Adjustment(lower=0, upper=100,
                             step_increment=1, page_increment=10)
        sbut = Gtk.ScaleButton()
        sbut.set_adjustment(adj)
        sbut.remove(sbut.get_child())
        sbut.set_hexpand(False)
        sbut.set_vexpand(False)
        label_text_widest = self.OPACITY_LABEL_TEXT_TEMPLATE % (100,)
        label = Gtk.Label(label_text_widest)
        label.set_width_chars(len(label_text_widest))
        # prog = Gtk.ProgressBar()
        # prog.set_show_text(False)
        sbut.add(label)
        self._opacity_scale_button = sbut
        # self._opacity_progress = prog
        self._opacity_label = label
        self._opacity_adj = adj
        grid.attach(sbut, 5, row, 1, 1)

        # Layer list and controls
        row += 1
        layersbox = Gtk.VBox()
        style = layersbox.get_style_context()
        style.add_class(Gtk.STYLE_CLASS_LINKED)
        style = view_scroll.get_style_context()
        style.set_junction_sides(Gtk.JunctionSides.BOTTOM)
        list_tools = inline_toolbar(
            self.app,
            [
                ("NewLayerGroupAbove", "mypaint-layer-group-new-symbolic"),
                ("NewPaintingLayerAbove", "mypaint-add-symbolic"),
                ("RemoveLayer", "mypaint-remove-symbolic"),
                ("RaiseLayerInStack", "mypaint-up-symbolic"),
                ("LowerLayerInStack", "mypaint-down-symbolic"),
                ("DuplicateLayer", None),
                ("MergeLayerDown", None),
            ]
        )
        style = list_tools.get_style_context()
        style.set_junction_sides(Gtk.JunctionSides.TOP)
        layersbox.pack_start(view_scroll, True, True, 0)
        layersbox.pack_start(list_tools, False, False, 0)
        layersbox.set_hexpand(True)
        layersbox.set_vexpand(True)
        grid.attach(layersbox, 0, row, 6, 1)

        # Background layer controls
        row += 1
        show_bg_btn = Gtk.CheckButton()
        change_bg_act = self.app.find_action("BackgroundWindow")
        change_bg_btn = widgets.borderless_button(action=change_bg_act)
        show_bg_act = self.app.find_action("ShowBackgroundToggle")
        show_bg_btn.set_related_action(show_bg_act)
        grid.attach(show_bg_btn, 0, row, 5, 1)
        grid.attach(change_bg_btn, 5, row, 1, 1)

        # Pack
        self.pack_start(grid, False, True, 0)
        # Updates from the real layers tree (TODO: move to lib/layers.py)
        self._processing_model_updates = False
        self._opacity_adj.connect('value-changed',
                                  self._opacity_adj_changed_cb)
        self._layer_mode_combo.connect('changed',
                                       self._layer_mode_combo_changed_cb)
        rootstack = docmodel.layer_stack
        rootstack.layer_properties_changed += self._layer_propchange_cb
        rootstack.current_path_updated += self._current_path_updated_cb
        # Initial update
        self.connect("show", self._show_cb)

    def _show_cb(self, event):
        self._processing_model_updates = True
        self._update_all()
        self._processing_model_updates = False

    ## Updates from the model

    def _current_path_updated_cb(self, rootstack, layerpath):
        """Respond to the current layer changing in the doc-model"""
        self._processing_model_updates = True
        self._update_all()
        self._processing_model_updates = False

    def _layer_propchange_cb(self, rootstack, path, layer, changed):
        if self._processing_model_updates:
            logger.debug("Property change skipped: already processing "
                         "an update from the document model")
        if layer is not rootstack.current:
            return
        self._processing_model_updates = True
        if "mode" in changed:
            self._update_layer_mode_combo()
        if "opacity" in changed or "mode" in changed:
            self._update_opacity_widgets()
        self._processing_model_updates = False

    ## Model update processing

    def _update_all(self):
        assert self._processing_model_updates
        self._update_context_menu()
        self._update_layer_mode_combo()
        self._update_opacity_widgets()

    def _update_layer_mode_combo(self):
        """Updates the layer mode combo's value from the model"""
        assert self._processing_model_updates
        combo = self._layer_mode_combo
        rootstack = self.app.doc.model.layer_stack
        current = rootstack.current
        if current is rootstack or not current:
            combo.set_sensitive(False)
            return
        elif not combo.get_sensitive():
            combo.set_sensitive(True)
        active_iter = None
        current_mode = current.mode
        for row in combo.get_model():
            mode = row[0]
            if mode == current_mode:
                active_iter = row.iter
            row[2] = (mode in current.PERMITTED_MODES)
        combo.set_active_iter(active_iter)
        label, desc = MODE_STRINGS.get(current_mode)
        template = self.LAYER_MODE_TOOLTIP_MARKUP_TEMPLATE
        tooltip = template.format(
            name = lib.xml.escape(label),
            description = lib.xml.escape(desc),
        )
        combo.set_tooltip_markup(tooltip)

    def _update_opacity_widgets(self):
        """Updates the opacity widgets from the model"""
        assert self._processing_model_updates

        # The opacity scale is only sensitive
        # when the opacity can be adjusted.
        sbut = self._opacity_scale_button
        rootstack = self.app.doc.model.layer_stack
        layer = rootstack.current
        opacity_is_adjustable = not (
            layer is None
            or layer is rootstack
            or layer.mode == PASS_THROUGH_MODE
        )
        sbut.set_sensitive(opacity_is_adjustable)

        # Update labels, scales etc.
        # to show an effective opacity value.
        if opacity_is_adjustable:
            opacity = layer.opacity
        else:
            opacity = 1.0

        percentage = opacity * 100
        adj = self._opacity_adj
        adj.set_value(percentage)

        template = self.OPACITY_SCALE_TOOLTIP_TEXT_TEMPLATE
        tooltip = template % (percentage,)
        sbut.set_tooltip_text(tooltip)

        label = self._opacity_label
        template = self.OPACITY_LABEL_TEXT_TEMPLATE
        text = template % (percentage,)
        label.set_text(text)

    def _update_context_menu(self):
        assert self._processing_model_updates
        layer = self.app.doc.model.layer_stack.current
        layer_class = layer.__class__
        if layer_class is self._layer_specific_ui_class:
            return
        ui_manager = self.app.ui_manager
        for old_mergeid in self._layer_specific_ui_mergeids:
            ui_manager.remove_ui(old_mergeid)
        self._layer_specific_ui_mergeids = []
        new_ui_matches = []
        for lclass, lui in LAYER_CLASS_UI:
            if isinstance(layer, lclass):
                new_ui_matches.append(lui)
        for new_ui in new_ui_matches:
            new_mergeid = ui_manager.add_ui_from_string(new_ui)
            self._layer_specific_ui_mergeids.append(new_mergeid)
        self._layer_specific_ui_class = layer_class

    ## Updates from the user

    def _layer_properties_cb(self, view):
        action = self.app.find_action("LayerProperties")
        action.activate()

    def _blink_current_layer_cb(self, view):
        self.app.doc.blink_layer()

    def _view_drag_began_cb(self, view):
        self._treeview_in_drag = True
        statusbar = self.app.statusbar
        statusbar_cid = self._drag_statusbar_context_id
        statusbar.remove_all(statusbar_cid)
        statusbar.push(statusbar_cid, self.STATUSBAR_DRAG_MSG)

    def _view_drag_ended_cb(self, view):
        self._treeview_in_drag = False
        statusbar = self.app.statusbar
        statusbar_cid = self._drag_statusbar_context_id
        statusbar.remove_all(statusbar_cid)

    def _opacity_adj_changed_cb(self, *ignore):
        if self._processing_model_updates:
            return
        opacity = self._opacity_adj.get_value() / 100.0
        docmodel = self.app.doc.model
        docmodel.set_current_layer_opacity(opacity)
        self._treeview.scroll_to_current_layer()

    def _layer_mode_combo_changed_cb(self, *ignored):
        """Propagate the user's choice of layer mode to the model"""
        if self._processing_model_updates:
            return
        docmodel = self.app.doc.model
        combo = self._layer_mode_combo
        model = combo.get_model()
        mode = model.get_value(combo.get_active_iter(), 0)
        if docmodel.layer_stack.current.mode == mode:
            return
        label, desc = MODE_STRINGS.get(mode)
        docmodel.set_current_layer_mode(mode)

    ## Utility methods

    def _popup_context_menu(self, event=None):
        """Display the popup context menu"""
        if event is None:
            time = Gtk.get_current_event_time()
            button = 0
        else:
            time = event.time
            button = event.button
        self._menu.popup(None, None, None, None, button, time)

    def _popup_menu_cb(self, widget, event=None):
        """Handler for "popup-menu" GtkEvents, and the view's @event"""
        self._popup_context_menu(event=event)
        return True
