# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (C) 2009 by Ilya Portnov <portnov@bk.ru>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layers panel"""


## Imports

from gettext import gettext as _
import os.path

from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GObject
from gi.repository import Pango

import lib.layer
from lib.tiledsurface import COMBINE_MODE_STRINGS, NUM_COMBINE_MODES
from lib.helpers import escape
import widgets
from widgets import inline_toolbar
from workspace import SizedVBoxToolWidget
import layers


## Module constants

#: UI XML for the current layer's class (framework: ``layerswindow.xml``)
LAYER_CLASS_UI = {
    lib.layer.LayerStack: """
        <popup name='LayersWindowPopup'>
            <placeholder name='AdvancedLayerProperties'>
                <menuitem action='LayerStackIsolated'/>
            </placeholder>
        </popup>
        """,
    lib.layer.PaintingLayer: """
        <popup name='LayersWindowPopup'>
            <placeholder name="BasicLayerActions">
                <menuitem action='CopyLayer'/>
                <menuitem action='PasteLayer'/>
                <menuitem action='ClearLayer'/>
            </placeholder>
            <placeholder name='AdvancedLayerActions'>
                <menuitem action='NormalizeLayerMode'/>
                <menuitem action='TrimLayer'/>
            </placeholder>
            <placeholder name="AdvancedListActions">
                <menuitem action='MergeLayer'/>
            </placeholder>
        </popup>
        """
    }


## Helper functions

def make_layer_mode_model():
    model = Gtk.ListStore(int, str, str)
    for mode in range(NUM_COMBINE_MODES):
        label, desc = COMBINE_MODE_STRINGS.get(mode)
        model.append([mode, label, desc])
    return model



## Class definitions


class LayersTool (SizedVBoxToolWidget):
    """Panel for arranging layers within a tree structure"""

    ## Class properties

    tool_widget_icon_name = "mypaint-layers-symbolic"
    tool_widget_title = _("Layers")
    tool_widget_description = _("Arrange layers and assign effects")

    #TRANSLATORS: layer mode tooltips
    tooltip_format = _("<b>{mode_name}</b>\n{mode_description}")

    __gtype_name__ = 'MyPaintLayersTool'


    ## Construction

    def __init__(self):
        GObject.GObject.__init__(self)
        from application import get_app
        app = get_app()
        self.app = app
        self.set_spacing(widgets.SPACING_CRAMPED)
        self.set_border_width(widgets.SPACING_TIGHT)
        # GtkTreeView init
        docmodel = app.doc.model
        view = Gtk.TreeView()
        treemodel = layers.RootStackTreeModelWrapper(docmodel)
        view.set_model(treemodel)
        self._treemodel = treemodel
        view.set_reorderable(True)
        view.set_headers_visible(False)
        view.connect("button-press-event", self._view_button_press_cb)
        self._treeview = view
        # View behaviour and appearance
        sel = view.get_selection()
        sel.set_mode(Gtk.SelectionMode.SINGLE)
        view_scroll = Gtk.ScrolledWindow()
        view_scroll.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
        scroll_pol = Gtk.PolicyType.AUTOMATIC
        view_scroll.set_policy(scroll_pol, scroll_pol)
        view_scroll.add(view)
        view_scroll.set_size_request(-1, 100)
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
        self._layer_specific_ui_mergeid = None
        self._layer_specific_ui_class = None
        # Type column
        cell = Gtk.CellRendererPixbuf()
        col = Gtk.TreeViewColumn(_("Type"))
        col.pack_start(cell, expand=False)
        datafunc = layers.layer_type_pixbuf_datafunc
        col.set_cell_data_func(cell, datafunc)
        col.set_max_width(24)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        view.append_column(col)
        self._type_col = col
        # Name column
        cell = Gtk.CellRendererText()
        cell.set_property("ellipsize", Pango.EllipsizeMode.END)
        col = Gtk.TreeViewColumn(_("Name"))
        col.pack_start(cell, expand=True)
        datafunc = layers.layer_name_text_datafunc
        col.set_cell_data_func(cell, datafunc)
        col.set_expand(True)
        col.set_min_width(48)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        view.append_column(col)
        self._name_col = col
        # Visibility column
        cell = Gtk.CellRendererPixbuf()
        col = Gtk.TreeViewColumn(_("Visible"))
        col.pack_start(cell, expand=False)
        datafunc = layers.layer_visible_pixbuf_datafunc
        col.set_cell_data_func(cell, datafunc)
        col.set_max_width(24)
        view.append_column(col)
        self._visible_col = col
        # Locked column
        cell = Gtk.CellRendererPixbuf()
        col = Gtk.TreeViewColumn(_("Locked"))
        col.pack_start(cell, expand=False)
        datafunc = layers.layer_locked_pixbuf_datafunc
        col.set_cell_data_func(cell, datafunc)
        col.set_max_width(24)
        view.append_column(col)
        self._locked_col = col
        # View appearance
        view.set_show_expanders(True)
        view.set_enable_tree_lines(True)
        view.set_expander_column(self._name_col)
        # Callbacks
        root_stack = docmodel.layer_stack
        root_stack.current_path_updated += self._current_path_updated_cb
        # Main layout grid
        grid = Gtk.Grid()
        grid.set_row_spacing(widgets.SPACING_TIGHT)
        grid.set_column_spacing(widgets.SPACING)
        # Mode dropdown
        row = 0
        layer_mode_lbl = Gtk.Label(label=_('Mode:'))
        layer_mode_lbl.set_tooltip_text(
          _("Blending mode: how the current layer combines with the "
            "layers underneath it."))
        layer_mode_lbl.set_alignment(0, 0.5)
        layer_mode_lbl.set_hexpand(False)
        self._layer_mode_model = make_layer_mode_model()
        self._layer_mode_combo = Gtk.ComboBox()
        self._layer_mode_combo.set_model(self._layer_mode_model)
        self._layer_mode_combo.set_hexpand(True)
        cell = Gtk.CellRendererText()
        self._layer_mode_combo.pack_start(cell)
        self._layer_mode_combo.add_attribute(cell, "text", 1)
        grid.attach(layer_mode_lbl, 0, row, 1, 1)
        grid.attach(self._layer_mode_combo, 1, row, 5, 1)
        # Opacity slider
        row += 1
        opacity_lbl = Gtk.Label(label=_('Opacity:'))
        opacity_lbl.set_tooltip_text(
          _("Layer opacity: how much of the current layer to use. "
            "Smaller values make it more transparent."))
        opacity_lbl.set_alignment(0, 0.5)
        opacity_lbl.set_hexpand(False)
        adj = Gtk.Adjustment(lower=0, upper=100,
                             step_incr=1, page_incr=10)
        self.opacity_scale = Gtk.HScale(adj)
        self.opacity_scale.set_draw_value(False)
        self.opacity_scale.set_hexpand(True)
        grid.attach(opacity_lbl, 0, row, 1, 1)
        grid.attach(self.opacity_scale, 1, row, 5, 1)
        # Layer list and controls
        row += 1
        layersbox = Gtk.VBox()
        style = layersbox.get_style_context()
        style.add_class(Gtk.STYLE_CLASS_LINKED)
        style = view_scroll.get_style_context()
        style.set_junction_sides(Gtk.JunctionSides.BOTTOM)
        list_tools = inline_toolbar(self.app, [
                ("NewLayerFG", "mypaint-add-symbolic"),
                ("RemoveLayer", "mypaint-remove-symbolic"),
                ("RaiseLayerInStack", "mypaint-up-symbolic"),
                ("LowerLayerInStack", "mypaint-down-symbolic"),
                ("DuplicateLayer", None),
                ("MergeLayer", None),
                ])
        style = list_tools.get_style_context()
        style.set_junction_sides(Gtk.JunctionSides.TOP)
        layersbox.pack_start(view_scroll, True, True)
        layersbox.pack_start(list_tools, False, False)
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
        # Updates
        doc = app.doc.model
        doc.doc_observers.append(self._update)
        self.opacity_scale.connect('value-changed',
                                   self._opacity_scale_changed_cb)
        self._layer_mode_combo.connect('changed',
                                   self._layer_mode_combo_changed_cb)
        self._is_updating = False
        self._update(doc)
        root_stack.expand_layer += self._rootstack_expand_layer_cb
        root_stack.collapse_layer += self._rootstack_collapse_layer_cb

        root_stack.layer_content_changed += self._layer_content_changed


    ## Model update handling

    def _update(self, doc):
        """Updates all controls from the working document"""
        if self._is_updating:
            return
        current_layer = doc.layer_stack.current
        self.opacity_scale.set_value(current_layer.opacity*100)
        self._update_opacity_tooltip()
        for lmm_row in self._layer_mode_model:
            lmm_mode, lmm_name, lmm_desc = lmm_row
            if lmm_mode == current_layer.mode:
                self._layer_mode_combo.set_active_iter(lmm_row.iter)
                tooltip = self.tooltip_format.format(
                    mode_name = escape(lmm_name),
                    mode_description = escape(lmm_desc))
                self._layer_mode_combo.set_tooltip_markup(tooltip)
        self._is_updating = False

    def _current_path_updated_cb(self, rootstack, layerpath):
        """Respond to the current layer changing in the doc-model"""
        # Update the context menu
        current_layer = rootstack.current
        new_layer_class = current_layer.__class__
        if new_layer_class is not self._layer_specific_ui_class:
            ui_manager = self.app.ui_manager
            old_mergeid = self._layer_specific_ui_mergeid
            if old_mergeid is not None:
                ui_manager.remove_ui(old_mergeid)
                self._layer_specific_ui_mergeid = None
            new_ui = LAYER_CLASS_UI.get(new_layer_class)
            if new_ui:
                new_mergeid = ui_manager.add_ui_from_string(new_ui)
                self._layer_specific_ui_mergeid = new_mergeid
            self._layer_specific_ui_class = new_layer_class
        # Have to make the parent row visible first
        if len(layerpath) > 1:
            self._treeview.expand_to_path(Gtk.TreePath(layerpath[:-1]))
        # Update the GTK selection marker to match the model
        old_layerpath = None
        sel = self._treeview.get_selection()
        model, selected_paths = sel.get_selected_rows()
        if len(selected_paths) > 0:
            old_treepath = selected_paths[0]
            old_layerpath = tuple(old_treepath.get_indices())
        if layerpath == old_layerpath:
            return
        treepath = Gtk.TreePath(layerpath)
        sel.unselect_all()
        sel.select_path(treepath)
        # Make it visible
        self._scroll_to_current_layer()

    def _update_opacity_tooltip(self):
        """Updates the opacity slider's tooltip to show the current opacity"""
        scale = self.opacity_scale
        tmpl = _("Layer opacity: %d%%")
        scale.set_tooltip_text(tmpl % (scale.get_value(),))

    def _scroll_to_current_layer(self):
        """Scroll the layers listview to show the current layer"""
        sel = self._treeview.get_selection()
        tree_model, sel_row_paths = sel.get_selected_rows()
        if len(sel_row_paths) > 0:
            sel_row_path = sel_row_paths[0]
            self._treeview.scroll_to_cell(sel_row_path)

    def _view_button_press_cb(self, view, event):
        """Handle button presses (visibility, locked, naming)"""
        if self._is_updating:
            return True
        # Basic details about the click
        modifiers_mask = ( Gdk.ModifierType.CONTROL_MASK |
                           Gdk.ModifierType.SHIFT_MASK )
        modifiers_held = (event.get_state() & modifiers_mask)
        double_click = (event.type == Gdk.EventType._2BUTTON_PRESS)
        is_menu = event.triggers_context_menu()
        # Determine which row & column was clicked
        x, y = int(event.x), int(event.y)
        bw_x, bw_y = view.convert_widget_to_bin_window_coords(x, y)
        click_info = view.get_path_at_pos(bw_x, bw_y)
        if click_info is None:
            return True
        treemodel = self._treemodel
        click_treepath, click_col, cell_x, cell_y = click_info
        layer = treemodel.get_layer(treepath=click_treepath)
        docmodel = self.app.doc.model
        rootstack = docmodel.layer_stack
        # Eye/visibility column toggles kinds of visibility
        if (click_col is self._visible_col) and not is_menu:
            if modifiers_held:
                current_solo = rootstack.current_layer_solo
                rootstack.current_layer_solo = not current_solo
            elif rootstack.current_layer_solo:
                rootstack.current_layer_solo = False
            else:
                new_visible = not layer.visible
                docmodel.set_layer_visibility(new_visible, layer)
            return True
        # Layer lock column
        elif (click_col is self._locked_col) and not is_menu:
            new_locked = not layer.locked
            docmodel.set_layer_locked(new_locked, layer)
            return True
        # Fancy clicks on names allow the layer to be renamed
        elif (click_col is self._name_col) and not is_menu:
            if modifiers_held or double_click:
                rename_action = self.app.find_action("RenameLayer")
                rename_action.activate()
                return True

        # Click an un-selected layer row to select it
        click_layerpath = tuple(click_treepath.get_indices())
        if click_layerpath != rootstack.current_path:
            docmodel.select_layer(path=click_layerpath)
            self.app.doc.layerblink_state.activate()

        # The type icon column allows a layer-type-specific action to be
        # invoked with a single click.
        if (click_col is self._type_col) and not is_menu:
            layer.activate_layertype_action()
            return True

        # Context menu
        if is_menu and event.type == Gdk.BUTTON_PRESS:
            self._popup_context_menu(event)
            return True

        # Default behaviours: allow expanders & drag-and-drop to work
        return False

    def _opacity_scale_changed_cb(self, *ignore):
        if self._is_updating:
            return
        self._is_updating = True
        opacity = self.opacity_scale.get_value() / 100.0
        docmodel = self.app.doc.model
        docmodel.set_layer_opacity(opacity)
        self._update_opacity_tooltip()
        self._scroll_to_current_layer()
        self._is_updating = False

    def _layer_mode_combo_changed_cb(self, *ignored):
        """Propagate the user's choice of layer mode to the model"""
        if self._is_updating:
            return
        self._is_updating = True
        docmodel = self.app.doc.model
        it = self._layer_mode_combo.get_active_iter()
        mode, label, desc = self._layer_mode_model.get(it, 0, 1, 2)
        docmodel.set_layer_mode(mode)
        tooltip = self.tooltip_format.format(
                    mode_name = escape(label),
                    mode_description = escape(desc), )
        self._layer_mode_combo.set_tooltip_markup(tooltip)
        self._scroll_to_current_layer()
        self._is_updating = False

    def _rootstack_expand_layer_cb(self, rootstack, path):
        treepath = Gtk.TreePath(path)
        self._treeview.expand_to_path(treepath)

    def _rootstack_collapse_layer_cb(self, rootstack, path):
        treepath = Gtk.TreePath(path)
        self._treeview.collapse_row(treepath)

    def _layer_content_changed(self, rootstack, layer, *args):
        self._scroll_to_current_layer()

    def _popup_context_menu(self, event=None):
        """Display the popup context menu"""
        if event is None:
            time = Gtk.get_current_event_time()
            button = 0
        else:
            time = event.time
            button = event.button
        self._menu.popup(None, None, None, None, button, time)

    def _popup_menu_cb(self, widget):
        """Handler for "popup-menu" GtkEvents"""
        self._popup_context_menu(None)
        return True
