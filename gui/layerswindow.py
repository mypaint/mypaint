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
from logging import getLogger
logger = getLogger(__name__)

from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GObject
from gi.repository import Pango

import lib.layer
from lib.helpers import escape
import widgets
from widgets import inline_toolbar
from workspace import SizedVBoxToolWidget
import layers


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
            </placeholder>
        </popup>
        """),
    (lib.layer.FileBackedLayer, """
        <popup name='LayersWindowPopup'>
            <placeholder name='BasicLayerActions'>
                <menuitem action='BeginExternalLayerEdit'/>
                <menuitem action='CommitExternalLayerEdit'/>
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

    #TRANSLATORS: tooltip for the layer mode dropdown (markup)
    LAYER_MODE_TOOLTIP_MARKUP_TEMPLATE = _("<b>{name}</b>\n{description}")

    #TRANSLATORS: tooltip for the opacity slider (text)
    OPACITY_SCALE_TOOLTIP_TEXT_TEMPLATE = _("Layer opacity: %d%%")

    __gtype_name__ = 'MyPaintLayersTool'

    STATUSBAR_CONTEXT = 'layerstool-dnd'

    #TRANSLATORS: status bar messages for drag, without/with modifiers
    STATUSBAR_DRAG_MSG = _("Move layer in stack...")
    STATUSBAR_DRAG_INTO_MSG = _("Move layer in stack (dropping into a "
                                "regular layer will create a new group)")


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
        # Motion and modifier keys during drag
        view.connect("drag-begin", self._view_drag_begin_cb)
        view.connect("drag-end", self._view_drag_end_cb)
        view.connect("drag-motion", self._view_drag_motion_cb)
        statusbar_cid = app.statusbar.get_context_id(self.STATUSBAR_CONTEXT)
        self._drag_statusbar_context_id = statusbar_cid
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
        self._layer_specific_ui_mergeids = []
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
        # Main layout grid
        grid = Gtk.Grid()
        grid.set_row_spacing(widgets.SPACING_TIGHT)
        grid.set_column_spacing(widgets.SPACING)

        # Mode dropdown
        row = 0
        label = Gtk.Label(label=_('Mode:'))
        label.set_tooltip_text(
            _("Blending mode: how the current layer combines with the "
              "layers underneath it."))
        label.set_alignment(0, 0.5)
        label.set_hexpand(False)
        grid.attach(label, 0, row, 1, 1)

        store = Gtk.ListStore(int, str, bool)
        modes = lib.layer.STACK_MODES + lib.layer.STANDARD_MODES
        for mode in modes:
            label, desc = lib.layer.MODE_STRINGS.get(mode)
            store.append([mode, label, True])
        combo = Gtk.ComboBox()
        combo.set_model(store)
        combo.set_hexpand(True)
        cell = Gtk.CellRendererText()
        combo.pack_start(cell)
        combo.add_attribute(cell, "text", 1)
        combo.add_attribute(cell, "sensitive", 2)
        self._layer_mode_combo = combo

        grid.attach(combo, 1, row, 5, 1)

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
        self._opacity_scale = Gtk.HScale(adj)
        self._opacity_scale.set_draw_value(False)
        self._opacity_scale.set_hexpand(True)
        grid.attach(opacity_lbl, 0, row, 1, 1)
        grid.attach(self._opacity_scale, 1, row, 5, 1)
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
        self._processing_model_updates = False
        self._opacity_scale.connect('value-changed',
                                    self._opacity_scale_changed_cb)
        self._layer_mode_combo.connect('changed',
                                       self._layer_mode_combo_changed_cb)
        rootstack = docmodel.layer_stack
        rootstack.expand_layer += self._expand_layer_cb
        rootstack.collapse_layer += self._collapse_layer_cb
        rootstack.layer_content_changed += self._layer_content_changed
        rootstack.layer_properties_changed += self._layer_propchange_cb
        rootstack.current_layer_solo_changed += self._treeview_redraw_all
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
            self._update_opacity_scale()
        self._processing_model_updates = False

    def _expand_layer_cb(self, rootstack, path):
        if not path:
            return
        treepath = Gtk.TreePath(path)
        self._treeview.expand_to_path(treepath)

    def _collapse_layer_cb(self, rootstack, path):
        if not path:
            return
        treepath = Gtk.TreePath(path)
        self._treeview.collapse_row(treepath)

    def _layer_content_changed(self, rootstack, layer, *args):
        if not layer:
            return
        self._scroll_to_current_layer()

    def _treeview_redraw_all(self, *_ignored):
        self._treeview.queue_draw()


    ## Model update processing

    def _update_all(self):
        assert self._processing_model_updates
        self._update_context_menu()
        self._update_layers_treeview_selection()
        self._update_layer_mode_combo()
        self._update_opacity_scale()

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
        label, desc = lib.layer.MODE_STRINGS.get(current_mode)
        template = self.LAYER_MODE_TOOLTIP_MARKUP_TEMPLATE
        tooltip = template.format(name=escape(label),
                                  description=escape(desc))
        combo.set_tooltip_markup(tooltip)

    def _update_opacity_scale(self):
        """Updates the opacity scale from the model"""
        assert self._processing_model_updates
        rootstack = self.app.doc.model.layer_stack
        layer = rootstack.current
        scale = self._opacity_scale
        opacity_is_adjustable = not (
            layer is None
            or layer is rootstack
            or layer.mode == lib.layer.PASS_THROUGH_MODE
        )
        scale.set_sensitive(opacity_is_adjustable)
        if not opacity_is_adjustable:
            return
        percentage = layer.opacity * 100
        scale.set_value(percentage)
        template = self.OPACITY_SCALE_TOOLTIP_TEXT_TEMPLATE
        tooltip = template % (percentage,)
        scale.set_tooltip_text(tooltip)

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

    def _update_layers_treeview_selection(self):
        assert self._processing_model_updates
        sel = self._treeview.get_selection()
        layerpath = self.app.doc.model.layer_stack.current_path
        if not layerpath:
            sel.unselect_all()
            return
        old_layerpath = None
        model, selected_paths = sel.get_selected_rows()
        if len(selected_paths) > 0:
            old_treepath = selected_paths[0]
            if old_treepath:
                old_layerpath = tuple(old_treepath.get_indices())
        if layerpath == old_layerpath:
            return
        sel.unselect_all()
        if len(layerpath) > 1:
            self._treeview.expand_to_path(Gtk.TreePath(layerpath[:-1]))
        if len(layerpath) > 0:
            sel.select_path(Gtk.TreePath(layerpath))
            self._scroll_to_current_layer()


    ## Updates from the user

    def _view_button_press_cb(self, view, event):
        """Handle button presses (visibility, locked, naming)"""
        if self._processing_model_updates:
            return
        # Basic details about the click
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
            if event.state & Gdk.ModifierType.CONTROL_MASK:
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
        # Double-clicking the name column presents the rename dialog.
        # (This can also be done via the menu)
        elif (click_col is self._name_col) and not is_menu:
            if double_click:
                rename_action = self.app.find_action("RenameLayer")
                rename_action.activate()
                return True
        # Click an un-selected layer row to select it
        click_layerpath = tuple(click_treepath.get_indices())
        if click_layerpath != rootstack.current_path:
            docmodel.select_layer(path=click_layerpath)
            self.app.doc.layerblink_state.activate()
        # The type icon column acts as an extra expander.
        # Some themes' expander arrows are very small.
        if (click_col is self._type_col) and not is_menu:
            self._treeview.expand_to_path(click_treepath)
            return True
        # Context menu
        if is_menu and event.type == Gdk.BUTTON_PRESS:
            self._popup_context_menu(event)
            return True
        # Default behaviours: allow expanders & drag-and-drop to work
        allow_drag_into = bool(event.state & Gdk.ModifierType.SHIFT_MASK)
        self._treemodel.allow_drag_into = allow_drag_into
        return False

    def _view_drag_begin_cb(self, view, context):
        self._treeview_in_drag = True
        statusbar = self.app.statusbar
        statusbar_cid = self._drag_statusbar_context_id
        statusbar.remove_all(statusbar_cid)
        statusbar.push(statusbar_cid, self.STATUSBAR_DRAG_MSG)

    def _view_drag_motion_cb(self, view, context, x, y, t):
        win = view.get_window()
        w, x, y, modifiers = win.get_pointer()
        statusbar = self.app.statusbar
        statusbar_cid = self._drag_statusbar_context_id
        statusbar.remove_all(statusbar_cid)
        if modifiers & Gdk.ModifierType.SHIFT_MASK:
            Gdk.drag_status(context, Gdk.DragAction.PRIVATE, t)
            self._treemodel.allow_drag_into = True
            statusbar.push(statusbar_cid, self.STATUSBAR_DRAG_INTO_MSG)
        else:
            Gdk.drag_status(context, Gdk.DragAction.MOVE, t)
            self._treemodel.allow_drag_into = False
            statusbar.push(statusbar_cid, self.STATUSBAR_DRAG_MSG)
        return False

    def _view_drag_end_cb(self, view, context):
        self._treeview_in_drag = False
        statusbar = self.app.statusbar
        statusbar_cid = self._drag_statusbar_context_id
        statusbar.remove_all(statusbar_cid)

    def _opacity_scale_changed_cb(self, *ignore):
        if self._processing_model_updates:
            return
        opacity = self._opacity_scale.get_value() / 100.0
        docmodel = self.app.doc.model
        docmodel.set_current_layer_opacity(opacity)
        self._scroll_to_current_layer()

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
        label, desc = lib.layer.MODE_STRINGS.get(mode)
        docmodel.set_current_layer_mode(mode)


    ## Utility methods

    def _scroll_to_current_layer(self, *_ignored):
        """Scroll the layers listview to show the current layer"""
        sel = self._treeview.get_selection()
        tree_model, sel_row_paths = sel.get_selected_rows()
        if len(sel_row_paths) > 0:
            sel_row_path = sel_row_paths[0]
            if sel_row_path:
                self._treeview.scroll_to_cell(sel_row_path)

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
