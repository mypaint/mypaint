# This file is part of MyPaint.
# Copyright (C) 2009 by Ilya Portnov <portnov@bk.ru>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layers panel"""


## Imports

from gettext import gettext as _
from difflib import SequenceMatcher
from logging import getLogger
logger = getLogger(__name__)
import os.path

from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GdkPixbuf
from gi.repository import GObject
from gi.repository import Pango

import dialogs
import lib.layer
from lib.tiledsurface import COMBINE_MODE_STRINGS
from lib.helpers import escape
import widgets
from workspace import SizedVBoxToolWidget


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

def inline_toolbar(app, tool_defs):
    bar = Gtk.Toolbar()
    bar.set_style(Gtk.ToolbarStyle.ICONS)
    bar.set_icon_size(widgets.ICON_SIZE_SMALL)
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
    return bar

def make_layer_mode_model():
    model = Gtk.ListStore(int, str, str)
    for mode, (label, desc) in enumerate(COMBINE_MODE_STRINGS):
        model.append([mode, label, desc])
    return model


## Module constants


TREESTORE_PATH_COL = 0
TREESTORE_LAYER_COL = 1



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
        self.set_border_width(widgets.SPACING_CRAMPED)

        # Layer treestore
        store = Gtk.TreeStore(object, object)  # layerpath, layer
        store.connect("row-deleted", self._treestore_row_deleted_cb)
        self.treestore = store

        # Layer treeview
        view = self.treeview = Gtk.TreeView(store)
        view.set_reorderable(True)
        view.set_headers_visible(False)
        view.connect("button-press-event", self._treeview_button_press_cb)
        view_scroll = Gtk.ScrolledWindow()
        view_scroll.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
        view_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        view_scroll.add(view)
        view_scroll.set_size_request(-1, 100)
        sel = view.get_selection()
        sel.set_mode(Gtk.SelectionMode.SINGLE)
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

        # Type and name
        renderer = Gtk.CellRendererPixbuf()
        col = self.type_col = Gtk.TreeViewColumn(_("Type"))
        col.pack_start(renderer, expand=False)
        col.set_cell_data_func(renderer, self._layer_type_datafunc)
        col.set_max_width(24)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        view.append_column(col)

        renderer = Gtk.CellRendererText()
        renderer.set_property("ellipsize", Pango.EllipsizeMode.END)
        col = self.name_col = Gtk.TreeViewColumn(_("Name"))
        col.pack_start(renderer, expand=True)
        col.set_cell_data_func(renderer, self._layer_name_datafunc)
        col.set_expand(True)
        col.set_min_width(48)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        view.append_column(col)

        # State icons
        renderer = Gtk.CellRendererPixbuf()
        col = self.visible_col = Gtk.TreeViewColumn(_("Visible"))
        col.pack_start(renderer, expand=False)
        col.set_cell_data_func(renderer, self._layer_visible_datafunc)
        col.set_max_width(24)
        view.append_column(col)

        renderer = Gtk.CellRendererPixbuf()
        col = self.locked_col = Gtk.TreeViewColumn(_("Locked"))
        col.pack_start(renderer, expand=False)
        col.set_cell_data_func(renderer, self._layer_locked_datafunc)
        col.set_max_width(24)
        view.append_column(col)

        # View appearance
        view.set_show_expanders(True)
        view.set_enable_tree_lines(True)
        view.set_expander_column(self.name_col)

        # Controls for the current layer

        layer_controls = Gtk.Table()
        layer_controls.set_row_spacings(widgets.SPACING_CRAMPED)
        layer_controls.set_col_spacings(widgets.SPACING_CRAMPED)
        row = 0

        layer_mode_lbl = Gtk.Label(label=_('Mode:'))
        layer_mode_lbl.set_tooltip_text(
          _("Blending mode: how the current layer combines with the "
            "layers underneath it."))
        layer_mode_lbl.set_alignment(0, 0.5)
        self.layer_mode_model = make_layer_mode_model()
        self._layer_mode_combo = Gtk.ComboBox()
        self._layer_mode_combo.set_model(self.layer_mode_model)
        cell1 = Gtk.CellRendererText()
        self._layer_mode_combo.pack_start(cell1)
        self._layer_mode_combo.add_attribute(cell1, "text", 1)
        layer_controls.attach(layer_mode_lbl, 0, 1, row, row+1,
                              Gtk.AttachOptions.FILL)
        layer_controls.attach(self._layer_mode_combo, 1, 2, row, row+1,
                              Gtk.AttachOptions.FILL|Gtk.AttachOptions.EXPAND)
        row += 1

        opacity_lbl = Gtk.Label(label=_('Opacity:'))
        opacity_lbl.set_tooltip_text(
          _("Layer opacity: how much of the current layer to use. Smaller "
            "values make it more transparent."))
        opacity_lbl.set_alignment(0, 0.5)
        adj = Gtk.Adjustment(lower=0, upper=100, step_incr=1, page_incr=10)
        self.opacity_scale = Gtk.HScale(adj)
        self.opacity_scale.set_draw_value(False)
        layer_controls.attach(opacity_lbl, 0, 1, row, row+1,
                              Gtk.AttachOptions.FILL)
        layer_controls.attach(self.opacity_scale, 1, 2, row, row+1,
                              Gtk.AttachOptions.FILL|Gtk.AttachOptions.EXPAND)

        # Background layer controls

        show_bg_btn = Gtk.CheckButton()
        change_bg_action = self.app.find_action("BackgroundWindow")
        change_bg_btn = widgets.borderless_button(action=change_bg_action)
        show_bg_action = self.app.find_action("ShowBackgroundToggle")
        show_bg_btn.set_related_action(show_bg_action)
        bg_hbox = Gtk.HBox()
        bg_hbox.pack_start(show_bg_btn, True, True, 0)
        bg_hbox.pack_start(change_bg_btn, False, True, 0)

        # Layer list action buttons

        list_tools = inline_toolbar(self.app, [
            ("NewLayerFG", "mypaint-add-symbolic"),
            ("RemoveLayer", "mypaint-remove-symbolic"),
            ("RaiseLayerInStack", "mypaint-up-symbolic"),
            ("LowerLayerInStack", "mypaint-down-symbolic"),
            ("DuplicateLayer", None),
            ("MergeLayer", None),
        ])

        view_grid = Gtk.Grid()
        view_grid.set_row_spacing(0)
        view_scroll.set_hexpand(True)
        view_scroll.set_vexpand(True)
        view_grid.attach(view_scroll, 0, 0, 1, 1)
        list_tools.set_hexpand(False)
        list_tools.set_vexpand(False)
        view_grid.attach(list_tools, 0, 1, 1, 1)

        # Pack and add to toplevel
        self.pack_start(layer_controls, False, True, 0)
        self.pack_start(view_grid, True, True, 0)
        self.pack_start(bg_hbox, False, True, 0)

        # Updates
        doc = app.doc.model
        doc.doc_observers.append(self._update)
        self.opacity_scale.connect('value-changed', self._opacity_scale_changed_cb)
        self._layer_mode_combo.connect('changed', self._layer_mode_combo_changed_cb)
        self.is_updating = False
        self._update(doc)

        # TODO: observe changes to the document & scroll to the highlighted
        # row when the user draws something.


    def _update(self, doc):
        """Updates all controls from the working document"""
        if self.is_updating:
            return
        self.is_updating = True
        selection = self.treeview.get_selection()
        selection.unselect_all()
        self._update_layers_treestore(doc)
        self._update_selection()
        self._update_compositing_widgets(doc)
        self.is_updating = False


    def _layers_treestore_deeprows(self):
        """Enumerates the treestore's rows in display order"""
        row_queue = list(self.treestore)
        while len(row_queue) > 0:
            row = row_queue.pop(0)
            yield row
            row_children = list(row.iterchildren())
            if len(row_children) > 0:
                row_queue[:0] = row_children


    def _layers_treestore_deepenumerate(self):
        """Enumerates the treestore's layers with layer paths, in render order

        If the models are identical, this is designed to produce the same
        sequence as `lib.layers.RootLayerStack.deepenumerate()`.
        """
        store = self.treestore
        queue = [((), None)]  # start with the root
        while len(queue) > 0:
            row_layerpath, row_iter = queue.pop(0)
            if row_iter is not None:
                # Process a row
                row_layer = store.get_value(row_iter, TREESTORE_LAYER_COL)
                yield (row_layerpath, row_layer)
            # Determine whether the row has any child rows
            nchildren = store.iter_n_children(row_iter)
            if nchildren <= 0:
                continue
            # Queue all the children next, in render order
            child_iter = store.iter_nth_child(row_iter, nchildren-1)
            i = 0
            while child_iter is not None:
                child_layerpath = tuple(list(row_layerpath) + [i])
                queue.insert(i, (child_layerpath, child_iter))
                child_iter = store.iter_previous(child_iter)
                i += 1


    def _update_layers_treestore(self, doc):
        """Updates the layers store from the model's layer stack"""
        assert self.is_updating
        self.treestore.clear()
        path2iter = {}
        for path, layer in doc.layer_stack.deepenumerate():
            assert len(path) > 0
            if len(path) == 1:
                parent_iter = None
            else:
                parent = path[:-1]
                parent_iter = path2iter.get(parent)
                assert parent_iter is not None
            i = self.treestore.prepend(parent_iter)
            self.treestore.set_value(i, TREESTORE_PATH_COL, path)
            self.treestore.set_value(i, TREESTORE_LAYER_COL, layer)
            path2iter[path] = i


    def _update_compositing_widgets(self, doc):
        """Updates widgets that control compositing from the current layer"""
        assert self.is_updating
        # Update the common widgets
        current_layer = doc.layer_stack.current
        self.opacity_scale.set_value(current_layer.opacity*100)
        self._update_opacity_tooltip()
        for lmm_row in self.layer_mode_model:
            lmm_mode, lmm_name, lmm_desc = lmm_row
            if lmm_mode == current_layer.mode:
                self._layer_mode_combo.set_active_iter(lmm_row.iter)
                tooltip = self.tooltip_format.format(
                    mode_name = escape(lmm_name),
                    mode_description = escape(lmm_desc))
                self._layer_mode_combo.set_tooltip_markup(tooltip)


    def _update_selection(self):
        """Update the TreeView's selected row from the document model"""
        # Move selection line to the model's current layer and scroll to it.
        # Layer stack paths are not the same thing as treestore paths.
        doc = self.app.doc.model
        current_layer_lspath = doc.layer_stack.current_path
        current_layer_tspath = None
        for row in self._layers_treestore_deeprows():
            row_lspath = row[TREESTORE_PATH_COL]
            if row_lspath == current_layer_lspath:
                current_layer_tspath = row.path
                break
        if current_layer_tspath is None:
            return
        selection = self.treeview.get_selection()
        was_selected = selection.path_is_selected(current_layer_tspath)
        selection.unselect_all()
        selection.select_path(current_layer_tspath)
        if not was_selected:
            # Only do this if the required layer is not already highlighted to
            # allow users to scroll and change distant layers' visibility or
            # locked state in batches: https://gna.org/bugs/?20330
            self.treeview.scroll_to_cell(current_layer_tspath)

        # Queue a redraw too - undoing/redoing lock and visible Commands
        # updates the underlying doc state, and we need to present the current
        # state via the icons at all times.
        self.treeview.queue_draw()

        # Update layer class specific parts of the UI
        current_layer = doc.layer_stack.current
        new_layer_class = current_layer.__class__
        if new_layer_class is not self._layer_specific_ui_class:
            old_mergeid = self._layer_specific_ui_mergeid
            if old_mergeid is not None:
                self.app.ui_manager.remove_ui(old_mergeid)
                self._layer_specific_ui_mergeid = None
            new_ui = LAYER_CLASS_UI.get(new_layer_class)
            if new_ui:
                new_mergeid = self.app.ui_manager.add_ui_from_string(new_ui)
                self._layer_specific_ui_mergeid = new_mergeid
            self._layer_specific_ui_class = new_layer_class

    def _update_opacity_tooltip(self):
        """Updates the opacity slider's tooltip to show the current opacity"""
        scale = self.opacity_scale
        tmpl = _("Layer opacity: %d%%")
        scale.set_tooltip_text(tmpl % (scale.get_value(),))

    def _scroll_to_current_layer(self):
        """Scroll the layers listview to show the current layer"""
        sel = self.treeview.get_selection()
        tree_model, sel_row_paths = sel.get_selected_rows()
        if len(sel_row_paths) > 0:
            sel_row_path = sel_row_paths[0]
            self.treeview.scroll_to_cell(sel_row_path)


    def _treeview_button_press_cb(self, treeview, event):
        """Handle button presses (visibility, locked, naming)"""
        if self.is_updating:
            return True
        modifiers = Gdk.ModifierType.CONTROL_MASK | Gdk.ModifierType.SHIFT_MASK
        modifiers_held = (event.get_state() & modifiers)
        double_click = (event.type == Gdk._2BUTTON_PRESS)
        x, y = int(event.x), int(event.y)
        bw_x, bw_y = treeview.convert_widget_to_bin_window_coords(x, y)
        path_info = treeview.get_path_at_pos(bw_x, bw_y)
        if path_info is None:
            return True
        treestore = self.treestore
        clicked_path, clicked_col, cell_x, cell_y = path_info
        clicked_iter = treestore.get_iter(clicked_path)
        layer = treestore.get_value(clicked_iter, TREESTORE_LAYER_COL)
        layer_path = treestore.get_value(clicked_iter, TREESTORE_PATH_COL)
        doc = self.app.doc
        model = doc.model

        is_menu = event.triggers_context_menu()

        # Eye/visibility column toggles kinds of visibility
        if clicked_col is self.visible_col:
            if modifiers_held:
                current = model.layer_stack.get_current_layer_solo()
                model.layer_stack.set_current_layer_solo(not current)
            elif model.layer_stack.get_current_layer_solo():
                model.layer_stack.set_current_layer_solo(False)
            else:
                model.set_layer_visibility(not layer.visible, layer)
                self.treeview.queue_draw()
            return True
        # Layer lock column
        elif clicked_col is self.locked_col and not is_menu:
            model.set_layer_locked(not layer.locked, layer)
            self.treeview.queue_draw()
            return True
        # Fancy clicks on names allow the layer to be renamed
        elif clicked_col is self.name_col and not is_menu:
            if modifiers_held or double_click:
                rename_action = self.app.find_action("RenameLayer")
                rename_action.activate()
                return True

        # Click an un-selected layer row to select it
        if layer_path != model.layer_stack.current_path:
            model.select_layer(path=layer_path)
            doc.layerblink_state.activate()

        # The type icon column allows a layer-type-specific action to be
        # invoked with a single click.
        if clicked_col is self.type_col and not is_menu:
            layer.activate_layertype_action()
            self._update(model)  # the action may require it
            return True

        if is_menu and event.type == Gdk.BUTTON_PRESS:
            self._popup_context_menu(event)
            return True

        # Allow the default drag initiation to happen if the user click+drags
        # starting with the current layer
        return False


    def _treestore_row_deleted_cb(self, store, path):
        """Detect layer drag moves when row-deleted is received"""
        # The deletion could be a result of update. Ignore those.
        if self.is_updating:
            return
        # If not, it can only be the end result of a drag-and-drop layer move
        # made by the user.
        self._handle_layer_drag()


    def _layers_treestore_deepenumerate(self):
        """Enumerates the treestore with layer paths, in render order

        If the models are identical, this is designed to produce the same
        sequence as `lib.layers.RootLayerStack.deepenumerate()`.
        """
        store = self.treestore
        queue = [((), None)]  # start with the root
        while len(queue) > 0:
            row_layerpath, row_iter = queue.pop(0)
            if row_iter is not None:
                # Process a row
                row_layer = store.get_value(row_iter, TREESTORE_LAYER_COL)
                yield (row_layerpath, row_layer)
            # Determine whether the row has any child rows
            nchildren = store.iter_n_children(row_iter)
            if nchildren <= 0:
                continue
            # Queue all the children next, in render order
            child_iter = store.iter_nth_child(row_iter, nchildren-1)
            i = 0
            while child_iter is not None:
                child_path = tuple(list(row_layerpath) + [i])
                queue.insert(i, (child_path, child_iter))
                child_iter = store.iter_previous(child_iter)
                i += 1

    def _handle_layer_drag(self):
        """Process a layer drag move detected by row-deleted"""
        doc = self.app.doc.model

        # Old and new layer stacking details, for comparison
        old_parent = {}
        old_children = {}
        old_path = {}
        for path, layer in doc.layer_stack.deepenumerate():
            parent_path = path[:-1]
            if len(parent_path) == 0:
                parent = doc.layer_stack
            else:
                parent = doc.layer_stack.deepget(parent_path)
            old_parent[layer] = parent
            if parent not in old_children:
                old_children[parent] = []
            old_children[parent].append(layer)
            old_path[layer] = path
        new_layers_enum = list(self._layers_treestore_deepenumerate())
        new_layers_map = dict(new_layers_enum)
        new_parent = {}
        new_children = {}
        new_path = {}
        for path, layer in new_layers_enum:
            parent_path = path[:-1]
            if len(parent_path) == 0:
                parent = doc.layer_stack
            else:
                parent = new_layers_map[parent_path]
            new_parent[layer] = parent
            if parent not in new_children:
                new_children[parent] = []
            new_children[parent].append(layer)
            new_path[layer] = path

        # Detect changes of parent first: doing this affects indices too
        layers = list(doc.layer_stack.deepiter())
        for layer in layers:
            if old_parent[layer] is not new_parent[layer]:
                logger.debug("layer reparented: %r", layer)
                doc.move_layer_in_stack(old_path[layer], new_path[layer])
                return

        # Detect changes of position within the same parent
        reordered_parent = None
        for layer in layers:
            if old_path[layer][-1] != new_path[layer][-1]:
                parent = old_parent[layer]
                if reordered_parent is None:
                    reordered_parent = parent
                assert reordered_parent is parent, ("Only changing the "
                                "ordering of a single parent is supported "
                                "by the drag-handling code")
        if reordered_parent is not None:
            logger.debug("layer reordered: %r", reordered_parent)
            oldkids = old_children[reordered_parent]
            newkids = new_children[reordered_parent]
            # All items in the parent may have just received a new path, so
            # use difflib to compute the minimal move.
            seqmatch = SequenceMatcher(None, oldkids, newkids)
            opcodes = seqmatch.get_opcodes()
            deletions = [op for op in opcodes if op[0] == 'delete']
            insertions = [op for op in opcodes if op[0] == 'insert']
            assert len(deletions) == 1
            assert len(insertions) == 1
            (del_i1, del_i2, del_j1, del_j2) = deletions[0][1:]
            (ins_i1, ins_i2, ins_j1, ins_j2) = insertions[0][1:]
            assert del_i2 == del_i1 + 1
            assert ins_j2 == ins_j1 + 1
            moved_child = oldkids[del_i1]
            assert moved_child is newkids[ins_j1]
            logger.debug("minimal move is %r alone", moved_child)
            doc.move_layer_in_stack(old_path[moved_child],
                                    new_path[moved_child])
            return

        # Otherwise, just ensure the selection is not lost.
        doc.select_layer(path=doc.layer_stack.current_path,
                         user_initiated=False)
    def _opacity_scale_changed_cb(self, *ignore):
        if self.is_updating:
            return
        self.is_updating = True
        doc = self.app.doc.model
        doc.set_layer_opacity(self.opacity_scale.get_value()/100.0)
        self._update_opacity_tooltip()
        self._scroll_to_current_layer()
        self.is_updating = False


    def _layer_name_datafunc(self, column, renderer, model, tree_iter,
                             *data_etc):
        layer = model.get_value(tree_iter, TREESTORE_LAYER_COL)
        path = model.get_path(tree_iter)
        name = layer.name
        if name is None or name == '':
            layers = self.app.doc.model.layer_stack
            layer.assign_unique_name(layers.get_names())
            name = layer.name
        attrs = Pango.AttrList()
        if isinstance(layer, lib.layer.LayerStack):
            markup = "<i>%s</i>" % (escape(name),)
            parse_result = Pango.parse_markup(markup, -1, '\000')
            parse_ok, attrs, name, accel_char = parse_result
            assert parse_ok
        renderer.set_property("attributes", attrs)
        renderer.set_property("text", name)


    def _layer_visible_datafunc(self, column, renderer, model, tree_iter,
                                *data_etc):
        layer = model.get_value(tree_iter, TREESTORE_LAYER_COL)
        layers = self.app.doc.model.layer_stack
        # Layer visibility is based on the layer's natural hidden/visible flag
        visible = layer.visible
        # But the layer stack can override that, and sometimes we need to
        # respect that and show a more appropriate icon
        greyed_out = False
        if layers.get_current_layer_solo():
            visible = (layer is layers.current)
            greyed_out = True
        # Pick icon
        vis_infix = "-visible" if visible else "-hidden"
        sens_infix = "-insensitive" if greyed_out else ""
        icon_name = "mypaint-object%s%s-symbolic" % (vis_infix, sens_infix)
        renderer.set_property("icon-name", icon_name)


    def _layer_locked_datafunc(self, column, renderer, model, tree_iter,
                               *data_etc):
        layer = model.get_value(tree_iter, TREESTORE_LAYER_COL)
        if layer.locked:
            icon_name = "mypaint-object-locked-symbolic"
        else:
            icon_name = "mypaint-object-unlocked-symbolic"
        renderer.set_property("icon-name", icon_name)


    def _layer_type_datafunc(self, column, renderer, model, tree_iter,
                             *data_etc):
        layer = model.get_value(tree_iter, TREESTORE_LAYER_COL)
        renderer.set_property("icon-name", layer.get_icon_name())


    def _layer_mode_combo_changed_cb(self, *ignored):
        """Propagate the user's choice of layer mode to the model"""
        if self.is_updating:
            return
        self.is_updating = True
        doc = self.app.doc.model
        i = self._layer_mode_combo.get_active_iter()
        mode, display_name, desc = self.layer_mode_model.get(i, 0, 1, 2)
        doc.set_layer_mode(mode)
        tooltip = self.tooltip_format.format(
            mode_name = escape(display_name),
            mode_description = escape(desc))
        self._layer_mode_combo.set_tooltip_markup(tooltip)
        self._scroll_to_current_layer()
        self.is_updating = False

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
