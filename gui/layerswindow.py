import pygtkcompat

import gtk
gdk = gtk.gdk
from gettext import gettext as _
import gobject
import pango

import dialogs
from lib.layer import COMPOSITE_OPS
from lib.helpers import escape

def stock_button(stock_id):
    b = gtk.Button()
    img = gtk.Image()
    img.set_from_stock(stock_id, gtk.ICON_SIZE_MENU)
    b.add(img)
    return b

def action_button(action):
    b = gtk.Button()
    b.set_related_action(action)
    if b.get_child() is not None:
        b.remove(b.get_child())
    img = action.create_icon(gtk.ICON_SIZE_MENU)
    img.set_tooltip_text(action.get_tooltip())
    b.add(img)
    return b

def make_composite_op_model():
    model = gtk.ListStore(str, str)
    for name, display_name in COMPOSITE_OPS:
        model.append([name, display_name])
    return model


class ToolWidget (gtk.VBox):

    stock_id = "mypaint-tool-layers"
    tool_widget_title = _("Layers")

    def __init__(self, app):
        gtk.VBox.__init__(self)
        self.app = app
        #self.set_size_request(200, 250)

        # Layer treeview
        # The 'object' column is a layer. All displayed columns use data from it.
        store = self.liststore = gtk.ListStore(object)
        store.connect("row-deleted", self.liststore_drag_row_deleted_cb)
        view = self.treeview = gtk.TreeView(store)
        view.connect("cursor-changed", self.treeview_cursor_changed_cb)
        view.set_reorderable(True)
        view.set_headers_visible(False)
        view.connect("button-press-event", self.treeview_button_press_cb)
        view_scroll = gtk.ScrolledWindow()
        view_scroll.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        view_scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        view_scroll.add(view)

        renderer = gtk.CellRendererPixbuf()
        col = self.visible_col = gtk.TreeViewColumn(_("Visible"))
        col.pack_start(renderer, expand=False)
        col.set_cell_data_func(renderer, self.layer_visible_datafunc)
        view.append_column(col)

        renderer = gtk.CellRendererPixbuf()
        col = self.locked_col = gtk.TreeViewColumn(_("Locked"))
        col.pack_start(renderer, expand=False)
        col.set_cell_data_func(renderer, self.layer_locked_datafunc)
        view.append_column(col)

        renderer = gtk.CellRendererText()
        col = self.name_col = gtk.TreeViewColumn(_("Name"))
        col.pack_start(renderer, expand=True)
        col.set_cell_data_func(renderer, self.layer_name_datafunc)
        view.append_column(col)

        # Common controls

        common_table = gtk.Table()
        row = 0

        layer_mode_lbl = gtk.Label(_('Mode:'))
        layer_mode_lbl.set_alignment(0, 0.5)
        self.layer_mode_model = make_composite_op_model()
        self.layer_mode_combo = gtk.ComboBox()
        self.layer_mode_combo.set_model(self.layer_mode_model)
        cell1 = gtk.CellRendererText()
        self.layer_mode_combo.pack_start(cell1)
        self.layer_mode_combo.add_attribute(cell1, "text", 1)
        common_table.attach(layer_mode_lbl, 0, 1, row, row+1, gtk.FILL)
        common_table.attach(self.layer_mode_combo, 1, 2, row, row+1, gtk.FILL|gtk.EXPAND)
        row += 1

        opacity_lbl = gtk.Label(_('Opacity:'))
        opacity_lbl.set_alignment(0, 0.5)
        adj = gtk.Adjustment(lower=0, upper=100, step_incr=1, page_incr=10)
        self.opacity_scale = gtk.HScale(adj)
        self.opacity_scale.set_value_pos(gtk.POS_RIGHT)
        common_table.attach(opacity_lbl, 0, 1, row, row+1, gtk.FILL)
        common_table.attach(self.opacity_scale, 1, 2, row, row+1, gtk.FILL|gtk.EXPAND)
        row += 1

        add_action = self.app.find_action("NewLayerFG")
        move_up_action = self.app.find_action("RaiseLayerInStack")
        move_down_action = self.app.find_action("LowerLayerInStack")
        merge_down_action = self.app.find_action("MergeLayer")
        del_action = self.app.find_action("RemoveLayer")
        duplicate_action = self.app.find_action("DuplicateLayer")

        add_button = self.add_button = action_button(add_action)
        move_up_button = self.move_up_button = action_button(move_up_action)
        move_down_button = self.move_down_button = action_button(move_down_action)
        merge_down_button = self.merge_down_button = action_button(merge_down_action)
        del_button = self.del_button = action_button(del_action)
        duplicate_button = self.duplicate_button = action_button(duplicate_action)

        buttons_hbox = gtk.HBox()
        buttons_hbox.pack_start(add_button)
        buttons_hbox.pack_start(move_up_button)
        buttons_hbox.pack_start(move_down_button)
        buttons_hbox.pack_start(duplicate_button)
        buttons_hbox.pack_start(merge_down_button)
        buttons_hbox.pack_start(del_button)

        # Pack and add to toplevel
        self.pack_start(view_scroll)
        self.pack_start(buttons_hbox, expand=False)
        self.pack_start(common_table, expand=False)

        # Names for anonymous layers
        # app.filehandler.file_opened_observers.append(self.init_anon_layer_names)
        ## TODO: may need to reset them with the new system too

        # Updates
        doc = app.doc.model
        doc.doc_observers.append(self.update)
        self.opacity_scale.connect('value-changed', self.on_opacity_changed)
        self.layer_mode_combo.connect('changed', self.on_layer_mode_changed)

        self.is_updating = False
        self.update(doc)


    def update(self, doc):
        if self.is_updating:
            return
        self.is_updating = True

        # Update the liststore and the selection to match the master layers
        # list in doc
        current_layer = doc.get_current_layer()
        self.treeview.get_selection().unselect_all()
        liststore_layers = [row[0] for row in self.liststore]
        liststore_layers.reverse()
        if doc.layers != liststore_layers:
            self.liststore.clear()
            for layer in doc.layers:
                self.liststore.prepend([layer])
        selected_path = (len(doc.layers) - (doc.layer_idx + 1), )

        # Queue a selection update too...
        gobject.idle_add(self.update_selection)

        # Update the common widgets
        self.opacity_scale.set_value(current_layer.opacity*100)
        mode = current_layer.compositeop
        def find_iter(model, path, iter, data):
            value = model.get_value(iter, 0)
            if value == mode:
                self.layer_mode_combo.set_active_iter(iter)
        self.layer_mode_model.foreach(find_iter, None)
        self.is_updating = False


    def update_selection(self):
        doc = self.app.doc.model
        # ... select_path() ust be queued with gobject.idle_add to avoid
        # glitches in the update after dragging the current row downwards.
        selected_path = (len(doc.layers) - (doc.layer_idx + 1), )
        self.treeview.get_selection().select_path(selected_path)
        self.treeview.scroll_to_cell(selected_path)


    def treeview_cursor_changed_cb(self, treeview, *data):
        if self.is_updating:
            return
        selection = treeview.get_selection()
        if selection is None:
            return
        store, t_iter = selection.get_selected()
        if t_iter is None:
            return
        layer = store.get_value(t_iter, 0)
        doc = self.app.doc
        if doc.model.get_current_layer() != layer:
            idx = doc.model.layers.index(layer)
            doc.model.select_layer(idx)
            doc.layerblink_state.activate()


    def treeview_button_press_cb(self, treeview, event):
        x, y = int(event.x), int(event.y)
        bw_x, bw_y = treeview.convert_widget_to_bin_window_coords(x, y)
        path_info = treeview.get_path_at_pos(bw_x, bw_y)
        if path_info is None:
            return False
        clicked_path, clicked_col, cell_x, cell_y = path_info
        if pygtkcompat.USE_GTK3:
            clicked_path = clicked_path.get_indices()
        layer, = self.liststore[clicked_path[0]]
        doc = self.app.doc.model
        if clicked_col is self.visible_col:
            doc.set_layer_visibility(not layer.visible, layer)
            self.treeview.queue_draw()
            return True
        elif clicked_col is self.locked_col:
            doc.set_layer_locked(not layer.locked, layer)
            self.treeview.queue_draw()
            return True
        elif clicked_col is self.name_col:
            if event.type == gdk._2BUTTON_PRESS:
                rename_action = self.app.find_action("RenameLayer")
                rename_action.activate()
                return True
        return False


    def liststore_drag_row_deleted_cb(self, liststore, path):
        if self.is_updating:
            return
        # Must be internally generated
        # The only way this can happen is at the end of a drag which reorders the list.
        self.resync_doc_layers()


    def resync_doc_layers(self):
        assert not self.is_updating
        new_order = [row[0] for row in self.liststore]
        new_order.reverse()
        doc = self.app.doc.model
        if new_order != doc.layers:
            doc.reorder_layers(new_order)
        else:
            doc.select_layer(doc.layer_idx)
            # otherwise the current layer selection is visually lost


    def on_opacity_changed(self, *ignore):
        if self.is_updating:
            return
        self.is_updating = True
        doc = self.app.doc.model
        doc.set_layer_opacity(self.opacity_scale.get_value()/100.0)
        self.is_updating = False


    def on_layer_del(self, button):
        doc = self.app.doc.model
        doc.remove_layer(layer=doc.get_current_layer())


    def layer_name_datafunc(self, column, renderer, model, tree_iter,
                            *data_etc):
        layer = model.get_value(tree_iter, 0)
        path = model.get_path(tree_iter)
        name = layer.name
        attrs = pango.AttrList()
        if not name:
            layer_num = self.app.doc.get_number_for_nameless_layer(layer)
            name = _(u"Untitled layer #%d") % layer_num
            markup = "<small><i>%s</i></small> " % (escape(name),)
            if pygtkcompat.USE_GTK3:
                parse_result = pango.parse_markup(markup, -1, '\000')
                parse_ok, attrs, name, accel_char = parse_result
                assert parse_ok
            else:
                parse_result = pango.parse_markup(markup)
                attrs, name, accel_char = parse_result
        renderer.set_property("attributes", attrs)
        renderer.set_property("text", name)


    def layer_visible_datafunc(self, column, renderer, model, tree_iter,
                               *data_etc):
        layer = model.get_value(tree_iter, 0)
        if layer.visible:
            pixbuf = self.app.pixmaps.eye_open
        else:
            pixbuf = self.app.pixmaps.eye_closed
        renderer.set_property("pixbuf", pixbuf)


    def layer_locked_datafunc(self, column, renderer, model, tree_iter,
                              *data_etc):
        layer = model.get_value(tree_iter, 0)
        if layer.locked:
            pixbuf = self.app.pixmaps.lock_closed
        else:
            pixbuf = self.app.pixmaps.lock_open
        renderer.set_property("pixbuf", pixbuf)


    def on_layer_mode_changed(self, *ignored):
        if self.is_updating:
            return
        self.is_updating = True
        doc = self.app.doc.model
        i = self.layer_mode_combo.get_active_iter()
        mode_name, display_name = self.layer_mode_model.get(i, 0, 1)
        print 'Change layer mode to %s (%s)' % (display_name, mode_name)
        doc.set_layer_compositeop(mode_name)
        self.is_updating = False
