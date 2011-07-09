import gtk
gdk = gtk.gdk
from gettext import gettext as _
import gobject
import pango

import dialogs

def stock_button(stock_id):
    b = gtk.Button()
    img = gtk.Image()
    img.set_from_stock(stock_id, gtk.ICON_SIZE_MENU)
    b.add(img)
    return b


class ToolWidget (gtk.VBox):

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
        adj = gtk.Adjustment(lower=0, upper=100, step_incr=1, page_incr=10)
        self.opacity_scale = gtk.HScale(adj)
        self.opacity_scale.set_value_pos(gtk.POS_RIGHT)
        opacity_lbl = gtk.Label(_('Opacity:'))
        opacity_hbox = gtk.HBox()
        opacity_hbox.pack_start(opacity_lbl, expand=False)
        opacity_hbox.pack_start(self.opacity_scale, expand=True)

        add_button = self.add_button = stock_button(gtk.STOCK_ADD)
        move_up_button = self.move_up_button = stock_button(gtk.STOCK_GO_UP)
        move_down_button = self.move_down_button = stock_button(gtk.STOCK_GO_DOWN)
        merge_down_button = self.merge_down_button = stock_button(gtk.STOCK_DND_MULTIPLE)  # XXX need a better one
        del_button = self.del_button = stock_button(gtk.STOCK_DELETE)

        add_button.connect('clicked', self.on_layer_add)
        move_up_button.connect('clicked', self.move_layer, 'up')
        move_down_button.connect('clicked', self.move_layer, 'down')
        merge_down_button.connect('clicked', self.merge_layer_down)
        del_button.connect('clicked', self.on_layer_del)

        merge_down_button.set_tooltip_text(_('Merge Down'))

        buttons_hbox = gtk.HBox()
        buttons_hbox.pack_start(add_button)
        buttons_hbox.pack_start(move_up_button)
        buttons_hbox.pack_start(move_down_button)
        buttons_hbox.pack_start(merge_down_button)
        buttons_hbox.pack_start(del_button)

        # Pack and add to toplevel
        self.pack_start(view_scroll)
        self.pack_start(buttons_hbox, expand=False)
        self.pack_start(opacity_hbox, expand=False)

        # Updates
        doc = app.doc.model
        doc.doc_observers.append(self.update)
        self.opacity_scale.connect('value-changed', self.on_opacity_changed)

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
        self.is_updating = False


    def update_selection(self):
        doc = self.app.doc.model

        # ... select_path() ust be queued with gobject.idle_add to avoid
        # glitches in the update after dragging the current row downwards.
        selected_path = (len(doc.layers) - (doc.layer_idx + 1), )
        self.treeview.get_selection().select_path(selected_path)
        self.treeview.scroll_to_cell(selected_path)

        ## Reflect position of current layer in the list
        sel_is_top = sel_is_bottom = False
        sel_is_bottom = doc.layer_idx == 0
        sel_is_top = doc.layer_idx == len(doc.layers)-1
        self.move_up_button.set_sensitive(not sel_is_top)
        self.move_down_button.set_sensitive(not sel_is_bottom)
        self.merge_down_button.set_sensitive(not sel_is_bottom)


    def treeview_cursor_changed_cb(self, treeview, *data):
        if self.is_updating:
            return
        store, t_iter = treeview.get_selection().get_selected()
        if t_iter is None:
            return
        layer = store.get_value(t_iter, 0)
        doc = self.app.doc.model
        if doc.get_current_layer() != layer:
            idx = doc.layers.index(layer)
            doc.select_layer(idx)


    def treeview_button_press_cb(self, treeview, event):
        x, y = int(event.x), int(event.y)
        bw_x, bw_y = treeview.convert_widget_to_bin_window_coords(x, y)
        path_info = treeview.get_path_at_pos(bw_x, bw_y)
        if path_info is None:
            return False
        clicked_path, clicked_col, cell_x, cell_y = path_info
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
                new_name = dialogs.ask_for_name(self, _("Name"), layer.name)
                if new_name:
                    layer.name = new_name
                    self.treeview.queue_draw()
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


    def move_layer(self, widget, action):
        doc = self.app.doc.model
        current_layer_pos = doc.layer_idx
        if action == 'up':
            new_layer_pos = current_layer_pos + 1
        elif action == 'down':
            new_layer_pos = current_layer_pos - 1
        else:
            return
        if new_layer_pos < len(doc.layers) and new_layer_pos >= 0:
            doc.move_layer(current_layer_pos, new_layer_pos, select_new=True)


    def merge_layer_down(self, widget):
        self.app.doc.model.merge_layer_down()


    def on_layer_add(self, button):
        doc = self.app.doc.model
        doc.add_layer(after=doc.get_current_layer())


    def on_layer_del(self, button):
        doc = self.app.doc.model
        doc.remove_layer(layer=doc.get_current_layer())


    def layer_name_datafunc(self, column, renderer, model, tree_iter):
        layer = model.get_value(tree_iter, 0)
        path = model.get_path(tree_iter)
        num_layers = len(model)
        layer_num = num_layers - path[0]
        name = layer.name
        attrs = pango.AttrList()
        if not name:
            attrs.change(pango.AttrScale(pango.SCALE_SMALL, 0, -1))
            attrs.change(pango.AttrStyle(pango.STYLE_ITALIC, 0, -1))
            name = _("Layer %d" % layer_num)
        renderer.set_property("attributes", attrs)
        renderer.set_property("text", name)


    def layer_visible_datafunc(self, column, renderer, model, tree_iter):
        layer = model.get_value(tree_iter, 0)
        if layer.visible:
            pixbuf = self.app.pixmaps.eye_open
        else:
            pixbuf = self.app.pixmaps.eye_closed
        renderer.set_property("pixbuf", pixbuf)


    def layer_locked_datafunc(self, column, renderer, model, tree_iter):
        layer = model.get_value(tree_iter, 0)
        if layer.locked:
            pixbuf = self.app.pixmaps.lock_closed
        else:
            pixbuf = self.app.pixmaps.lock_open
        renderer.set_property("pixbuf", pixbuf)

