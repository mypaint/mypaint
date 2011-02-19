import gtk
gdk = gtk.gdk

from os.path import basename
import tempfile
from cStringIO import StringIO
from gettext import gettext as _
from math import asin, pi
from lib import command
import dialogs
import windowing

DRAG_LAYER_INDEX = 100
DRAG_LAYER_PNG = 101
DRAG_LAYER_URI = 102
LAYER_INDEX_MIME = "application/x-mypaint-layer-index"

def stock_button(stock_id):
    b = gtk.Button()
    img = gtk.Image()
    img.set_from_stock(stock_id, gtk.ICON_SIZE_MENU)
    b.add(img)
    return b

class PixbufToggleButton (gtk.ToggleButton):

    def __init__(self, app, active, tooltip, on_toggle, active_pixbuf, inactive_pixbuf):
        gtk.ToggleButton.__init__(self)
        self.app = app
        self.active_pixbuf = active_pixbuf
        self.inactive_pixbuf = inactive_pixbuf
        self.set_active(active)
        self.image = gtk.Image()
        self.set_border_width(0)
        self.set_size_request(24, 24)
        self.add(self.image)
        self.on_toggle = on_toggle
        self.update_image()
        self.connect('toggled', self.on_toggled)
        self.set_relief(gtk.RELIEF_NONE)
        self.set_tooltip_text(tooltip)
        self.set_focus_on_click(False)
        self.set_property("can-focus", False)
        sty = self.get_modifier_style()
        sty.xthickness = 0
        sty.ythickness = 0
        self.modify_style(sty)

    def update_image(self):
        if self.get_active():
            pixbuf = self.active_pixbuf
        else:
            pixbuf = self.inactive_pixbuf
        self.image.set_from_pixbuf(pixbuf)

    def on_toggled(self, b):
        self.update_image()
        self.on_toggle(b)

def small_pack(box, widget):
    b = box()
    b.pack_start(widget, expand=False)
    return b

class LayerWidget(gtk.EventBox):
    def __init__(self, parent, layer=None):
        gtk.EventBox.__init__(self)
        self.set_border_width(2)
        self.add_events( gdk.BUTTON_PRESS_MASK | gdk.BUTTON_RELEASE_MASK ) 
        self.connect("button-press-event", self.on_button_press)
        self.connect("button-release-event", self.on_button_release)
        self.button_pressed = False
        self.selected = False
        self.layer = layer
        self.list = parent
        self.app = parent.app

        # Widgets
        self.layer_name = gtk.Label()
        self.hidden_button = PixbufToggleButton(self.app,
            False, _('Layer visibility'), self.on_hidden_toggled, 
            self.app.pixmaps.eye_closed, self.app.pixmaps.eye_open)
        locked = self.layer and self.layer.locked
        self.lock_button = PixbufToggleButton(self.app,
            locked, _('Layer lock'), self.on_lock_toggled, 
            self.app.pixmaps.lock_closed, self.app.pixmaps.lock_open)

        # Pack and add to self
        self.main_hbox = gtk.HBox()
        self.main_hbox.pack_start(self.hidden_button, expand=False)
        self.main_hbox.pack_start(self.lock_button, expand=False)
        self.main_hbox.pack_start(self.layer_name)
        self.add(self.main_hbox)

        # Drag/drop for moving layers
        # FIXME: Broken, the callbacks are not being called
#         self.connect('drag_data_received', self.drag_data)
#         self.connect('drag_drop', self.drag_drop)
#         self.connect('drag_data_get', self.drag_get)
#         self.drag_dest_set(gtk.DEST_DEFAULT_MOTION | gtk.DEST_DEFAULT_HIGHLIGHT | gtk.DEST_DEFAULT_DROP,
#                  [(LAYER_INDEX_MIME,0,DRAG_LAYER_INDEX),
#                   ("text/plain", 0, DRAG_LAYER_URI)],
#                  gdk.ACTION_MOVE)
#         self.drag_source_set(gdk.BUTTON1_MASK,
#                 [(LAYER_INDEX_MIME,0,DRAG_LAYER_INDEX),
#                  ("image/png", 0, DRAG_LAYER_PNG),
#                  ("text/uri-list", 0, DRAG_LAYER_URI)],
#                 gdk.ACTION_MOVE)

        self.clicked = 0

        self.set_layer(layer)

    def on_button_press(self, widget, event):
        if event.type == gdk.BUTTON_PRESS:
            self.clicked = 1
            return
        elif event.type == gdk._2BUTTON_PRESS:
            self.clicked = 2
            return
        self.clicked = 0

    def on_button_release(self, widget, event):
        if self.clicked == 1:
            if self is not self.list.selected:
                self.list.selected = self
        elif self.clicked == 2:
            if self is self.list.selected:
                self.change_name()
        self.clicked = 0

    def drag_data(self, widget, context, x,y, selection, targetType, time):
        if targetType==DRAG_LAYER_INDEX:
            idx = int(selection.data)
            self.list.swap(self, idx)
        elif targetType==DRAG_LAYER_URI:
            filename = selection.data.strip().replace('file://','')
            doc = self.app.doc.model
            idx = self.list.widgets.index(self)
            try:
                pixbuf = gdk.pixbuf_new_from_file(filename)
            except Exception, e:
                print e
                return
            else:
                doc.add_layer(insert_idx=idx)
                doc.load_layer_from_pixbuf(pixbuf)
                doc.layer.name = basename(filename)
                self.list.update()

    def drag_drop(self, widget, context, x,y, time):
        return True

    def drag_get(self, widget, context, selection, targetType, time):
        tmp = StringIO()
        def stringIO_saver(buf):
            global tmp
            tmp.write(buf)

        if targetType==DRAG_LAYER_INDEX:
            idx = self.list.widgets.index(self)
            selection.set(selection.target, 8, str(idx))
        elif targetType==DRAG_LAYER_PNG:
            pixbuf = self.layer.surface.render_as_pixbuf()
            pixbuf.save_to_callback(stringIO_saver, 'png', {'alpha':'True'})
            s = tmp.getvalue()
            selection.set(selection.target, 8, s)
        elif targetType==DRAG_LAYER_URI:
            pixbuf = self.layer.surface.render_as_pixbuf()
            tmpfile = tempfile.mktemp(prefix='mypaint', suffix='.png')
            pixbuf.save(tmpfile, 'png', {'alpha': 'True'})
            selection.set(selection.target, 8, "file://"+tmpfile+"\n")

    def set_layer(self, layer):
        if not layer:
            return
        self.callbacks_active = False
        self.hidden_button.set_active(not layer.visible)
        self.lock_button.set_active(layer.locked)
        layer_text = layer.name
        if not layer_text:
            if self is self.list.selected:
                layer_text = _('<small>Double click to enter name</small>')
            else:
                layer_text = _('<small>Click to select this layer</small>')
        self.layer_name.set_markup(layer_text)
        self.callbacks_active = True

    def change_name(self, *ignore):
        layer_name = dialogs.ask_for_name(self, _("Name"), self.layer.name)
        if layer_name:
            self.layer.name = layer_name
            self.layer_name.set_text(layer_name)

    def on_hidden_toggled(self, checkbox):
        if not self.callbacks_active:
            return
        visible = not self.layer.visible
        self.app.doc.model.set_layer_visibility(visible, self.layer)

    def on_lock_toggled(self, checkbox):
        if not self.callbacks_active:
            return
        locked = not self.layer.locked
        self.app.doc.model.set_layer_locked(locked, self.layer)

    def set_selected(self):
        style = self.get_style()
        color = style.bg[gtk.STATE_SELECTED]
        def mark(w):
            w.modify_bg(gtk.STATE_NORMAL, color)
            if isinstance(w, gtk.Box):
                w.foreach(mark)
        mark(self)
        self.main_hbox.foreach(mark)

    def set_unselected(self):
        def unmark(w):
            w.modify_bg(gtk.STATE_NORMAL, None)
            if isinstance(w, gtk.Box):
                w.foreach(unmark)
        unmark(self)
        self.main_hbox.foreach(unmark)

class LayersList(gtk.VBox):
    def __init__(self, app, layers=[]):
        gtk.VBox.__init__(self)
        self.layers = layers
        self.widgets = []
        self.selected = None
        self.app = app
        self.disable_selected_callback = False
        self.pack_layers()

    def set_layers(self, layers):
        def find_widget(l):
            for w in self.widgets:
                if w.layer is l:
                    return w
            return None
        self.layers = layers
        ws = []
        for l in layers:
            w = find_widget(l)
            if w:
                ws.append(w)
            else:
                w = LayerWidget(self, l)
                ws.append(w)
        self.widgets = ws
        def remove_or_destroy(w):
            self.remove(w)
            if w not in self.widgets:
                w.destroy() # fix for memory leak that would keep layerdata alive
        self.foreach(remove_or_destroy)
        for w in self.widgets:
            self.pack_end(w,expand=False)

        # Prevent callback loop
        self.disable_selected_callback = True
        idx = self.app.doc.model.layer_idx
        self.selected = self.widgets[idx]
        self.disable_selected_callback = False
        self.update()

        self.show_all()

    def repack_layers(self):
        self.foreach(self.remove)
        self.widgets = []
        self.pack_layers()

    def pack_layers(self):
        for layer in self.layers:
            widget = LayerWidget(self, layer)
            self.widgets.append(widget)
            self.pack_end(widget, expand=False)
        self.show_all()

    def update(self):
        for widget in self.widgets:
            widget.set_layer(widget.layer)

    def set_selected(self, widget):
        self._selected = widget
        if widget:
            for item in self.widgets:
                if item is widget:
                    item.set_selected()
                else:
                    item.set_unselected()
        if widget in self.widgets and not self.disable_selected_callback:
            idx = self.widgets.index(widget)
            doc = self.app.doc
            doc.model.select_layer(idx)
            doc.layerblink_state.activate()

    def get_selected(self):
        return self._selected

    selected = property(get_selected, set_selected)

    def swap(self, widget, another_widget_idx):
        target_idx = self.widgets.index(widget)
        doc = self.app.doc.model
        doc.move_layer(another_widget_idx, target_idx)

class ToolWidget (gtk.VBox):

    tool_widget_title = _("Layers")

    def __init__(self, app):
        gtk.VBox.__init__(self)
        self.app = app
        self.set_size_request(200, 150)
        self.callbacks_active = True # Used to prevent callback loops

        # Widgets
        # Layer list
        layers_scroll = gtk.ScrolledWindow()
        layers_scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        self.layers_list = LayersList(app)
        layers_scroll.add_with_viewport(self.layers_list)

        # Common controls
        adj = gtk.Adjustment(lower=0, upper=100, step_incr=1, page_incr=10)
        self.opacity_scale = gtk.HScale(adj)
        self.opacity_scale.set_value_pos(gtk.POS_RIGHT)
        opacity_lbl = gtk.Label(_('Opacity:'))
        opacity_hbox = gtk.HBox()
        opacity_hbox.pack_start(opacity_lbl, expand=False)
        opacity_hbox.pack_start(self.opacity_scale, expand=True)

        add_button = stock_button(gtk.STOCK_ADD)
        move_up_button = stock_button(gtk.STOCK_GO_UP)
        move_down_button = stock_button(gtk.STOCK_GO_DOWN)
        merge_down_button = stock_button(gtk.STOCK_DND_MULTIPLE)  # XXX need a better one
        del_button = stock_button(gtk.STOCK_DELETE)

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
        self.pack_start(layers_scroll)
        self.pack_start(buttons_hbox, expand=False)
        self.pack_start(opacity_hbox, expand=False)

        # Updates
        doc = app.doc.model
        doc.doc_observers.append(self.update)
        self.opacity_scale.connect('value-changed', self.on_opacity_changed)

        self.update(doc)

    def update(self, doc):
        if not self.callbacks_active:
            return

        # Update the layer list
        self.layers_list.set_layers(doc.layers)

        # Update the common widgets
        self.callbacks_active = False
        self.opacity_scale.set_value(doc.get_current_layer().opacity*100)
        self.callbacks_active = True

    def on_opacity_changed(self, *ignore):
        if not self.callbacks_active:
            return

        self.callbacks_active = False
        doc = self.app.doc.model
        doc.set_layer_opacity(self.opacity_scale.get_value()/100.0)
        self.callbacks_active = True

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
            # TODO: avoid calling two actions as this is actually one operation
            doc.move_layer(current_layer_pos, new_layer_pos)
            doc.select_layer(new_layer_pos)

    def merge_layer_down(self, widget):
        self.app.doc.model.merge_layer_down()

    def on_layer_add(self, button):
        doc = self.app.doc.model
        doc.add_layer(after=doc.get_current_layer())

    def on_layer_del(self, button):
        doc = self.app.doc.model
        doc.remove_layer(layer=doc.get_current_layer())
