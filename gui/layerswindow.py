import gtk
gdk = gtk.gdk

from os.path import basename
import tempfile
from cStringIO import StringIO
from gettext import gettext as _
from math import asin, pi
from lib import command

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


alpha = asin(0.8)
class EyeOnly(gtk.DrawingArea):
    def __init__(self, size=20):
        gtk.DrawingArea.__init__(self)
        self.set_size_request(size,size)
        self.set_events(gdk.BUTTON_PRESS_MASK | gdk.BUTTON_RELEASE_MASK)
        self.connect('button-press-event', self.button_press)
        self.connect('button-release-event', self.on_button_release)
        self.connect('expose-event', self.draw)
        self.set_tooltip_text(_('Layer visibility'))
        self.size = size
        self.active = False
        self.button_pressed = False

    def set_active(self,v):
        self.active = v
        self.queue_draw()

    def get_active(self):
        return self.active

    def on_toggle(self, w):
        pass

    def on_button_press(self):
        pass

    def button_press(self, w, event):
        self.button_pressed = True
        self.on_button_press()

    def on_button_release(self, w, event):
        if not self.button_pressed:
            return
        self.set_active(not self.active)
        self.on_toggle(w)

    def draw(self,widget, event):
        cr = self.window.cairo_create()
        r = self.get_allocation()
        w,h = r.width, r.height
        r = w/4.0
        R = 0.625*w
        x0 = w/2.0
        y0 = h/2.0
        cr.set_line_width(1.4)
        if self.active:
            cr.set_source_rgb(0.0,0.0,0.0)
            cr.arc(x0, y0 - R + r, R, pi/2-alpha, pi/2+alpha)
            cr.stroke()
            cr.arc(x0, y0 + R - r, R, -pi/2-alpha, -pi/2+alpha)
            cr.stroke()
            cr.arc(x0, y0, r, 0.0, 2*pi)
            cr.fill()
        else:
            cr.set_source_rgb(0.4,0.4,0.4)
            cr.arc(x0, y0 - R + r, R, pi/2-alpha, pi/2+alpha)
            cr.stroke()
            cr.set_source_rgb(0.6,0.6,0.6)
            cr.arc(x0, y0 + R - r, R, -pi/2-alpha, -pi/2+alpha)
            cr.stroke()

class Eye(gtk.ToggleButton):
    def __init__(self):
        gtk.ToggleButton.__init__(self)
        self.set_active(True)
        self.eye = EyeOnly()
        self.eye.set_active(True)
        self.add(self.eye)
        self.connect('toggled', self.toggled)

    def toggled(self,b):
        self.eye.set_active(not self.eye.get_active())
        self.on_toggle(b)

    def on_toggle(self,w):
        pass

def small_pack(box, widget):
    b = box()
    b.pack_start(widget, expand=False)
    return b

class LayerWidget(gtk.EventBox):
    def __init__(self,parent,layer=None):
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

#         vbox = gtk.VBox()
#         vbox.pack_start(gtk.HSeparator(), expand=False)

        add_button = stock_button(gtk.STOCK_ADD)
        add_button.connect('clicked', self.on_layer_add)
        del_button = stock_button(gtk.STOCK_DELETE)
        del_button.connect('clicked', self.on_layer_del)

        # Widgets
        self.visibility_button = Eye()
        self.visibility_button.on_toggle = self.on_visibility_toggled

        # Clickable label with layer name
        self.layer_name = gtk.Label("LAYER!!!")
        layer_name_box = gtk.EventBox()
        layer_name_box.add(self.layer_name)

        # Pack and add to self
        self.main_hbox = gtk.HBox()
        self.main_hbox.pack_start(self.visibility_button, expand=False)
        self.main_hbox.pack_start(layer_name_box)
        self.main_hbox.pack_start(add_button, expand=False)
        self.main_hbox.pack_start(del_button, expand=False)
        self.add(self.main_hbox)

        # Drag/drop for moving layers
        self.connect('drag_data_received',self.drag_data)
        self.connect('drag_drop', self.drag_drop)
        self.connect('drag_data_get', self.drag_get)
        self.drag_dest_set(gtk.DEST_DEFAULT_MOTION | gtk.DEST_DEFAULT_HIGHLIGHT | gtk.DEST_DEFAULT_DROP,
                 [(LAYER_INDEX_MIME,0,DRAG_LAYER_INDEX),
                  ("text/plain", 0, DRAG_LAYER_URI)],
                 gdk.ACTION_MOVE)
        self.drag_source_set(gdk.BUTTON1_MASK,
                [(LAYER_INDEX_MIME,0,DRAG_LAYER_INDEX),
                 ("image/png", 0, DRAG_LAYER_PNG),
                 ("text/uri-list", 0, DRAG_LAYER_URI)],
                gdk.ACTION_MOVE)

        self.clicked = 0
        self.button_pressed = False
        self.time_pressed = 0

        self.set_layer(layer)

    def on_button_press(self, widget, event):
        self.button_pressed = True

    def on_button_release(self, widget, event):
        if self.button_pressed:
            self.list.selected = self
            self.button_pressed = False
        if self.clicked == 0:
            self.time_pressed = event.time
            self.clicked = 1
        elif self.clicked == 1 and event.time - self.time_pressed < 700:
            self.clicked = 2
        else:
            self.clicked = 0
#         if self.clicked == 2:
#             self.clicked = 0
#             self.name_entry.show_edit()

    def drag_data(self, widget, context, x,y, selection, targetType, time):
        if targetType==DRAG_LAYER_INDEX:
            idx = int(selection.data)
            self.list.swap(self, idx)
        elif targetType==DRAG_LAYER_URI:
            filename = selection.data.strip().replace('file://','')
            doc = self.app.drawWindow.tdw.doc
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
            pixbuf.save_to_callback(stringIO_saver, 'png', {'aplha':'True'})
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
        self.visibility_button.set_active(layer.visible)
#         self.layer_name.set_text(layer.name)
        self.callbacks_active = True

    def on_layer_add(self,button):
        doc = self.app.drawWindow.tdw.doc
        doc.add_layer(after=self.layer)

    def on_layer_del(self,button):
        doc = self.app.drawWindow.tdw.doc
        doc.remove_layer(layer=self.layer)

#     def on_name_changed(self,entry):
#         if not self.callbacks_active:
#             return
#         text =  entry.get_text()
#         if text:
#             entry.to_show = True
#         self.layer.name = text

    def on_opacity_changed(self,scale):
        if not self.callbacks_active:
            return
        doc = self.app.drawWindow.tdw.doc
        cmd = doc.get_last_command()
        if isinstance(cmd, command.SetLayerOpacity) and cmd.layer is self.layer:
            doc.undo()
        opacity = scale.get_value()/100.0
        doc.do(command.SetLayerOpacity(doc, opacity, self.layer))

    def on_visibility_toggled(self, checkbox):
        if not self.callbacks_active:
            return
        self.layer.visible = not self.layer.visible
        self.app.drawWindow.tdw.queue_draw()

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
    def __init__(self,app,layers=[]):
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
        self.foreach(self.remove)
        for w in self.widgets:
            self.pack_end(w,expand=False)

        # Prevent callback loop
        self.disable_selected_callback = True
        idx = self.app.drawWindow.doc.layer_idx
        self.selected = self.widgets[idx]
        self.disable_selected_callback = False

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
            dw = self.app.drawWindow
            dw.doc.select_layer(idx)
            dw.layerblink_state.activate()

    def get_selected(self):
        return self._selected

    selected = property(get_selected, set_selected)

    def swap(self, widget, another_widget_idx):
        target_idx = self.widgets.index(widget)
        doc = self.app.drawWindow.tdw.doc
        doc.move_layer(another_widget_idx, target_idx)

class Window(gtk.Window):
    def __init__(self,app):
        gtk.Window.__init__(self)
        self.set_title(_("Layers"))
        self.set_role("Layers")
        self.app = app
        self.connect('delete-event', self.app.hide_window_cb)
        self.app.kbm.add_window(self)
        doc = app.drawWindow.tdw.doc
        doc.doc_observers.append(self.update)
        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        self.layers_list = LayersList(app)
        scroll.add_with_viewport(self.layers_list)
        self.add(scroll)
        self.set_size_request(300, 300)
        self.update(doc)

    def update(self, doc, action='edit', idx=None):
        self.layers_list.set_layers(doc.layers)


