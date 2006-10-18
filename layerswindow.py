"layer dialog"
# TODO

import gtk
import gobject
gdk = gtk.gdk

# gtk-style-concept: "subscribe" to changes.
# ==> the object with does the composition stores the dirty-status
#     together with the incoming connection.
# a changed-dirty-rect signal is not emitted for each dab
# - but after each event processed (so interpolated dabs are combined)

class DummyLayer:
    def __init__(self, name):
        self.name = name
        self.visible = True
        self.thumb = None
        self.pixbuf = None

    def load(self, filename):
        self.pixbuf = gdk.pixbuf_new_from_file(filename)
        self.thumb = self.pixbuf.scale_simple(64, 64, gdk.INTERP_BILINEAR)

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app

        self.set_title('Layers')
        self.connect('delete-event', self.app.hide_window_cb)

        vbox = gtk.VBox()
        self.add(vbox)

        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        vbox.pack_start(scroll)

        view = self.create_view()
        scroll.add(view)

        hbox = gtk.HBox()
        hbox.set_border_width(8)
        vbox.pack_start(hbox, expand=False, fill=False)

        b = gtk.Button('new')
        #b.connect('clicked', self.add_as_new_cb)
        hbox.pack_start(b, expand=False)

        b = gtk.Button('duplicate')
        hbox.pack_start(b, expand=False)

        b = gtk.Button('up')
        hbox.pack_start(b, expand=False)

        b = gtk.Button('down')
        hbox.pack_start(b, expand=False)

        self.tooltips = gtk.Tooltips()

        self.set_size_request(150, 300)

    def create_view(self):
        model = gtk.TreeStore(gobject.TYPE_PYOBJECT, gobject.TYPE_STRING, gtk.gdk.Pixbuf)

        # ? how do I show thumbs ?

        for i in range(3):
            layer = DummyLayer('Layer%d' % i)
            layer.load('layer%d.png' % i)
            iter = model.insert_before(None, None)
            model.set_value(iter, 0, layer)
            model.set_value(iter, 1, layer.name)
            model.set_value(iter, 2, layer.thumb)

        view = gtk.TreeView(model)

        renderer = gtk.CellRendererPixbuf()
        column = gtk.TreeViewColumn("Thumb", renderer, pixbuf=2)
        view.append_column(column)

        renderer = gtk.CellRendererText()
        renderer.set_property("editable", True)
        column = gtk.TreeViewColumn("Name", renderer, text=1)
        view.append_column(column)

        return view
