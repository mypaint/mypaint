"select color window"
import gtk

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.add_accel_group(self.app.accel_group)

        self.set_title('Color')

        vbox = gtk.VBox()
        self.add(vbox)

        cs = gtk.ColorSelection()
        cs.connect('color-changed', self.color_changed_cb)
        vbox.pack_start(cs)

    def color_changed_cb(self, cs):
        c = cs.get_current_color()
        r = float(c.red  ) * 255 / 65535
        g = float(c.green) * 255 / 65535
        b = float(c.blue ) * 255 / 65535
        #print r, g, b
        self.app.brush.set_color(int(r), int(g), int(b))

