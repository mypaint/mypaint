"select color window"
import gtk
import colorsys

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.add_accel_group(self.app.accel_group)

        self.set_title('Color')

        vbox = gtk.VBox()
        self.add(vbox)

        self.cs = gtk.ColorSelection()
        self.cs.connect('color-changed', self.color_changed_cb)
        vbox.pack_start(self.cs)

    def color_changed_cb(self, cs):
        self.app.brush.set_color(self.get_color_rgb())

    def update(self):
        self.set_color_rgb(self.app.brush.get_color())

    def get_color_rgb(self):
        c = self.cs.get_current_color()
        r = float(c.red  ) * 255 / 65535
        g = float(c.green) * 255 / 65535
        b = float(c.blue ) * 255 / 65535
        return [int(r), int(g), int(b)]

    def set_color_rgb(self, rgb):
        r, g, b  = rgb
        c = gtk.gdk.Color(int(r*65535.0/255.0+0.5), int(g*65535.0/255.0+0.5), int(b*65535.0/255.0+0.5))
        self.cs.set_current_color(c)

    def get_color_hsv(self):
        c = self.cs.get_current_color()
        r = float(c.red  ) / 65535
        g = float(c.green) / 65535
        b = float(c.blue ) / 65535
        assert r >= 0.0
        assert g >= 0.0
        assert b >= 0.0
        assert r <= 1.0
        assert g <= 1.0
        assert b <= 1.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return (h, s, v)

    def set_color_hsv(self, hsv):
        h, s, v = hsv
        while h > 1.0: h -= 1.0
        while h < 0.0: h += 1.0
        if s > 1.0: s = 1.0
        if s < 0.0: s = 0.0
        if v > 1.0: v = 1.0
        if v < 0.0: v = 0.0
        r, g, b  = colorsys.hsv_to_rgb(h, s, v)
        c = gtk.gdk.Color(int(r*65535+0.5), int(g*65535+0.5), int(b*65535+0.5))
        self.cs.set_current_color(c)

