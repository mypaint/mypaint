"select color window (GTK and an own window)"
import gtk, gobject
import colorsys

# GTK selector
class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.add_accel_group(self.app.accel_group)

        self.set_title('Color')
        self.connect('delete-event', self.app.hide_window_cb)

        vbox = gtk.VBox()
        self.add(vbox)

        self.cs = gtk.ColorSelection()
        self.cs.connect('color-changed', self.color_changed_cb)
        vbox.pack_start(self.cs)

        self.alternative = None

    def show_change_color_window(self):
        if self.alternative:
            # second press: <strike>pick color</strike> cancel and remove the window
            #self.pick_color_at_pointer()
            self.alternative.remove_cleanly()
        else:
            self.alternative = AlternativeColorSelectorWindow(self)

    def color_changed_cb(self, cs):
        self.app.brush.set_color(self.get_color_rgb())

    def update(self):
        self.set_color_rgb(self.app.brush.get_color())

    def pick_color_at_pointer(self):
        # grab screen color at cursor
        # inspired by gtkcolorsel.c function grab_color_at_mouse()
        screen = self.get_screen()
        colormap = screen.get_system_colormap()
        root = screen.get_root_window()
        display = self.get_display()
        screen_trash, x_root, y_root, modifiermask_trash = display.get_pointer()
        image = root.get_image(x_root, y_root, 1, 1)
        pixel = image.get_pixel(0, 0)
        color = colormap.query_color(pixel)
        #print color.red, color.green, color.blue
        self.cs.set_current_color(color)
        
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


# own color selector
# see also get_colorselection_pixbuf in gtkmybrush.c
class AlternativeColorSelectorWindow(gtk.Window):
    def __init__(self, colorselectionwindow):
        gtk.Window.__init__(self, gtk.WINDOW_POPUP)
        self.set_gravity(gtk.gdk.GRAVITY_CENTER)
        self.set_position(gtk.WIN_POS_MOUSE)
        
        self.colorselectionwindow = colorselectionwindow
        self.app = colorselectionwindow.app
        self.add_accel_group(self.app.accel_group)

        #self.set_title('Color')
        self.connect('delete-event', self.app.hide_window_cb)

        self.image = image = gtk.Image()
        self.add(image)

        self.image.set_from_pixbuf(self.app.brush.get_colorselection_pixbuf())

	self.set_events(gtk.gdk.BUTTON_PRESS_MASK |
                        gtk.gdk.BUTTON_RELEASE_MASK |
                        gtk.gdk.ENTER_NOTIFY |
                        gtk.gdk.LEAVE_NOTIFY
                        )
        self.connect("enter-notify-event", self.enter_notify_cb)
        self.connect("leave-notify-event", self.leave_notify_cb)
        self.connect("button-release-event", self.button_release_cb)
        self.connect("button-press-event", self.button_press_cb)

        self.destroy_timer = None
        self.button_pressed = False

        self.show_all()

        self.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.CROSSHAIR))
        
    def button_press_cb(self, widget, event):
        if event.button == 1:
            self.colorselectionwindow.pick_color_at_pointer()
        self.button_pressed = True

    def remove_cleanly(self):
        self.colorselectionwindow.alternative = None
        if self.destroy_timer is not None:
            gobject.source_remove(self.destroy_timer)
            self.destroy_timer = None
        self.destroy()

    def button_release_cb(self, widget, event):
        if self.button_pressed:
            self.remove_cleanly()

    def enter_notify_cb(self, widget, event):
        if self.destroy_timer is not None:
            gobject.source_remove(self.destroy_timer)
            self.destroy_timer = None

    def leave_notify_cb(self, widget, event):
        # allow to leave the window for a short time
        if self.destroy_timer is not None:
            gobject.source_remove(self.destroy_timer)
        self.destroy_timer = gobject.timeout_add(80, self.remove_cleanly)
