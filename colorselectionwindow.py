"select color window"
import gtk
import colorsys
from time import time

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
            # second press: pick color and remove the window
            self.pick_color_at_pointer()
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


class AlternativeColorSelectorWindow(gtk.Window):
    def __init__(self, colorselectionwindow):
        start = time()
        gtk.Window.__init__(self)
        self.colorselectionwindow = colorselectionwindow
        self.app = colorselectionwindow.app
        self.add_accel_group(self.app.accel_group)

        self.set_title('Similar Color')
        self.connect('delete-event', self.app.hide_window_cb)

        self.image = image = gtk.Image()
        #image.set_from_pixbuf(self.app.brush.get_colorselection_pixbuf())
        vbox = gtk.VBox()
        self.add(vbox)

        vbox.pack_start(image)

        #label = gtk.Label("press any key to select, move mouse outside to cancel")
        #label.set_alignment(0.5, 0.5)
        #vbox.pack_start(label)

        t = time(); print t - start; start = t
        self.image.set_from_pixbuf(self.app.brush.get_colorselection_pixbuf())
        t = time(); print t - start; start = t


	self.set_events(gtk.gdk.BUTTON_PRESS_MASK |
                        gtk.gdk.BUTTON_RELEASE_MASK |
                        gtk.gdk.ENTER_NOTIFY |
                        gtk.gdk.LEAVE_NOTIFY
                        )
        self.connect("enter-notify-event", self.enter_notify_cb)
        self.connect("leave-notify-event", self.leave_notify_cb)
        self.connect("button-release-event", self.button_release_cb)
        self.connect("button-press-event", self.button_press_cb)

        self.first_enter_time = None
        self.button_pressed = False

        # window manager stuff
        #self.set_transient_for(self.application.??)
        self.set_decorated(False)
        self.set_resizable(False)
        #self.set_gravity(gtk.gdk.GRAVITY_CENTER)

        # place window center at pointer
        screen_trash, x_root, y_root, modifiermask_trash = self.get_display().get_pointer()
        width, height = 256, 256
        # carefully avoid drawing before the window is placed correctly :)
        self.move(x_root - width/2, y_root - height/2)
        self.show()
        self.move(x_root - width/2, y_root - height/2)
        self.show_all()
        self.move(x_root - width/2, y_root - height/2)
        #self.present()
        t = time(); print t - start; start = t
        
    def button_press_cb(self, widget, event):
        if event.button == 1:
            self.colorselectionwindow.pick_color_at_pointer()
        self.button_pressed = True

    def remove_cleanly(self):
        self.colorselectionwindow.alternative = None
        self.destroy()

    def button_release_cb(self, widget, event):
        if self.button_pressed:
            self.remove_cleanly()

    def enter_notify_cb(self, widget, event):
        if not self.first_enter_time:
            self.first_enter_time = event.time

    def leave_notify_cb(self, widget, event):
        # when creating the window, we sometimes get leave&enter notifications
        # without evident reason; block them out.
        if not self.first_enter_time: return
        if self.first_enter_time - event.time < 200: return
        self.remove_cleanly()
