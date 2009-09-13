import math
import gtk, gobject
gdk = gtk.gdk
import cairo

popup_width = 80
popup_height = 80

# TODO: end eraser modus when done with picking (or even before starting to pick)


class ColorPicker(gtk.Window):
    outside_popup_timeout = 0
    def __init__(self, app, doc):
        gtk.Window.__init__(self, gtk.WINDOW_POPUP)
        # TODO: put the mouse position onto the selected color
        self.set_position(gtk.WIN_POS_MOUSE)

        self.app = app
        self.app.kbm.add_window(self)

        self.set_events(gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.ENTER_NOTIFY |
                        gdk.LEAVE_NOTIFY
                        )
        self.connect("expose_event", self.expose_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)

        self.set_size_request(popup_width, popup_height)

        self.doc = doc

        self.idle_handler = None

        # we need this, otherwise we can't prevent painting events
        # from reaching the canvas window
        self.set_extension_events (gdk.EXTENSION_EVENTS_ALL)

    def pick(self):
        size = int(self.app.brush.get_actual_radius() * math.sqrt(math.pi))
        if size < 6:
            size = 6
        self.app.colorSelectionWindow.pick_color_at_pointer(size)

    def enter(self):
        self.pick()

        # popup placement
        x, y = self.get_position()
        self.move(x, y + popup_height)
        self.show_all()

        gdk.pointer_grab(self.window, event_mask=gdk.POINTER_MOTION_MASK)
        self.popup_state.register_mouse_grab(self)
    
    def leave(self, reason):
        gdk.pointer_ungrab()
        if self.idle_handler:
            gobject.source_remove(self.idle_handler)
            self.idle_handler = None
        self.hide()

    def motion_notify_cb(self, widget, event):
        def update():
            self.idle_handler = None
            self.pick()
            self.queue_draw()

        if not self.idle_handler:
            self.idle_handler = gobject.idle_add(update)

    def expose_cb(self, widget, event):
        cr = self.window.cairo_create()

        #cr.set_source_rgb (1.0, 1.0, 1.0)
        cr.set_source_rgba (1.0, 1.0, 1.0, 0.0) # transparent
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        color = self.app.brush.get_color_rgb()

        line_width = 3.0
        distance = 2*line_width
        rect = [0, 0, popup_height, popup_width]
        rect[0] += distance/2.0
        rect[1] += distance/2.0
        rect[2] -= distance
        rect[3] -= distance
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.rectangle(*rect)
        cr.set_source_rgb(*color)
        cr.fill_preserve()
        cr.set_line_width(line_width)
        cr.set_source_rgb(0, 0, 0)
        cr.stroke()
        
        return True
