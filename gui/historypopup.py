import math
import gtk
gdk = gtk.gdk

import random, colorsys
import numpy

"""
Worklist (planning/prototyping)
- short keypress switches between two colors (==> still possible to do black/white the old way)
- long keypress allows to move the mouse over another color
  - the mouseover color is selected when the key is released
- (unsure) pressing the key twice (without painting in-between) cycles through the colors
  - the problem is that you don't see the colors
  ==> different concept:
    - short keypress opens the color ring with cursor centered on hole (ring stays open)
    - the ring disappears as soon as you touch it (you can just continue painting)
    - pressing the key again cycles the ring
- recent colors should be saved with painting

Observation:
- it seems quite unnatural that you can have a /shorter/ popup duration by pressing the key /longer/
  ==> you rather want a /minimum/ duration
"""

num_colors = 6

class HistoryPopup(gtk.Window):
    outside_popup_timeout = 0
    def __init__(self, app, doc):
        gtk.Window.__init__(self, gtk.WINDOW_POPUP)
        self.set_gravity(gdk.GRAVITY_CENTER)
        self.set_position(gtk.WIN_POS_MOUSE)

        self.app = app
        self.app.kbm.add_window(self)

	self.set_events(gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.ENTER_NOTIFY |
                        gdk.LEAVE_NOTIFY
                        )
        self.connect("button-release-event", self.button_release_cb)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("expose_event", self.expose_cb)

        self.set_size_request(300, 50)

        self.selection = None

        self.colorhist = [(random.random(), random.random(), random.random()) for i in range(num_colors)]

        self.doc = doc
        doc.stroke_observers.append(self.stroke_finished_cb)

    def enter(self):
        def hsv_equal(a, b):
            # hack required because we somewhere have an rgb<-->hsv conversion roundtrip
            a_ = numpy.array(colorsys.hsv_to_rgb(*a))
            b_ = numpy.array(colorsys.hsv_to_rgb(*b))
            return ((a_ - b_)**2).sum() < (3*1.0/256)**2

        # finish pending stroke, if any (causes stroke_finished_cb to get called)
        self.doc.split_stroke()
        if self.selection is None:
            self.selection = num_colors - 1
            color = self.app.brush.get_color_hsv()
            if hsv_equal(self.colorhist[self.selection], color):
                self.selection -= 1
        else:
            self.selection = (self.selection - 1) % num_colors

        self.app.brush.set_color_hsv(self.colorhist[self.selection])
        self.show_all()
        self.window.set_cursor(gdk.Cursor(gdk.CROSSHAIR))
    
    def leave(self, reason):
        self.hide()

    def button_press_cb(self, widget, event):
        pass

    def button_release_cb(self, widget, event):
        pass

    def stroke_finished_cb(self, stroke, brush):
        print 'stroke finished', stroke.total_painting_time, 'seconds'
        self.selection = None
        if not brush.is_eraser():
            color = brush.get_color_hsv()
            if color in self.colorhist:
                self.colorhist.remove(color)
            self.colorhist = (self.colorhist + [color])[-num_colors:]

    def expose_cb(self, widget, event):
        cr = self.window.cairo_create()
        aloc = self.get_allocation()
        #pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, True, 8, size, size)

        # translate to center
        cx = aloc.x + aloc.width / 2
        cy = aloc.y + aloc.height / 2
        r = aloc.height/2.0 - 2.0
        #cr.translate(aloc.width-1 -r-1.0, cy)
        cr.translate(0+r+1.0, cy)

        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.paint()

        for i, c in enumerate(self.colorhist):
            cr.arc(0, 0, r, 0, 2 * math.pi)

            cr.set_source_rgb(*colorsys.hsv_to_rgb(*c))
        
            if i == self.selection:
                cr.fill_preserve()
                cr.set_source_rgb(0, 0, 0)
                cr.stroke()
            else:
                cr.fill()
                #cr.set_source_rgb(0.5, 0, 0)
                #cr.stroke()

            cr.translate(+2*r+2.0, 0)

        #pixmap, mask = pixbuf.render_pixmap_and_mask()
        #self.image.set_from_pixmap(pixmap, mask)
        #self.shape_combine_mask(mask,0,0)

        return True
