"extending the C myDrawWidget a bit, eg with infinite canvas"
import gtk, gc
from mydrawwidget import MyDrawWidget
from helpers import Rect

class InfiniteMyDrawWidget(MyDrawWidget):
    def __init__(self):
        MyDrawWidget.__init__(self)
        self.init_canvas()
        MyDrawWidget.clear(self)
        self.connect("size-allocate", self.size_allocate_event_cb)
        self.connect("dragging_finished", self.dragging_finished_cb)

    def init_canvas(self):
        self.canvas_w = 1
        self.canvas_h = 1
        self.viewport_x = 0.0
        self.viewport_y = 0.0

    def clear(self):
        self.discard_and_resize(1, 1)
        self.init_canvas()
        MyDrawWidget.clear(self)
        if self.window: 
            self.resize_if_needed()

    def allow_dragging(self, allow=True):
        if allow:
            MyDrawWidget.allow_dragging(self, 1)
        else:
            MyDrawWidget.allow_dragging(self, 0)

    def load(self, filename):
        pixbuf = gtk.gdk.pixbuf_new_from_file(filename)
        if pixbuf.get_has_alpha():
            print 'Loaded file has an alpha channel. Rendering it on white instead.'
            print 'NOT IMPLEMENTED'
            return
            TODO
            w, h = pixbuf.get_width(), pixbuf.get_height()
            new_pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, w, h)
            new_pixbuf.fill(0xffffffff) # white

            pixbuf = new_pixbuf
            print 'got pixbuf from file.'
        self.canvas_w = pixbuf.get_width()
        self.canvas_h = pixbuf.get_height()
        self.viewport_x = 0.0
        self.viewport_y = 0.0
        # will need a bigger canvas size than that
        self.resize_if_needed(old_pixbuf = pixbuf)

    def save(self, filename):
        pixbuf = self.get_nonwhite_as_pixbuf()
        pixbuf.save(filename, 'png')

    def dragging_finished_cb(self, widget):
        self.viewport_x = self.get_viewport_x()
        self.viewport_y = self.get_viewport_y()
        self.resize_if_needed()
        
    def size_allocate_event_cb(self, widget, allocation):
        size = (allocation.width, allocation.height)
        self.resize_if_needed(size=size)

    def zoom(self, new_zoom):
        vp_w, vp_h = self.window.get_size()
        old_zoom = self.get_zoom()

        center_x = self.viewport_x + vp_w / old_zoom / 2.0
        center_y = self.viewport_y + vp_h / old_zoom / 2.0

        self.set_zoom(new_zoom)

        self.viewport_x = center_x - vp_w / new_zoom / 2.0
        self.viewport_y = center_y - vp_h / new_zoom / 2.0

        self.set_viewport(self.viewport_x, self.viewport_y)
        self.resize_if_needed()

    def scroll(self, dx, dy):
        zoom = self.get_zoom()
        self.viewport_x += dx/zoom
        self.viewport_y += dy/zoom
        self.set_viewport(self.viewport_x, self.viewport_y)
        self.resize_if_needed()

    def resize_if_needed(self, old_pixbuf = None, size = None):
        vp_w, vp_h = size or self.window.get_size()
        zoom = self.get_zoom()
        vp_w = int(vp_w/zoom)
        vp_h = int(vp_h/zoom)

        # calculation is done in canvas coordinates
        oldCanvas = Rect(0, 0, self.canvas_w, self.canvas_h)
        viewport  = Rect(int(self.viewport_x+0.5), int(self.viewport_y+0.5), vp_w, vp_h)
        
        # add space; needed to draw into the non-visible part at the border
        expanded = viewport.copy()
        border = max(30, min(vp_w/4, vp_h/4)) # quite arbitrary
        expanded.expand(border)

        if (expanded in oldCanvas) and (not old_pixbuf):
            # canvas is big enough already
            return 

        # need a new, bigger canvas
        if old_pixbuf is None:
            old_pixbuf = self.get_as_pixbuf()
        # let's see what size we need it.
        expanded.expand(1*border) # expand even further to avoid too frequent resizing
        # now, combine the (possibly already painted) rect with the (visible) viewport
        newCanvas = oldCanvas.copy()
        newCanvas.expandToIncludeRect(expanded)
        self.canvas_w = newCanvas.w
        self.canvas_h = newCanvas.h
        self.discard_and_resize(self.canvas_w, self.canvas_h)
        translate_x = oldCanvas.x - newCanvas.x
        translate_y = oldCanvas.y - newCanvas.y
        assert translate_x >= 0 and translate_y >= 0 # because new canvas must include old one

        # paste old image back
        w, h = newCanvas.w, newCanvas.h
        print 'Resizing canvas to %dx%d = %.3fMB' % (w, h, w*h*3/1024.0/1024.0)
        new_pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, w, h)
        new_pixbuf.fill(0xffffffff) # white
        old_pixbuf.copy_area(src_x=0, src_y=0,
                             width=old_pixbuf.get_width(), height=old_pixbuf.get_height(),
                             dest_pixbuf=new_pixbuf,
                             dest_x=translate_x, dest_y=translate_y)
        self.set_from_pixbuf(new_pixbuf)

        self.viewport_x += translate_x
        self.viewport_y += translate_y
        self.set_viewport(self.viewport_x, self.viewport_y)

        # free that huge memory again
        del new_pixbuf, old_pixbuf
        gc.collect()
