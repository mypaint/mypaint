"extending the C myDrawWidget a bit, eg with infinite canvas"
import gtk, gc
from mydrawwidget import MyDrawWidget
from helpers import Rect

class InfiniteMyDrawWidget(MyDrawWidget):
    def __init__(self):
        MyDrawWidget.__init__(self, 1, 1)
        self.canvas_w = 1
        self.canvas_h = 1
        self.viewport_x = 0
        self.viewport_y = 0
        self.clear()
        self.connect("size-allocate", self.size_allocate_event_cb)

    def load(self, filename):
        pixbuf = gtk.gdk.pixbuf_new_from_file(filename)
        self.canvas_w = pixbuf.get_width()
        self.canvas_h = pixbuf.get_height()
        self.viewport_x = 0
        self.viewport_y = 0
        # will need a bigger canvas size than that
        self.resize_if_needed(old_pixbuf = pixbuf)

    def save(self, filename):
        pixbuf = self.get_nonwhite_as_pixbuf()
        pixbuf.save(filename, 'png')

    def size_allocate_event_cb(self, widget, allocation):
        size = (allocation.width, allocation.height)
        self.resize_if_needed(size=size)

    def scroll(self, dx, dy):
        self.viewport_x += dx
        self.viewport_y += dy
        self.set_viewport(self.viewport_x, self.viewport_y)
        self.resize_if_needed()

    def resize_if_needed(self, old_pixbuf = None, size = None):
        vp_w, vp_h = size or self.window.get_size()

        # calculation is done in canvas coordinates
        oldCanvas = Rect(0, 0, self.canvas_w, self.canvas_h)
        viewport  = Rect(self.viewport_x, self.viewport_y, vp_w, vp_h)

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
