"select brush window"
import gtk
import brush, mydrawwidget

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.app.brush_selected_callbacks.insert(0, self.brush_selected_cb)
        self.add_accel_group(self.app.accel_group)

        self.set_title('Brush selection')
        self.connect('delete-event', self.app.hide_window_cb)

        # TODO: load available brushes
        # bad idea - self.brushes.append(brush.Brush())

        vbox = gtk.VBox()
        self.add(vbox)

        self.brushlist = BrushList(self.app)
        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        scroll.add_with_viewport(self.brushlist)
        vbox.pack_start(scroll)
        #scroll.resize_children() # whyever

        #vbox.pack_start(self.brushlist, expand=True, fill=True)

        vbox.pack_start(gtk.HSeparator(), expand=False)

        hbox = gtk.HBox()
        hbox.set_border_width(8)
        vbox.pack_start(hbox, expand=False, fill=False)
        # bad, fixed maximal size -- No, that's actually good!
        self.mdw = mydrawwidget.MyDrawWidget(128, 128)
        #self.mdw = mydrawwidget.MyDrawWidget(512, 512)
        self.mdw.clear()
        self.mdw.set_brush(self.app.brush)
        self.mdw.set_size_request(128, 128)
        hbox.pack_start(self.mdw, expand=False, fill=False)

        vbox2 = gtk.VBox()
        hbox.pack_end(vbox2, expand=False, fill=False)
        #hbox.properties.padding = 10
        #hbox.set_spacing(10)

        b = gtk.Button('Clear')
        def clear_cb(window, mdw):
            mdw.clear()
        b.connect('clicked', clear_cb, self.mdw)
        vbox2.pack_start(b, expand=False)

        b = gtk.Button('add as new')
        b.connect('clicked', self.add_as_new_cb)
        vbox2.pack_start(b, expand=False)

        b = gtk.Button('save preview')
        b.connect('clicked', self.update_preview_cb)
        vbox2.pack_start(b, expand=False)

        b = gtk.Button('save settings')
        b.connect('clicked', self.update_settings_cb)
        vbox2.pack_start(b, expand=False)

        b = gtk.Button('delete selected')
        b.connect('clicked', self.delete_selected_cb)
        vbox2.pack_start(b, expand=False)

        self.resize(300, 500)

    def set_preview_pixbuf(self, pixbuf):
        if pixbuf is None:
            self.mdw.clear()
        else:
            self.mdw.set_from_pixbuf(pixbuf)

    def get_preview_pixbuf(self):
        pixbuf = self.mdw.get_as_pixbuf()
        # TODO: cut only painted area, please
        return pixbuf

    def add_as_new_cb(self, window):
        b = brush.Brush()
        b.copy_settings_from(self.app.brush)
        b.update_preview(self.get_preview_pixbuf())
        self.app.brushes.insert(0, b)
        self.brushlist.redraw_thumbnails()
        self.app.select_brush(b)
        b.save(self.app.brushpath)

    def update_preview_cb(self, window):
        pixbuf = self.mdw.get_as_pixbuf()
        b = self.app.selected_brush
        if b is None:
            # no brush selected
            display = gtk.gdk.display_get_default()
            display.beep()
            return
        b.update_preview(pixbuf)
        b.save(self.app.brushpath)
        self.brushlist.redraw_thumbnails()

    def update_settings_cb(self, window):
        b = self.app.selected_brush
        if b is None:
            # no brush selected
            display = gtk.gdk.display_get_default()
            display.beep()
            return
        b.copy_settings_from(self.app.brush)
        b.save(self.app.brushpath)

    def delete_selected_cb(self, window):
        b = self.app.selected_brush
        if b is None: return

        d = gtk.Dialog("Really delete this brush?",
             self,
             gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
             (gtk.STOCK_YES, gtk.RESPONSE_ACCEPT,
              gtk.STOCK_NO, gtk.RESPONSE_REJECT))
        response = d.run()
        d.destroy()
        if response != gtk.RESPONSE_ACCEPT: return

        self.app.select_brush(None)
        self.app.brushes.remove(b)
        b.delete(self.app.brushpath)
        self.brushlist.redraw_thumbnails()

    def brush_selected_cb(self, brush):
        if brush is None: return
        if brush is self.app.selected_brush:
            # selected same brush twice: load pixmap
            self.set_preview_pixbuf(brush.preview)


preview_spacing_outside = 0
preview_border_visible = 1
preview_spacing_inside = 1
preview_total_border = preview_border_visible + preview_spacing_inside + preview_spacing_outside
preview_total_w = brush.thumb_w + 2*preview_total_border
preview_total_h = brush.thumb_h + 2*preview_total_border

class BrushList(gtk.DrawingArea):
    "choose a brush by preview"
    def __init__(self, app):
        gtk.DrawingArea.__init__(self)
        self.pixbuf = None
        self.app = app
        self.app.brush_selected_callbacks.append(self.brush_selected_cb)

        self.tiles_w = 4

        self.connect("expose-event", self.expose_cb)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("configure-event", self.configure_event_cb)
	self.set_events(gtk.gdk.EXPOSURE_MASK |
                        gtk.gdk.BUTTON_PRESS_MASK)
        self.redraw_thumbnails()

    def redraw_thumbnails(self, width = None, height = None):
        if width is None:
            if not self.pixbuf: return
            width = self.pixbuf.get_width()
            height = self.pixbuf.get_height()
        self.tiles_w = (width / preview_total_w) or 1
        self.tiles_h = len(self.app.brushes)/self.tiles_w + 1
        height = self.tiles_h * preview_total_h
        self.set_size_request(0, height)
        self.pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, width, height)
        self.pixbuf.fill(0xffffffff) # white
        i = 0
        for b in self.app.brushes:
            x = (i % self.tiles_w) * preview_total_w
            y = (i / self.tiles_w) * preview_total_h
            x += preview_total_border
            y += preview_total_border
            b.preview_thumb.copy_area(0, 0, brush.thumb_w, brush.thumb_h, self.pixbuf, x, y)
            i += 1
        self.queue_draw()

    def button_press_cb(self, widget, event):
        x, y = int(event.x), int(event.y)
        i = x / preview_total_w
        if i >= self.tiles_w: return
        i = i + self.tiles_w * (y / preview_total_h)
        if i >= len(self.app.brushes): return
        # keep the color setting
        color = self.app.brush.get_color()
        brush = self.app.brushes[i]

        # brush changed on harddisk?
        # (selecting another one might save over it because of drawing time update)
        changed = brush.reload_if_changed(self.app.brushpath)
        if changed:
            self.redraw_thumbnails()

        self.app.select_brush(brush)
        self.app.brush.set_color(color)

    def brush_selected_cb(self, brush):
        self.queue_draw()

    #def size_request_cb(self, widget, size):
    def configure_event_cb(self, widget, size):
        if self.pixbuf and self.pixbuf.get_width() == size.width:
            if self.pixbuf.get_height() == size.height:
                return
        self.redraw_thumbnails(size.width, size.height)

    def expose_cb(self, widget, event):
        rowstride = self.pixbuf.get_rowstride()
        pixels = self.pixbuf.get_pixels()
        
        # cut to maximal size
        e_x, e_y = event.area.x, event.area.y
        e_w, e_h = event.area.width, event.area.height
        p_w, p_h = self.pixbuf.get_width(), self.pixbuf.get_height()

        widget.window.draw_rgb_image(
            widget.style.black_gc,
            0, 0, p_w, p_h,
            'normal',
            pixels, rowstride)

        # draw borders
        i = 0
        for b in self.app.brushes:
            if b is self.app.selected_brush:
                gc = widget.style.black_gc
            else:
                gc = widget.style.white_gc
            x = (i % self.tiles_w) * preview_total_w
            y = (i / self.tiles_w) * preview_total_h
            w = preview_total_w
            h = preview_total_h
            def shrink(pixels, x, y, w, h):
                x += pixels
                y += pixels
                w -= 2*pixels
                h -= 2*pixels
                return (x, y, w, h)
            x, y, w, h = shrink(preview_spacing_outside, x, y, w, h)
            for j in range(preview_border_visible):
                widget.window.draw_rectangle(gc, False, x, y, w-1, h-1)
                x, y, w, h = shrink(1, x, y, w, h)
            i += 1

        return True



