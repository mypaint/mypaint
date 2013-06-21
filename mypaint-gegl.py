import sys, os.path

import gi
from gi.repository import Gegl, Gtk
from gi.repository import GeglGtk3 as GeglGtk

from lib import mypaintlib, tiledsurface, brush


def draw_test_data(surface, brush):

    # WARNING: can pull in GTK2, depending on matplotlib config.
    # This will crash application. Use the Tk backend of matplotlib instead
    import pylab

    events = pylab.loadtxt('tests/painting30sec.dat')
    s, b = surface, brush

    for i in range(10):
        t_old = events[0][0]
        s.begin_atomic()
        for t, x, y, pressure in events:
            dtime = t - t_old
            t_old = t
            b.stroke_to (s.backend, x, y, pressure, 0.0, 0.0, dtime)
        s.end_atomic()

def find_widgets(widget, predicate):
    """Finds widgets in a container's tree by predicate."""
    queue = [widget]
    found = []
    while len(queue) > 0:
        w = queue.pop(0)
        if predicate(w):
            found.append(w)
        if hasattr(w, "get_children"):
            for w2 in w.get_children():
                queue.append(w2)
    return found


# TODO: Create a lib.document.Document instance and use that instead of
# using tiledsurface.GeglSurface() directly
# TODO: Refactor eventhandling out of gui.tileddrawwidget.TiledDrawWiget to
# separate widget subclassgin Gtk.EventBox and reuse it here.
class MyPaintGeglApplication(object):

    def __init__(self):
        self.brush_info = brush.BrushInfo(open('tests/brushes/charcoal.myb').read())
        self.brush_info.set_color_rgb((0.0, 0.0, 0.0))

        self.brush = brush.Brush(self.brush_info)
        self.surface = tiledsurface.GeglSurface()
        self.display_node = self.surface.get_node()
        self.graph = Gegl.Node();

        self.button_pressed = False
        self.last_event = (0.0, 0.0, 0.0) # (x, y, time)

        self.init_ui()

    def init_ui(self):

        window = Gtk.Window()
        window.connect("destroy", self.quit)

        top_box = Gtk.VBox()

        save_button = Gtk.Button("Save file")
        save_button.connect("clicked", self.save_handler)

        load_button = Gtk.Button("Load file")
        load_button.connect("clicked", self.load_handler)

        brush_button = Gtk.Button("Load brush")
        brush_button.connect("clicked", self.change_brush_handler)

        self.color_selector = Gtk.ColorSelection()
        self.color_selector.connect("color-changed", self.color_change_handler)
        
        # Extract only the color triangle
        hsv_selector = find_widgets(self.color_selector, lambda w: w.get_name() == 'GtkHSV')[0]
        hsv_selector.unparent()

        self.view_widget = GeglGtk.View()
        self.view_widget.set_node(self.display_node)
        self.view_widget.set_autoscale_policy(GeglGtk.ViewAutoscale.DISABLED)
        self.view_widget.set_size_request(400, 400)
        self.view_widget.connect("draw-background", self.draw_background)

        event_box = Gtk.EventBox()
        event_box.connect("motion-notify-event", self.motion_to)
        event_box.connect("button-press-event", self.button_press)
        event_box.connect("button-release-event", self.button_release)

        button_box = Gtk.VBox()
        ui_box = Gtk.HBox()

        event_box.add(self.view_widget)
        button_box.pack_start(save_button, expand=False, fill=True, padding=0)
        button_box.pack_start(load_button, expand=False, fill=True, padding=0)
        button_box.pack_start(brush_button, expand=False, fill=True, padding=0)
        ui_box.pack_start(hsv_selector, expand=False, fill=True, padding=0)
        ui_box.pack_start(button_box, expand=True, fill=True, padding=0)
        top_box.pack_start(ui_box, expand=False, fill=True, padding=0)
        top_box.pack_start(event_box, expand=True, fill=True, padding=0)
        window.add(top_box)
        window.show_all()
        
        self.window = window

    def run(self):
        return Gtk.main()

    def quit(self, *ignored):
        Gtk.main_quit()

    def motion_to(self, widget, event):

        (x, y, time) = event.x, event.y, event.time

        # FIXME: crashes?
        #matrix = self.view_widget.get_transformation()
        #matrix.invert()
        #x, y = matrix.transform_point2(x, y)

        pressure = 0.5
        dtime = (time - self.last_event[2])/1000.0
        if self.button_pressed:
            self.surface.begin_atomic()
            self.brush.stroke_to(self.surface.backend, x, y, pressure, 0.0, 0.0, dtime)
            self.surface.end_atomic()

        self.last_event = (x, y, time)

    def button_press(self, widget, event):
        self.button_pressed = True

    def button_release(self, widget, event):
        self.button_pressed = False
        self.brush.reset()

    def draw_background(self, widget, cr, rect):
        # Draw white background
        cr.set_source_rgba(1.0, 1.0, 1.0, 1.0);
        cr.rectangle(rect.x, rect.y, rect.width, rect.height);
        cr.fill();

    def load_from_png(self, filename):
        self.surface.load_from_png(filename, 0, 0)

    def save_as_png(self, filename):
        self.surface.save_as_png(filename)

    def save_handler(self, widget):

        buttons = (Gtk.STOCK_OK, Gtk.ResponseType.OK,
                   Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
        action = Gtk.FileChooserAction.SAVE
        chooser = Gtk.FileChooserDialog("Save file", self.window,
                                        action=action, buttons=buttons)
        
        response = chooser.run()
        if response == Gtk.ResponseType.OK:
            self.save_as_png(chooser.get_filename())

        chooser.destroy()

    def load_handler(self, widget):
        
        buttons = (Gtk.STOCK_OK, Gtk.ResponseType.OK,
                   Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
        action = Gtk.FileChooserAction.OPEN
        chooser = Gtk.FileChooserDialog("Load file", self.window,
                                        action=action, buttons=buttons)
        
        response = chooser.run()
        if response == Gtk.ResponseType.OK:
            self.load_from_png(chooser.get_filename())

        chooser.destroy()

    def change_brush_handler(self, widget):

        buttons = (Gtk.STOCK_OK, Gtk.ResponseType.OK,
                   Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
        chooser = Gtk.FileChooserDialog("Load brush", self.window, buttons=buttons)
        chooser.set_current_folder("brushes/deevad")
        
        response = chooser.run()
        if response == Gtk.ResponseType.OK:
            brush_settings = open(chooser.get_filename()).read()
            self.brush_info.load_from_string(brush_settings)

        chooser.destroy()

    def color_change_handler(self, widget):
        c = widget.get_current_color()
        rgb = (c.red/65535.0, c.green/65535.0, c.blue/65535.0)
        self.brush_info.set_color_rgb(rgb)

if __name__ == '__main__':

    Gegl.init([])
    Gtk.init([])

    app = MyPaintGeglApplication()

    #draw_test_data(app.surface, app.brush)

    # TEMP:
    #app.save_as_png("mypaint-gegl.png")
    app.run()




