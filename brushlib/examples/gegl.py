
from gi.repository import GeglGtk3 as GeglGtk
from gi.repository import Gegl, Gtk
from gi.repository import MyPaint, MyPaintGegl

class Application(object):

    def __init__(self):

        self.brush = MyPaint.Brush()
        self.gegl_surface = MyPaintGegl.TiledSurface()
        self.surface = self.gegl_surface.interface()

        self.graph = Gegl.Node();
        self.display_node = self.graph.create_child("gegl:buffer-source")

        # FIXME: does not seem to have any effect
        print self.gegl_surface.get_buffer()
        #self.display_node.set_property("buffer", self.gegl_surface.get_buffer())

        self.button_pressed = False
        self.last_event = (0.0, 0.0, 0.0) # (x, y, time)

        self.init_ui()

    def init_ui(self):

        window = Gtk.Window()
        window.connect("destroy", self.quit)

        top_box = Gtk.VBox()

        self.view_widget = GeglGtk.View.new_for_buffer(self.gegl_surface.get_buffer())
        #self.view_widget.set_node(self.display_node)
        self.view_widget.set_autoscale_policy(GeglGtk.ViewAutoscale.DISABLED)
        self.view_widget.set_size_request(400, 400)
        self.view_widget.connect("draw-background", self.draw_background)

        event_box = Gtk.EventBox()
        event_box.connect("motion-notify-event", self.motion_to)
        event_box.connect("button-press-event", self.button_press)
        event_box.connect("button-release-event", self.button_release)

        event_box.add(self.view_widget)
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
            print "stroke_to"
            self.surface.begin_atomic()
            self.brush.stroke_to(self.surface, x, y, pressure, 0.0, 0.0, dtime)
            self.surface.end_atomic()

        self.last_event = (x, y, time)

    def button_press(self, widget, event):
        self.button_pressed = True
        self.brush.new_stroke()

    def button_release(self, widget, event):
        self.button_pressed = False
        self.brush.reset()

    def draw_background(self, widget, cr, rect):
        # Draw white background
        cr.set_source_rgba(1.0, 1.0, 1.0, 1.0);
        cr.rectangle(rect.x, rect.y, rect.width, rect.height);
        cr.fill();


if __name__ == '__main__':

    Gegl.init(0, "")
    Gtk.init([])

    app = Application()

    app.run()

    Gegl.exit()

def something():

    #MyPaint.init()

    # Create a brush, load from disk
    brush = MyPaint.Brush()
    brush_def = open("brushes/classic/brush.myb").read()
    brush.from_string(brush_def)

    # List all settings
    # TODO: Is there a better way to list all enums with GI?
    settings = [getattr(MyPaint.BrushSetting, attr) for attr in dir(MyPaint.BrushSetting) if attr.startswith("SETTING_")]
    print "Available settings: %s\n" % str(settings)

    # Get info about a given setting
    setting = MyPaint.BrushSetting.SETTING_RADIUS_LOGARITHMIC
    info = MyPaint.brush_setting_info(setting)

    # TODO: rename "def_" to "default"
    print "Setting: %s\n\t Max: %f \n\t Default: %f \n\t Min: %f" % (info.cname, info.max, info.def_, info.min)
    print "\t Name: %s\n\t Tooltip: '%s'\n" % (info.get_name(), info.get_tooltip()) # Use the getters so that i18n works

    # TODO: should be MyPaint.BrushSetting.from_cname
    # Same with MyPaint.Brush.input_from_cname
    assert (MyPaint.Brush.setting_from_cname(info.cname) == setting)

    # Get/Set current base value for the given setting
    print "Base value is: %f" % brush.get_base_value(setting)
    brush.set_base_value(setting, 2.0)
    assert brush.get_base_value(setting) ==  2.0

    # Get dynamics for given setting
    inputs = [getattr(MyPaint.BrushInput, a) for a in dir(MyPaint.BrushInput) if a.startswith('INPUT_')]
    if not brush.is_constant(setting):
        for input in inputs:
            mapping_points = brush.get_mapping_n(setting, input)
            if mapping_points > 1: # If 0, no dynamics for this input
                points = [brush.get_mapping_point(setting, input, i) for i in range(mapping_points)]
                print "Has dynamics for input %s:\n%s" % (input, str(points))


    # Create a surface to paint on
    Gegl.init(0, "")
    surface = MyPaintGegl.TiledSurface()
    s = surface.interface()

    print surface.get_buffer()

    for x, y in [(0.0, 0.0), (100.0, 100.0), (100.0, 200.0)]:
        dtime = 0.1 # XXX: Important to set correctly for speed calculations
        s.begin_atomic()
        brush.stroke_to(s, x, y, pressure=1.0, xtilt=0.0, ytilt=0.0, dtime=dtime)
        rect = s.end_atomic()
        print rect.x, rect.y, rect.width, rect.height

    Gegl.exit()
