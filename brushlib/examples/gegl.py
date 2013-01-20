
# from gi.repository import Gegl, GeglGtk3
from gi.repository import MyPaint, MyPaintGegl

if __name__ == '__main__':

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
    surface = MyPaint.Surface()
    # FIXME: Must use a real surface here, using just the interface wont paint anything
    # TypeError: cannot allocate disguised struct MyPaintGegl.TiledSurface;
    # surface = MyPaintGegl.TiledSurface()

    for x, y in [(0.0, 0.0), (10.0, 10.0), (10.0, 20.0)]:
        dtime = 100.0 # XXX: Important to set correctly for speed calculations
        brush.stroke_to(surface, x, y, pressure=1.0, xtilt=0.0, ytilt=0.0, dtime=dtime)

