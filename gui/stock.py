import gtk
from gtk import gdk
from gettext import gettext as _

# Symbolic names for our custom stock items.  The string values are also the
# names of the icons used, trying to mirror those in stantard or widespread
# sets.  Mypaint ships with defaults for all of them using the Tango palette
# under the "hicolor" theme.

TOOL_BRUSH = "mypaint-tool-brush"
TOOL_COLOR_SELECTOR = "mypaint-tool-color-triangle"
TOOL_COLOR_SAMPLER = "mypaint-tool-hue-wheel"
TOOL_LAYERS = "mypaint-tool-layers"
ROTATE_LEFT = "object-rotate-left"
ROTATE_RIGHT = "object-rotate-right"
MIRROR_HORIZONTAL = "object-flip-horizontal"
MIRROR_VERTICAL = "object-flip-vertical"
BRUSH_BLEND_MODES = "mypaint-brush-blend-modes"
BRUSH_BLEND_MODE_NORMAL = "mypaint-brush-blend-mode-normal"
BRUSH_BLEND_MODE_ERASER = "mypaint-brush-blend-mode-eraser"
BRUSH_BLEND_MODE_ALPHA_LOCK = "mypaint-brush-blend-mode-alpha-lock"


_stock_items = [
    (TOOL_BRUSH, _("Brush List..."), 0, ord("b"), None),
    (TOOL_COLOR_SELECTOR, _("Color Triangle..."), 0, ord("g"), None),
    (TOOL_COLOR_SAMPLER, _("Color Sampler..."), 0, ord("t"), None),
    (TOOL_LAYERS, _("Layers..."), 0, ord("l"), None),
    (ROTATE_LEFT, _("Rotate Counterclockwise"),
        gdk.CONTROL_MASK, gdk.keyval_from_name("Left"), None),
    (ROTATE_RIGHT, _("Rotate Clockwise"),
        gdk.CONTROL_MASK, gdk.keyval_from_name("Right"), None),
    (MIRROR_HORIZONTAL, _("Mirror Horizontal"), 0, ord("i"), None),
    (MIRROR_VERTICAL, _("Mirror Vertical"), 0, ord("u"), None),
    (BRUSH_BLEND_MODES, _("Blend Mode"), 0, 0, None),
    (BRUSH_BLEND_MODE_NORMAL, _("Normal"), 0, ord('n'), None),
    (BRUSH_BLEND_MODE_ERASER, _("Eraser"), 0, ord('e'), None),
    (BRUSH_BLEND_MODE_ALPHA_LOCK, _("Lock Alpha Channel"),
        gdk.SHIFT_MASK, ord('l'), None),
]

def init_custom_stock_items():
    """Initialise the set of custom stock items defined here.

    Called at application start.
    """
    factory = gtk.IconFactory()
    factory.add_default()
    gtk.stock_add(_stock_items)
    for item_spec in _stock_items:
        stock_id = item_spec[0]
        source = gtk.IconSource()
        source.set_icon_name(stock_id)
        iconset = gtk.IconSet()
        iconset.add_source(source)
        factory.add(stock_id, iconset)
