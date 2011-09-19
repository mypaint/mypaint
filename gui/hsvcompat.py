# This file is part of MyPaint.
# Copyright (C) 2011 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layout constants and constructor functions for common widgets.
"""

import gtk
from gtk import gdk
import dialogs
from lib.helpers import clamp
from widgets import find_widgets


class ColorChangerHSV (gtk.Alignment):
    """Wrapper around a GtkHSV triangle widget, bound to the app instance.

    The widget is extracted from a `gtk.ColorSelector` for greater
    compatibility. The code's a bit ugly, but this is necessary to support
    pre-2.18 versions of (Py)GTK.
    """

    __gtype_name__ = 'ColorChangerHSV'

    def __init__(self, app=None, minimal=False, details=True):
        """Construct, bound to an app instance.

            :`minimal`:
                Do not show the colour previews or the eyedropper.
            :`details`:
                Make double-clicking the colour previews show a colour details
                dialog. Updating the dialog updates the current colour.
        """
        gtk.Alignment.__init__(self, xscale=1.0, yscale=1.0)
        self.hsv_changed_observers = []
        self.app = None
        self.color_sel = gtk.ColorSelection()
        self.hsv_widget = None   #: an actual `gtk.HSV`, if bound by PyGTK.
        self._init_hsv_widgets(minimal, details)
        self.color_sel.connect("color-changed", self.on_color_changed)
        self.in_brush_modified_cb = False
        if app is not None:
            self.set_app(app)


    def set_app(self, app):
        self.app = app
        self.set_color_hsv(self.app.brush.get_color_hsv())
        self.app.brush.observers.append(self.brush_modified_cb)
        app.ch.color_pushed_observers.append(self.color_pushed_cb)


    def _init_hsv_widgets(self, minimal, details):
        hsv, = find_widgets(self.color_sel, lambda w: w.get_name() == 'GtkHSV')
        hsv.unset_flags(gtk.CAN_FOCUS)
        hsv.unset_flags(gtk.CAN_DEFAULT)
        hsv.set_size_request(150, 150)
        container = hsv.parent
        if minimal:
            container.remove(hsv)
            self.add(hsv)
        else:
            container.parent.remove(container)
            # Make the packing box give extra space to the HSV widget
            container.set_child_packing(hsv, True, True, 0, gtk.PACK_START)
            container.set_spacing(0)
            if details:
                self.add_details_dialogs(container)
            self.add(container)
        # When extra space is given, grow the HSV wheel.
        # We can only control the GtkHSV's radius if PyGTK exposes it to us
        # as the undocumented gtk.HSV.
        if hasattr(hsv, "set_metrics"):
            def set_hsv_metrics(hsvwidget, alloc):
                radius = min(alloc.width, alloc.height)
                ring_width = max(12, int(radius/16))
                hsvwidget.set_metrics(radius, ring_width)
                hsvwidget.queue_draw()
            hsv.connect("size-allocate", set_hsv_metrics)
            self.hsv_widget = hsv


    def color_pushed_cb(self, pushed_color):
        rgb = self.app.ch.last_color
        color = gdk.Color(*[int(c*65535) for c in rgb])
        self.color_sel.set_previous_color(color)


    def brush_modified_cb(self, settings):
        if not settings.intersection(('color_h', 'color_s', 'color_v')):
            return
        brush_color = self.app.brush.get_color_hsv()
        if brush_color != self.get_color_hsv():
            self.in_brush_modified_cb = True  # do we still need this?
            self.set_color_hsv(brush_color)
            self.in_brush_modified_cb = False


    def get_color_hsv(self):
        if self.hsv_widget is not None:
            # if we can, it's better to go to the GtkHSV widget direct
            return self.hsv_widget.get_color()
        else:
            # not as good, loses hue information if saturation == 0
            color = self.color_sel.get_current_color()
            return color.hue, color.saturation, color.value


    def set_color_hsv(self, hsv):
        h, s, v = hsv
        while h > 1.0: h -= 1.0
        while h < 0.0: h += 1.0
        s = clamp(s, 0.0, 1.0)
        v = clamp(v, 0.0, 1.0)
        if self.hsv_widget is not None:
            self.hsv_widget.set_color(h, s, v)
        else:
            color = gdk.color_from_hsv(h, s, v)
            self.color_sel.set_current_color(color)


    def on_color_changed(self, color_sel):
        h, s, v = self.get_color_hsv()
        for cb in self.hsv_changed_observers:
            cb(h, s, v)
        color_finalized = not color_sel.is_adjusting()
        if color_finalized and not self.in_brush_modified_cb:
            if self.app is not None:
                b = self.app.brush
                b.set_color_hsv((h, s, v))


    def add_details_dialogs(self, hsv_container):
        prev, current = find_widgets(hsv_container,
            lambda w: w.get_name() == 'GtkDrawingArea')
        def on_button_press(swatch, event):
            if event.type != gdk._2BUTTON_PRESS:
                return False
            dialogs.change_current_color_detailed(self.app)
        current.connect("button-press-event", on_button_press)
        prev.connect("button-press-event", on_button_press)


if __name__ == '__main__':
    win = gtk.Window()
    win.set_title("hsvcompat test")
    win.connect("destroy", gtk.main_quit)
    hsv = ColorChangerHSV()
    win.add(hsv)
    win.show_all()
    gtk.main()
