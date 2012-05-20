# This file is part of MyPaint.
# Copyright (C) 2011,2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""HSV widget compatability layer.
"""

import gtk
from gtk import gdk

from util import clamp
from gettext import gettext as _

from adjbases import ColorAdjuster, HSVColor, UIColor
from combined import CombinedAdjusterPage


def find_widgets(widget, predicate):
    """Finds widgets in a container's tree by predicate.
    """
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


class HSVTrianglePage (CombinedAdjusterPage):

    __table = None
    __adj = None

    def __init__(self):
        adj = HSVTriangle(minimal=True)
        self.__adj = adj
        self.__table = gtk.Table(rows=1, columns=1)
        opts = gtk.FILL|gtk.EXPAND
        self.__table.attach(adj, 0,1, 0,1, opts, opts, 3, 3)

    @classmethod
    def get_page_icon_name(class_):
        return "mypaint-tool-color-triangle"

    @classmethod
    def get_page_title(class_):
        return _("HSV Triangle")

    @classmethod
    def get_page_description(class_):
        return _("The standard GTK color selector")

    def get_page_table(self):
        return self.__table

    def set_color_manager(self, manager):
        ColorAdjuster.set_color_manager(self, manager)
        self.__adj.set_color_manager(manager)


class HSVTriangle (gtk.Alignment, ColorAdjuster):
    """Wrapper around a GtkHSV triangle widget, bound to the app instance.

    The widget is extracted from a `gtk.ColorSelector` for greater
    compatibility. The code's a bit ugly, but this is necessary to support
    pre-2.18 versions of (Py)GTK.

    """

    __gtype_name__ = 'HSVTriangle'


    def __init__(self, minimal=False, details=True):
        """Initiailize.

        :`minimal`:
            Do not show the colour previews or the eyedropper.
        :`details`:
            Make double-clicking the colour previews show a colour details
            dialog. Updating the dialog updates the current colour.

        """
        gtk.Alignment.__init__(self, xscale=1.0, yscale=1.0)
        self.hsv_changed_observers = []
        self.color_sel = gtk.ColorSelection()
        self.hsv_widget = None   #: an actual `gtk.HSV`, if bound by PyGTK.
        self._init_hsv_widgets(minimal, details)
        self.color_sel.connect("color-changed", self.color_changed_cb)


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


    def update_cb(self):
        color = self.get_managed_color()
        self.set_current_color(color)
        color = self.get_color_manager().get_previous_color()
        gdk_color = color.to_gdk_color()
        self.color_sel.set_previous_color(gdk_color)


    def get_color_hsv(self):
        if self.hsv_widget is not None:
            # if we can, it's better to go to the GtkHSV widget direct
            return self.hsv_widget.get_color()
        else:
            # not as good, loses hue information if saturation == 0
            color = self.color_sel.get_current_color()
            return color.hue, color.saturation, color.value


    def set_current_color(self, color):
        if self.hsv_widget is not None:
            self.hsv_widget.set_color(*color.get_hsv())
        else:
            self.color_sel.set_current_color(color.to_gdk_color())


    def color_changed_cb(self, color_sel):
        h, s, v = self.get_color_hsv()
        for cb in self.hsv_changed_observers:
            cb(h, s, v)
        color_finalized = not color_sel.is_adjusting()
        if color_finalized:
            color = HSVColor(h, s, v)
            self.set_managed_color(color)


    def add_details_dialogs(self, hsv_container):
        prev, current = find_widgets(hsv_container,
            lambda w: w.get_name() == 'GtkDrawingArea')
        def on_button_press(swatch, event):
            if event.type != gdk._2BUTTON_PRESS:
                return False
            mgr = self.get_color_manager()
            col = mgr.get_color()
            prev_col = mgr.get_previous_color()
            col = UIColor.new_from_dialog(title=_("Color details"),
                                          color=col, previous_color=prev_col,
                                          parent=self.get_toplevel())
            if col is not None:
                self.set_managed_color(col)
        current.connect("button-press-event", on_button_press)
        prev.connect("button-press-event", on_button_press)


if __name__ == '__main__':
    win = gtk.Window()
    win.set_title("hsvtriangle test")
    win.connect("destroy", gtk.main_quit)
    hsv = HSVTriangle()
    win.add(hsv)
    win.show_all()
    gtk.main()
