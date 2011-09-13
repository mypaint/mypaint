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
from gettext import gettext as _
from lib.helpers import clamp
import dialogs


# Spacing constants

SPACING_TIGHT = 6
SPACING = 2 * SPACING_TIGHT
SPACING_LOOSE = 3*SPACING_TIGHT


# Useful style defaults for named widgets

gtk.rc_parse_string ("""
    style "borderless-style" {
        GtkWidget::focus-padding = 0
        GtkWidget::focus-line-width = 0
        xthickness = 0
        ythickness = 0
    }
    style "thinborder-style" {
        GtkWidget::focus-padding = 0
        GtkWidget::focus-line-width = 0
        xthickness = 1
        ythickness = 1
    }
    style "borderless-toolbar-style" {
        # Confuses some theme engines:
        #GtkToolbar::shadow-type = GTK_SHADOW_NONE
        # Following suffices to comply with Fitts's Law in fullscreen:
        GtkToolbar::internal-padding = 0
        xthickness = 0
        ythickness = 0
    }
    widget "*.borderless*" style "borderless-style"
    widget "*.thinborder*" style "thinborder-style"
    widget "*.toolbar1" style "borderless-toolbar-style"
    """)

BORDERLESS_BUTTON_NAME = "borderless-button"


def borderless_button(stock_id=None, size=gtk.ICON_SIZE_BUTTON, tooltip=None):
    button = gtk.Button()
    if stock_id is not None:
        image = gtk.Image()
        image.set_from_stock(stock_id, size)
        button.add(image)
    button.set_name(BORDERLESS_BUTTON_NAME)
    button.set_relief(gtk.RELIEF_NONE)
    button.set_can_default(False)
    button.set_can_focus(False)
    button.set_tooltip_text(tooltip)
    return button


def section_frame(label_text):
    frame = gtk.Frame()
    label_markup = "<b>%s</b>" % label_text
    label = gtk.Label(label_markup)
    label.set_use_markup(True)
    frame.set_label_widget(label)
    frame.set_shadow_type(gtk.SHADOW_NONE)
    return frame


class ItemSpinBox (gtk.HBox):
    """Control for selecting one of a small number of text items.

    Somewhat like a `gtk.SpinButton`, but with a textual ``(value,
    display_value)`` list model, and with the buttons positioned at either
    end of the central label. Hopefully nothing too unusual.

    Intended as an alternative to `gtk.ComboBox` which doesn't involve any
    pointer grab breaking, which makes it more useful for dropdown panels.
    """

    ARROW_SHADOW_TYPE = gtk.SHADOW_OUT

    def __init__(self, model, changed_cb, value=None):
        gtk.HBox.__init__(self)
        self._left_button = borderless_button(tooltip=_("Previous item"))
        self._left_button.add(gtk.Arrow(gtk.ARROW_LEFT, self.ARROW_SHADOW_TYPE))
        self._left_button.connect("clicked", self._spin_button_clicked, -1)
        self._right_button = borderless_button(tooltip=_("Next item"))
        self._right_button.add(gtk.Arrow(gtk.ARROW_RIGHT, self.ARROW_SHADOW_TYPE))
        self._right_button.connect("clicked", self._spin_button_clicked, 1)
        self._label = gtk.Label()
        self.pack_start(self._left_button, False, False)
        self.pack_start(self._label, True, True)
        self.pack_start(self._right_button, False, False)
        self._changed_cb = changed_cb
        self._model = None
        self._model_index = None
        self.set_model(model, value)

    def set_model(self, model, value=None):
        """Set the model.

        The `model` argument is either `None`, or a list of pairs of the form
        ``[(value, text), ...]``. The ``value``s must be unique and testable by
        equality. The ``text`` values are what is displayed in the label area
        of the `ItemSpinBox` widget.
        """
        old_index = self._model_index
        old_value = None
        old_text = None
        if old_index is not None:
            old_value, old_text = self.model[old_index]

        self._model = model
        self._model_index = None
        text = None
        if value is None:
            value = old_value
        self.set_value(value)

        model_valid = self._is_model_valid()
        buttons_sensitive = model_valid and len(self._model) > 1
        self._left_button.set_sensitive(buttons_sensitive)
        self._right_button.set_sensitive(buttons_sensitive)


    def _is_model_valid(self):
        if self._model is None: return False
        if not self._model: return False
        if self._model_index is None: return False
        if self._model_index < 0: return False
        if self._model_index >= len(self._model): return False
        return True


    def _update_label(self):
        if not self._is_model_valid():
            text = _("(Nothing to show)")
            self._label.set_sensitive(False)
        else:
            value, text = self._model[self._model_index]
            self._label.set_sensitive(True)
        self._label.set_text(text)


    def get_value(self):
        if not self._is_model_valid():
            return None
        value, text = self._model[self._model_index]
        return value


    def set_value(self, value, notify=False):
        new_value = None
        if not self._model:
            self._model_index = None
            self._update_label()
            new_value = None
        else:
            found = False
            for i, entry in enumerate(self._model):
                v, t = entry
                if v == value:
                    self._model_index = i
                    self._update_label()
                    new_value = v
                    found = True
                    break
            if not found:
                self._model_index = 0
                self._update_label()
                new_value = self._model[0][0]
        if notify:
            self._changed_cb(new_value)


    def _spin_button_clicked(self, widget, delta):
        if not self._is_model_valid():
            return
        i = self._model_index + delta
        while i < 0:
            i += len(self._model)
        i %= len(self._model)
        self._model_index = i
        self._update_label()
        self._changed_cb(self.get_value())


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


class ColorChangerHSV (gtk.Alignment):
    """Wrapper around a GtkHSV triangle widget, bound to the app instance.

    The widget is extracted from a `gtk.ColorSelector` for greater
    compatibility. The code's a bit ugly, but this is necessary to support
    pre-2.18 versions of (Py)GTK.
    """

    def __init__(self, app, minimal=False, details=True):
        """Construct, bound to an app instance.

            :`minimal`:
                Do not show the colour previews or the eyedropper.
            :`details`:
                Make double-clicking the colour previews show a colour details
                dialog. Updating the dialog updates the current colour.
        """
        gtk.Alignment.__init__(self, xscale=1.0, yscale=1.0)
        self.hsv_changed_observers = []
        self.app = app
        self.color_sel = gtk.ColorSelection()
        self.hsv_widget = None   #: an actual `gtk.HSV`, if bound by PyGTK.
        self._init_hsv_widgets(minimal, details)
        self.color_sel.connect("color-changed", self.on_color_changed)
        self.in_brush_modified_cb = False
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
    win.set_title("misc widgets")
    win.connect("destroy", gtk.main_quit)

    outer_vbox = gtk.VBox()
    outer_vbox.set_border_width(SPACING)
    win.add(outer_vbox)

    frame = section_frame("Item spin buttons")
    outer_vbox.pack_start(frame, True, True)
    vbox = gtk.VBox()
    vbox.set_border_width(SPACING)
    vbox.set_spacing(SPACING_TIGHT)
    frame.add(vbox)

    def changed_cb(new_value):
        print "SSB changed:", new_value

    sb0 = ItemSpinBox(None, changed_cb)
    vbox.pack_start(sb0, False, False)

    model1 = list(enumerate("Apple Orange Pear Banana Lychee Herring Guava".split()))
    sb1 = ItemSpinBox(model1, changed_cb)
    vbox.pack_start(sb1, False, False)

    sb1a = ItemSpinBox(model1, changed_cb, 4)
    vbox.pack_start(sb1a, False, False)

    model2 = [(0, "Single value")]
    sb2 = ItemSpinBox(model2, changed_cb)
    vbox.pack_start(sb2, False, False)

    #win.set_size_request(200, 150)

    win.show_all()
    gtk.main()
