# This file is part of MyPaint.
# Copyright (C) 2011-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function

from gettext import gettext as _

from lib.gibindings import Gtk

from .widgets import borderless_button


class ItemSpinBox (Gtk.HBox):
    """Control for selecting one of a small number of text items.

    Somewhat like a `Gtk.SpinButton`, but with a textual ``(value,
    display_value)`` list model, and with the buttons positioned at either
    end of the central label. Hopefully nothing too unusual.

    Intended as an alternative to `Gtk.ComboBox` which doesn't involve any
    pointer grab breaking, which makes it more useful for dropdown panels.
    """

    __gtype_name__ = "ItemSpinBox"

    ARROW_SHADOW_TYPE = Gtk.ShadowType.OUT

    def __init__(self, model=None, changed_cb=None, value=None):
        super(ItemSpinBox, self).__init__()
        self._left_button = borderless_button(tooltip=_("Previous item"))
        self._left_button.add(
            Gtk.Arrow.new(Gtk.ArrowType.LEFT, self.ARROW_SHADOW_TYPE))
        self._left_button.connect("clicked", self._spin_button_clicked, -1)
        self._right_button = borderless_button(tooltip=_("Next item"))
        self._right_button.add(
            Gtk.Arrow.new(Gtk.ArrowType.RIGHT, self.ARROW_SHADOW_TYPE))
        self._right_button.connect("clicked", self._spin_button_clicked, 1)
        self._label = Gtk.Label()
        self.pack_start(self._left_button, False, False, 0)
        self.pack_start(self._label, True, True, 0)
        self.pack_start(self._right_button, False, False, 0)
        self._changed_cb = None
        self.set_changed_callback(changed_cb)
        self._model = None
        self._model_index = None
        self.set_model(model, value)

    def set_changed_callback(self, changed_cb):
        """Set the value-changed callback.

        `changed_cb` will be called each time the user chooses a different item
        in the list, with the ``value`` member from the model (see
        `set_model()`).
        """
        self._changed_cb = changed_cb

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
            old_value, old_text = self._model[old_index]

        self._model = model
        self._model_index = None
        if value is None:
            value = old_value
        self.set_value(value)

        model_valid = self._is_model_valid()
        buttons_sensitive = model_valid and len(self._model) > 1
        self._left_button.set_sensitive(buttons_sensitive)
        self._right_button.set_sensitive(buttons_sensitive)

    def _is_model_valid(self):
        if self._model is None:
            return False
        if not self._model:
            return False
        if self._model_index is None:
            return False
        if self._model_index < 0:
            return False
        if self._model_index >= len(self._model):
            return False
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
        if notify and self._changed_cb is not None:
            self._changed_cb(new_value)

    def _spin_button_clicked(self, widget, delta):
        self._spin(delta)

    def _spin(self, delta):
        if not self._is_model_valid():
            return
        i = self._model_index + delta
        while i < 0:
            i += len(self._model)
        i %= len(self._model)
        self._model_index = i
        self._update_label()
        if self._changed_cb is not None:
            self._changed_cb(self.get_value())

    def next(self):
        """Spin to the next item"""
        self._spin(1)

    def previous(self):
        """Spin to the previous item"""
        self._spin(-1)


if __name__ == '__main__':
    win = Gtk.Window()
    win.set_title("spinbox test")
    win.connect("destroy", Gtk.main_quit)

    vbox = Gtk.VBox()
    win.add(vbox)

    def changed_cb(new_value):
        print("SSB changed:", new_value)

    sb0 = ItemSpinBox(None, changed_cb)
    vbox.pack_start(sb0, False, False, 0)

    fruits = "Apple Orange Pear Banana Lychee Herring Guava".split()
    model1 = list(enumerate(fruits))
    sb1 = ItemSpinBox(model1, changed_cb)
    vbox.pack_start(sb1, False, False, 0)

    sb1a = ItemSpinBox(model1, changed_cb, 4)
    vbox.pack_start(sb1a, False, False, 0)

    model2 = [(0, "Single value")]
    sb2 = ItemSpinBox(model2, changed_cb)
    vbox.pack_start(sb2, False, False, 0)

    sb3 = ItemSpinBox()
    vbox.pack_start(sb3, False, False, 0)

    win.show_all()
    Gtk.main()
