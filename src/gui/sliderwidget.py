# This file is part of MyPaint.
# Copyright (C) 2020 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Custom GtkScale/GtkSpinButton combination widget."""

from __future__ import division, print_function

import weakref

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GObject

from lib.pycompat import with_metaclass

from lib.observable import event


# Can't access the constant as a normal attribute - it starts with a digit.
_DOUBLE_CLICK = getattr(Gdk.EventType, "2BUTTON_PRESS")


class ScaleDelegator(type(Gtk.Bin)):
    """Metaclass automatically copying properties from Gtk.Scale

    The purpose of this is to allow setting properties on InputSlider
    instances as if they were Gtk.Scale. In turn, this is to allow use of
    glade/xml to create and set up such instances.
    """

    def __init__(cls, name, bases, dict):
        # Existing properties from shared ancestry
        base = {p.name for p in Gtk.Bin.list_properties()}
        # Properties in GtkScale, but not GtkBin
        to_add = [p for p in Gtk.Scale.list_properties() if p.name not in base]
        for prop in to_add:
            val_type = prop.value_type
            setattr(
                cls, prop.name.replace('-', '_'), GObject.Property(
                type=val_type.pytype if val_type.pytype else val_type,
                default=prop.default_value)
            )
        # Store newly created property names to determine which to delegate
        cls._scale_props = {p.name for p in to_add}
        super(ScaleDelegator, cls).__init__(name, bases, dict)


class InputSlider (with_metaclass(ScaleDelegator, Gtk.Bin)):
    """ Custom container widget switching between slider and spinner box

    This widget is a container with a single child - normally a slider, but
    which can be toggled to a spin button to allow manual adjustment of the
    value by entering a number.

    The metaclass creates copies of the properties from GtkScale + superclasses
    to this class (excepting pre-existing properties through shared ancestry).
    This is so that instances of this class can be constructed from glade/xml
    files. The property values are delegated to the instance of GtkScale.

    Going by the old saying: "Magic is an abomination", it would be preferable
    to have a widget deriving either GtkScale or GtkSpinButton, complementing
    their existing functionality to produce a similar result as this widget,
    without having to use a container and class-init/delegation magic.

    The spin button is created when needed, and not retained when switching
    back to the slider. To change the properties of the spinbutton when it is
    created, set the modify_spinbutton hook, which is called with a reference
    to the slider and a weak reference to the newly created spin button.
    """

    # Needed for instantiation via glade/xml
    __gtype_name__ = 'InputSlider'

    # If the scale/slider does not define a limit on precision, this
    # value is used instead.
    MAX_SPIN_BUTTON_DIGITS = 5
    # Upper limit for the gtk widget
    _REAL_MAX_DIGITS = 20

    def _notify(self, _, prop):
        """Delegate property changes to the scale instance"""
        name = prop.name
        if name in self._scale_props:
            self._scale.set_property(name, self.get_property(name))

    def __init__(self, adj=None):
        super(InputSlider, self).__init__()
        # Set minimum height of this widget based on a spin button,
        # to avoid any layout shifting when switching input mode.
        height_ref = Gtk.SpinButton()
        height_ref.show()
        min_height, _ = height_ref.get_preferred_height()
        self.set_size_request(-1, min_height)

        # Hook up double-click switching
        self.connect("button-press-event", self._bin_button_press_cb)
        scale = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, adj)
        self.add(scale)
        self._scale = scale
        self._scale_mode = True
        self._old_value = None
        self._tooltip_cb_id = None
        # Hook up property delegation
        self.connect("notify", self._notify)

    @property
    def dynamic_tooltip(self):
        return self._tooltip_cb_id is not None

    @dynamic_tooltip.setter
    def dynamic_tooltip(self, enabled):
        if enabled and self._tooltip_cb_id is None:
            self._scale.set_has_tooltip(True)
            self._tooltip_cb_id = self._scale.connect(
                "query-tooltip", self._dynamic_tooltip
            )
        elif not enabled and self._tooltip_cb_id is not None:
            self._scale.set_has_tooltip(False)
            self._scale.disconnect(self._tooltip_cb_id)
            self._tooltip_cb_id = None

    def _dynamic_tooltip(self, scale, x, y, kb_mode, tooltip):
        if kb_mode:
            return False
        else:
            digits = self._scale_precision(scale)
            scale_value = round(scale.get_value(), digits)
            tooltip.set_text(str(scale_value))
            return True

    @property
    def scale(self):
        return self._scale

    @event
    def spin_button_created(self, scale, button_weakref):
        """Event allowing users of this class to modify spinbutton setup"""

    def __getattr__(self, attr):
        """Delegate attribute access to the scale instance"""
        return getattr(self._scale, attr)

    def _bin_button_press_cb(self, widget, event):
        """Trigger mode switch o double click"""
        if self._scale_mode and event.type == _DOUBLE_CLICK:
            self._swap_out()
        # Absorb all button press events - this is to prevent unconsumed
        # press events from the spin button bubbling to the window & triggering
        # widget focus to be cleared (which in turn would trigger the spin
        # button to be removed).
        return True

    def _swap_out(self):
        """Move from scale/slider to spin button"""
        self._old_value = self._scale.get_value()
        self._scale_mode = False
        # Switching sensitivity on and off is a workaround to avoid the slider
        # switching (at least visually) to fine-tune mode when it is swapped
        # back - because it does not receive a mouse button release event
        # when it is swapped out, the last button press is interpreted as a
        # prolonged press, which triggers the fine-tune mode.
        self._scale.set_sensitive(False)
        self.remove(self._scale)
        self.add(self._make_spin_button())
        self.show_all()
        # Set focus to the spin button, to guarantee that the focus-out-event
        # signal will be sent when the spin button no longer has focus.
        self.get_toplevel().set_focus(self.get_child())

    def _make_spin_button(self):
        """Construct a spin_button based on the current state of the scale"""
        scale = self._scale
        adj = scale.get_adjustment()

        precision = self._scale_precision(scale)

        spin_button = Gtk.SpinButton(adjustment=adj, digits=precision)

        # Set increments based on digits of precision - not always sensible,
        # and can be overridden by hooking into the creation event.
        step_increment = 10 ** (0 - precision)
        page_increment = 10 ** (1 - precision)
        spin_button.set_increments(step_increment, page_increment)

        spin_button.connect("key-press-event", self._spin_button_key_event)
        spin_button.connect("key-release-event", self._spin_button_key_event)
        self._focus_cb_id = spin_button.connect(
            "focus-out-event", self._spin_button_focus_out)
        self.spin_button_created(scale, weakref.ref(spin_button))
        return spin_button

    def _scale_precision(self, scale):
        precision = scale.get_round_digits()
        if precision == -1:
            upper = min(self.MAX_SPIN_BUTTON_DIGITS, self._REAL_MAX_DIGITS)
            precision = max(1, upper)
        return precision

    def _spin_button_focus_out(self, *args):
        """Return to scale mode & treat changes to the value as intentional"""
        self._swap_back()

    def _spin_button_key_event(self, spinbut, event):
        """ Switch back on return/enter/escape - reset old value on escape """
        if event.keyval in {Gdk.KEY_Return, Gdk.KEY_KP_Enter}:
            self._swap_back()
            return True
        # Swap back without changing the value whe pressing escape.
        # Only resets on key release, but absorbs both events to prevent the
        # default behavior of a complete mode reset (see gui/keyboard.py)
        if event.keyval == Gdk.KEY_Escape:
            if event.type == Gdk.EventType.KEY_RELEASE:
                self._swap_back(True)
            return True
        return False

    def _swap_back(self, reset_value=False):
        """Return to scale mode, optionally resetting the value"""
        spin_button = self.get_child()
        spin_button.disconnect(self._focus_cb_id)
        self.remove(spin_button)
        self.add(self._scale)
        self._scale.set_sensitive(True)
        if reset_value:
            self._scale.set_value(self._old_value)
        self._scale_mode = True

    def trigger_box_resize(self):
        """Remove & add back scale to trigger resizing of the value box

        Only needed for the sliders with the value box enabled (brush editor)
        There is probably a better way of doing this.
        """
        self.remove(self._scale)
        self.add(self._scale)
