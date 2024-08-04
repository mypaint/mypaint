# This file is part of MyPaint.
# Copyright (C) 2013-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Dockable panel showing options for the current mode"""

## Imports

from __future__ import division, print_function
import logging

from .toolstack import SizedVBoxToolWidget, TOOL_WIDGET_NATURAL_HEIGHT_SHORT
import lib.xml
from lib.gettext import C_

from lib.gibindings import Gtk

logger = logging.getLogger(__name__)


## Class defs

class ModeOptionsTool (SizedVBoxToolWidget):
    """Dockable panel showing options for the current mode

    This panel has a title and an icon reflecting the current mode, and
    displays its options widget if it has one: define an object method named
    ``get_options_widget()`` returning an arbitrary GTK widget. Singletons work
    well here, and are encouraged. ``get_options_widget()`` can also return
    `None` if the mode is a temporary mode which has no sensible options. In
    this case, any widget already displayed will not be replaced, which is
    particularly appropriate for modes which only persist for the length of
    time the mouse button is held, and stack on top of other modes.
    """

    ## Class constants

    SIZED_VBOX_NATURAL_HEIGHT = TOOL_WIDGET_NATURAL_HEIGHT_SHORT

    tool_widget_icon_name = "mypaint-options-symbolic"
    tool_widget_title = C_(
        "options panel: tab tooltip: title",
        "Tool Options",
    )
    tool_widget_description = C_(
        "options panel: tab tooltip: description",
        "Specialized settings for the current editing tool",
    )

    __gtype_name__ = 'MyPaintModeOptionsTool'

    OPTIONS_MARKUP = C_(
        "options panel: header",
        "<b>{mode_name}</b>",
    )
    NO_OPTIONS_MARKUP = C_(
        "options panel: body",
        "<i>No options available</i>",
    )

    ## Method defs

    def __init__(self):
        """Construct, and connect internal signals & callbacks"""
        SizedVBoxToolWidget.__init__(self)
        from gui.application import get_app
        self._app = get_app()
        self._app.doc.modes.changed += self._modestack_changed_cb
        self.set_border_width(3)
        self.set_spacing(6)
        # Placeholder in case a mode has no options
        label = Gtk.Label()
        label.set_markup(self.NO_OPTIONS_MARKUP)
        self._no_options_label = label
        # Container for an options widget exposed by the current mode
        self._mode_icon = Gtk.Image()
        label = Gtk.Label()
        label.set_text("<options-label>")
        self._options_label = label
        label.set_alignment(0.0, 0.5)
        label_hbox = Gtk.HBox()
        label_hbox.set_spacing(3)
        label_hbox.set_border_width(3)
        label_hbox.pack_start(self._mode_icon, False, False, 0)
        label_hbox.pack_start(self._options_label, True, True, 0)
        align = Gtk.Alignment.new(0.5, 0.5, 1.0, 1.0)
        align.set_padding(0, 0, 0, 0)
        align.set_border_width(3)
        self._options_bin = align
        self.pack_start(label_hbox, False, False, 0)
        self.pack_start(align, True, True, 0)
        self.connect("show", lambda *a: self._update_ui())
        # Fallback
        self._update_ui_with_options_widget(
            self._no_options_label,
            self.tool_widget_title,
            self.tool_widget_icon_name,
        )

    def _modestack_changed_cb(self, modestack, old, new):
        """Update the UI when the mode changes"""
        self._update_ui()

    def _update_ui(self):
        """Update the UI to show the options widget of the current mode"""
        mode = self._app.doc.modes.top
        self._update_ui_for_mode(mode)

    def _update_ui_for_mode(self, mode):
        # Get the new options widget
        try:
            get_options_widget = mode.get_options_widget
        except AttributeError:
            get_options_widget = None
        if get_options_widget:
            new_options = get_options_widget()
        else:
            new_options = self._no_options_label
        if not new_options:
            # Leave existing widget as-is, even if it's the default.
            # XXX maybe we should be doing something stack-based here?
            return
        icon_name = mode.get_icon_name()
        name = mode.get_name()
        self._update_ui_with_options_widget(new_options, name, icon_name)

    def _update_ui_with_options_widget(self, new_options, name, icon_name):
        old_options = self._options_bin.get_child()
        logger.debug("name: %r, icon name: %r", name, icon_name)
        if name:
            markup = self.OPTIONS_MARKUP.format(
                mode_name=lib.xml.escape(name),
            )
            self._options_label.set_markup(markup)
        if icon_name:
            self._mode_icon.set_from_icon_name(
                icon_name,
                Gtk.IconSize.SMALL_TOOLBAR,
            )
        # Options widget: only update if there's a change
        if new_options is not old_options:
            if old_options:
                old_options.hide()
                self._options_bin.remove(old_options)
            self._options_bin.add(new_options)
            new_options.show()
        self._options_bin.show_all()
