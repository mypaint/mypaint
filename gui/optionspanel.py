# This file is part of MyPaint.
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Dockable panel showing options for the current mode"""

## Imports

import workspace
from lib.helpers import escape

from gettext import gettext as _

import gi
from gi.repository import Gtk

import logging
logger = logging.getLogger(__name__)


## Class defs

class ModeOptionsTool (workspace.SizedVBoxToolWidget):
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

    SIZED_VBOX_NATURAL_HEIGHT = workspace.TOOL_WIDGET_NATURAL_HEIGHT_SHORT

    tool_widget_icon_name = "mypaint-brush-mods-active"
    tool_widget_title = _("Tool Options")
    tool_widget_description = _("Specialized settings for the current "
                                "editing tool")

    __gtype_name__ = 'MyPaintModeOptionsTool'

    OPTIONS_MARKUP = _("<b>%s</b>")
    NO_OPTIONS_MARKUP = _("No options available")

    ## Method defs

    def __init__(self):
        """Construct, and connect internal signals & observers"""
        workspace.SizedVBoxToolWidget.__init__(self)
        from application import get_app
        self._app = get_app()
        self._app.doc.modes.observers.append(self._mode_changed_cb)
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
        label_hbox.pack_start(self._mode_icon, False, False)
        label_hbox.pack_start(self._options_label, True, True)
        align = Gtk.Alignment(0.5, 0.5, 1.0, 1.0)
        align.set_padding(0, 0, 0, 0)
        align.set_border_width(3)
        self._options_bin = align
        self.pack_start(label_hbox, False, False, 0)
        self.pack_start(align, True, True, 0)
        self.connect("show", lambda *a: self._update_ui())

    def _mode_changed_cb(self, mode):
        """Update the UI when the mode changes"""
        self._update_ui()

    def _update_ui(self):
        """Update the UI to show the options widget of the current mode"""
        mode = self._app.doc.modes.top
        # Get the new options widget
        old_options = self._options_bin.get_child()
        new_options = self._no_options_label
        if hasattr(mode, "get_options_widget"):
            new_options = mode.get_options_widget()
        # Only update if there's a change
        if new_options and new_options is not old_options:
            # Label
            markup = self.OPTIONS_MARKUP % (escape(mode.get_name()),)
            self._options_label.set_markup(markup)
            # Icon
            icon_name = mode.get_icon_name()
            self._mode_icon.set_from_icon_name(icon_name,
                                               Gtk.IconSize.SMALL_TOOLBAR)
            # Options widget
            if old_options:
                old_options.hide()
                self._options_bin.remove(old_options)
            self._options_bin.add(new_options)
            new_options.show()
        self._options_bin.show_all()

