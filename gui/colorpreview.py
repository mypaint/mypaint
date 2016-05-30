# This file is part of MyPaint.
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or


"""Color preview widget / current color indicator, for the status bar."""

# TODO: This *might* evolve to a color preview + alpha selector, possibly
# TODO:   with a history row taking up the bottom. For now let's draw it at
# TODO:   an aspect ratio of about 1:5 and see how users like it.

from __future__ import division, print_function

from colors import PreviousCurrentColorAdjuster

from gi.repository import Gdk


class BrushColorIndicator (PreviousCurrentColorAdjuster):
    """Previous/Current color adjuster bound to app.brush_color_manager"""

    __gtype_name__ = "MyPaintBrushColorIndicator"

    HAS_DETAILS_DIALOG = False

    def __init__(self):
        PreviousCurrentColorAdjuster.__init__(self)
        self.connect("realize", self._init_color_manager)
        self.connect("button-press-event", self._button_press_cb)
        self.connect("button-release-event", self._button_release_cb)
        self._button = None
        self._app = None

    def _init_color_manager(self, widget):
        from application import get_app
        self._app = get_app()
        mgr = self._app.brush_color_manager
        assert mgr is not None
        self.set_color_manager(mgr)

    def _button_press_cb(self, widget, event):
        """Clicking on the current color side shows the quick color chooser"""
        if not self._app:
            return False
        if event.button != 1:
            return False
        if event.type != Gdk.EventType.BUTTON_PRESS:
            return False
        width = widget.get_allocated_width()
        if event.x > width // 2:
            return False
        self._button = event.button
        return True

    def _button_release_cb(self, widget, event):
        if event.button != self._button:
            return False
        self._button = None
        chooser = self._app.drawWindow.color_chooser
        if chooser.get_visible():
            chooser.hide()
        else:
            chooser.popup(
                widget=self,
                above=True,
                textwards=True,
                event=event,
            )
        return True
