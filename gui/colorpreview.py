# This file is part of MyPaint.
# Copyright (C) 2013-2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or


"""Color preview widget / current color indicator, for the status bar."""

# TODO: This *might* evolve to a color preview + alpha selector, possibly
# TODO:   with a history row taking up the bottom. For now let's draw it at
# TODO:   an aspect ratio of about 1:5 and see how users like it.

from __future__ import division, print_function

from .colors import PreviousCurrentColorAdjuster


class BrushColorIndicator (PreviousCurrentColorAdjuster):
    """Previous/Current color adjuster bound to app.brush_color_manager"""

    __gtype_name__ = "MyPaintBrushColorIndicator"

    HAS_DETAILS_DIALOG = False

    def __init__(self):
        PreviousCurrentColorAdjuster.__init__(self)
        self.connect("realize", self._init_color_manager)
        self._app = None
        self.clicked += self._clicked_cb

    def _init_color_manager(self, widget):
        from gui.application import get_app
        self._app = get_app()
        mgr = self._app.brush_color_manager
        assert mgr is not None
        self.set_color_manager(mgr)

    def _clicked_cb(self, adj, event, pos):
        x0, y0 = pos
        w = self.get_allocated_width()
        if x0 > w // 2:
            return
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
