# This file is part of MyPaint.
# Copyright (C) 2011-2018 by the MyPaint Development Team.
# Copyright (C) 2011 by Ben O'Steen <bosteen@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Scratchpad panel"""

## Imports

from __future__ import division, print_function
import logging

from lib.gettext import gettext as _
from .toolstack import SizedVBoxToolWidget, TOOL_WIDGET_NATURAL_HEIGHT_SHORT
from .widgets import inline_toolbar

from lib.gibindings import Gtk

logger = logging.getLogger(__name__)


## Class defs

class ScratchpadTool (SizedVBoxToolWidget):

    __gtype_name__ = 'MyPaintScratchpadTool'

    SIZED_VBOX_NATURAL_HEIGHT = TOOL_WIDGET_NATURAL_HEIGHT_SHORT

    tool_widget_title = _("Scratchpad")
    tool_widget_icon_name = 'mypaint-scratchpad-symbolic'
    tool_widget_description = _("Mix colors and make sketches on "
                                "separate scrap pages")

    def __init__(self):
        super(SizedVBoxToolWidget, self).__init__()
        from gui.application import get_app
        app = get_app()
        self.app = app
        toolbar = inline_toolbar(
            app, [
                ("ScratchNew", "mypaint-add-symbolic"),
                ("ScratchLoad", None),
                ("ScratchSaveAs", "mypaint-document-save-symbolic"),
                ("ScratchRevert", "mypaint-document-revert-symbolic"),
            ])
        scratchpad_view = app.scratchpad_doc.tdw
        scratchpad_view.set_size_request(64, 64)
        self.connect("destroy-event", self._save_cb)
        self.connect("delete-event", self._save_cb)
        scratchpad_box = Gtk.EventBox()
        scratchpad_box.add(scratchpad_view)
        self.pack_start(scratchpad_box, True, True, 0)
        self.pack_start(toolbar, False, True, 0)

    def _save_cb(self, action):
        filename = self.app.scratchpad_filename
        logger.info("Saving the scratchpad to %r", filename)
        self.app.filehandler.save_scratchpad(filename)
