# This file is part of MyPaint.
# Copyright (C) 2012 by Ali Lown <ali@lown.me.uk>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gettext import gettext as _
import gtk, gobject, pango
gdk = gtk.gdk
import dialogs
import tileddrawwidget


class ToolWidget(gtk.VBox):
    """Tool widget for previewing the whole canvas"""

    stock_id = "mypaint-tool-preview-window"
    tool_widget_title = _("Preview")

    def __init__(self, app):
        gtk.VBox.__init__(self)
        self.set_size_request(250, 250)

        self.tdw = tileddrawwidget.TiledDrawWidget(app, app.doc.model)
        self.tdw.set_size_request(250, 250)
        #TODO: perhaps configure this based on used area?
        self.tdw.scale = 0.03
        self.tdw.set_sensitive(False)
        self.add(self.tdw)
