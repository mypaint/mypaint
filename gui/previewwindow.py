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

        self.doc = app.doc.model
        self.tdw = tileddrawwidget.TiledDrawWidget(app, self.doc)
        self.tdw.zoom_min = 1/50.0
        self.tdw.set_size_request(250, 250)
        self.tdw.set_sensitive(False)
        self.add(self.tdw)
        self.doc.canvas_observers.append(self.doc_modified_cb)

    def doc_modified_cb(self, x, y, w, h):
      winx, winy = self.size_request()

      #Calculate new zoom level
      rect = self.doc.get_bbox()
      fw = rect.w
      fh = rect.h

      if fw == 0 or fh == 0:
        return

      zoom_x = float(winx) / fw
      zoom_y = float(winy) / fh

      scale = min(zoom_x, zoom_y)

      self.tdw.scale = scale
      self.tdw.recenter_document()
