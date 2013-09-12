# This file is part of MyPaint.
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Flood fill tool"""

## Imports

import gi
from gi.repository import Gdk
from gettext import gettext as _

import canvasevent

from application import get_app

## Class defs

class FloodFillMode (canvasevent.SwitchableModeMixin,
                     canvasevent.ScrollableModeMixin,
                     canvasevent.SingleClickMode):
    """Mode for flood-filling with the current brush color"""

    ## Class constants

    __action_name__ = "FloodFillMode"
    permitted_switch_actions = set([
        'RotateViewMode', 'ZoomViewMode', 'PanViewMode',
        'ColorPickMode',
        ])


    ## Instance vars (and defaults)

    _inside_frame = True
    _x = None
    _y = None

    @property
    def cursor(self):
        app = get_app()
        if self._inside_frame:
            name = "cursor_crosshair_precise_open"
        else:
            name = "cursor_arrow_forbidden"
        return app.cursors.get_action_cursor(self.__action_name__, name)


    ## Method defs

    def enter(self, **kwds):
        super(FloodFillMode, self).enter(**kwds)
        self._tdws = set()

    @classmethod
    def get_name(cls):
        return _(u'Flood Fill')

    def get_usage(self):
        return _(u"Click to fill with the current color")

    def __init__(self, ignore_modifiers=False, **kwds):
        super(FloodFillMode, self).__init__(**kwds)
    
    def clicked_cb(self, tdw, event):
        x, y = tdw.display_to_model(event.x, event.y)
        self._x = x
        self._y = y
        self._tdws.add(tdw)
        self._update_cursor()
        color = self.doc.app.brush_color_manager.get_color()
        self.doc.model.flood_fill(x, y, color.get_rgb())
        return False

    def motion_notify_cb(self, tdw, event):
        x, y = tdw.display_to_model(event.x, event.y)
        self._x = x
        self._y = y
        self._tdws.add(tdw)
        self._update_cursor()
        return super(FloodFillMode, self).motion_notify_cb(tdw, event)

    def model_structure_changed_cb(self, doc):
        super(FloodFillMode, self).model_structure_changed_cb(doc)
        self._update_cursor()

    def _update_cursor(self):
        x, y = self._x, self._y
        model = self.doc.model
        was_inside = self._inside_frame
        if not model.frame_enabled:
            self._inside_frame and not model.layer.is_empty()
        else:
            fx1, fy1, fw, fh = model.get_frame()
            fx2, fy2 = fx1+fw, fy1+fh
            self._inside_frame = (x >= fx1 and y >= fy1 and
                                  x < fx2 and y < fy2)
        if was_inside != self._inside_frame:
            for tdw in self._tdws:
                tdw.set_override_cursor(self.cursor)
