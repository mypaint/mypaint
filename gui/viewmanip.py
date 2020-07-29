# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or


"""Modes for manipulating the view"""


## Imports
from __future__ import division, print_function

import gui.mode

import math

from gettext import gettext as _


## Class defs

class PanViewMode (gui.mode.ScrollableModeMixin, gui.mode.OneshotDragMode):
    """A mode for translating the viewport by dragging."""

    ACTION_NAME = 'PanViewMode'

    pointer_behavior = gui.mode.Behavior.CHANGE_VIEW
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW
    supports_button_switching = True

    permitted_switch_actions = set([
        'RotateViewMode',
        'ZoomViewMode'
    ] + gui.mode.BUTTON_BINDING_ACTIONS)

    @classmethod
    def get_name(cls):
        return _(u"Scroll View")

    def get_usage(self):
        return _(u"Drag the canvas view")

    @property
    def inactive_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME)

    @property
    def active_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME)

    def drag_update_cb(self, tdw, event, ev_x, ev_y, dx, dy):
        # NOTE: keep in line with gui.mode.ScrollableViewMixin
        tdw.scroll(-dx, -dy)
        self.doc.notify_view_changed()
        super(PanViewMode, self).drag_update_cb(
            tdw, event, ev_x, ev_y, dx, dy)


class ZoomViewMode (gui.mode.ScrollableModeMixin, gui.mode.OneshotDragMode):
    """A mode for zooming the viewport by dragging."""

    ACTION_NAME = 'ZoomViewMode'

    pointer_behavior = gui.mode.Behavior.CHANGE_VIEW
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW
    supports_button_switching = True

    permitted_switch_actions = set([
        'PanViewMode'
        'RotateViewMode',
    ] + gui.mode.BUTTON_BINDING_ACTIONS)

    @classmethod
    def get_name(cls):
        return _(u"Zoom View")

    def get_usage(self):
        return _(u"Zoom the canvas view")

    @property
    def active_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME)

    @property
    def inactive_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME)

    def drag_update_cb(self, tdw, event, ev_x, ev_y, dx, dy):
        tdw.scroll(-dx, -dy)
        tdw.zoom(math.exp(dy/100.0), center=(ev_x, ev_y))
        # TODO: Let modifiers constrain the zoom amount to
        # TODO: the defined steps. Shift seems to be conventional now...
        self.doc.notify_view_changed()
        super(ZoomViewMode, self).drag_update_cb(
            tdw, event, ev_x, ev_y, dx, dy)
        # TODO: The constrain amount and threshold should be similar to
        # TODO: what gui.mode.ScrollableModeMixin now does.


class RotateViewMode (gui.mode.ScrollableModeMixin, gui.mode.OneshotDragMode):
    """A mode for rotating the viewport by dragging."""

    ACTION_NAME = 'RotateViewMode'

    pointer_behavior = gui.mode.Behavior.CHANGE_VIEW
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW
    supports_button_switching = True

    permitted_switch_actions = set([
        'PanViewMode'
        'ZoomViewMode',
    ] + gui.mode.BUTTON_BINDING_ACTIONS)

    @classmethod
    def get_name(cls):
        return _(u"Rotate View")

    def get_usage(cls):
        return _(u"Rotate the canvas view")

    @property
    def active_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME)

    @property
    def inactive_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME)

    def drag_update_cb(self, tdw, event, ev_x, ev_y, dx, dy):
        # calculate angular velocity from the rotation center
        cx, cy = tdw.get_center()
        x, y = ev_x - cx, ev_y - cy
        phi2 = math.atan2(y, x)
        phi1 = math.atan2(y - dy, x - dx)
        tdw.rotate(phi2-phi1, center=(cx, cy))
        self.doc.notify_view_changed()
        # TODO: Allow modifiers to constrain the transformation angle
        #       to 22.5 degree steps.
        super(RotateViewMode, self).drag_update_cb(
            tdw, event, ev_x, ev_y, dx, dy)
