# This file is part of MyPaint.
# Copyright (C) 2007-2011 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2011-2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (C) 2014-2019 by The MyPaint Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function

import logging
from math import pi

from lib.gibindings import Gtk, Gdk

from lib.helpers import clamp

logger = logging.getLogger(__name__)

RADIUS = 2


class CurveWidget(Gtk.DrawingArea):
    """Widget for modifying a (restricted) nonlinear curve.
    """
    __gtype_name__ = 'CurveWidget'
    _SNAP_TO = tuple(float(n)/100 for n in range(0, 105, 5))
    _WHINED_ABOUT_ALPHA = False

    def __init__(self, changed_cb=None, magnetic=True, npoints=None,
                 ylockgroups=()):
        Gtk.DrawingArea.__init__(self)
        self.points = [(0.0, 0.2), (.25, .5), (.75, .75), (1.0, 1.0)]
        self._ylock = {}
        self.ylockgroups = ylockgroups

        self.maxpoints = None
        self._npoints = None
        self.npoints = npoints
        self.grabbed = None
        if changed_cb is None:
            self.changed_cb = lambda *a: None
        else:
            self.changed_cb = changed_cb
        self.magnetic = magnetic

        self.connect("draw", self.draw_cb)

        self.connect("button-press-event", self.button_press_cb)
        self.connect("button-release-event", self.button_release_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)
        self.set_events(Gdk.EventMask.EXPOSURE_MASK |
                        Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.BUTTON_RELEASE_MASK |
                        Gdk.EventMask.POINTER_MOTION_MASK
                        )
        self.set_size_request(300, 200)

        self.graypoint = None

    @property
    def npoints(self):
        """If the curve contains a fixed number of points, return that number.

        :return: The point count if the count is fixed, otherwise None
        :rtype: int | None
        """
        return self._npoints

    @npoints.setter
    def npoints(self, n):
        """Set the number of points for a fixed curve, or disable fixedness.

        :param n: The number of points, or None to disable
        :type n: int | None
        """
        self._npoints = n
        self.maxpoints = 64 if not n else n

    @property
    def ylockgroups(self):
        """Get a copy of the y-lock groups used by the curve.

        :return: Dictionary of index-> (i1, i2, ...) associations
        :rtype: dict
        """
        return {k: w for k, w in self._ylock.items()}

    @ylockgroups.setter
    def ylockgroups(self, ylockgroups):
        """Set y-lock groups from a list of index tuples.

        Each tuple should contain a up a set of indices that represent curve
        points that will share the same y-value. When the y-value of one point
        in a group is changed, all other points in that same group are changed.

        :param ylockgroups: List of tuples of indices
        :type ylockgroups: [(int, int, ..)]
        """
        self._ylock.clear()
        for group in ylockgroups:
            for idx in group:
                self._ylock[idx] = tuple(i for i in group if i != idx)

    def eventpoint(self, event_x, event_y):
        width, height = self.get_display_area()
        x, y = event_x, event_y
        x -= RADIUS
        y -= RADIUS
        x = x / width
        y = y / height
        return x, y

    def get_display_area(self):
        alloc = self.get_allocation()
        width, height = alloc.width, alloc.height
        width -= 2*RADIUS
        height -= 2*RADIUS
        width = width / 4 * 4
        height = height / 4 * 4
        return width, height

    def set_point(self, index, value):
        y = value[1]
        self.points[index] = value
        if index in self._ylock:
            for locked_to in self._ylock[index]:
                self.points[locked_to] = (self.points[locked_to][0], y)

    def button_press_cb(self, widget, event):
        if not (self.points or event.button == 1):
            return
        x, y = self.eventpoint(event.x, event.y)

        # Note: Squared distance used for comparisons
        def dist_squared(p):
            return abs(x - p[0])**2 + abs(y - p[1])**2
        points = self.points
        dsq, pos = min((dist_squared(p), i) for i, p in enumerate(points))

        # Unless the number of points are fixed, maxed out, or the intent
        # was to move an existing point, insert a new curve point.
        if not (self.npoints or dsq <= 0.003 or len(points) >= self.maxpoints):
            candidates = [i+1 for i, (px, _) in enumerate(points) if px < x]
            insert_pos = candidates and candidates[-1]
            if insert_pos and insert_pos < len(points):
                points.insert(insert_pos, (x, clamp(y, 0.0, 1.0)))
                pos = insert_pos
                self.queue_draw()

        self.grabbed = pos

    def button_release_cb(self, widget, event):
        if not event.button == 1:
            return
        if self.grabbed:
            if self.points[self.grabbed] is None:
                self.points.pop(self.grabbed)
        self.grabbed = None
        # notify user of the widget
        self.changed_cb(self)

    def motion_notify_cb(self, widget, event):
        if self.grabbed is None:
            return
        x, y = self.eventpoint(event.x, event.y)
        i = self.grabbed
        points = self.points
        # XXX this may fail for non contiguous groups.
        if i in self._ylock:
            i_candidate = None
            if x > points[max(self._ylock[i])][0]:
                i_candidate = max((i,) + self._ylock[i])
            elif x < points[min(self._ylock[i])][0]:
                i_candidate = min((i,) + self._ylock[i])
            if (i_candidate is not None and
                    abs(points[i][0] - points[i_candidate][0]) < 0.001):
                i = i_candidate
        out = False  # by default, points cannot be removed
        if i == len(points) - 1:
            # last point stays right
            left_bound = right_bound = 1.0
        elif i == 0:
            # first point stays left
            left_bound = right_bound = 0.0
        else:
            # other points can be dragged out
            left_bound = points[i-1][0]
            right_bound = points[i+1][0]
            margin = 0.02
            inside_x_bounds = left_bound - margin < x < right_bound + margin
            inside_y_bounds = -0.1 <= y <= 1.1
            out = not (self.npoints or (inside_x_bounds and inside_y_bounds))
        if out:
            points[i] = None
        else:
            y = clamp(y, 0.0, 1.0)
            if self.magnetic:
                x_diff = [abs(x - v) for v in self._SNAP_TO]
                y_diff = [abs(y - v) for v in self._SNAP_TO]
                if min(x_diff) < 0.015 and min(y_diff) < 0.015:
                    y = self._SNAP_TO[y_diff.index(min(y_diff))]
                    x = self._SNAP_TO[x_diff.index(min(x_diff))]
            x = clamp(x, left_bound, right_bound)
            self.set_point(i, (x, y))
        self.queue_draw()

    def draw_cb(self, widget, cr):

        def gdk2rgb(c):
            if c.alpha < 1 and not self.__class__._WHINED_ABOUT_ALPHA:
                logger.warning('The GTK3 style is reporting a color with '
                               'alpha less than 1. This should be harmless, '
                               'but please report any glitches with the curve '
                               'widget. Adwaita is thought not to suffer from '
                               'this problem.')
                self.__class__._WHINED_ABOUT_ALPHA = True
            return c.red, c.green, c.blue
        style = widget.get_style_context()
        state = widget.get_state_flags()
        fg = gdk2rgb(style.get_color(state))

        width, height = self.get_display_area()
        if width <= 0 or height <= 0:
            return

        # 1-pixel width, align lines with pixels
        # (but filled rectangles are off by 0.5px now)
        cr.translate(0.5, 0.5)

        # draw feint grid lines
        cr.set_line_width(0.333)
        cr.set_source_rgb(*fg)
        for i in range(11):
            cr.move_to(RADIUS, i*height/10 + RADIUS)
            cr.line_to(width + RADIUS, i*height/10 + RADIUS)
            cr.move_to(i*width/10 + RADIUS, RADIUS)
            cr.line_to(i*width/10 + RADIUS, height + RADIUS)
            cr.stroke()

        # The graypoint is represented by a dashed vertical line spanning the
        # entire graph, with a gap where a circle marks the vertical position.
        if self.graypoint:
            x1, y1 = self.graypoint
            x1 = int(x1*width) + RADIUS
            y1 = int(y1*height) + RADIUS
            cr.set_line_width(0.5)
            cr.set_dash([height/50.0])
            cr.move_to(x1, RADIUS)
            cr.line_to(x1, y1 - 2 * RADIUS)
            cr.move_to(x1, y1 + 2 * RADIUS)
            cr.line_to(x1, height + RADIUS)
            cr.stroke()
            cr.set_dash([])
            cr.arc(x1, y1, 2 * RADIUS, 0, 2*pi)
            cr.stroke()

        # back to regular weight
        cr.set_line_width(1.0)

        # draw points
        prev_x = prev_y = 0
        for p in self.points:
            if p is None:
                continue
            current_x = int(p[0] * width) + RADIUS
            current_y = int(p[1] * height) + RADIUS

            cr.rectangle(
                current_x-RADIUS-0.5, current_y-RADIUS-0.5,
                2*RADIUS+1, 2*RADIUS+1
            )
            cr.fill()

            # If it's the first point, we won't draw any lines yet
            if p is not self.points[0]:
                cr.move_to(prev_x, prev_y)
                cr.line_to(current_x, current_y)
                cr.stroke()

            prev_x, prev_y = current_x, current_y

        return True


def _test(case=1):
    logging.basicConfig()
    win = Gtk.Window()
    curve = CurveWidget()
    if case == 1:
        curve.ylockgroups = [(1, 2), (3, 4)]
        curve.npoints = 6
        curve.points = [
            (0., 0.), (.2, .5), (.4, .75), (.6, .5), (.8, .3), (1., 1.)
        ]
        curve.graypoint = (0.5, 0.0)
    win.add(curve)
    win.set_title("curve test")
    win.connect("destroy", lambda *a: Gtk.main_quit())
    win.show_all()
    Gtk.main()


if __name__ == '__main__':
    _test()
