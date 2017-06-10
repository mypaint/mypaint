# This file is part of MyPaint.
# Copyright (C) 2007-2011 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function

from warnings import warn
import logging
logger = logging.getLogger(__name__)

from gi.repository import Gtk, Gdk

RADIUS = 2


class CurveWidget(Gtk.DrawingArea):
    """Widget for modifying a (restricted) nonlinear curve.
    """
    __gtype_name__ = 'CurveWidget'
    _SNAP_TO = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0)
    _WHINED_ABOUT_ALPHA = False

    def __init__(self, changed_cb=None, magnetic=True, npoints=None,
                 ylockgroups=()):
        Gtk.DrawingArea.__init__(self)
        self.points = [(0.0, 0.2), (.25, .5), (.75, .75), (1.0, 1.0)]  # doesn't matter
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
        return self._npoints

    @npoints.setter
    def npoints(self, n):
        self._npoints = n
        self.maxpoints = 64 if not n else n

    @property
    def ylock(self):
        warn("Deprecated, use ylockgroups instead", DeprecationWarning)
        return self._ylock

    @property
    def ylockgroups(self):
        return keys(self._ylock)

    @ylockgroups.setter
    def ylockgroups(self, ylockgroups):
        self._ylock.clear()
        for items in ylockgroups:
            for thisitem in items:
                others = list(items)
                others.remove(thisitem)
                self._ylock[thisitem] = tuple(others)

    def eventpoint(self, event_x, event_y):
        width, height = self.get_display_area()
        x, y = event_x, event_y
        x -= RADIUS
        y -= RADIUS
        x = x / width
        y = y / height
        return (x, y)

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
        if index in self.ylock:
            for lockedto in self.ylock[index]:
                self.points[lockedto] = (self.points[lockedto][0], y)

    def button_press_cb(self, widget, event):
        if not event.button == 1:
            return
        x, y = self.eventpoint(event.x, event.y)
        nearest = None
        for i in range(len(self.points)):
            px, py = self.points[i]
            dist = abs(px - x) + 0.5*abs(py - y)
            if nearest is None or dist < mindist:
                mindist = dist
                nearest = i
        if not self.npoints and mindist > 0.05 and len(self.points) < self.maxpoints:
            insertpos = 0
            for i in range(len(self.points)):
                if self.points[i][0] < x:
                    insertpos = i + 1
            if insertpos > 0 and insertpos < len(self.points):
                if y > 1.0:
                    y = 1.0
                if y < 0.0:
                    y = 0.0
                self.points.insert(insertpos, (x, y))
                # XXX and update ylockgroups?
                #
                nearest = insertpos
                self.queue_draw()

        #if nearest == 0:
        #    # first point cannot be grabbed
        #    display = gdk.display_get_default()
        #    display.beep()
        #else:
        #assert self.grabbed is None # This did happen. I think it's save to ignore?
        # I guess it's because gtk can generate button press event without corresponding release.
        self.grabbed = nearest

    def button_release_cb(self, widget, event):
        if not event.button == 1:
            return
        if self.grabbed:
            i = self.grabbed
            if self.points[i] is None:
                self.points.pop(i)
        self.grabbed = None
        # notify user of the widget
        self.changed_cb(self)

    def motion_notify_cb(self, widget, event):
        if self.grabbed is None:
            return
        x, y = self.eventpoint(event.x, event.y)
        i = self.grabbed
        # XXX this may fail for non contiguous groups.
        if i in self.ylock:
            possiblei = None
            if x > self.points[max(self.ylock[i])][0]:
                possiblei = max((i,) + self.ylock[i])
            elif x < self.points[min(self.ylock[i])][0]:
                possiblei = min((i,) + self.ylock[i])
            if (possiblei is not None and
                    abs(self.points[i][0] - self.points[possiblei][0]) < 0.001):
                i = possiblei
        out = False  # by default, the point cannot be removed by drawing it out
        if i == len(self.points)-1:
            # last point stays right
            leftbound = rightbound = 1.0
        elif i == 0:
            # first point stays left
            leftbound = rightbound = 0.0
        else:
            # other points can be dragged out
            if not self.npoints and (y > 1.1 or y < -0.1):
                out = True
            leftbound = self.points[i-1][0]
            rightbound = self.points[i+1][0]
            if not self.npoints and (x <= leftbound - 0.02 or x >= rightbound + 0.02):
                out = True
        if out:
            self.points[i] = None
        else:
            if y > 1.0:
                y = 1.0
            if y < 0.0:
                y = 0.0
            if self.magnetic:
                xdiff = [abs(x - v) for v in self._SNAP_TO]
                ydiff = [abs(y - v) for v in self._SNAP_TO]
                if min(xdiff) < 0.015 and min(ydiff) < 0.015:
                    y = self._SNAP_TO[ydiff.index(min(ydiff))]
                    x = self._SNAP_TO[xdiff.index(min(xdiff))]
            if x < leftbound:
                x = leftbound
            if x > rightbound:
                x = rightbound
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
            return (c.red, c.green, c.blue)
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

        # back to regular weight
        cr.set_line_width(1.0)

        if self.graypoint:
            x1, y1 = self.graypoint
            x1 = int(x1*width) + RADIUS
            y1 = int(y1*height) + RADIUS
            cr.rectangle(x1-RADIUS-1+0.5, y1-RADIUS-1+0.5, 2*RADIUS+1, 2*RADIUS+1)
            cr.fill()

        cr.set_source_rgb(*fg)

        # draw points
        current_x = current_y = prev_x = prev_y = 0
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


if __name__ == '__main__':
    logging.basicConfig()
    win = Gtk.Window()
    curve = CurveWidget()
    curve.ylockgroups = [(1, 2), (3, 4)]
    curve.npoints = 6
    curve.points = [(0., 0.), (.2, .5), (.4, .75), (.6, .5), (.8, .3), (1., 1.)]
    win.add(curve)
    win.set_title("curve test")
    win.connect("destroy", lambda *a: Gtk.main_quit())
    win.show_all()
    Gtk.main()
