# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Miscellaneous helpers & drawing utility functions for Cairo.
"""

from __future__ import division, print_function
import math

from lib.pycompat import xrange


def clamp(v, bottom, top):
    """Returns `v`, clamped to within a particular range.
    """
    if v > top:
        return top
    if v < bottom:
        return bottom
    return v


def add_distance_fade_stops(gr, rgb, nstops=3, gamma=2, alpha=1.0):
    """Adds rgba stops to a Cairo gradient approximating a power law fade.

    The stops have even spacing between the 0 and 1 positions, and alpha
    values diminishing from 1 to 0. When `gamma` is greater than 1, the
    generated fades or glow diminishes faster than a linear gradient. This
    seems to reduce halo artifacts on some LCD backlit displays.
    """
    red, green, blue = rgb
    nstops = int(nstops) + 2
    for s in xrange(nstops+1):
        a = alpha * (((nstops - s) / nstops) ** gamma)
        stop = s / nstops
        gr.add_color_stop_rgba(stop, red, green, blue, a)


def draw_marker_circle(cr, x, y, size=2):
    """Draws an outlined circular marker.
    """
    cr.save()
    cr.set_source_rgb(0, 0, 0)
    cr.set_line_width(size+2)
    cr.arc(x, y, (2*size)+0.5, 0, 2*math.pi)
    cr.stroke_preserve()
    cr.set_source_rgb(1, 1, 1)
    cr.set_line_width(size)
    cr.stroke()
    cr.restore()
