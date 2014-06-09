# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Graphical rendering helpers (splines, alpha checks, brush preview)"""

## Imports

import logging
logger = logging.getLogger(__name__)

import math
from numpy import array
import cairo

from lib.helpers import clamp

import gi
from gi.repository import GdkPixbuf
from gi.repository import Gdk

from lib.brush import Brush, BrushInfo
import lib.tiledsurface
from lib.pixbufsurface import render_as_pixbuf


## Module constants

_ALPHA_CHECK_SIZE = 16
_ALPHA_CHECK_COLOR_1 = (0.45, 0.45, 0.45)
_ALPHA_CHECK_COLOR_2 = (0.50, 0.50, 0.50)

_BRUSH_PREVIEW_POINTS = [
        # px,  py,   press, xtilt, ytilt  # px,  py,   press, xtilt, ytilt
        (0.00, 0.00,  0.00,  0.00, 0.00), (1.00, 0.05,  0.00, -0.06, 0.05),
        (0.10, 0.10,  0.20,  0.10, 0.05), (0.90, 0.15,  0.90, -0.05, 0.05),
        (0.11, 0.30,  0.90,  0.08, 0.05), (0.86, 0.35,  0.90, -0.04, 0.05),
        (0.13, 0.50,  0.90,  0.06, 0.05), (0.84, 0.55,  0.90, -0.03, 0.05),
        (0.17, 0.70,  0.90,  0.04, 0.05), (0.83, 0.75,  0.90, -0.02, 0.05),
        (0.25, 0.90,  0.20,  0.02, 0.00), (0.81, 0.95,  0.00,  0.00, 0.00),
        (0.41, 0.95,  0.00,  0.00, 0.00), (0.80, 1.00,  0.00,  0.00, 0.00),
    ]


## Drawing functions

def spline_4p(t, p_1, p0, p1, p2):
    """Interpolated point using a Catmull-Rom spline

    :param float t: Time parameter, between 0.0 and 1.0
    :param array p_1: Point p[-1]
    :param array p0: Point p[0]
    :param array p1: Point p[1]
    :param array p2: Point p[2]
    :returns: Interpolated point, between p0 and p1
    :rtype: array

    Used for a succession of points, this function makes smooth curves
    passing through all specified points, other than the first and last.
    For each pair of points, and their immediate predecessor and
    successor points, the `t` parameter should be stepped incrementally
    from 0 (for point p0) to 1 (for point p1).  See also:

    * `spline_iter()`
    * http://en.wikipedia.org/wiki/Cubic_Hermite_spline
    * http://stackoverflow.com/questions/1251438
    """
    return ( t*((2-t)*t - 1)    * p_1 +
            (t*t*(3*t - 5) + 2) * p0  +
            t*((4 - 3*t)*t + 1) * p1  +
            (t-1)*t*t           * p2   ) / 2


def spline_iter(tuples, double_first=True, double_last=True):
    """Converts an list of control point tuples to interpolatable arrays

    :param list tuples: Sequence of tuples of floats
    :param bool double_first: Repeat 1st point, putting it in the result
    :param bool double_last: Repeat last point, putting it in the result
    :returns: Iterator producing (p-1, p0, p1, p2)

    The resulting sequence of 4-tuples is intended to be fed into
    spline_4p().  The start and end points are therefore normally
    doubled, producing a curve that passes through them, along a vector
    aimed at the second or penultimate point respectively.

    """
    cint = [None, None, None, None]
    if double_first:
        cint[0:3] = cint[1:4]
        cint[3] = array(tuples[0])
    for ctrlpt in tuples:
        cint[0:3] = cint[1:4]
        cint[3] = array(ctrlpt)
        if None not in cint:
            yield cint
    if double_last:
        cint[0:3] = cint[1:4]
        cint[3] = array(tuples[-1])
        yield cint


def _variable_pressure_scribble(w, h, tmult):
    points = _BRUSH_PREVIEW_POINTS
    px, py, press, xtilt, ytilt = points[0]
    yield (10, px*w, py*h, 0.0, xtilt, ytilt)
    event_dtime = 0.005
    point_time = 0.1
    for p_1, p0, p1, p2 in spline_iter(points, True, True):
        dt = 0.0
        while dt < point_time:
            t = dt/point_time
            px, py, press, xtilt, ytilt = spline_4p(t, p_1, p0, p1, p2)
            yield (event_dtime, px*w, py*h, press, xtilt, ytilt)
            dt += event_dtime
    px, py, press, xtilt, ytilt = points[-1]
    yield (10, px*w, py*h, 0.0, xtilt, ytilt)


def render_brush_preview_pixbuf(brushinfo, max_edge_tiles=4):
    """Renders brush preview images

    :param lib.brush.BrushInfo brushinfo: settings to render
    :param int max_edge_tiles: Use at most this many tiles along an edge
    :returns: Preview image, at 128x128 pixels
    :rtype: GdkPixbuf

    This generates the preview image (128px icon) used for brushes which
    don't have saved ones. These include brushes picked from .ORA files
    where the parent_brush_name doesn't correspond to a brush in the
    user's MyPaint brushes - they're used as the default, and for the
    Auto button in the Brush Icon editor.

    Brushstrokes are inherently unpredictable in size, so the allowable
    area is grown until the brush fits or until the rendering becomes
    too big. `max_edge_tiles` limits this growth.
    """
    assert max_edge_tiles >= 1
    brushinfo = brushinfo.clone() # avoid capturing a ref
    brush = Brush(brushinfo)
    surface = lib.tiledsurface.Surface()
    N = lib.tiledsurface.N
    for size_in_tiles in range(1, max_edge_tiles):
        width = N * size_in_tiles
        height = N * size_in_tiles
        surface.clear()
        fg, spiral = _brush_preview_bg_fg(surface, size_in_tiles, brushinfo)
        brushinfo.set_color_rgb(fg)
        brush.reset()
        # Curve
        #cx = width/2.0
        #cy = height/2.0
        #r = width/3.0
        #if spiral:
        #    shape = _variable_pressure_spiral(cx, cy, r, size_in_tiles, 5)
        #else:
        #    shape = _variable_pressure_circle(cx, cy, r, size_in_tiles)
        shape = _variable_pressure_scribble(width, height, size_in_tiles)
        surface.begin_atomic()
        for dt, x, y, p, xt, yt in shape:
            brush.stroke_to(surface.backend, x, y, p, xt, yt, dt)
        surface.end_atomic()
        # Check rendered size
        tposs = surface.tiledict.keys()
        outside =             min({tx for tx,ty in tposs}) < 0
        outside = outside or (min({ty for tx,ty in tposs}) < 0)
        outside = outside or (max({tx for tx,ty in tposs}) >= size_in_tiles)
        outside = outside or (max({ty for tx,ty in tposs}) >= size_in_tiles)
        bbox = surface.get_bbox()
        if not outside:
            break
    # Convert to pixbuf at the right scale
    rect = (0, 0, width, height)
    pixbuf = render_as_pixbuf(surface, *rect, alpha=True)
    if max(width, height) != 128:
        interp = (GdkPixbuf.InterpType.NEAREST if max(width, height) < 128
                  else GdkPixbuf.InterpType.BILINEAR)
        pixbuf = pixbuf.scale_simple(128, 128, interp)
    # Composite over a checquered bg via Cairo: shows erases
    size = _ALPHA_CHECK_SIZE
    nchecks = int(128 / size)
    cairo_surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 128, 128)
    cr = cairo.Context(cairo_surf)
    render_checks(cr, size, nchecks)
    Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
    cr.paint()
    cairo_surf.flush()
    return Gdk.pixbuf_get_from_surface(cairo_surf, 0, 0, 128, 128)


def _brush_preview_bg_fg(surface, size_in_tiles, brushinfo):
    """Render the background for brush previews, return paint colour"""
    # The background colour represents the overall nature of the brush
    col1 = (0.85, 0.85, 0.80) # Boring grey, with a hint of paper-yellow
    col2 = (0.80, 0.80, 0.80) # Grey, but will appear blueish in contrast
    fgcol = (0.05, 0.15, 0.20)  # Hint ofcolour shows off HSV varier brushes
    spiral = False
    N = lib.tiledsurface.N
    fx = [
        ("eraser", # pink=rubber=eraser; red=danger
            (0.8, 0.7, 0.7),  # pink/red tones: pencil eraser/danger
            (0.75, 0.60, 0.60),
            False, fgcol ),
        ("colorize",
            (0.8, 0.8, 0.8),  # orange on gray
            (0.6, 0.6, 0.6),
            False, (0.6, 0.2, 0.0)),
        ("smudge",  # blue=water=wet, with some contrast
            (0.85, 0.85, 0.80),  # same as the regular paper colour
            (0.60, 0.60, 0.70),  # bluer (water, wet); more contrast
            True, fgcol),
        ]
    for cname, c1, c2, c_spiral, c_fg, in fx:
        if brushinfo.has_large_base_value(cname):
            col1 = c1
            col2 = c2
            fgcol = c_fg
            spiral = c_spiral
            break

    never_smudger = (brushinfo.has_small_base_value("smudge") and
                     brushinfo.has_only_base_value("smudge"))
    colorizer = brushinfo.has_large_base_value("colorize")

    if never_smudger and not colorizer:
        col2 = col1

    a = 1<<15
    col1_fix15 = [c*a for c in col1] + [a]
    col2_fix15 = [c*a for c in col2] + [a]
    for ty in range(0, size_in_tiles):
        tx_thres = max(0, size_in_tiles - ty - 1)
        for tx in range(0, size_in_tiles):
            topcol = col1_fix15
            botcol = col1_fix15
            if tx > tx_thres:
                topcol = col2_fix15
            if tx >= tx_thres:
                botcol = col2_fix15
            with surface.tile_request(tx, ty, readonly=False) as dst:
                if topcol == botcol:
                    dst[:] = topcol
                else:
                    for i in range(N):
                        dst[0:N-i, i, ...] = topcol
                        dst[N-i:N, i, ...] = botcol
    return fgcol, spiral


def render_checks(cr, size, nchecks):
    """Renders a checquerboard pattern to a cairo surface"""
    cr.set_source_rgb(*_ALPHA_CHECK_COLOR_1)
    cr.paint()
    cr.set_source_rgb(*_ALPHA_CHECK_COLOR_2)
    for i in xrange(0, nchecks):
        for j in xrange(0, nchecks):
            if (i+j) % 2 == 0:
                continue
            cr.rectangle(i*size, j*size, size, size)
            cr.fill()


## Test code

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import sys
    for myb_file in sys.argv[1:]:
        if not myb_file.lower().endswith(".myb"):
            logger.warning("Ignored %r: not a .myb file", myb_file)
            continue
        myb_fp = open(myb_file, 'r')
        myb_json = myb_fp.read()
        myb_fp.close()
        myb_brushinfo = BrushInfo(myb_json)
        myb_pixbuf = render_brush_preview_pixbuf(myb_brushinfo)
        if myb_pixbuf is not None:
            myb_basename = myb_file[:-4]
            png_file = "%s_autopreview.png" % (myb_file,)
            logger.info("Saving to %r...", png_file)
            myb_pixbuf.savev(png_file, "png", [], [])

