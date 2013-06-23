# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk2compat

import gtk
from gtk import gdk
import cairo
import math
from cStringIO import StringIO

# Absolute minimum size
BRUSH_CURSOR_MIN_SIZE = 3

# Cursor style constants
BRUSH_CURSOR_STYLE_NORMAL = 0
BRUSH_CURSOR_STYLE_ERASER = 1
BRUSH_CURSOR_STYLE_LOCK_ALPHA = 2
BRUSH_CURSOR_STYLE_COLORIZE = 3

# cache only the last cursor
last_cursor_info = None
last_cursor = None
max_cursor_size = None


def get_brush_cursor(radius, style, prefs={}):
    """Returns a gdk.Cursor for use with a brush of a particular size+type.
    """
    global last_cursor, last_cursor_info, max_cursor_size

    display = gtk2compat.gdk.display_get_default()
    if not max_cursor_size:
        max_cursor_size = max(display.get_maximal_cursor_size())
    d = int(radius*2)
    min_size = max(prefs.get("cursor.freehand.min_size", 4),
                   BRUSH_CURSOR_MIN_SIZE)
    if d < min_size:
        d = min_size
    if d+1 > max_cursor_size:
        d = max_cursor_size-1
    cursor_info = (d, style, min_size)
    if cursor_info != last_cursor_info:
        last_cursor_info = cursor_info
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, d+1, d+1)
        cr = cairo.Context(surf)
        cr.set_source_rgba(1, 1, 1, 0)
        cr.paint()
        draw_brush_cursor(cr, d, style, prefs)
        surf.flush()

        # Calculate hotspot. Zero means topmost or leftmost. Cursors with an
        # even pixel diameter are interesting because they can never be
        # perfectly centred on their hotspot. Rounding down and not up may be
        # more "arrow-cursor" like in this case.
        #
        # NOTE: is it worth adjusting the freehand drawing code to add half a
        # pixel to the position passed on to brushlib for the even case?
        hot_x = hot_y = int(d/2)

        pixbuf = image_surface_to_pixbuf(surf)
        if gtk2compat.USE_GTK3:
            last_cursor = gdk.Cursor.new_from_pixbuf(display, pixbuf,
                                                     hot_x, hot_y)
        else:
            last_cursor = gdk.Cursor(display, pixbuf, hot_x, hot_y)

    return last_cursor


def image_surface_to_pixbuf(surf):
    if gtk2compat.USE_GTK3:
        w = surf.get_width()
        h = surf.get_height()
        return gdk.pixbuf_get_from_surface(surf, 0, 0, w, h)
    # GTK2.
    # Ugly PNG-based fudge to work around cairo.ImageSurface.get_data()
    # returning packed BGRA or ARGB pixels depending on endianness.
    png_dp = StringIO()
    surf.write_to_png(png_dp)
    pixbuf_loader = gdk.PixbufLoader()
    pixbuf_loader.write(png_dp.getvalue())
    pixbuf_loader.close()
    return pixbuf_loader.get_pixbuf()


def draw_brush_cursor(cr, d, style=BRUSH_CURSOR_STYLE_NORMAL, prefs={}):
    """Draw a brush cursor into a Cairo context, assumed to be of w=h=d+1
    """

    # The cursor consists of an inner circle drawn over an outer one. The
    # inmost edge of the inner ring, and the outmode edge of the outer ring are
    # pixel-edge aligned. If vertical bars represent pixel edges, then
    # conceptually,
    #
    #   |<--------------    Integer radius
    #   |OOOOOO             Outer ring pixels
    #   |------->|          Integer inset
    #       IIIII|          Inner ring pixels
    #
    # Which results in the cleanest possible edges inside and outside. Note the
    # overlap, and that line widths don't have to be integers.

    # Outer and inner line widths
    width1 = float(prefs.get("cursor.freehand.outer_line_width", 1.25))
    width2 = float(prefs.get("cursor.freehand.inner_line_width", 1.25))
    inset = int(prefs.get("cursor.freehand.inner_line_inset", 2))

    # Colors
    col_bg = tuple(prefs.get("cursor.freehand.outer_line_color", (0,0,0,1)))
    col_fg = tuple(prefs.get("cursor.freehand.inner_line_color", (1,1,1,0.75)))

    # Cursor style
    arcs = []
    if style == BRUSH_CURSOR_STYLE_ERASER:
        # divide into eighths, alternating on and off
        k = math.pi / 4
        k2 = k/2
        arcs.append((k2,     k2+k))
        arcs.append((k2+2*k, k2+3*k))
        arcs.append((k2+4*k, k2+5*k))
        arcs.append((k2+6*k, k2+7*k))
    elif style == BRUSH_CURSOR_STYLE_LOCK_ALPHA:
        # same thing, but the two side voids are filled
        k = math.pi/4
        k2 = k/2
        arcs.append((k2+6*k, k2+k))
        arcs.append((k2+2*k, k2+5*k))
    elif style == BRUSH_CURSOR_STYLE_COLORIZE:
        # same as lock-alpha, but with the voids turned through 90 degrees
        k = math.pi/4
        k2 = k/2
        arcs.append((k2,     k2+3*k))
        arcs.append((k2+4*k, k2+7*k))
    else:
        # Regular drawing mode
        arcs.append((0, 2*math.pi))

    # Pick centre to ensure pixel alignedness for the outer edge of the
    # black outline.
    if d%2 == 0:
        r0 = int(d/2)
    else:
        r0 = int(d/2) + 0.5
    cx = cy = r0

    # Outer "bg" line.
    cr.set_line_cap(cairo.LINE_CAP_BUTT)
    cr.set_source_rgba(*col_bg)
    cr.set_line_width(width1)
    r = r0 - (width1 / 2.0)
    for a1, a2 in arcs:
        cr.new_sub_path()
        cr.arc(cx, cy, r, a1, a2)
    cr.stroke()

    # Inner line: also pixel aligned, but to its inner edge.
    cr.set_line_cap(cairo.LINE_CAP_ROUND)
    cr.set_source_rgba(*col_fg)
    cr.set_line_width(width2)
    r = r0 - inset + (width2 / 2.0)
    for a1, a2 in arcs:
        cr.new_sub_path()
        cr.arc(cx, cy, r, a1, a2)
    cr.stroke()



if __name__ == '__main__':
    from random import randint
    win = gtk.Window()
    win.set_title("cursor test")

    min_size = 5
    max_size = 64
    nsteps = 8
    w = nsteps * max_size
    h = 4 * max_size
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    cr = cairo.Context(surf)
    cr.set_source_rgb(.7, .7, .7)
    cr.paint()

    for style in xrange(4):
        col = 0
        for size in xrange(min_size, max_size+1, (max_size-min_size)/nsteps):
            cr.save()
            y = (style * max_size) + ((max_size - size)/2)
            x = (col * max_size) + ((max_size - size)/2)
            cr.translate(x, y)
            draw_brush_cursor(cr, size, style)
            cr.restore()
            col += 1
    pixbuf = image_surface_to_pixbuf(surf)
    image = gtk.Image()
    image.set_from_pixbuf(pixbuf)
    image.set_size_request(w, h)

    display = gtk2compat.gdk.display_get_default()
    max_size = max(display.get_maximal_cursor_size())
    num_styles = 4
    style = 0
    def _enter_cb(widget, event):
        global style, max_size
        r = randint(3, max_size/2)
        e = False
        l = False
        c = False
        print "DEBUG: radius=%s, style=%s" % (r, style)
        cursor = get_brush_cursor(r, style)
        widget.get_window().set_cursor(cursor)
        style += 1
        if style >= num_styles:
            style = 0
    win.connect("enter-notify-event", _enter_cb)
    win.add(image)
    win.connect("destroy", lambda *a: gtk.main_quit())
    win.show_all()
    gtk.main()
