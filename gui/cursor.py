# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import pygtkcompat

import gtk
from gtk import gdk
import cairo
import math

# Module config
BRUSH_CURSOR_MIN_SIZE = 5
BRUSH_CURSOR_MIN_SIZE_BIG_SCREEN = 8

# Cursor style constants
BRUSH_CURSOR_STYLE_NORMAL = 0
BRUSH_CURSOR_STYLE_ERASER = 1
BRUSH_CURSOR_STYLE_LOCK_ALPHA = 2
BRUSH_CURSOR_STYLE_COLORIZE = 3

# cache only the last cursor
last_cursor_info = None
last_cursor = None
max_cursor_size = None
largest_screen_size = None


def get_largest_screen_size(display):
    global largest_screen_size
    if not largest_screen_size:
        largest_screen_size = 0
        for s in xrange(display.get_n_screens()):
            screen = display.get_screen(s)
            size = max(screen.get_width(), screen.get_height())
            if size > largest_screen_size:
                largest_screen_size = size
    return largest_screen_size


def get_brush_cursor(radius, style):
    """Returns a gdk.Cursor for use with a brush of a particular size+type.
    """
    global last_cursor, last_cursor_info, max_cursor_size

    display = pygtkcompat.gdk.display_get_default()
    if not max_cursor_size:
        max_cursor_size = max(display.get_maximal_cursor_size())
    d = int(radius)*2
    min_size = BRUSH_CURSOR_MIN_SIZE
    screen_size = get_largest_screen_size(display)
    if screen_size > 1024:
        min_size = BRUSH_CURSOR_MIN_SIZE_BIG_SCREEN
    if d < min_size:
        d = min_size
    if d+1 > max_cursor_size:
        d = max_cursor_size-1
    cursor_info = (d, style, screen_size)
    if cursor_info != last_cursor_info:
        last_cursor_info = cursor_info
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, d+1, d+1)
        cr = cairo.Context(surf)
        draw_brush_cursor(cr, d, screen_size, style)
        hot_x = hot_y = int((d+1)/2)
        if pygtkcompat.USE_GTK3:
            pixbuf = gdk.pixbuf_get_from_surface(surf, 0, 0, d+1, d+1)
            last_cursor = gdk.Cursor.new_from_pixbuf(display, pixbuf,
                                                     hot_x, hot_y)
        else:
            pixbuf = gdk.pixbuf_new_from_data(surf.get_data(),
                gdk.COLORSPACE_RGB, True, 8, d+1, d+1, surf.get_stride())
            last_cursor = gdk.Cursor(display, pixbuf, hot_x, hot_y)

    return last_cursor


def draw_brush_cursor(cr, d, screen_size, style=BRUSH_CURSOR_STYLE_NORMAL):
    # Draw a brush cursor into a Cairo context, assumed to be of w=h=d+1
    col_bg = (0, 0, 0)
    col_fg = (1, 1, 1)

    # Outer border width
    #if d >= 60:
    #    width1 = 6
    #    width2 = width1 - 2
    if d >= 50:
        width1 = 5
        width2 = width1 - 1.75
    elif d >= 30 or screen_size > 1024:
        width1 = 4
        width2 = width1 - 1.666
    else:
        width1 = 3
        width2 = width1 - 1.5

    #if screen_size > 1024:
    #    width1 += 1
    #    width2 += 1

    # Shadow size
    if d > 10:
        shadow_x = 1
        shadow_y = 1.5
    else:
        shadow_x = shadow_y = 0

    # Ensure pixel alignedness for the outer edges of the outer black border
    assert width1 == int(width1)
    if int(width1)%2 == 1:
        rdelta = 0.5
    else:
        rdelta = 0
    rint = int((d-max(shadow_y, shadow_x))/2)
    cx = cy = rint + rdelta

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

    # Shadow
    cr.set_line_cap(cairo.LINE_CAP_BUTT)
    cr.set_source_rgba(0, 0, 0, 0.2)
    cr.set_line_width(width1)
    r = rint - (width1 / 2)
    for a1, a2 in arcs:
        cr.new_sub_path()
        cr.arc(cx+shadow_x, cy+shadow_y, r, a1, a2)
    cr.stroke()

    # Outer borders
    cr.set_line_cap(cairo.LINE_CAP_BUTT)
    cr.set_source_rgb(*col_bg)
    cr.set_line_width(width1)
    r = rint - (width1 / 2)
    for a1, a2 in arcs:
        cr.new_sub_path()
        cr.arc(cx, cy, r, a1, a2)
    cr.stroke_preserve()

    # Inner line
    cr.set_line_cap(cairo.LINE_CAP_ROUND)
    cr.set_source_rgb(*col_fg)
    cr.set_line_width(width2)
    cr.stroke()



if __name__ == '__main__':
    from random import randint
    win = gtk.Window()
    win.set_title("cursor test")
    label = gtk.Label("Hover and enter/leave to set cursor")
    label.set_size_request(400, 200)
    num_styles = 4
    style = 0
    display = gdk.display_get_default()
    m = max(display.get_maximal_cursor_size())
    def _enter_cb(widget, event):
        global style, m
        r = randint(3, m/2)
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
    win.add(label)
    win.connect("destroy", lambda *a: gtk.main_quit())
    win.show_all()
    gtk.main()
