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

# caching only the last cursor
last_cursor_info = None
last_cursor = None
max_cursor_size = None

def get_brush_cursor(radius, is_eraser, is_lockalpha):
    global last_cursor, last_cursor_info, max_cursor_size

    display = pygtkcompat.gdk.display_get_default()
    if not max_cursor_size:
        max_cursor_size = max(display.get_maximal_cursor_size())

    d = int(radius)*2
    if d < 3: d = 3
    if is_eraser and d < 8: d = 8
    if d+1 > max_cursor_size:
        d = max_cursor_size-1
    cursor_info = (d, is_eraser, is_lockalpha)
    if cursor_info != last_cursor_info:
        last_cursor_info = cursor_info

        if pygtkcompat.USE_GTK3:
            # Drawing constants
            border = 2
            width1 = 4
            width2 = 2
            cx = cy = int(d+1)/2
            w = h = d+1
            r = int((d-width1)/2)

            surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
            cr = cairo.Context(surf)
            cr.set_line_width(width1)
            cr.set_source_rgb(0, 0, 0)
            arcs = []
            if is_eraser:
                # divide into eighths, alternating on and off
                k = math.pi / 4
                k2 = k/2
                arcs.append((k2,     k2+k))
                arcs.append((k2+2*k, k2+3*k))
                arcs.append((k2+4*k, k2+5*k))
                arcs.append((k2+6*k, k2+7*k))
            elif is_lockalpha:
                # same thing, but the two side voids are filled
                k = math.pi/4
                k2 = k/2
                arcs.append((k2+6*k, k2+k))
                arcs.append((k2+2*k, k2+5*k))
            else:
                # Regular drawing mode
                arcs.append((0, 2*math.pi))
            cr.set_line_cap(cairo.LINE_CAP_ROUND)
            for a1, a2 in arcs:
                cr.new_sub_path()
                cr.arc(cx, cy, r, a1, a2)
            cr.stroke_preserve()
            cr.set_line_width(width2)
            cr.set_source_rgb(1, 1, 1)
            cr.stroke()
            pixbuf = gdk.pixbuf_get_from_surface(surf, 0, 0, d+1, d+1)
            last_cursor = gdk.Cursor.new_from_pixbuf(display, pixbuf,
                                int((d+1)/2), int((d+1)/2))

        else:
            # GTK2/PyGTK
            cursor = gdk.Pixmap(None, d+1, d+1,1)
            mask   = gdk.Pixmap(None, d+1, d+1,1)
            colormap = gdk.colormap_get_system()
            black = colormap.alloc_color('black')
            white = colormap.alloc_color('white')

            bgc = cursor.new_gc(foreground=black)
            wgc = cursor.new_gc(foreground=white)
            cursor.draw_rectangle(bgc, True, 0, 0, d+1, d+1)
            cursor.draw_arc(wgc,False, 0, 0, d, d, 0, 360*64)

            bgc = mask.new_gc(foreground=black)
            wgc = mask.new_gc(foreground=white)
            mask.draw_rectangle(bgc, True, 0, 0, d+1, d+1)
            mask.draw_arc(wgc, False, 0, 0, d, d, 0, 360*64)
            mask.draw_arc(wgc, False, 1, 1, d-2, d-2, 0, 360*64)

            if is_eraser:
                thickness = d/8
                mask.draw_rectangle(bgc, True, d/2-thickness, 0, 2*thickness+1, d+1)
                mask.draw_rectangle(bgc, True, 0, d/2-thickness, d+1, 2*thickness+1)
            elif is_lockalpha:
                thickness = int(d/4+0.5)
                mask.draw_rectangle(bgc, True, d/2-thickness, 0, 2*thickness+1, d+1)

            last_cursor = gdk.Cursor(cursor,mask,gdk.color_parse('black'), gdk.color_parse('white'),(d+1)/2,(d+1)/2)

    return last_cursor


if __name__ == '__main__':
    from random import randint
    win = gtk.Window()
    win.set_title("cursor test")
    label = gtk.Label("Hover and enter/leave to set cursor")
    label.set_size_request(400, 200)
    def _enter_cb(widget, event):
        m = max_cursor_size
        if m is None:
            m = 16
        r = randint(3, m/2)
        e = False
        l = False
        style = randint(1, 3)
        if style == 2:
            e = True
        elif style == 3:
            l = True
        print "DEBUG: radius=%s, eraser=%s, lockalpha=%s" % (r,e,l)
        cursor = get_brush_cursor(r, e, l)
        widget.get_window().set_cursor(cursor)
    win.connect("enter-notify-event", _enter_cb)
    win.add(label)
    win.connect("destroy", lambda *a: gtk.main_quit())
    win.show_all()
    gtk.main()
