# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team
# Copyright (C) 2009-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function
import cairo
import math
import logging

import gui.drawutils
from lib.pycompat import xrange

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GdkPixbuf

logger = logging.getLogger(__name__)

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


class Name:
    """Cursor name constants

    These correspond to the names of small (22x22px) png files in the
    installation's pixmaps directory.

    """

    ARROW = "cursor_arrow"
    ARROW_MOVE = "cursor_arrow_move"
    PENCIL = "cursor_pencil"
    HAND_OPEN = "cursor_hand_open"
    HAND_CLOSED = "cursor_hand_closed"
    CROSSHAIR_OPEN = "cursor_crosshair_open"
    CROSSHAIR_OPEN_PRECISE = "cursor_crosshair_precise_open"
    CROSSHAIR_CLOSED = "cursor_crosshair_closed"
    INVERTED_CROSSHAIR_DIAGONAL = "cursor_crosshair_diagonal"
    MOVE_WEST_OR_EAST = "cursor_move_w_e"
    MOVE_NORTHWEST_OR_SOUTHEAST = "cursor_move_nw_se"
    MOVE_NORTH_OR_SOUTH = "cursor_move_n_s"
    MOVE_NORTHEAST_OR_SOUTHWEST = "cursor_move_ne_sw"
    FORBIDDEN_EVERYWHERE = "cursor_forbidden_everywhere"
    ARROW_FORBIDDEN = "cursor_arrow_forbidden"
    REMOVE = "cursor_remove"
    ADD = "cursor_add"
    PICKER = "cursor_color_picker"
    ERASER = "cursor_eraser"
    ALPHA_LOCK = "cursor_alpha_lock"
    COLORIZE = "cursor_colorize"


def get_brush_cursor(radius, style, prefs=None):
    """Get a cursor for use with a brush of a particular size and type

    :param float radius: Radius of the ring
    :param int style: A cursor style constant (for now, see the source)
    :param dict prefs: User preferences dict

    """
    prefs = prefs or {}

    global last_cursor, last_cursor_info, max_cursor_size

    display = Gdk.Display.get_default()
    if not max_cursor_size:
        # get_maximal_cursor_size returns a 2-tuple with max width/height
        max_cursor_size = max(display.get_maximal_cursor_size())
    min_size_prefs = prefs.get("cursor.freehand.min_size", 4)
    min_size = max(min_size_prefs, BRUSH_CURSOR_MIN_SIZE)
    # Adjust cursor size for dynamic crosshair, if enabled
    threshold = None
    if prefs.get("cursor.dynamic_crosshair", False):
        threshold = int(prefs.get("cursor.dynamic_crosshair_threshold", 8))
    # calculate initial cursor dimensions
    d = max(int(radius * 2), min_size)
    # and visual radius based on the initial dimensions
    r = min(max_cursor_size - 1, d) // 2
    # if the circle radius is smaller than the crosshair threshold,
    # the size is set so the cursor can contains the crosshair
    if threshold and threshold > r:
        d = threshold * 2
    d = min(d, max_cursor_size - 1)
    cursor_info = (r, style, min_size, threshold)
    if cursor_info != last_cursor_info:
        last_cursor_info = cursor_info
        dim = d + 1
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, dim, dim)
        cr = cairo.Context(surf)
        cr.set_source_rgba(1, 1, 1, 0)
        cr.paint()
        draw_brush_cursor(cr, r, dim/2, style, prefs, threshold)
        surf.flush()

        # Calculate hotspot. Zero means topmost or leftmost. Cursors with an
        # even pixel diameter are interesting because they can never be
        # perfectly centred on their hotspot. Rounding down and not up may be
        # more "arrow-cursor" like in this case.
        #
        # NOTE: is it worth adjusting the freehand drawing code to add half a
        # pixel to the position passed on to brushlib for the even case?
        hot_x = hot_y = int(dim // 2)

        pixbuf = _image_surface_to_pixbuf(surf)
        last_cursor = Gdk.Cursor.new_from_pixbuf(display, pixbuf,
                                                 hot_x, hot_y,)
    return last_cursor


def _image_surface_to_pixbuf(surf):
    """Convert a Cairo surface to a GdkPixbuf"""
    w = surf.get_width()
    h = surf.get_height()
    return Gdk.pixbuf_get_from_surface(surf, 0, 0, w, h)


def draw_brush_cursor(
        cr, radius, cc,
        style=BRUSH_CURSOR_STYLE_NORMAL, prefs=None, threshold=None):
    """Draw a brush cursor into a Cairo context

    :param cr: A Cairo context
    :param radius: The radius of the brush cursor circle
    :param cc: The center of the cursor (single value)
    :param int style: A cursor style constant (for now, see the source)
    :param dict prefs: User preferences dict
    :param threshold: if set, draw crosshair lines if the brush circle radius
           is smaller than the threshold value.

    The Cairo context is assumed to have width and height `d`+1, at
    least.

    """
    prefs = prefs or {}

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
    col_bg = tuple(prefs.get(
        "cursor.freehand.outer_line_color",
        (0, 0, 0, 1),
    ))
    col_fg = tuple(prefs.get(
        "cursor.freehand.inner_line_color",
        (1, 1, 1, 0.75),
    ))

    # Cursor style
    arcs = cursor_arc_segments(style)

    cx = cy = cc
    if threshold:
        lines = cursor_line_segments(style, radius, threshold)
    else:
        lines = []

    # Outer "bg" line.
    cr.set_line_cap(cairo.LINE_CAP_BUTT)
    cr.set_source_rgba(*col_bg)
    cr.set_line_width(width1)
    r_outer = radius - (width1 / 2.0)
    for a1, a2 in arcs:
        cr.new_sub_path()
        cr.arc(cx, cy, r_outer, a1, a2)
    cr.stroke()

    # Inner line: also pixel aligned, but to its inner edge.
    cr.set_line_cap(cairo.LINE_CAP_ROUND)
    cr.set_source_rgba(*col_fg)
    cr.set_line_width(width2)
    r_inner = radius - inset + (width2 / 2.0)
    for a1, a2 in arcs:
        cr.new_sub_path()
        cr.arc(cx, cy, r_inner, a1, a2)
    cr.stroke()

    # Crosshair lines, if enabled and radius is below threshold
    if lines:
        def stroke_lines():
            for (x_offs_0, y_offs_0), (x_offs_1, y_offs_1) in lines:
                cr.new_sub_path()
                cr.move_to(cx + x_offs_0, cy + y_offs_0)
                cr.line_to(cx + x_offs_1, cy + y_offs_1)
            cr.stroke()

        cr.set_line_cap(cairo.LINE_CAP_BUTT)
        cr.set_source_rgba(*col_bg)
        cr.set_line_width(3)
        stroke_lines()

        cr.set_line_cap(cairo.LINE_CAP_SQUARE)
        cr.set_source_rgba(*col_fg)
        cr.set_line_width(1)
        stroke_lines()


def cursor_line_segments(style, r, threshold):
    """Returns tuples of crosshair line coordinates for the given style"""
    if r >= threshold:
        return []

    offs = math.ceil(threshold - r)

    if style == BRUSH_CURSOR_STYLE_ERASER:
        k = math.cos(math.pi / 4)
        offs1 = r * k
        offs2 = (r + offs) * k
        return [
            ((-offs1, -offs1), (-offs2, -offs2)),
            ((offs1, -offs1), (offs2, -offs2)),
            ((offs1, offs1), (offs2, offs2)),
            ((-offs1, offs1), (-offs2, offs2)),
        ]
    elif style == BRUSH_CURSOR_STYLE_LOCK_ALPHA:
        return [((-r - offs, 0), (-r, 0)), ((r, 0), (r + offs, 0))]
    elif style == BRUSH_CURSOR_STYLE_COLORIZE:
        return [((0, -r - offs), (0, -r)), ((0, r), (0, r + offs))]
    else:
        return [
            ((-r - offs, 0), (-r, 0)), ((r, 0), (r + offs, 0)),
            ((0, -r - offs), (0, -r)), ((0, r), (0, r + offs)),
        ]


def cursor_arc_segments(style):
    """Returns a list of arc segments based on style - pairs of radians"""
    k = math.pi / 4
    k2 = k / 2
    if style == BRUSH_CURSOR_STYLE_ERASER:
        # divide into eighths, alternating on and off
        return [
            (k2, k2 + k),
            (k2 + 2 * k, k2 + 3 * k),
            (k2 + 4 * k, k2 + 5 * k),
            (k2 + 6 * k, k2 + 7 * k)
        ]
    elif style == BRUSH_CURSOR_STYLE_LOCK_ALPHA:
        # same thing, but the two side voids are filled
        return [(k2 + 6 * k, k2 + k), (k2 + 2 * k, k2 + 5 * k)]
    elif style == BRUSH_CURSOR_STYLE_COLORIZE:
        # same as lock-alpha, but with the voids turned through 90 degrees
        return [(k2, k2 + 3 * k), (k2 + 4 * k, k2 + 7 * k)]
    else:
        # Regular drawing mode
        return [(0, 2 * math.pi)]


class CustomCursorMaker (object):
    """Factory and cache of custom cursors for actions."""

    CURSOR_HOTSPOTS = {
        Name.ARROW: (1, 1),
        Name.ARROW_MOVE: (1, 1),
        Name.PENCIL: (7, 22),
        Name.HAND_OPEN: (11, 12),
        Name.HAND_CLOSED: (11, 12),
        Name.CROSSHAIR_OPEN: (11, 11),
        Name.CROSSHAIR_CLOSED: (11, 11),
        Name.CROSSHAIR_OPEN_PRECISE: (12, 11),
        Name.INVERTED_CROSSHAIR_DIAGONAL: (11, 11),
        Name.MOVE_WEST_OR_EAST: (11, 11),
        Name.MOVE_NORTHWEST_OR_SOUTHEAST: (11, 11),
        Name.MOVE_NORTH_OR_SOUTH: (11, 11),
        Name.MOVE_NORTHEAST_OR_SOUTHWEST: (11, 11),
        Name.FORBIDDEN_EVERYWHERE: (11, 11),
        Name.ARROW_FORBIDDEN: (7, 4),
        Name.REMOVE: (11, 11),
        Name.ADD: (11, 11),
        Name.PICKER: (3, 15),
        Name.ERASER: (12, 11),
        Name.ALPHA_LOCK: (12, 11),
        Name.COLORIZE: (12, 11),
    }

    def __init__(self, app):
        object.__init__(self)
        self.app = app
        self.cache = {}

    def _get_overlay_cursor(self, icon_pixbuf, cursor_name=Name.ARROW):
        """Returns an overlay cursor. Not cached.

        :param icon_pixbuf: a GdkPixbuf.Pixbuf containing a small (~22px)
           image, or None
        :param cursor_name: name of a pixmaps/ cursor image to use for the
           pointer part, minus the .png

        The overlay icon will be overlaid to the bottom and right of the
        returned cursor image.

        """

        pointer_pixbuf = getattr(self.app.pixmaps, cursor_name)
        pointer_w = pointer_pixbuf.get_width()
        pointer_h = pointer_pixbuf.get_height()
        hot_x, hot_y = self.CURSOR_HOTSPOTS.get(cursor_name, (None, None))
        if hot_x is None:
            hot_x = 1
            hot_y = 1

        cursor_pixbuf = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, True,
                                             8, 32, 32)
        cursor_pixbuf.fill(0x00000000)

        pointer_pixbuf.composite(
            cursor_pixbuf, 0, 0, pointer_w, pointer_h, 0, 0, 1, 1,
            GdkPixbuf.InterpType.NEAREST, 255
        )
        if icon_pixbuf is not None:
            icon_w = icon_pixbuf.get_width()
            icon_h = icon_pixbuf.get_height()
            icon_x = 32 - icon_w
            icon_y = 32 - icon_h
            icon_pixbuf.composite(
                cursor_pixbuf, icon_x, icon_y, icon_w, icon_h,
                icon_x, icon_y, 1, 1, GdkPixbuf.InterpType.NEAREST, 255
            )

        display = self.app.drawWindow.get_display()
        cursor = Gdk.Cursor.new_from_pixbuf(display, cursor_pixbuf,
                                            hot_x, hot_y)
        return cursor

    def get_freehand_cursor(self, cursor_name=Name.CROSSHAIR_OPEN_PRECISE):
        """Returns a cursor for the current app.brush. Cached.

        :param cursor_name: name of a pixmaps/ image to use, minus the .png

        An icon for the brush blend mode will be overlaid to the bottom and
        right of the cursor image.

        """
        # Pick an icon
        if self.app.brush.is_eraser():
            icon_name = "mypaint-eraser-symbolic"
        elif self.app.brush.is_alpha_locked():
            icon_name = "mypaint-lock-alpha-symbolic"
        elif self.app.brush.is_colorize():
            icon_name = "mypaint-colorize-symbolic"
        else:
            icon_name = None
        return self.get_icon_cursor(icon_name, cursor_name)

    def get_action_cursor(self, action_name, cursor_name=Name.ARROW):
        """Returns an overlay cursor for a named action. Cached.

        :param action_name: the name of a GtkAction defined in resources.xml
        :param cursor_name: name of a pixmaps/ image to use, minus the .png

        The action's icon will be overlaid at a small size to the bottom and
        right of the cursor image.

        """
        # Find a small action icon for the overlay
        action = self.app.find_action(action_name)
        if action is None:
            return Gdk.Cursor.new(Gdk.CursorType.ARROW)
        icon_name = action.get_icon_name()
        if icon_name is None:
            return Gdk.Cursor.new(Gdk.CursorType.ARROW)
        return self.get_icon_cursor(icon_name, cursor_name)

    def get_icon_cursor(self, icon_name, cursor_name=Name.ARROW):
        """Returns an overlay cursor for a named icon. Cached.

        :param icon_name: themed icon system name.
        :param cursor_name: name of a pixmaps/ image to use, minus the .png

        The icon will be overlaid at a small size to the bottom and right of
        the cursor image.

        """

        # Return from cache, if we have an entry
        cache_key = ("actions", icon_name, cursor_name)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if icon_name is not None:
            if "symbolic" in icon_name:
                icon_pixbuf = gui.drawutils.load_symbolic_icon(
                    icon_name, 18,
                    fg=(1, 1, 1, 1),
                    outline=(0, 0, 0, 1),
                )
            else:
                # Look up icon via the user's current theme
                icon_theme = Gtk.IconTheme.get_default()
                size_range = [Gtk.IconSize.SMALL_TOOLBAR, Gtk.IconSize.MENU]
                for icon_size in size_range:
                    valid, width, height = Gtk.icon_size_lookup(icon_size)
                    if not valid:
                        continue
                    size = min(width, height)
                    if size > 24:
                        continue
                    flags = 0
                    icon_pixbuf = icon_theme.load_icon(icon_name, size, flags)
                    if icon_pixbuf:
                        break
                if not icon_pixbuf:
                    logger.warning(
                        "Can't find icon %r for cursor. Search path: %r",
                        icon_name,
                        icon_theme.get_search_path(),
                    )
        else:
            icon_pixbuf = None

        # Build cursor
        cursor = self._get_overlay_cursor(icon_pixbuf, cursor_name)

        # Cache and return
        self.cache[cache_key] = cursor
        return cursor


def get_move_cursor_name_for_angle(angle):
    """Cursor name to use for a motion in a given direction

    To get the cursor appropriate for edge adjustments, provide the angle
    perpendicular to the angle of the edge.

      >>> f = get_move_cursor_name_for_angle
      >>> pi = math.pi
      >>> offs = pi / 16 - 0.01
      >>> (Name.MOVE_WEST_OR_EAST
      ...  == f(0) == f(-offs) == f(offs))
      True
      >>> (Name.MOVE_NORTHEAST_OR_SOUTHWEST
      ...  == f(pi/4) == f(pi/4 - offs) == f(pi/4 + offs))
      True
      >>> (Name.MOVE_NORTH_OR_SOUTH
      ...  == f(pi/2) == f(pi/2 - offs) == f(pi/2 + offs))
      True
      >>> (Name.MOVE_NORTHWEST_OR_SOUTHEAST
      ...  == f(3 * pi/4) == f(3 * pi/4 - offs) == f(3 * pi/4 + offs))
      True
    """
    return [
        Name.MOVE_WEST_OR_EAST,
        Name.MOVE_NORTHEAST_OR_SOUTHWEST,
        Name.MOVE_NORTH_OR_SOUTH,
        Name.MOVE_NORTHWEST_OR_SOUTHEAST,
    ][int(round((angle % math.pi) / (math.pi / 4))) % 4]


# Interactive testing

if __name__ == '__main__':
    from random import randint
    win = Gtk.Window()
    win.set_title("cursor test")

    _min_size = 2
    _max_size = 64
    nsteps = 8
    w = nsteps * _max_size
    h = 4 * _max_size
    _surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    _cr = cairo.Context(_surf)
    _cr.set_source_rgb(.7, .7, .7)
    _cr.paint()

    for _style in xrange(4):
        col = 0
        for size in xrange(_min_size, _max_size + 1,
                           (_max_size - _min_size) // nsteps):
            _cr.save()
            y = (_style * _max_size) + ((_max_size - size) / 2)
            x = (col * _max_size) + ((_max_size - size) / 2)
            _cr.translate(x, y)
            draw_brush_cursor(_cr, size//2, size//2, _style)
            _cr.restore()
            col += 1
    pixbuf = _image_surface_to_pixbuf(_surf)
    image = Gtk.Image()
    image.set_from_pixbuf(pixbuf)
    image.set_size_request(w, h)

    _max_size = max(Gdk.Display.get_default().get_maximal_cursor_size())
    num_styles = 4
    _style = 0

    def _enter_cb(widget, event):
        global _style, _max_size
        r = randint(3, _max_size // 2)
        print("DEBUG: radius=%s, style=%s" % (r, _style))
        cursor = get_brush_cursor(r, _style)
        widget.get_window().set_cursor(cursor)
        _style += 1
        if _style >= num_styles:
            _style = 0
    win.connect("enter-notify-event", _enter_cb)
    win.add(image)
    win.connect("destroy", lambda *a: Gtk.main_quit())
    win.show_all()
    Gtk.main()
