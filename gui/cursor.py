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

from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GdkPixbuf

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


def get_brush_cursor(radius, style, prefs={}):
    """Get a cursor for use with a brush of a particular size and type

    :param float radius: Radius of the ring
    :param int style: A cursor style constant (for now, see the source)
    :param dict prefs: User preferences dict

    """
    global last_cursor, last_cursor_info, max_cursor_size

    display = Gdk.Display.get_default()
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
        hot_x = hot_y = int(d // 2)

        pixbuf = _image_surface_to_pixbuf(surf)
        last_cursor = Gdk.Cursor.new_from_pixbuf(display, pixbuf,
                                                 hot_x, hot_y,)
    return last_cursor


def _image_surface_to_pixbuf(surf):
    """Convert a Cairo surface to a GdkPixbuf"""
    w = surf.get_width()
    h = surf.get_height()
    return Gdk.pixbuf_get_from_surface(surf, 0, 0, w, h)


def draw_brush_cursor(cr, d, style=BRUSH_CURSOR_STYLE_NORMAL, prefs={}):
    """Draw a brush cursor into a Cairo context

    :param cr: A Cairo context
    :param d: The diameter of the resultant brush cursor
    :param int style: A cursor style constant (for now, see the source)
    :param dict prefs: User preferences dict

    The Cairo context is assumed to have width and height `d`+1, at
    least.

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
    col_bg = tuple(prefs.get(
        "cursor.freehand.outer_line_color",
        (0, 0, 0, 1),
    ))
    col_fg = tuple(prefs.get(
        "cursor.freehand.inner_line_color",
        (1, 1, 1, 0.75),
    ))

    # Cursor style
    arcs = []
    if style == BRUSH_CURSOR_STYLE_ERASER:
        # divide into eighths, alternating on and off
        k = math.pi / 4
        k2 = k/2
        arcs.append((k2, k2+k))
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
        arcs.append((k2, k2+3*k))
        arcs.append((k2+4*k, k2+7*k))
    else:
        # Regular drawing mode
        arcs.append((0, 2*math.pi))

    # Pick centre to ensure pixel alignedness for the outer edge of the
    # black outline.
    if d % 2 == 0:
        r0 = int(d // 2)
    else:
        r0 = int(d // 2) + 0.5
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


def get_move_cursor_name_for_angle(theta):
    """Cursor name to use for a motion in a given direction"""
    while theta < 2*math.pi:
        theta += 2*math.pi
    theta %= 2*math.pi
    assert theta >= 0
    assert theta < 2*math.pi
    cursor_strs = [
        (1, Name.MOVE_WEST_OR_EAST),
        (3, Name.MOVE_NORTHWEST_OR_SOUTHEAST),
        (5, Name.MOVE_NORTH_OR_SOUTH),
        (7, Name.MOVE_NORTHEAST_OR_SOUTHWEST),
        (9, Name.MOVE_WEST_OR_EAST),
        (11, Name.MOVE_NORTHWEST_OR_SOUTHEAST),
        (13, Name.MOVE_NORTH_OR_SOUTH),
        (15, Name.MOVE_NORTHEAST_OR_SOUTHWEST),
        (17, Name.MOVE_WEST_OR_EAST),
    ]
    cursor_str = None
    for i, s in cursor_strs:
        if theta < i*(2.0/16)*math.pi:
            cursor_str = s
            break
    assert cursor_str is not None
    return cursor_str


## Interactive testing

if __name__ == '__main__':
    from random import randint
    win = Gtk.Window()
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
        for size in xrange(min_size, max_size + 1,
                           (max_size - min_size) // nsteps):
            cr.save()
            y = (style * max_size) + ((max_size - size)/2)
            x = (col * max_size) + ((max_size - size)/2)
            cr.translate(x, y)
            draw_brush_cursor(cr, size, style)
            cr.restore()
            col += 1
    pixbuf = _image_surface_to_pixbuf(surf)
    image = Gtk.Image()
    image.set_from_pixbuf(pixbuf)
    image.set_size_request(w, h)

    display = Gdk.Display.get_default()
    max_size = max(display.get_maximal_cursor_size())
    num_styles = 4
    style = 0

    def _enter_cb(widget, event):
        global style, max_size
        r = randint(3, max_size // 2)
        print("DEBUG: radius=%s, style=%s" % (r, style))
        cursor = get_brush_cursor(r, style)
        widget.get_window().set_cursor(cursor)
        style += 1
        if style >= num_styles:
            style = 0
    win.connect("enter-notify-event", _enter_cb)
    win.add(image)
    win.connect("destroy", lambda *a: Gtk.main_quit())
    win.show_all()
    Gtk.main()
