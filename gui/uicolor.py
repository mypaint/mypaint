# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""UIColor conversion routines

These are functions which convert
our display color classes (see `lib.color.UIColor`)
to and from GDK's equivalents.
They can't be part of lib/ because of the GDK dependency.

"""

import struct

from lib.gibindings import Gdk

from lib.color import RGBColor
from lib.helpers import clamp


def from_gdk_color(gdk_color: Types.ELLIPSIS) -> Types.NONE:
    """Construct a new UIColor from a Gdk.Color.

    Args:
        gdk_color: 

    Returns:

    Raises:

    >>> from_gdk_color(Gdk.Color(0.0000, 0x8000, 0xffff))
    <RGBColor r=0.0000, g=0.5000, b=1.0000>
    """
    rgb16 = (gdk_color.red, gdk_color.green, gdk_color.blue)
    return RGBColor(*[c / 65535 for c in rgb16])


def to_gdk_color(color: Types.ELLIPSIS) -> Types.NONE:
    """Convert a UIColor to a Gdk.Color.

    Args:
        color: 

    Returns:

    Raises:

    >>> gcol = to_gdk_color(RGBColor(1,1,1))
    >>> gcol.to_string()
    '#ffffffffffff'
    """
    return Gdk.Color(*[int(c * 65535) for c in color.get_rgb()])


def from_gdk_rgba(gdk_rgba: Types.ELLIPSIS) -> Types.NONE:
    """Construct a new UIColor from a `Gdk.RGBA` (omitting alpha)

    Args:
        gdk_rgba: 

    Returns:

    Raises:

    >>> from_gdk_rgba(Gdk.RGBA(0.5, 0.8, 0.2, 1))
    <RGBColor r=0.5000, g=0.8000, b=0.2000>
    """
    rgbflt = (gdk_rgba.red, gdk_rgba.green, gdk_rgba.blue)
    return RGBColor(*[clamp(c, 0.0, 1.0) for c in rgbflt])


def to_gdk_rgba(color: Types.ELLIPSIS) -> Types.NONE:
    """Convert to a `GdkRGBA` (with alpha=1.0).

    Args:
        color: 

    Returns:

    Raises:

    >>> col = RGBColor(1,1,1)
    >>> rgba = to_gdk_rgba(col)
    >>> rgba.to_string()
    'rgb(255,255,255)'
    """
    rgba = list(color.get_rgb())
    rgba.append(1.0)
    return Gdk.RGBA(*rgba)


def from_drag_data(bytes: Types.ELLIPSIS) -> Types.NONE:
    """Construct from drag+dropped bytes of type application/x-color.
    
    The data format is 8 bytes, RRGGBBAA, with assumed native endianness.
    Alpha is ignored.

    Args:
        bytes: 

    Returns:

    Raises:

    """
    r, g, b, a = [h / 0xFFFF for h in struct.unpack("=HHHH", bytes)]
    return RGBColor(r, g, b)
    # TODO: check endianness


def to_drag_data(color: Types.ELLIPSIS) -> Types.NONE:
    """Converts to bytes for dragging as application/x-color.

    Args:
        color: 

    Returns:

    Raises:

    """
    rgba = [int(c * 0xFFFF) for c in color.get_rgb()]
    rgba.append(0xFFFF)
    return struct.pack("=HHHH", *rgba)


def _test():
    """ """
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    _test()
