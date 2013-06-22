# coding=utf-8

# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Colour objects and transformation functions.

Colour objects are lightweight polymorphic structures which can be cloned and
freely substituted no matter what subtype: they all offer methods for getting
RGB or HSV triples, and can all be constructed from any other type of colour
object via a common interface. This gets around some of the UI awkwardnesses of
supporting multiple colour spaces: only when you update a central shared colour
with an adjuster does its type change to match the control's colour space.

"""

# TODO: Convert code to GTK3.
# TODO: Simplify the HCY implementation. KDE's is nice (see kcolorspaces.cpp).
# TODO: Move all GTK code elsewhere, strip down to GUI-free code.
# TODO: Move this module to lib/ (keep the name, since it'll be UI-agnostic)
# TODO:   - required to support moving palette.py.


## Imports

import re
from colorsys import *
import struct

import gtk
from gtk import gdk

from util import clamp


##
## Lightweight colour objects
##

class UIColor (object):
    """Base class for colour objects which can be manipulated via the UI.

    This base provides a common interface allowing concrete subclasses to be
    instantiated from other UIColors: this provides a mechanism for conversions
    between colour models by mixed systems of user interface components as
    needed. Colour objects are typically instantiated individually by
    specifying their components, but they may be constructed from other
    `UIColor` objects too:

      >>> col1 = RGBColor(r=0.1, g=0.33, b=0.5)
      >>> col2 = HSVColor(color=col1)

    Subclasses must implement `get_rgb()`, and a ``color`` keyword to
    their constructor which takes as one of its arguments a `UIColor` object.

    """

    def get_rgb(self):
        """Extracts a floating-point R,G,B tuple representation of the colour.

        This is unimplemented at this level, but is required by most
        conversions. Subclasses are required to define this function, which
        must return a floating-point ``(r, g, b)`` representation of the colour
        with the channel samples lying in the range 0.0 and 1.0 inclusive.

        """
        raise NotImplementedError


    def get_hsv(self):
        """Extracts a floating-point H,S,V tuple representation of the colour.

        All terms in the returned ``(h, s, v)`` triple must be scaled to lie in
        the range 0.0 to 1.0 inclusive, compatible with `colorsys`. At this
        level, the operation is defined using an invocation of `get_rgb()`, but
        this behaviour can be overridden by subclasses.

        """
        return rgb_to_hsv(*self.get_rgb())


    ## Read-only properties (at this level)
    #rgb = property(get_rgb)
    #hsv = property(get_hsv)

    # RGB read-only
    @property
    def r(self):
        """Read-only RGB red value."""
        return self.get_rgb()[0]
    @r.setter
    def r(self, n): raise NotImplementedError

    @property
    def g(self):
        """Read-only RGB green value."""
        return self.get_rgb()[1]
    @g.setter
    def g(self, n): raise NotImplementedError

    @property
    def b(self):
        """Read-only RGB blue value."""
        return self.get_rgb()[2]
    @b.setter
    def b(self, n): raise NotImplementedError


    # HSV read-only
    @property
    def h(self):
        """Read-only hue angle."""
        return self.get_hsv()[0]
    @h.setter
    def h(self, n): raise NotImplementedError

    @property
    def s(self):
        """Read-only HSV saturation."""
        return self.get_hsv()[1]
    @s.setter
    def s(self, n): raise NotImplementedError

    @property
    def v(self):
        """Read-only HSV value."""
        return self.get_hsv()[2]
    @v.setter
    def v(self, n): raise NotImplementedError


    # Utility methods

    def get_luma(self):
        """Returns a perceptually-weighted brightness suitable for drawing.

          >>> col = RGBColor(0.6, 0.6, 0.6)
          >>> col.get_luma()
          0.6

        The weightings are fixed and rather generic, and the conversion relies
        on a potential RGB conversion. Thus, it's better to use a UIColor
        subclass if this needs to be manipulated. In particular, there's no
        way of just setting the luma defined at this level.

        """
        r, g, b = self.get_rgb()
        return 0.299*r + 0.587*g + 0.114*b


    def to_greyscale(self):
        """Returns a greyscaled version of the colour.

          >>> col = RGBColor(r=1.0, g=0.8, b=0.2)
          >>> col = col.to_greyscale()
          >>> min(col.get_rgb()) == max(col.get_rgb())
          True

        Based on `get_luma()`, so the same caveats apply. The returned object
        is itself a `UIColor`.

        """
        luma = self.get_luma()
        return RGBColor(luma, luma, luma)


    def to_contrasting(self, k=0.333):
        """Returns a contrasting `UIColor` suitable for drawing.

          >>> col = RGBColor(r=1.0, g=0.8, b=0.2)
          >>> col != col.to_contrasting()
          True

        """
        luma = self.get_luma()
        c = (luma + k) % 1.0
        return RGBColor(c, c, c)


    def __eq__(self, col):
        """Two colour objects are equal if their RGB form is equal.
        """
        # Round to 24bit for comparison
        rgb1 = [int(c * 0xff) for c in self.get_rgb()]
        try:
            rgb2 = [int(c * 0xff) for c in col.get_rgb()]
        except AttributeError:
            return False
        return rgb1 == rgb2
        ## colorhistory.py uses
        # a_ = numpy.array(helpers.hsv_to_rgb(*a))
        # b_ = numpy.array(helpers.hsv_to_rgb(*b))
        # return ((a_ - b_)**2).sum() < (3*1.0/256)**2


    def __copy__(self):
        """Clones the object using its own constructor; see `copy.copy()`.
        """
        color_class = type(self)
        return color_class(color=self)


    def __deepcopy__(self, memo):
        """Clones the object using its own constructor; see `copy.deepcopy()`.
        """
        color_class = type(self)
        return color_class(color=self)


    @staticmethod
    def new_from_gdk_color(gdk_color):
        """Construct a new `UIColor` from a `gdk.Color`.
        """
        rgb16 = (gdk_color.red, gdk_color.green, gdk_color.blue)
        return RGBColor(*[float(c)/65535 for c in rgb16])


    def to_gdk_color(self):
        """Convert to a `gdk.Color`.
        """
        return gdk.Color(*[int(c*65535) for c in self.get_rgb()])


    @staticmethod
    def new_from_gdk_rgba(gdk_rgba):
        """Construct a new `UIColor` from a ``GdkRGBA`` (omitting alpha)
        """
        rgbflt = (gdk_rgba.red, gdk_rgba.green, gdk_rgba.blue)
        return RGBColor(*[clamp(c, 0., 1.) for c in rgbflt])


    def to_gdk_rgba(self):
        """Convert to a `GdkRGBA` (with alpha=1.0).
        """
        rgba = list(self.get_rgb())
        rgba.append(1.0)
        return gdk.RGBA(*rgba)


    __HEX_PARSE_TABLE = [
      (re.compile('^(?:#|0x)' + '([0-9a-fA-F]{2})' * 3 + '$'), 0xff ),
      (re.compile('^(?:#|0x)' + '([0-9a-fA-F])' * 3    + '$'), 0xf  ),  ]


    @classmethod
    def new_from_hex_str(class_, hex_str, default=[0.5, 0.5, 0.5]):
        """Construct from an RGB hex string, e.g. ``#ff0000``.
        """
        hex_str = str(hex_str)
        r, g, b = default
        for pr, pd in class_.__HEX_PARSE_TABLE:
            m = pr.match(hex_str)
            if m:
                r, g, b = [float.fromhex(x)/pd for x in m.groups()]
                break
        return RGBColor(r, g, b)


    def to_hex_str(self, prefix='#'):
        """Converts to an RGB hex string of the form ``#RRGGBB``
        """
        r, g, b = [int(c * 0xff) for c in self.get_rgb()]
        return "%s%02x%02x%02x" % (prefix, r, g, b)


    @classmethod
    def new_from_drag_data(class_, bytes):
        """Construct from drag+dropped bytes of type application/x-color.

        The data format is 8 bytes, RRGGBBAA, with assumed native endianness.
        Alpha is ignored.
        """
        r,g,b,a = [float(h)/0xffff for h in struct.unpack("=HHHH", bytes)]
        return RGBColor(r, g, b)
        # TODO: check endianness


    def to_drag_data(self):
        """Converts to bytes for dragging as application/x-color.
        """
        rgba = [int(c * 0xffff) for c in self.get_rgb()]
        rgba.append(0xffff)
        return struct.pack("=HHHH", *rgba)


    def to_fill_pixel(self):
        """Converts to a pixel value for `gdk.Pixbuf.fill()`.
        """
        r, g, b = [int(c * 0xff) for c in self.get_rgb()]
        pixel = (r<<24) | (g<<16) | (b<<8) | 0xff
        return pixel


    @classmethod
    def new_from_dialog(class_, title,
                        color=None, previous_color=None,
                        parent=None):
        """Returns a colour chosen by the user via a modal dialog.

        The dialog is a standard `gtk.ColorSelectionDialog`. The returned value
        may be `None`, reflecting the user pressing Cancel in the dialog.

        """
        if color is None:
            color = RGBColor(0.5, 0.5, 0.5)
        if previous_color is None:
            previous_color = RGBColor(0.5, 0.5, 0.5)
        dialog = gtk.ColorSelectionDialog(title)
        sel = dialog.get_color_selection()
        sel.set_current_color(color.to_gdk_color())
        sel.set_previous_color(previous_color.to_gdk_color())
        dialog.set_position(gtk.WIN_POS_MOUSE)
        dialog.set_modal(True)
        dialog.set_resizable(False)
        if parent is not None:
            dialog.set_transient_for(parent)
        dialog.set_default_response(gtk.RESPONSE_OK)
        response_id = dialog.run()
        result = None
        if response_id == gtk.RESPONSE_OK:
            col_gdk = sel.get_current_color()
            result = class_.new_from_gdk_color(col_gdk)
        dialog.destroy()
        return result


    @classmethod
    def new_from_pixbuf_average(class_, pixbuf):
        """Returns the the average of all colours in a pixbuf."""
        assert pixbuf.get_colorspace() == gdk.COLORSPACE_RGB
        assert pixbuf.get_bits_per_sample() == 8
        n_channels = pixbuf.get_n_channels()
        assert n_channels in (3, 4)
        if n_channels == 3:
            assert not pixbuf.get_has_alpha()
        else:
            assert pixbuf.get_has_alpha()
        data = pixbuf.get_pixels()
        w, h = pixbuf.get_width(), pixbuf.get_height()
        rowstride = pixbuf.get_rowstride()
        n_pixels = w*h
        r = g = b = 0
        for y in xrange(h):
            for x in xrange(w):
                offs = y*rowstride + x*n_channels
                r += ord(data[offs])
                g += ord(data[offs+1])
                b += ord(data[offs+2])
        r = float(r) / n_pixels
        g = float(g) / n_pixels
        b = float(b) / n_pixels
        return RGBColor(r/255, g/255, b/255)


    def interpolate(self, other, steps):
        """Generator: interpolate between this color and another."""
        raise NotImplementedError


class RGBColor (UIColor):
    """Additive Red/Green/Blue representation of a colour.
    """
    r = None  #: Read/write red channel, range 0.0 to 1.0
    g = None  #: Read/write green channel, range 0.0 to 1.0
    b = None  #: Read/write blue channel, range 0.0 to 1.0

    def __init__(self, r=None, g=None, b=None, rgb=None, color=None):
        """Initializes from individual values, or another UIColor

          >>> col1 = RGBColor(1, 0, 1)
          >>> col2 = RGBColor(r=1, g=0.0, b=1)
          >>> col1 == col2
          True
          >>> RGBColor(color=HSVColor(0.0, 0.0, 0.5))
          <RGBColor r=0.5000, g=0.5000, b=0.5000>
        """
        UIColor.__init__(self)
        if color is not None:
            r, g, b = color.get_rgb()
        if rgb is not None:
            r, g, b = rgb
        self.r = r; assert self.r is not None
        self.g = g; assert self.g is not None
        self.b = b; assert self.b is not None

    def get_rgb(self):
        return self.r, self.g, self.b

    def __repr__(self):
        return "<RGBColor r=%0.4f, g=%0.4f, b=%0.4f>" \
            % (self.r, self.g, self.b)


    def interpolate(self, other, steps):
        """RGB interpolation.

        >>> white = RGBColor(r=1, g=1, b=1)
        >>> black = RGBColor(r=0, g=0, b=0)
        >>> [c.to_hex_str() for c in white.interpolate(black, 3)]
        ['#ffffff', '#7f7f7f', '#000000']
        >>> [c.to_hex_str() for c in black.interpolate(white, 3)]
        ['#000000', '#7f7f7f', '#ffffff']

        """
        assert steps >= 3
        other = RGBColor(color=other)
        for step in xrange(steps):
            p = float(step) / (steps - 1)
            r = self.r + (other.r - self.r) * p
            g = self.g + (other.g - self.g) * p
            b = self.b + (other.b - self.b) * p
            yield RGBColor(r=r, g=g, b=b)


class HSVColor (UIColor):
    """Cylindrical Hue/Saturation/Value representation of a colour.

      >>> col = HSVColor(0.6, 0.5, 0.4)
      >>> col.h = 0.7
      >>> col.s = 0.0
      >>> col.v = 0.1
      >>> col.get_rgb()
      (0.1, 0.1, 0.1)

    """

    h = None  #: Read/write hue angle, scaled to the range 0.0 to 1.0
    s = None  #: Read/write HSV saturation, 0.0 to 1.0
    v = None  #: Read/write HSV value, 0.0 to 1.0

    def __init__(self, h=None, s=None, v=None, hsv=None, color=None):
        """Initializes from individual values, or another UIColor

          >>> col1 = HSVColor(1.0, 0.5, 0.7)
          >>> col2 = HSVColor(h=1, s=0.5, v=0.7)
          >>> col1 == col2
          True
          >>> HSVColor(color=RGBColor(0.5, 0.5, 0.5))
          <HSVColor h=0.0000, s=0.0000, v=0.5000>
        """
        UIColor.__init__(self)
        if color is not None:
            h, s, v = color.get_hsv()
        if hsv is not None:
            h, s, v = hsv
        self.h = h; assert self.h is not None
        self.s = s; assert self.s is not None
        self.v = v; assert self.v is not None

    def get_hsv(self):
        return self.h, self.s, self.v

    def get_rgb(self):
        return hsv_to_rgb(self.h, self.s, self.v)

    def __repr__(self):
        return "<HSVColor h=%0.4f, s=%0.4f, v=%0.4f>" \
            % (self.h, self.s, self.v)


    def interpolate(self, other, steps):
        """HSV interpolation, sometimes nicer looking than RGB.

        >>> red_hsv = HSVColor(h=0, s=1, v=1)
        >>> green_hsv = HSVColor(h=1./3, s=1, v=1)
        >>> [c.to_hex_str() for c in green_hsv.interpolate(red_hsv, 3)]
        ['#00ff00', '#ffff00', '#ff0000']
        >>> [c.to_hex_str() for c in red_hsv.interpolate(green_hsv, 3)]
        ['#ff0000', '#ffff00', '#00ff00']

        Note the pure yellow. Interpolations in RGB space are duller looking:

        >>> red_rgb = RGBColor(color=red_hsv)
        >>> [c.to_hex_str() for c in red_rgb.interpolate(green_hsv, 3)]
        ['#ff0000', '#7f7f00', '#00ff00']

        """
        assert steps >= 3
        other = HSVColor(color=other)
        # Calculate the shortest angular distance
        # Normalize first
        ha = self.h % 1.0
        hb = other.h % 1.0
        # If the shortest distance doesn't pass through zero, then
        hdelta = hb - ha
        # But the shortest distance might pass through zero either antilockwise
        # or clockwise. Smallest magnitude wins.
        for hdx0 in -(ha+1-hb), (hb+1-ha):
            if abs(hdx0) < abs(hdelta):
                hdelta = hdx0
        # Interpolate, using shortest angular dist for hue
        for step in xrange(steps):
            p = float(step) / (steps - 1)
            h = (self.h + hdelta * p) % 1.0
            s = self.s + (other.s - self.s) * p
            v = self.v + (other.v - self.v) * p
            yield HSVColor(h=h, s=s, v=v)


class HCYColor (UIColor):
    """Cylindrical Hue/Chroma/Luma colour, with perceptually weighted luma.

    Not an especially common colour space. Sometimes referred to as HSY, HSI,
    or (occasionally and wrongly) as HSL. The Hue `h` term is identical to that
    used by `HSVColor`. Luma `y`, however, is a perceptually-weighted
    representation of the brightness. This ordinarily would make an assymetric
    colourspace solid not unlike the Y'CbCr one because the red, green and blue
    primaries underlying it do not contribute equally to the human perception
    of brightness. Therefore the Chroma `c` term is the fraction of the maximum
    permissible saturation at the given `h` and `y`: this scaling to within the
    legal RGB gamut causes the resultant colour space to be a regular cylinder.

    """

    h = None  #: Read/write hue angle, scaled to the range 0.0 to 1.0
    c = None  #: Read/write HCY chroma, 0.0 to 1.0
    y = None  #: Read/write HCY luma, 0.0 to 1.0

    def __init__(self, h=None, c=None, y=None, hcy=None, color=None):
        """Initializes from individual values, or another UIColor

          >>> col1 = HCYColor(0, 0.1, 0.2)
          >>> col2 = HCYColor(h=0, c=0.1, y=.2)
          >>> col3 = HCYColor(hcy=[0, 0.1, .2])
          >>> col1 == col2 and col2 == col3
          True
          >>> HCYColor(color=RGBColor(0.5, 0.5, 0.5))
          <HCYColor h=0.0000, c=0.0000, y=0.5000>
        """
        UIColor.__init__(self)
        if color is not None:
            if isinstance(color, HCYColor):
                h = color.h
                c = color.c
                y = color.y
            else:
                h, s, v = color.get_hsv()
                h_, c, y = RGB_to_HCY(hsv_to_rgb(h, s, v))
        if hcy is not None:
            h, c, y = hcy
        self.h = h; assert self.h is not None
        self.c = c; assert self.c is not None
        self.y = y; assert self.y is not None

    def get_hsv(self):
        rgb = self.get_rgb()
        h, s, v = rgb_to_hsv(*rgb)
        return self.h, s, v

    def get_rgb(self):
        return HCY_to_RGB((self.h, self.c, self.y))

    def get_luma(self):
        return self.y

    def __repr__(self):
        return "<HCYColor h=%0.4f, c=%0.4f, y=%0.4f>" \
            % (self.h, self.c, self.y)


    def interpolate(self, other, steps):
        """HCY interpolation.

        >>> red = HCYColor(0, 0.8, 0.5)
        >>> green = HCYColor(1./3, 0.8, 0.5)
        >>> [c.to_hex_str() for c in green.interpolate(red, 5)]
        ['#19c619', '#5ea319', '#8c8c19', '#c46f19', '#e55353']
        >>> [c.to_hex_str() for c in red.interpolate(green, 5)]
        ['#e55353', '#c46f19', '#8c8c19', '#5ea319', '#19c619']

        HCY is a cylindrical space, so interpolations between two endpoints of
        the same chroma will preserve that chroma. RGB interpoloation tends to
        diminish because the interpolation will pass near the diagonal of zero
        chroma.

        >>> [i.c for i in red.interpolate(green, 5)]
        [0.8, 0.8, 0.8, 0.8, 0.8]
        >>> red_rgb = RGBColor(color=red)
        >>> [round(HCYColor(color=i).c, 3)
        ...       for i in red_rgb.interpolate(green, 5)]
        [0.8, 0.457, 0.571, 0.686, 0.8]

        """
        assert steps >= 3
        other = HCYColor(color=other)
        # Like HSV, interpolate using the shortest angular distance.
        ha = self.h % 1.0
        hb = other.h % 1.0
        hdelta = hb - ha
        for hdx0 in -(ha+1-hb), (hb+1-ha):
            if abs(hdx0) < abs(hdelta):
                hdelta = hdx0
        for step in xrange(steps):
            p = float(step) / (steps - 1)
            h = (self.h + hdelta * p) % 1.0
            c = self.c + (other.c - self.c) * p
            y = self.y + (other.y - self.y) * p
            yield HCYColor(h=h, c=c, y=y)



class YCbCrColor (UIColor):
    """YUV-type colour, using the BT601 definition.

    This implementation uses the BT601 Y'CbCr definition. Luma (`Y`) ranges
    from 0 to 1, the chroma components (`Cb` and `Cr`) range from -0.5 to 0.5.
    The projection of this space onto the Y=0 plane is similar to a slightly
    tilted regular hexagon.

    This colour space is derived from the displayable RGB space. The luma or
    chroma components may be manipluated, but because the envelope of the RGB
    cube does not align with this space's axes it's quite easy to go out of
    the displayable gamut. Two methods are provided for clipping or scaling
    out of gamut values back to RGB.

    """

    Y = None  #: Read/write BT601 luma, 0.0 to 1.0
    Cb = None  #: Read/write BT601 blue-difference chroma, -0.5 to 0.5.
    Cr = None  #: Read/write BT601 red-difference chroma, -0.5 to 0.5.
    __Y0 = None
    __Cb0 = None
    __Cr0 = None

    def __init__(self, Y=None, Cb=None, Cr=None, YCbCr=None, color=None):
        """Initializes from individual values, or another UIColor
        """
        UIColor.__init__(self)
        if color is not None:
            if isinstance(color, YCbCrColor):
                Y = color.Y
                Cb = color.Cb
                Cr = color.Cr
            else:
                rgb = color.get_rgb()
                Y, Cb, Cr = RGB_to_YCbCr_BT601(rgb)
        if YCbCr is not None:
            Y, Cb, Cr = YCbCr
        self.Y = Y
        assert self.Y is not None
        self.Cb = Cb
        assert self.Cb is not None
        self.Cr = Cr
        assert self.Cr is not None
        self.__Y0 = Y
        self.__Cb0 = Cb
        self.__Cr0 = Cr

    def get_luma(self):
        return self.Y

    def get_rgb(self):
        """Gets a raw RGB triple, possibly out of gamut.
        """
        return YCbCr_to_RGB_BT601((self.Y, self.Cb, self.Cr))


    def in_gamut(self):
        """Returns whether the colur is within the RGB gamut.
        """
        rgb = self.get_rgb()
        return min(rgb) >= 0 and max(rgb) <= 1


    def gamut_clip(self):
        """Clip to the RGB gamut; for use after altering luma.
        """
        assert self.Cb == self.__Cb0
        assert self.Cr == self.__Cr0
        if self.Y == self.__Y0:
            return
        tau_c = gamut_adaptive_tau_c(self.Y, self.get_rgb())
        self.Cb *= tau_c
        self.Cr *= tau_c
        self.__Cb0 = self.Cb
        self.__Cr0 = self.Cr


    def gamut_scale(self):
        """Scale to the RGB gamut; for use after altering chroma.
        """
        assert self.Y == self.__Y0
        if self.Cb == self.__Cb0 and self.Cr == self.__Cr0:
            return
        rgb0 = YCbCr_to_RGB_BT601((self.__Y0, self.__Cb0, self.__Cr0))
        rgb = YCbCr_to_RGB_BT601((self.Y, self.Cb, self.Cr))
        tau_s = gamut_adaptive_tau_s(self.__Y0, rgb0, self.Y, rgb)
        self.Cb = self.__Cb0 = self.Cb * tau_s
        self.Cr = self.__Cr0 = self.Cr * tau_s


    def __repr__(self):
        return "<YCbCrColor Y=%0.4f, Cb=%0.4f, Cr=%0.4f>" \
            % (self.Y, self.Cb, self.Cr)


    def interpolate(self, other, steps):
        """YCbCr interpolation.

        >>> yellow = YCbCrColor(color=RGBColor(1,1,0))
        >>> red = YCbCrColor(color=RGBColor(1,0,0))
        >>> [c.to_hex_str() for c in yellow.interpolate(red, 3)]
        ['#feff00', '#ff7f00', '#ff0000']

        This colorspace is a simple transformation of the RGB cube, so to
        within a small margin of error, the results of this interpolation are
        identical to an interpolation in RGB space.

        >>> y_rgb = RGBColor(1,1,0)
        >>> r_rgb = RGBColor(1,0,0)
        >>> [c.to_hex_str() for c in y_rgb.interpolate(r_rgb, 3)]
        ['#ffff00', '#ff7f00', '#ff0000']

        """
        assert steps >= 3
        other = YCbCrColor(color=other)
        # Like HSV, interpolate using the shortest angular distance.
        for step in xrange(steps):
            p = float(step) / (steps - 1)
            Y = self.Y + (other.Y - self.Y) * p
            Cb = self.Cb + (other.Cb - self.Cb) * p
            Cr = self.Cr + (other.Cr - self.Cr) * p
            yield YCbCrColor(Y=Y, Cb=Cb, Cr=Cr)

## 
## Linear/sRGB transformations.
## Here for completeness, but unused.
## 

def _apow(v, p):
    return (v >= 0) and (v**p) or -((-v)**p)

def sRGB_component_to_linearRGB(c):
    if abs(c) <= 0.04045:
        return c / 12.92
    else:
        return _apow(((c+0.055) / 1.055), 2.4)

def linearRGB_component_to_sRGB(c):
    if abs(c) <= 0.0031308:
        return 12.92 * c
    else:
        return 1.055 * _apow(c, 1/2.4) - 0.055

def sRGB_to_linearRGB(rgb):
    return tuple([sRGB_component_to_linearRGB(c) for c in rgb])

def linearRGB_to_sRGB(rgb):
    return tuple([linearRGB_component_to_sRGB(c) for c in rgb])



##
## Clipping and scaling manipulated and potentially out-gamut RGB triples
## based on target luma. Useful after transforms in YCC (YUV, YIQ, YCbCr)
## spaces.
##


def __gamut_adaptive_gamfunc(c, Y):
    if c == Y:
        return 1.0
    return max((1-Y)/(c-Y), (-Y)/(c-Y))


def gamut_adaptive_tau_c(Y, rgb):
    r, g, b = rgb
    if Y < 0: return 0.0
    if Y > 1: return 0.0
    grY = __gamut_adaptive_gamfunc(r, Y)
    ggY = __gamut_adaptive_gamfunc(g, Y)
    gbY = __gamut_adaptive_gamfunc(b, Y)
    return min(1.0, grY, ggY, gbY)


def gamut_adaptive_clip(Y, rgb):
    """Gamut-adaptive clipping, for use after chrominance processing.

    Returns a clipped ``(r,g,b)`` triple, in the RGB gamut. `Y` is the
    processed YCbCr luminance component, and `rgb` is the processed RGB triple,
    potentially out of gamut.

    ref: http://dx.doi.org/10.1109/ICIP.2010.5652000
    """
    if Y < 0: return (0.0, 0.0, 0.0)
    elif Y > 1: return (1.0, 1.0, 1.0)
    tau_c = gamut_adaptive_tau_c(Y, rgb)
    return tuple(tau_c*c + (1-tau_c)*Y for c in rgb)


def gamut_adaptive_tau_s(Y_old, rgb_old, Y_new, rgb_new):
    if Y_new < 0: return 0.0
    elif Y_new > 1: return 0.0
    if rgb_new[0] == rgb_new[1] == rgb_new[2]:
        return 0.0
    else:
        min_newgam = min(__gamut_adaptive_gamfunc(c, Y_new) for c in rgb_new)
        min_oldgam = min(__gamut_adaptive_gamfunc(c, Y_old) for c in rgb_old)
        return min_newgam / min_oldgam


def gamut_adaptive_scale(Y_old, rgb_old, Y_new, rgb_new):
    """Gamut-adaptive scaling, for use after luminance processing.

    Returns a scaled ``(r,g,b)`` triple within the RGB gamut. `rgb_old` is the
    colour before luminance processing was applied, and `Y_old` is the
    corresponding luminance component. The ``*_new`` values are the values
    after the change of luminance, where `rgb_new` may be out of gamut. The
    returned value is scaled so as to retain the old value's saturation, i.e.
    its relative position between grey and the gamut envelope along an
    equilumiant plane.

    ref: http://dx.doi.org/10.1109/ICIP.2010.5652000
    """
    if Y_new < 0: return (0.0, 0.0, 0.0)
    elif Y_new > 1: return (1.0, 1.0, 1.0)
    tau_s = gamut_adaptive_tau_s(Y_old, rgb_old, Y_new, rgb_new)
    return tuple(tau_s*c + (1-tau_s)*Y_new for c in rgb_new)



##############################################################################
###
###  Non-OO colour transforms
###


##
## YCC spaces
##

## ITU.BT-601 Y'CbCr renormalized values (Cb, Cr between -0.5 and 0.5).

# A YCC space, i.e. one luma dimension and two orthogonal chroma axes derived
# directly from an RGB model. Planes of constant Y are roughly equiluminant,
# but the colour solid is asymmetrical.
# 
# Of marginal interest, the projection of the pure-tone {R,Y,G,C,B,M} onto the
# Y=0 plane is very close to exactly hexagonal. Shame that cross-sections of
# the colour solid are irregular triangles, rectangles and pentagons following
# a parallelepiped standing on a point.
#
# ref http://www.itu.int/rec/R-REC-BT.601/en


def RGB_to_YCbCr_BT601(rgb):
    R, G, B = rgb
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.500 * B
    Cr = 0.500 * R - 0.419 * G - 0.081 * B
    return Y, Cb, Cr

def YCbCr_to_RGB_BT601(YCbCr):
    Y, U, V = YCbCr
    R = Y             + 1.403 * V
    G = Y - 0.344 * U - 0.714 * V
    B = Y + 1.773 * U
    return R, G, B



## ITU.BT.709 Y'CbCr renormalized values (Cb, Cr between -0.5 and 0.5).

# The one used for HDTV. The projection is a skewed hexagon.
# ref http://www.itu.int/rec/R-REC-BT.709/en


def RGB_to_YCbCr_BT709(rgb):  # ITU.BT-709 Y'CbCr
    R, G, B = rgb
    Y = 0.2215 * R + 0.7154 * G + 0.0721 * B
    Cb = -0.1145 * R - 0.3855 * G + 0.5000 * B
    Cr = 0.5016 * R - 0.4556 * G - 0.0459 * B
    return Y, Cb, Cr

def YCbCr_to_RGB_BT709(YCbCr):
    Y, Cb, Cr = YCbCr
    R = Y               + 1.5701 * Cr
    G = Y - 0.1870 * Cb - 0.4664 * Cr
    B = Y + 1.8556 * Cb
    return R, G, B


## Generic, nonmatrix YUV

def RGB_to_YUV_generic(rgb, Wr=0.299, Wb=0.0114, Umax=0.436, Vmax=0.615):
    Wg = 1 - Wr - Wb
    r, g, b = rgb
    Y = Wr*r + Wb*b + Wg*g
    U = Umax * (b - Y) / (1 - Wb)
    V = Vmax * (r - Y) / (1 - Wr)
    return Y, U, V

def YUV_to_RGB_generic(YUV, Wr=0.299, Wb=0.0114, Umax=0.436, Vmax=0.615):
    Wg = 1 - Wr - Wb
    Y, U, V = YUV
    R = Y + V*(1-Wr)/Vmax
    G = Y - U*Wb*(1-Wb)/(Umax*Wg) - V*Wr*(1-Wr)/(Vmax*Wg)
    B = Y + U*(1-Wb)/Umax
    return R, G, B



## YDbDr colour space.

# A YCC space very similar to the Rec. 601 YCbCr one. Definitions
# from Wikipedia.


def RGB_to_YDbDr(rgb):
    R, G, B = rgb
    Y  = +0.299*R + 0.587*G + 0.114*B
    Db = -0.450*R - 0.883*G + 1.333*B
    Dr = -1.333*R + 1.116*G + 0.217*B
    return Y, Db, Dr

def YDbDr_to_RGB(ydbdr):
    Y, Db, Dr = ydbdr
    R = Y            - 0.526*Dr
    G = Y - 0.129*Db + 0.269*Dr
    B = Y + 0.665*Db
    return R, G, B



##
## Cylindrical colour spaces
##

## HCY colour space.

# Frequently referred to as HSY, Hue/Chroma/Luma, HsY, HSI etc.  It's
# equivalent to a cylindrical remapping of the YCbCr solid: the "C" term is the
# proportion of the maximum permissible chroma within the RGB gamut at a given
# hue and luma. Planes of constant Y are equiluminant.
# 
# ref https://code.google.com/p/colour-space-viewer/
# ref Joblove G.H., Greenberg D., Color spaces for computer graphics.


# For consistency, use the same weights that the Color and Luminosity layer
# blend modes use, as also used by brushlib's Colorize brush blend mode. All
# following http://dvcs.w3.org/hg/FXTF/rawfile/tip/compositing/index.html
# here. BT.601 YCbCr has a nearly identical definition of luma.

_SVGFX_RED_WEIGHT = 0.3
_SVGFX_GREEN_WEIGHT = 0.59
_SVGFX_BLUE_WEIGHT = 0.11

def RGB_to_HCY(rgb):
    _r, _g, _b = rgb
    r_weight = _SVGFX_RED_WEIGHT
    g_weight = _SVGFX_GREEN_WEIGHT
    b_weight = _SVGFX_BLUE_WEIGHT
    M = max(_r, _g, _b)
    m = min(_r, _g, _b)
    y_ = r_weight*_r + g_weight*_g + b_weight*_b
    H_sec = 0
    H_insec = 0.0
    Y_peak = 0.0
    c_ = M - m
    if c_ != 0:
        if M == _r:
            if m == _g:
                H_sec = 5
                X = _b - m
                H_insec = 1.0 - X/c_
                Y_peak = (1.0-g_weight) + H_insec*(r_weight - (1.-g_weight))
            else:
                H_sec = 0
                X = _g - m
                H_insec = X/c_
                Y_peak = r_weight + H_insec*((1.0-b_weight) - r_weight)
        elif M == _g:
            if m == _b:
                H_sec = 1
                X = _r - m
                H_insec = 1.0 - X/c_
                Y_peak = (1.0 - b_weight) + H_insec*(g_weight - (1.0-b_weight))
            else:
                H_sec = 2
                X = _b - m
                H_insec = X/c_
                Y_peak = g_weight + H_insec*((1.0-r_weight) - g_weight)
        else:
            if m == _r:
                H_sec = 3
                X = _g - m
                H_insec = 1.0 - X/c_
                Y_peak = (1.0-r_weight) + H_insec * (b_weight - (1.-r_weight))
            else:
                H_sec = 4
                X = _r - m
                H_insec = X/c_
                Y_peak = b_weight + H_insec * ((1.-g_weight) - b_weight)
    if y_ > 0.0 and y_ < 1.0:
        if y_ < Y_peak:
            c_ /= y_ / Y_peak
        else:
            c_ /= (1.0 - y_) / (1.0 - Y_peak)
    h_ = (H_sec + H_insec) / 6.0
    return h_, c_, y_



def HCY_to_RGB(hcy):
    _h, _c, _y = hcy
    r_weight = _SVGFX_RED_WEIGHT
    g_weight = _SVGFX_GREEN_WEIGHT
    b_weight = _SVGFX_BLUE_WEIGHT

    # wtf
    if _h >= 1.0:
        _h -= int(_h)
    _h *= 6.0
    H_sec = int(_h)
    H1 = (H_sec // 2) * 2
    H2 = _h - H1

    Y_peak = 0
    H_insec = _h - H_sec

    if H_sec == 0:
        Y_peak =    r_weight  + H_insec * ((1-b_weight) -    r_weight )
    elif H_sec == 1:
        Y_peak = (1-b_weight) + H_insec * (    g_weight - (1-b_weight))
    elif H_sec == 2:
        Y_peak =    g_weight  + H_insec * ((1-r_weight) -    g_weight )
    elif H_sec == 3:
        Y_peak = (1-r_weight) + H_insec * (    b_weight - (1-r_weight))
    elif H_sec == 4:
        Y_peak =    b_weight  + H_insec * ((1-g_weight) -    b_weight )
    else:
        Y_peak = (1-g_weight) + H_insec * (    r_weight - (1-g_weight))


    if _y < Y_peak:
        _c *= _y / Y_peak
    else:
        _c *= (1.0 - _y) / (1.0 - Y_peak)

    X = _c * (1.0 - abs(H2 - 1.0))

    r_ = g_ = b_ = 0.0
    if H_sec == 0:
        r_ = _c; g_ = X
    elif H_sec == 1:
        r_ = X;  g_ = _c
    elif H_sec == 2:
        g_ = _c; b_ = X
    elif H_sec == 3:
        g_ = X;  b_ = _c
    elif H_sec == 4:
        r_ = X; b_ = _c
    else:
        r_ = _c; b_ = X

    m = _y - (r_weight * r_ + g_weight * g_ + b_weight * b_)

    r_ += m
    g_ += m
    b_ += m
    return r_, g_, b_


## Improved HLS colour space.

# Pretty much HCY without the cylindrical expansion.
# 
# ref: Hanbury, The Taming of the Hue, Saturation and Brightness Colour Space
# ref: Hanbury, Circular Statistics Applied to Colour Images

from math import pi, sin, cos, acos, sqrt

def RGB_to_IHLS(rgb):
    r, g, b = [float(c) for c in rgb]
    root3 = sqrt(3)
    root32 = 0.5 * root3
    Y  = 0.2126*r + 0.7154*g + 0.0721*b
    C1 =        r -    0.5*g -    0.5*b
    C2 =          - root32*g + root32*b
    C = sqrt((C1**2) + (C2**2))
    if C == 0:
        H = 0
    elif C2 <= 0:
        H = acos(C1/C)
    else:
        H = 2*pi - acos(C1/C)
    # H* = H - k * 60deg where k in {0, 1, 2, 3, 4, 5} so that 0 <= H* <= 60deg
    Hstar = None
    for k in range(6):
        Hstar = H - k*pi/3
        if Hstar >= 0 and Hstar <= pi/3:
            break
    assert Hstar is not None
    S = 2 * C * sin(2*pi/3 - Hstar) / root3
    return H, S, Y


def IHLS_to_RGB(IHLS):
    H, S, Y = IHLS
    root3 = sqrt(3)
    root32 = 0.5 * root3

    Hstar = None
    for k in range(6):
        Hstar = H - k*pi/3
        if Hstar >= 0 and Hstar <= pi/3:
            break
    assert Hstar is not None

    C = S * root3 / 2 * sin(2*pi/3 - Hstar)
    if C == 0:
        C1 = 0
        C2 = 0
    else:
        C1 = C * cos(H)
        C2 = -C * cos(H)
    R = Y + 0.7875*C1 + 0.3714*C2
    G = Y - 0.2125*C1 - 0.2059*C2
    B = Y - 0.2125*C1 + 0.9488*C2
    return R, G, B

## 
## Other colour spaces
##


## I₁I₂I₃ colour space.

# Somewhat YUV-like but with a more regular colour solid. Horizontal planes
# of constant I₁ are not equiluminant.
# ref http://www.couleur.org/index.php?page=transformations#I1I2I3


def RGB_to_I1I2I3(RGB):
    R, G, B = RGB
    I1 = (R + G + B)/3.0
    I2 = (R - B)/2.0
    I3 = (2.0*G - R - B)/4.0
    return I1, I2, I3

def I1I2I3_to_RGB(III):
    I1, I2, I3 = III
    b = I1 - I2 - 2.0 * I3 / 3.0
    r = 2.0 * I2 + b
    g = 3.0 * I1 - r - b
    return r, g, b


if __name__ == '__main__':
    import doctest
    doctest.testmod()
