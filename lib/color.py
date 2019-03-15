# coding=utf-8
# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Color objects and transformation functions.

Color objects are lightweight polymorphic structures which can be cloned and
freely substituted no matter what subtype: they all offer methods for getting
RGB or HSV triples, and can all be constructed from any other type of color
object via a common interface. This gets around some of the UI awkwardnesses of
supporting multiple color spaces: only when you update a central shared color
with an adjuster does its type change to match the control's color space.

"""

## Imports

from __future__ import division, print_function
import re
import colorsys
import numpy as np
import math
from scipy.cluster.vq import kmeans2
import colour
from collections import namedtuple
import sys
import random

from gi.repository import GdkPixbuf
from lib import helpers

from lib.pycompat import xrange
from lib.pycompat import PY3


## Lightweight color objects


class UIColor (object):
    """Base class for color objects which can be manipulated via the UI.

    This base provides a common interface allowing concrete subclasses to be
    instantiated from other UIColors: this provides a mechanism for conversions
    between color models by mixed systems of user interface components as
    needed. Color objects are typically instantiated individually by
    specifying their components, but they may be constructed from other
    `UIColor` objects too:

      >>> col1 = RGBColor(r=0.1, g=0.33, b=0.5)
      >>> col2 = HSVColor(color=col1)

    Subclasses must implement `get_rgb()`, and a ``color`` keyword to
    their constructor which takes as one of its arguments a `UIColor` object.

    """

    def get_rgb(self):
        """Extracts a floating-point R,G,B tuple representation of the color.

        This is unimplemented at this level, but is required by most
        conversions. Subclasses are required to define this function, which
        must return a floating-point ``(r, g, b)`` representation of the color
        with the channel samples lying in the range 0.0 and 1.0 inclusive.

        """
        raise NotImplementedError

    def get_hsv(self):
        """Extracts a floating-point H,S,V tuple representation of the color.

        All terms in the returned ``(h, s, v)`` triple must be scaled to lie in
        the range 0.0 to 1.0 inclusive, compatible with `colorsys`. At this
        level, the operation is defined using an invocation of `get_rgb()`, but
        this behaviour can be overridden by subclasses.

        """
        return colorsys.rgb_to_hsv(*self.get_rgb())

    ## Read-only properties (at this level)
    # rgb = property(get_rgb)
    # hsv = property(get_hsv)

    # RGB read-only
    @property
    def r(self):
        """Read-only RGB red value."""
        return self.get_rgb()[0]

    @property
    def g(self):
        """Read-only RGB green value."""
        return self.get_rgb()[1]

    @property
    def b(self):
        """Read-only RGB blue value."""
        return self.get_rgb()[2]

    # HSV read-only
    @property
    def h(self):
        """Read-only hue angle."""
        return self.get_hsv()[0]

    @property
    def s(self):
        """Read-only HSV saturation."""
        return self.get_hsv()[1]

    @property
    def v(self):
        """Read-only HSV value."""
        return self.get_hsv()[2]

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
        """Returns a greyscaled version of the color.

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
        """Two base color objects are equal if their RGB form is equal.

        Subclasses should override this with a type-specific alternative
        which compares in a more appropriate fashion. This
        implementation is only intended for comparing between dissimilar
        colour classes.

        """
        # Round to 8bpc for comparison
        rgb1 = [int(c * 0xff) for c in self.get_rgb()]
        try:
            rgb2 = [int(c * 0xff) for c in col.get_rgb()]
        except AttributeError:
            return False
        return rgb1 == rgb2
        # colorhistory.py uses
        #   a_ = np.array(helpers.hsv_to_rgb(*a))
        #   b_ = np.array(helpers.hsv_to_rgb(*b))
        #   return ((a_ - b_)**2).sum() < (3*1.0/256)**2

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

    __HEX_PARSE_TABLE = [
        (re.compile('^(?:#|0x)' + '([0-9a-fA-F]{2})' * 3 + '$'), 0xff),
        (re.compile('^(?:#|0x)' + '([0-9a-fA-F])' * 3 + '$'), 0xf),
    ]

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

    def to_fill_pixel(self):
        """Converts to a pixel value for `Gdk.Pixbuf.fill()`.

          >>> col = RGBColor(1,1,1)
          >>> "%08x" % (col.to_fill_pixel(),)
          'ffffffff'

        """
        r, g, b = [int(c * 0xff) for c in self.get_rgb()]
        pixel = (r << 24) | (g << 16) | (b << 8) | 0xff
        return pixel

    @classmethod
    def new_from_pixbuf_average(class_, pixbuf, size=3):
        """Returns the the kmeans dominant color in a pixbuf.

        >>> p = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, True, 8, 5, 5)
        >>> p.fill(0x880088ff)
        >>> UIColor.new_from_pixbuf_average(p).to_hex_str()
        '#880088'

        """
        assert pixbuf.get_colorspace() == GdkPixbuf.Colorspace.RGB
        assert pixbuf.get_bits_per_sample() == 8
        n_channels = pixbuf.get_n_channels()
        assert n_channels in (3, 4)
        if n_channels == 3:
            assert not pixbuf.get_has_alpha()
        else:
            assert pixbuf.get_has_alpha()
        data = pixbuf.get_pixels()

        use_kmeans = False
        try:
            from gui.application import get_app
            app = get_app()
            p = app.preferences
            use_kmeans = p['color.average_use_dominant']
        except AttributeError:
            use_kmeans = False

        if use_kmeans is True:
            arr = helpers.gdkpixbuf2numpy(pixbuf)

            # Use kmeans2 for dominant color
            # adapted from https://github.com/despawnerer/palletize
            # MIT License

            # flatten and discard alpha
            arr = arr.reshape(-1, 4)
            arr = np.delete(arr, 3, axis=1)

            count = size
            clusters = size
            iterations = 30
            assert count > 0
            assert clusters is None or clusters >= count
            assert iterations > 0

            if clusters is None:
                clusters = count if count > 1 else 3

            # find centroids of color clusters
            centroids, labels = kmeans2(
                arr.astype(float), clusters, iterations, minit='points',
                check_finite=False,
            )

            # reorder them by prominence
            _, counts = np.unique(labels, return_counts=True)
            best_centroid_indices = np.argsort(counts)[::-1]
            dominant_colors = centroids[best_centroid_indices].astype(int)
            result = [tuple(color) for color in dominant_colors[:count]]

            return RGBColor(result[0][0]/255,
                            result[0][1]/255, result[0][2]/255)
        else:
            assert isinstance(data, bytes)
            w, h = pixbuf.get_width(), pixbuf.get_height()
            rowstride = pixbuf.get_rowstride()
            n_pixels = w*h
            r = g = b = 0
            for y in xrange(h):
                for x in xrange(w):
                    offs = y*rowstride + x*n_channels
                    if PY3:
                        # bytes=bytes. Indexing produces ints.
                        r += data[offs]
                        g += data[offs+1]
                        b += data[offs+2]
                    else:
                        # bytes=str. Indexing of produces a str of len 1.
                        r += ord(data[offs])
                        g += ord(data[offs+1])
                        b += ord(data[offs+2])
            r = r / n_pixels
            g = g / n_pixels
            b = b / n_pixels
            return RGBColor(r/255, g/255, b/255)

    def interpolate(self, other, steps, modifier=None):
        """Generator: interpolate between this color and another."""
        raise NotImplementedError


class RGBColor (UIColor):
    """Additive Red/Green/Blue representation of a color."""

    # Base class overrides: make r,g,b attributes read/write
    r = None
    g = None
    b = None

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
        self.r = r  #: Read/write red channel, range 0.0 to 1.0
        self.g = g  #: Read/write green channel, range 0.0 to 1.0
        self.b = b  #: Read/write blue channel, range 0.0 to 1.0
        assert self.r is not None
        assert self.g is not None
        assert self.b is not None

    def get_rgb(self):
        return self.r, self.g, self.b

    def __repr__(self):
        return "<RGBColor r=%0.4f, g=%0.4f, b=%0.4f>" \
            % (self.r, self.g, self.b)

    def interpolate(self, other, steps, modifier=None):
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
            p = step / (steps - 1)
            r = self.r + (other.r - self.r) * p
            g = self.g + (other.g - self.g) * p
            b = self.b + (other.b - self.b) * p
            yield RGBColor(r=r, g=g, b=b)

    def mix(self, other, ratio):
        """linear mix 0-1 ratio.

        """
        other = RGBColor(color=other)
        r = (self.r + (other.r - self.r) * ratio)
        g = (self.g + (other.g - self.g) * ratio)
        b = (self.b + (other.b - self.b) * ratio)
        return RGBColor(r=r, g=g, b=b)

    def __eq__(self, other):
        """Equality test (override)

        >>> c1 = RGBColor(0.7, 0.45, 0.55)
        >>> c2 = RGBColor(0.4, 0.55, 0.45)
        >>> c2hcy = HCYColor(color=c2)
        >>> c1 == c2
        False
        >>> c1 == c1 and c2 == c2
        True
        >>> c2 == c2hcy
        True
        >>> c1 == c2hcy
        False

        """
        try:
            t1 = self.get_rgb()
            t2 = other.get_rgb()
        except AttributeError:
            return UIColor.__eq__(self, other)
        else:
            t1 = [round(c, 3) for c in t1]
            t2 = [round(c, 3) for c in t2]
            return t1 == t2


class LinearRGBColor (UIColor):
    """Additive Linear Light Red/Green/Blue representation of a color."""

    # Base class overrides: make r,g,b attributes read/write
    r = None
    g = None
    b = None

    def __init__(self, r=None, g=None, b=None, rgb=None, color=None,
                 gamma=2.2):
        """Initializes from individual values, or another UIColor

          >>> col1 = LinearRGBColor(1, 0, 1)
          >>> col2 = LinearRGBColor(r=1, g=0.0, b=1)
          >>> col1 == col2
          True
          >>> LinearRGBColor(color=HSVColor(0.0, 0.0, 0.5))
          <LinearRGBColor r=0.2176, g=0.2176, b=0.2176>
        """
        UIColor.__init__(self)
        if color is not None:
            r, g, b = color.get_rgb()
        if rgb is not None:
            r, g, b = rgb
        if gamma is not None:
            r, g, b = r**gamma, g**gamma, b**gamma
        self.r = r  #: Read/write red channel, range 0.0 to 1.0
        self.g = g  #: Read/write green channel, range 0.0 to 1.0
        self.b = b  #: Read/write blue channel, range 0.0 to 1.0
        self.gamma = gamma
        assert self.r is not None
        assert self.g is not None
        assert self.b is not None

    def get_rgb(self):
        # returns sRGB
        if self.gamma is not None:
            return (self.r**(1/self.gamma), self.g**(1/self.gamma),
                    self.b**(1/self.gamma))
        else:
            return self.r, self.g, self.b

    def __repr__(self):
        return "<LinearRGBColor r=%0.4f, g=%0.4f, b=%0.4f>" \
            % (self.r, self.g, self.b)

    def interpolate(self, other, steps, modifier=None):
        """Linear RGB interpolation.

        >>> white = LinearRGBColor(r=1, g=1, b=1)
        >>> black = LinearRGBColor(r=0, g=0, b=0)
        >>> [c.to_hex_str() for c in white.interpolate(black, 3)]
        ['#ffffff', '#bababa', '#000000']
        >>> [c.to_hex_str() for c in black.interpolate(white, 3)]
        ['#000000', '#bababa', '#ffffff']

        """
        assert steps >= 3
        other = LinearRGBColor(color=other, gamma=self.gamma)
        for step in xrange(steps):
            p = step / (steps - 1)
            r = (self.r + (other.r - self.r) * p)**(1/self.gamma)
            g = (self.g + (other.g - self.g) * p)**(1/self.gamma)
            b = (self.b + (other.b - self.b) * p)**(1/self.gamma)
            yield LinearRGBColor(r=r, g=g, b=b, gamma=self.gamma)

    def mix(self, other, ratio):
        """linear mix 0-1 ratio.

        """
        other = LinearRGBColor(color=other, gamma=self.gamma)
        r = (self.r + (other.r - self.r) * ratio)**(1/self.gamma)
        g = (self.g + (other.g - self.g) * ratio)**(1/self.gamma)
        b = (self.b + (other.b - self.b) * ratio)**(1/self.gamma)
        return LinearRGBColor(r=r, g=g, b=b, gamma=self.gamma)

    def __eq__(self, other):
        """Equality test (override)

        >>> c1 = LinearRGBColor(0.7, 0.45, 0.55)
        >>> c2 = LinearRGBColor(0.4, 0.55, 0.45)
        >>> c2hcy = HCYColor(color=c2)
        >>> c1 == c2
        False
        >>> c1 == c1 and c2 == c2
        True
        >>> c2 == c2hcy
        True
        >>> c1 == c2hcy
        False

        """
        try:
            t1 = self.get_rgb()
            t2 = other.get_rgb()
        except AttributeError:
            return UIColor.__eq__(self, other)
        else:
            t1 = [round(c, 3) for c in t1]
            t2 = [round(c, 3) for c in t2]
            return t1 == t2


class PigmentColor (UIColor):
    """Subtractive Spectral-upsampled representation of a color."""

    # Base class overrides: make r,g,b attributes read/write
    r = None
    g = None
    b = None

    def __init__(self, spd=None, color=None, gamma=2.2):
        """Initializes from individual values, or another UIColor
          # noqa: E519,E527
          >>> col1 = PigmentColor(spd=np.ones(10))
          >>> col2 = PigmentColor(color=RGBColor(r=1.0, g=0.957, b=1.0))
          >>> col1 == col2
          True
          >>> PigmentColor(color=HSVColor(0.0, 0.0, 0.5))
          <PigmentColor spd=[ 0.1195239   0.12194163  0.13034062  0.22245692  0.23211225  0.20936331
            0.2226059   0.22407617  0.22382034  0.2237448 ]>
        """
        UIColor.__init__(self)
        self.gamma = gamma
        if color is not None:
            r, g, b = color.get_rgb()
            if gamma is not None:
                r, g, b = r**gamma, g**gamma, b**gamma
        if spd is None:
            self.spd = RGB_to_Spectral((r, g, b))
        else:
            self.spd = spd
        assert self.spd is not None
        self.cachedrgb = None

    def get_rgb(self):
        if self.cachedrgb is not None:
            return self.cachedrgb
        # returns sRGB
        self.r, self.g, self.b = np.clip(Spectral_to_RGB(self.spd), 0.0, 1.0)
        if self.gamma is not None:
            self.cachedrgb = (self.r**(1/self.gamma), self.g**(1/self.gamma),
                              self.b**(1/self.gamma))
            return self.cachedrgb
        else:
            self.cachedrgb = (self.r, self.g, self.b)
            return self.cachedrgb

    def __repr__(self):
        return "<PigmentColor spd=%s>" \
            % (np.array2string(self.spd))

    def interpolate(self, other, steps, modifier=1.0):
        """WGM Spectral interpolation. Modifier controls multiply vs geo mean

        >>> white = PigmentColor(color=RGBColor(r=1, g=1, b=1))
        >>> black = PigmentColor(color=RGBColor(r=0, g=0, b=0))
        >>> [c.to_hex_str() for c in white.interpolate(black, 3)]
        ['#fffefe', '#1f1f1f', '#030303']
        >>> [c.to_hex_str() for c in black.interpolate(white, 3)]
        ['#030303', '#1f1f1f', '#fffefe']

        """
        wgm_ratio = modifier
        assert steps >= 3
        other = PigmentColor(color=other, gamma=self.gamma)
        for step in xrange(steps):
            p = step / (steps - 1)
            spd_wgm = Spectral_Mix_WGM(self.spd, other.spd, p)
            if wgm_ratio < 1.0:
                spd_mult = Spectral_Mix_MULT(self.spd, other.spd)
                p = -0.5 * (math.cos(2 * math.pi * p) - 1)
                spd = (spd_wgm * wgm_ratio**p + spd_mult
                       * (1 - wgm_ratio)**(1 - p))
                yield PigmentColor(spd=spd, gamma=self.gamma)
            else:
                yield PigmentColor(spd=spd_wgm, gamma=self.gamma)

    def mix(self, other, ratio):
        """WGM Spectral mix 0-1 ratio.

        """
        other = PigmentColor(color=other, gamma=self.gamma)
        spd = Spectral_Mix_WGM(self.spd, other.spd, ratio)
        return PigmentColor(spd=spd, gamma=self.gamma)

    def __eq__(self, other):
        """Equality test (override)

        >>> c1 = PigmentColor(color=RGBColor(r=0.7, g=0.45, b=0.55))
        >>> c2 = PigmentColor(color=RGBColor(r=0.4, g=0.55, b=0.45))
        >>> c2pig =RGBColor(color=c2)
        >>> c1 == c2
        False
        >>> c1 == c1 and c2 == c2
        True
        >>> c2 == c2pig
        True
        >>> c1 == c2pig
        False

        """
        try:
            t1 = self.get_rgb()
            t2 = other.get_rgb()
        except AttributeError:
            return UIColor.__eq__(self, other)
        else:
            t1 = [round(c, 3) for c in t1]
            t2 = [round(c, 3) for c in t2]
            return t1 == t2


class HSVColor (UIColor):
    """Cylindrical Hue/Saturation/Value representation of a color.

      >>> col = HSVColor(0.6, 0.5, 0.4)
      >>> col.h = 0.7
      >>> col.s = 0.0
      >>> col.v = 0.1
      >>> col.get_rgb()
      (0.1, 0.1, 0.1)

    """

    # Base class overrides: make h,s,v attributes read/write
    h = None
    s = None
    v = None

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
        self.h = h  #: Read/write hue angle, scaled to the range 0.0 to 1.0
        self.s = s  #: Read/write HSV saturation, 0.0 to 1.0
        self.v = v  #: Read/write HSV value, 0.0 to 1.0
        assert self.h is not None
        assert self.s is not None
        assert self.v is not None

    def get_hsv(self):
        return self.h, self.s, self.v

    def get_rgb(self):
        return colorsys.hsv_to_rgb(self.h, self.s, self.v)

    def __repr__(self):
        return "<HSVColor h=%0.4f, s=%0.4f, v=%0.4f>" \
            % (self.h, self.s, self.v)

    def interpolate(self, other, steps, modifier=None):
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
            p = step / (steps - 1)
            h = (self.h + hdelta * p) % 1.0
            s = self.s + (other.s - self.s) * p
            v = self.v + (other.v - self.v) * p
            yield HSVColor(h=h, s=s, v=v)

    def mix(self, other, ratio):
        """linear mix 0-1 ratio.

        """
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
        h = (self.h + hdelta * ratio) % 1.0
        s = self.s + (other.s - self.s) * ratio
        v = self.v + (other.v - self.v) * ratio
        return HSVColor(h=h, s=s, v=v)

    def __eq__(self, other):
        """Equality test (override)

        >>> c1 = HSVColor(0.7, 0.45, 0.55)
        >>> c2 = HSVColor(0.4, 0.55, 0.45)
        >>> c2rgb = RGBColor(color=c2)
        >>> c1 == c2
        False
        >>> c1 == c1 and c2 == c2
        True
        >>> c2 == c2rgb
        True
        >>> c1 == c2rgb
        False

        Colours with zero value but differing hues or saturations must
        test equal. The same isn't true of the other end of the
        cylinder.

        >>> HSVColor(0.7, 0.45, 0.0) == HSVColor(0.4, 0.55, 0.0)
        True
        >>> HSVColor(0.7, 0.45, 1.0) == HSVColor(0.4, 0.55, 1.0)
        False

        """
        try:
            t1 = self.get_hsv()
            t2 = other.get_hsv()
        except AttributeError:
            return UIColor.__eq__(self, other)
        else:
            t1 = [round(c, 3) for c in t1]
            t2 = [round(c, 3) for c in t2]
            if t1[-1] == t2[-1] == 0:
                return True
            return t1 == t2


class CAM16Color (UIColor):
    """CAM16 representation of a color.  Use VSH as stand-ins for axes

      >>> col = CAM16Color(55.67306142, 28.26474923, 106.14599451)
      >>> col.h = 106.14599451
      >>> col.s = 28.26474923
      >>> col.v = 95.67306142
      >>> result = col.get_rgb()
      >>> print (round(result[0],3), round(result[1],3), round(result[2],3))
      1.0 0.966 0.633

    """

    # Base class overrides: make h,s,v attributes read/write
    v = None
    s = None
    h = None

    def __init__(self, v=None, s=None, h=None, vsh=None, color=None,
                 cieaxes=None, illuminant=None,
                 gamutmapping="relativeColorimetric"):
        """Initializes from individual values, or another UIColor

          >>> col1 = CAM16Color(95.67306142,   58.26474923,  106.14599451)
          >>> col2 = CAM16Color(h=106.14599451, s=58.26474923, v=95.67306142)
          >>> col1 == col2
          True
          >>> CAM16Color(color=RGBColor(0.5, 0.5, 0.5))
          <CAM16, v=53.0488, s=2.4757, h=209.5203, illuminant=95.0456, 100.0000, 108.9058>
        """
        UIColor.__init__(self)

        self.cieconfig = None

        # gamut mapping strategy
        # relativeColorimetric, highlight, or False
        # highlight will flag out of gamut colors as magenta similar to GIMP
        # False will simply clip the RGB values
        self.gamutmapping = gamutmapping

        # The entire cam16 config needs to follow around the color
        if cieaxes is not None:
            self.cieaxes = cieaxes
        else:
            self.cieaxes = "JMh"

        if illuminant is not None:
            self.illuminant = np.array(illuminant)

        else:
            self.illuminant = np.array(
                colour.xy_to_XYZ(
                    colour.ILLUMINANTS['cie_2_1931']['D65']) * 100.0)

        self.L_A = 20.0
        self.Y_b = 4.5
        self.surround = colour.CAM16_VIEWING_CONDITIONS['Dim']

        # maybe we want to know if the gamut was constrained
        # so we can halt sliders and adjusters from going farther
        self.gamutexceeded = None
        self.displayexceeded = None

        # limit color purity?
        self.limit_purity = None
        self.reset_intent = False

        # try getting from preferences but fallback to avoid breaking doctest
        try:
            from gui.application import get_app
            app = get_app()
            p = app.preferences
            if p['color.limit_purity'] >= 0.0:
                self.limit_purity = p['color.limit_purity']
            self.reset_intent = p['color.reset_intent_after_gamut_map']
        except:
            self.limit_purity = None
            self.reset_intent = False
        # don't cache this until get_rgb, so we can modify 1st via
        # adjusters
        self.cachedrgb = None
        if color is not None:
            if isinstance(color, CAM16Color):
                # convert from one to another (handle whitepoint changes)
                v, s, h = color.v, color.s, color.h
                self.illuminant = color.illuminant
            else:
                # any other UIColor is assumed to be sRGB
                rgb = color.get_rgb()
                v, s, h = RGB_to_CAM16(self, rgb)

        if vsh is not None:
            try:
                v, s, h = vsh
            except ValueError:
                v = s = h = 0.0
        self.h = h  #: Read/write hue angle, 0-360
        self.s = s  #: Read/write CAM16 saturation, no scaling
        self.v = v  #: Read/write CAM16 value, no scaling
        assert self.h is not None
        assert self.s is not None
        assert self.v is not None

    def get_hsv(self):
        rgb = self.get_rgb()
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        return h, s, v

    def get_rgb(self):
        if self.cachedrgb:
            return self.cachedrgb
        return CAM16_to_RGB(self)

    def __repr__(self):
        return ("<CAM16, v=%0.4f, s=%0.4f, h=%0.4f, "
                + "illuminant=%0.4f, %0.4f, %0.4f>") \
            % (self.v, self.s, self.h, self.illuminant[0],
               self.illuminant[1],
               self.illuminant[2])

    def interpolate(self, other, steps, modifier=None):
        """CAM16 interpolation, sometimes nicer looking than anything else.

        >>> red_hsv = CAM16Color(h=32.1526953,s=80.46644073,v=46.9250674 )
        >>> green_hsv = CAM16Color(h=136.6478602,s=76.64436113,v=79.7493805)
        >>> [c.to_hex_str() for c in green_hsv.interpolate(red_hsv, 3)]
        ['#79e725', '#fecc00', '#c82c00']
        >>> [c.to_hex_str() for c in red_hsv.interpolate(green_hsv, 3)]
        ['#c82c00', '#fecc00', '#79e725']

        """
        assert steps >= 3
        other = CAM16Color(color=other)
        # Calculate the shortest angular distance
        # Normalize first
        ha = self.h % 360.
        hb = other.h % 360.
        # If the shortest distance doesn't pass through zero, then
        hdelta = hb - ha
        # But the shortest distance might pass through zero either antilockwise
        # or clockwise. Smallest magnitude wins.
        for hdx0 in -(ha+360.-hb), (hb+360.-ha):
            if abs(hdx0) < abs(hdelta):
                hdelta = hdx0
        # Interpolate, using shortest angular dist for hue
        for step in xrange(steps):
            p = step / (steps - 1)
            h = (self.h + hdelta * p)
            s = self.s + (other.s - self.s) * p
            v = self.v + (other.v - self.v) * p
            yield CAM16Color(h=h, s=s, v=v, illuminant=self.illuminant)

    def mix(self, other, ratio):
        """linear mix 0-1 ratio.

        """
        other = CAM16Color(color=other)
        # Calculate the shortest angular distance
        # Normalize first
        ha = self.h % 360.
        hb = other.h % 360.
        # If the shortest distance doesn't pass through zero, then
        hdelta = hb - ha
        # But the shortest distance might pass through zero either antilockwise
        # or clockwise. Smallest magnitude wins.
        for hdx0 in -(ha+360.-hb), (hb+360.-ha):
            if abs(hdx0) < abs(hdelta):
                hdelta = hdx0
        h = (self.h + hdelta * ratio) % 360.
        s = self.s + (other.s - self.s) * ratio
        v = self.v + (other.v - self.v) * ratio
        return CAM16Color(h=h, s=s, v=v, illuminant=self.illuminant)

    def __eq__(self, other):
        """Equality test (override)

        >>> c1 = CAM16Color(color=HSVColor(0.7, 0.45, 0.55))
        >>> c2ill = c1.illuminant - 20.0
        >>> c2 = CAM16Color(color=HSVColor(0.7, 0.45, 0.55), illuminant=np.array([ 109.84660695, 100, 35.58228003]))
        >>> c2rgb = RGBColor(color=c2)
        >>> c1 == c2
        False
        >>> c1 == c1 and c2 == c2
        True
        >>> c2 == c2rgb
        True
        >>> c1 == c2rgb
        False

        """
        try:
            t1 = (self.v, self.s, self.h, self.limit_purity)
            t1_ill = self.illuminant
            t2 = (other.v, other.s, other.h, other.limit_purity)
            t2_ill = other.illuminant
        except AttributeError:
            return UIColor.__eq__(self, other)
        else:
            return t1 == t2 and np.array_equal(t1_ill, t2_ill)



class HCYColor (UIColor):
    """Cylindrical Hue/Chroma/Luma color, with perceptually weighted luma.

    Not an especially common color space. Sometimes referred to as HSY, HSI,
    or (occasionally and wrongly) as HSL. The Hue `h` term is identical to that
    used by `HSVColor`. Luma `y`, however, is a perceptually-weighted
    representation of the brightness. This ordinarily would make an asymmetric
    colorspace solid not unlike the Y'CbCr one because the red, green and blue
    primaries underlying it do not contribute equally to the human perception
    of brightness. Therefore the Chroma `c` term is the fraction of the maximum
    permissible saturation at the given `h` and `y`: this scaling to within the
    legal RGB gamut causes the resultant color space to be a regular cylinder.

    In practical terms, adjusting luma alone moves the color along a shading
    series of uniform relative saturation towards either white or black. This
    feature is useful for gamut masking especially, and when working in
    painting styles where value is drawn first and color applied later.
    However the pure "digital" colors appear at different heights in the
    color solid of this model, which can be confusing.

    """

    # Base class override: make h attribute read/write
    h = None

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
                h_, c, y = RGB_to_HCY(colorsys.hsv_to_rgb(h, s, v))
        if hcy is not None:
            h, c, y = hcy
        self.h = h  #: Read/write hue angle, scaled to the range 0.0 to 1.0
        self.c = c  #: Read/write HCY chroma, 0.0 to 1.0
        self.y = y  #: Read/write HCY luma, 0.0 to 1.0
        assert self.h is not None
        assert self.c is not None
        assert self.y is not None

    def get_hsv(self):
        rgb = self.get_rgb()
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        return self.h, s, v

    def get_rgb(self):
        return HCY_to_RGB((self.h, self.c, self.y))

    def get_luma(self):
        return self.y

    def __repr__(self):
        return "<HCYColor h=%0.4f, c=%0.4f, y=%0.4f>" \
            % (self.h, self.c, self.y)

    def interpolate(self, other, steps, modifier=None):
        """HCY interpolation.

        >>> red = HCYColor(0, 0.8, 0.5)
        >>> green = HCYColor(1./3, 0.8, 0.5)
        >>> [c.to_hex_str() for c in green.interpolate(red, 5)]
        ['#19a819', '#579519', '#878719', '#cc7219', '#e56363']
        >>> [c.to_hex_str() for c in red.interpolate(green, 5)]
        ['#e56363', '#cc7219', '#878719', '#579519', '#19a819']

        HCY is a cylindrical space, so interpolations between two endpoints of
        the same chroma will preserve that chroma. RGB interpoloation tends to
        diminish because the interpolation will pass near the diagonal of zero
        chroma.

        >>> [i.c for i in red.interpolate(green, 5)]
        [0.8, 0.8, 0.8, 0.8, 0.8]
        >>> red_rgb = RGBColor(color=red)
        >>> [round(HCYColor(color=i).c, 3)
        ...       for i in red_rgb.interpolate(green, 5)]
        [0.8, 0.4, 0.508, 0.654, 0.8]

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
            p = step / (steps - 1)
            h = (self.h + hdelta * p) % 1.0
            c = self.c + (other.c - self.c) * p
            y = self.y + (other.y - self.y) * p
            yield HCYColor(h=h, c=c, y=y)

    def mix(self, other, ratio):
        """linear mix 0-1 ratio.

        """
        other = HCYColor(color=other)
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
        h = (self.h + hdelta * ratio) % 1.0
        c = self.c + (other.c - self.c) * ratio
        y = self.y + (other.y - self.y) * ratio
        return HCYColor(h=h, c=c, y=y)

    def __eq__(self, other):
        """Equality test (override)

        >>> c1 = HCYColor(0.7, 0.45, 0.55)
        >>> c2 = HCYColor(0.4, 0.55, 0.45)
        >>> c2rgb = RGBColor(color=c2)
        >>> c1 == c2
        False
        >>> c1 == c1 and c2 == c2
        True
        >>> c2rgb == c2
        True
        >>> c1 == c2rgb
        False

        Two colours with identical lumas but with differing hues or
        saturations must test equal if their luma is pure black or pure
        white.

        >>> HCYColor(0.7, 0.45, 0.0) == HCYColor(0.4, 0.55, 0.0)
        True
        >>> HCYColor(0.7, 0.45, 1.0) == HCYColor(0.4, 0.55, 1.0)
        True

        """
        try:
            t1 = (self.h, self.c, self.y)
            t2 = (other.h, other.c, other.y)
        except AttributeError:
            return UIColor.__eq__(self, other)
        else:
            t1 = [round(c, 3) for c in t1]
            t2 = [round(c, 3) for c in t2]
            if t1[-1] == t2[-1] == 0:
                return True
            if t1[-1] == t2[-1] == 1:
                return True
            return t1 == t2


class YCbCrColor (UIColor):
    """YUV-type color, using the BT601 definition.

    This implementation uses the BT601 Y'CbCr definition. Luma (`Y`) ranges
    from 0 to 1, the chroma components (`Cb` and `Cr`) range from -0.5 to 0.5.
    The projection of this space onto the Y=0 plane is similar to a slightly
    tilted regular hexagon.

    This color space is derived from the displayable RGB space. The luma or
    chroma components may be manipluated, but because the envelope of the RGB
    cube does not align with this space's axes it's quite easy to go out of
    the displayable gamut.

    """

    def __init__(self, Y=None, Cb=None, Cr=None, YCbCr=None, color=None):
        """Initializes from individual values, or another UIColor"""
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
        self.Y = Y  #: Read/write BT601 luma, 0.0 to 1.0
        self.Cb = Cb  #: Read/write BT601 blue-difference chroma, -0.5 to 0.5.
        self.Cr = Cr  #: Read/write BT601 red-difference chroma, -0.5 to 0.5.
        assert self.Y is not None
        assert self.Cb is not None
        assert self.Cr is not None

    def get_luma(self):
        return self.Y

    def get_rgb(self):
        """Gets a raw RGB triple, possibly out of gamut.
        """
        return YCbCr_to_RGB_BT601((self.Y, self.Cb, self.Cr))

    def __repr__(self):
        return "<YCbCrColor Y=%0.4f, Cb=%0.4f, Cr=%0.4f>" \
            % (self.Y, self.Cb, self.Cr)

    def interpolate(self, other, steps, modifier=None):
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
            p = step / (steps - 1)
            Y = self.Y + (other.Y - self.Y) * p
            Cb = self.Cb + (other.Cb - self.Cb) * p
            Cr = self.Cr + (other.Cr - self.Cr) * p
            yield YCbCrColor(Y=Y, Cb=Cb, Cr=Cr)

    def mix(self, other, ratio):
        """linear mix 0-1 ratio.

        """
        other = YCbCrColor(color=other)
        Y = self.Y + (other.Y - self.Y) * ratio
        Cb = self.Cb + (other.Cb - self.Cb) * ratio
        Cr = self.Cr + (other.Cr - self.Cr) * ratio
        return YCbCrColor(Y=Y, Cb=Cb, Cr=Cr)

    def __eq__(self, other):
        """Equality test (override)

        >>> c1 = YCbCrColor(0.7, 0.45, 0.55)
        >>> c2 = YCbCrColor(0.4, 0.55, 0.45)
        >>> c2rgb = RGBColor(color=c2)
        >>> c1 == c2
        False
        >>> c1 == c1 and c2 == c2
        True
        >>> c2 == c2rgb
        True
        >>> c1 == c2rgb
        False

        """
        try:
            t1 = (self.Y, self.Cb, self.Cr)
            t2 = (other.Y, other.Cb, other.Cr)
        except AttributeError:
            return UIColor.__eq__(self, other)
        else:
            t1 = [round(c, 3) for c in t1]
            t2 = [round(c, 3) for c in t2]
            return t1 == t2


## ITU.BT-601 Y'CbCr renormalized values (Cb, Cr between -0.5 and 0.5).

# A YCC space, i.e. one luma dimension and two orthogonal chroma axes derived
# directly from an RGB model. Planes of constant Y are roughly equiluminant,
# but the color solid is asymmetrical.
#
# Of marginal interest, the projection of the pure-tone {R,Y,G,C,B,M} onto the
# Y=0 plane is very close to exactly hexagonal. Shame that cross-sections of
# the color solid are irregular triangles, rectangles and pentagons following
# a rectangular cuboid standing on a point.
#
# ref http://www.itu.int/rec/R-REC-BT.601/en


def RGB_to_YCbCr_BT601(rgb):
    """RGB → BT601 YCbCr: R,G,B,Y ∈ [0, 1]; Cb,Cr ∈ [-0.5, 0.5]"""
    R, G, B = rgb
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.500 * B
    Cr = 0.500 * R - 0.419 * G - 0.081 * B
    return Y, Cb, Cr


def YCbCr_to_RGB_BT601(YCbCr):
    """BT601 YCbCr → RGB: R,G,B,Y ∈ [0, 1]; Cb,Cr ∈ [-0.5, 0.5]"""
    Y, U, V = YCbCr
    R = Y + 1.403 * V
    G = Y - 0.344 * U - 0.714 * V
    B = Y + 1.773 * U
    return R, G, B


## HCY color space.

# Frequently referred to as HSY, Hue/Chroma/Luma, HsY, HSI etc.  It can be
# thought of as a cylindrical remapping of the YCbCr solid: the "C" term is the
# proportion of the maximum permissible chroma within the RGB gamut at a given
# hue and luma. Planes of constant Y are equiluminant.
#
# ref https://code.google.com/p/colour-space-viewer/
# ref git://anongit.kde.org/kdelibs in kdeui/colors/kcolorspaces.cpp
# ref http://blog.publicfields.net/2011/12/rgb-hue-saturation-luma.html
# ref Joblove G.H., Greenberg D., Color spaces for computer graphics.
# ref http://www.cs.rit.edu/~ncs/color/t_convert.html
# ref http://en.literateprograms.org/RGB_to_HSV_color_space_conversion_(C)
# ref http://lodev.org/cgtutor/color.html
# ref Levkowitz H., Herman G.T., "GLHS: a generalized lightness, hue, and
#     saturation color model"

# For consistency, use the same weights that the Color and Luminosity layer
# blend modes use, as also used by brushlib's Colorize brush blend mode. We
# follow http://www.w3.org/TR/compositing/ here. BT.601 YCbCr has a nearly
# identical definition of luma.

_HCY_RED_LUMA = 0.2126
_HCY_GREEN_LUMA = 0.7152
_HCY_BLUE_LUMA = 0.0722


def RGB_to_HCY(rgb):
    """RGB → HCY: R,G,B,H,C,Y ∈ [0, 1]

    :param rgb: Color expressed as an additive RGB triple.
    :type rgb: tuple (r, g, b) where 0≤r≤1, 0≤g≤1, 0≤b≤1.
    :rtype: tuple (h, c, y) where 0≤h<1, but 0≤c≤2 and 0≤y≤1.

    """
    r, g, b = rgb

    # Luma is just a weighted sum of the three components.
    y = _HCY_RED_LUMA*r + _HCY_GREEN_LUMA*g + _HCY_BLUE_LUMA*b

    # Hue. First pick a sector based on the greatest RGB component, then add
    # the scaled difference of the other two RGB components.
    p = max(r, g, b)
    n = min(r, g, b)
    d = p - n   # An absolute measure of chroma: only used for scaling.
    if n == p:
        h = 0.0
    elif p == r:
        h = (g - b)/d
        if h < 0:
            h += 6.0
    elif p == g:
        h = ((b - r)/d) + 2.0
    else:  # p==b
        h = ((r - g)/d) + 4.0
    h /= 6.0

    # Chroma, relative to the RGB gamut envelope.
    if r == g == b:
        # Avoid a division by zero for the achromatic case.
        c = 0.0
    else:
        # For the derivation, see the GLHS paper.
        c = max((y-n)/y, (p-y)/(1-y))
    return h, c, y


def HCY_to_RGB(hcy):
    """HCY → RGB: R,G,B,H,C,Y ∈ [0, 1]

    :param hcy: Color expressed as a Hue/relative-Chroma/Luma triple.
    :type hcy: tuple (h, c, y) where 0≤h<1, but 0≤c≤2 and 0≤y≤1.
    :rtype: tuple (r, g, b) where 0≤r≤1, 0≤g≤1, 0≤b≤1.

    >>> n = 32
    >>> diffs = [sum( [abs(c1-c2) for c1, c2 in
    ...                zip( HCY_to_RGB(RGB_to_HCY([r/n, g/n, b/n])),
    ...                     [r/n, g/n, b/n] ) ] )
    ...          for r in range(int(n+1))
    ...            for g in range(int(n+1))
    ...              for b in range(int(n+1))]
    >>> sum(diffs) < n*1e-6
    True

    """
    h, c, y = hcy

    if c == 0:
        return y, y, y

    h %= 1.0
    h *= 6.0
    if h < 1:
        # implies (p==r and h==(g-b)/d and g>=b)
        th = h
        tm = _HCY_RED_LUMA + _HCY_GREEN_LUMA * th
    elif h < 2:
        # implies (p==g and h==((b-r)/d)+2.0 and b<r)
        th = 2.0 - h
        tm = _HCY_GREEN_LUMA + _HCY_RED_LUMA * th
    elif h < 3:
        # implies (p==g and h==((b-r)/d)+2.0 and b>=g)
        th = h - 2.0
        tm = _HCY_GREEN_LUMA + _HCY_BLUE_LUMA * th
    elif h < 4:
        # implies (p==b and h==((r-g)/d)+4.0 and r<g)
        th = 4.0 - h
        tm = _HCY_BLUE_LUMA + _HCY_GREEN_LUMA * th
    elif h < 5:
        # implies (p==b and h==((r-g)/d)+4.0 and r>=g)
        th = h - 4.0
        tm = _HCY_BLUE_LUMA + _HCY_RED_LUMA * th
    else:
        # implies (p==r and h==(g-b)/d and g<b)
        th = 6.0 - h
        tm = _HCY_RED_LUMA + _HCY_BLUE_LUMA * th

    # Calculate the RGB components in sorted order
    if tm >= y:
        p = y + y*c*(1-tm)/tm
        o = y + y*c*(th-tm)/tm
        n = y - (y*c)
    else:
        p = y + (1-y)*c
        o = y + (1-y)*c*(th-tm)/(1-tm)
        n = y - (1-y)*c*tm/(1-tm)

    # Back to RGB order
    if h < 1:
        return (p, o, n)
    elif h < 2:
        return (o, p, n)
    elif h < 3:
        return (n, p, o)
    elif h < 4:
        return (n, o, p)
    elif h < 5:
        return (o, n, p)
    else:
        return (p, n, o)


def RGB_to_CAM16(self, rgb):
    xyz = colour.sRGB_to_XYZ(rgb)

    cam16 = colour.XYZ_to_CAM16(xyz*100.0, self.illuminant, self.L_A,
                                self.Y_b, self.surround)
    axes = list(self.cieaxes)
    cam16_vsh = np.array([getattr(cam16, axes[0]),
                          getattr(cam16, axes[1]), getattr(cam16, axes[2])])

    return cam16_vsh


def CAM16_to_RGB(self):
    if self.illuminant is None:
        self.illuminant = np.array(
            colour.xy_to_XYZ(
                colour.ILLUMINANTS['cie_2_1931']['D65']) * 100.0)
    maxcolorfulness = self.limit_purity
    axes = list(self.cieaxes)
    v, s, h = max(0.00001, self.v), max(0.00001, self.s), self.h
    # reset gamut/display flags since we don't know yet
    self.gamutexceeded = False
    self.displayexceeded = False
    # max colorfulness is optional limiter to help enforce limited palette
    # treat this like exceeding the gamut
    if maxcolorfulness is not None:
        # only return stripes for Chroma slider when limiting purity/chroma
        if self.gamutmapping == "highlightC" and self.s > maxcolorfulness:
            self.gamutexceeded = True
            return 0.5, 0.5, 0.5, 0
        if self.s > maxcolorfulness:
            self.gamutexceeded = True
            s = maxcolorfulness
    # build CAM spec
    zipped = zip(axes, (v, s, h))
    cam = colour.utilities.as_namedtuple(dict((x, y) for x, y in zipped),
                                         colour.CAM16_Specification)
    xyz = colour.CAM16_to_XYZ(cam, self.illuminant,
                              self.L_A, self.Y_b, self.surround)
    # convert CAM16 to sRGB, but it may be out of gamut which
    # we'll handle by either gamut mapping or highlighting (stripes)
    result = colour.XYZ_to_sRGB(xyz/100.0)

    # max RGB of the illuminant scale to 0-1
    # for D65 this is 1.0, 1.0, 1.0
    # other illuminants will be different, example 1.0, 0.9, 0.75
    linearRGB = colour.XYZ_to_sRGB(self.illuminant/100.0,
                                   apply_encoding_cctf=False)
    maxRGB = np.clip(colour.models.oetf_sRGB(
                     linearRGB / max(linearRGB)), 0.0, 1.0)

    # clip to our gamut and see if should gamut map or return
    x = np.clip(result, 0, maxRGB)
    if (((result <= maxRGB).all() and (result > -0.01).all())
       or self.gamutmapping is False):
        r, g, b = x
        self.cachedrgb = (r, g, b)
        if "highlight" in self.gamutmapping:
            # we are in gamut and should return the color w/ alpha 1.0
            # for the slider to render w/ transparency
            return r, g, b, 1.0
        else:
            # if we're not on a slider, just return r, g, b w/o alpha
            return r, g, b
    # only flag if negative RGB
    # this is to halt adjusters from going father for no reason
    if (result < 0).any():
        self.gamutexceeded = True

    # return zero alpha for guis and sliders to know this is out of gamut
    # this lets the "stripes" show through for these areas
    if "highlight" in self.gamutmapping:
        return 0.5, 0.5, 0.5, 0
    # print("before", result, maxRGB)
    if (result < 0).any():
        amount_to_add = (abs(min(result)) / maxRGB[np.argmin(result)]) * maxRGB
        result += amount_to_add
    if (result > maxRGB + 0.01).any():
        result /= result[np.argmax(result)] / maxRGB[np.argmax(result)]
    r, g, b = np.clip(result, 0.0, maxRGB)
    # cache the rgb for faster get_rgb calls
    # must reset this to None if changing any properties
    self.cachedrgb = (r, g, b)
    return r, g, b


# weighted geometric mean must avoid absolute zero
_WGM_EPSILON = 0.0001

# These do not sum to 1.0.  No normalization,
# but CMFs have been weighted w/ D65 SPD


def RGB_to_Spectral(rgb):
    """Converts RGB to 10 segments spectral power distribution curve.
    Upsamples to spectral primaries and sums them together into one SPD
    Based on work by Scott Allen Burns.
    """

    r, g, b = rgb
    r = max(r, _WGM_EPSILON)
    g = max(g, _WGM_EPSILON)
    b = max(b, _WGM_EPSILON)
    # Spectral primaries derived by an optimization routine devised by
    # Allen Burns. Smooth curves <= 1.0 to match XYZ

    spectral_r = r * np.array([0.009281362787953, 0.009732627042016,
                               0.011254252737167, 0.015105578649573,
                               0.024797924177217, 0.083622585502406,
                               0.977865045723212, 1.000000000000000,
                               0.999961046144372, 0.999999992756822])

    spectral_g = g * np.array([0.002854127435775, 0.003917589679914,
                               0.012132151699187, 0.748259205918013,
                               1.000000000000000, 0.865695937531795,
                               0.037477469241101, 0.022816789725717,
                               0.021747419446456, 0.021384940572308])

    spectral_b = b * np.array([0.537052150373386, 0.546646402401469,
                               0.575501819073983, 0.258778829633924,
                               0.041709923751716, 0.012662638828324,
                               0.007485593127390, 0.006766900622462,
                               0.006699764779016, 0.006676219883241])

    return np.sum([spectral_r, spectral_g, spectral_b], axis=0)


def Spectral_to_RGB(spd):
    """Converts 10 segments spectral power distribution curve to RGB.
    Based on work by Scott Allen Burns.
    """

    # Spectral_to_XYZ CMFS premultiplied with XYZ to RGB matrix
    T_MATRIX = (np.array([[0.026595621243689, 0.049779426257903,
                           0.022449850859496, -0.218453689278271,
                           -0.256894883201278, 0.445881722194840,
                           0.772365886289756, 0.194498761382537,
                           0.014038157587820, 0.007687264480513],
                          [-0.032601672674412, -0.061021043498478,
                          -0.052490001018404, 0.206659098273522,
                          0.572496335158169, 0.317837248815438,
                          -0.021216624031211, -0.019387668756117,
                          -0.001521339050858, -0.000835181622534],
                          [0.339475473216284, 0.635401374177222,
                          0.771520797089589, 0.113222640692379,
                          -0.055251113343776, -0.048222578468680,
                          -0.012966666339586, -0.001523814504223,
                          -0.000094718948810, -0.000051604594741]]))

    r, g, b = np.sum(spd * T_MATRIX, axis=1)
    return r, g, b


def Spectral_Mix_WGM(spd_a, spd_b, ratio):
    """Mixes two SPDs via weighted geomtric mean and returns an SPD.
    Based on work by Scott Allen Burns.
    """
    return spd_a**(1.0 - ratio) * spd_b**ratio


def Spectral_Mix_MULT(spd_a, spd_b):
    """Multiplies two SPDs and returns an SPD.
    Based on work by Scott Allen Burns.
    """
    return spd_a * spd_b


def CCT_to_RGB(CCT):
    """Accepts a color temperature 0-25000.  Returns RGB 0-1"""
    linearRGB = colour.XYZ_to_sRGB(colour.xy_to_XYZ(
        colour.temperature.CCT_to_xy(CCT)), apply_encoding_cctf=False)
    sRGB = colour.models.oetf_sRGB(linearRGB/max(linearRGB))
    return sRGB


def RGB_to_CCT(RGB):
    """Accepts an RGB and returns CCT 0-25000 """
    CCT = colour.temperature.xy_to_CCT_Hernandez1999(
        colour.XYZ_to_xy(colour.sRGB_to_XYZ(RGB)))
    return CCT


def color_diff(c1, c2):
    """Accepts two UIColors and returns delta_E_CAM16UCS"""
    if not isinstance(c1, CAM16Color):
        c1 = CAM16Color(color=c1)
    if not isinstance(c2, CAM16Color):
        c2 = CAM16Color(color=c2)
    c1_ucs = colour.JMh_CAM16_to_CAM16UCS([c1.v, c1.s, c1.h])
    c2_ucs = colour.JMh_CAM16_to_CAM16UCS([c2.v, c2.s, c2.h])
    return colour.difference.delta_E_CAM16UCS(c1_ucs, c2_ucs)

## Module testing


def _test():
    """Run all doctests in this module"""
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _test()
