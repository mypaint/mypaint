# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from math import floor, ceil
import colorsys

from gtk import gdk # for gdk_pixbuf stuff
import mypaintlib


class Rect:
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h))
    def empty(self):
        return self.w == 0 or self.h == 0
    def copy(self):
        return Rect(self.x, self.y, self.w, self.h)
    def expand(self, border):
        self.w += 2*border
        self.h += 2*border
        self.x -= border
        self.y -= border
    def __contains__(self, other):
        return (
            other.x >= self.x and
            other.y >= self.y and
            other.x + other.w <= self.x + self.w and
            other.y + other.h <= self.y + self.h
            )
    def __eq__(self, other):
        return tuple(self) == tuple(other)
    def overlaps(r1, r2):
        if max(r1.x, r2.x) >= min(r1.x+r1.w, r2.x+r2.w): return False
        if max(r1.y, r2.y) >= min(r1.y+r1.h, r2.y+r2.h): return False
        return True
    def expandToIncludePoint(self, x, y):
        if self.w == 0 or self.h == 0:
            self.x = x
            self.y = y
            self.w = 1
            self.h = 1
            return
        if x < self.x:
            self.w += self.x - x
            self.x = x
        if y < self.y:
            self.h += self.y - y
            self.y = y
        if x > self.x + self.w - 1:
            self.w += x - (self.x + self.w - 1)
        if y > self.y + self.h - 1:
            self.h += y - (self.y + self.h - 1)
    def expandToIncludeRect(self, other):
        if other.empty(): return
        self.expandToIncludePoint(other.x, other.y)
        self.expandToIncludePoint(other.x + other.w - 1, other.y + other.h - 1)
    def __repr__(self):
        return 'Rect(%d, %d, %d, %d)' % (self.x, self.y, self.w, self.h)

def iter_rect(x, y, w, h):
    assert w>=0 and h>=0
    for yy in xrange(y, y+h):
        for xx in xrange(x, x+w):
            yield (xx, yy)


def rotated_rectangle_bbox(corners):
    list_y = [y for (x, y) in corners]
    list_x = [x for (x, y) in corners]
    x1 = int(floor(min(list_x)))
    y1 = int(floor(min(list_y)))
    x2 = int(ceil(max(list_x)))
    y2 = int(ceil(max(list_y)))
    return x1, y1, x2-x1+1, y2-y1+1

def clamp(x, lo, hi):
    if x < lo: return lo
    if x > hi: return hi
    return x


def gdkpixbuf2numpy(pixbuf):
    # workaround for pygtk still returning Numeric instead of numpy arrays
    # (see gdkpixbuf2numpy.hpp)
    arr = pixbuf.get_pixels_array()
    return mypaintlib.gdkpixbuf_numeric2numpy(arr)

def pixbuf_thumbnail(src, w, h):
    """
    Creates a centered thumbnail of a gdk.pixbuf.
    """
    src_w = src.get_width()
    src_h = src.get_height()

    w2, h2 = src_w, src_h
    if w2 > w:
        w2 = w
        h2 = h2*w/src_w
    if h2 > h:
        w2 = w2*h/src_h
        h2 = h
    assert w2 <= w and h2 <= h
    src2 = src.scale_simple(w2, h2, gdk.INTERP_BILINEAR)
    
    dst = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, w, h)
    dst.fill(0xffffffff) # white background

    src2.copy_area(0, 0, w2, h2, dst, (w-w2)/2, (h-h2)/2)
    return dst

def rgb_to_hsv(r, g, b):
    r = clamp(r, 0.0, 1.0)
    g = clamp(g, 0.0, 1.0)
    b = clamp(b, 0.0, 1.0)
    return colorsys.rgb_to_hsv(r, g, b)

def hsv_to_rgb(h, s, v):
    h = clamp(h, 0.0, 1.0)
    s = clamp(s, 0.0, 1.0)
    v = clamp(v, 0.0, 1.0)
    return colorsys.hsv_to_rgb(h, s, v)

def indent_etree(elem, level=0):
    """
    Indent an XML etree. This does not seem to come with python?
    Source: http://effbot.org/zone/element-lib.htm#prettyprint
    """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_etree(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

if __name__ == '__main__':
    big = Rect(-3, 2, 180, 222)
    a = Rect(0, 10, 5, 15)
    b = Rect(2, 10, 1, 15)
    c = Rect(-1, 10, 1, 30)
    assert b in a
    assert a not in b
    assert a in big and b in big and c in big
    for r in [a, b, c]:
        assert r in big
        assert big.overlaps(r)
        assert r.overlaps(big)
    assert a.overlaps(b)
    assert b.overlaps(a)
    assert not a.overlaps(c)
    assert not c.overlaps(a)


    r1 = Rect( -40, -40, 5, 5 )
    r2 = Rect( -40-1, -40+5, 5, 500 )
    assert not r1.overlaps(r2)
    assert not r2.overlaps(r1)
    r1.y += 1
    assert r1.overlaps(r2)
    assert r2.overlaps(r1)
    r1.x += 999
    assert not r1.overlaps(r2)
    assert not r2.overlaps(r1)

    print 'Tests passed.'

