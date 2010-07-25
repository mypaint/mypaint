# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from math import floor, ceil
import colorsys, urllib, gc

from gtk import gdk # for gdk_pixbuf stuff
import mypaintlib

import hashlib
import os
import zipfile

try:
    from json import dumps as json_dumps, loads as json_loads
    print "builtin python 2.6 json support"
except ImportError:
    try:
        from cjson import encode as json_dumps, decode as json_loads
        print "external python-cjson"
    except ImportError:
        try:
            from json import write as json_dumps, read as json_loads
            print "external python-json"
        except ImportError:
            try:
                from simplejson import dumps as json_dumps, loads as json_loads
                print "external python-simplejson"
            except ImportError:
                raise ImportError("Could jot import json. You either need to use python >= 2.6 or install one of python-cjson, python-json or python-simplejson.")

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

def get_freedesktop_thumbnail(filename):
    """
    Tries to fetch a thumbnail from ~/.thumbnails.
    If there is no thumbnail for the specified filename,
    a new thumbnail will be generated and stored according to the FDO spec.
    A thumbnail will also get regenerated if the MTimes (as in "modified")
    of thumbnail and original image do not match.
    """
    file_hash = hashlib.md5('file://'+filename).hexdigest()
    tb_filename_normal = os.path.join(os.path.expanduser('~/.thumbnails/normal'), file_hash) + '.png'
    tb_filename_large = os.path.join(os.path.expanduser('~/.thumbnails/large'), file_hash) + '.png'
    if os.path.isfile(tb_filename_normal):
        pixbuf = gdk.pixbuf_new_from_file(tb_filename_normal)
    elif os.path.isfile(tb_filename_large):
        pixbuf = gdk.pixbuf_new_from_file(tb_filename_large)
    else:
        pixbuf = None
    pixbuf = save_freedesktop_thumbnail(pixbuf, filename) # save thumbnail or regenerate if MTimes do not match
    return pixbuf

def save_freedesktop_thumbnail(pixbuf, filename):
    """
    Saves a thumbnail according to the FDO spec.
    """
    file_hash = hashlib.md5('file://'+filename).hexdigest()
    tb_filename_normal = os.path.join(os.path.expanduser('~/.thumbnails/normal'), file_hash) + '.png'
    file_mtime = str(int(os.stat(filename).st_mtime))
    if (not os.path.isfile(tb_filename_normal)) or (not pixbuf) or (file_mtime != pixbuf.get_option("tEXt::Thumb::MTime")):
        pixbuf = get_pixbuf(filename)
        if pixbuf:
            pixbuf = scale_proportionally(pixbuf, 128,128)
            pixbuf.save(tb_filename_normal, 'png', {"tEXt::Thumb::MTime" : file_mtime, "tEXt::Thumb::URI" : ('file://'+filename)})
            return pixbuf
    else:
        return pixbuf

def get_pixbuf(filename):
    try:
        if os.path.splitext(filename)[1].lower() == ".ora":
            ora = zipfile.ZipFile(file(filename))
            data = ora.read("Thumbnails/thumbnail.png")
            loader = gdk.PixbufLoader("png")
            loader.write(data)
            loader.close()
            pixbuf = loader.get_pixbuf()
            return pixbuf
        else:
            pixbuf = gdk.pixbuf_new_from_file(filename)
            return pixbuf;
    except:
        pass

def scale_proportionally(pixbuf, w, h, shrink_only=True):
    width, height = pixbuf.get_width(), pixbuf.get_height()
    scale = min(w / float(width), h / float(height))
    if shrink_only and scale >= 1:
        return pixbuf
    new_width, new_height = int(width * scale), int(height * scale)
    if new_width > 0 and new_height > 0:
        pixbuf = pixbuf.scale_simple(new_width, new_height, gdk.INTERP_BILINEAR)
    return pixbuf

def pixbuf_thumbnail(src, w, h, alpha=False):
    """
    Creates a centered thumbnail of a gdk.pixbuf.
    """
    src2 = scale_proportionally(src, w, h)
    w2, h2 = src2.get_width(), src2.get_height()
    dst = gdk.Pixbuf(gdk.COLORSPACE_RGB, alpha, 8, w, h)
    if alpha:
        dst.fill(0xffffff00) # transparent background
    else:
        dst.fill(0xffffffff) # white background
    src2.copy_area(0, 0, w2, h2, dst, (w-w2)/2, (h-h2)/2)
    return dst

def get_file_path_from_dnd_dropped_uri(uri):
    # code from http://faq.pygtk.org/index.py?req=show&file=faq23.031.htp
    # get the path to file
    path = ""
    if uri.startswith('file:\\\\\\'): # windows
        path = uri[8:] # 8 is len('file:///')
    elif uri.startswith('file://'): # nautilus, rox
        path = uri[7:] # 7 is len('file://')
    elif uri.startswith('file:'): # xffm
        path = uri[5:] # 5 is len('file:')
        
    path = urllib.url2pathname(path) # escape special chars
    path = path.strip('\r\n\x00') # remove \r\n and NULL
    
    return path

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

def run_garbage_collector():
    print 'MEM: garbage collector run, collected', gc.collect(), 'objects'
    print 'MEM: gc.garbage contains', len(gc.garbage), 'items of uncollectible garbage'

old_stats = []
def record_memory_leak_status(print_diff=False):
    run_garbage_collector()
    print 'MEM: collecting info (can take some time)...'
    new_stats = []
    for obj in gc.get_objects():
        if 'A' <= getattr(obj, '__name__', ' ')[0] <= 'Z':
            cnt = len(gc.get_referrers(obj))
            new_stats.append((obj.__name__ + ' ' + str(obj), cnt))
    new_stats.sort()
    print 'MEM: ...done collecting.'
    global old_stats
    if old_stats:
        if print_diff:
            d = {}
            for obj, cnt in old_stats:
                d[obj] = cnt
            for obj, cnt in new_stats:
                cnt_old = d.get(obj, 0)
                if cnt != cnt_old:
                    print 'MEM: DELTA %+d %s' % (cnt - cnt_old, obj)
    else:
        print 'MEM: Stored stats to compare with the next info collection.'
    old_stats = new_stats

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

