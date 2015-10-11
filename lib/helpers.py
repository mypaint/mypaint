# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from math import floor, ceil, isnan
import os
import sys
import hashlib
import zipfile
import colorsys
import urllib
import gc
import numpy
import logging
logger = logging.getLogger(__name__)

from gi.repository import GdkPixbuf
from gi.repository import GLib
from gettext import gettext as _

import mypaintlib
import lib.pixbuf
import lib.glib


try:
    from json import dumps as json_dumps_builtin, loads as json_loads
    logger.debug("Using builtin python 2.6 json support")
    json_dumps = lambda obj: json_dumps_builtin(obj, indent=2)
except ImportError:
    try:
        from cjson import encode as json_dumps, decode as json_loads
        logger.debug("Using external python-cjson")
    except ImportError:
        try:
            from json import write as json_dumps, read as json_loads
            logger.debug("Using external python-json")
        except ImportError:
            try:
                from simplejson import dumps as json_dumps, loads as json_loads
                logger.debug("Using external python-simplejson")
            except ImportError:
                raise ImportError("Could not import json. You either need to use python >= 2.6 or install one of python-cjson, python-json or python-simplejson.")


class Rect (object):
    """Representation of a rectangular area.

    We use our own class here because (around GTK 3.18.x, at least) it's
    less subject to typelib omissions than Gdk.Rectangle.

    Ref: https://github.com/mypaint/mypaint/issues/437

    """

    def __init__(self, x=0, y=0, w=0, h=0):
        """Initializes, with optional location and dimensions."""
        object.__init__(self)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @classmethod
    def new_from_gdk_rectangle(cls, gdk_rect):
        """Creates a new Rect based on a Gdk.Rectangle."""
        return Rect(
            x = gdk_rect.x,
            y = gdk_rect.x,
            w = gdk_rect.width,
            h = gdk_rect.height,
        )

    def __iter__(self):
        """Allows iteration, and thus casting to tuples and lists.

        The sequence returned is always 4 items long, and in the order
        x, y, w, h.

        """
        return iter((self.x, self.y, self.w, self.h))

    def empty(self):
        """Returns true if the rectangle has zero area."""
        return self.w == 0 or self.h == 0

    def copy(self):
        """Copies and returns the Rect."""
        return Rect(self.x, self.y, self.w, self.h)

    def expand(self, border):
        """Expand the area by a fixed border size."""
        self.w += 2*border
        self.h += 2*border
        self.x -= border
        self.y -= border

    def contains(self, other):
        """Returns true if this rectangle entirely contains another."""
        return (
            other.x >= self.x and
            other.y >= self.y and
            other.x + other.w <= self.x + self.w and
            other.y + other.h <= self.y + self.h
        )

    def __eq__(self, other):
        """Returns true if this rectangle is identical to another."""
        try:
            return tuple(self) == tuple(other)
        except TypeError:  # e.g. comparison to None
            return False

    def overlaps(r1, r2):
        """Returns true if this rectangle intersects another."""
        if max(r1.x, r2.x) >= min(r1.x+r1.w, r2.x+r2.w):
            return False
        if max(r1.y, r2.y) >= min(r1.y+r1.h, r2.y+r2.h):
            return False
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
        if other.empty():
            return
        self.expandToIncludePoint(other.x, other.y)
        self.expandToIncludePoint(other.x + other.w - 1, other.y + other.h - 1)

    def __repr__(self):
        return 'Rect(%d, %d, %d, %d)' % (self.x, self.y, self.w, self.h)


def rotated_rectangle_bbox(corners):
    list_y = [y for (x, y) in corners]
    list_x = [x for (x, y) in corners]
    x1 = int(floor(min(list_x)))
    y1 = int(floor(min(list_y)))
    x2 = int(floor(max(list_x)))
    y2 = int(floor(max(list_y)))
    return x1, y1, x2-x1+1, y2-y1+1


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def gdkpixbuf2numpy(pixbuf):
    # gdk.Pixbuf.get_pixels_array() is no longer wrapped; use our own
    # implementation.
    return mypaintlib.gdkpixbuf_get_pixels_array(pixbuf)
    ## Can't do the following - the created generated array is immutable
    #w, h = pixbuf.get_width(), pixbuf.get_height()
    #assert pixbuf.get_bits_per_sample() == 8
    #assert pixbuf.get_has_alpha()
    #assert pixbuf.get_n_channels() == 4
    #arr = numpy.frombuffer(pixbuf.get_pixels(), dtype=numpy.uint8)
    #arr = arr.reshape(h, w, 4)
    #return arr


def freedesktop_thumbnail(filename, pixbuf=None):
    """Fetch or (re-)generate the thumbnail in $XDG_CACHE_HOME/thumbnails.

    If there is no thumbnail for the specified filename, a new
    thumbnail will be generated and stored according to the FDO spec.
    A thumbnail will also get regenerated if the MTimes (as in "modified")
    of thumbnail and original image do not match.

    When pixbuf is given, it will be scaled and used as thumbnail
    instead of reading the file itself. In this case the file is still
    accessed to get its mtime, so this method must not be called if
    the file is still open.

    Returns the large (256x256) thumbnail.
    """

    uri = lib.glib.filename_to_uri(os.path.abspath(filename))
    logger.debug("thumb: uri=%r", uri)
    file_hash = hashlib.md5(uri).hexdigest()

    cache_dir = lib.glib.get_user_cache_dir()
    base_directory = os.path.join(cache_dir, 'thumbnails')

    directory = os.path.join(base_directory, 'normal')
    tb_filename_normal = os.path.join(directory, file_hash) + '.png'

    if not os.path.exists(directory):
        os.makedirs(directory, 0700)
    directory = os.path.join(base_directory, 'large')
    tb_filename_large = os.path.join(directory, file_hash) + '.png'
    if not os.path.exists(directory):
        os.makedirs(directory, 0700)

    file_mtime = str(int(os.stat(filename).st_mtime))

    save_thumbnail = True

    if filename.lower().endswith('.ora'):
        # don't bother with normal (128x128) thumbnails when we can
        # get a large one (256x256) from the file in an instant
        acceptable_tb_filenames = [tb_filename_large]
    else:
        # prefer the large thumbnail, but accept the normal one if
        # available, for the sake of performance
        acceptable_tb_filenames = [tb_filename_large, tb_filename_normal]

    for fn in acceptable_tb_filenames:
        if not pixbuf and os.path.isfile(fn):
            # use the largest stored thumbnail that isn't obsolete
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(fn)
            if file_mtime == pixbuf.get_option("tEXt::Thumb::MTime"):
                save_thumbnail = False
            else:
                pixbuf = None

    if not pixbuf:
        # try to load a pixbuf from the file
        pixbuf = get_pixbuf(filename)

    if pixbuf:
        pixbuf = scale_proportionally(pixbuf, 256, 256)
        if save_thumbnail:
            png_opts = {"tEXt::Thumb::MTime": file_mtime,
                        "tEXt::Thumb::URI": uri}
            logger.debug("thumb: png_opts=%r", png_opts)
            lib.pixbuf.save(
                pixbuf,
                tb_filename_large,
                type='png',
                **png_opts
            )
            logger.debug("thumb: saved large (256x256) thumbnail to %r",
                         tb_filename_large)
            # save normal size too, in case some implementations don't
            # bother with large thumbnails
            pixbuf_normal = scale_proportionally(pixbuf, 128, 128)
            lib.pixbuf.save(
                pixbuf_normal,
                tb_filename_normal,
                type='png',
                **png_opts
            )
            logger.debug("thumb: saved normal (128x128) thumbnail to %r",
                         tb_filename_normal)
    return pixbuf


def get_pixbuf(filename):
    """Loads a thumbnail pixbuf loaded from a file.

    :param filename: File to get a thumbnail image from.
    :returns: Thumbnail puixbuf, or None.
    :rtype: GdkPixbuf.Pixbuf

    >>> get_pixbuf("pixmaps/mypaint_logo.png")  # doctest: +ELLIPSIS
    <Pixbuf ...>
    >>> get_pixbuf("tests/bigimage.ora")  # doctest: +ELLIPSIS
    <Pixbuf ...>
    >>> get_pixbuf("desktop/icons")   # Non-files return None.
    >>> get_pixbuf("pixmaps/nonexistent.foo")  # None also.

    """
    if not os.path.isfile(filename):
        logger.debug("No thumb pixbuf for %r: not a file", filename)
        return None
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".ora":
        thumb_entry = "Thumbnails/thumbnail.png"
        try:
            orazip = zipfile.ZipFile(filename)
            pixbuf = lib.pixbuf.load_from_zipfile(orazip, thumb_entry)
        except:
            logger.exception(
                "Failed to read %r entry of %r",
                thumb_entry,
                filename,
            )
            return None
        if not pixbuf:
            logger.error(
                "Failed to parse %r entry of %r",
                thumb_entry,
                filename,
            )
            return None
        logger.debug(
            "Parsed %r entry of %r successfully",
            thumb_entry,
            filename,
        )
        return pixbuf
    else:
        try:
            return lib.pixbuf.load_from_file(filename)
        except:
            logger.exception(
                "Failed to load thumbnail pixbuf from %r",
                filename,
            )
            return None


def scale_proportionally(pixbuf, w, h, shrink_only=True):
    width, height = pixbuf.get_width(), pixbuf.get_height()
    scale = min(w / float(width), h / float(height))
    if shrink_only and scale >= 1:
        return pixbuf
    new_width, new_height = int(width * scale), int(height * scale)
    new_width = max(new_width, 1)
    new_height = max(new_height, 1)
    return pixbuf.scale_simple(new_width, new_height,
                               GdkPixbuf.InterpType.BILINEAR)


def pixbuf_thumbnail(src, w, h, alpha=False):
    """Creates a centered thumbnail of a GdkPixbuf.
    """
    src2 = scale_proportionally(src, w, h)
    w2, h2 = src2.get_width(), src2.get_height()
    dst = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, alpha, 8, w, h)
    if alpha:
        dst.fill(0xffffff00)  # transparent background
    else:
        dst.fill(0xffffffff)  # white background
    src2.composite(dst, (w-w2)/2, (h-h2)/2, w2, h2, (w-w2)/2, (h-h2)/2, 1, 1,
                   GdkPixbuf.InterpType.BILINEAR, 255)
    return dst


def rgb_to_hsv(r, g, b):
    assert not isnan(r)
    r = clamp(r, 0.0, 1.0)
    g = clamp(g, 0.0, 1.0)
    b = clamp(b, 0.0, 1.0)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    assert not isnan(h)
    return h, s, v


def hsv_to_rgb(h, s, v):
    h = clamp(h, 0.0, 1.0)
    s = clamp(s, 0.0, 1.0)
    v = clamp(v, 0.0, 1.0)
    return colorsys.hsv_to_rgb(h, s, v)


def zipfile_writestr(z, arcname, data):
    """Write a string into a zipfile entry, with standard permissions

    :param zipfile.ZipFile z: A zip file open for write.
    :param unicode arcname: Name of the file entry to add.
    :param bytes data: Content to add.

    Work around bad permissions with the standard
    `zipfile.Zipfile.writestr`: http://bugs.python.org/issue3394. The
    original zero-permissions defect was fixed upstream, but do we want
    more public permissions than the fix's 0600?

    """
    zi = zipfile.ZipInfo(arcname)
    zi.external_attr = 0o644 << 16  # wider perms, should match z.write()
    zi.external_attr |= 0o100000 << 16  # regular file
    z.writestr(zi, data)


def run_garbage_collector():
    logger.info('MEM: garbage collector run, collected %d objects',
                gc.collect())
    logger.info('MEM: gc.garbage contains %d items of uncollectible garbage',
                len(gc.garbage))

old_stats = []


def record_memory_leak_status(print_diff=False):
    run_garbage_collector()
    logger.info('MEM: collecting info (can take some time)...')
    new_stats = []
    for obj in gc.get_objects():
        if 'A' <= getattr(obj, '__name__', ' ')[0] <= 'Z':
            cnt = len(gc.get_referrers(obj))
            new_stats.append((obj.__name__ + ' ' + str(obj), cnt))
    new_stats.sort()
    logger.info('MEM: ...done collecting.')
    global old_stats
    if old_stats:
        if print_diff:
            d = {}
            for obj, cnt in old_stats:
                d[obj] = cnt
            for obj, cnt in new_stats:
                cnt_old = d.get(obj, 0)
                if cnt != cnt_old:
                    logger.info('MEM: DELTA %+d %s', cnt - cnt_old, obj)
    else:
        logger.info('MEM: Stored stats to compare with the next '
                    'info collection.')
    old_stats = new_stats


def fmt_time_period_abbr(t):
    """Get a localized abbreviated minutes+seconds string

    :param int t: A positive number of seconds
    :returns: short localized string
    :rtype: unicode

    The result looks like like "<minutes>m<seconds>s",
    or just "<seconds>s".

    """
    if t < 0:
        raise ValueError("Parameter t cannot be negative")
    days = int(t / (24*60*60))
    hours = int(t - days*24*60*60) / (60*60)
    minutes = int(t - hours*60*60) / 60
    seconds = int(t - minutes*60)
    #TRANSLATORS: I'm assuming that time periods in places where
    #TRANSLATORS: abbreviations make sense don't need ngettext()
    if t > 24*60*60:
        template = _("{days}d{hours}h")
    elif t > 60*60:
        template = _("{hours}h{minutes}m")
    elif t > 60:
        template = _("{minutes}m{seconds}s")
    else:
        template = _("{seconds}s")
    return template.format(
        days = days,
        hours = hours,
        minutes = minutes,
        seconds = seconds,
    )



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

    r1 = Rect(-40, -40, 5, 5)
    r2 = Rect(-40-1, -40+5, 5, 500)
    assert not r1.overlaps(r2)
    assert not r2.overlaps(r1)
    r1.y += 1
    assert r1.overlaps(r2)
    assert r2.overlaps(r1)
    r1.x += 999
    assert not r1.overlaps(r2)
    assert not r2.overlaps(r1)

    import doctest
    doctest.testmod()

    print 'Tests passed.'
