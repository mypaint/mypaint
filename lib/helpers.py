# This file is part of MyPaint.
# Copyright (C) 2012-2019 by the MyPaint Development Team.
# Copyright (C) 2007-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function

import itertools
from math import floor, isnan
import os
import hashlib
import zipfile
import colorsys
import gc
import logging
import sys

from lib.gibindings import GdkPixbuf
from lib.gettext import C_

from . import mypaintlib
import lib.pixbuf
import lib.glib
from lib.pycompat import PY2
from lib.pycompat import unicode

logger = logging.getLogger(__name__)


class Rect (object):
    """Representation of a rectangular area.

    We use our own class here because (around GTK 3.18.x, at least) it's
    less subject to typelib omissions than Gdk.Rectangle.

    Ref: https://github.com/mypaint/mypaint/issues/437

    >>> big = Rect(-3, 2, 180, 222)
    >>> a = Rect(0, 10, 5, 15)
    >>> b = Rect(2, 10, 1, 15)
    >>> c = Rect(-1, 10, 1, 30)
    >>> a.contains(b)
    True
    >>> not b.contains(a)
    True
    >>> [big.contains(r) for r in [a, b, c]]
    [True, True, True]
    >>> [big.overlaps(r) for r in [a, b, c]]
    [True, True, True]
    >>> [r.overlaps(big) for r in [a, b, c]]
    [True, True, True]
    >>> a.overlaps(b) and b.overlaps(a)
    True
    >>> (not a.overlaps(c)) and (not c.overlaps(a))
    True

    >>> r1 = Rect(-40, -40, 5, 5)
    >>> r2 = Rect(-40 - 1, - 40 + 5, 5, 500)
    >>> assert not r1.overlaps(r2)
    >>> assert not r2.overlaps(r1)
    >>> r1.y += 1
    >>> assert r1.overlaps(r2)
    >>> assert r2.overlaps(r1)
    >>> i = r1.intersection(r2)
    >>> assert i.h == 1
    >>> assert i.w == 4
    >>> assert i.x == r1.x
    >>> assert i.y == r2.y
    >>> r1.x += 999
    >>> assert not r1.overlaps(r2)
    >>> assert not r2.overlaps(r1)

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
            y = gdk_rect.y,
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
        self.w += 2 * border
        self.h += 2 * border
        self.x -= border
        self.y -= border

    def expanded(self, border):
        """Return a copy of this rectangle, expanded by a fixed border size."""
        copy = self.copy()
        copy.expand(border)
        return copy

    def contains(self, other):
        """Returns true if this rectangle entirely contains another."""
        return (
            other.x >= self.x and
            other.y >= self.y and
            other.x + other.w <= self.x + self.w and
            other.y + other.h <= self.y + self.h
        )

    def contains_pixel(self, x, y):
        """Checks if pixel coordinates lie inside this rectangle"""
        return (self.x <= x <= self.x + self.w - 1 and
                self.y <= y <= self.y + self.h - 1)

    def clamped_point(self, x, y):
        """Returns the given point, clamped to the area of this rectangle"""
        cx = clamp(x, self.x, self.x + self.w)
        cy = clamp(y, self.y, self.y + self.h)
        return cx, cy

    def __eq__(self, other):
        """Returns true if this rectangle is identical to another."""
        try:
            return tuple(self) == tuple(other)
        except TypeError:  # e.g. comparison to None
            return False

    def overlaps(self, r2):
        """Returns true if this rectangle intersects another."""
        if max(self.x, r2.x) >= min(self.x + self.w, r2.x + r2.w):
            return False
        if max(self.y, r2.y) >= min(self.y + self.h, r2.y + r2.h):
            return False
        return True

    def expand_to_include_point(self, x, y):
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

    def expand_to_include_rect(self, other):
        if other.empty():
            return
        self.expand_to_include_point(other.x, other.y)
        self.expand_to_include_point(
            other.x + other.w - 1,
            other.y + other.h - 1,
        )

    def intersection(self, other):
        """Creates new Rect for the intersection with another
        If the rectangles do not intersect, None is returned
        :rtype: Rect
        """
        if not self.overlaps(other):
            return None

        x = max(self.x, other.x)
        y = max(self.y, other.y)
        rx = min(self.x + self.w, other.x + other.w)
        ry = min(self.y + self.h, other.y + other.h)
        return Rect(x, y, rx - x, ry - y)

    def __repr__(self):
        return 'Rect(%d, %d, %d, %d)' % (self.x, self.y, self.w, self.h)


def coordinate_bounds(tile_coords):
    """Find min/max x, y bounds of (x, y) pairs

    If the input iterable's length is 0, None is returned
    :param iterable tile_coords: iterable of (x, y)
    :returns: (min x, min y, max x, max y) or None
    :rtype: (int, int, int, int) | None

    >>> coordinate_bounds([])
    >>> coordinate_bounds([(0, 0)])
    (0, 0, 0, 0)
    >>> coordinate_bounds([(-10, 5), (0, 0)])
    (-10, 0, 0, 5)
    >>> coordinate_bounds([(3, 5), (0, 0), (-3, 7), (20, -10)])
    (-3, -10, 20, 7)
    """
    lim = float('inf')
    min_x, min_y, max_x, max_y = lim, lim, -lim, -lim
    # Determine minima and maxima in one pass
    for x, y in tile_coords:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    if min_x == lim:
        return None
    else:
        return min_x, min_y, max_x, max_y


def rotated_rectangle_bbox(corners):
    list_y = [y for (x, y) in corners]
    list_x = [x for (x, y) in corners]
    x1 = int(floor(min(list_x)))
    y1 = int(floor(min(list_y)))
    x2 = int(floor(max(list_x)))
    y2 = int(floor(max(list_y)))
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


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
    # Can't do the following - the created generated array is immutable
    # w, h = pixbuf.get_width(), pixbuf.get_height()
    # assert pixbuf.get_bits_per_sample() == 8
    # assert pixbuf.get_has_alpha()
    # assert pixbuf.get_n_channels() == 4
    # arr = np.frombuffer(pixbuf.get_pixels(), dtype=np.uint8)
    # arr = arr.reshape(h, w, 4)
    # return arr


def freedesktop_thumbnail(filename, pixbuf=None, force=False):
    """Fetch or (re-)generate the thumbnail in $XDG_CACHE_HOME/thumbnails.

    If there is no thumbnail for the specified filename, a new
    thumbnail will be generated and stored according to the FDO spec.
    A thumbnail will also get regenerated if the file modification times
    of thumbnail and original image do not match.

    :param GdkPixbuf.Pixbuf pixbuf: Thumbnail to save, optional.
    :param bool force: Force rengeneration (skip mtime checks).
    :returns: the large (256x256) thumbnail, or None.
    :rtype: GdkPixbuf.Pixbuf

    When pixbuf is given, it will be scaled and used as thumbnail
    instead of reading the file itself. In this case the file is still
    accessed to get its mtime, so this method must not be called if
    the file is still open.

    >>> image = "svg/thumbnail-test-input.svg"
    >>> p1 = freedesktop_thumbnail(image, force=True)
    >>> isinstance(p1, GdkPixbuf.Pixbuf)
    True
    >>> p2 = freedesktop_thumbnail(image)
    >>> isinstance(p2, GdkPixbuf.Pixbuf)
    True
    >>> p2.to_string() == p1.to_string()
    True
    >>> p2.get_width() == p2.get_height() == 256
    True

    """

    uri = lib.glib.filename_to_uri(os.path.abspath(filename))
    logger.debug("thumb: uri=%r", uri)
    if not isinstance(uri, bytes):
        uri = uri.encode("utf-8")
    file_hash = hashlib.md5(uri).hexdigest()

    cache_dir = lib.glib.get_user_cache_dir()
    base_directory = os.path.join(cache_dir, u'thumbnails')

    directory = os.path.join(base_directory, u'normal')
    tb_filename_normal = os.path.join(directory, file_hash) + u'.png'

    if not os.path.exists(directory):
        os.makedirs(directory, 0o700)
    directory = os.path.join(base_directory, u'large')
    tb_filename_large = os.path.join(directory, file_hash) + u'.png'
    if not os.path.exists(directory):
        os.makedirs(directory, 0o700)

    file_mtime = str(int(os.stat(filename).st_mtime))

    save_thumbnail = True

    if filename.lower().endswith(u'.ora'):
        # don't bother with normal (128x128) thumbnails when we can
        # get a large one (256x256) from the file in an instant
        acceptable_tb_filenames = [tb_filename_large]
    else:
        # prefer the large thumbnail, but accept the normal one if
        # available, for the sake of performance
        acceptable_tb_filenames = [tb_filename_large, tb_filename_normal]

    # Use the largest stored thumbnail that isn't obsolete,
    # Unless one was passed in,
    # or regeneration is being forced.
    for fn in acceptable_tb_filenames:
        if pixbuf or force or (not os.path.isfile(fn)):
            continue
        try:
            pixbuf = lib.pixbuf.load_from_file(fn)
        except Exception as e:
            logger.warning(
                u"thumb: cache file %r looks corrupt (%r). "
                u"It will be regenerated.",
                fn, unicode(e),
            )
            pixbuf = None
        else:
            assert pixbuf is not None
            if file_mtime == pixbuf.get_option("tEXt::Thumb::MTime"):
                save_thumbnail = False
                break
            else:
                pixbuf = None

    # Try to load a pixbuf from the file, if we still need one.
    if not pixbuf:
        pixbuf = get_pixbuf(filename)

    # Update the fd.o thumbs cache.
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

    # Return the 256x256 scaled version.
    return pixbuf


def get_pixbuf(filename):
    """Loads a thumbnail pixbuf loaded from a file.

    :param filename: File to get a thumbnail image from.
    :returns: Thumbnail puixbuf, or None.
    :rtype: GdkPixbuf.Pixbuf

    >>> p = get_pixbuf("pixmaps/mypaint_logo.png")
    >>> isinstance(p, GdkPixbuf.Pixbuf)
    True
    >>> p = get_pixbuf("tests/bigimage.ora")
    >>> isinstance(p, GdkPixbuf.Pixbuf)
    True
    >>> get_pixbuf("desktop/icons") is None
    True
    >>> get_pixbuf("pixmaps/nonexistent.foo") is None
    True

    """
    if not os.path.isfile(filename):
        logger.debug("No thumb pixbuf for %r: not a file", filename)
        return None
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".ora":
        thumb_entry = "Thumbnails/thumbnail.png"
        try:
            with zipfile.ZipFile(filename) as orazip:
                pixbuf = lib.pixbuf.load_from_zipfile(orazip, thumb_entry)
        except Exception:
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
        except Exception:
            logger.exception(
                "Failed to load thumbnail pixbuf from %r",
                filename,
            )
            return None


def scale_proportionally(pixbuf, w, h, shrink_only=True):
    width, height = pixbuf.get_width(), pixbuf.get_height()
    scale = min(w / width, h / height)
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
    src2.composite(
        dst,
        (w - w2) // 2, (h - h2) // 2,
        w2, h2,
        (w - w2) // 2, (h - h2) // 2,
        1, 1,
        GdkPixbuf.InterpType.BILINEAR,
        255,
    )
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


def transform_hsv(hsv, eotf):
    r, g, b = hsv_to_rgb(*hsv)
    return rgb_to_hsv(r**eotf, g**eotf, b**eotf)


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


def utf8(string):
    """Return the input as bytes encoded by utf-8"""
    return string.encode('utf-8')


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
    days = int(t // (24 * 60 * 60))
    hours = int(t - days * 24 * 60 * 60) // (60 * 60)
    minutes = int(t - hours * 60 * 60) // 60
    seconds = int(t - minutes * 60)
    if t > 24 * 60 * 60:
        # TRANSLATORS: Assumption for all "Time period abbreviations":
        # TRANSLATORS: they don't need ngettext (to support plural/singular)
        template = C_("Time period abbreviations", u"{days}d{hours}h")
    elif t > 60 * 60:
        template = C_("Time period abbreviations", u"{hours}h{minutes}m")
    elif t > 60:
        template = C_("Time period abbreviation", u"{minutes}m{seconds}s")
    else:
        template = C_("Time period abbreviation", u"{seconds}s")
    return template.format(
        days = days,
        hours = hours,
        minutes = minutes,
        seconds = seconds,
    )


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks

    :param iterable: An iterable
    :param int n: How many items to chunk the iterator by
    :param fillvalue: Filler value when iterable length isn't a multiple of n
    :returns: An iterable with tuples n items from the source iterable
    :rtype: iterable

    >>> actual = grouper('ABCDEFG', 3, fillvalue='x')
    >>> expected = [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]
    >>> [a_val == e_val for a_val, e_val in zip(actual, expected)]
    [True, True, True]
    """
    args = [iter(iterable)] * n
    if PY2:
        return itertools.izip_longest(*args, fillvalue=fillvalue)
    else:
        return itertools.zip_longest(*args, fillvalue=fillvalue)


def casefold(s):
    """Converts a unicode string into a case-insensitively comparable form.

    Forward-compat marker for things that should be .casefold() in
    Python 3, but which need to be .lower() in Python2.

    :param str s: The string to convert.
    :rtype: str
    :returns: The converted string.

    >>> casefold("Xyz") == u'xyz'
    True

    """
    if sys.version_info <= (3, 0, 0):
        s = unicode(s)
        return s.lower()
    else:
        s = str(s)
        return s.casefold()


def _test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _test()
