# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Pythonic wrappers for some standard GLib routines.

MyPaint is strict about using Unicode internally for everything, but
when GLib returns a filename, all hell can break loose (the data isn't
unicode, and may not even be UTF-8). This module works around that.

"""


## Imports

from __future__ import division, print_function
import logging
import sys

from lib.pycompat import PY3
from lib.pycompat import unicode

from lib.gibindings import GLib

logger = logging.getLogger(__name__)


## File path getter functions

def filename_to_unicode(opsysstring):
    """Converts a str representing a filename from GLib to unicode.

    :param bytes opsysstring: a string in the (GLib) encoding for filenames
    :returns: the converted filename
    :rtype: unicode|str

    >>> filename_to_unicode(b'/ascii/only/path') == u'/ascii/only/path'
    True
    >>> filename_to_unicode(None) is None
    True

    This is just a more Pythonic wrapper around g_filename_to_utf8() for
    now, which works around a ton of weird bugs and corner cases with
    the typelib annotations for it. It is intended for cleaning up the
    output of certain GLib functions.

    Currently, if you're using Python 3 and the input is already unicode
    then this function assumes that GLib+GI have already done the work,
    and that the unicode string was correct. You get the same string
    back.

    For Python 2, this accepts only "bytes" string input. If we find a
    corner case where GLib functions return degenerate unicode, we can
    adapt it for that case (those funcs need their own wrappers though).

    """
    if opsysstring is None:
        return None

    # Let's assume that if the string is already unicode under Python 3,
    # then it's already correct.
    if PY3 and isinstance(opsysstring, unicode):
        return opsysstring

    # On Windows, they're always UTF-8 regardless.
    # That's what the docs say.
    if sys.platform == "win32":
        return opsysstring.decode("utf-8")

    # Other systems are dependent in opaque ways on the environment.
    if not isinstance(opsysstring, bytes):
        raise TypeError("Argument must be bytes")
    opsysstring_degenerate_unicode = opsysstring.decode("latin_1")

    # This function's annotation seems to vary quite a bit.
    # See https://github.com/mypaint/mypaint/issues/634
    ustring = None

    # The sensible, modern case! Byte strings in, unicode strings
    # out hopefully, and the C func's arguments are correctly
    # [out]-annotated. It works like this as of...
    #
    # - Python 2.7.14 OR Python 3.6.4
    # - gobject-introspection 1.54.1
    # - glib2 2.54.3
    # - Debian buster/sid amd64 OR MSYS2 MINGW64 on Windows 7 64-bit.

    if ustring is None:
        for s in [opsysstring, opsysstring_degenerate_unicode]:
            try:
                ustring, _bytes_read, _bytes_written \
                    = GLib.filename_to_utf8(s, -1)
                break
            except TypeError:
                pass

    # Try an older, bad typelib's form.
    # This is the case for Ubuntu 14.04 LTS "trusty" (which is ancient,
    # but that's what our current Travis CI solution uses). For the
    # record, this weirdness is applicable to the following combination:
    #
    # - Python 2.7.6 OR Python 3.4.3
    # - gobject-introspection 1.40.0
    # - glib2 2.40.2
    # - Ubuntu 14.04.5 LTS amd64.
    #
    # Of note: the Py3 wrappings are weird in Trusty. Other GLib funcs
    # return bytes, but GLib.filename_to_utf8() expects those degenerate
    # unicode strings. Byte strings will not do. Unusual tastes.

    if ustring is None:
        for s in [opsysstring, opsysstring_degenerate_unicode]:
            try:
                ustring = GLib.filename_to_utf8(s, -1, 0, 0)
                break
            except TypeError:
                pass

    # Congratulations! You found a new bug.

    if ustring is None:
        raise UnicodeDecodeError(
            "New or unknown bugs in g_filename_to_utf8()'s typelib. "
            "Failed to convert %r. Please tell the developers about this."
            % (opsysstring,)
        )

    # Python2's wrappers tended to do this.
    # I suspect it's reasonable to convert for all, now that we're
    # reasonably sure that the data would be utf-8.

    if isinstance(ustring, bytes):
        ustring = ustring.decode("utf-8")

    return ustring


def get_user_config_dir():
    """Like g_get_user_config_dir(), but always unicode"""
    d_fs = GLib.get_user_config_dir()
    return filename_to_unicode(d_fs)


def get_user_data_dir():
    """Like g_get_user_data_dir(), but always unicode"""
    d_fs = GLib.get_user_data_dir()
    return filename_to_unicode(d_fs)


def get_user_cache_dir():
    """Like g_get_user_cache_dir(), but always unicode"""
    d_fs = GLib.get_user_cache_dir()
    return filename_to_unicode(d_fs)


def get_user_special_dir(d_id):
    """Like g_get_user_special_dir(), but always unicode"""
    d_fs = GLib.get_user_special_dir(d_id)
    return filename_to_unicode(d_fs)


## First-import cache forcing

def init_user_dir_caches():
    """Caches the GLib user directories

    >>> init_user_dir_caches()

    The first time this module is imported is from a particular point in
    the launch script, after all the i18n setup is done and before
    lib.mypaintlib is imported. If they're not cached up-front in this
    manner, get_user_config_dir() & friends may return literal "?"s in
    place of non-ASCII characters (Windows systems with non-ASCII user
    profile dirs are known to trigger this).

    The debugging prints may be useful too.

    """
    logger.debug("Init g_get_user_config_dir(): %r", get_user_config_dir())
    logger.debug("Init g_get_user_data_dir(): %r", get_user_data_dir())
    logger.debug("Init g_get_user_cache_dir(): %r", get_user_cache_dir())
    # It doesn't matter if some of these are None
    for i in range(GLib.UserDirectory.N_DIRECTORIES):
        k = GLib.UserDirectory(i)
        logger.debug(
            "Init g_get_user_special_dir(%s): %r",
            k.value_name,
            get_user_special_dir(k),
        )


## Filename <-> URI conversion


def filename_to_uri(abspath, hostname=None):
    """More Pythonic & stable g_filename_to_uri(), with OS workarounds.

    >>> import os.path
    >>> relpath = os.path.join(u'tmp', u'smile (\u263a).ora')
    >>> abspath = os.path.abspath(relpath)
    >>> uri = filename_to_uri(abspath)
    >>> isinstance(uri, str)
    True
    >>> uri.endswith('/tmp/smile%20(%E2%98%BA).ora')
    True
    >>> uri.startswith('file:///')
    True

    """
    if hostname:
        raise ValueError("Only NULL hostnames are supported")
    # GLib.filename_to_uri is *present* on Windows, for both i686 and
    # x64_64 (MSYS2 builds), however attempting to *call* it on the
    # 64-bit build results in
    # "Error: g-invoke-error-quark: Could not locate g_filename_to_uri"
    # as reported in https://github.com/mypaint/mypaint/issues/374
    # Use the _utf8 variant instead on platforms where it exists.
    try:
        g_filename_to_uri = GLib.filename_to_uri_utf8
    except AttributeError:
        g_filename_to_uri = GLib.filename_to_uri
    hostname = ""
    return g_filename_to_uri(abspath, hostname)


def filename_from_uri(uri):
    """More Pythonic & stable g_filename_from_uri(), with OS workarounds.

    >>> import os.path
    >>> relpath = os.path.join(u'tmp', u'smile (\u263a).ora')
    >>> abspath1 = os.path.abspath(relpath)
    >>> uri = filename_to_uri(abspath1)
    >>> abspath2, hostname = filename_from_uri(uri)
    >>> if PY3:
    ...     unicode = str
    >>> isinstance(abspath2, unicode)
    True
    >>> abspath2.replace('\\\\', "/") == abspath1.replace('\\\\', "/")
    True

    """
    # First find the right g_filename_from_uri.
    # See the note above.
    try:
        g_filename_from_uri = GLib.filename_from_uri_utf8
    except AttributeError:
        g_filename_from_uri = GLib.filename_from_uri
    # But oh joy. We have to support more than one typelib.
    try:
        # Newer GLib typelibs on Linux mark it as a return.
        abspath, hostname = g_filename_from_uri(uri)
    except TypeError:
        # Older GLib typelibs,
        # including the one shipped with Ubuntu Server 12.04 (Travis!)
        # And that windows _utf8 mess still uses it too.
        abspath = g_filename_from_uri(uri, "")
        hostname = None
    assert (not hostname), ("Only URIs without hostnames are supported.")
    return (filename_to_unicode(abspath), None)


## Module testing

def _test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _test()
