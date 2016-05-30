# This file is part of MyPaint.
# Copyright (C) 2015-2016 by the MyPaint Development Team.
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

import sys
import logging

from gi.repository import GLib

logger = logging.getLogger(__name__)


## File path getter functions


def filename_to_unicode(opsysstring):
    """Converts a str representing a filename from GLib to unicode.

    :param str opsysstring: a string in the (GLib) encoding for filenames
    :returns: the converted filename
    :rtype: unicode

    >>> filename_to_unicode('/ascii/only/path')
    u'/ascii/only/path'
    >>> filename_to_unicode(None) is None
    True

    This is just a more Pythonic wrapper around g_filename_to_utf8() for
    now. If there are compatibility reasons to change it, fallbacks
    involving sys.getfilesystemencoding exist.

    """
    if opsysstring is None:
        return None
    # On Windows, they're always UTF-8 regardless.
    if sys.platform == "win32":
        return opsysstring.decode("utf-8")
    # Other systems are dependent in opaque ways on the environment.
    if not isinstance(opsysstring, str):
        raise TypeError("Argument must be bytes")
    # This function's annotation seems to vary quite a bit.
    # See https://github.com/mypaint/mypaint/issues/634
    try:
        ustring, _, _ = GLib.filename_to_utf8(opsysstring, -1)
    except TypeError:
        ustring = GLib.filename_to_utf8(opsysstring, -1, 0, 0)
    if ustring is None:
        raise UnicodeDecodeError(
            "GLib failed to convert %r to a UTF-8 string. "
            "Consider setting G_FILENAME_ENCODING if your file system's "
            "filename encoding scheme is not UTF-8."
            % (opsysstring,)
        )
    return ustring.decode("utf-8")


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
