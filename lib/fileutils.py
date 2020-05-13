# This file is part of MyPaint.
# Copyright (C) 2007-2018 by the MyPaint Development Team
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Utility functions for dealing with files, file URIs and filenames"""


## Imports

from __future__ import division, print_function

import os
import os.path
import sys
import functools
import logging
import shutil
import unicodedata

import lib.helpers
from lib.pycompat import unicode

from lib.gibindings import GLib
from lib.gibindings import Gio

logger = logging.getLogger(__name__)


## Module configuration


VIA_TEMPFILE_MAKES_BACKUP_COPY = True
VIA_TEMPFILE_BACKUP_COPY_SUFFIX = '~'


## Utility funcs


def expanduser_unicode(s):
    """Expands a ~/ on the front of a unicode path, where meaningful.

    :param s: path to expand, coercable to unicode
    :returns: The expanded path
    :rtype: unicode

    This doesn't do anything on the Windows platform other than coerce
    its argument to unicode. On other platforms, it converts a "~"
    component on the front of a relative path to the user's absolute
    home, like os.expanduser().

    Certain workarounds for OS and filesystem encoding issues are
    implemented here too.

    """
    s = unicode(s)
    # The sys.getfilesystemencoding() on Win32 (mbcs) is for encode
    # only, and isn't roundtrippable. Luckily ~ is not meaningful on
    # Windows, and MyPaint uses a better default for the scrap prefix on
    # the Windows platform anyway.
    if sys.platform == "win32":
        return s
    # expanduser() doesn't handle non-ascii characters in environment variables
    # https://gna.org/bugs/index.php?17111
    s = s.encode(sys.getfilesystemencoding())
    s = os.path.expanduser(s)
    s = s.decode(sys.getfilesystemencoding())
    return s


def via_tempfile(save_method):
    """Filename save method decorator: write via a tempfile

    :param callable save_method: A valid save method to be wrapped
    :returns: a new decorated method

    This decorator wraps save methods which operate only on filenames
    to write to tempfiles in the same location. Rename is then used to
    atomically overwrite the original file, where possible.

    Any method with a filename as its first non-self parameter which
    creates a file of that name can be wrapped by this decorator. Other
    args passed to the decorated method are passed on to the save method
    itself.
    """
    @functools.wraps(save_method)
    def _wrapped_save_method(self, filename, *args, **kwds):
        # Where the user told us to save into.
        # Any backup files are written to this folder.
        user_specified_dirname = os.path.dirname(filename)
        # However if a file being overwritten is a symlink, the file
        # being pointed at is the one which should be atomically
        # overwritten.
        target_path = os.path.realpath(filename)
        target_dirname, target_basename = os.path.split(target_path)
        stemname, ext = os.path.splitext(target_basename)
        # Try to save up front, don't rotate backups if it fails
        temp_basename = ".tmpsave.%s%s" % (stemname, ext)
        temp_path = os.path.join(target_dirname, temp_basename)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        try:
            logger.debug("Writing to temp path %r", temp_path)
            save_result = save_method(self, temp_path, *args, **kwds)
        except Exception as ex:
            logger.exception("Save method failed")
            try:
                os.remove(temp_path)
            except Exception:
                logger.error("cleanup: failed to remove temp path too")
            raise ex
        if not os.path.exists(temp_path):
            logger.warning("Save method did not create %r", temp_path)
            return save_result
        # Maintain a backup copy, because filesystems suck
        if VIA_TEMPFILE_MAKES_BACKUP_COPY:
            suffix = VIA_TEMPFILE_BACKUP_COPY_SUFFIX
            backup_basename = "%s%s%s" % (stemname, ext, suffix)
            backup_path = os.path.join(user_specified_dirname, backup_basename)
            if os.path.exists(target_path):
                if os.path.exists(backup_path):
                    logger.debug("Removing old backup %r", backup_path)
                    os.remove(backup_path)
                with open(target_path, 'rb') as target_fp:
                    with open(backup_path, 'wb') as backup_fp:
                        logger.debug("Making new backup %r", backup_path)
                        shutil.copyfileobj(target_fp, backup_fp)
                        backup_fp.flush()
                        os.fsync(backup_fp.fileno())
                assert os.path.exists(backup_path)
        # Finally, replace the original
        logger.debug("Replacing %r with %r", target_path, temp_path)
        replace(temp_path, target_path)
        assert os.path.exists(target_path)
        return save_result

    return _wrapped_save_method


try:
    _replace = os.replace   # python 3
except AttributeError:
    if sys.platform == 'win32':
        try:
            import win32api
            import win32con

            def _replace(s, d):
                win32api.MoveFileEx(
                    s, d, win32con.MOVEFILE_REPLACE_EXISTING,
                )
        except ImportError:
            import ctypes
            _MoveFileEx = ctypes.windll.kernel32.MoveFileExW
            _MoveFileEx.argtypes = (
                ctypes.c_wchar_p,
                ctypes.c_wchar_p,
                ctypes.c_uint32,
            )
            _MoveFileEx.restype = ctypes.c_bool

            def _replace(s, d):
                if not _MoveFileEx(s, d, 1):  # MOVEFILE_REPLACE_EXISTING
                    raise OSError("_MoveFileEx(%r, %r)" % (s, d))
    else:
        _replace = os.rename


def replace(src, dst):
    """os.replace compat wrapper

    This has the semantics of a simple os.replace in Python3.

    >>> import tempfile, shutil
    >>> t = tempfile.mkdtemp()
    >>> f1 = os.path.join(t, "f1");
    >>> f2 = os.path.join(t, "f2");
    >>> for i, f in enumerate([f1, f2]):
    ...     with open(f, "w") as fp:
    ...         pass
    >>> assert os.path.isfile(f1)
    >>> assert os.path.isfile(f2)
    >>> replace(f1, f2)
    >>> assert not os.path.isfile(f1)
    >>> assert os.path.isfile(f2)
    >>> shutil.rmtree(t)

    Idea adapted from <http://stupidpythonideas.blogspot.co.uk/2014/07/
    getting-atomic-writes-right.html>.

    """
    _replace(src, dst)


def startfile(filepath, operation="open"):
    """os.startfile / g_app_info_launch_default_for_uri compat

    This has the similar semantics to os.startfile, where it's
    supported: it launches the given file or folder path with the
    default app. On Windows, operation can be set to "edit" to use the
    default editor for a file. The operation parameter is ignored on
    other systems, and GIO's equivalent routine is used.

    The relevant app is started in the background, and there are no
    means for getting its pid.

    """
    try:
        if os.name == 'nt':
            os.startfile(filepath, operation)  # raises: WindowsError
        else:
            uri = GLib.filename_to_uri(filepath)
            Gio.app_info_launch_default_for_uri(uri, None)  # raises: GError
        return True
    except Exception:
        logger.exception(
            "Failed to launch the default application for %r (op=%r)",
            filepath,
            operation,
        )
        return False


def _test():
    """Run doctests"""
    import doctest
    doctest.testmod()


def safename(s, fragment=False):
    """Returns a safe filename based on its unicode-string argument.

    Returns a safe filename or filename fragment based on an arbitrary
    string. The name generated in this way should be good for all OSes.

    Slashes, colons and other special characters will be stripped. The
    string will have its leading and trailing whitespace trimmed.

    :param unicode s: The string to convert.
    :param bool fragment: Name will never be used as a complete file name.

    Normally, extra checks are applied that assume the returned name
    will be used for a complete file basename, including extension.  If
    the "fragment" parameter is True, these additional safety
    conversions will be ignored, and its is the caller's responsibility
    to make the name safe. Appending other safe fragments or an
    extension will make the combined name safe again.

    >>> safename("test 1/4") == u'test 1_4'
    True
    >>> (safename("test 2/4 with a \022 and a trailing space ")
    ...  == u'test 2_4 with a _ and a trailing space')
    True
    >>> safename("lpt3") == u'_lpt3'
    True
    >>> safename("lpt3", fragment=True) == u'lpt3'
    True

    Note that fragments can be blank.  Whole names cannot.
    A completely blank name is treated like the reserved words.

    >>> safename("   ", fragment=True) == u''
    True
    >>> safename("   ", fragment=False) == u'_'
    True

    """
    # A little cleanup first
    s = unicode(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()

    # Strip characters that break interop.
    forbidden = [ord(c) for c in u"\\|/:*?\"<>"]
    forbidden += list(range(31))   # control chars
    table = {n: ord("_") for n in forbidden}
    s = s.translate(table)

    if not fragment:
        # Certain whole-filenames are reserved on Windows.
        reserved = lib.helpers.casefold(u"""
            NUL CON PRN AUX
            COM1 COM2 COM3 COM4 COM5 COM6 COM7 COM8 COM9
            LPT1 LPT2 LPT3 LPT4 LPT5 LPT6 LPT7 LPT8 LPT9
        """).strip().split()

        # A blank name is invalid for all systems.
        reserved += [u""]
        if lib.helpers.casefold(s) in reserved:
            s = "_" + s

        # Windows file names cannot end with a dot or a space.
        # Spaces are already handled.
        if s.endswith("."):
            s = s + "_"

    return s


if __name__ == '__main__':
    _test()
