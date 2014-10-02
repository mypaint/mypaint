# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Utility functions for dealing with files, file URIs and filenames"""


## Imports

from math import floor, ceil, isnan
import os
import os.path
import sys
import hashlib
import zipfile
import colorsys
import urllib
import gc
import functools
import numpy
import logging
logger = logging.getLogger(__name__)
import shutil

from gi.repository import GdkPixbuf
from gi.repository import GLib

import mypaintlib



## Utiility funcs


def expanduser_unicode(s):
    # expanduser() doesn't handle non-ascii characters in environment variables
    # https://gna.org/bugs/index.php?17111
    s = s.encode(sys.getfilesystemencoding())
    s = os.path.expanduser(s)
    s = s.decode(sys.getfilesystemencoding())
    return s


def uri2filename(uri):
    # code from http://faq.pyGtk.org/index.py?req=show&file=faq23.031.htp
    # get the path to file
    path = ""
    if uri.startswith('file:\\\\\\'):  # windows
        path = uri[8:]  # 8 is len('file:///')
    elif uri.startswith('file://'):  # nautilus, rox
        path = uri[7:]  # 7 is len('file://')
    elif uri.startswith('file:'):  # xffm
        path = uri[5:]  # 5 is len('file:')
    path = urllib.url2pathname(path)  # escape special chars
    path = path.strip('\r\n\x00')  # remove \r\n and NULL
    path = path.decode('utf-8')  # return unicode object (for Windows)
    return path


def filename2uri(path):
    path = os.path.abspath(path)
    path = urllib.pathname2url(path.encode('utf-8'))
    # Workaround for Windows. For some reason (wtf?) urllib adds
    # trailing slashes on Windows. It converts "C:\blah" to "//C:\blah".
    # This would result in major problems when using the URI later.
    # (However, it seems we must add a single slash on Windows.)
    # One effect of this bug was that the last save directory was not remembered.
    while path.startswith('/'):
        path = path[1:]
    return 'file:///' + path


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
        target_path = filename
        dirname, target_basename = os.path.split(target_path)
        stemname, ext = os.path.splitext(target_basename)
        # Try to save up front, don't rotate backups if it fails
        temp_basename = ".tmpsave.%s%s" % (stemname, ext)
        temp_path = os.path.join(dirname, temp_basename)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        try:
            logger.debug("Writing to temp path %r", temp_path)
            save_result = save_method(self, temp_path, *args, **kwds)
        except Exception as ex:
            logger.exception("Save method failed")
            os.remove(temp_path)
            raise ex
        if not os.path.exists(temp_path):
            logger.warning("Save method did not create %r", temp_path)
            return save_result

        # Maintain a backup copy, because filesystems suck
        backup_basename = "%s%s.BAK" % (stemname, ext)
        backup_path = os.path.join(dirname, backup_basename)
        if os.path.exists(target_path):
            if os.path.exists(backup_path):
                logger.debug("Removing old backup %r", backup_path)
                os.remove(backup_path)
            with open(target_path, 'rb') as target_fp:
                backup_fp = open(backup_path, 'wb')
                logger.debug("Making new backup %r", backup_path)
                shutil.copyfileobj(target_fp, backup_fp)
                backup_fp.flush()
                os.fsync(backup_fp.fileno())
                backup_fp.close()
            assert os.path.exists(backup_path)

        # Renaming the tempfile over the target will fail under Windows,
        # but has advantages with Linux ext4: newer versions detect
        # this, and aggressively flush data to the disk.
        try:
            logger.debug("Renaming %r to %r", temp_path, target_path)
            os.rename(temp_path, target_path)
        except:
            logger.exception(
                "Rename %r into place failed (normal under Windows)",
                temp_path,
                )
            logger.info(
                "Retrying, after first removing %r",
                target_path,
                )
            os.remove(target_path)
            os.rename(temp_path, target_path)
        assert os.path.exists(target_path)
        return save_result

    return _wrapped_save_method


