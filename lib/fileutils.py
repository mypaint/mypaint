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
import os, sys, hashlib, zipfile, colorsys, urllib, gc
import os.path
import functools
import numpy
import logging
logger = logging.getLogger(__name__)

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
    if uri.startswith('file:\\\\\\'): # windows
        path = uri[8:] # 8 is len('file:///')
    elif uri.startswith('file://'): # nautilus, rox
        path = uri[7:] # 7 is len('file://')
    elif uri.startswith('file:'): # xffm
        path = uri[5:] # 5 is len('file:')
    path = urllib.url2pathname(path) # escape special chars
    path = path.strip('\r\n\x00') # remove \r\n and NULL
    path = path.decode('utf-8') # return unicode object (for Windows)
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


