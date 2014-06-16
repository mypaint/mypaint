# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Pixbuf loading utils"""

## Imports

from gi.repository import GdkPixbuf

import logging
logger = logging.getLogger(__name__)


## Constants

LOAD_CHUNK_SIZE = 64*1024



## Utility functions


def pixbuf_from_stream(fp, feedback_cb=None):
    """Extract and return a GdkPixbuf from file-like object"""
    loader = GdkPixbuf.PixbufLoader()
    while True:
        if feedback_cb is not None:
            feedback_cb()
        buf = fp.read(LOAD_CHUNK_SIZE)
        if buf == '':
            break
        loader.write(buf)
    loader.close()
    return loader.get_pixbuf()


def pixbuf_from_zipfile(datazip, filename, feedback_cb=None):
    """Extract and return a GdkPixbuf from a zipfile entry"""
    try:
        datafp = datazip.open(filename, mode='r')
    except KeyError:
        # Support for bad zip files (saved by old versions of the
        # GIMP ORA plugin)
        datafp = datazip.open(filename.encode('utf-8'), mode='r')
        logger.warning('Bad ZIP file. There is an utf-8 encoded '
                       'filename that does not have the utf-8 '
                       'flag set: %r', filename)
    pixbuf = pixbuf_from_stream(datafp, feedback_cb=feedback_cb)
    datafp.close()
    return pixbuf

