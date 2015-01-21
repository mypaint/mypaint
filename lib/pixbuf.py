# This file is part of MyPaint.
# Copyright (C) 2014-2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""GdkPixbuf utils and compatibility layer

The GdkPixbuf.Pixbuf interface varies between platforms and typelibs.
The functions here provide a more consistent interface to write code
against.

The names are patterned after the gdk_pixbuf_{load,save}* functions
which are exposed on POSIX platforms.

"""

## Imports

from gi.repository import GdkPixbuf

import os
import logging
logger = logging.getLogger(__name__)


## Constants

LOAD_CHUNK_SIZE = 64*1024


## Utility functions


def save(pixbuf, filename, type='png', **kwargs):
    """Save pixbuf to a named file (compatibility wrapper)

    :param GdkPixbuf.Pixbuf pixbuf: the pixbuf to save
    :param unicode filename: file path to save as
    :param str type: type to save as: 'jpeg'/'png'/...
    :param \*\*kwargs: passed through to GdkPixbuf
    :rtype: bool
    :returns: whether the file was saved fully

    """
    if os.name == 'nt':
        # Backhanded. Ideally want pixbuf.savev_utf8(filename, [...])
        # but the typelib on MSYS2/MinGW wraps this incorrectly.
        fp = open(filename, 'wb')
        writer = lambda buf, size, data: fp.write(buf) or True
        result = pixbuf.save_to_callbackv(
            save_func=writer,
            user_data=fp,
            type=type,
            option_keys=kwargs.keys(),
            option_values=kwargs.values(),
        )
        fp.close()
        return result
    else:
        return pixbuf.savev(filename, type, kwargs.keys(), kwargs.values())


def load_from_file(filename, feedback_cb=None):
    """Load a pixbuf from a named file

    :param unicode filename: name of the file to open and read
    :param callable feedback_cb: invoked to provide feedback to the user
    :rtype: GdkPixbuf.Pixbuf
    :returns: the loaded pixbuf
    """
    fp = open(filename, 'rb')
    pixbuf = load_from_stream(fp, feedback_cb)
    fp.close()
    return pixbuf


def load_from_stream(fp, feedback_cb=None):
    """Load a pixbuf from an open file-like object

    :param fp: file-like object opened for reading
    :param callable feedback_cb: invoked to provide feedback to the user
    :rtype: GdkPixbuf.Pixbuf
    :returns: the loaded pixbuf
    """
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


def load_from_zipfile(datazip, filename, feedback_cb=None):
    """Extract and return a pixbuf from a zipfile entry

    :param zipfile.ZipFile datazip: ZipFile object opened for extracting
    :param unicode filename: pixbuf entry (file name) in the zipfile
    :param callable feedback_cb: invoked to provide feedback to the user
    :rtype: GdkPixbuf.Pixbuf
    :returns: the loaded pixbuf
    """
    try:
        datafp = datazip.open(filename, mode='r')
    except KeyError:
        # Support for bad zip files (saved by old versions of the
        # GIMP ORA plugin)
        datafp = datazip.open(filename.encode('utf-8'), mode='r')
        logger.warning('Bad ZIP file. There is an utf-8 encoded '
                       'filename that does not have the utf-8 '
                       'flag set: %r', filename)
    pixbuf = load_from_stream(datafp, feedback_cb=feedback_cb)
    datafp.close()
    return pixbuf
