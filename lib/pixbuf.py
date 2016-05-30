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
from __future__ import print_function

from gi.repository import GdkPixbuf

import logging
logger = logging.getLogger(__name__)


## Constants

LOAD_CHUNK_SIZE = 64 * 1024


## Utility functions


def save(pixbuf, filename, type='png', **kwargs):
    """Save pixbuf to a named file (compatibility wrapper)

    :param GdkPixbuf.Pixbuf pixbuf: the pixbuf to save
    :param unicode filename: file path to save as
    :param str type: type to save as: 'jpeg'/'png'/...
    :param \*\*kwargs: passed through to GdkPixbuf
    :rtype: bool
    :returns: whether the file was saved fully

    >>> import tempfile, shutil, os
    >>> p = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB,True,8,64,64)
    >>> d = tempfile.mkdtemp()
    >>> save(p, os.path.join(d, "test.png"), type="png",
    ...      **{"tEXt::greeting": "Hello, world"})
    True
    >>> shutil.rmtree(d, ignore_errors=True)

    """
    with open(filename, 'wb') as fp:
        try:
            save_to_callbackv = pixbuf.save_to_callbackv
        except AttributeError:
            # save_to_callbackv disappeared in GdkPixbuf 2.31.2
            # and returned as of GdkPixbuf 2.31.5
            # https://bugzilla.gnome.org/show_bug.cgi?id=670372#c12
            save_to_callbackv = pixbuf.save_to_callback
        # Keyword args are not compatible with 2.26 (Ubuntu 12.04,
        # a.k.a. precise, a.k.a. "what Travis-CI runs")
        result = save_to_callbackv(
            lambda buf, size, data: fp.write(buf) or True,  # save_func
            fp,      # user_data
            type,      # type
            kwargs.keys(),   # option_keys
            kwargs.values(),  # option_values
        )
        return result


def load_from_file(filename, feedback_cb=None):
    """Load a pixbuf from a named file

    :param unicode filename: name of the file to open and read
    :param callable feedback_cb: invoked to provide feedback to the user
    :rtype: GdkPixbuf.Pixbuf
    :returns: the loaded pixbuf

    >>> p = load_from_file("pixmaps/mypaint_logo.png")
    >>> isinstance(p, GdkPixbuf.Pixbuf)
    True

    """
    with open(filename, 'rb') as fp:
        return load_from_stream(fp, feedback_cb)


def load_from_stream(fp, feedback_cb=None):
    """Load a pixbuf from an open file-like object

    :param fp: file-like object opened for reading
    :param callable feedback_cb: invoked to provide feedback to the user
    :rtype: GdkPixbuf.Pixbuf
    :returns: the loaded pixbuf

    >>> fp = open("pixmaps/mypaint_logo.png", "rb")
    >>> p = load_from_stream(fp)
    >>> isinstance(p, GdkPixbuf.Pixbuf)
    True

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

    >>> import zipfile
    >>> z = zipfile.ZipFile("tests/smallimage.ora", mode="r")
    >>> p = load_from_zipfile(z, "Thumbnails/thumbnail.png")
    >>> isinstance(p, GdkPixbuf.Pixbuf)
    True

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


## Module testing

def _test():
    """Run doctest strings"""
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _test()
