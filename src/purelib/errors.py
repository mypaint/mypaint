# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Error classes which may be raised by gui-independent code"""

from __future__ import division, print_function


class FileHandlingError (Exception):
    """Simple problem loading or saving files; user-facing string

    Covers expected things like missing required path elements, missing
    files, unsupported formats etc.

    The stringification of a FileHandlingError should always be
    presentable to the user directly, and marked for translation at the
    point where it's raised. It may contain diagnostic information, but
    should always be prefixed or suffixed with gloss for ordinary users.

    In general, if one of these is raised as a response to another
    exception, log that error with (yourmodule.logger.exception()) with
    programmer-focussed diagnostic info, and favour user-presentable
    info for the message string.

    """

    def __init__(self, msg, investigate_dir=None):
        super(FileHandlingError, self).__init__(msg)
        self.investigate_dir = investigate_dir


class AllocationError (Exception):
    """Indicates a failure to construct a required internal object.

    This may be used as a stopgap to cover probable out-of-memory
    conditions which GI hasn't wrapped as nice Pythonic `MemoryError`s.
    The GdkPixbuf wrappers are quite prone to this.

    The stringification of an AllocationError should always be
    presentable to the user directly, and marked for translation at the
    point where it's raised. It may contain diagnostic information, but
    should always be prefixed or suffixed with gloss for ordinary users.

    In general, if one of these is raised as a response to another
    exception, log that error with (yourmodule.logger.exception()) with
    programmer-focussed diagnostic info, and favour user-presentable
    info for the message string.

    """



