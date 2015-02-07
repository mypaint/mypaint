# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Error classes which may be raised by gui-independent code"""


class FileHandlingError (Exception):
    """Simple problem loading or saving files; user-facing string

    Covers expected things like missing required ath elements, missing
    files, unsupported formats etc.

    The stringification of a FileHandlingError should always be
    presentable to the user directly, and marked for translation at the
    point where it's raised. It may contain diagnostic information, but
    should always be prefixed or suffixed with gloss for ordinary users.

    """

