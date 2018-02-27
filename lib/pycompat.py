# This file is part of MyPaint.
# Copyright (C) 2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Convenient consts for use while porting problem Python code.

Trying to avoid six here; the versions we're porting between have the
str/bytes b"" and unicode/str u"" literals, and that seems to be half
the battle for a lot of this stuff.

"""

import sys

PY3 = (sys.version_info >= (3,))
PY2 = (sys.version_info < (3,))

if PY3:
    xrange = range
    unicode = str
else:
    xrange = xrange
    unicode = unicode
