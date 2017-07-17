# This file is part of MyPaint.
# Copyright (C) 2011-2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (C) 2007-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Exception classes"""

from __future__ import division, print_function


class LoadingFailed (Exception):
    """Raised when loading to indicate that a layer cannot be loaded"""
    pass
