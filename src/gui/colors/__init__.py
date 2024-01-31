# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Color manipulation submodule."""

from __future__ import division, print_function

from .adjbases import ColorManager
from .adjbases import ColorAdjuster
from .adjbases import PreviousCurrentColorAdjuster
from .hsvsquare import HSVSquare

__all__ = [
    "ColorManager",
    "ColorAdjuster",
    "PreviousCurrentColorAdjuster",
    "HSVSquare",
]
