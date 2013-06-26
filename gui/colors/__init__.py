# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Colour manipulation submodule.
"""

from adjbases import ColorManager, ColorAdjuster, PreviousCurrentColorAdjuster
from picker import ColorPickerButton, get_color_at_pointer
from hsvtriangle import HSVTriangle
from uicolor import RGBColor, HSVColor, HCYColor

