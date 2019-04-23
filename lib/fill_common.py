# This file is part of MyPaint.
# Copyright (C) 2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Functions and constants common to fill and morphological operations"""

import lib.mypaintlib

# Constant alpha value for fully opaque pixels
_OPAQUE = 1 << 15

# Keeping track of fully opaque tiles allows for potential
# substantial performance benefits for both morphological
# operations as well as feathering and compositing
_FULL_TILE = lib.mypaintlib.TileConstants.OPAQUE_ALPHA_TILE()
_FULL_TILE.flags.writeable = False

# Keeping track of empty tiles (which are less likely to
# be produced during the fill) permits skipping the compositing
# step for these tiles
_EMPTY_TILE = lib.mypaintlib.TileConstants.TRANSPARENT_ALPHA_TILE()
_EMPTY_TILE.flags.writeable = False


def nine_grid(tile_coord):
    """ Return the input coordinate along with its neighbours.

    Return tile coordinates of the full nine-grid,
    relative to the input coordinate, in the following order:

    8 1 5
    4 0 2
    7 3 6
    """
    tile_x, tile_y = tile_coord
    offsets = [
        (0, 0), (0, -1), (1, 0), (0, 1), (-1, 0),
        (1, -1), (1, 1), (-1, 1), (-1, -1)
    ]
    return [(tile_x+o[0], tile_y+o[1]) for o in offsets]


def adjacent(tile_coord):
    """ Return the coordinates adjacent to the input coordinate.

    Return coordinates of the neighbourhood
    of the input coordinate, in the following order:

    7 0 4
    3   1
    6 2 5
    """
    return nine_grid(tile_coord)[1:]
