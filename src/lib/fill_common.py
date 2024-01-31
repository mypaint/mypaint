# This file is part of MyPaint.
# Copyright (C) 2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Functions and constants common to fill and morphological operations"""

import lib.helpers
import lib.mypaintlib
import numpy


N = lib.mypaintlib.TILE_SIZE

# Constant alpha value for fully opaque pixels
_OPAQUE = 1 << 15

# Keeping track of fully opaque tiles allows for potential
# substantial performance benefits for both morphological
# operations as well as feathering and compositing
_FULL_TILE = lib.mypaintlib.ConstTiles.ALPHA_OPAQUE()
_FULL_TILE.flags.writeable = False

# Keeping track of empty tiles (which are less likely to
# be produced during the fill) permits skipping the compositing
# step for these tiles
_EMPTY_TILE = lib.mypaintlib.ConstTiles.ALPHA_TRANSPARENT()
_EMPTY_TILE.flags.writeable = False


def new_full_tile(value, dimensions=(N, N), value_type='uint16'):
    """Return a new tile filled with the given value"""
    tile = numpy.empty(dimensions, value_type)
    tile.fill(value)
    return tile


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


def orthogonal(tile_coord):
    """ Return the coordinates orthogonal to the input coordinate.

    Return coordinates orthogonal to the input coordinate,
    in the following order:

      0
    3   1
      2
    """
    return nine_grid(tile_coord)[1:5]


class TileBoundingBox(object):
    """ Bounding box helper for checking tiles by their coordinates

    Defines a bounding box in pixel coordinates that allows
    checking conditions and retrieving in-tile pixel bounds
    for individual tiles, based on their coordinates.
    """

    def __init__(self, bbox):
        """
        Create a new TileBoundingBox based on a pixel bounding box.
        :param bbox: x, y, w, h bounding box in model pixel units
        :type bbox: lib.helpers.Rect
        """
        bbx, bby, bbw, bbh = bbox
        bb_rx, bb_ry = bbx + bbw - 1, bby + bbh - 1

        self.min_tx = int(bbx // N)
        self.min_ty = int(bby // N)
        self.max_tx = int(bb_rx // N)
        self.max_ty = int(bb_ry // N)

        self.min_px = int(bbx % N)
        self.min_py = int(bby % N)
        self.max_px = int(bb_rx % N)
        self.max_py = int(bb_ry % N)
        self.no_tile_crossing = (
            (self.min_px, self.min_py, self.max_px, self.max_py) ==
            (0, 0, N - 1, N - 1)
        )

    def tile_bounds(self, tc):
        """ Return the in-tile pixel bounds as a 4-tuple.
        Bounds cover the entire tile, unless it crosses
        an edge of the bounding box. Does not check if
        the tile actually lies inside the bounding box.
        """
        if self.no_tile_crossing:
            return 0, 0, N - 1, N - 1
        tx, ty = tc
        min_x = self.min_px if tx == self.min_tx else 0
        min_y = self.min_py if ty == self.min_ty else 0
        max_x = self.max_px if tx == self.max_tx else N - 1
        max_y = self.max_py if ty == self.max_ty else N - 1
        return min_x, min_y, max_x, max_y

    def outside(self, tc):
        """ Check if tile is outside bounding box.
        Checks if the tile of the given coordinate
        lies completely outside of the bounding box.
        """
        tx, ty = tc
        return (
            tx < self.min_tx or tx > self.max_tx or
            ty < self.min_ty or ty > self.max_ty
        )

    def crossing(self, tc):
        """ Check if tile crosses the bounding box.
        Checks if the tile of the given coordinate
        crosses at least one edge of the bounding box.
        """
        if self.no_tile_crossing:
            return False
        tx, ty = tc
        return (
            (tx == self.min_tx and self.min_px != 0) or
            (ty == self.min_ty and self.min_py != 0) or
            (tx == self.max_tx and self.max_px != (N - 1)) or
            (ty == self.max_ty and self.max_py != (N - 1))
        )

    def inside(self, tc):
        """ Check if tile is inside the bounding box.
        Checks if the tile of the given coordinate
        is fully enclosed by the bounding box.
        """
        tx, ty = tc
        if self.crossing(tc):
            return False
        return (
            self.min_tx <= tx <= self.max_tx and
            self.min_ty <= ty <= self.max_ty
        )
