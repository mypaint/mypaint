# This file is part of MyPaint.
# Copyright (C) 2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""This module implements tile-based floodfill and related operations."""

import time
import logging

import numpy as np

import lib.helpers
import lib.mypaintlib as myplib
import lib.surface

from lib.pycompat import PY3

logger = logging.getLogger(__name__)

TILE_SIZE = N = myplib.TILE_SIZE

# Constant alpha value for fully opaque pixels
_OPAQUE = 1 << 15

# Keeping track of fully opaque tiles allows for potential
# substantial performance benefits for both morphological
# operations as well as feathering.
_FULL_TILE = np.full((N, N), _OPAQUE, 'uint16')
_FULL_TILE.flags.writeable = False

# Keeping track of empty tiles (which are less likely to
# be produced during the fill) permits skipping the compositing
# step for these tiles
_EMPTY_TILE = np.zeros((N, N), 'uint16')
_EMPTY_TILE.flags.writeable = False


def is_full(t):
    return t is _FULL_TILE


def nine_grid(tc):
    """ Return the input coordinate along with its neighbours.

    Return tile coordinates of the full nine-grid,
    relative to the input coordinate, in the following order:

    8 1 5
    4 0 2
    7 3 6
    """
    tx, ty = tc
    offsets = [
        (0, 0), (0, -1), (1, 0), (0, 1), (-1, 0),
        (1, -1), (1, 1), (-1, 1), (-1, -1)
    ]
    return [(tx+o[0], ty+o[1]) for o in offsets]


def adjacent(tc):
    """ Return the coordinates adjacent to the input coordinate.

    Return coordinates of the neighbourhood
    of the input coordinate, in the following order:

    7 0 4
    3   1
    6 2 5
    """
    return nine_grid(tc)[1:]


def orthogonal(tc):
    """ Return the coordinates orthogonal to the input coordinate.

    Return coordinates orthogonal to the input coordinate,
    in the following order:

      0
    3   1
      2
    """
    return nine_grid(tc)[1:5]


# Tile boundary condition helpers

def out_of_bounds(point, bbox):
    """Test if a 2d coordinate is outside the given 2d bounding box
    """
    x, y = point
    min_x, min_y, max_x, max_y = bbox
    return x < min_x or x > max_x or y < min_y or y > max_y


def across_bounds(point, bbox):
    """Test if a 2d coordinate is on the edge of the given 2d bounding box
    """
    x, y = point
    min_x, min_y, max_x, max_y = bbox
    return x == min_x or x == max_x or y == min_y or y == max_y


def inside_bounds(point, bbox):
    """Test if a 2d coordinate is inside of the given 2d bounding box
    """
    x, y = point
    min_x, min_y, max_x, max_y = bbox
    return x > min_x and x < max_x and y > min_y and y < max_y


def enqueue_overflows(queue, tile_coord, seeds, bbox, *p):
    """ Conditionally add (coordinate, seed list, data...) tuples to a queue.

    :param queue: the queue which may be appended
    :type queue: list
    :param tile_coord: the 2d coordinate in the middle of the seed coordinates
    :type tile_coord: (int, int)
    :param seeds: 4-tuple of seed lists for n, e, s, w, relative to tile_coord
    :type seeds: (list, list, list, list)
    :param bbox: the bounding box of the fill operation
    :type bbox: (int, int, int, int)
    :param *p: tuples of length >= 4, items added to queue items w. same index

    NOTE: This function improves readability significantly in exchange for a
    small performance hit. Replace with explicit queueing if too slow.
    """
    for edge in zip(*(orthogonal(tile_coord), seeds) + p):
        edge_coord = edge[0]
        edge_seeds = edge[1]
        if edge_seeds and not out_of_bounds(edge_coord, bbox):
            queue.append(edge)
        pass
    pass


# Main fill handling function

def flood_fill(src, x, y, color, tolerance, bbox, dst, empty_rgba):
    """ Top-level flood fill interface, initiating and delegating actual fill

    :param src: Source surface-like object
    :type src: Anything supporting readonly tile_request()
    :param x: Starting point X coordinate
    :param y: Starting point Y coordinate
    :param color: an RGB color
    :type color: tuple
    :param bbox: Bounding box: limits the fill
    :type bbox: lib.helpers.Rect or equivalent 4-tuple
    :param tolerance: how much filled pixels are permitted to vary
    :type tolerance: float [0.0, 1.0]
    :param dst: Target surface
    :type dst: lib.tiledsurface.MyPaintSurface
    :param empty_rgba: reference to transparent_tile.rgba in lib.tiledsurface
    :type empty_rgba; numpy.ndarray

    The fill is performed with reference to src.
    The resulting tiles are composited into dst.
    """
    # Color to fill with
    fill_r, fill_g, fill_b = color

    # Limits
    tolerance = lib.helpers.clamp(tolerance, 0.0, 1.0)

    # Maximum area to fill: tile and in-tile pixel extents
    bbx, bby, bbw, bbh = bbox
    if bbh <= 0 or bbw <= 0:
        return
    bbbrx = bbx + bbw - 1
    bbbry = bby + bbh - 1

    min_tx = int(bbx // N)
    min_ty = int(bby // N)
    max_tx = int(bbbrx // N)
    max_ty = int(bbbry // N)

    min_px = int(bbx % N)
    min_py = int(bby % N)
    max_px = int(bbbrx % N)
    max_py = int(bbbry % N)

    tiles_bbox = (min_tx, min_ty, max_tx, max_ty)

    def tile_bounds(tile_coords):
        tx, ty = tile_coords
        min_x = min_px if tx == min_tx else 0
        min_y = min_py if ty == min_ty else 0
        max_x = max_px if tx == max_tx else N-1
        max_y = max_py if ty == max_ty else N-1
        return min_x, min_y, max_x, max_y

    # Tile and pixel addressing for the seed point
    init_tx, init_ty = int(x // N), int(y // N)
    px, py = int(x % N), int(y % N)

    # Sample the pixel color there to obtain the target color
    with src.tile_request(init_tx, init_ty, readonly=True) as start:
        targ_r, targ_g, targ_b, targ_a = [int(c) for c in start[py][px]]
    if targ_a == 0:
        targ_r, targ_g, targ_b = 0, 0, 0

    # Set of coordinates of fully opaque filled tiles, used to potentially
    # bypass dilation/erosion and blur operations for contiguous opaque areas
    full_opaque = set({})

    filler = myplib.Filler(targ_r, targ_g, targ_b, targ_a, tolerance)
    init = (init_tx, init_ty, px, py)
    args = (src, init, tiles_bbox, tile_bounds, filler, empty_rgba)

    # Profiling
    t0 = time.time()

    filled = scanline_fill(*(args + (full_opaque,)))

    t1 = time.time()
    logger.info("%.3f seconds to fill", t1 - t0)

    # When filling large areas, copying a full tile directly
    # when possible greatly improves performance.
    full_rgba = myplib.full_rgba_tile(fill_r, fill_g, fill_b)

    # Composite filled tiles into the destination surface
    tiles_to_composite = filled.items() if PY3 else filled.iteritems()
    for tc, src_tile in tiles_to_composite:
        # Omit tiles outside of the bounding box _if_ the frame is enabled
        if out_of_bounds(tc, tiles_bbox):
            continue
        with dst.tile_request(*tc, readonly=False) as dst_tile:
            # Skip empty tiles
            if src_tile is _EMPTY_TILE:
                continue
            # Copy full tiles directly if not on the bounding box edge
            if is_full(src_tile) and not across_bounds(tc, tiles_bbox):
                myplib.tile_copy_rgba16_into_rgba16(full_rgba, dst_tile)
                continue
            # Otherwise, composite the section with provided bounds into the
            # destination tile, most often the entire tile
            t_bounds = tile_bounds(tc)
            myplib.fill_composite(
                fill_r, fill_g, fill_b, src_tile, dst_tile, *t_bounds
            )
        dst._mark_mipmap_dirty(*tc)
    bbox = lib.surface.get_tiles_bbox(filled)
    dst.notify_observers(*bbox)
    logger.info("Total time for fill: %.3f seconds", time.time() - t0)


def scanline_fill(
        src, init, tiles_bbox, bounds,
        filler, empty_rgba, full_opaque):
    """ Perform a scanline fill and return the filled tiles

    Perform a scanline fill using the given starting point and tile,
    with reference to the src surface and given bounding box, using the
    provided filler instance.

    Uniform tiles which should be filled fully will have their coordinates
    added to the full_opaque set.

    :param src: Source surface-like object
    :param init: coordinates for starting tile and pixel
    :type init: (int, int, int, int)
    :param tiles_bbox: min/max bounds for tiles (min_x, min_y, max_x, max_y)
    :type tiles_bbox: (int, int, int, int)
    :param bounds: func returning tile-relative pixel bounds for a tile
    :type bounds: ((int, int)) -> (int, int, int, int)
    :param filler: filler instance performing the per-tile fill operation
    :type filler: mypaintlib.Filler
    :param empty_rgba: reference to transparent_tile.rgba in lib.tiledsurface
    :type empty_rgba; numpy.ndarray
    :param full_opaque: set of coords to be amended by coords of full tiles
    :type full_opaque: set

    :returns: a dictionary of coord->tile mappings for the filled tiles
    """

    # Set of coordinates for tiles that only need to be
    # handled once in the fill loop (uniformly colored tiles)
    final = set({})

    # Dict of coord->tile data populated during the fill
    filled = {}

    full_overflows = [
        ((), [(0, N-1)], [(0, N-1)], [(0, N-1)]),         # from north
        ([(0, N-1)], (), [(0, N-1)], [(0, N-1)]),         # from east
        ([(0, N-1)], [(0, N-1)], (), [(0, N-1)]),         # from south
        ([(0, N-1)], [(0, N-1)], [(0, N-1)], ()),         # from west
        ([(0, N-1)], [(0, N-1)], [(0, N-1)], [(0, N-1)])  # from within
    ]

    inv_edges = (
        myplib.edges.south,
        myplib.edges.west,
        myplib.edges.north,
        myplib.edges.east
    )

    init_tx, init_ty = init[0:2]
    init_px, init_py = init[2:4]
    tileq = [((init_tx, init_ty), (init_px, init_py), myplib.edges.none)]

    # Dict of alpha->tile, used for uniform non-opaque tile fills
    # NOTE: these are usually not a result of an intentional fill, but
    # clicking a pixel with color very similar to the intended target pixel
    uniform_tiles = {}

    def uniform_tile(alpha):
        if alpha not in uniform_tiles:
            uniform_tiles[alpha] = np.full((N, N), alpha, 'uint16')
        return uniform_tiles[alpha]

    while len(tileq) > 0:
        tc, seeds, direction = tileq.pop(0)
        # Skip if the tile has been fully processed already
        if tc in final:
            continue
        # Flood-fill one tile
        (tx, ty) = tc
        with src.tile_request(tx, ty, readonly=True) as src_tile:
            overflows = None
            if tc not in filled:
                # The first time a tile is encountered, run a uniformity
                # test, which is used to quickly process e.g. tiles that
                # are empty but not instances of transparent_tile.rgba
                alpha = None
                if inside_bounds(tc, tiles_bbox):
                    is_empty = src_tile is empty_rgba
                    # Returns the alpha of the fill for the tile's color if
                    # the tile is uniform, otherwise returns None
                    alpha = filler.tile_uniformity(is_empty, src_tile)
                if alpha is None:
                    # No shortcut can be taken, create new tile
                    filled[tc] = np.zeros((N, N), 'uint16')
                else:
                    # Tile is uniform, so there is no need to process
                    # it again in the fill loop, either set as
                    # a uniformly filled alpha tile or skip it if it
                    # cannot be filled at all (an unlikely scenario)
                    final.add(tc)
                    if alpha == _OPAQUE:
                        filled[tc] = _FULL_TILE
                        full_opaque.add(tc)
                    elif alpha != 0:
                        filled[tc] = uniform_tile(alpha)
                    else:
                        filled[tc] = _EMPTY_TILE
                        # No seeds to process, skip seed handling
                        continue
                    # For a filled uniform tile
                    overflows = full_overflows[direction]
            if overflows is None:
                overflows = filler.fill(
                    src_tile, filled[tc], seeds,
                    direction, *bounds(tc)
                )
        enqueue_overflows(tileq, tc, overflows, tiles_bbox, inv_edges)
    return filled
