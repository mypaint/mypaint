# This file is part of MyPaint.
# Copyright (C) 2018-2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""This module implements tile-based floodfill and related operations."""

import logging
import sys

import numpy as np

import lib.helpers
import lib.mypaintlib as myplib
import lib.surface
import lib.fill_common as fc
from lib.fill_common import _OPAQUE, _FULL_TILE, _EMPTY_TILE
import lib.morphology

from lib.pycompat import PY3

logger = logging.getLogger(__name__)

TILE_SIZE = N = myplib.TILE_SIZE

# This should point to the array transparent_tile.rgba
# defined in tiledsurface.py
_EMPTY_RGBA = None

# Distance data for tiles with no detected distances
_GAPLESS_TILE = np.full((N, N), 2*N*N, 'uint16')
_GAPLESS_TILE.flags.writeable = False


def is_full(tile):
    """Check if the given tile is the fully opaque alpha tile"""
    return tile is _FULL_TILE


class GapClosingOptions():
    """Container of parameters for gap closing fill operations
    to avoid updates to the callchain in case the parameter set
    is altered.
    """
    def __init__(self, max_gap_size, retract_seeps):
        self.max_gap_size = max_gap_size
        self.retract_seeps = retract_seeps


def orthogonal(tile_coord):
    """ Return the coordinates orthogonal to the input coordinate.

    Return coordinates orthogonal to the input coordinate,
    in the following order:

      0
    3   1
      2
    """
    return fc.nine_grid(tile_coord)[1:5]


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


def starting_coordinates(x, y):
    """Get the coordinates of starting tile and pixel (tx, ty, px, py)"""
    init_tx, init_ty = int(x // N), int(y // N)
    init_x, init_y = int(x % N), int(y % N)
    return init_tx, init_ty, init_x, init_y


def tiles_bbox_and_bounds(bbox):
    """
    Get tile bounds from pixel bounds
    and tile->pixel bound helper function

    :param bbox: Bounding box with pixel units
    :return: tile coordinate bounds and
    (tile coordinate) -> (pixel bound) function
    """
    bbx, bby, bbw, bbh = bbox
    bb_rx, bb_ry = bbx + bbw - 1, bby + bbh - 1

    min_tx, min_ty = int(bbx // N), int(bby // N)
    max_tx, max_ty = int(bb_rx // N), int(bb_ry // N)

    min_px, min_py = int(bbx % N), int(bby % N)
    max_px, max_py = int(bb_rx % N), int(bb_ry % N)

    def tile_bounds(tc):
        """ Return the in-tile pixel bounds as a 4-tuple
        Bounds cover the entire tile, unless it is located
        on the edge of the bounding box.
        """
        tile_x, tile_y = tc
        min_x = min_px if tile_x == min_tx else 0
        min_y = min_py if tile_y == min_ty else 0
        max_x = max_px if tile_x == max_tx else N - 1
        max_y = max_py if tile_y == max_ty else N - 1
        return min_x, min_y, max_x, max_y

    return (min_tx, min_ty, max_tx, max_ty), tile_bounds


def get_target_color(src, tx, ty, px, py):
    """Get the pixel color for the given tile/pixel coordinates"""
    with src.tile_request(tx, ty, readonly=True) as start:
        targ_r, targ_g, targ_b, targ_a = [
            int(c) for c in start[py][px]
        ]
    if targ_a == 0:
        targ_r, targ_g, targ_b = 0, 0, 0

    return targ_r, targ_g, targ_b, targ_a


# Main fill handling function

def flood_fill(
        src, x, y, color, tolerance, offset, feather,
        gap_closing_options, mode, framed, bbox, dst):
    """ Top-level flood fill interface, initiating and delegating actual fill

    :param src: Source surface-like object
    :type src: Anything supporting readonly tile_request()
    :param x: Starting point X coordinate
    :param y: Starting point Y coordinate
    :param color: an RGB color
    :type color: tuple
    :param tolerance: how much filled pixels are permitted to vary
    :type tolerance: float [0.0, 1.0]
    :param offset: the post-fill expansion/contraction radius in pixels
    :type offset: int [-TILE_SIZE, TILE_SIZE]
    :param feather: the amount to blur the fill, after offset is applied
    :type feather: int [0, TILE_SIZE]
    :param gap_closing_options: parameters for gap closing fill, or None
    :type gap_closing_options: lib.floodfill.GapClosingOptions
    :param mode: Fill blend mode - normal, erasing or alpha locked
    :type mode: int (Any of the Combine* modes in mypaintlib)
    :param framed: Whether the frame is enabled or not.
    :type framed: bool
    :param bbox: Bounding box: limits the fill
    :type bbox: lib.helpers.Rect or equivalent 4-tuple
    :param dst: Target surface
    :type dst: lib.tiledsurface.MyPaintSurface

    The fill is performed with reference to src.
    The resulting tiles are composited into dst.
    """

    _, _, w, h = bbox
    if w <= 0 or h <= 0:
        return

    # Basic safety clamping
    tolerance = lib.helpers.clamp(tolerance, 0.0, 1.0)
    offset = lib.helpers.clamp(offset, -TILE_SIZE, TILE_SIZE)
    feather = lib.helpers.clamp(feather, 0, TILE_SIZE)

    # Initial parameters
    starting_point = starting_coordinates(x, y)
    r, g, b, a = get_target_color(src, *starting_point)
    filler = myplib.Filler(r, g, b, a, tolerance)

    fill_args = (src, starting_point, bbox, filler)

    if gap_closing_options:
        fill_args += (gap_closing_options,)
        filled = gap_closing_fill(*fill_args)
    else:
        filled = scanline_fill(*fill_args)

    # Dilate/Erode (Grow/Shrink)
    if offset != 0:
        filled = lib.morphology.morph(offset, filled)

    # Feather (Fake gaussian blur)
    if feather != 0:
        filled = lib.morphology.blur(feather, filled)

    # When dilating or blurring the fill, only respect the
    # bounding box limits if they are set by an active frame
    trim_result = framed and (offset > 0 or feather != 0)
    composite(mode, color, trim_result, filled, bbox, dst)


def composite(mode, fill_col, trim_result, filled, outer_bbox, dst):
    """Composite the filled tiles into the destination surface"""

    tiles_bbox, tile_pixel_bounds = tiles_bbox_and_bounds(outer_bbox)

    # Prepare opaque color rgba tile for copying
    full_rgba = myplib.fill_rgba(
        _FULL_TILE, *(fill_col + (0, 0, N-1, N-1)))

    # Composite filled tiles into the destination surface
    tiles_to_composite = filled.items() if PY3 else filled.iteritems()
    for tile_coord, src_tile in tiles_to_composite:

        # Omit tiles outside of the bounding box _if_ the frame is enabled
        # Note:filled tiles outside bbox only originates from dilation/blur
        if trim_result and out_of_bounds(tile_coord, tiles_bbox):
            continue
        # Skip empty source tiles (no fill to process)
        if src_tile is _EMPTY_TILE:
            continue
        with dst.tile_request(*tile_coord, readonly=False) as dst_tile:
            # Skip empty destination tiles for erasing and alpha locking
            if dst_tile is _EMPTY_RGBA and mode != myplib.CombineNormal:
                continue
            # Copy full tiles directly if not on the bounding box edge
            # unless the fill is dilated or blurred with no frame set
            cut_off = trim_result and across_bounds(tile_coord, tiles_bbox)
            if is_full(src_tile) and not cut_off:
                if mode == myplib.CombineNormal:
                    myplib.tile_copy_rgba16_into_rgba16(full_rgba, dst_tile)
                    continue
                elif mode == myplib.CombineDestinationOut:
                    myplib.tile_copy_rgba16_into_rgba16(_EMPTY_RGBA, dst_tile)
                    continue

            # Otherwise, composite the section with provided bounds into the
            # destination tile, most often the entire tile
            if trim_result:
                tile_bounds = tile_pixel_bounds(tile_coord)
            else:
                tile_bounds = (0, 0, N-1, N-1)
            src_tile_rgba = myplib.fill_rgba(
                src_tile, *(fill_col + tile_bounds))
            myplib.tile_combine(mode, src_tile_rgba, dst_tile, True, 1.0)

        dst._mark_mipmap_dirty(*tile_coord)
    bbox = lib.surface.get_tiles_bbox(filled)
    dst.notify_observers(*bbox)


def scanline_fill(src, init, bbox, filler):
    """ Perform a scanline fill and return the filled tiles

    Perform a scanline fill using the given starting point and tile,
    with reference to the src surface and given bounding box, using the
    provided filler instance.

    :param src: Source surface-like object
    :param init: coordinates for starting tile and pixel
    :type init: (int, int, int, int)
    :param bbox: Bounding box for the fill
    :type bbox: lib.helpers.Rect or equivalent 4-tuple
    :param filler: filler instance performing the per-tile fill operation
    :type filler: mypaintlib.Filler
    :returns: a dictionary of coord->tile mappings for the filled tiles
    """

    # Dict of coord->tile data populated during the fill
    filled = {}

    inv_edges = (
        myplib.edges.south,
        myplib.edges.west,
        myplib.edges.north,
        myplib.edges.east
    )

    # Starting coordinates + direction of origin (from within)
    _tx, _ty, _px, _py = init
    tileq = [((_tx, _ty), (_px, _py), myplib.edges.none)]

    tiles_bbox, tile_pixel_bounds = tiles_bbox_and_bounds(bbox)
    tfs = _TileFillSkipper(tiles_bbox, filler, set({}))

    while len(tileq) > 0:
        tile_coord, seeds, from_dir = tileq.pop(0)
        # Skip if the tile has been fully processed already
        if tile_coord in tfs.final:
            continue
        # Flood-fill one tile
        with src.tile_request(*tile_coord, readonly=True) as src_tile:
            # See if the tile can be skipped
            overflows = tfs.check(tile_coord, src_tile, filled, from_dir)
            if overflows is None:
                if tile_coord not in filled:
                    filled[tile_coord] = np.zeros((N, N), 'uint16')
                overflows = filler.fill(
                    src_tile, filled[tile_coord], seeds,
                    from_dir, *tile_pixel_bounds(tile_coord)
                )
        enqueue_overflows(tileq, tile_coord, overflows, tiles_bbox, inv_edges)
    return filled


class _TileFillSkipper:
    """Provides checking for, and handling of, uniform tiles"""

    FULL_OVERFLOWS = [
        ((), [(0, N-1)], [(0, N-1)], [(0, N-1)]),         # from north
        ([(0, N-1)], (), [(0, N-1)], [(0, N-1)]),         # from east
        ([(0, N-1)], [(0, N-1)], (), [(0, N-1)]),         # from south
        ([(0, N-1)], [(0, N-1)], [(0, N-1)], ()),         # from west
        ([(0, N-1)], [(0, N-1)], [(0, N-1)], [(0, N-1)])  # from within
    ]

    def __init__(self, tiles_bbox, filler, final):

        self.uniform_tiles = {}
        self.final = final
        self.bbox = tiles_bbox
        self.filler = filler

    # Dict of alpha->tile, used for uniform non-opaque tile fills
    # NOTE: these are usually not a result of an intentional fill, but
    # clicking a pixel with color very similar to the intended target pixel
    def uniform_tile(self, alpha):
        """ Return a reference to a uniform alpha tile

        If no uniform tile with the given alpha value exists, one is created
        """
        if alpha not in self.uniform_tiles:
            self.uniform_tiles[alpha] = np.full((N, N), alpha, 'uint16')
        return self.uniform_tiles[alpha]

    def check(self, tile_coord, src_tile, filled, from_dir):
        """Check if the tile can be handled without using the fill loop.

        The first time the tile is encountered, check if it is uniform
        and if so, handle it immediately depending on whether it is
        fillable or not.

        If the tile can be handled immediately, returns the overflows
        (new seed ranges), otherwise return None to indicate that the
        fill algorithm needs to be invoked.
        """
        if tile_coord in filled or not inside_bounds(tile_coord, self.bbox):
            return None

        # Returns the alpha of the fill for the tile's color if
        # the tile is uniform, otherwise returns None
        is_empty = src_tile is _EMPTY_RGBA
        alpha = self.filler.tile_uniformity(is_empty, src_tile)

        if alpha is None:
            # No shortcut can be taken, create new tile
            return None
        # Tile is uniform, so there is no need to process
        # it again in the fill loop, either set as
        # a uniformly filled alpha tile or skip it if it
        # cannot be filled at all (unlikely, but not impossible)
        self.final.add(tile_coord)
        if alpha == 0:
            filled[tile_coord] = _EMPTY_TILE
            return [(), (), (), ()]
        elif alpha == _OPAQUE:
            filled[tile_coord] = _FULL_TILE
        else:
            filled[tile_coord] = self.uniform_tile(alpha)
        return self.FULL_OVERFLOWS[from_dir]


def gap_closing_fill(src, init, bbox, filler, gap_closing_options):
    """ Fill loop that finds and uses gap data to avoid unwanted leaks

    Gaps are defined as distances of fillable pixels enclosed on two sides
    by unfillable pixels. Each tile considered, and their neighbours, are
    flooded with alpha values based on the target color and threshold values.
    The resulting alphas are then searched for gaps, and the size of these gaps
    are marked in separate tiles - one for each tile filled.
    """
    tiles_bbox, tile_bounds = tiles_bbox_and_bounds(bbox)

    full_alphas = {}
    distances = {}
    unseep_q = []
    filled = {}

    options = gap_closing_options
    max_gap_size = lib.helpers.clamp(options.max_gap_size, 1, TILE_SIZE)
    gc_filler = myplib.GapClosingFiller(max_gap_size, options.retract_seeps)
    distbucket = myplib.DistanceBucket(max_gap_size)

    init_tx, init_ty, init_px, init_py = init
    tileq = [((init_tx, init_ty), (init_px, init_py))]

    total_px = 0

    def gap_free(north, east, south, west):
        """Returns true if no gaps can possible cross the corner of the tile
        in the center of the given neighboring tiles
        """
        return myplib.no_corner_gaps(
            max_gap_size, north, east, south, west
        )

    while len(tileq) > 0:
        tile_coord, seeds = tileq.pop(0)
        # Pixel limits within tiles vary at the bounding box edges
        px_bounds = tile_bounds(tile_coord)
        # Create distance-data and alpha output tiles for the fill
        if tile_coord not in distances:
            # Ensure that alpha data exists for the tile and its neighbours
            prep_alphas(tile_coord, full_alphas, src, filler)
            grid = [full_alphas[ftc] for ftc in fc.nine_grid(tile_coord)]
            full = [is_full(tile) for tile in grid]
            # Skip full gap distance searches when possible
            # (marginal overall difference, but can reduce allocations)
            if all(full) or (is_full(grid[0]) and gap_free(*(grid[1:5]))):
                distances[tile_coord] = _GAPLESS_TILE
            else:
                dist_data = np.full((N, N), 2*N*N, 'uint16')
                # Search and mark any gap distances for the tile
                myplib.find_gaps(distbucket, dist_data, *grid)
                distances[tile_coord] = dist_data
            filled[tile_coord] = np.zeros((N, N), 'uint16')
        if isinstance(seeds, tuple):  # Fetch distance for initial seed coord
            dists = distances[tile_coord]
            init_x, init_y = seeds
            seeds = [(init_x, init_y, dists[init_y][init_x])]
        # Run the gap-closing fill for the tile
        result = gc_filler.fill(
            full_alphas[tile_coord], distances[tile_coord],
            filled[tile_coord], seeds, *px_bounds)
        overflows = result[0:4]
        enqueue_overflows(tileq, tile_coord, overflows, tiles_bbox)
        fill_edges, px_f = result[4:6]
        total_px += px_f
        if fill_edges:
            unseep_q.append((tile_coord, fill_edges, True))

    # Seep inversion is basically just a four-way 0-alpha fill
    # with different conditions. It only backs off into the original
    # fill and therefore does not require creation of new tiles
    backup = {}
    while len(unseep_q) > 0:
        tile_coord, seeds, is_initial = unseep_q.pop(0)
        if tile_coord not in distances or tile_coord not in filled:
            continue
        if tile_coord not in backup:
            backup[tile_coord] = np.copy(filled[tile_coord])
        result = gc_filler.unseep(
            distances[tile_coord], filled[tile_coord], seeds, is_initial
        )
        overflows = result[0:4]
        num_erased_pixels = result[4]
        total_px -= num_erased_pixels
        enqueue_overflows(
            unseep_q, tile_coord, overflows, tiles_bbox, (False,)*4
        )
    if total_px <= 0:
        # For small areas, when starting on a distance-marked pixel,
        # backing off may remove the entire fill, in which case we
        # roll back the tiles that were processed
        backup_pairs = backup.items() if PY3 else backup.iteritems()
        for tile_coord, tile in backup_pairs:
            filled[tile_coord] = tile

    # Check which tiles (if any) are fully opaque, and replace them
    for tc in filled:
        if (filled[tc] == _FULL_TILE).all():
            filled[tc] = _FULL_TILE

    return filled


def prep_alphas(tile_coord, full_alphas, src, filler):
    """When needed, create and calculate alpha tiles for distance searching.

    For the tile of the given coordinate, ensure that a corresponding tile
    of alpha values (based on the tolerance function) exists in the full_alphas
    dict for both the tile and all of its neighbors

    """
    for ntc in fc.nine_grid(tile_coord):
        if ntc not in full_alphas:
            with src.tile_request(
                ntc[0], ntc[1], readonly=True
            ) as src_tile:
                is_empty = src_tile is _EMPTY_RGBA
                alpha = filler.tile_uniformity(is_empty, src_tile)
                if alpha == _OPAQUE:
                    full_alphas[ntc] = _FULL_TILE
                else:
                    alpha_tile = np.empty((N, N), 'uint16')
                    filler.flood(src_tile, alpha_tile)
                    full_alphas[ntc] = alpha_tile
