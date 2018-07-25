# This file is part of MyPaint.
# Copyright (C) 2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""This module implements tile-based floodfill and related operations."""

import math
import time
import logging

import multiprocessing as mp
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
# operations as well as feathering and compositing
_FULL_TILE = np.full((N, N), _OPAQUE, 'uint16')
_FULL_TILE.flags.writeable = False

# Keeping track of empty tiles (which are less likely to
# be produced during the fill) permits skipping the compositing
# step for these tiles
_EMPTY_TILE = np.zeros((N, N), 'uint16')
_EMPTY_TILE.flags.writeable = False


def is_full(t):
    return t is _FULL_TILE


# Constants acting as placeholders when distributing
# heavy morphological workloads across worker processes
_EMPTY_TILE_PH = 0
_FULL_TILE_PH = 1


# Switches out proxy references for real ones when handling
# tiles returned from a worker process
def unproxy(tile):
    if type(tile) is int:
        return [_EMPTY_TILE, _FULL_TILE][tile]
    else:
        return tile


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


def adjacent_tiles(tc, filled):
    """ Return a tuple of tiles adjacent to the input tile coordinate.
    Adjacent tiles that are not in the tileset are replaced by the empty tile.
    """
    return tuple([filled.get(c, _EMPTY_TILE) for c in adjacent(tc)])


def complement_adjacent(tiles):
    """ Ensure that each tile in the input tileset has a full neighbourhood
    of eight tiles, setting missing tiles to the empty tile.

    The new set should only be used as input to tile operations, as the empty
    tile is readonly.
    """
    new = {}
    for tc in tiles.keys():
        for c in adjacent(tc):
            if c not in tiles and c not in new:
                new[c] = _EMPTY_TILE
    tiles.update(new)


def contig_vertical(coords):
    """ Given a list of (x,y)-coordinates, group them in x, y order
    where groups consist of elements with the same x coordinate
    and consecutive y-coordinates

    (e.g) [[(1, 1),  (1, 2)], [(1, 4)], [(2, 4), (2, 5)]]
    """
    result = []
    group = []
    prev = None
    for (tx, ty) in sorted(coords):
        if prev is None or (prev[0] == tx and prev[1] == ty - 1):
            group.append((tx, ty))
        else:
            result.append(group)
            group = [(tx, ty)]
        prev = (tx, ty)
    if group:
        result.append(group)
    return result


def triples(num):
    """ Return a tuple of three minimally different
    terms whose sum equals the given integer argument
    """
    k = num / 3.0
    p = num // 3
    floor = int(math.floor(k))
    ceil = int(math.ceil(k))
    if k - p >= 0.5:
        return (ceil, ceil, floor)
    else:
        return (ceil, floor, floor)


def morph(offset, tiles, full_opaque):
    """ Either dilate or erode the given set of alpha tiles, depending
    on the sign of the offset, returning the set of morphed tiles.
    """
    op = myplib.dilate if offset > 0 else myplib.erode
    sz = abs(offset)
    # When dilating, create new tiles to account for edge overflow
    # (without checking if they are actually needed)
    if offset > 0:
        complement_adjacent(tiles)

    # Split up the coordinates of the tiles to morphed into
    # contiguous strands, which can be processed more efficiently
    strands = contig_vertical(tiles.keys())
    morphed = {}

    # Use a rough heuristic based on the number of tiles that need
    # processing and the size of the erosion/dilation
    cpus = mp.cpu_count()
    num_workers = int(min(cpus, math.sqrt((len(tiles) * sz)) // 50))

    if num_workers > 1:
        # Use worker processes for large/heavy morphs
        mt0 = time.time()

        # Set up IPC communication channels and tile constants
        strand_queue = mp.Queue()
        morph_results = mp.Queue()

        # Use int constants instead of tile references, since
        # the references won't be the same for the workers
        skip_t = _EMPTY_TILE_PH if offset < 0 else _FULL_TILE_PH

        # Create and start the worker processes
        for w in range(num_workers):
            wp = mp.Process(
                target=morph_worker,
                args=(
                    tiles, full_opaque, strand_queue,
                    morph_results, offset, op, skip_t
                )
            )
            wp.start()

        # Populate the work queue with strands
        for s in strands:
            strand_queue.put(s)
        # Add a stop-signal value for each worker
        for w in range(num_workers):
            strand_queue.put(None)

        # Merge the resulting tile dicts, replacing proxy constants
        # with their corresponding references for full/empty tiles
        for w in range(num_workers):
            i = morph_results.get()
            results = i.items() if PY3 else i.iteritems()
            for tc, tile in results:
                morphed[tc] = unproxy(tile)

        logger.info(
            "%.3f s. to morph with %d workers",
            time.time() - mt0, num_workers)
        return morphed
    else:
        # Don't use workers for small workloads
        mt1 = time.time()
        skip_t = _EMPTY_TILE if offset < 0 else _FULL_TILE
        for s in strands:
            morph_strand(
                tiles, full_opaque, offset > 0,
                myplib.MorphBucket(sz), op,
                skip_t, _FULL_TILE, s, morphed
            )
        logger.info("%.3f s. to morph without workers", time.time() - mt1)
        return morphed


def morph_strand(
        tiles, full_opaque, skip_full, mb,
        op, skip_t, full_t, keys, morphed):
    """ Apply morphological operation to a strand of alpha tiles.
    """
    can_update = False  # reuse most of the data from the previous operation
    for tc in keys:
        # For dilation, skip all full tiles
        # For erosion, skip full tiles when all of its neighbours are full
        if tc in full_opaque:
            adj_full = map(lambda c: c in full_opaque, adjacent(tc))
            if skip_full or all(adj_full):
                morphed[tc] = full_t
                can_update = False
                continue
        # Perform the dilation/erosion
        no_skip, morphed_tile = op(
            mb, can_update, tiles[tc], *(adjacent_tiles(tc, tiles))
        )
        # For very large radiuses, a small search is performed to see
        # if the actual morph operation can be skipped with the result
        # being either an empty or a full alpha tile.
        if no_skip:
            morphed[tc] = morphed_tile
            can_update = True
        else:
            morphed[tc] = skip_t
            can_update = False


def morph_worker(
        tiles, full_opaque, strand_queue, results,
        offset, op, skip_tile):
    """ tile morphing worker function invoked by separate processes
    """
    mb = myplib.MorphBucket(abs(offset))
    morphed = {}
    # Fetch and process strands from the work queue
    # until a stop signal value is fetched
    while True:
        keys = strand_queue.get()
        if keys is None:
            break
        morph_strand(
            tiles, full_opaque, offset > 0, mb, op,
            skip_tile, _FULL_TILE_PH, keys, morphed)
    results.put(morphed)


def blur(feather, tiles):
    """ Return the set of blurred tiles based on the input tiles.
    """
    t0 = time.time()
    # Single pixel feathering uses a single box blur
    # radiuses > 2 uses three iterations with radiuses
    # adding up to the feather radius
    if feather == 1:
        radiuses = (1,)
    elif feather == 2:
        radiuses = (1, 1)
    else:
        radiuses = triples(feather)

    # Only expand the the tile coverage once, assuming a maximum
    # total blur radius (feather value) of TILE_SIZE
    complement_adjacent(tiles)
    prev_r = 0
    bb = None
    for r in radiuses:
        if prev_r != r:
            bb = myplib.BlurBucket(r)
        # For each pass, we create a new tile set for the blurred output,
        # which are then used as input for the next pass
        blurred = {}
        for strand in contig_vertical(tiles.keys()):
            can_update = False
            for tc in strand:
                alpha_tile = tiles[tc]
                adj = adjacent_tiles(tc, tiles)
                adj_full = map(is_full, adj)

                # Skip tile if the full 9-tile neighbourhood is full
                if is_full(alpha_tile) and all(adj_full):
                    blurred[tc] = _FULL_TILE
                    can_update = False
                    continue

                # Unless skipped, create a new output tile
                # and run the box blur on the input tiles
                blurred[tc] = np.empty((N, N), 'uint16')
                myplib.blur(bb, can_update, alpha_tile, blurred[tc], *adj)
                can_update = True
        tiles = blurred
    logger.info("Time to blur: %.3f seconds", time.time() - t0)
    return tiles


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

def flood_fill(
        src, x, y, color, tolerance, offset, feather,
        framed, bbox, dst, empty_rgba):
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
    :param framed: Whether the frame is enabled or not.
    :type framed: bool
    :param bbox: Bounding box: limits the fill
    :type bbox: lib.helpers.Rect or equivalent 4-tuple
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
    offset = lib.helpers.clamp(offset, -TILE_SIZE, TILE_SIZE)
    feather = lib.helpers.clamp(feather, 0, TILE_SIZE)

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

    # Dilate/Erode, Grow/Contract, Expand/Shrink
    if offset != 0:
        filled = morph(offset, filled, full_opaque)

    # Feather (Fake gaussian blur)
    if feather != 0:
        filled = blur(feather, filled)

    # When filling large areas, copying a full tile directly
    # when possible greatly improves performance.
    full_rgba = myplib.full_rgba_tile(fill_r, fill_g, fill_b)

    # When dilating the fill, only respect the
    # bounding box limits if they are set by an active frame
    trim_result = framed and (offset > 0 or feather != 0)

    # Composite filled tiles into the destination surface
    tiles_to_composite = filled.items() if PY3 else filled.iteritems()
    for tc, src_tile in tiles_to_composite:
        # Omit tiles outside of the bounding box _if_ the frame is enabled
        # Note:filled tiles outside bbox only originates from dilation/blur
        if trim_result and out_of_bounds(tc, tiles_bbox):
            continue
        with dst.tile_request(*tc, readonly=False) as dst_tile:
            # Skip empty tiles
            if src_tile is _EMPTY_TILE:
                continue
            # Copy full tiles directly if not on the bounding box edge
            # unless the fill is dilated or blurred with no frame
            frame_constrained = trim_result and across_bounds(tc, tiles_bbox)
            if is_full(src_tile) and not frame_constrained:
                myplib.tile_copy_rgba16_into_rgba16(full_rgba, dst_tile)
                continue
            # Otherwise, composite the section with provided bounds into the
            # destination tile, most often the entire tile
            if trim_result:
                t_bounds = tile_bounds(tc)
            else:
                t_bounds = (0, 0, N-1, N-1)
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
                    # cannot be filled at all (unlikely, but not impossible)
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
