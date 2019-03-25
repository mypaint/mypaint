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


# Constants acting as placeholders when distributing
# heavy morphological workloads across worker processes
_EMPTY_TILE_PH = 0
_FULL_TILE_PH = 1


def unproxy(tile):
    """Switch out proxy values to corresponding tile references

    This is used when distributing heavy morphological operations
    across multiple working processes, where the direct references
    cannot be used because the memory is not shared.
    """
    if isinstance(tile, int):
        return [_EMPTY_TILE, _FULL_TILE][tile]
    else:
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


def adjacent_tiles(tile_coord, filled):
    """ Return a tuple of tiles adjacent to the input tile coordinate.
    Adjacent tiles that are not in the tileset are replaced by the empty tile.
    """
    return tuple([filled.get(c, _EMPTY_TILE) for c in adjacent(tile_coord)])


def complement_adjacent(tiles):
    """ Ensure that each tile in the input tileset has a full neighbourhood
    of eight tiles, setting missing tiles to the empty tile.

    The new set should only be used as input to tile operations, as the empty
    tile is readonly.
    """
    new = {}
    for tile_coord in tiles.keys():
        for adj_coord in adjacent(tile_coord):
            if adj_coord not in tiles and adj_coord not in new:
                new[adj_coord] = _EMPTY_TILE
    tiles.update(new)


def directly_below(coord1, coord2):
    """ Return true if the first coordinate is directly below the second"""
    return coord1[0] == coord2[0] and coord1[1] == coord2[1] + 1


def contig_vertical(coords):
    """ Given a list of (x,y)-coordinates, group them in x, y order
    where groups consist of elements with the same x coordinate
    and consecutive y-coordinates

    (e.g) [[(1, 1),  (1, 2)], [(1, 4)], [(2, 4), (2, 5)]]
    """
    result = []
    group = []
    previous = None
    for tile_coord in sorted(coords):
        if previous is None or directly_below(tile_coord, previous):
            group.append(tile_coord)
        else:
            result.append(group)
            group = [tile_coord]
        previous = tile_coord
    if group:
        result.append(group)
    return result


def triples(num):
    """ Return a tuple of three minimally different
    terms whose sum equals the given integer argument
    """
    fraction = num / 3.0
    whole = num // 3
    floor = int(math.floor(fraction))
    ceil = int(math.ceil(fraction))
    if fraction - whole >= 0.5:
        return (ceil, ceil, floor)
    else:
        return (ceil, floor, floor)


def morph(offset, tiles, full_opaque):
    """ Either dilate or erode the given set of alpha tiles, depending
    on the sign of the offset, returning the set of morphed tiles.
    """
    operation = myplib.dilate if offset > 0 else myplib.erode
    # Radius of the structuring element used in the morph
    se_size = abs(offset)
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
    num_workers = int(min(cpus, math.sqrt((len(tiles) * se_size)) // 50))

    if num_workers > 1:
        # Use worker processes for large/heavy morphs
        mt0 = time.time()

        # Set up IPC communication channels and tile constants
        strand_queue = mp.Queue()
        morph_results = mp.Queue()

        # Use int constants instead of tile references, since
        # the references won't be the same for the workers
        skip_tile = _EMPTY_TILE_PH if offset < 0 else _FULL_TILE_PH

        # Create and start the worker processes
        for _ in range(num_workers):
            worker = mp.Process(
                target=morph_worker,
                args=(
                    tiles, full_opaque, strand_queue,
                    morph_results, offset, operation, skip_tile
                )
            )
            worker.start()

        # Populate the work queue with strands
        for strand in strands:
            strand_queue.put(strand)
        # Add a stop-signal value for each worker
        for signal in (None,) * num_workers:
            strand_queue.put(signal)

        # Merge the resulting tile dicts, replacing proxy constants
        # with their corresponding references for full/empty tiles
        for _ in range(num_workers):
            result = morph_results.get()
            result_items = result.items() if PY3 else result.iteritems()
            for tile_coord, tile in result_items:
                morphed[tile_coord] = unproxy(tile)

        logger.info(
            "%.3f s. to morph with %d workers",
            time.time() - mt0, num_workers)
        return morphed
    else:
        # Don't use workers for small workloads
        mt1 = time.time()
        skip_t = _EMPTY_TILE if offset < 0 else _FULL_TILE
        for strand in strands:
            morph_strand(
                tiles, full_opaque, offset > 0,
                myplib.MorphBucket(se_size), operation,
                skip_t, _FULL_TILE, strand, morphed
            )
        logger.info("%.3f s. to morph without workers", time.time() - mt1)
        return morphed


def morph_strand(
        tiles, full_opaque, skip_full, morph_bucket,
        operation, skip_tile, full_tile, keys, morphed):
    """ Apply a morphological operation to a strand of alpha tiles.

    We operate on vertical strands of tiles (same x-coordinate) due to
    maximize the potential reuse of the UW* lookup table when moving from
    one tile to the next. Skipping tiles is still faster and therefore
    always prioritized when possible.

    * Urbach-Wilkinson (https://doi.org/10.1109/TIP.2007.9125824)
    """
    can_update = False  # reuse most of the data from the previous operation
    for tile_coord in keys:
        if tile_coord in full_opaque:
            # For dilation, skip all full tiles
            # For erosion, skip full tiles when all neighbours are full too
            if skip_full or all(
                    [coord in full_opaque for coord in adjacent(tile_coord)]
            ):
                morphed[tile_coord] = full_tile
                can_update = False
                continue
        # Perform the dilation/erosion
        no_skip, morphed_tile = operation(
            morph_bucket, can_update, tiles[tile_coord],
            *(adjacent_tiles(tile_coord, tiles))
        )
        # For very large radiuses, a small search is performed to see
        # if the actual morph operation can be skipped with the result
        # being either an empty or a full alpha tile.
        if no_skip:
            morphed[tile_coord] = morphed_tile
            can_update = True
        else:
            morphed[tile_coord] = skip_tile
            can_update = False


def morph_worker(
        tiles, full_opaque, strand_queue, results,
        offset, morph_op, skip_tile):
    """ tile morphing worker function invoked by separate processes
    """
    morph_bucket = myplib.MorphBucket(abs(offset))
    morphed = {}
    # Fetch and process strands from the work queue
    # until a stop signal value is fetched
    while True:
        keys = strand_queue.get()
        if keys is None:
            break
        morph_strand(
            tiles, full_opaque, offset > 0, morph_bucket, morph_op,
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
    prev_radius = 0
    blur_bucket = None
    for radius in radiuses:
        if prev_radius != radius:
            blur_bucket = myplib.BlurBucket(radius)
        tiles = blur_pass(tiles, blur_bucket)
    logger.info("Time to blur: %.3f seconds", time.time() - t0)
    return tiles


def blur_pass(tiles, blur_bucket):
    """Perform a single box blur pass for the given input tiles,
    returning the (potential) superset of blurred tiles"""
    # For each pass, we create a new tile set for the blurred output,
    # which are then used as input for the next pass
    blurred = {}
    for strand in contig_vertical(tiles.keys()):
        can_update = False
        for tile_coord in strand:
            alpha_tile = tiles[tile_coord]
            adj = adjacent_tiles(tile_coord, tiles)
            adj_full = [is_full(tile) for tile in adj]
            # Skip tile if the full 9-tile neighbourhood is full
            if is_full(alpha_tile) and all(adj_full):
                blurred[tile_coord] = _FULL_TILE
                can_update = False
                continue
            # Unless skipped, create a new output tile
            # and run the box blur on the input tiles
            blurred[tile_coord] = np.empty((N, N), 'uint16')
            myplib.blur(
                blur_bucket, can_update,
                alpha_tile, blurred[tile_coord], *adj
            )
            can_update = True
    return blurred


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


# Main fill handling function

def flood_fill(
        src, x, y, color, tolerance, offset, feather,
        gap_closing_options, framed, bbox, dst):
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
    :param framed: Whether the frame is enabled or not.
    :type framed: bool
    :param bbox: Bounding box: limits the fill
    :type bbox: lib.helpers.Rect or equivalent 4-tuple
    :param dst: Target surface
    :type dst: lib.tiledsurface.MyPaintSurface

    The fill is performed with reference to src.
    The resulting tiles are composited into dst.
    """

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
        """ Return the in-tile pixel bounds as a 4-tuple
        Bounds cover the entire tile, unless it is located
        on the edge of the bounding box.
        """
        tile_x, tile_y = tile_coords
        min_x = min_px if tile_x == min_tx else 0
        min_y = min_py if tile_y == min_ty else 0
        max_x = max_px if tile_x == max_tx else N-1
        max_y = max_py if tile_y == max_ty else N-1
        return min_x, min_y, max_x, max_y

    # Tile and pixel addressing for the seed point
    init_tx, init_ty = int(x // N), int(y // N)
    init_x, init_y = int(x % N), int(y % N)

    # Sample the pixel color there to obtain the target color
    with src.tile_request(init_tx, init_ty, readonly=True) as start:
        targ_r, targ_g, targ_b, targ_a = [
            int(c) for c in start[init_y][init_x]
        ]
    if targ_a == 0:
        targ_r, targ_g, targ_b = 0, 0, 0

    # Set of coordinates of fully opaque filled tiles, used to potentially
    # bypass dilation/erosion and blur operations for contiguous opaque areas

    filler = myplib.Filler(targ_r, targ_g, targ_b, targ_a, tolerance)
    init = (init_tx, init_ty, init_x, init_y)
    fill_args = (src, init, tiles_bbox, tile_bounds, filler)

    # Profiling
    t0 = time.time()

    if gap_closing_options:
        filled = gap_closing_fill(*(fill_args + (gap_closing_options,)))
        full_opaque = set({})
    else:
        filled, full_opaque = scanline_fill(*(fill_args))

    t1 = time.time()
    logger.info("%.3f seconds to fill", t1 - t0)

    # Dilate/Erode, Grow/Contract, Expand/Shrink
    if offset != 0:
        filled = morph(offset, filled, full_opaque)

    # Feather (Fake gaussian blur)
    if feather != 0:
        filled = blur(feather, filled)

    # When dilating or blurring the fill, only respect the
    # bounding box limits if they are set by an active frame
    trim_result = framed and (offset > 0 or feather != 0)
    mode = myplib.CombineNormal
    composite(
        mode, color, trim_result,
        filled, tiles_bbox, tile_bounds, dst)

    logger.info("Total time for fill: %.3f seconds", time.time() - t0)


def composite(
        mode, fill_col, trim_result,
        filled, bbox, bounds, dst):
    """Composite the filled tiles into the destination surface"""

    # When filling large areas, copying a full tile directly
    # (when possible) greatly improves performance.
    full_rgba = myplib.fill_rgba(
        _FULL_TILE, *(fill_col + (0, 0, N-1, N-1)))

    # Composite filled tiles into the destination surface
    tiles_to_composite = filled.items() if PY3 else filled.iteritems()
    for tile_coord, src_tile in tiles_to_composite:

        # Omit tiles outside of the bounding box _if_ the frame is enabled
        # Note:filled tiles outside bbox only originates from dilation/blur
        if trim_result and out_of_bounds(tile_coord, bbox):
            continue
        # Skip empty source tiles (no fill to process)
        if src_tile is _EMPTY_TILE:
            continue
        with dst.tile_request(*tile_coord, readonly=False) as dst_tile:
            # Skip empty destination tiles if we are erasing
            if dst_tile is _EMPTY_RGBA and mode == myplib.CombineSourceAtop:
                continue
            # Copy full tiles directly if not on the bounding box edge
            # unless the fill is dilated or blurred with no frame set
            cut_off = trim_result and across_bounds(tile_coord, bbox)
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
                tile_bounds = bounds(tile_coord)
            else:
                tile_bounds = (0, 0, N-1, N-1)
            src_tile_rgba = myplib.fill_rgba(
                src_tile, *(fill_col + tile_bounds))
            myplib.tile_combine(mode, src_tile_rgba, dst_tile, True, 1.0)

        dst._mark_mipmap_dirty(*tile_coord)
    bbox = lib.surface.get_tiles_bbox(filled)
    dst.notify_observers(*bbox)


def scanline_fill(
        src, init, tiles_bbox, bounds,
        filler):
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
    :param full_opaque: set of coords to be amended by coords of full tiles
    :type full_opaque: set

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

    tileq = [(init[0:2], init[2:4], myplib.edges.none)]

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
                    from_dir, *bounds(tile_coord)
                )
        enqueue_overflows(tileq, tile_coord, overflows, tiles_bbox, inv_edges)
    return filled, tfs.full_opaque


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
        self.full_opaque = set({})
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
            self.full_opaque.add(tile_coord)
        else:
            filled[tile_coord] = self.uniform_tile(alpha)
        return self.FULL_OVERFLOWS[from_dir]


def gap_closing_fill(
        src, init, tiles_bbox, tile_bounds,
        filler, gap_closing_options):
    """ Fill loop that finds and uses gap data to avoid unwanted leaks

    Gaps are defined as distances of fillable pixels enclosed on two sides
    by unfillable pixels. Each tile considered, and their neighbours, are
    flooded with alpha values based on the target color and threshold values.
    The resulting alphas are then searched for gaps, and the size of these gaps
    are marked in separate tiles - one for each tile filled.
    """
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
            grid = [full_alphas[ftc] for ftc in nine_grid(tile_coord)]
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
    return filled


def prep_alphas(tile_coord, full_alphas, src, filler):
    """When needed, create and calculate alpha tiles for distance searching.

    For the tile of the given coordinate, ensure that a corresponding tile
    of alpha values (based on the tolerance function) exists in the full_alphas
    dict for both the tile and all of its neighbors

    """
    for ntc in nine_grid(tile_coord):
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
