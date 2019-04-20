# This file is part of MyPaint.
# Copyright (C) 2018-2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""This module implements tile-based morphological operations;
dilation, erosion and blur
"""
import math
import logging
import sys

import multiprocessing as mp
import numpy as np

import lib.mypaintlib as myplib

import lib.fill_common as fc
from lib.fill_common import _FULL_TILE, _EMPTY_TILE

from lib.pycompat import PY3

N = myplib.TILE_SIZE

logger = logging.getLogger(__name__)


def adjacent_tiles(tile_coord, filled):
    """ Return a tuple of tiles adjacent to the input tile coordinate.
    Adjacent tiles that are not in the tileset are replaced by the empty tile.
    """
    return tuple([filled.get(c, _EMPTY_TILE) for c in fc.adjacent(tile_coord)])


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


def complement_adjacent(tiles):
    """ Ensure that each tile in the input tileset has a full neighbourhood
    of eight tiles, setting missing tiles to the empty tile.

    The new set should only be used as input to tile operations, as the empty
    tile is readonly.
    """
    new = {}
    for tile_coord in tiles.keys():
        for adj_coord in fc.adjacent(tile_coord):
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

    # Try to use worker processes for large/heavy morphs
    if num_workers > 1 and sys.platform != "win32":
        try:
            return morph_multi(
                num_workers, offset, tiles, full_opaque,
                operation, strands, morphed
            )
        except Exception:
            logger.warn("Multiprocessing failed, using single core fallback")

    # Don't use workers for small workloads
    skip_t = _EMPTY_TILE if offset < 0 else _FULL_TILE
    for strand in strands:
        morph_strand(
            tiles, full_opaque, offset > 0,
            myplib.MorphBucket(se_size), operation,
            skip_t, _FULL_TILE, strand, morphed
        )
    return morphed


def morph_multi(
    num_workers, offset, tiles, full_opaque,
    operation, strands, morphed
):
    """Set up worker processes and a work queue to
    split up the morphological operations
    """
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
    return morphed


def morph_strand(
        tiles, full_opaque, skip_full, morph_bucket,
        operation, skip_tile, full_tile, keys, morphed):
    """ Apply a morphological operation to a strand of alpha tiles.

    Operates on vertical strands of tiles (same x-coordinate) to
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
                    [coord in full_opaque for coord in fc.adjacent(tile_coord)]
            ):
                morphed[tile_coord] = full_tile
                can_update = False
                continue

        # Perform the dilation/erosion
        center_tile = tiles[tile_coord]
        no_skip, morphed_tile = operation(
            morph_bucket, can_update, center_tile,
            *(adjacent_tiles(tile_coord, tiles))
        )
        # For very large radii, a small search is performed to see
        # if the actual morph operation can be skipped with the result
        # being either an empty or a full alpha tile.
        if no_skip:
            can_update = True
            # Skip the resulting tile if it is empty
            if center_tile is _EMPTY_TILE and not morphed_tile.any():
                continue
            morphed[tile_coord] = morphed_tile
        else:
            can_update = False
            morphed[tile_coord] = skip_tile


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
    return tiles


def blur_pass(tiles, blur_bucket):
    """Perform a single box blur pass for the given input tiles,
    returning the (potential) superset of blurred tiles"""
    # For each pass, create a new tile set for the blurred output,
    # which is then used as input for the next pass
    blurred = {}
    for strand in contig_vertical(tiles.keys()):
        can_update = False
        for tile_coord in strand:
            alpha_tile = tiles[tile_coord]
            adj = adjacent_tiles(tile_coord, tiles)
            adj_full = [(tile is _FULL_TILE) for tile in adj]
            # Skip tile if the full 9-tile neighbourhood is full
            if alpha_tile is _FULL_TILE and all(adj_full):
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
