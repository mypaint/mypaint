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
import logging

import lib.mypaintlib as myplib

import lib.fill_common as fc
from lib.fill_common import _FULL_TILE, _EMPTY_TILE

N = myplib.TILE_SIZE

logger = logging.getLogger(__name__)


def adjacent_tiles(tile_coord, filled: Types.ELLIPSIS) -> Types.NONE:
    """

    Args:
        tile_coord: 
        filled: 

    Returns:
        Adjacent tiles that are not in the tileset are replaced by the empty tile.

    Raises:

    """
    return tuple([filled.get(c, _EMPTY_TILE) for c in fc.adjacent(tile_coord)])


def complement_adjacent(tiles: Types.ELLIPSIS) -> Types.NONE:
    """Ensure that each tile in the input tileset has a full neighbourhood
    of eight tiles, setting missing tiles to the empty tile.
    
    The new set should only be used as input to tile operations, as the empty
    tile is readonly.

    Args:
        tiles: 

    Returns:

    Raises:

    """
    new = {}
    for tile_coord in tiles.keys():
        for adj_coord in fc.adjacent(tile_coord):
            if adj_coord not in tiles and adj_coord not in new:
                new[adj_coord] = _EMPTY_TILE
    tiles.update(new)


def directly_below(coord1, coord2: Types.ELLIPSIS) -> Types.NONE:
    """

    Args:
        coord1: 
        coord2: 

    Returns:
        

    Raises:

    """
    return coord1[0] == coord2[0] and coord1[1] == coord2[1] + 1


def strand_partition(tiles, dilating: Types.ELLIPSIS = False) -> Types.NONE:
    """Partition input tiles for easier processing
    This function partitions a tile dictionary into
    two parts: one dictionary containing tiles that
    do not need to be processed further (see note),
    and list of coordinate lists, where each list
    contains vertically contiguous coordinates,
    ordered from low to high.
    
    note: Tiles that never need further processing are
    those that are fully opaque and with a full neighbourhood
    of identical tiles. If the "dilating" parameter is set
    to true, just being fully opaque is enough.
    :return: (final_dict, strands_list)

    Args:
        tiles: 
        dilating:  (Default value = False)

    Returns:

    Raises:

    """
    # Dict of coord->tile for tiles that need no further processing
    final_tiles = {}
    # Groups of contiguous tile coordinates
    strands = []
    strand = []
    previous = None
    coords = tiles.keys()
    for tile_coord in sorted(coords):
        is_full_tile = tiles[tile_coord] is _FULL_TILE
        if is_full_tile and (dilating or adj_full(tile_coord, tiles)):
            # Tile needs no processing
            final_tiles[tile_coord] = _FULL_TILE
            previous = None
            if strand:
                strands.append(strand)
                strand = []
        elif previous is None or directly_below(tile_coord, previous):
            # Either beginning of new strand, or adds to existing one
            strand.append(tile_coord)
        else:
            # Neither final, nor contiguous, begin new strand
            strands.append(strand)
            strand = [tile_coord]
        previous = tile_coord
    if strand:
        strands.append(strand)
    return final_tiles, strands


def morph(handler, offset, tiles):
    # type: (Types.ELLIPSIS) -> Types.NONE
    """Either dilate or erode the given set of alpha tiles, depending
    on the sign of the offset, returning the set of morphed tiles.

    Args:
        handler: 
        offset: 
        tiles: 

    Returns:

    Raises:

    """
    # When dilating, create new tiles to account for edge overflow
    # (without checking if they are actually needed)
    if offset > 0:
        complement_adjacent(tiles)

    handler.set_stage(handler.MORPH, len(tiles))

    # Split up the coordinates of the tiles to morph, into vertically
    # contiguous strands, which can be processed more efficiently
    morphed, strands = strand_partition(tiles, offset > 0)
    # Run the morph operation (C++, conditionally threaded)
    myplib.morph(offset, morphed, tiles, strands, handler.controller)
    return morphed


def blur(handler, radius, tiles):
    # type: (Types.ELLIPSIS) -> Types.NONE
    """

    Args:
        handler: 
        radius: 
        tiles: 

    Returns:
        

    Raises:

    """
    complement_adjacent(tiles)

    handler.set_stage(handler.BLUR, len(tiles))

    blurred, strands = strand_partition(tiles, dilating=False)
    myplib.blur(radius, blurred, tiles, strands, handler.controller)
    return blurred


def adj_full(coord, tiles: Types.ELLIPSIS) -> Types.NONE:
    """

    Args:
        coord: 
        tiles: 

    Returns:

    Raises:

    """
    return all(t is _FULL_TILE for t in adjacent_tiles(coord, tiles))
