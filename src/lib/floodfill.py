# This file is part of MyPaint.
# Copyright (C) 2018-2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""This module implements tile-based floodfill and related operations."""

import logging

import numpy as np
import threading

from lib.gibindings import GLib

import lib.helpers
import lib.mypaintlib as myplib
import lib.surface
import lib.tiledsurface
from lib.gettext import C_
import lib.fill_common as fc
from lib.fill_common import _OPAQUE, _FULL_TILE, _EMPTY_TILE
import lib.modes
import lib.morphology

from lib.pycompat import iteritems

logger = logging.getLogger(__name__)

TILE_SIZE = N = myplib.TILE_SIZE
INF_DIST = 2*N*N

# This should point to the array transparent_tile.rgba
# defined in tiledsurface.py
_EMPTY_RGBA = None

# Distance data for tiles with no detected distances
_GAPLESS_TILE = fc.new_full_tile(INF_DIST)
_GAPLESS_TILE.flags.writeable = False

EDGE = myplib.edges


class GapClosingOptions:
    """Container of parameters for gap closing fill operations
    to avoid updates to the call chain in case the parameter set
    is altered.
    """
    def __init__(self, max_gap_size, retract_seeps):
        self.max_gap_size = max_gap_size
        self.retract_seeps = retract_seeps


def enqueue_overflows(queue, tile_coord, seeds, tiles_bbox, *p):
    """ Conditionally add (coordinate, seed list, data...) tuples to a queue.

    :param queue: the queue which may be appended
    :type queue: list
    :param tile_coord: the 2d coordinate in the middle of the seed coordinates
    :type tile_coord: (int, int)
    :param seeds: 4-tuple of seed lists for n, e, s, w, relative to tile_coord
    :type seeds: (list, list, list, list)
    :param tiles_bbox: the bounding box of the fill operation
    :type tiles_bbox: lib.fill_common.TileBoundingBox
    :param p: tuples of length >= 4, items added to queue items w. same index

    NOTE: This function improves readability significantly in exchange for a
    small performance hit. Replace with explicit queueing if too slow.
    """
    for edge in zip(*(fc.orthogonal(tile_coord), seeds) + p):
        edge_coord = edge[0]
        edge_seeds = edge[1]
        if edge_seeds and not tiles_bbox.outside(edge_coord):
            queue.append(edge)


def starting_coordinates(x, y):
    """Get the coordinates of starting tile and pixel (tx, ty, px, py)"""
    init_tx, init_ty = int(x // N), int(y // N)
    init_x, init_y = int(x % N), int(y % N)
    return init_tx, init_ty, init_x, init_y


def seeds_by_tile(seeds):
    """Partition and convert seed coordinates

    Partition a list of model-space seed coordinates into lists of
    in-tile coordinates associated to their respective tile coordinate
    in a dictionary.
    """
    tile_seeds = dict()
    for (x, y) in seeds:
        tx, ty, px, py = starting_coordinates(x, y)
        seed_list = tile_seeds.get((tx, ty), [])
        seed_list.append((px, py))
        tile_seeds[(tx, ty)] = seed_list
    return tile_seeds


def get_target_color(src, tx, ty, px, py):
    """Get the pixel color for the given tile/pixel coordinates"""
    with src.tile_request(tx, ty, readonly=True) as start:
        targ_r, targ_g, targ_b, targ_a = [
            int(c) for c in start[py][px]
        ]
    if targ_a == 0:
        targ_r, targ_g, targ_b = 0, 0, 0

    return targ_r, targ_g, targ_b, targ_a


# Main fill interface

class FillHandler:
    """Handles fill status and cancellation

    The fill is run in a separate thread, and this controller
    is used to start and (optionally) cancel the fill, as well
    as provide information about its current status.
    """

    # Stages that the fill operation can be in
    FILL = 0
    MORPH = 1
    BLUR = 2
    COMPOSITE = 3
    FINISHING = 4

    STAGE_STRINGS = [
        C_("floodfill status message: use active tense", "Filling"),
        C_("floodfill status message: use active tense", "Morphing"),
        C_("floodfill status message: use active tense", "Blurring"),
        C_("floodfill status message: use active tense", "Compositing"),
        C_("floodfill status message: use active tense", "Finishing up")
    ]
    TILES_STRING = C_("uniform square region of pixels, plural noun", "tiles")
    TILES_TEMPLATE = "{t} " + TILES_STRING

    def __init__(self):
        # An object with a "keep running" flag and "tiles processed"
        # data, for easier C++ access in morph/blur stages
        self.controller = myplib.Controller()
        # Separate "keep running" flag checked in Python code
        self.run = True
        self.stage = None
        self.set_stage(self.FILL)
        # When morphing, blurring or compositing,
        # this is the total amount of tiles to process
        self.tiles_max = 0
        self.fill_thread = None

    @property
    def tiles_processed(self):
        """The number of tiles processed for the current stage"""
        return self.controller.num_processed()

    def inc_processed(self):
        """Increment the number of tiles processed by 1"""
        self.controller.inc_processed(1)

    def set_stage(self, stage, num_tiles_to_process=None):
        """Change stage, updating strings and tile data"""
        self.stage = stage
        self.controller.reset()
        self.stage_string = self.STAGE_STRINGS[stage]
        if num_tiles_to_process:
            self.tiles_max = num_tiles_to_process
            self.stage_string += (
                " " + self.TILES_TEMPLATE.format(t=num_tiles_to_process))

    @property
    def progress_string(self):
        """Progress for the current stage"""
        if self.stage == self.FILL:
            return self.TILES_TEMPLATE.format(t=self.tiles_processed)
        elif self.stage < self.FINISHING:
            return str(int(100*self.tiles_processed/self.tiles_max)) + "%"
        else:
            return ""

    def wait(self, t=None):
        """Wait t seconds for the fill to complete"""
        self.fill_thread.join(t)

    def running(self):
        """Check if the fill is still running"""
        return self.fill_thread.is_alive()

    def cancel(self):
        """Tell the fill to stop as soon as possible"""
        self.controller.stop()
        self.run = False


class FloodFillArguments(object):
    """Container holding a set of flood fill arguments
    The purpose of this class is to avoid unnecessary
    call chain updates when changing/adding parameters.
    """

    def __init__(
            self, target_pos, seeds, color, tolerance, offset, feather,
            gap_closing_options, mode, lock_alpha, opacity, framed, bbox
    ):
        """ Create a new fill argument set
        :param target_pos: pixel coordinate of target color
        :type target_pos: tuple
        :param seeds: set of seed pixel coordinates {(x, y)...}
        :type seeds: set
        :param color: an RGB color
        :type color: tuple
        :param tolerance: how much filled pixels are permitted to vary
        :type tolerance: float [0.0, 1.0]
        :param offset: the post-fill expansion/contraction radius in pixels
        :type offset: int [-TILE_SIZE, TILE_SIZE]
        :param feather: the amount to blur the fill, after offset is applied
        :type feather: int [0, TILE_SIZE]
        :param gap_closing_options: parameters for gap closing fill, or None
        :type gap_closing_options: GapClosingOptions
        :param mode: Fill blend mode - normal, erasing or alpha locked
        :type mode: int (Any of the Combine* modes in mypaintlib)
        :param lock_alpha: Lock alpha of the destination layer
        :type lock_alpha: bool
        :param opacity: opacity of the fill
        :type opacity: float
        :param framed: Whether the frame is enabled or not.
        :type framed: bool
        :param bbox: Bounding box: limits the fill
        :type bbox: lib.helpers.Rect or equivalent 4-tuple
        """
        self.target_pos = target_pos
        self.seeds = seeds
        self.color = color
        self.tolerance = tolerance
        self.offset = offset
        self.feather = feather
        self.gap_closing_options = gap_closing_options
        self.mode = mode
        self.lock_alpha = lock_alpha
        self.opacity = opacity
        self.framed = framed
        self.bbox = bbox

    def skip_empty_dst(self):
        """If true, compositing to empty tiles does nothing"""
        return (
            self.lock_alpha or self.mode in [
                myplib.CombineSourceAtop,
                myplib.CombineDestinationOut,
                myplib.CombineDestinationIn,
            ]
        )

    def no_op(self):
        """If true, compositing will never alter the output layer

        These are comp modes for which alpha locking does not really
        make any sense, as any visible change caused by them requires
        the alpha of the destination to change as well.
        """
        return self.lock_alpha and (
            self.mode in lib.modes.MODES_DECREASING_BACKDROP_ALPHA
        )


def flood_fill(src, fill_args, dst):
    """Top-level fill interface
    Delegates actual fill in separate thread and returns a FillHandler

    :param src: source, surface-like object
    :type src: anything supporting readonly tile_request()
    :param fill_args: arguments common to all fill calls
    :type fill_args: FloodFillArguments
    :param dst: target surface
    :type dst: lib.tiledsurface.MyPaintSurface
    """
    handler = FillHandler()
    fill_function_args = (src, fill_args, dst, handler)
    fill_thread = threading.Thread(target=_flood_fill, args=fill_function_args)
    handler.fill_thread = fill_thread
    fill_thread.start()
    return handler


def _flood_fill(src, args, dst, handler):
    """Main flood fill function

    The fill is performed with reference to src.
    The resulting tiles are composited into dst.

    :param src: source, surface-like object
    :type src: anything supporting readonly tile_request()
    :param args: arguments common to all fill calls
    :type args: FloodFillArguments
    :param dst: target surface
    :type dst: lib.tiledsurface.MyPaintSurface
    :param handler: controller used to track state and cancel fill
    :type handler: FillHandler
    """
    _, _, width, height = args.bbox
    if width <= 0 or height <= 0 or args.no_op():
        return

    tiles_bbox = fc.TileBoundingBox(args.bbox)

    # Basic safety clamping
    tolerance = lib.helpers.clamp(args.tolerance, 0.0, 1.0)
    offset = lib.helpers.clamp(args.offset, -TILE_SIZE, TILE_SIZE)
    feather = lib.helpers.clamp(args.feather, 0, TILE_SIZE)

    # Initial parameters
    target_color = get_target_color(
        src, *starting_coordinates(*args.target_pos)
    )
    filler = myplib.Filler(*(target_color + (tolerance,)))
    seed_lists = seeds_by_tile(args.seeds)

    fill_args = (handler, src, seed_lists, tiles_bbox, filler)

    if args.gap_closing_options:
        fill_args += (args.gap_closing_options,)
        filled = gap_closing_fill(*fill_args)
    else:
        filled = scanline_fill(*fill_args)

    # Dilate/Erode (Grow/Shrink)
    if offset != 0 and handler.run:
        filled = lib.morphology.morph(handler, offset, filled)

    # Feather (Gaussian blur)
    if feather != 0 and handler.run:
        filled = lib.morphology.blur(handler, feather, filled)

    # When dilating or blurring the fill, only respect the
    # bounding box limits if they are set by an active frame
    trim_result = args.framed and (offset > 0 or feather != 0)
    if handler.run:
        composite(
            handler, args,
            trim_result, filled, tiles_bbox, dst
        )


def update_bbox(bbox, tx, ty):
    """Update given the min/max, x/y bounding box
    If a coordinate lies outside of the current
    bounds, set the bounds based on that coordinate
    """
    if bbox:
        min_tx, min_ty, max_tx, max_ty = bbox
        if tx < min_tx:
            min_tx = tx
        elif tx > max_tx:
            max_tx = tx
        if ty < min_ty:
            min_ty = ty
        elif ty > max_ty:
            max_ty = ty
        return min_tx, min_ty, max_tx, max_ty
    else:
        return tx, ty, tx, ty


def composite(
        handler, fill_args,
        trim_result, filled, tiles_bbox, dst):
    """Composite the filled tiles into the destination surface"""

    handler.set_stage(handler.COMPOSITE, len(filled))

    fill_col = fill_args.color

    # Prepare opaque color rgba tile for copying
    full_rgba = myplib.rgba_tile_from_alpha_tile(
        _FULL_TILE, *(fill_col + (0, 0, N-1, N-1)))

    # Bounding box of tiles that need updating
    dst_changed_bbox = None
    dst_tiles = dst.get_tiles()

    skip_empty_dst = fill_args.skip_empty_dst()
    mode = fill_args.mode
    lock_alpha = fill_args.lock_alpha
    opacity = fill_args.opacity

    tile_combine = myplib.tile_combine

    # Composite filled tiles into the destination surface
    for tile_coord, src_tile in iteritems(filled):

        if not handler.run:
            break

        handler.inc_processed()

        # Omit tiles outside of the bounding box _if_ the frame is enabled
        # Note:filled tiles outside bbox only originates from dilation/blur
        if trim_result and tiles_bbox.outside(tile_coord):
            continue

        # Skip empty destination tiles for erasing and alpha locking
        # Avoids completely unnecessary tile allocation and copying
        if skip_empty_dst and tile_coord not in dst_tiles:
            continue

        with dst.tile_request(*tile_coord, readonly=False) as dst_tile:

            # Only at this point might the bounding box need to be updated
            dst_changed_bbox = update_bbox(dst_changed_bbox, *tile_coord)

            # Under certain conditions, direct copies and dict manipulation
            # can be used instead of compositing operations.
            cut_off = trim_result and tiles_bbox.crossing(tile_coord)
            full_inner = src_tile is _FULL_TILE and not cut_off
            if full_inner:
                if mode == myplib.CombineNormal and opacity == 1.0:
                    myplib.tile_copy_rgba16_into_rgba16(full_rgba, dst_tile)
                    continue
                elif mode == myplib.CombineDestinationOut and opacity == 1.0:
                    dst_tiles.pop(tile_coord)
                    continue
                elif mode == myplib.CombineDestinationIn and opacity == 1.0:
                    continue
                # Even if opacity != 1.0, we can reuse the full rgba tile
                src_tile_rgba = full_rgba
            else:
                if trim_result:
                    tile_bounds = tiles_bbox.tile_bounds(tile_coord)
                else:
                    tile_bounds = (0, 0, N-1, N-1)
                src_tile_rgba = myplib.rgba_tile_from_alpha_tile(
                    src_tile, *(fill_col + tile_bounds)
                )

            # If alpha locking is enabled in combination with a mode other than
            # CombineNormal, we need to copy the dst tile to mask the result
            if lock_alpha and mode != myplib.CombineSourceAtop:
                mask = np.copy(dst_tile)
                mask_mode = myplib.CombineDestinationAtop
                tile_combine(mode, src_tile_rgba, dst_tile, True, opacity)
                tile_combine(mask_mode, mask, dst_tile, True, 1.0)
            else:
                tile_combine(mode, src_tile_rgba, dst_tile, True, opacity)

    # Handle dst-out and dst-atop: clear untouched tiles
    if mode in [myplib.CombineDestinationIn, myplib.CombineDestinationAtop]:
        for tile_coord in list(dst_tiles.keys()):
            if not handler.run:
                break
            if tile_coord not in filled:
                dst_changed_bbox = update_bbox(
                    dst_changed_bbox, *tile_coord
                )
                with dst.tile_request(*tile_coord, readonly=False):
                    dst_tiles.pop(tile_coord)

    if dst_changed_bbox and handler.run:
        min_tx, min_ty, max_tx, max_ty = dst_changed_bbox
        bbox = (
            min_tx * N, min_ty * N,
            (1 + max_tx - min_tx) * N,
            (1 + max_ty - min_ty) * N,
        )
        # Even for large fills on slow machines, this stage
        # will almost always be too short to even notice.
        # It is not cancellable once entered.
        handler.set_stage(FillHandler.FINISHING)

        # The observers may directly or indirectly use the
        # Gtk API, so the call is scheduled on the gui thread.
        GLib.idle_add(dst.notify_observers, *bbox)


def scanline_fill(handler, src, seed_lists, tiles_bbox, filler):
    """ Perform a scanline fill and return the filled tiles

    Perform a scanline fill using the given starting point and tile,
    with reference to the src surface and given bounding box, using the
    provided filler instance.

    :param handler: updates fill status and permits cancelling
    :type handler: FillHandler
    :param src: Source surface-like object
    :param seed_lists: dictionary, pairing tile coords with lists of seeds
    :type seed_lists: dict
    :param tiles_bbox: Bounding box for the fill
    :type tiles_bbox: lib.fill_common.TileBoundingBox
    :param filler: filler instance performing the per-tile fill operation
    :type filler: myplib.Filler
    :returns: a dictionary of coord->tile mappings for the filled tiles
    """
    # Dict of coord->tile data populated during the fill
    filled = {}

    inv_edges = (
        EDGE.south,
        EDGE.west,
        EDGE.north,
        EDGE.east
    )

    # Starting coordinates + direction of origin (from within)
    tileq = []
    for seed_tile_coord, seeds in iteritems(seed_lists):
        tileq.append((seed_tile_coord, seeds, myplib.edges.none))

    tfs = _TileFillSkipper(tiles_bbox, filler, set({}))

    while len(tileq) > 0 and handler.run:
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
                    handler.inc_processed()
                    filled[tile_coord] = np.zeros((N, N), 'uint16')
                overflows = filler.fill(
                    src_tile, filled[tile_coord], seeds,
                    from_dir, *tiles_bbox.tile_bounds(tile_coord)
                )
            else:
                handler.inc_processed()
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
        self.tiles_bbox = tiles_bbox
        self.filler = filler

    # Dict of alpha->tile, used for uniform non-opaque tile fills
    # NOTE: these are usually not a result of an intentional fill, but
    # clicking a pixel with color very similar to the intended target pixel
    def uniform_tile(self, alpha):
        """ Return a reference to a uniform alpha tile

        If no uniform tile with the given alpha value exists, one is created
        """
        if alpha not in self.uniform_tiles:
            self.uniform_tiles[alpha] = fc.new_full_tile(alpha)
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
        if tile_coord in filled or self.tiles_bbox.crossing(tile_coord):
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
            return [(), (), (), ()]
        elif alpha == _OPAQUE:
            filled[tile_coord] = _FULL_TILE
        else:
            filled[tile_coord] = self.uniform_tile(alpha)
        return self.FULL_OVERFLOWS[from_dir]


def gap_closing_fill(
        handler, src, seed_lists, tiles_bbox, filler, gap_closing_options):
    """ Fill loop that finds and uses gap data to avoid unwanted leaks

    Gaps are defined as distances of fillable pixels enclosed on two sides
    by unfillable pixels. Each tile considered, and their neighbours, are
    flooded with alpha values based on the target color and threshold values.
    The resulting alphas are then searched for gaps, and the size of these gaps
    are marked in separate tiles - one for each tile filled.
    """

    unseep_queue = []
    filled = {}
    final = set({})

    seed_queue = []
    for seed_tile_coord, seeds in iteritems(seed_lists):
        seed_queue.append((seed_tile_coord, seeds))

    options = gap_closing_options
    max_gap_size = lib.helpers.clamp(options.max_gap_size, 1, TILE_SIZE)
    gc_filler = myplib.GapClosingFiller(max_gap_size, options.retract_seeps)
    gc_handler = _GCTileHandler(final, max_gap_size, tiles_bbox, filler, src)
    total_px = 0
    skip_unseeping = False

    while len(seed_queue) > 0 and handler.run:
        tile_coord, seeds = seed_queue.pop(0)
        if tile_coord in final:
            continue
        # Create distance-data and alpha output tiles for the fill
        # and check if the tile can be skipped directly
        alpha_t, dist_t, overflows = gc_handler.get_gc_data(tile_coord, seeds)
        if overflows:
            handler.inc_processed()
            filled[tile_coord] = _FULL_TILE
        else:
            # Complement data for initial seeds (if they are initial seeds)
            seeds, any_not_max = complement_gc_seeds(seeds, dist_t)
            # If the fill is starting at a point with a detected distance,
            # disable seep retraction - otherwise it is very likely
            # that the result will be completely empty.
            if any_not_max:
                skip_unseeping = True
            # Pixel limits within tiles can vary at the bounding box edges
            px_bounds = tiles_bbox.tile_bounds(tile_coord)
            # Create new output tile if not already present
            if tile_coord not in filled:
                handler.inc_processed()
                filled[tile_coord] = np.zeros((N, N), 'uint16')
            # Run the gap-closing fill for the tile
            result = gc_filler.fill(
                alpha_t, dist_t, filled[tile_coord], seeds, *px_bounds
            )
            overflows = result[0:4]
            fill_edges, px_f = result[4:6]
            # The entire tile was filled, despite potential gaps;
            # replace data w. constant and mark tile as final.
            if px_f == N*N:
                final.add(tile_coord)
            # When seep inversion is enabled, track total pixels filled
            # and coordinates where the fill stopped due to distance conditions
            total_px += px_f
            if not skip_unseeping and fill_edges:
                unseep_queue.append((tile_coord, fill_edges, True))
        # Enqueue overflows, whether skipping or not
        enqueue_overflows(seed_queue, tile_coord, overflows, tiles_bbox)

    # If enabled, pull the fill back into the gaps to stop before them
    if not skip_unseeping and handler.run:
        unseep(
            unseep_queue, filled, gc_filler,
            total_px, tiles_bbox, gc_handler.distances
        )
    return filled


class _GCTileHandler(object):
    """Gap-closing-fill Tile Handler

    Manages input alpha tiles and distance tiles necessary to perform
    gap closing fill operations.
    """
    OVERFLOWS = [
        [(), (EDGE.west,), (EDGE.north,), (EDGE.east,)],
        [(EDGE.south,), (), (EDGE.north,), (EDGE.east,)],
        [(EDGE.south,), (EDGE.west,), (), (EDGE.east,)],
        [(EDGE.south,), (EDGE.west,), (EDGE.north,), ()],
        [(EDGE.south,), (EDGE.west,), (EDGE.north,), (EDGE.east,)],
    ]

    def __init__(self, final, max_gap_size, tiles_bbox, filler, src):
        self._src = src
        self.final = final
        self.distances = dict()
        self._alpha_tiles = dict()
        self._dist_data = None
        self._bbox = tiles_bbox
        self._filler = filler
        self._distbucket = myplib.DistanceBucket(max_gap_size)

    def get_gc_data(self, tile_coord, seeds):
        """Get the data necessary to run a gap-closing fill

        For the given tile coordinate, prepare the data necessary to
        run a gap-closing fill operation for that tile, namely the
        corresponding input alpha tile and distance tile.

        The first time a coordinate is reached, also check if it can be
        skipped directly, and return the overflows if that is the case.

        :returns: (alpha_tile, distance_tile, overflows)
        :rtype: tuple
        """
        if tile_coord not in self.distances:
            # Ensure that alpha data exists for the tile and its neighbours
            grid, all_full = self.alpha_grid(tile_coord)
            # The search is skipped if we have a 9-grid of only full tiles
            # since there cannot be any gaps in that case. Otherwise, if no
            # gaps were found during the search, use a constant distance tile
            if all_full or not self.find_gaps(*grid):
                self.distances[tile_coord] = _GAPLESS_TILE
                # Check if fill can be skipped directly
                can_skip_fill = (
                    (all_full or grid[0] is _FULL_TILE) and
                    not self._bbox.crossing(tile_coord) and
                    gc_seeds_skippable(seeds)
                )
                if can_skip_fill:
                    self.final.add(tile_coord)
                    if isinstance(seeds, list):
                        overflows = self.OVERFLOWS[EDGE.none]
                    else:
                        overflows = self.OVERFLOWS[seeds[0]]
                    return _FULL_TILE, _GAPLESS_TILE, overflows
            else:
                self.distances[tile_coord] = self._dist_data
                self._dist_data = None
        # The distance data is already present, meaning the skip checks have
        # already been tried, no skipping possible.
        return self._alpha_tiles[tile_coord], self.distances[tile_coord], ()

    def find_gaps(self, *grid):
        """Search for and mark gaps, given a nine-grid of alpha tiles

        :param grid: nine-grid of alpha tiles
        :return: True if any gaps were found, otherwise false
        :rtype: bool
        """
        if self._dist_data is None:
            self._dist_data = fc.new_full_tile(INF_DIST)
        return myplib.find_gaps(self._distbucket, self._dist_data, *grid)

    def alpha_grid(self, tile_coord):
        """When needed, create and calculate alpha tiles for distance searching.

        For the tile of the given coordinate, ensure that a corresponding tile
        of alpha values (based on the tolerance function) exists in the
        full_alphas dict for both the tile and all of its neighbors

        :returns: Tuple with the grid and a boolean value indicating
        whether every tile in the grid is the constant full alpha tile
        """
        all_full = True
        alpha_tiles = self._alpha_tiles
        grid = []
        for ntc in fc.nine_grid(tile_coord):
            if ntc not in alpha_tiles:
                with self._src.tile_request(
                        ntc[0], ntc[1], readonly=True
                ) as src_tile:
                    is_empty = src_tile is _EMPTY_RGBA
                    alpha = self._filler.tile_uniformity(is_empty, src_tile)
                    if alpha == _OPAQUE:
                        alpha_tiles[ntc] = _FULL_TILE
                    elif alpha == 0:
                        alpha_tiles[ntc] = _EMPTY_TILE
                    elif alpha:
                        alpha_tiles[ntc] = fc.new_full_tile(alpha)
                    else:
                        alpha_tile = np.empty((N, N), 'uint16')
                        self._filler.flood(src_tile, alpha_tile)
                        alpha_tiles[ntc] = alpha_tile
            tile = alpha_tiles[ntc]
            grid.append(tile)
            all_full = all_full and tile is _FULL_TILE
        return grid, all_full


def unseep(seed_queue, filled, gc_filler, total_px, tiles_bbox, distances):
    """Seep inversion is basically a four-way 0-alpha fill
    with different conditions. It only backs off into the original
    fill and therefore does not require creation of new tiles or use
    of an input alpha tile.
    """
    backup = {}
    while len(seed_queue) > 0:
        tile_coord, seeds, is_initial = seed_queue.pop(0)
        if tile_coord not in distances or tile_coord not in filled:
            continue
        if tile_coord not in backup:
            if filled[tile_coord] is _FULL_TILE:
                backup[tile_coord] = _FULL_TILE
                filled[tile_coord] = fc.new_full_tile(1 << 15)
            else:
                backup[tile_coord] = np.copy(filled[tile_coord])
        result = gc_filler.unseep(
            distances[tile_coord], filled[tile_coord], seeds, is_initial
        )
        overflows = result[0:4]
        num_erased_pixels = result[4]
        total_px -= num_erased_pixels
        enqueue_overflows(
            seed_queue, tile_coord, overflows, tiles_bbox, (False,) * 4
        )
    if total_px <= 0:
        # For small areas, when starting on a distance-marked pixel,
        # backing off may remove the entire fill, in which case we
        # roll back the tiles that were processed
        for tile_coord, tile in iteritems(backup):
            filled[tile_coord] = tile


def complement_gc_seeds(seeds, distance_tile):
    """Add distances to initial seeds, check if all seeds lie on detected gaps

    If the input seeds are not initial seeds, they are returned unchanged.

    Returns a tuple with complemented seeds and a boolean indicating whether
    all seeds lie on detected gaps (this check is only done for initial seeds)

    """
    if isinstance(seeds, list) and len(seeds[0]) < 3:
        # Fetch distance for initial seed coord
        complemented_seeds = []
        any_not_max = False
        for (px, py) in seeds:
            distance = distance_tile[py][px]
            if distance < INF_DIST:
                any_not_max = True
            complemented_seeds.append((px, py, distance))
        return complemented_seeds, any_not_max
    else:
        return seeds, False


def gc_seeds_skippable(seeds):
    return (
        isinstance(seeds, tuple) or  # edge constant - a full edge of seeds
        len(seeds[0]) == 2 or  # initial seeds
        any([s[2] == INF_DIST for s in seeds])  # one seed can fill everything
    )
