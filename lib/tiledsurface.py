# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This module implements an unbounded tiled surface for painting.

import numpy
from numpy import *
import time
import sys
import os
import contextlib
import functools
import logging
logger = logging.getLogger(__name__)

import mypaintlib
import helpers
import math

TILE_SIZE = N = mypaintlib.TILE_SIZE
MAX_MIPMAP_LEVEL = 4

use_gegl = True if os.environ.get('MYPAINT_ENABLE_GEGL', 0) else False

from layer import DEFAULT_COMPOSITE_OP

import pixbufsurface


class Tile (object):
    def __init__(self, copy_from=None):
        object.__init__(self)
        # note: pixels are stored with premultiplied alpha
        #       15bits are used, but fully opaque or white is stored as 2**15 (requiring 16 bits)
        #       This is to allow many calcuations to divide by 2**15 instead of (2**16-1)
        if copy_from is None:
            self.rgba = zeros((N, N, 4), 'uint16')
        else:
            self.rgba = copy_from.rgba.copy()
        self.readonly = False

    def copy(self):
        return Tile(copy_from=self)



svg2mypaintlibmode = {
    'svg:src-over': mypaintlib.BlendingModeNormal,
    'svg:multiply': mypaintlib.BlendingModeMultiply,
    'svg:screen': mypaintlib.BlendingModeScreen,
    'svg:overlay': mypaintlib.BlendingModeOverlay,
    'svg:darken': mypaintlib.BlendingModeDarken,
    'svg:lighten': mypaintlib.BlendingModeLighten,
    'svg:hard-light': mypaintlib.BlendingModeHardLight,
    'svg:soft-light': mypaintlib.BlendingModeSoftLight,
    'svg:color-burn': mypaintlib.BlendingModeColorBurn,
    'svg:color-dodge': mypaintlib.BlendingModeColorDodge,
    'svg:difference': mypaintlib.BlendingModeDifference,
    'svg:exclusion': mypaintlib.BlendingModeExclusion,
    'svg:hue': mypaintlib.BlendingModeHue,
    'svg:saturation': mypaintlib.BlendingModeSaturation,
    'svg:color': mypaintlib.BlendingModeColor,
    'svg:luminosity': mypaintlib.BlendingModeLuminosity,
    }

svg2composite_func = {}
for svg_id, mode_enum in svg2mypaintlibmode.iteritems():
    svg2composite_func[svg_id] = functools.partial(mypaintlib.tile_composite, mode_enum)

# tile for read-only operations on empty spots
transparent_tile = Tile()
transparent_tile.readonly = True

# tile with invalid pixel memory (needs refresh)
mipmap_dirty_tile = Tile()
del mipmap_dirty_tile.rgba

def get_tiles_bbox(tiles):
    res = helpers.Rect()
    for tx, ty in tiles:
        res.expandToIncludeRect(helpers.Rect(N*tx, N*ty, N, N))
    return res

class SurfaceSnapshot (object):
    pass

if use_gegl:

    class GeglSurface():

        def __init__(self, mipmap_level=0, looped=False, looped_size=(0,0)):
            self.observers = []
            self._backend = mypaintlib.GeglBackedSurface(self)

            # Forwarding API
            self.begin_atomic = self._backend.begin_atomic
            self.end_atomic = self._backend.end_atomic
            self.get_color = self._backend.get_color
            self.get_alpha = self._backend.get_alpha
            self.draw_dab = self._backend.draw_dab
#            self.set_symmetry_state = self._backend.set_symmetry_state
            self.get_node = self._backend.get_node

        @property
        def backend(self):
            return self._backend

        def notify_observers(self, *args):
            for f in self.observers:
                f(*args)

        def get_bbox(self):
            rect = helpers.Rect(*self.get_bbox_c())
            return rect

        def clear(self):
            pass

        def save_as_png(self, path, *args, **kwargs):
            return self.save_as_png_c(str(path))

        def load_from_png(self, path, x, y, *args, **kwargs):
            return self.load_from_png_c(str(path))

        def save_snapshot(self):
            sshot = SurfaceSnapshot()
            sshot.tiledict = {}
            return sshot

        def load_snapshot(self, sshot):
            pass

        def is_empty(self):
            return False

        def remove_empty_tiles(self):
            pass

        def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0, opacity=1.0,
                           mode=DEFAULT_COMPOSITE_OP):
            pass

        def load_from_numpy(self, arr, x, y):
            return (0, 0, 0, 0)

        def load_from_surface(self, other):
            pass

        def get_tiles(self):
            return {}

        def set_symmetry_state(self, enabled, center_axis):
            pass

# TODO:
# - move the tile storage from MyPaintSurface to a separate class
class MyPaintSurface (object):
    # the C++ half of this class is in tiledsurface.hpp
    def __init__(self, mipmap_level=0, mipmap_surfaces=None,
                 looped=False, looped_size=(0,0)):
        object.__init__(self)

        # TODO: pass just what it needs access to, not all of self
        self._backend = mypaintlib.TiledSurface(self)
        self.tiledict = {}
        self.observers = []

        # Used to implement repeating surfaces, like Background
        if looped_size[0] % N or looped_size[1] % N:
            raise ValueError, 'Looped size must be multiples of tile size'
        self.looped = looped
        self.looped_size = looped_size

        self.mipmap_level = mipmap_level
        self.mipmaps = mipmap_surfaces

        if mipmap_level == 0:
            mipmaps = [self]
            for level in range(1, MAX_MIPMAP_LEVEL+1):
                s = MyPaintSurface(mipmap_level=level, mipmap_surfaces=mipmaps)
                mipmaps.append(s)
            self.mipmaps = mipmaps

            # for quick lookup
            for level, s in enumerate(mipmaps):
                try:
                    s.parent = mipmaps[level-1]
                except IndexError:
                    s.parent = None
                try:
                    s.mipmap = mipmaps[level+1]
                except IndexError:
                    s.mipmap = None

        # Forwarding API
        self.set_symmetry_state = self._backend.set_symmetry_state
        self.begin_atomic = self._backend.begin_atomic
        self.end_atomic = self._backend.end_atomic
        self.get_color = self._backend.get_color
        self.get_alpha = self._backend.get_alpha
        self.draw_dab = self._backend.draw_dab


    @property
    def backend(self):
        return self._backend

    def notify_observers(self, *args):
        for f in self.observers:
            f(*args)

    def clear(self):
        tiles = self.tiledict.keys()
        self.tiledict = {}
        self.notify_observers(*get_tiles_bbox(tiles))
        if self.mipmap: self.mipmap.clear()


    def trim(self, rect):
        """Trim the layer to a rectangle, discarding data outside it

        :param rect: A trimming rectangle in model coordinates
        :type rect: tuple (x, y, w, h)

        Only complete tiles are discarded by this method.
        """
        x, y, w, h = rect
        logger.info("Trim %dx%d%+d%+d", w, h, x, y)
        trimmed = []
        for tx, ty in list(self.tiledict.keys()):
            if tx*N+N < x or ty*N+N < y or tx*N > x+w or ty*N > y+h:
                trimmed.append((tx, ty))
                self.tiledict.pop((tx, ty))
                self._mark_mipmap_dirty(tx, ty)
        self.notify_observers(*get_tiles_bbox(trimmed))


    @contextlib.contextmanager
    def tile_request(self, tx, ty, readonly):
        """Context manager that fetches a tile as a NumPy array,
        and then puts the potentially modified tile back into the
        tile backing store. To be used with the 'with' statement."""

        numpy_tile = self._get_tile_numpy(tx, ty, readonly)
        yield numpy_tile
        self._set_tile_numpy(tx, ty, numpy_tile, readonly)

    def _regenerate_mipmap(self, t, tx, ty):
        t = Tile()
        self.tiledict[(tx, ty)] = t
        empty = True

        for x in xrange(2):
            for y in xrange(2):
                src = self.parent.tiledict.get((tx*2 + x, ty*2 + y), transparent_tile)
                if src is mipmap_dirty_tile:
                    src = self.parent._regenerate_mipmap(src, tx*2 + x, ty*2 + y)
                mypaintlib.tile_downscale_rgba16(src.rgba, t.rgba, x*N/2, y*N/2)
                if src.rgba is not transparent_tile.rgba:
                    empty = False
        if empty:
            # rare case, no need to speed it up
            del self.tiledict[(tx, ty)]
            t = transparent_tile
        return t

    def _get_tile_numpy(self, tx, ty, readonly):
        # OPTIMIZE: do some profiling to check if this function is a bottleneck
        #           yes it is
        # Note: we must return memory that stays valid for writing until the
        # last end_atomic(), because of the caching in tiledsurface.hpp.

        if self.looped:
            tx = tx % (self.looped_size[0] / N)
            ty = ty % (self.looped_size[1] / N)

        t = self.tiledict.get((tx, ty))
        if t is None:
            if readonly:
                t = transparent_tile
            else:
                t = Tile()
                self.tiledict[(tx, ty)] = t
        if t is mipmap_dirty_tile:
            t = self._regenerate_mipmap(t, tx, ty)
        if t.readonly and not readonly:
            # shared memory, get a private copy for writing
            t = t.copy()
            self.tiledict[(tx, ty)] = t
        if not readonly:
            # assert self.mipmap_level == 0
            self._mark_mipmap_dirty(tx, ty)
        return t.rgba

    def _set_tile_numpy(self, tx, ty, obj, readonly):
        pass # Data can be modified directly, no action needed

    def _mark_mipmap_dirty(self, tx, ty):
        #assert self.mipmap_level == 0
        for level, mipmap in enumerate(self.mipmaps):
            if level == 0:
                continue
            fac = 2**(level)
            if mipmap.tiledict.get((tx/fac, ty/fac), None) == mipmap_dirty_tile:
                break
            mipmap.tiledict[(tx/fac, ty/fac)] = mipmap_dirty_tile

    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, mipmap_level=0):
        # used mainly for saving (transparent PNG)

        #assert dst_has_alpha is True

        if self.mipmap_level < mipmap_level:
            return self.mipmap.blit_tile_into(dst, dst_has_alpha, tx, ty, mipmap_level)

        assert dst.shape[2] == 4

        with self.tile_request(tx, ty, readonly=True) as src:

            if src is transparent_tile.rgba:
                #dst[:] = 0 # <-- notably slower than memset()
                mypaintlib.tile_clear(dst)
            else:

                if dst.dtype == 'uint16':
                    # this will do memcpy, not worth to bother skipping the u channel
                    mypaintlib.tile_copy_rgba16_into_rgba16(src, dst)
                elif dst.dtype == 'uint8':
                    if dst_has_alpha:
                        mypaintlib.tile_convert_rgba16_to_rgba8(src, dst)
                    else:
                        mypaintlib.tile_convert_rgbu16_to_rgbu8(src, dst)
                else:
                    raise ValueError, 'Unsupported destination buffer type'

    def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0, opacity=1.0,
                       mode=DEFAULT_COMPOSITE_OP):
        """Composite one tile of this surface over a NumPy array.

        Composite one tile of this surface over the array dst, modifying only dst.
        """
        if self.mipmap_level < mipmap_level:
            return self.mipmap.composite_tile(dst, dst_has_alpha, tx, ty, mipmap_level, opacity, mode)
        if not (tx,ty) in self.tiledict:
            return

        with self.tile_request(tx, ty, readonly=True) as src:
            func = svg2composite_func[mode]
            func(src, dst, dst_has_alpha, opacity)


    ## Snapshotting

    def save_snapshot(self):
        """Creates and returns a snapshot of the surface"""
        sshot = SurfaceSnapshot()
        for t in self.tiledict.itervalues():
            t.readonly = True
        sshot.tiledict = self.tiledict.copy()
        return sshot


    def load_snapshot(self, sshot):
        """Loads a saved snapshot, replacing the internal tiledict"""
        self._load_tiledict(sshot.tiledict)


    def _load_tiledict(self, d):
        """Efficiently loads a tiledict, and notifies the observers"""
        if d == self.tiledict:
            # common case optimization, called from split_stroke() via stroke.redo()
            # testcase: comparison above (if equal) takes 0.6ms, code below 30ms
            return
        old = set(self.tiledict.iteritems())
        self.tiledict = d.copy()
        new = set(self.tiledict.iteritems())
        dirty = old.symmetric_difference(new)
        for pos, tile in dirty:
            self._mark_mipmap_dirty(*pos)
        bbox = get_tiles_bbox([pos for (pos, tile) in dirty])
        if not bbox.empty():
            self.notify_observers(*bbox)


    ## Loading tile data


    def load_from_surface(self, other):
        """Loads tile data from another surface, via a snapshot"""
        self.load_snapshot(other.save_snapshot())


    def _load_from_pixbufsurface(self, s):
        dirty_tiles = set(self.tiledict.keys())
        self.tiledict = {}

        for tx, ty in s.get_tiles():
            with self.tile_request(tx, ty, readonly=False) as dst:
                s.blit_tile_into(dst, True, tx, ty)

        dirty_tiles.update(self.tiledict.keys())
        bbox = get_tiles_bbox(dirty_tiles)
        self.notify_observers(*bbox)


    def load_from_numpy(self, arr, x, y):
        """Loads tile data from a numpy array

        :param arr: Array containing the pixel data
        :type arr: numpy.ndarray of uint8, dimensions HxWx3 or HxWx4
        :param x: X coordinate for the array
        :param y: Y coordinate for the array
        :returns: the dimensions of the loaded surface, as (x,y,w,h)

        """
        h, w, channels = arr.shape
        if h <= 0 or w <= 0:
            return (x, y, w, h)

        if arr.dtype == 'uint8':
            s = pixbufsurface.Surface(x, y, w, h, data=arr)
            self._load_from_pixbufsurface(s)
        else:
            raise ValueError, "Only uint8 data is supported by MyPaintSurface"

        return (x, y, w, h)

    def load_from_png(self, filename, x, y, feedback_cb=None):
        """Load from a PNG, one tilerow at a time, discarding empty tiles.
        """
        dirty_tiles = set(self.tiledict.keys())
        self.tiledict = {}

        state = {}
        state['buf'] = None # array of height N, width depends on image
        state['ty'] = y/N # current tile row being filled into buf
        state['frame_size'] = None

        def get_buffer(png_w, png_h):
            state['frame_size'] = x, y, png_w, png_h
            if feedback_cb:
                feedback_cb()
            buf_x0 = x/N*N
            buf_x1 = ((x+png_w-1)/N+1)*N
            buf_y0 = state['ty']*N
            buf_y1 = buf_y0+N
            buf_w = buf_x1-buf_x0
            buf_h = buf_y1-buf_y0
            assert buf_w % N == 0
            assert buf_h == N
            if state['buf'] is not None:
                consume_buf()
            else:
                state['buf'] = empty((buf_h, buf_w, 4), 'uint8')

            png_x0 = x
            png_x1 = x+png_w
            subbuf = state['buf'][:,png_x0-buf_x0:png_x1-buf_x0]
            if 1: # optimize: only needed for first and last
                state['buf'].fill(0)
                png_y0 = max(buf_y0, y)
                png_y1 = min(buf_y0+buf_h, y+png_h)
                assert png_y1 > png_y0
                subbuf = subbuf[png_y0-buf_y0:png_y1-buf_y0,:]

            state['ty'] += 1
            return subbuf

        def consume_buf():
            ty = state['ty']-1
            for i in xrange(state['buf'].shape[1]/N):
                tx = x/N + i
                src = state['buf'][:,i*N:(i+1)*N,:]
                if src[:,:,3].any():
                    with self.tile_request(tx, ty, readonly=False) as dst:
                        mypaintlib.tile_convert_rgba8_to_rgba16(src, dst)

        filename_sys = filename.encode(sys.getfilesystemencoding()) # FIXME: should not do that, should use open(unicode_object)
        flags = mypaintlib.load_png_fast_progressive(filename_sys, get_buffer)
        consume_buf() # also process the final chunk of data
        logger.debug("PNG loader flags: %r", flags)

        dirty_tiles.update(self.tiledict.keys())
        bbox = get_tiles_bbox(dirty_tiles)
        self.notify_observers(*bbox)

        # return the bbox of the loaded image
        return state['frame_size']

    def render_as_pixbuf(self, *args, **kwargs):
        if not self.tiledict:
            logger.warning('empty surface')
        t0 = time.time()
        kwargs['alpha'] = True
        res = pixbufsurface.render_as_pixbuf(self, *args, **kwargs)
        logger.debug('%.3fs rendering layer as pixbuf', time.time() - t0)
        return res

    def save_as_png(self, filename, *args, **kwargs):
        if not 'alpha' in kwargs:
            kwargs['alpha'] = True

        if len(self.tiledict) == 1:
            kwargs['single_tile_pattern'] = True
        pixbufsurface.save_as_png(self, filename, *args, **kwargs)

    def get_tiles(self):
        return self.tiledict

    def get_bbox(self):
        return get_tiles_bbox(self.tiledict)

    def is_empty(self):
        return not self.tiledict

    def remove_empty_tiles(self):
        """Removes tiles from the tiledict which contain no data"""
        for pos, data in self.tiledict.items():
            if not data.rgba.any():
                self.tiledict.pop(pos)

    def get_move(self, x, y, sort=True):
        """Returns a move object for this surface

        :param x: Start position for the move, X coord
        :param y: Start position for the move, X coord
        :param sort: If true, sort tiles to move by distance from (x,y)
        :rtype: TiledSurfaceMove

        It's up to the caller to ensure that only one move is active at a
        any single instant in time.
        """
        return TiledSurfaceMove(self, x, y, sort=sort)


    def flood_fill(self, x, y, color, bbox, tolerance, dst_surface):
        """Fills connected areas of this surface

        :param x: Starting point X coordinate
        :param y: Starting point Y coordinate
        :param color: an RGB color
        :type color: tuple
        :param bbox: Bounding box: limits the fill
        :type bbox: lib.helpers.Rect or equivalent 4-tuple
        :param tolerance: how much filled pixels are permitted to vary
        :type tolerance: float [0.0, 1.0]
        :param dst_surface: Target surface
        :type dst_surface: MyPaintSurface

        See also `lib.layer.Layer.flood_fill()`.
        """
        # Colour to fill with
        fill_r, fill_g, fill_b = color

        # Limits
        tolerance = helpers.clamp(tolerance, 0.0, 1.0)

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

        # Tile and pixel addressing for the seed point
        tx, ty = int(x//N), int(y//N)
        px, py = int(x%N), int(y%N)

        # Sample the pixel colour there to obtain the target colour
        with self.tile_request(tx, ty, readonly=True) as start:
            targ_r, targ_g, targ_b, targ_a = [int(c) for c in start[py][px]]
        if targ_a == 0:
            targ_r = 0
            targ_g = 0
            targ_b = 0
            targ_a = 0

        # Flood-fill loop
        filled = {}
        tileq = [ ((tx, ty), [(px, py)]) ]
        while len(tileq) > 0:
            (tx, ty), seeds = tileq.pop(0)
            # Bbox-derived limits
            if tx > max_tx or ty > max_ty:
                continue
            if tx < min_tx or ty < min_ty:
                continue
            # Pixel limits within this tile...
            min_x = 0
            min_y = 0
            max_x = N-1
            max_y = N-1
            # ... vary at the edges
            if tx == min_tx:
                min_x = min_px
            if ty == min_ty:
                min_y = min_py
            if tx == max_tx:
                max_x = max_px
            if ty == max_ty:
                max_y = max_py
            # Flood-fill one tile
            one = 1<<15
            col = (int(fill_r*one), int(fill_g*one), int(fill_b*one), one)
            with self.tile_request(tx, ty, readonly=True) as src:
                dst = filled.get((tx, ty), None)
                if dst is None:
                    dst = zeros((N, N, 4), 'uint16')
                    filled[(tx, ty)] = dst
                overflows = mypaintlib.tile_flood_fill(src, dst, seeds,
                               targ_r, targ_g, targ_b, targ_a,
                               fill_r, fill_g, fill_b,
                               min_x, min_y, max_x, max_y,
                               tolerance)
                seeds_n, seeds_e, seeds_s, seeds_w = overflows
            # Enqueue overflows in each cardinal direction
            if seeds_n and ty > min_ty:
                tpos = (tx, ty-1)
                tileq.append((tpos, seeds_n))
            if seeds_w and tx > min_tx:
                tpos = (tx-1, ty)
                tileq.append((tpos, seeds_w))
            if seeds_s and ty < max_ty:
                tpos = (tx, ty+1)
                tileq.append((tpos, seeds_s))
            if seeds_e and tx < max_tx:
                tpos = (tx+1, ty)
                tileq.append((tpos, seeds_e))

        # Composite filled tiles into the destination surface
        comp = functools.partial(mypaintlib.tile_composite,
                                 mypaintlib.BlendingModeNormal)
        for (tx, ty), src in filled.iteritems():
            with dst_surface.tile_request(tx, ty, readonly=False) as dst:
                comp(src, dst, True, 1.0)
            dst_surface._mark_mipmap_dirty(tx, ty)
        bbox = get_tiles_bbox(filled)
        dst_surface.notify_observers(*bbox)


class TiledSurfaceMove (object):
    """Ongoing move state for a tiled surface, processed in chunks

    Tile move processing involves slicing and copying data from a snapshot of
    the surface's original tile arrays into an active surface within the model
    document. It's therefore potentially very slow for huge layers: doing this
    interactively requires the move to be processed in chunks in idle routines.

    Moves are created by a surface's get_move() method starting at a particular
    point in model coordinates. During an interactive move, they are then
    updated in response to the user moving the pointer, and get processed in
    chunks of a few hundred tiles in an idle routine. They can also be
    processed non-interactively by calling all the different phases together.

    """

    def __init__(self, surface, x, y, sort=True):
        """Starts the move, recording state in the Move object

        :param x: Where to start, model X coordinate
        :param y: Where to start, model Y coordinate
        :param sort: If true, sort tiles to move by distance from (x,y)

        Sorting tiles by distance makes the move look nicer when moving
        interactively, but it's pointless for non-interactive moves.

        """
        object.__init__(self)
        self.surface = surface
        self.snapshot = surface.save_snapshot()
        self.chunks = self.snapshot.tiledict.keys()
        self.sort = sort
        tx = x // N
        ty = y // N
        self.start_pos = (x, y)
        if self.sort:
            manhattan_dist = lambda p: abs(tx - p[0]) + abs(ty - p[1])
            self.chunks.sort(key=manhattan_dist)
        self.chunks_i = 0
        # Tile state tracking for individual update cycles
        self.written = set()
        self.blank_queue = []


    def update(self, dx, dy):
        """Updates the offset during a move

        :param dx: New move offset: relative to the constructor x.
        :param dy: New move offset: relative to the constructor y.

        This causes all the move's work to be re-queued.
        """
        # Nothing has been written in this pass yet
        self.written = set()
        # Tile indices to be cleared during processing, unless written
        self.blank_queue = self.surface.tiledict.keys()
        if self.sort:
            x, y = self.start_pos
            tx = (x + dx) // N
            ty = (y + dy) // N
            manhattan_dist = lambda p: abs(tx - p[0]) + abs(ty - p[1])
            self.blank_queue.sort(key=manhattan_dist)
        # Calculate offsets
        self.slices_x = calc_translation_slices(int(dx))
        self.slices_y = calc_translation_slices(int(dy))
        # Need to process every source chunk
        self.chunks_i = 0


    def cleanup(self):
        """Cleans up after processing all tile moves in one update cycle

        This should be called after `process()` indicates that all tiles have
        been sliced and moved. It should always happen at the end of a move,
        but for an iteractive move, this can happen in the middle of
        proceedings too if the user pauses their drag for long enough. The move
        can still be updated and processed after this method has been called.
        """
        # Process any remaining work. Caller should have done this already.
        if self.chunks_i < len(self.chunks) or len(self.blank_queue) > 0:
            logger.warning("Stuff left to do at end of move cleanup(). May "
                           "result in poor interactive appearance. "
                           "chunks=%d/%d, blanks=%d", self.chunks_i,
                           len(self.chunks), len(self.blank_queue))
            logger.warning("Doing cleanup now...")
            self.process(n=-1)
        assert self.chunks_i >= len(self.chunks)
        assert len(self.blank_queue) == 0
        # Remove empty tiles created by Layer Move
        self.surface.remove_empty_tiles()


    def process(self, n=200):
        """Process a number of pending tile moves

        :param n: The number of source tiles to process in this call.
                  Specify a negative `n` to process all remaining tiles.
        :returns: whether there are any more tiles to process
        :type: bool

        """
        updated = set()
        moves_remaining = self._process_moves(n, updated)
        blanks_remaining = self._process_blanks(n, updated)
        for pos in updated:
            self.surface._mark_mipmap_dirty(*pos)
        bbox = get_tiles_bbox(updated)
        self.surface.notify_observers(*bbox)
        return blanks_remaining or moves_remaining


    def _process_moves(self, n, updated):
        """Process the tile movement queue"""
        if self.chunks_i > len(self.chunks):
            return False
        if n <= 0:
            n = len(self.chunks)  # process all remaining
        is_integral = len(self.slices_x) == 1 and len(self.slices_y) == 1
        for src_t in self.chunks[self.chunks_i : self.chunks_i + n]:
            src_tx, src_ty = src_t
            src_tile = self.snapshot.tiledict[src_t]
            for (src_x0, src_x1), (targ_tdx, targ_x0, targ_x1) in self.slices_x:
                for (src_y0, src_y1), (targ_tdy, targ_y0, targ_y1) in self.slices_y:
                    targ_tx = src_tx + targ_tdx
                    targ_ty = src_ty + targ_tdy
                    targ_t = targ_tx, targ_ty
                    if is_integral:
                        # We're lucky. Perform a straight data copy.
                        self.surface.tiledict[targ_t] = src_tile.copy()
                        continue
                    # Get a tile to write
                    targ_tile = None
                    if targ_t in self.written:
                        # Reuse a blank tile made earlier in this update cycle
                        targ_tile = self.surface.tiledict.get(targ_t, None)
                    if targ_tile is None:
                        # Create and store a new blank tile to avoid corruption
                        targ_tile = Tile()
                        self.surface.tiledict[targ_t] = targ_tile
                        self.written.add(targ_t)
                    # Copy this source slice to the desination
                    targ_tile.rgba[targ_y0:targ_y1, targ_x0:targ_x1] \
                                = src_tile.rgba[src_y0:src_y1, src_x0:src_x1]
                    updated.add(targ_t)
            # The source tile has been fully processed at this point, and can be blanked
            # if it's safe to do so
            if src_t in self.surface.tiledict:
                if src_t not in self.written:
                    self.surface.tiledict.pop(src_t, None)
                    updated.add(src_t)
        # Move on, and return whether we're complete
        self.chunks_i += n
        return self.chunks_i < len(self.chunks)


    def _process_blanks(self, n, updated):
        """Internal: process blanking-out queue"""
        if n <= 0:
            n = len(self.blank_queue)
        while len(self.blank_queue) > 0 and n > 0:
            t = self.blank_queue.pop(0)
            if t not in self.written:
                self.surface.tiledict.pop(t, None)
                updated.add(t)
                n -= 1
        return len(self.blank_queue) > 0


def calc_translation_slices(dc):
    """Returns a list of offsets and slice extents for a translation of `dc`.

    The returned slice list's members are of the form

        ((src_c0, src_c1), (targ_tdc, targ_c0, targ_c1))

    where ``src_c0`` and ``src_c1`` determine the extents of the source slice
    within a tile, their ``targ_`` equivalents specify where to put that slice
    in the target tile, and ``targ_tdc`` is the tile offset.
    """
    dcr = dc % N
    tdc = (dc // N)
    if dcr == 0:
        return [ ((0, N), (tdc, 0, N)) ]
    else:
        return [ ((0, N-dcr), (tdc, dcr, N)) ,
                 ((N-dcr, N), (tdc+1, 0, dcr)) ]


# Set which surface backend to use
Surface = GeglSurface if use_gegl else MyPaintSurface


def new_surface():
    """Creates a new Surface object. Used by mypaintlib internals."""
    return Surface()


class BackgroundError(Exception):
    """Errors raised by Background during failed initiailizations"""
    pass


class Background (Surface):
    """A background layer surface, with a repeating image"""

    def __init__(self, obj, mipmap_level=0):
        """Construct from a color or from a NumPy array

        :param obj: RGB triple (uint8), or a HxWx4 or HxWx3 numpy array which
           can be either uint8 or uint16.
        :param mipmap_level: mipmap level, used internally. Root is zero.
        """

        if not isinstance(obj, numpy.ndarray):
            r, g, b = obj
            obj = numpy.zeros((N, N, 3), dtype='uint8')
            obj[:,:,:] = r, g, b

        height, width = obj.shape[0:2]
        if height % N or width % N:
            raise BackgroundError, 'unsupported background tile size: %dx%d' % (width, height)

        super(Background, self).__init__(mipmap_level=0, looped=True,
                                         looped_size=(width, height))
        self.load_from_numpy(obj, 0, 0)

        # Generate mipmap
        if mipmap_level <= MAX_MIPMAP_LEVEL:
            mipmap_obj = numpy.zeros((height, width, 4), dtype='uint16')
            for ty in range(height/N*2):
                for tx in range(width/N*2):
                    with self.tile_request(tx, ty, readonly=True) as src:
                        mypaintlib.tile_downscale_rgba16(src, mipmap_obj, tx*N/2, ty*N/2)

            self.mipmap = Background(mipmap_obj, mipmap_level+1)
            self.mipmap.parent = self
            self.mipmap_level = mipmap_level


    def load_from_numpy(self, arr, x, y):
        """Loads tile data from a numpy array

        This extends the base class's implementation with additional support
        for tile-aligned uint16 data.

        """
        h, w, channels = arr.shape
        if h <= 0 or w <= 0:
            return (x, y, w, h)
        if arr.dtype == 'uint16':
            assert w % N == 0 and h % N == 0
            assert x == 0 and y == 0
            for ty in range(h/N):
                for tx in range(w/N):
                    with self.tile_request(tx, ty, readonly=False) as dst:
                        dst[:,:,:] = arr[ty*N:(ty+1)*N, tx*N:(tx+1)*N, :]
            return (x, y, w, h)
        else:
            return super(Background, self).load_from_numpy(arr, x, y)
