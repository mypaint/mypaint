# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2018 by the MyPaint Development Team#
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Common interfaces & routines for surface and surface-like objects"""

from __future__ import division, print_function

import abc
import os
import logging

import numpy as np

from . import mypaintlib
import lib.helpers
from lib.errors import FileHandlingError
from lib.gettext import C_
import lib.feedback
from lib.pycompat import xrange


logger = logging.getLogger(__name__)

N = mypaintlib.TILE_SIZE

# throttle excesssive calls to the save/render progress monitor objects
TILES_PER_CALLBACK = 256


class Bounded (object):
    """Interface for objects with an inherent size"""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_bbox(self):
        """Returns the bounding box of the object, in model coords

        :returns: the data bounding box
        :rtype: lib.helpers.Rect

        """


class TileAccessible (Bounded):
    """Interface for objects whose memory is accessible by tile"""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def tile_request(self, tx, ty, readonly):
        """Access by tile, read-only or read/write

        :param int tx: Tile X coord (multiply by TILE_SIZE for pixels)
        :param int ty: Tile Y coord (multiply by TILE_SIZE for pixels)
        :param bool readonly: get a read-only tile

        Implementations must be `@contextlib.contextmanager`s which
        yield one tile array (NxNx16, fix15 data). If called in
        read/write mode, implementations must either put back changed
        data, or alternatively they must allow the underlying data to be
        manipulated directly via the yielded object.

        See lib.tiledsurface.MyPaintSurface.tile_request() for a fuller
        explanation of this interface and its expectations.

        """


class TileBlittable (Bounded):
    """Interface for unconditional copying by tile"""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, *args, **kwargs):
        """Copies one tile from this object into a NumPy array

        :param numpy.ndarray dst: destination array
        :param bool dst_has_alpha: destination has an alpha channel
        :param int tx: Tile X coord (multiply by TILE_SIZE for pixels)
        :param int ty: Tile Y coord (multiply by TILE_SIZE for pixels)
        :param \*args: Implementation may extend this interface
        :param \*\*kwargs: Implementation may extend this interface

        The destination is typically of dimensions NxNx4, and is
        typically of type uint16 or uint8. Implementations are expected
        to check the details, and should raise ValueError if dst doesn't
        have a sensible shape or type.

        This is an unconditional copy of this object's raw visible data,
        ignoring any flags or opacities on the object itself which would
        otherwise control what you see.

        If the source object really consists of multiple compositables
        with special rendering flags, they should be composited normally
        into an empty tile, and that resultant tile blitted.

        """


class TileCompositable (Bounded):
    """Interface for compositing by tile, with modes/opacities/flags"""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       *args, **kwargs):
        """Composites one tile from this object over a NumPy array.

        :param dst: target tile array (uint16, NxNx4, 15-bit scaled int)
        :param dst_has_alpha: alpha channel in dst should be preserved
        :param int tx: Tile X coord (multiply by TILE_SIZE for pixels)
        :param int ty: Tile Y coord (multiply by TILE_SIZE for pixels)
        :param int mode: mode to use when compositing
        :param \*args: Implementation may extend this interface
        :param \*\*kwargs: Implementation may extend this interface

        Composite one tile of this surface over the array dst, modifying
        only dst. Unlike `blit_tile_into()`, this method must respect
        any special rendering settings on the object itself.

        """


def get_tiles_bbox(tile_coords):
    """Convert tile coords to a data bounding box

    :param tile_coords: iterable of (tx, ty) coordinate pairs

    >>> coords = [(0, 0), (-10, 4), (5, -2), (-3, 7)]
    >>> get_tiles_bbox(coords[0:1])
    Rect(0, 0, 64, 64)
    >>> get_tiles_bbox(coords)
    Rect(-640, -128, 1024, 640)
    >>> get_tiles_bbox(coords[1:])
    Rect(-640, -128, 1024, 640)
    >>> get_tiles_bbox(coords[1:-1])
    Rect(-640, -128, 1024, 448)
    """
    bounds = lib.helpers.coordinate_bounds(tile_coords)
    if bounds is None:
        return lib.helpers.Rect()
    else:
        x0, y0, x1, y1 = bounds
        return lib.helpers.Rect(
            N * x0, N * y0, N * (x1 - x0 + 1), N * (y1 - y0 + 1)
        )


def scanline_strips_iter(surface, rect, alpha=False,
                         single_tile_pattern=False, **kwargs):
    """Generate (render) scanline strips from a tile-blittable object

    :param TileBlittable surface: Surface to iterate over
    :param bool alpha: If true, write a PNG with alpha
    :param bool single_tile_pattern: True if surface is a one tile only.
    :param tuple \*\*kwargs: Passed to blit_tile_into.

    The `alpha` parameter is passed to the surface's `blit_tile_into()`.
    Rendering is skipped for all but the first line of single-tile patterns.

    The scanline strips yielded by this generator are suitable for
    feeding to a mypaintlib.ProgressivePNGWriter.

    """
    # Sizes
    x, y, w, h = rect
    assert w > 0
    assert h > 0

    # calculate bounding box in full tiles
    render_tx = x // N
    render_ty = y // N
    render_tw = (x + w - 1) // N - render_tx + 1
    render_th = (y + h - 1) // N - render_ty + 1

    # buffer for rendering one tile row at a time
    arr = np.empty((N, render_tw * N, 4), 'uint8')  # rgba or rgbu
    # view into arr without the horizontal padding
    arr_xcrop = arr[:, x-render_tx*N:x-render_tx*N+w, :]

    first_row = render_ty
    last_row = render_ty+render_th-1

    for ty in range(render_ty, render_ty+render_th):
        skip_rendering = False
        if single_tile_pattern:
            # optimization for simple background patterns
            # e.g. solid color
            if ty != first_row:
                skip_rendering = True

        for tx_rel in xrange(render_tw):
            # render one tile
            dst = arr[:, tx_rel*N:(tx_rel+1)*N, :]
            if not skip_rendering:
                tx = render_tx + tx_rel
                try:
                    surface.blit_tile_into(dst, alpha, tx, ty, **kwargs)
                except Exception:
                    logger.exception("Failed to blit tile %r of %r",
                                     (tx, ty), surface)
                    mypaintlib.tile_clear_rgba8(dst)

        # yield a numpy array of the scanline without padding
        res = arr_xcrop
        if ty == last_row:
            res = res[:y+h-ty*N, :, :]
        if ty == first_row:
            res = res[y-render_ty*N:, :, :]
        yield res


def save_as_png(surface, filename, *rect, **kwargs):
    """Saves a tile-blittable surface to a file in PNG format

    :param TileBlittable surface: Surface to save
    :param unicode filename: The file to write
    :param tuple \*rect: Rectangle (x, y, w, h) to save
    :param bool alpha: If true, write a PNG with alpha
    :param progress: Updates a UI every scanline strip.
    :type progress: lib.feedback.Progress or None
    :param bool single_tile_pattern: True if surface is one tile only.
    :param bool save_srgb_chunks: Set to False to not save sRGB flags.
    :param tuple \*\*kwargs: Passed to blit_tile_into (minus the above)

    The `alpha` parameter is passed to the surface's `blit_tile_into()`
    method, as well as to the PNG writer.  Rendering is
    skipped for all but the first line for single-tile patterns.
    If `*rect` is left unspecified, the surface's own bounding box will
    be used.
    If `save_srgb_chunks` is set to False, sRGB (and associated fallback
    cHRM and gAMA) will not be saved. MyPaint's default behaviour is
    currently to save these chunks.

    Raises `lib.errors.FileHandlingError` with a descriptive string if
    something went wrong.

    """
    # Horrible, dirty argument handling
    alpha = kwargs.pop('alpha', False)
    progress = kwargs.pop('progress', None)
    single_tile_pattern = kwargs.pop("single_tile_pattern", False)
    save_srgb_chunks = kwargs.pop("save_srgb_chunks", True)

    # Sizes. Save at least one tile to allow empty docs to be written
    if not rect:
        rect = surface.get_bbox()
    x, y, w, h = rect
    if w == 0 or h == 0:
        x, y, w, h = (0, 0, 1, 1)
        rect = (x, y, w, h)

    if not progress:
        progress = lib.feedback.Progress()
    num_strips = int((1 + ((y + h) // N)) - (y // N))
    progress.items = num_strips

    try:
        logger.debug(
            "Writing %r (%dx%d) alpha=%r srgb=%r",
            filename,
            w, h,
            alpha,
            save_srgb_chunks,
        )
        with open(filename, "wb") as writer_fp:
            pngsave = mypaintlib.ProgressivePNGWriter(
                writer_fp,
                w, h,
                alpha,
                save_srgb_chunks,
            )
            scanline_strips = scanline_strips_iter(
                surface, rect,
                alpha=alpha,
                single_tile_pattern=single_tile_pattern,
                **kwargs
            )
            for scanline_strip in scanline_strips:
                pngsave.write(scanline_strip)
                if not progress:
                    continue
                try:
                    progress += 1
                except Exception:
                    logger.exception(
                        "Failed to update lib.feedback.Progress: "
                        "dropping it"
                    )
                    progress = None
            pngsave.close()
        logger.debug("Finished writing %r", filename)
        if progress:
            progress.close()
    except (IOError, OSError, RuntimeError) as err:
        logger.exception(
            "Caught %r from C++ png-writer code, re-raising as a "
            "FileHandlingError",
            err,
        )
        raise FileHandlingError(C_(
            "low-level PNG writer failure report (dialog)",
            u"Failed to write “{basename}”.\n\n"
            u"Reason: {err}\n"
            u"Target folder: “{dirname}”."
        ).format(
            err = err,
            basename = os.path.basename(filename),
            dirname = os.path.dirname(filename),
        ))
        # Other possible exceptions include TypeError, ValueError, but
        # those indicate incorrect coding usually; just raise them
        # normally.


if __name__ == "__main__":
    import doctest
    doctest.testmod()
