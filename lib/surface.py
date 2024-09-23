# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2018 by the MyPaint Development Team#
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Common interfaces & routines for surface and surface-like objects"""

import abc
import os
import logging

import numpy as np

from . import mypaintlib
import lib.helpers
from lib.errors import FileHandlingError
from lib.gettext import C_
import lib.feedback


logger = logging.getLogger(__name__)

N = mypaintlib.TILE_SIZE

# throttle excesssive calls to the save/render progress monitor objects
TILES_PER_CALLBACK = 256


class Bounded(metaclass=abc.ABCMeta):
    """Interface for objects with an inherent size"""

    @abc.abstractmethod
    def get_bbox(self):
        """Returns the bounding box of the object, in model coords

        Args:

        Returns:
            lib.helpers.Rect: the data bounding box

        Raises:

        """


class TileAccessible(Bounded, metaclass=abc.ABCMeta):
    """Interface for objects whose memory is accessible by tile"""

    @abc.abstractmethod
    def tile_request(self, tx: int, ty: int, readonly: bool) -> Types.NONE:
        """Access by tile, read-only or read/write

        Args:
            tx: Tile X coord (multiply by TILE_SIZE for pixels)
            ty: Tile Y coord (multiply by TILE_SIZE for pixels)
            readonly: get a read-only tile
        
        Implementations must be `@contextlib.contextmanager`s which
        yield one tile array (NxNx16, fix15 data). If called in
        read/write mode, implementations must either put back changed
        data, or alternatively they must allow the underlying data to be
        manipulated directly via the yielded object.
        
        See lib.tiledsurface.MyPaintSurface.tile_request() for a fuller
        explanation of this interface and its expectations.

        Returns:

        Raises:

        """


class TileBlittable(Bounded, metaclass=abc.ABCMeta):
    """Interface for unconditional copying by tile"""

    @abc.abstractmethod
    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, *args, **kwargs):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Copies one tile from this object into a NumPy array

        Args:
            dst: destination array
            dst_has_alpha: destination has an alpha channel
            tx: Tile X coord (multiply by TILE_SIZE for pixels)
            ty: Tile Y coord (multiply by TILE_SIZE for pixels)
            *args: 
            **kwargs: 

        Returns:

        Raises:

        """


class TileCompositable(Bounded, metaclass=abc.ABCMeta):
    """Interface for compositing by tile, with modes/opacities/flags"""

    @abc.abstractmethod
    def composite_tile(
        self, dst, dst_has_alpha, tx, ty, mipmap_level=0, *args, **kwargs
    ):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Composites one tile from this object over a NumPy array.

        Args:
            dst: target tile array (uint16, NxNx4, 15-bit scaled int)
            dst_has_alpha: alpha channel in dst should be preserved
            tx: Tile X coord (multiply by TILE_SIZE for pixels)
            ty: Tile Y coord (multiply by TILE_SIZE for pixels)
            mipmap_level:  (Default value = 0)
            *args: 
            **kwargs: 

        Returns:

        Raises:

        """


def get_tiles_bbox(tile_coords: Types.ELLIPSIS) -> Types.NONE:
    """Convert tile coords to a data bounding box

    Args:
        tile_coords: iterable of (tx, ty) coordinate pairs

    Returns:

    Raises:

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
        return lib.helpers.Rect(N * x0, N * y0, N * (x1 - x0 + 1), N * (y1 - y0 + 1))


def scanline_strips_iter(
    surface, rect, alpha=False, single_tile_pattern=False, **kwargs
):
    """Generate (render) scanline strips from a tile-blittable object

    Args:
        surface (TileBlittable): Surface to iterate over
        rect: 
        alpha (bool, optional): If true, write a PNG with alpha (Default value = False)
        single_tile_pattern (bool, optional): True if surface is a one tile only. (Default value = False)
        **kwargs: 

    Returns:

    Raises:

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
    arr = np.empty((N, render_tw * N, 4), "uint8")  # rgba or rgbu
    # view into arr without the horizontal padding
    arr_xcrop = arr[:, x - render_tx * N : x - render_tx * N + w, :]

    first_row = render_ty
    last_row = render_ty + render_th - 1

    for ty in range(render_ty, render_ty + render_th):
        skip_rendering = False
        if single_tile_pattern:
            # optimization for simple background patterns
            # e.g. solid color
            if ty != first_row:
                skip_rendering = True

        for tx_rel in range(render_tw):
            # render one tile
            dst = arr[:, tx_rel * N : (tx_rel + 1) * N, :]
            if not skip_rendering:
                tx = render_tx + tx_rel
                try:
                    surface.blit_tile_into(dst, alpha, tx, ty, **kwargs)
                except Exception:
                    logger.exception("Failed to blit tile %r of %r", (tx, ty), surface)
                    mypaintlib.tile_clear_rgba8(dst)

        # yield a numpy array of the scanline without padding
        res = arr_xcrop
        if ty == last_row:
            res = res[: y + h - ty * N, :, :]
        if ty == first_row:
            res = res[y - render_ty * N :, :, :]
        yield res


def save_as_png(surface, filename, *rect, **kwargs):
    # type: (Types.ELLIPSIS) -> Types.NONE
    """Saves a tile-blittable surface to a file in PNG format

    Args:
        surface: Surface to save
        filename: The file to write
        *rect: 
        **kwargs: 

    Returns:

    Raises:

    """
    # Horrible, dirty argument handling
    alpha = kwargs.pop("alpha", False)
    progress = kwargs.pop("progress", None)
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
            w,
            h,
            alpha,
            save_srgb_chunks,
        )
        with open(filename, "wb") as writer_fp:
            pngsave = mypaintlib.ProgressivePNGWriter(
                writer_fp,
                w,
                h,
                alpha,
                save_srgb_chunks,
            )
            scanline_strips = scanline_strips_iter(
                surface,
                rect,
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
                        "Failed to update lib.feedback.Progress: " "dropping it"
                    )
                    progress = None
            pngsave.close()
        logger.debug("Finished writing %r", filename)
        if progress:
            progress.close()
    except (IOError, OSError, RuntimeError) as err:
        logger.exception(
            "Caught %r from C++ png-writer code, re-raising as a " "FileHandlingError",
            err,
        )
        raise FileHandlingError(
            C_(
                "low-level PNG writer failure report (dialog)",
                "Failed to write “{basename}”.\n\n"
                "Reason: {err}\n"
                "Target folder: “{dirname}”.",
            ).format(
                err=err,
                basename=os.path.basename(filename),
                dirname=os.path.dirname(filename),
            )
        )
        # Other possible exceptions include TypeError, ValueError, but
        # those indicate incorrect coding usually; just raise them
        # normally.


if __name__ == "__main__":
    import doctest

    doctest.testmod()
