# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2019 by The Mypaint Development Team
# Copyright (C) 2011-2017 by Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (C) 2007-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Data layer classes"""


## Imports
from __future__ import division, print_function

import zlib
import logging
import os
import time
import tempfile
import shutil
from copy import deepcopy
from random import randint
import uuid
import struct
import contextlib

from lib.brush import BrushInfo
from lib.gettext import C_
from lib.tiledsurface import N
import lib.tiledsurface as tiledsurface
import lib.strokemap
import lib.helpers as helpers
import lib.fileutils
import lib.pixbuf
import lib.modes
import lib.mypaintlib
from . import core
import lib.layer.error
import lib.autosave
import lib.xml
import lib.feedback
from . import rendering
from lib.pycompat import PY3
from lib.pycompat import unicode

if PY3:
    from io import StringIO
    from io import BytesIO
else:
    from cStringIO import StringIO


logger = logging.getLogger(__name__)


## Base classes


class SurfaceBackedLayer (core.LayerBase, lib.autosave.Autosaveable):
    """Minimal Surface-backed layer implementation

    This minimal implementation is backed by a surface, which is used
    for rendering by by the main application; subclasses are free to
    choose whether they consider the surface to be the canonical source
    of layer data or something else with the surface being just a
    preview.
    """

    #: Suffixes allowed in load_from_openraster().
    #: Values are strings with leading dots.
    #: Use a list containing "" to allow *any* file to be loaded.
    #: The first item in the list can be used as a default extension.
    ALLOWED_SUFFIXES = []

    #: Substitute content if the layer cannot be loaded.
    FALLBACK_CONTENT = None

    ## Initialization

    def __init__(self, surface=None, **kwargs):
        """Construct a new SurfaceBackedLayer

        :param surface: Surface to use, overriding the default.
        :param **kwargs: passed to superclass.

        If `surface` is specified, content observers will not be attached, and
        the layer will not be cleared during construction. The default is to
        instantiate and use a new, observed, `tiledsurface.Surface`.
        """
        super(SurfaceBackedLayer, self).__init__(**kwargs)

        # Pluggable surface implementation
        # Only connect observers if using the default tiled surface
        if surface is None:
            self._surface = tiledsurface.Surface()
            self._surface.observers.append(self._content_changed)
        else:
            self._surface = surface

    @classmethod
    def new_from_surface_backed_layer(cls, src):
        """Clone from another SurfaceBackedLayer

        :param cls: Called as a @classmethod
        :param SurfaceBackedLayer src: Source layer
        :return: A new instance of type `cls`.

        """
        if not isinstance(src, SurfaceBackedLayer):
            raise ValueError("Source must be a SurfaceBacedLayer")
        layer = cls()
        src_snap = src.save_snapshot()
        assert isinstance(src_snap, SurfaceBackedLayerSnapshot)
        SurfaceBackedLayerSnapshot.restore_to_layer(src_snap, layer)
        return layer

    def load_from_surface(self, surface):
        """Load the backing surface image's tiles from another surface"""
        self._surface.load_from_surface(surface)

    def load_from_strokeshape(self, strokeshape, bbox=None, center=None):
        """Load image tiles from a stroke shape object.

        :param strokemap.StrokeShape strokeshape: source shape
        :param tuple bbox: Optional (x,y,w,h) pixel bbox to render in.
        :param tuple center: Optional (x,y) center of interest.

        """
        strokeshape.render_to_surface(self._surface, bbox=bbox, center=center)

    ## Loading

    def load_from_openraster(self, orazip, elem, cache_dir, progress,
                             x=0, y=0, **kwargs):
        """Loads layer flags and bitmap/surface data from a .ora zipfile

        The normal behaviour is to load the surface data directly from
        the OpenRaster zipfile without using a temporary file. This
        method also checks the src attribute's suffix against
        ALLOWED_SUFFIXES before attempting to load the surface.

        See: _load_surface_from_orazip_member()

        """
        # Load layer flags
        super(SurfaceBackedLayer, self).load_from_openraster(
            orazip,
            elem,
            cache_dir,
            progress,
            x=x, y=y,
            **kwargs
        )
        # Read bitmap content into the surface
        attrs = elem.attrib
        src = attrs.get("src", None)
        src_rootname, src_ext = os.path.splitext(src)
        src_rootname = os.path.basename(src_rootname)
        src_ext = src_ext.lower()
        x += int(attrs.get('x', 0))
        y += int(attrs.get('y', 0))
        logger.debug(
            "Trying to load %r at %+d%+d, as %r",
            src,
            x, y,
            self.__class__.__name__,
        )
        suffixes = self.ALLOWED_SUFFIXES
        if ("" not in suffixes) and (src_ext not in suffixes):
            logger.debug(
                "Abandoning load attempt, cannot load %rs from a %r "
                "(supported file extensions: %r)",
                self.__class__.__name__,
                src_ext,
                suffixes,
            )
            raise lib.layer.error.LoadingFailed(
                "Only %r are supported" % (suffixes,),
            )
        # Delegate the actual loading part
        self._load_surface_from_orazip_member(
            orazip,
            cache_dir,
            src,
            progress,
            x, y,
        )

    def _load_surface_from_orazip_member(self, orazip, cache_dir,
                                         src, progress, x, y):
        """Loads the surface from a member of an OpenRaster zipfile

        Intended strictly for override by subclasses which need to first
        extract and then keep the file around afterwards.

        """
        pixbuf = lib.pixbuf.load_from_zipfile(
            datazip=orazip,
            filename=src,
            progress=progress,
        )
        self.load_surface_from_pixbuf(pixbuf, x=x, y=y)

    def load_from_openraster_dir(self, oradir, elem, cache_dir, progress,
                                 x=0, y=0, **kwargs):
        """Loads layer flags and data from an OpenRaster-style dir"""
        # Load layer flags
        super(SurfaceBackedLayer, self).load_from_openraster_dir(
            oradir,
            elem,
            cache_dir,
            progress,
            x=x, y=y,
            **kwargs
        )
        # Read bitmap content into the surface
        attrs = elem.attrib
        src = attrs.get("src", None)
        src_rootname, src_ext = os.path.splitext(src)
        src_rootname = os.path.basename(src_rootname)
        src_ext = src_ext.lower()
        x += int(attrs.get('x', 0))
        y += int(attrs.get('y', 0))
        logger.debug(
            "Trying to load %r at %+d%+d, as %r",
            src,
            x, y,
            self.__class__.__name__,
        )
        suffixes = self.ALLOWED_SUFFIXES
        if ("" not in suffixes) and (src_ext not in suffixes):
            logger.debug(
                "Abandoning load attempt, cannot load %rs from a %r "
                "(supported file extensions: %r)",
                self.__class__.__name__,
                src_ext,
                suffixes,
            )
            raise lib.layer.error.LoadingFailed(
                "Only %r are supported" % (suffixes,),
            )
        # Delegate the actual loading part
        self._load_surface_from_oradir_member(
            oradir,
            cache_dir,
            src,
            progress,
            x, y,
        )

    def _load_surface_from_oradir_member(self, oradir, cache_dir,
                                         src, progress, x, y):
        """Loads the surface from a file in an OpenRaster-like folder

        Intended strictly for override by subclasses which need to
        make copies to manage.

        """
        self.load_surface_from_pixbuf_file(
            os.path.join(oradir, src),
            x, y,
            progress,
        )

    def load_surface_from_pixbuf_file(self, filename, x=0, y=0,
                                      progress=None, image_type=None):
        """Loads the layer's surface from any file which GdkPixbuf can open"""
        if progress:
            if progress.items is not None:
                raise ValueError(
                    "load_surface_from_pixbuf_file() expects "
                    "unsized progress objects"
                )
            s = os.stat(filename)
            progress.items = int(s.st_size)
        try:
            with open(filename, 'rb') as fp:
                pixbuf = lib.pixbuf.load_from_stream(fp, progress, image_type)
        except Exception as err:
            if self.FALLBACK_CONTENT is None:
                raise lib.layer.error.LoadingFailed(
                    "Failed to load %r: %r" % (filename, str(err)),
                )
            logger.warning("Failed to load %r: %r", filename, str(err))
            logger.info("Using fallback content instead of %r", filename)
            pixbuf = lib.pixbuf.load_from_stream(
                StringIO(self.FALLBACK_CONTENT),
            )
        return self.load_surface_from_pixbuf(pixbuf, x, y)

    def load_surface_from_pixbuf(self, pixbuf, x=0, y=0):
        """Loads the layer's surface from a GdkPixbuf"""
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        surface = tiledsurface.Surface()
        bbox = surface.load_from_numpy(arr, x, y)
        self.load_from_surface(surface)
        return bbox

    def clear(self):
        """Clears the layer"""
        self._surface.clear()

    ## Info methods

    @property
    def effective_opacity(self):
        """The opacity used when compositing a layer: zero if invisible"""
        if self.visible:
            return self.opacity
        else:
            return 0.0

    def get_alpha(self, x, y, radius):
        """Gets the average alpha within a certain radius at a point"""
        return self._surface.get_alpha(x, y, radius)

    def get_bbox(self):
        """Returns the inherent bounding box of the surface, tile aligned"""
        return self._surface.get_bbox()

    def is_empty(self):
        """Tests whether the surface is empty"""
        return self._surface.is_empty()

    ## Flood fill

    def flood_fill(self, fill_args, dst_layer=None):
        """Fills a point on the surface with a color

        See `PaintingLayer.flood_fill() for parameters and semantics. This
        implementation does nothing.
        """
        pass

    ## Rendering

    def get_tile_coords(self):
        return self._surface.get_tiles().keys()

    def get_render_ops(self, spec):
        """Get rendering instructions."""

        visible = self.visible
        mode = self.mode
        opacity = self.opacity

        if spec.layers is not None:
            if self not in spec.layers:
                return []

        mode_default = lib.modes.default_mode()
        if spec.previewing:
            mode = mode_default
            opacity = 1.0
            visible = True
        elif spec.solo:
            if self is spec.current:
                visible = True

        if not visible:
            return []

        ops = []
        if (spec.current_overlay is not None) and (self is spec.current):
            # Temporary special effects, e.g. layer blink.
            ops.append((rendering.Opcode.PUSH, None, None, None))
            ops.append((
                rendering.Opcode.COMPOSITE, self._surface, mode_default, 1.0,
            ))
            ops.extend(spec.current_overlay.get_render_ops(spec))
            ops.append(rendering.Opcode.POP, None, mode, opacity)
        else:
            # The 99%+ case☺
            ops.append((
                rendering.Opcode.COMPOSITE, self._surface, mode, opacity,
            ))
        return ops

    ## Translating

    def get_move(self, x, y):
        """Get a translation/move object for this layer

        :param x: Model X position of the start of the move
        :param y: Model X position of the start of the move
        :returns: A move object

        """
        return SurfaceBackedLayerMove(self, x, y)

    ## Saving

    @lib.fileutils.via_tempfile
    def save_as_png(self, filename, *rect, **kwargs):
        """Save to a named PNG file

        :param filename: filename to save to
        :param *rect: rectangle to save, as a 4-tuple
        :param **kwargs: passed to the surface's save_as_png() method
        :rtype: Gdk.Pixbuf
        """
        self._surface.save_as_png(filename, *rect, **kwargs)

    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        """Saves the layer's data into an open OpenRaster ZipFile"""
        rect = self.get_bbox()
        return self._save_rect_to_ora(orazip, tmpdir, "layer", path,
                                      frame_bbox, rect, **kwargs)

    def queue_autosave(self, oradir, taskproc, manifest, bbox, **kwargs):
        """Queues the layer for auto-saving"""

        # Queue up a task which writes the surface as a PNG. This will
        # be the file that's indexed by the <layer/>'s @src attribute.
        #
        # For looped layers - currently just the background layer - this
        # PNG file has to fill the requested save bbox so that other
        # apps will understand it. Other kinds of layer will just use
        # their inherent data bbox size, which may be smaller.
        #
        # Background layers save a simple tile too, but with a
        # mypaint-specific attribute name. If/when OpenRaster
        # standardizes looped layer data, that code should be moved
        # here.

        png_basename = self.autosave_uuid + ".png"
        png_relpath = os.path.join("data", png_basename)
        png_path = os.path.join(oradir, png_relpath)
        png_bbox = self._surface.looped and bbox or tuple(self.get_bbox())
        if self.autosave_dirty or not os.path.exists(png_path):
            task = tiledsurface.PNGFileUpdateTask(
                surface = self._surface,
                filename = png_path,
                rect = png_bbox,
                alpha = (not self._surface.looped),  # assume that means bg
                **kwargs
            )
            taskproc.add_work(task)
            self.autosave_dirty = False
        # Calculate appropriate offsets
        png_x, png_y = png_bbox[0:2]
        ref_x, ref_y = bbox[0:2]
        x = png_x - ref_x
        y = png_y - ref_y
        assert (x == y == 0) or not self._surface.looped
        # Declare and index what is about to be written
        manifest.add(png_relpath)
        elem = self._get_stackxml_element("layer", x, y)
        elem.attrib["src"] = png_relpath
        return elem

    @staticmethod
    def _make_refname(prefix, path, suffix, sep='-'):
        """Internal: standardized filename for something with a path"""
        assert "." in suffix
        path_ref = sep.join([("%02d" % (n,)) for n in path])
        if not suffix.startswith("."):
            suffix = sep + suffix
        return "".join([prefix, sep, path_ref, suffix])

    def _save_rect_to_ora(self, orazip, tmpdir, prefix, path,
                          frame_bbox, rect, progress=None, **kwargs):
        """Internal: saves a rectangle of the surface to an ORA zip"""
        # Write PNG data via a tempfile
        pngname = self._make_refname(prefix, path, ".png")
        pngpath = os.path.join(tmpdir, pngname)
        t0 = time.time()
        self._surface.save_as_png(pngpath, *rect, progress=progress, **kwargs)
        t1 = time.time()
        logger.debug('%.3fs surface saving %r', t1 - t0, pngname)
        # Archive and remove
        storepath = "data/%s" % (pngname,)
        orazip.write(pngpath, storepath)
        os.remove(pngpath)
        # Return details
        png_bbox = tuple(rect)
        png_x, png_y = png_bbox[0:2]
        ref_x, ref_y = frame_bbox[0:2]
        x = png_x - ref_x
        y = png_y - ref_y
        assert (x == y == 0) or not self._surface.looped
        elem = self._get_stackxml_element("layer", x, y)
        elem.attrib["src"] = storepath
        return elem

    ## Painting symmetry axis

    def set_symmetry_state(
            self, active, center, symmetry_type, symmetry_lines, angle):
        """Set the surface's painting symmetry axis and active flag.

        See `LayerBase.set_symmetry_state` for the params.
        """
        cx, cy = center
        self._surface.set_symmetry_state(
            bool(active),
            float(cx), float(cy),
            int(symmetry_type), int(symmetry_lines),
            float(angle)
        )

    ## Snapshots

    def save_snapshot(self):
        """Snapshots the state of the layer, for undo purposes"""
        return SurfaceBackedLayerSnapshot(self)

    ## Trimming

    def get_trimmable(self):
        return True

    def trim(self, rect):
        """Trim the layer to a rectangle, discarding data outside it

        :param rect: A trimming rectangle in model coordinates
        :type rect: tuple (x, y, w, h)

        Only complete tiles are discarded by this method.
        If a tile is neither fully inside nor fully outside the
        rectangle, the part of the tile outside the rectangle will be
        cleared.
        """
        self.autosave_dirty = True
        self._surface.trim(rect)

    ## Cleanup

    def remove_empty_tiles(self):
        """Removes empty tiles.

        :returns: Stats about the removal: (nremoved, ntotal)
        :rtype: tuple

        """
        removed, total = self._surface.remove_empty_tiles()
        return (removed, total)


class SurfaceBackedLayerMove (object):
    """Move object wrapper for surface-backed layers

    Layer Subclasses should extend this minimal implementation to
    provide functionality for doing things other than the surface tiles
    around.

    """

    def __init__(self, layer, x, y):
        super(SurfaceBackedLayerMove, self).__init__()
        surface_move = layer._surface.get_move(x, y)
        self._wrapped = surface_move

    def update(self, dx, dy):
        self._wrapped.update(dx, dy)

    def cleanup(self):
        self._wrapped.cleanup()

    def process(self, n=200):
        return self._wrapped.process(n)


class SurfaceBackedLayerSnapshot (core.LayerBaseSnapshot):
    """Minimal layer implementation's snapshot

    Snapshots are stored in commands, and used to implement undo and redo.
    They must be independent copies of the data, although copy-on-write
    semantics are fine. Snapshot objects don't have to be _full and exact_
    clones of the layer's data, but they do need to capture _inherent_
    qualities of the layer. Mere metadata can be ignored. For the base
    layer implementation, this means the surface tiles and the layer's
    opacity.
    """

    def __init__(self, layer):
        super(SurfaceBackedLayerSnapshot, self).__init__(layer)
        self.surface_sshot = layer._surface.save_snapshot()

    def restore_to_layer(self, layer):
        super(SurfaceBackedLayerSnapshot, self).restore_to_layer(layer)
        layer._surface.load_snapshot(self.surface_sshot)


class FileBackedLayer (SurfaceBackedLayer, core.ExternallyEditable):
    """A layer with primarily file-based storage

    File-based layers use temporary files for storage, and create one
    file per edit of the layer in an external application. The only
    operation which can change the file's content is editing the file in
    an external app. The layer's position on the MyPaint canvas, its
    mode and its opacity can be changed as normal.

    The internal surface is used only to store and render a bitmap
    preview of the layer's content.

    """

    ## Class constants

    ALLOWED_SUFFIXES = []
    REVISIONS_SUBDIR = u"revisions"

    ## Construction

    def __init__(self, x=0, y=0, **kwargs):
        """Construct, with blank internal fields"""
        super(FileBackedLayer, self).__init__(**kwargs)
        self._workfile = None
        self._x = int(round(x))
        self._y = int(round(y))
        self._keywords = kwargs.copy()
        self._keywords["x"] = x
        self._keywords["y"] = y

    def _ensure_valid_working_file(self):
        if self._workfile is not None:
            return
        ext = self.ALLOWED_SUFFIXES[0]
        rev0_fp = tempfile.NamedTemporaryFile(
            mode = "wb",
            suffix = ext,
            dir = self.revisions_dir,
            delete = False,
        )
        self.write_blank_backing_file(rev0_fp, **self._keywords)
        rev0_fp.close()
        self._workfile = _ManagedFile(rev0_fp.name)
        logger.info("Loading new blank working file from %r", rev0_fp.name)
        self.load_surface_from_pixbuf_file(
            rev0_fp.name,
            x=self._x,
            y=self._y,
        )
        redraw_bbox = self.get_full_redraw_bbox()
        self._content_changed(*redraw_bbox)

    @property
    def revisions_dir(self):
        cache_dir = self.root.doc.cache_dir
        revisions_dir = os.path.join(cache_dir, self.REVISIONS_SUBDIR)
        if not os.path.isdir(revisions_dir):
            os.makedirs(revisions_dir)
        return revisions_dir

    def write_blank_backing_file(self, file, **kwargs):
        """Write out the zeroth backing file revision.

        :param file: open file-like object to write bytes into.
        :param **kwargs: all construction params, including x and y.

        This operation is deferred until the file is needed.

        """
        raise NotImplementedError

    def _load_surface_from_orazip_member(self, orazip, cache_dir,
                                         src, progress, x, y):
        """Loads the surface from a member of an OpenRaster zipfile

        This override retains a managed copy of the extracted file in
        the REVISIONS_SUBDIR of the cache folder.

        """
        # Extract a copy of the file, and load that
        tmpdir = os.path.join(cache_dir, "tmp")
        if not os.path.isdir(tmpdir):
            os.makedirs(tmpdir)
        orazip.extract(src, path=tmpdir)
        tmp_filename = os.path.join(tmpdir, src)
        self.load_surface_from_pixbuf_file(
            tmp_filename,
            x, y,
            progress,
        )
        # Move it to the revisions subdir, and manage it there.
        revisions_dir = os.path.join(cache_dir, self.REVISIONS_SUBDIR)
        if not os.path.isdir(revisions_dir):
            os.makedirs(revisions_dir)
        self._workfile = _ManagedFile(
            unicode(tmp_filename),
            move=True,
            dir=revisions_dir,
        )
        # Record its loaded position
        self._x = x
        self._y = y

    def _load_surface_from_oradir_member(self, oradir, cache_dir,
                                         src, progress, x, y):
        """Loads the surface from a file in an OpenRaster-like folder

        This override makes a managed copy of the original file in the
        REVISIONS_SUBDIR of the cache folder.

        """
        # Load the displayed surface tiles
        super(FileBackedLayer, self)._load_surface_from_oradir_member(
            oradir, cache_dir,
            src, progress,
            x, y,
        )
        # Copy it to the revisions subdir, and manage it there.
        revisions_dir = os.path.join(cache_dir, self.REVISIONS_SUBDIR)
        if not os.path.isdir(revisions_dir):
            os.makedirs(revisions_dir)
        self._workfile = _ManagedFile(
            unicode(os.path.join(oradir, src)),
            copy=True,
            dir=revisions_dir,
        )
        # Record its loaded position
        self._x = x
        self._y = y

    ## Snapshots & cloning

    def save_snapshot(self):
        """Snapshots the state of the layer and its strokemap for undo"""
        return FileBackedLayerSnapshot(self)

    def __deepcopy__(self, memo):
        clone = super(FileBackedLayer, self).__deepcopy__(memo)
        clone._workfile = deepcopy(self._workfile)
        return clone

    ## Moving

    def get_move(self, x, y):
        """Start a new move for the layer"""
        return FileBackedLayerMove(self, x, y)

    ## Trimming (no-op for file-based layers)

    def get_trimmable(self):
        return False

    def trim(self, rect):
        pass

    ## Saving

    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        """Saves the working file to an OpenRaster zipfile"""
        # No supercall in this override, but the base implementation's
        # attributes method is useful.
        ref_x, ref_y = frame_bbox[0:2]
        x = self._x - ref_x
        y = self._y - ref_y
        elem = self._get_stackxml_element("layer", x, y)
        # Pick a suitable name to store under.
        self._ensure_valid_working_file()
        src_path = unicode(self._workfile)
        src_rootname, src_ext = os.path.splitext(src_path)
        src_ext = src_ext.lower()
        storename = self._make_refname("layer", path, src_ext)
        storepath = "data/%s" % (storename,)
        # Archive (but do not remove) the managed tempfile
        orazip.write(src_path, storepath)
        # Return details of what was written.
        elem.attrib["src"] = unicode(storepath)
        return elem

    def queue_autosave(self, oradir, taskproc, manifest, bbox, **kwargs):
        """Queues the layer for auto-saving"""
        # Again, no supercall. Autosave the backing file by copying it.
        ref_x, ref_y = bbox[0:2]
        x = self._x - ref_x
        y = self._y - ref_y
        elem = self._get_stackxml_element("layer", x, y)
        # Pick a suitable name to store under.
        self._ensure_valid_working_file()
        src_path = unicode(self._workfile)
        src_rootname, src_ext = os.path.splitext(src_path)
        src_ext = src_ext.lower()
        final_basename = self.autosave_uuid + src_ext
        final_relpath = os.path.join("data", final_basename)
        final_path = os.path.join(oradir, final_relpath)
        if self.autosave_dirty or not os.path.exists(final_path):
            final_dir = os.path.join(oradir, "data")
            tmp_fp = tempfile.NamedTemporaryFile(
                mode = "wb",
                prefix = final_basename,
                dir = final_dir,
                delete = False,
            )
            tmp_path = tmp_fp.name
            # Copy the managed tempfile now.
            # Though perhaps this could be processed in chunks
            # like other layers.
            with open(src_path, "rb") as src_fp:
                shutil.copyfileobj(src_fp, tmp_fp)
            tmp_fp.close()
            lib.fileutils.replace(tmp_path, final_path)
            self.autosave_dirty = False
        # Return details of what gets written.
        manifest.add(final_relpath)
        elem.attrib["src"] = unicode(final_relpath)
        return elem

    ## Editing via external apps

    def new_external_edit_tempfile(self):
        """Get a tempfile for editing in an external app"""
        if self.root is None:
            return
        self._ensure_valid_working_file()
        self._edit_tempfile = _ManagedFile(
            unicode(self._workfile),
            copy = True,
            dir = self.external_edits_dir,
        )
        return unicode(self._edit_tempfile)

    def load_from_external_edit_tempfile(self, tempfile_path):
        """Load content from an external-edit tempfile"""
        redraw_bboxes = []
        redraw_bboxes.append(self.get_full_redraw_bbox())
        x = self._x
        y = self._y
        self.load_surface_from_pixbuf_file(tempfile_path, x=x, y=y)
        redraw_bboxes.append(self.get_full_redraw_bbox())
        self._workfile = _ManagedFile(
            tempfile_path,
            copy = True,
            dir = self.revisions_dir,
        )
        self._content_changed(*tuple(core.combine_redraws(redraw_bboxes)))
        self.autosave_dirty = True


class FileBackedLayerSnapshot (SurfaceBackedLayerSnapshot):
    """Snapshot subclass for file-backed layers"""

    def __init__(self, layer):
        super(FileBackedLayerSnapshot, self).__init__(layer)
        self.workfile = layer._workfile
        self.x = layer._x
        self.y = layer._y

    def restore_to_layer(self, layer):
        super(FileBackedLayerSnapshot, self).restore_to_layer(layer)
        layer._workfile = self.workfile
        layer._x = self.x
        layer._y = self.y
        layer.autosave_dirty = True


class FileBackedLayerMove (SurfaceBackedLayerMove):
    """Move object wrapper for file-backed layers"""

    def __init__(self, layer, x, y):
        super(FileBackedLayerMove, self).__init__(layer, x, y)
        self._layer = layer
        self._start_x = layer._x
        self._start_y = layer._y

    def update(self, dx, dy):
        super(FileBackedLayerMove, self).update(dx, dy)
        # Update file position too.
        self._layer._x = int(round(self._start_x + dx))
        self._layer._y = int(round(self._start_y + dy))
        # The file itself is the canonical source of the data,
        # and just setting the position doesn't change that.
        # So no need to set autosave_dirty here for these layers.


## Utility classes


class _ManagedFile (object):
    """Working copy of a file, as used by file-backed layers

    Managed files take control of an unmanaged file on disk when they
    are created, and unlink it from the disk when their object is
    destroyed. If you need a fresh copy to work on, the standard copy()
    implementation handles that in the way you'd expect.

    The underlying filename can be accessed by converting to `unicode`.

    """

    def __init__(self, file_path, copy=False, move=False, dir=None):
        """Initialize, taking control of an unmanaged file or a copy

        :param unicode file_path: File to manage or manage a copy of
        :param bool copy: Copy first, and manage the copy
        :param bool move: Move first, and manage under the new name
        :param unicode dir: Target folder for move or copy.

        The file can be automatically copied or renamed first,
        in which case the new file is managed instead of the original.
        The new file will preserve the original's file extension,
        but otherwise use UUID (random) syntax.
        If `targdir` is undefined, this new file will be
        created in the same folder as the original.

        Creating these objects, or copying them, should only be
        attempted from the main thread.

        """
        assert isinstance(file_path, unicode)
        assert os.path.isfile(file_path)
        if dir:
            assert os.path.isdir(dir)
        super(_ManagedFile, self).__init__()
        file_path = self._get_file_to_manage(
            file_path,
            copy=copy,
            move=move,
            dir=dir,
        )
        file_dir, file_basename = os.path.split(file_path)
        self._dir = file_dir
        self._basename = file_basename

    def __copy__(self):
        """Shallow copies work just like deep copies"""
        return deepcopy(self)

    def __deepcopy__(self, memo):
        """Deep-copying a _ManagedFile copies the file"""
        orig_path = unicode(self)
        clone_path = self._get_file_to_manage(orig_path, copy=True)
        logger.debug("_ManagedFile: cloned %r as %r within %r",
                     self._basename, os.path.basename(clone_path), self._dir)
        return _ManagedFile(clone_path)

    @staticmethod
    def _get_file_to_manage(orig_path, copy=False, move=False, dir=None):
        """Obtain a file path to manage. Same params as constructor.

        If asked to copy or rename first,
        UUID-based naming is used without much error checking.
        This should be sufficient for MyPaint's usage
        because the document working dir is atomically constructed.
        However it's not truly atomic or threadsafe.

        """
        assert os.path.isfile(orig_path)
        if not (copy or move):
            return orig_path
        orig_dir, orig_basename = os.path.split(orig_path)
        orig_rootname, orig_ext = os.path.splitext(orig_basename)
        if dir is None:
            dir = orig_dir
        new_unique_path = None
        while new_unique_path is None:
            new_rootname = unicode(uuid.uuid4())
            new_basename = new_rootname + orig_ext
            new_path = os.path.join(dir, new_basename)
            if os.path.exists(new_path):  # yeah, paranoia
                logger.warn("UUID clash: %r exists", new_path)
                continue
            if move:
                os.rename(orig_path, new_path)
            else:
                shutil.copy2(orig_path, new_path)
            new_unique_path = new_path
        assert os.path.isfile(new_unique_path)
        return new_unique_path

    def __str__(self):
        if PY3:
            return self.__unicode__()
        else:
            return self.__bytes__()  # Always an error under Py2

    def __bytes__(self):
        raise NotImplementedError("Use unicode strings for file names.")

    def __unicode__(self):
        file_path = os.path.join(self._dir, self._basename)
        assert isinstance(file_path, unicode)
        return file_path

    def __repr__(self):
        return "_ManagedFile(%r)" % (self,)

    def __del__(self):
        try:
            file_path = unicode(self)
        except Exception:
            logger.exception("_ManagedFile: cleanup of incomplete object. "
                             "File may still exist on disk.")
            return
        if os.path.exists(file_path):
            logger.debug("_ManagedFile: %r is no longer referenced, deleting",
                         file_path)
            os.unlink(file_path)
        else:
            logger.debug("_ManagedFile: %r was already removed, not deleting",
                         file_path)


## Data layer classes


class BackgroundLayer (SurfaceBackedLayer):
    """Background layer, with a repeating tiled image

    By convention only, there is just a single non-editable background
    layer in any document, hidden behind an API in the document's
    RootLayerStack. In the MyPaint application, the working document's
    background layer cannot be manipulated by the user except through
    the background dialog.
    """

    # This could be generalized as a repeating tile for general use in
    # the layers stack, extending the FileBackedLayer concept.  Think
    # textures!

    # The legacy non-namespaced attribute is no longer _written_
    # to files as of the 2.0 release. 2.0 .ora files will not
    # be stable in 1.2.1 and earlier in the general case, so
    # there is no point in pretending that they are.

    # MyPaint will support _reading_ .ora files using the legacy
    # background tile attribute through the 2.x releases, but
    # no distinction is made when such files are subseqently saved.

    ORA_BGTILE_LEGACY_ATTR = "background_tile"
    ORA_BGTILE_ATTR = "{%s}background-tile" % (
        lib.xml.OPENRASTER_MYPAINT_NS,
    )

    def __init__(self, bg, **kwargs):
        if isinstance(bg, tiledsurface.Background):
            surface = bg
        else:
            surface = tiledsurface.Background(bg)
        super(BackgroundLayer, self).__init__(name=u"background",
                                              surface=surface, **kwargs)
        self.locked = False
        self.visible = True
        self.mode = lib.mypaintlib.CombineNormal
        self.opacity = 1.0

    def set_surface(self, surface):
        """Sets the surface from a tiledsurface.Background"""
        assert isinstance(surface, tiledsurface.Background)
        self.autosave_dirty = True
        self._surface = surface

    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox,
                           progress=None, **kwargs):

        if not progress:
            progress = lib.feedback.Progress()
        progress.items = 2

        # Item 1: save as a regular layer for other apps.
        # Background surfaces repeat, so just the bit filling the frame.
        elem = self._save_rect_to_ora(
            orazip, tmpdir, "background", path,
            frame_bbox, frame_bbox,
            progress=progress.open(),
            **kwargs
        )

        # Item 2: also save as single pattern (with corrected origin)
        x0, y0 = frame_bbox[0:2]
        x, y, w, h = self.get_bbox()

        pngname = self._make_refname("background", path, "tile.png")
        tmppath = os.path.join(tmpdir, pngname)
        t0 = time.time()
        self._surface.save_as_png(
            tmppath,
            x=x + x0,
            y=y + y0,
            w=w,
            h=h,
            progress=progress.open(),
            **kwargs
        )
        t1 = time.time()
        storename = 'data/%s' % (pngname,)
        logger.debug('%.3fs surface saving %s', t1 - t0, storename)
        orazip.write(tmppath, storename)
        os.remove(tmppath)
        elem.attrib[self.ORA_BGTILE_ATTR] = storename

        progress.close()
        return elem

    def queue_autosave(self, oradir, taskproc, manifest, bbox, **kwargs):
        """Queues the layer for auto-saving"""
        # Arrange for the tile PNG to be rewritten, if necessary
        tilepng_basename = self.autosave_uuid + "-tile.png"
        tilepng_relpath = os.path.join("data", tilepng_basename)
        manifest.add(tilepng_relpath)
        x0, y0 = bbox[0:2]
        x, y, w, h = self.get_bbox()
        tilepng_bbox = (x + x0, y + y0, w, h)
        tilepng_path = os.path.join(oradir, tilepng_relpath)
        if self.autosave_dirty or not os.path.exists(tilepng_path):
            task = tiledsurface.PNGFileUpdateTask(
                surface = self._surface,
                filename = tilepng_path,
                rect = tilepng_bbox,
                alpha = False,
                **kwargs
            )
            taskproc.add_work(task)
        # Supercall will clear the dirty flag, no need to do it here
        elem = super(BackgroundLayer, self).queue_autosave(
            oradir, taskproc, manifest, bbox,
            **kwargs
        )
        elem.attrib[self.ORA_BGTILE_LEGACY_ATTR] = tilepng_relpath
        elem.attrib[self.ORA_BGTILE_ATTR] = tilepng_relpath
        return elem

    def save_snapshot(self):
        """Snapshots the state of the layer, for undo purposes"""
        return BackgroundLayerSnapshot(self)


class BackgroundLayerSnapshot (core.LayerBaseSnapshot):
    """Snapshot of a root layer stack's state"""

    def __init__(self, layer):
        super(BackgroundLayerSnapshot, self).__init__(layer)
        self.surface = layer._surface

    def restore_to_layer(self, layer):
        super(BackgroundLayerSnapshot, self).restore_to_layer(layer)
        layer._surface = self.surface


class VectorLayer (FileBackedLayer):
    """SVG-based vector layer

    Vector layers respect a wider set of construction parameters than
    most layers:

    :param float x: SVG document X coordinate, in model coords
    :param float y: SVG document Y coordinate, in model coords
    :param float w: SVG document width, in model pixels
    :param float h: SVG document height, in model pixels
    :param iterable outline: Initial shape, absolute ``(X, Y)`` points

    The outline shape is drawn with a random color, and a thick dashed
    surround. It is intended to indicate where the SVG file goes on the
    canvas initially, to help avoid confusion.

    The document bounding box should enclose all points of the outline.

    """

    DEFAULT_NAME = C_(
        "layer default names",
        # TRANSLATORS: Short default name for vector (SVG/Inkscape) layers
        u"Vectors",
    )

    TYPE_DESCRIPTION = C_(
        "layer type descriptions",
        u"Vector Layer",
    )

    ALLOWED_SUFFIXES = [".svg"]

    def get_icon_name(self):
        return "mypaint-layer-vector-symbolic"

    def load_surface_from_pixbuf_file(self, *args, **kwds):
        """Overrides pixbuf loading to explicitly handle svg data"""
        kwds.update({"image_type": "svg"})
        return super(VectorLayer, self).load_surface_from_pixbuf_file(
            *args, **kwds)

    def write_blank_backing_file(self, file, **kwargs):
        x = kwargs.get("x", 0)
        y = kwargs.get("y", 0)
        outline = kwargs.get("outline")
        if outline:
            outline = [(px - x, py - y) for (px, py) in outline]
        else:
            outline = [(0, 0), (0, N), (N, N), (N, 0)]
        svg = (
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
            '<!-- Created by MyPaint (http://mypaint.org/) -->'
            '<svg version="1.1" width="{w}" height="{h}">'
            '<path d="M '
        ).format(**kwargs)
        for px, py in outline:
            svg += "{x},{y} ".format(x=px, y=py)
        rgb = tuple([randint(0x33, 0x99) for i in range(3)])
        col = "#%02x%02x%02x" % rgb
        svg += (
            'Z" id="path0" '
            'style="fill:none;stroke:{col};stroke-width:5;'
            'stroke-linecap:round;stroke-linejoin:round;'
            'stroke-dasharray:9, 9;stroke-dashoffset:0" />'
            '</svg>'
        ).format(col=col)

        if not isinstance(svg, bytes):
            svg = svg.encode("utf-8")
        file.write(svg)

    def flood_fill(self, fill_args, dst_layer=None):
        """Fill to dst_layer, with ref. to a rasterization of this layer.
        This implementation is virtually identical to the one in LayerStack.
        """
        assert dst_layer is not self
        assert dst_layer is not None

        root = self.root
        if root is None:
            raise ValueError(
                "Cannot flood_fill() into a vector layer which is not "
                "a descendent of a RootLayerStack."
            )
        src = root.get_tile_accessible_layer_rendering(self)
        dst = dst_layer._surface
        return tiledsurface.flood_fill(src, fill_args, dst)


class FallbackBitmapLayer (FileBackedLayer):
    """An unpaintable, fallback bitmap layer"""

    def get_icon_name(self):
        return "mypaint-layer-fallback-symbolic"

    DEFAULT_NAME = C_(
        "layer default names",
        # TRANSLATORS: Short default name for renderable fallback layers
        "Bitmap",
    )

    TYPE_DESCRIPTION = C_(
        "layer type descriptions",
        u"Bitmap Data",
    )

    #: Any suffix is allowed, no preference for defaults
    ALLOWED_SUFFIXES = [""]


class FallbackDataLayer (FileBackedLayer):
    """An unpaintable, fallback, non-bitmap layer"""

    def get_icon_name(self):
        return "mypaint-layer-fallback-symbolic"

    DEFAULT_NAME = C_(
        "layer default names",
        # TRANSLATORS: Short default name for non-renderable fallback layers
        u"Data",
    )

    TYPE_DESCRIPTION = C_(
        "layer type descriptions",
        u"Unknown Data",
    )

    #: Any suffix is allowed, favour ".dat".
    ALLOWED_SUFFIXES = [".dat", ""]

    #: Use a silly little icon so that the layer can be positioned
    FALLBACK_CONTENT = (
        '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
        <svg width="64" height="64" version="1.1"
                xmlns="http://www.w3.org/2000/svg">
        <rect width="62" height="62" x="1.5" y="1.5"
                style="{rectstyle};fill:{shadow};stroke:{shadow}" />
            <rect width="62" height="62" x="0.5" y="0.5"
                style="{rectstyle};fill:{base};stroke:{basestroke}" />
            <text x="33.5" y="50.5"
                style="{textstyle};fill:{textshadow};stroke:{textshadow}"
                >?</text>
            <text x="32.5" y="49.5"
                style="{textstyle};fill:{text};stroke:{textstroke}"
                >?</text>
        </svg>''').format(
        rectstyle="stroke-width:1",
        shadow="#000",
        base="#eee",
        basestroke="#fff",
        textstyle="text-align:center;text-anchor:middle;"
                  "font-size:48px;font-weight:bold;font-family:sans",
        text="#9c0",
        textshadow="#360",
        textstroke="#ad1",
    )


## User-paintable layer classes

class SimplePaintingLayer (SurfaceBackedLayer):
    """A layer you can paint on, but not much else."""

    ## Class constants

    ALLOWED_SUFFIXES = [".png"]

    DEFAULT_NAME = C_(
        "layer default names",
        # TRANSLATORS: Default name for new normal, paintable layers
        u"Layer",
    )

    TYPE_DESCRIPTION = C_(
        "layer type descriptions",
        u"Painting Layer",
    )

    ## Flood fill

    def get_fillable(self):
        """True if this layer currently accepts flood fill"""
        return not self.locked

    def flood_fill(self, fill_args, dst_layer=None):
        """Fills a point on the surface with a color

        :param fill_args: Parameters common to all fill calls
        :type fill_args: lib.floodfill.FloodFillArguments
        :param dst_layer: Optional target layer (default is self!)
        :type dst_layer: StrokemappedPaintingLayer

        The `tolerance` parameter controls how much pixels are permitted to
        vary from the starting (target) color. This is calculated based on the
        rgba channel with the largest difference to the corresponding channel
        of the starting color, scaled to a number in [0,1] and also determines
        the alpha of filled pixels.

        The default target layer is `self`. This method invalidates the filled
        area of the target layer's surface, queueing a redraw if it is part of
        a visible document.
        """
        if dst_layer is None:
            dst_layer = self
        dst_layer.autosave_dirty = True   # XXX hmm, not working?
        return self._surface.flood_fill(fill_args, dst=dst_layer._surface)

    ## Simple painting

    def get_paintable(self):
        """True if this layer currently accepts painting brushstrokes"""
        return (
            self.visible
            and not self.locked
            and self.branch_visible
            and not self.branch_locked
        )


    def stroke_to(self, brush, x, y, pressure, xtilt, ytilt, dtime,
                  viewzoom, viewrotation, barrel_rotation):
        """Render a part of a stroke to the canvas surface

        :param brush: The brush to use for rendering dabs
        :type brush: lib.brush.Brush
        :param x: Input event's X coord, translated to document coords
        :param y: Input event's Y coord, translated to document coords
        :param pressure: Input event's pressure
        :param xtilt: Input event's tilt component in the document X direction
        :param ytilt: Input event's tilt component in the document Y direction
        :param dtime: Time delta, in seconds
        :returns: whether the stroke should now be split
        :rtype: bool

        This method renders zero or more dabs to the surface of this
        layer, but it won't affect any strokemap maintained by this
        object (even if subclasses add one). That's because this method
        is for tiny increments, not big brushstrokes.

        Use this for the incremental painting of segments of a stroke
        corresponding to single input events.  The return value tells
        the caller whether to finalize the lib.stroke.Stroke which is
        currently recording the user's input, and begin recording a new
        one. You can choose to ignore it if you're just using a
        SimplePaintingLayer and not recording strokes.

        """
        self._surface.begin_atomic()
        split = brush.stroke_to(
            self._surface.backend, x, y,
            pressure, xtilt, ytilt, dtime, viewzoom,
            viewrotation, barrel_rotation
        )
        self._surface.end_atomic()
        self.autosave_dirty = True
        return split

    @contextlib.contextmanager
    def cairo_request(self, x, y, w, h, mode=lib.modes.default_mode):
        """Get a Cairo context for a given area, then put back changes.

        See lib.tiledsurface.MyPaintSurface.cairo_request() for details.
        This is just a wrapper.

        """
        with self._surface.cairo_request(x, y, w, h, mode) as cr:
            yield cr
        self.autosave_dirty = True

    ## Type-specific stuff

    def get_icon_name(self):
        return "mypaint-layer-painting-symbolic"


class StrokemappedPaintingLayer (SimplePaintingLayer):
    """Painting layer with a record of user brushstrokes.

    This class definition adds a strokemap to the simple implementation.
    The stroke map is a stack of `strokemap.StrokeShape` objects in
    painting order, allowing strokes and their associated brush and
    color information to be picked from the canvas.

    The caller of stroke_to() is expected to also maintain a current
    lib.stroke.Stroke object which records user input for the current
    stroke, but no shape info. When stroke_to() says to break the
    stroke, or when the caller wishes to break a stroke, feed these
    details back to the layer via add_stroke_shape() to update the
    strokemap.

    """

    ## Class constants

    # The un-namespaced legacy attribute name is deprecated since
    # MyPaint v1.2.0, and painting layers in OpenRaster files will not
    # be saved with it beginning with v2.0.0.
    # MyPaint will support reading .ora files using the legacy strokemap
    # attribute (and the "v2" strokemap format, if the format changes)
    # throughout v2.x.

    _ORA_STROKEMAP_ATTR = "{%s}strokemap" % (lib.xml.OPENRASTER_MYPAINT_NS,)
    _ORA_STROKEMAP_LEGACY_ATTR = "mypaint_strokemap_v2"

    ## Initializing & resetting

    def __init__(self, **kwargs):
        super(StrokemappedPaintingLayer, self).__init__(**kwargs)
        #: Stroke map.
        #: List of strokemap.StrokeShape instances (not stroke.Stroke),
        #: ordered by depth.
        self.strokes = []

    def clear(self):
        """Clear both the surface and the strokemap"""
        super(StrokemappedPaintingLayer, self).clear()
        self.strokes = []

    def load_from_surface(self, surface):
        """Load the surface image's tiles from another surface"""
        super(StrokemappedPaintingLayer, self).load_from_surface(surface)
        self.strokes = []

    def load_from_openraster(self, orazip, elem, cache_dir, progress,
                             x=0, y=0, invert_strokemaps=False, **kwargs):
        """Loads layer flags, PNG data, and strokemap from a .ora zipfile"""
        # Load layer tile data and flags
        super(StrokemappedPaintingLayer, self).load_from_openraster(
            orazip,
            elem,
            cache_dir,
            progress,
            x=x, y=y,
            **kwargs
        )
        self._load_strokemap_from_ora(
            elem, x, y, invert_strokemaps, orazip=orazip
        )

    def load_from_openraster_dir(self, oradir, elem, cache_dir, progress,
                                 x=0, y=0, **kwargs):
        """Loads layer flags and data from an OpenRaster-style dir"""
        # Load layer tile data and flags
        super(StrokemappedPaintingLayer, self).load_from_openraster_dir(
            oradir,
            elem,
            cache_dir,
            progress,
            x=x, y=y,
            **kwargs
        )
        self._load_strokemap_from_ora(elem, x, y, False, oradir=oradir)

    def _load_strokemap_from_ora(
            self, elem, x, y, invert=False, orazip=None, oradir=None
    ):
        """Load the strokemap from a layer elem & an ora{zip|dir}."""
        attrs = elem.attrib
        x += int(attrs.get('x', 0))
        y += int(attrs.get('y', 0))
        supported_strokemap_attrs = [
            self._ORA_STROKEMAP_ATTR,
            self._ORA_STROKEMAP_LEGACY_ATTR,
        ]
        strokemap_name = None
        for attr_qname in supported_strokemap_attrs:
            strokemap_name = attrs.get(attr_qname, None)
            if strokemap_name is None:
                continue
            logger.debug(
                "Found strokemap %r in %r",
                strokemap_name,
                attr_qname,
            )
            break
        if strokemap_name is None:
            return
        # This is a hacky way of identifying files which need their stroke
        # maps inverted, due to storing visually inconsistent colors.
        # These files are distinguished by lacking both the legacy strokemap
        # attribute and the eotf attribute. This support will be temporary.
        invert = invert and not attrs.get(self._ORA_STROKEMAP_LEGACY_ATTR)
        if orazip:
            if PY3:
                ioclass = BytesIO
            else:
                ioclass = StringIO
            sio = ioclass(orazip.read(strokemap_name))
            self._load_strokemap_from_file(sio, x, y, invert)
            sio.close()
        elif oradir:
            with open(os.path.join(oradir, strokemap_name), "rb") as sfp:
                self._load_strokemap_from_file(sfp, x, y, invert)
        else:
            raise ValueError("either orazip or oradir must be specified")

    ## Stroke recording and rendering

    def render_stroke(self, stroke):
        """Render a whole captured stroke to the canvas

        :param stroke: The stroke to render
        :type stroke: lib.stroke.Stroke
        """
        stroke.render(self._surface)
        self.autosave_dirty = True

    def add_stroke_shape(self, stroke, before):
        """Adds a rendered stroke's shape to the strokemap

        :param stroke: the stroke sequence which has been rendered
        :type stroke: lib.stroke.Stroke
        :param before: layer snapshot taken before the stroke started
        :type before: lib.layer.StrokemappedPaintingLayerSnapshot

        The StrokeMap is a stack of lib.strokemap.StrokeShape objects which
        encapsulate the shape of a rendered stroke, and the brush settings
        which were used to render it.  The shape of the rendered stroke is
        determined by visually diffing snapshots taken before the stroke
        started and now.

        """
        after_sshot = self._surface.save_snapshot()
        shape = lib.strokemap.StrokeShape.new_from_snapshots(
            before.surface_sshot,
            after_sshot,
        )
        if shape is not None:
            shape.brush_string = stroke.brush_settings
            self.strokes.append(shape)

    ## Snapshots

    def save_snapshot(self):
        """Snapshots the state of the layer and its strokemap for undo"""
        return StrokemappedPaintingLayerSnapshot(self)

    ## Translating

    def get_move(self, x, y):
        """Get an interactive move object for the surface and its strokemap"""
        return StrokemappedPaintingLayerMove(self, x, y)

    ## Trimming

    def trim(self, rect):
        """Trim the layer and its strokemap"""
        super(StrokemappedPaintingLayer, self).trim(rect)
        empty_strokes = []
        for stroke in self.strokes:
            if not stroke.trim(rect):
                empty_strokes.append(stroke)
        for stroke in empty_strokes:
            logger.debug("Removing emptied stroke %r", stroke)
            self.strokes.remove(stroke)

    ## Strokemap load and save

    def _load_strokemap_from_file(self, f, translate_x, translate_y, invert):
        assert not self.strokes
        brushes = []
        x = int(translate_x // N) * N
        y = int(translate_y // N) * N
        dx = translate_x % N
        dy = translate_y % N
        while True:
            t = f.read(1)
            if t == b"b":
                length, = struct.unpack('>I', f.read(4))
                tmp = f.read(length)
                b_string = zlib.decompress(tmp)
                if invert:
                    b_string = BrushInfo.brush_string_inverted_eotf(b_string)
                brushes.append(b_string)
            elif t == b"s":
                brush_id, length = struct.unpack('>II', f.read(2 * 4))
                stroke = lib.strokemap.StrokeShape()
                tmp = f.read(length)
                stroke.init_from_string(tmp, x, y)
                stroke.brush_string = brushes[brush_id]
                # Translate non-aligned strokes
                if (dx, dy) != (0, 0):
                    stroke.translate(dx, dy)
                self.strokes.append(stroke)
            elif t == b"}":
                break
            else:
                errmsg = "Invalid strokemap (initial char=%r)" % (t,)
                raise ValueError(errmsg)

    ## Strokemap querying

    def get_stroke_info_at(self, x, y):
        """Get the stroke at the given point"""
        x, y = int(x), int(y)
        for s in reversed(self.strokes):
            if s.touches_pixel(x, y):
                return s

    def get_last_stroke_info(self):
        if not self.strokes:
            return None
        return self.strokes[-1]

    ## Saving

    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        """Save the strokemap too, in addition to the base implementation"""
        # Save the layer normally

        elem = super(StrokemappedPaintingLayer, self).save_to_openraster(
            orazip, tmpdir, path,
            canvas_bbox, frame_bbox, **kwargs
        )
        # Store stroke shape data too
        x, y, w, h = self.get_bbox()
        if PY3:
            sio = BytesIO()
        else:
            sio = StringIO()
        t0 = time.time()
        _write_strokemap(sio, self.strokes, -x, -y)
        t1 = time.time()
        data = sio.getvalue()
        sio.close()
        datname = self._make_refname("layer", path, "strokemap.dat")
        logger.debug("%.3fs strokemap saving %r", t1 - t0, datname)
        storepath = "data/%s" % (datname,)
        helpers.zipfile_writestr(orazip, storepath, data)
        # Add strokemap XML attrs and return.
        # See comment above for compatibility strategy.
        elem.attrib[self._ORA_STROKEMAP_ATTR] = storepath
        return elem

    def queue_autosave(self, oradir, taskproc, manifest, bbox, **kwargs):
        """Queues the layer for auto-saving"""
        dat_basename = u"%s-strokemap.dat" % (self.autosave_uuid,)
        dat_relpath = os.path.join("data", dat_basename)
        dat_path = os.path.join(oradir, dat_relpath)
        # Have to do this before the supercall because that will clear
        # the dirty flag.
        if self.autosave_dirty or not os.path.exists(dat_path):
            x, y, w, h = self.get_bbox()
            task = _StrokemapFileUpdateTask(
                self.strokes,
                dat_path,
                -x, -y,
            )
            taskproc.add_work(task)
        # Supercall to queue saving PNG and obtain basic XML
        elem = super(StrokemappedPaintingLayer, self).queue_autosave(
            oradir, taskproc, manifest, bbox,
            **kwargs
        )
        # Add strokemap XML attrs and return.
        # See comment above for compatibility strategy.
        elem.attrib[self._ORA_STROKEMAP_ATTR] = dat_relpath
        manifest.add(dat_relpath)
        return elem


class PaintingLayer (StrokemappedPaintingLayer, core.ExternallyEditable):
    """The normal paintable bitmap layer that the user sees."""

    def __init__(self, **kwargs):
        super(PaintingLayer, self).__init__(**kwargs)
        self._external_edit = None

    def new_external_edit_tempfile(self):
        """Get a tempfile for editing in an external app"""
        # Uniquely named tempfile. Will be overwritten.
        if not self.root:
            return
        tmp_filename = os.path.join(
            self.external_edits_dir,
            u"%s%s" % (unicode(uuid.uuid4()), u".png"),
        )
        # Overwrite, saving only the data area.
        # Record the data area for later.
        rect = self.get_bbox()
        if rect.w <= 0:
            rect.w = N
        if rect.h <= 0:
            rect.h = N
        self._surface.save_as_png(tmp_filename, *rect, alpha=True)
        edit_info = (tmp_filename, _ManagedFile(tmp_filename), rect)
        self._external_edit = edit_info
        return tmp_filename

    def load_from_external_edit_tempfile(self, tempfile_path):
        """Load content from an external-edit tempfile"""
        # Try to load the layer data back where it came from.
        # Only works if the file being loaded is the one most recently
        # created using new_external_edit_tempfile().
        x, y, __, __ = self.get_bbox()
        edit_info = self._external_edit
        if edit_info:
            tmp_filename, __, rect = edit_info
            if tempfile_path == tmp_filename:
                x, y, __, __ = rect
        redraw_bboxes = []
        redraw_bboxes.append(self.get_full_redraw_bbox())
        self.load_surface_from_pixbuf_file(tempfile_path, x=x, y=y)
        redraw_bboxes.append(self.get_full_redraw_bbox())
        self._content_changed(*tuple(core.combine_redraws(redraw_bboxes)))
        self.autosave_dirty = True


## Stroke-mapped layer implementation details and helpers

def _write_strokemap(f, strokes, dx, dy):
    brush2id = {}
    for stroke in strokes:
        _write_strokemap_stroke(f, stroke, brush2id, dx, dy)
    f.write(b'}')


def _write_strokemap_stroke(f, stroke, brush2id, dx, dy):

    # save brush (if not already recorderd)
    b = stroke.brush_string
    if b not in brush2id:
        brush2id[b] = len(brush2id)
        if isinstance(b, unicode):
            b = b.encode("utf-8")
        b = zlib.compress(b)
        f.write(b'b')
        f.write(struct.pack('>I', len(b)))
        f.write(b)

    # save stroke
    s = stroke.save_to_string(dx, dy)
    f.write(b's')
    f.write(struct.pack('>II', brush2id[stroke.brush_string], len(s)))
    f.write(s)


class _StrokemapFileUpdateTask (object):
    """Updates a strokemap file in chunked calls (for autosave)"""

    def __init__(self, strokes, filename, dx, dy):
        super(_StrokemapFileUpdateTask, self).__init__()
        tmp = tempfile.NamedTemporaryFile(
            mode = "wb",
            prefix = os.path.basename(filename),
            dir = os.path.dirname(filename),
            delete = False,
        )
        self._tmp = tmp
        self._final_name = filename
        self._dx = dx
        self._dy = dy
        self._brush2id = {}
        self._strokes = strokes[:]
        self._strokes_i = 0
        logger.debug("autosave: scheduled update of %r", self._final_name)

    def __call__(self):
        if self._tmp.closed:
            raise RuntimeError("Called too many times")
        if self._strokes_i < len(self._strokes):
            stroke = self._strokes[self._strokes_i]
            _write_strokemap_stroke(
                self._tmp,
                stroke,
                self._brush2id,
                self._dx, self._dy,
            )
            self._strokes_i += 1
            return True
        else:
            self._tmp.write(b'}')
            self._tmp.close()
            lib.fileutils.replace(self._tmp.name, self._final_name)
            logger.debug("autosave: updated %r", self._final_name)
            return False


class StrokemappedPaintingLayerSnapshot (SurfaceBackedLayerSnapshot):
    """Snapshot subclass for painting layers with strokemaps"""

    def __init__(self, layer):
        super(StrokemappedPaintingLayerSnapshot, self).__init__(layer)
        self.strokes = layer.strokes[:]

    def restore_to_layer(self, layer):
        super(StrokemappedPaintingLayerSnapshot, self).restore_to_layer(layer)
        layer.strokes = self.strokes[:]
        layer.autosave_dirty = True


class StrokemappedPaintingLayerMove (SurfaceBackedLayerMove):
    """Move object wrapper for painting layers with strokemaps"""

    def __init__(self, layer, x, y):
        super(StrokemappedPaintingLayerMove, self).__init__(layer, x, y)
        self._layer = layer
        self._final_dx = 0
        self._final_dy = 0

    def update(self, dx, dy):
        super(StrokemappedPaintingLayerMove, self).update(dx, dy)
        self._final_dx = dx
        self._final_dy = dy

    def cleanup(self):
        super(StrokemappedPaintingLayerMove, self).cleanup()
        dx = self._final_dx
        dy = self._final_dy
        # Arrange for the strokemap to be moved too;
        # this happens in its own background idler.
        for stroke in self._layer.strokes:
            stroke.translate(dx, dy)
            # Minor problem: huge strokemaps take a long time to move, and the
            # translate must be forced to completion before drawing or any
            # further layer moves. This can cause apparent hangs for no
            # reason later on. Perhaps it would be better to process them
            # fully in this hourglass-cursor phase after all?
        # The tile memory is the canonical source of a painting layer,
        # so we'll need to autosave it.
        self._layer.autosave_dirty = True


## Module testing


def _test():
    """Run doctest strings"""
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _test()
