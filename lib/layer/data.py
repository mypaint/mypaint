# This file is part of MyPaint.
# Copyright (C) 2011-2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (C) 2007-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Data layer classes"""


## Imports

import zlib
import logging
import os
from cStringIO import StringIO
import time
import zipfile
logger = logging.getLogger(__name__)
import tempfile
import shutil
from copy import deepcopy
from random import randint
import uuid
import struct

from gettext import gettext as _

import lib.tiledsurface as tiledsurface
import lib.strokemap
import lib.helpers as helpers
import lib.fileutils
import lib.pixbuf
from consts import *
import core
import lib.layer.error


## Base classes


class SurfaceBackedLayer (core.LayerBase):
    """Minimal Surface-backed layer implementation

    This minimal implementation is backed by a surface, which is used
    for rendering by by the main application; subclasses are free to
    choose whether they consider the surface to be the canonical source
    of layer data or something else with the surface being just a
    preview.
    """

    ## Class constants: capabilities

    #: Whether the surface can be painted to (if not locked)
    IS_PAINTABLE = False

    #: Whether the surface can be filled (if not locked)
    IS_FILLABLE = False

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

    def load_from_strokeshape(self, strokeshape):
        """Load image tiles from a strokemap.StrokeShape"""
        strokeshape.render_to_surface(self._surface)

    ## Loading

    def load_from_openraster(self, orazip, elem, tempdir, feedback_cb,
                             x=0, y=0, extract_and_keep=False, **kwargs):
        """Loads layer flags and bitmap/surface data from a .ora zipfile

        :param extract_and_keep: Set to true to extract and keep a copy

        The normal behaviour is to load the data file directly from `orazip`
        without using a temporary file.  If `extract_and_keep` is set, an
        alternative method is used which extracts

            os.path.join(tempdir, elem.attrib["src"])

        and reads from that. The caller is then free to do what it likes with
        this file.
        """
        # Load layer flags
        super(SurfaceBackedLayer, self) \
            .load_from_openraster(orazip, elem, tempdir, feedback_cb,
                                  x=x, y=y, **kwargs)
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
        t0 = time.time()
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
        if extract_and_keep:
            orazip.extract(src, path=tempdir)
            tmp_filename = os.path.join(tempdir, src)
            self.load_surface_from_pixbuf_file(tmp_filename, x, y, feedback_cb)
        else:
            pixbuf = lib.pixbuf.load_from_zipfile(
                datazip=orazip,
                filename=src,
                feedback_cb=feedback_cb,
            )
            self.load_surface_from_pixbuf(pixbuf, x=x, y=y)
        t1 = time.time()
        logger.debug("Loaded %r successfully", self.__class__.__name__)
        logger.debug("Spent %.3fs loading and converting %r", t1 - t0, src)

    def load_surface_from_pixbuf_file(self, filename, x=0, y=0,
                                      feedback_cb=None):
        """Loads the layer's surface from any file which GdkPixbuf can open"""
        fp = open(filename, 'rb')
        try:
            pixbuf = lib.pixbuf.load_from_stream(fp, feedback_cb)
        except Exception as err:
            if self.FALLBACK_CONTENT is None:
                raise lib.layer.error.LoadingFailed(
                    "Failed to load %r: %r" % (filename, str(err)),
                )
            logger.info("Using fallback content for %r", filename)
            pixbuf = lib.pixbuf.load_from_stream(
                StringIO(self.FALLBACK_CONTENT),
            )
        finally:
            fp.close()
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
        # Mirror what composite_tile does.
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

    def get_paintable(self):
        """True if this layer currently accepts painting brushstrokes"""
        return self.IS_PAINTABLE and not self.locked

    def get_fillable(self):
        """True if this layer currently accepts flood fill"""
        return self.IS_FILLABLE and not self.locked

    ## Flood fill

    def flood_fill(self, x, y, color, bbox, tolerance, dst_layer=None):
        """Fills a point on the surface with a color

        See `PaintingLayer.flood_fill() for parameters and semantics. This
        implementation does nothing.
        """
        pass

    ## Rendering

    def get_tile_coords(self):
        return self._surface.get_tiles().keys()

    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       **kwargs):
        """Unconditionally copy one tile's data into an array without options

        The minimal surface-based implementation composites one tile of the
        backing surface over the array dst, modifying only dst.
        """
        self._surface.composite_tile(
            dst, dst_has_alpha, tx, ty,
            mipmap_level=mipmap_level,
            opacity=1, mode=DEFAULT_MODE
        )

    def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       layers=None, previewing=None, solo=None, **kwargs):
        """Composite a tile's data into an array, respecting flags/layers list

        The minimal surface-based implementation composites one tile of the
        backing surface over the array dst, modifying only dst.
        """
        mode = self.mode
        opacity = self.opacity
        if layers is not None:
            if self not in layers:
                return
        elif not self.visible:
            return
        if self is previewing:  # not solo though - we show the effect of that
            mode = DEFAULT_MODE
            opacity = 1.0
        self._surface.composite_tile(
            dst, dst_has_alpha, tx, ty,
            mipmap_level=mipmap_level,
            opacity=opacity, mode=mode
        )

    def render_as_pixbuf(self, *rect, **kwargs):
        """Renders this layer as a pixbuf"""
        return self._surface.render_as_pixbuf(*rect, **kwargs)

    ## Translating

    def get_move(self, x, y):
        """Get a translation/move object for this layer

        :param x: Model X position of the start of the move
        :param y: Model X position of the start of the move
        :returns: A move object

        Subclasses should extend this minimal implementation to provide
        additional functionality for moving things other than the surface tiles
        around.
        """
        return self._surface.get_move(x, y)

    ## Saving

    @lib.fileutils.via_tempfile
    def save_as_png(self, filename, *rect, **kwargs):
        """Save to a named PNG file

        :param filename: filename to save to
        :param *rect: rectangle to save, as a 4-tuple
        :param **kwargs: passed to pixbufsurface.save_as_png()
        :rtype: Gdk.Pixbuf
        """
        self._surface.save_as_png(filename, *rect, **kwargs)

    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        """Saves the layer's data into an open OpenRaster ZipFile"""
        rect = self.get_bbox()
        return self._save_rect_to_ora(orazip, tmpdir, "layer", path,
                                      frame_bbox, rect, **kwargs)

    @staticmethod
    def _make_refname(prefix, path, suffix, sep='-'):
        """Internal: standardized filename for something wiith a path"""
        assert "." in suffix
        path_ref = sep.join([("%02d" % (n,)) for n in path])
        if not suffix.startswith("."):
            suffix = sep + suffix
        return "".join([prefix, sep, path_ref, suffix])

    def _save_rect_to_ora(self, orazip, tmpdir, prefix, path,
                          frame_bbox, rect, **kwargs):
        """Internal: saves a rectangle of the surface to an ORA zip"""
        # Write PNG data via a tempfile
        pngname = self._make_refname(prefix, path, ".png")
        pngpath = os.path.join(tmpdir, pngname)
        t0 = time.time()
        self._surface.save_as_png(pngpath, *rect, **kwargs)
        t1 = time.time()
        logger.debug('%.3fs surface saving %r', t1-t0, pngname)
        # Archive and remove
        storepath = "data/%s" % (pngname,)
        orazip.write(pngpath, storepath)
        os.remove(pngpath)
        # Return details
        data_bbox = tuple(rect)
        data_x, data_y = data_bbox[0:2]
        frame_x, frame_y = frame_bbox[0:2]
        elem = self._get_stackxml_element(
            "layer",
            x=(data_x - frame_x),
            y=(data_y - frame_y),
        )
        elem.attrib["src"] = storepath
        return elem

    ## Painting symmetry axis

    def set_symmetry_state(self, active, center_x):
        """Set the surface's painting symmetry axis and active flag.

        See `LayerBase.set_symmetry_state` for the params.
        """
        self._surface.set_symmetry_state(bool(active), float(center_x))

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
        self._surface.trim(rect)


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

    IS_FILLABLE = False
    IS_PAINTABLE = False
    ALLOWED_SUFFIXES = []

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
        tempdir = self.root.doc.tempdir
        ext = self.ALLOWED_SUFFIXES[0]
        rev0_fd, rev0_filename = tempfile.mkstemp(suffix=ext, dir=tempdir)
        self.write_blank_backing_file(rev0_filename, **self._keywords)
        os.close(rev0_fd)
        self._workfile = _ManagedFile(rev0_filename)
        logger.info("Loading new blank working file from %r", rev0_filename)
        self.load_surface_from_pixbuf_file(rev0_filename, x=self._x, y=self._y)
        redraw_bbox = self.get_full_redraw_bbox()
        self._content_changed(*redraw_bbox)

    def write_blank_backing_file(self, filename, **kwargs):
        """Write out the zeroth backing file revision.

        :param filename: name of the file to write.
        :param **kwargs: all construction params, including x and y.

        This operation is deferred until the file is needed.

        """
        raise NotImplementedError

    def load_from_openraster(self, orazip, elem, tempdir, feedback_cb,
                             x=0, y=0, **kwargs):
        """Loads layer data and attrs from an OpenRaster zipfile"""
        # Load layer flags and raster data
        super(FileBackedLayer, self) \
            .load_from_openraster(orazip, elem, tempdir, feedback_cb,
                                  x=x, y=y, extract_and_keep=True, **kwargs)
        # Use the extracted file as the zero revision, and record layer
        # working parameters.
        attrs = elem.attrib
        src = attrs.get("src", None)
        src_rootname, src_ext = os.path.splitext(src)
        src_ext = src_ext.lower()
        tmp_filename = os.path.join(tempdir, src)
        if not os.path.exists(tmp_filename):
            raise lib.layer.error.LoadingFailed(
                "tmpfile missing after extract_and_keep: %r"
                % (tmp_filename,),
            )
        self._workfile = _ManagedFile(tmp_filename, move=True, dir=tempdir)
        self._x = x + int(attrs.get('x', 0))
        self._y = y + int(attrs.get('y', 0))

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
        surface_move = super(FileBackedLayer, self).get_move(x, y)
        return FileBackedLayerMove(self, surface_move)

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
        data_x, data_y = (self._x, self._y)
        frame_x, frame_y = frame_bbox[0:2]
        elem = self._get_stackxml_element(
            "layer",
            x=(data_x - frame_x),
            y=(data_y - frame_y),
        )
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

    ## Editing via external apps

    def new_external_edit_tempfile(self):
        """Get a tempfile for editing in an external app"""
        if self.root is None:
            return
        self._ensure_valid_working_file()
        self._edit_tempfile = deepcopy(self._workfile)
        return unicode(self._edit_tempfile)

    def load_from_external_edit_tempfile(self, tempfile_path):
        """Load content from an external-edit tempfile"""
        redraw_bboxes = []
        redraw_bboxes.append(self.get_full_redraw_bbox())
        x = self._x
        y = self._y
        self.load_surface_from_pixbuf_file(tempfile_path, x=x, y=y)
        redraw_bboxes.append(self.get_full_redraw_bbox())
        self._workfile = _ManagedFile(tempfile_path, copy=True)
        self._content_changed_aggregated(redraw_bboxes)


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


class FileBackedLayerMove (object):
    """Move object wrapper for file-backed layers"""

    def __init__(self, layer, surface_move):
        super(FileBackedLayerMove, self).__init__()
        self._wrapped = surface_move
        self._layer = layer
        self._start_x = layer._x
        self._start_y = layer._y

    def update(self, dx, dy):
        self._layer._x = int(round(self._start_x + dx))
        self._layer._y = int(round(self._start_y + dy))
        self._wrapped.update(dx, dy)

    def cleanup(self):
        self._wrapped.cleanup()

    def process(self, n=200):
        return self._wrapped.process(n)


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
        raise NotImplementedError("Under Python 2.x, use unicode()")

    def __unicode__(self):
        file_path = os.path.join(self._dir, self._basename)
        assert isinstance(file_path, unicode)
        return file_path

    def __repr__(self):
        return "_ManagedFile(%r)" % (self,)

    def __del__(self):
        try:
            file_path = unicode(self)
        except:
            logger.warning("_ManagedFile: cleanup of incomplete object, file "
                           "may still exist on disk")
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

    def __init__(self, bg, **kwargs):
        if isinstance(bg, tiledsurface.Background):
            surface = bg
        else:
            surface = tiledsurface.Background(bg)
        super(BackgroundLayer, self).__init__(name=u"background",
                                              surface=surface, **kwargs)
        self.locked = False
        self.visible = True
        self.opacity = 1.0

    def save_snapshot(self):
        raise NotImplementedError("BackgroundLayer cannot be snapshotted yet")

    def load_snapshot(self):
        raise NotImplementedError("BackgroundLayer cannot be snapshotted yet")

    def set_surface(self, surface):
        """Sets the surface from a tiledsurface.Background"""
        assert isinstance(surface, tiledsurface.Background)
        self._surface = surface

    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        # Save as a regular layer for other apps.
        # Background surfaces repeat, so just the bit filling the frame.
        elem = self._save_rect_to_ora(
            orazip, tmpdir, "background", path,
            frame_bbox, frame_bbox, **kwargs
        )

        # Also save as single pattern (with corrected origin)
        x0, y0 = frame_bbox[0:2]
        x, y, w, h = self.get_bbox()
        rect = (x+x0, y+y0, w, h)

        pngname = self._make_refname("background", path, "tile.png")
        tmppath = os.path.join(tmpdir, pngname)
        t0 = time.time()
        self._surface.save_as_png(tmppath, *rect, **kwargs)
        t1 = time.time()
        storename = 'data/%s' % (pngname,)
        logger.debug('%.3fs surface saving %s', t1 - t0, storename)
        orazip.write(tmppath, storename)
        os.remove(tmppath)
        elem.attrib['background_tile'] = storename
        return elem


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

    #TRANSLATORS: Short default name for vector (SVG/Inkscape) layers
    DEFAULT_NAME = _(u"Vector Layer")

    ALLOWED_SUFFIXES = [".svg"]

    def get_icon_name(self):
        return "mypaint-layer-vector-symbolic"

    def write_blank_backing_file(self, filename, **kwargs):
        N = tiledsurface.N
        x = kwargs.get("x", 0)
        y = kwargs.get("y", 0)
        outline = kwargs.get("outline")
        if outline:
            outline = [(px-x, py-y) for (px, py) in outline]
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
        fp = open(filename, 'wb')
        fp.write(svg)
        fp.flush()
        fp.close()


class FallbackBitmapLayer (FileBackedLayer):
    """An unpaintable, fallback bitmap layer"""

    def get_icon_name(self):
        return "mypaint-layer-fallback-symbolic"

    #TRANSLATORS: Short default name for renderable fallback layers
    DEFAULT_NAME = _(u"Unknown Bitmap Layer")

    #: Any suffix is allowed, no preference for defaults
    ALLOWED_SUFFIXES = [""]


class FallbackDataLayer (FileBackedLayer):
    """An unpaintable, fallback, non-bitmap layer"""

    def get_icon_name(self):
        return "mypaint-layer-fallback-symbolic"

    #TRANSLATORS: Short default name for non-renderable fallback layers
    DEFAULT_NAME = _(u"Unknown Data Layer")

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


class PaintingLayer (SurfaceBackedLayer, core.ExternallyEditable):
    """A paintable, bitmap layer

    Painting layers add a strokemap to the base implementation. The
    stroke map is a stack of `strokemap.StrokeShape` objects in painting
    order, allowing strokes and their associated brush and color
    information to be picked from the canvas.
    """

    ## Class constants

    IS_PAINTABLE = True
    IS_FILLABLE = True
    ALLOWED_SUFFIXES = [".png"]

    #TRANSLATORS: Default name for new normal, paintable layers
    DEFAULT_NAME = _(u"Layer")

    ## Initializing & resetting

    def __init__(self, **kwargs):
        super(PaintingLayer, self).__init__(**kwargs)
        self._external_edit = None
        #: Stroke map.
        #: List of strokemap.StrokeShape instances (not stroke.Stroke),
        #: ordered by depth.
        self.strokes = []

    def clear(self):
        """Clear both the surface and the strokemap"""
        super(PaintingLayer, self).clear()
        self.strokes = []

    def load_from_surface(self, surface):
        """Load the surface image's tiles from another surface"""
        super(PaintingLayer, self).load_from_surface(surface)
        self.strokes = []

    def load_from_openraster(self, orazip, elem, tempdir, feedback_cb,
                             x=0, y=0, **kwargs):
        """Loads layer flags, PNG data, and strokemap from a .ora zipfile"""
        # Load layer tile data and flags
        super(PaintingLayer, self).load_from_openraster(
            orazip,
            elem,
            tempdir,
            feedback_cb,
            x=x, y=y,
            **kwargs)
        # Strokemap too
        attrs = elem.attrib
        x += int(attrs.get('x', 0))
        y += int(attrs.get('y', 0))
        strokemap_name = attrs.get('mypaint_strokemap_v2', None)
        if strokemap_name is not None:
            t2 = time.time()
            sio = StringIO(orazip.read(strokemap_name))
            self.load_strokemap_from_file(sio, x, y)
            sio.close()
            t3 = time.time()
            logger.debug('%.3fs loading strokemap %r',
                         t3 - t2, strokemap_name)

    ## Flood fill

    def flood_fill(self, x, y, color, bbox, tolerance, dst_layer=None):
        """Fills a point on the surface with a color

        :param x: Starting point X coordinate
        :param y: Starting point Y coordinate
        :param color: an RGB color
        :type color: tuple
        :param bbox: Bounding box: limits the fill
        :type bbox: lib.helpers.Rect or equivalent 4-tuple
        :param tolerance: how much filled pixels are permitted to vary
        :type tolerance: float [0.0, 1.0]
        :param dst_layer: Optional target layer (default is self!)
        :type dst_layer: SurfaceBackedLayer

        The `tolerance` parameter controls how much pixels are permitted to
        vary from the starting color.  We use the 4D Euclidean distance from
        the starting point to each pixel under consideration as a metric,
        scaled so that its range lies between 0.0 and 1.0.

        The default target layer is `self`. This method invalidates the filled
        area of the target layer's surface, queueing a redraw if it is part of
        a visible document.
        """
        if dst_layer is None:
            dst_layer = self
        self._surface.flood_fill(x, y, color, bbox, tolerance,
                                 dst_surface=dst_layer._surface)

    ## Painting

    def stroke_to(self, brush, x, y, pressure, xtilt, ytilt, dtime):
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

        This method renders zero or more dabs to the surface of this layer,
        but does not affect the strokemap. Use this for the incremental
        painting of segments of a stroke sorresponding to single input events.
        The return value decides whether to finalize the lib.stroke.Stroke
        which is currently recording the user's input, and begin recording a
        new one.
        """
        self._surface.begin_atomic()
        split = brush.stroke_to(
            self._surface.backend, x, y,
            pressure, xtilt, ytilt, dtime
        )
        self._surface.end_atomic()
        return split

    def render_stroke(self, stroke):
        """Render a whole captured stroke to the canvas

        :param stroke: The stroke to render
        :type stroke: lib.stroke.Stroke
        """
        stroke.render(self._surface)

    def add_stroke_shape(self, stroke, before):
        """Adds a rendered stroke's shape to the strokemap

        :param stroke: the stroke sequence which has been rendered
        :type stroke: lib.stroke.Stroke
        :param before: layer snapshot taken before the stroke started
        :type before: lib.layer.PaintingLayerSnapshot

        The StrokeMap is a stack of lib.strokemap.StrokeShape objects which
        encapsulate the shape of a rendered stroke, and the brush settings
        which were used to render it.  The shape of the rendered stroke is
        determined by visually diffing snapshots taken before the stroke
        started and now.
        """
        shape = lib.strokemap.StrokeShape()
        after_sshot = self._surface.save_snapshot()
        shape.init_from_snapshots(before.surface_sshot, after_sshot)
        shape.brush_string = stroke.brush_settings
        self.strokes.append(shape)

    ## Snapshots

    def save_snapshot(self):
        """Snapshots the state of the layer and its strokemap for undo"""
        return PaintingLayerSnapshot(self)

    ## Translating

    def get_move(self, x, y):
        """Get an interactive move object for the surface and its strokemap"""
        surface_move = super(PaintingLayer, self).get_move(x, y)
        return PaintingLayerMove(self, surface_move)

    ## Trimming

    def trim(self, rect):
        """Trim the layer and its strokemap"""
        super(PaintingLayer, self).trim(rect)
        empty_strokes = []
        for stroke in self.strokes:
            if not stroke.trim(rect):
                empty_strokes.append(stroke)
        for stroke in empty_strokes:
            logger.debug("Removing emptied stroke %r", stroke)
            self.strokes.remove(stroke)

    ## Strokemap

    def load_strokemap_from_file(self, f, translate_x, translate_y):
        assert not self.strokes
        brushes = []
        N = tiledsurface.N
        x = int(translate_x//N) * N
        y = int(translate_y//N) * N
        dx = translate_x % N
        dy = translate_y % N
        while True:
            t = f.read(1)
            if t == 'b':
                length, = struct.unpack('>I', f.read(4))
                tmp = f.read(length)
                brushes.append(zlib.decompress(tmp))
            elif t == 's':
                brush_id, length = struct.unpack('>II', f.read(2*4))
                stroke = lib.strokemap.StrokeShape()
                tmp = f.read(length)
                stroke.init_from_string(tmp, x, y)
                stroke.brush_string = brushes[brush_id]
                # Translate non-aligned strokes
                if (dx, dy) != (0, 0):
                    stroke.translate(dx, dy)
                self.strokes.append(stroke)
            elif t == '}':
                break
            else:
                assert False, 'invalid strokemap'

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

    def _save_strokemap_to_file(self, f, translate_x, translate_y):
        brush2id = {}
        for stroke in self.strokes:
            s = stroke.brush_string
            # save brush (if not already known)
            if s not in brush2id:
                brush2id[s] = len(brush2id)
                s = zlib.compress(s)
                f.write('b')
                f.write(struct.pack('>I', len(s)))
                f.write(s)
            # save stroke
            s = stroke.save_to_string(translate_x, translate_y)
            f.write('s')
            f.write(struct.pack('>II', brush2id[stroke.brush_string], len(s)))
            f.write(s)
        f.write('}')

    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        """Save the strokemap too, in addition to the base implementation"""
        # Save the layer normally

        elem = super(PaintingLayer, self).save_to_openraster(
            orazip, tmpdir, path,
            canvas_bbox, frame_bbox, **kwargs
        )
        # Store stroke shape data too
        x, y, w, h = self.get_bbox()
        sio = StringIO()
        t0 = time.time()
        self._save_strokemap_to_file(sio, -x, -y)
        t1 = time.time()
        data = sio.getvalue()
        sio.close()
        datname = self._make_refname("layer", path, "strokemap.dat")
        logger.debug("%.3fs strokemap saving %r", t1-t0, datname)
        storepath = "data/%s" % (datname,)
        helpers.zipfile_writestr(orazip, storepath, data)
        # Return details
        elem.attrib['mypaint_strokemap_v2'] = storepath
        return elem

    ## Type-specific stuff

    def get_icon_name(self):
        return "mypaint-layer-painting-symbolic"

    ## Editing via external apps

    def new_external_edit_tempfile(self):
        """Get a tempfile for editing in an external app"""
        # Uniquely named tempfile. Will be overwritten.
        if not self.root:
            return
        tempdir = self.root.doc.tempdir
        tmp_fd, tmp_filename = tempfile.mkstemp(suffix=".png", dir=tempdir)
        tmp_filename = unicode(tmp_filename)
        os.close(tmp_fd)
        # Overwrite, saving only the data area.
        # Record the data area for later.
        rect = self.get_bbox()
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
        self._content_changed_aggregated(redraw_bboxes)


class PaintingLayerSnapshot (SurfaceBackedLayerSnapshot):
    """Snapshot subclass for painting layers"""

    def __init__(self, layer):
        super(PaintingLayerSnapshot, self).__init__(layer)
        self.strokes = layer.strokes[:]

    def restore_to_layer(self, layer):
        super(PaintingLayerSnapshot, self).restore_to_layer(layer)
        layer.strokes = self.strokes[:]


class PaintingLayerMove (object):
    """Move object wrapper for painting layers"""

    def __init__(self, layer, surface_move):
        super(PaintingLayerMove, self).__init__()
        self._wrapped = surface_move
        self._layer = layer
        self._final_dx = 0
        self._final_dy = 0

    def update(self, dx, dy):
        self._final_dx = dx
        self._final_dy = dy
        return self._wrapped.update(dx, dy)

    def cleanup(self):
        self._wrapped.cleanup()
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

    def process(self, n=200):
        return self._wrapped.process(n)


## Module testing


def _test():
    """Run doctest strings"""
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _test()
