# This file is part of MyPaint.
# Copyright (C) 2007-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

import os
import sys
import zipfile
import tempfile
import time
import traceback
from os.path import join
from cStringIO import StringIO
import xml.etree.ElementTree as ET
from warnings import warn
import logging
logger = logging.getLogger(__name__)

from gi.repository import Gdk
from gi.repository import GdkPixbuf
from gi.repository import GObject

import numpy
from gettext import gettext as _

import helpers
import tiledsurface
import pixbufsurface
import mypaintlib
import command
import stroke
import layer
import brush


## Module constants

# Sizes
N = tiledsurface.N
LOAD_CHUNK_SIZE = 64*1024

# Compositing
from layer import DEFAULT_COMPOSITE_OP
from layer import VALID_COMPOSITE_OPS


## Class defs

class SaveLoadError(Exception):
    """Expected errors on loading or saving

    Covers stuff like missing permissions or non-existing files.

    """
    pass


class DeprecatedAPIWarning (UserWarning):
    pass


class Document (object):
    """In-memory representation of everything to be worked on & saved

    This is the "model" in the Model-View-Controller design for the drawing
    canvas. The View mostly resides in `gui.tileddrawwidget`, and the
    Controller is mostly in `gui.document` and `gui.canvasevent`.

    The model contains everything that the user would want to save. It is
    possible to use the model without any GUI attached (see ``../tests/``).
    """
    # Please note the following difficulty with the undo stack:
    #
    #   Most of the time there is an unfinished (but already rendered)
    #   stroke pending, which has to be turned into a command.Action
    #   or discarded as empty before any other action is possible.
    #   (split_stroke)

    ## Class constants

    TEMPDIR_STUB_NAME = "mypaint"

    ## Initialization and cleanup

    def __init__(self, brushinfo=None, painting_only=False):
        """Initialize

        :param brushinfo: the lib.brush.BrushInfo instance to use
        :param painting_only: only use painting layers

        If painting_only is true, then no tempdir will be created by the
        document when it is initialized or cleared.
        """
        object.__init__(self)
        if not brushinfo:
            brushinfo = brush.BrushInfo()
            brushinfo.load_defaults()
        self._layers = layer.RootLayerStack(self)
        self.brush = brush.Brush(brushinfo)
        self.brush.brushinfo.observers.append(self.brushsettings_changed_cb)
        self.stroke = None
        self.canvas_observers = []  #: See `layer_modified_cb()`
        self.stroke_observers = [] #: See `split_stroke()`
        self.doc_observers = [] #: See `call_doc_observers()`
        self.frame_observers = []
        self.symmetry_observers = []  #: See `set_symmetry_axis()`
        self._symmetry_axis = None
        self.command_stack = command.CommandStack()
        self._painting_only = painting_only
        self._tempdir = None

        self._frame = [0, 0, 0, 0]
        self._frame_enabled = False

        # Backgrounds for rendering
        blank_arr = numpy.zeros((N, N, 4), dtype='uint16')
        self._blank_bg_surface = tiledsurface.Background(blank_arr)

        # Compatibility
        self.layers = _LayerStackMapping(self)

        self.clear(True)


    ## Layer stack access


    @property
    def layer_stack(self):
        return self._layers
        # TODO: rename this to just "layers" one day.


    ## Backwards API compat

    @property
    def layer_idx(self):
        return self.layers.layer_idx

    @layer_idx.setter
    def layer_idx(self, value):
        self.layers.layer_idx = value

    def get_current_layer(self):
        warn("Use doc.layer_stack.get_current() instead",
             DeprecatedAPIWarning, stacklevel=2)
        return self._layers.get_current()

    def get_background_visible(self):
        warn("Use doc.layer_stack.background_visible instead",
             DeprecatedAPIWarning, stacklevel=2)
        return self._layers.get_background_visible()

    def set_background(self, *args, **kwargs):
        warn("Use doc.layer_stack.set_background instead",
             DeprecatedAPIWarning, stacklevel=2)
        self._layers.set_background(*args, **kwargs)

    @property
    def layer(self):
        """Compatibility hack - access the current layer"""
        warn("Use doc.layer_stack.current instead",
             DeprecatedAPIWarning, stacklevel=2)
        return self._layers.current

    ## Working-doc tempdir


    def _create_tempdir(self):
        """Internal: creates the working-document tempdir"""
        if self._painting_only:
            return
        assert self._tempdir is None
        tempdir = tempfile.mkdtemp(self.TEMPDIR_STUB_NAME)
        if not isinstance(tempdir, unicode):
            tempdir = tempdir.decode(sys.getfilesystemencoding())
        logger.debug("Created working-doc tempdir %r", tempdir)
        self._tempdir = tempdir


    def _cleanup_tempdir(self):
        """Internal: recursively delete the working-document tempdir"""
        if self._painting_only:
            return
        assert self._tempdir is not None
        tempdir = self._tempdir
        self._tempdir = None
        for root, dirs, files in os.walk(tempdir, topdown=False):
            for name in files:
                tempfile = os.path.join(root, name)
                try:
                    os.remove(tempfile)
                except OSError, err:
                    logger.warning("Cannot remove %r: %r", tempfile, err)
            for name in dirs:
                subtemp = os.path.join(root, name)
                try:
                    os.rmdir(subtemp)
                except OSError, err:
                    logger.warning("Cannot rmdir %r: %r", subtemp, err)
        try:
            os.rmdir(tempdir)
        except OSError, err:
            logger.warning("Cannot rmdir %r: %r", subtemp, err)
        if os.path.exists(tempdir):
            logger.error("Failed to remove working-doc tempdir %r", tempdir)
        else:
            logger.debug("Successfully removed working-doc tempdir %r", tempdir)


    def cleanup(self):
        """Cleans up any persistent state belonging to the document.

        Currently this just removes the working-document tempdir. This method
        is called by the main app's exit routine after confirmation.
        """
        self._cleanup_tempdir()


    ## Document frame


    def get_frame(self):
        return self._frame


    def set_frame(self, frame, user_initiated=False):
        x, y, w, h = frame
        self.update_frame(x=x, y=y, width=w, height=h,
                          user_initiated=user_initiated)


    frame = property(get_frame, set_frame)


    def update_frame(self, x=None, y=None, width=None, height=None,
                     user_initiated=False):
        """Update parts of the frame"""
        frame = [x, y, width, height]
        if user_initiated:
            if isinstance(self.get_last_command(), command.UpdateFrame):
                self.update_last_command(frame=frame)
            else:
                self.do(command.UpdateFrame(self, frame))
        else:
            for i, var in enumerate([x, y, width, height]):
                if var is not None:
                    self._frame[i] = int(var)
            self.call_frame_observers()


    def get_frame_enabled(self):
        return self._frame_enabled


    def set_frame_enabled(self, enabled, user_initiated=False):
        if self._frame_enabled == bool(enabled):
            return
        if user_initiated:
            self.do(command.SetFrameEnabled(self, enabled))
        else:
            self._frame_enabled = bool(enabled)
            self.call_frame_observers()


    frame_enabled = property(get_frame_enabled)


    def set_frame_to_current_layer(self, user_initiated=False):
        x, y, w, h = self.get_current_layer().get_bbox()
        self.update_frame(x, y, w, h, user_initiated=user_initiated)


    def set_frame_to_document(self, user_initiated=False):
        x, y, w, h = self.get_bbox()
        self.update_frame(x, y, w, h, user_initiated=user_initiated)


    def trim_layer(self):
        """Trim the current layer to the extent of the document frame

        This has no effect if the frame is not currently enabled.

        """
        if not self._frame_enabled:
            return
        self.do(command.TrimLayer(self))


    ## Observer convenience methods


    def call_frame_observers(self):
        for func in self.frame_observers:
            func()


    def call_doc_observers(self):
        """Announce major structural changes via `doc_observers`.

        This is invoked to announce major structural changes such as the layers
        changing or a new document being loaded. The callbacks in the list are
        invoked with a single argument, `self`.

        """
        for f in self.doc_observers:
            f(self)
        return True


    ## Symmetry axis


    def get_symmetry_axis(self):
        """Gets the active painting symmetry X axis value.
        """
        return self._symmetry_axis


    def set_symmetry_axis(self, x):
        """Sets the active painting symmetry X axis value.

        A value of `None` inactivates symmetrical painting. After setting, all
        registered `symmetry_observers` are called without arguments.
        """
        # TODO: make this undoable?
        for layer in self._layers.deepiter():
            layer.set_symmetry_axis(x)
        self._symmetry_axis = x
        for func in self.symmetry_observers:
            func()


    def clear(self, init=False):
        """Clears everything, and resets the command stack

        :param init: Set to true to suppress notification of the
          canvas_observers (used during init).

        This results in an empty layers stack, no undo history, and a new empty
        working-document temp directory.
        """
        self.split_stroke()
        self.set_symmetry_axis(None)
        if not init:
            bbox = self.get_bbox()
        # Clean up any persistent state belonging to the last load
        if self._tempdir is not None:
            self._cleanup_tempdir()
        self._create_tempdir()
        # throw everything away, including undo stack
        self.command_stack.clear()
        self._layers.clear()
        self.add_layer(0)
        # disallow undo of the first layer
        self.command_stack.clear()
        self.unsaved_painting_time = 0.0

        if not init:
            for f in self.canvas_observers:
                f(*bbox)

        self.call_doc_observers()



    def split_stroke(self):
        """Splits the current stroke, announcing the newly stacked stroke

        The stroke being drawn is pushed onto to the command stack and the
        callbacks in the list `self.stroke_observers` are invoked with two
        arguments: the newly completed stroke, and the brush used. The brush
        argument is a temporary read-only convenience object.

        This is called every so often when drawing a single long brushstroke on
        input to allow parts of a long line to be undone.

        """
        if not self.stroke:
            return
        self.stroke.stop_recording()
        if not self.stroke.empty:
            cmd = command.Stroke(self, self.stroke,
                                 self.snapshot_before_stroke)
            self.command_stack.do(cmd)
            del self.snapshot_before_stroke
            self.unsaved_painting_time += self.stroke.total_painting_time
            for f in self.stroke_observers:
                f(self.stroke, self.brush)
        self.stroke = None


    def brushsettings_changed_cb(self, settings, lightweight_settings=set([
            'radius_logarithmic', 'color_h', 'color_s', 'color_v',
            'opaque', 'hardness', 'slow_tracking', 'slow_tracking_per_dab'
            ])):
        # The lightweight brush settings are expected to change often in
        # mid-stroke e.g. by heavy keyboard usage. If only those change, we
        # don't create a new undo step. (And thus also no separate pickable
        # stroke in the strokemap.)
        if settings - lightweight_settings:
            self.split_stroke()

    def select_layer(self, index=None, path=None, layer=None,
                     user_initiated=True):
        """Selects a layer, and notifies about it

        If user_initiated is false, selection and notification is handled here.
        This form is used for preserving the selection in the GUI by certain
        internal mechanisms which permute the layer stacking order. To keep
        the GUI layer view's selection happy in this case, noninteractive calls
        queue the selection and notification so that it doesn't process in the
        same event as the GUI's internal selection manipulation.

        If user_initiated is true, the selection and notification is performed
        wrapped as an undoable command.
        """
        if user_initiated:
            self.do(command.SelectLayer(self, index=index, path=path, layer=layer))
        else:
            layers = self.layer_stack
            sel_path = layers.canonpath(index=index, path=path, layer=layer)
            GObject.idle_add(self.__select_layer_path_and_notify, sel_path)


    def __select_layer_path_and_notify(self, path):
        layers = self.layer_stack
        layers.set_current_path(path)
        self.call_doc_observers()


    ## Layer (x, y) position


    def record_layer_move(self, layer, dx, dy):
        """Records that a layer has moved"""
        layer_idx = self.layers.index(layer)
        self.do(command.MoveLayer(self, layer_idx, dx, dy, True))


    ## Layer stack (z) position


    def reorder_layer(self, was_idx, new_idx, select_new=False):
        """Reorder a layer by index (deprecated)"""
        # Use move_layer_in_stack() instead...
        self.do(command.ReorderSingleLayer(self, was_idx, new_idx, select_new))

    def move_layer_in_stack(self, old_path, new_path):
        """Moves a layer in the stack by path (undoable)"""
        logger.debug("move %r to %r", old_path, new_path)
        self.do(command.ReorderLayerInStack(self, old_path, new_path))


    ## Misc layer command frontends


    def duplicate_layer(self, insert_idx=None, name=''):
        self.do(command.DuplicateLayer(self, insert_idx, name))

    def clear_layer(self):
        if not self.layer.is_empty():
            self.do(command.ClearLayer(self))


    ## Drawing/painting strokes


    def stroke_to(self, dtime, x, y, pressure, xtilt, ytilt):
        """Draws a stroke to the current layer with the current brush.

        This is called by GUI code in response to motion events on the canvas -
        both with and without pressure. If enough time has elapsed,
        `split_stroke()` is called.

        :param self:
            This is an object method.
        :param float dtime:
            Floating-point number of seconds since the last call to this,
            function, for motion interpolation etc.
        :param float x:
            Document X position of the end-point of this stroke.
        :param float y:
            Document Y position of the end-point of this stroke.
        :param float pressure:
            Pressure, ranging from 0.0 to 1.0.
        :param float xtilt:
            X-axis tilt, ranging from -1.0 to 1.0.
        :param float ytilt:
            Y-axis tilt, ranging from -1.0 to 1.0.

        """

        current_layer = self._layers.current
        if not current_layer.get_paintable():
            split = True
        else:
            if not self.stroke:
                self.stroke = stroke.Stroke()
                self.stroke.start_recording(self.brush)
                self.snapshot_before_stroke = current_layer.save_snapshot()
            self.stroke.record_event(dtime, x, y, pressure, xtilt, ytilt)
            split = current_layer.stroke_to(self.brush, x, y,
                                            pressure, xtilt, ytilt, dtime)
        if split:
            self.split_stroke()


    def redo_last_stroke_with_different_brush(self, brush):
        cmd = self.get_last_command()
        if not isinstance(cmd, command.Stroke):
            return
        cmd = self.undo()
        assert isinstance(cmd, command.Stroke)
        new_stroke = cmd.stroke.copy_using_different_brush(brush)
        snapshot_before = self.layer.save_snapshot()
        new_stroke.render(self.layer._surface)
        self.do(command.Stroke(self, new_stroke, snapshot_before))


    ## Other painting/drawing


    def flood_fill(self, x, y, color, tolerance=0.1,
                   sample_merged=False, make_new_layer=False):
        """Flood-fills a point on the current layer with a colour

        :param x: Starting point X coordinate
        :param y: Starting point Y coordinate
        :param color: The RGB color to fill connected pixels with
        :type color: tuple
        :param tolerance: How much filled pixels are permitted to vary
        :type tolerance: float [0.0, 1.0]
        :param sample_merged: Use all visible layers instead of just current
        :type sample_merged: bool
        :param make_new_layer: Write output to a new layer above the current
        :type make_new_layer: bool

        Filling an infinite canvas requires limits. If the frame is enabled,
        this limits the maximum size of the fill, and filling outside the frame
        is not possible.

        Otherwise, if the entire document is empty, the limits are dynamic.
        Initially only a single tile will be filled. This can then form one
        corner for the next fill's limiting rectangle. This is a little quirky,
        but allows big areas to be filled rapidly as needed on blank layers.
        """
        bbox = helpers.Rect(*tuple(self.get_effective_bbox()))
        if not self.layer.get_fillable():
            make_new_layer = True
        if bbox.empty():
            bbox = helpers.Rect()
            bbox.x = N*int(x//N)
            bbox.y = N*int(y//N)
            bbox.w = N
            bbox.h = N
        elif not self.frame_enabled:
            bbox.expandToIncludePoint(x, y)
        cmd = command.FloodFill(self, x, y, color, bbox, tolerance,
                                sample_merged, make_new_layer)
        self.do(cmd)


    ## Graphical refresh


    def layer_modified_cb(self, *args):
        """Forwards region modify notifications (area invalidations)

        GUI code can respond to these notifications by appending callbacks to
        `self.canvas_observers`. Each callback is invoked with the bounding box
        of the changed region: ``cb(x, y, w, h)``, or ``cb(0, 0, 0, 0)`` to
        denote that everything needs to be redrawn.

        See also: `invalidate_all()`.

        """
        # for now, any layer modification is assumed to be visible
        for f in self.canvas_observers:
            f(*args)


    def invalidate_all(self):
        """Marks everything as invalid.

        Invokes the callbacks in `self.canvas_observers` passing the notation
        for "everything" as arguments. See `layer_modified_cb()` for details.

        """
        for f in self.canvas_observers:
            f(0, 0, 0, 0)


    ## Undo/redo command stack


    def undo(self):
        self.split_stroke()
        while 1:
            cmd = self.command_stack.undo()
            if not cmd or not cmd.automatic_undo:
                return cmd

    def redo(self):
        self.split_stroke()
        while 1:
            cmd = self.command_stack.redo()
            if not cmd or not cmd.automatic_undo:
                return cmd

    def do(self, cmd):
        self.split_stroke()
        self.command_stack.do(cmd)


    def update_last_command(self, **kwargs):
        self.split_stroke()
        return self.command_stack.update_last_command(**kwargs)


    def get_last_command(self):
        self.split_stroke()
        return self.command_stack.get_last_command()


    ## Utility methods

    def get_bbox(self):
        """Returns the dynamic bounding box of the document.

        This is currently the union of all the bounding boxes of all of the
        layers. It disregards the user-chosen frame.

        """
        res = helpers.Rect()
        for layer in self.layer_stack.deepiter():
            # OPTIMIZE: only visible layers...
            # careful: currently saving assumes that all layers are included
            bbox = layer.get_bbox()
            res.expandToIncludeRect(bbox)
        return res


    def get_effective_bbox(self):
        """Return the effective bounding box of the document.

        If the frame is enabled, this is the bounding box of the frame, 
        else the (dynamic) bounding box of the document.

        """
        return self.get_frame() if self.frame_enabled else self.get_bbox()


    ## Rendering tiles

    def blit_tile_into( self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                        layers=None, background=None ):
        """Blit composited tiles into a destination surface"""
        self.layer_stack.blit_tile_into( dst, dst_has_alpha, tx, ty,
                                         mipmap_level, layers=layers )

    def get_rendered_image_behind_current_layer(self, tx, ty):
        dst = numpy.empty((N, N, 4), dtype='uint16')
        l = self.layers[0:self.layer_idx]
        self.blit_tile_into(dst, False, tx, ty, layers=l)
        return dst


    ## More layer stack commands


    def add_layer(self, insert_idx, name=''):
        insert_path = self.layers.get_insert_path(insert_idx)
        self.do(command.AddLayer(self, insert_path, name=name))


    def remove_layer(self,layer=None):
        self.do(command.RemoveLayer(self, layer))


    def rename_layer(self, layer, name):
        self.do(command.RenameLayer(self, name, layer))

    def convert_layer_to_normal_mode(self):
        self.do(command.ConvertLayerToNormalMode(self, self.layer))

    def merge_layer_down(self):
        dst_idx = self.layer_idx - 1
        if dst_idx < 0:
            return False
        self.do(command.MergeLayer(self, dst_idx))
        return True



    ## Layer import/export


    def load_layer_from_pixbuf(self, pixbuf, x=0, y=0):
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        s = tiledsurface.Surface()
        bbox = s.load_from_numpy(arr, x, y)
        self.do(command.LoadLayer(self, s))
        return bbox


    def load_layer_from_png(self, filename, x=0, y=0, feedback_cb=None):
        s = tiledsurface.Surface()
        bbox = s.load_from_png(filename, x, y, feedback_cb)
        self.do(command.LoadLayer(self, s))
        return bbox


    ## Even more layer command frontends


    def set_layer_visibility(self, visible, layer):
        """Sets the visibility of a layer."""
        cmd = self.get_last_command()
        if isinstance(cmd, command.SetLayerVisibility) and cmd.layer is layer:
            self.update_last_command(visible=visible)
        else:
            self.do(command.SetLayerVisibility(self, visible, layer))


    def set_layer_locked(self, locked, layer):
        """Sets the input-locked status of a layer."""
        cmd = self.get_last_command()
        if isinstance(cmd, command.SetLayerLocked) and cmd.layer is layer:
            self.update_last_command(locked=locked)
        else:
            self.do(command.SetLayerLocked(self, locked, layer))


    def set_layer_opacity(self, opacity, layer=None):
        """Sets the opacity of a layer.

        If layer=None, works on the current layer.

        """
        cmd = self.get_last_command()
        if isinstance(cmd, command.SetLayerOpacity):
            self.undo()
        self.do(command.SetLayerOpacity(self, opacity, layer))


    def set_layer_compositeop(self, compositeop, layer=None):
        """Sets the compositing operator for a layer.

        If layer=None, works on the current layer.

        """
        if compositeop not in VALID_COMPOSITE_OPS:
            compositeop = DEFAULT_COMPOSITE_OP
        cmd = self.get_last_command()
        if isinstance(cmd, command.SetLayerCompositeOp):
            self.undo()
        self.do(command.SetLayerCompositeOp(self, compositeop, layer))


    ## Saving and loading


    def load_from_pixbuf(self, pixbuf):
        """Load a document from a pixbuf."""
        self.clear()
        bbox = self.load_layer_from_pixbuf(pixbuf)
        self.set_frame(bbox, user_initiated=False)


    def save(self, filename, **kwargs):
        """Save the document to a file.

        :param str filename:
            The filename to save to. The extension is used to determine format,
            and a ``save_*()`` method is chosen to perform the save.
        :param dict kwargs:
            Passed on to the chosen save method.
        :raise SaveLoadError:
            The error string will be set to something descriptive and
            presentable to the user.

        """
        self.split_stroke()
        junk, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        save = getattr(self, 'save_' + ext, self._unsupported)
        try:
            save(filename, **kwargs)
        except GObject.GError, e:
            traceback.print_exc()
            if e.code == 5:
                #add a hint due to a very consfusing error message when there is no space left on device
                raise SaveLoadError, _('Unable to save: %s\nDo you have enough space left on the device?') % e.message
            else:
                raise SaveLoadError, _('Unable to save: %s') % e.message
        except IOError, e:
            traceback.print_exc()
            raise SaveLoadError, _('Unable to save: %s') % e.strerror
        self.unsaved_painting_time = 0.0


    def load(self, filename, **kwargs):
        """Load the document from a file.

        :param str filename:
            The filename to load from. The extension is used to determine
            format, and a ``load_*()`` method is chosen to perform the load.
        :param dict kwargs:
            Passed on to the chosen loader method.
        :raise SaveLoadError:
            The error string will be set to something descriptive and
            presentable to the user.

        """
        if not os.path.isfile(filename):
            raise SaveLoadError, _('File does not exist: %s') % repr(filename)
        if not os.access(filename,os.R_OK):
            raise SaveLoadError, _('You do not have the necessary permissions to open file: %s') % repr(filename)
        junk, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        load = getattr(self, 'load_' + ext, self._unsupported)
        try:
            load(filename, **kwargs)
        except GObject.GError, e:
            traceback.print_exc()
            raise SaveLoadError, _('Error while loading: GError %s') % e
        except IOError, e:
            traceback.print_exc()
            raise SaveLoadError, _('Error while loading: IOError %s') % e
        self.command_stack.clear()
        self.unsaved_painting_time = 0.0
        self.call_doc_observers()


    def _unsupported(self, filename, *args, **kwargs):
        raise SaveLoadError, _('Unknown file format extension: %s') % repr(filename)

    def render_as_pixbuf(self, *args, **kwargs):
        return pixbufsurface.render_as_pixbuf(self, *args, **kwargs)

    def render_thumbnail(self):
        t0 = time.time()
        x, y, w, h = self.get_effective_bbox()
        if w == 0 or h == 0:
            # workaround to save empty documents
            x, y, w, h = 0, 0, tiledsurface.N, tiledsurface.N
        mipmap_level = 0
        while mipmap_level < tiledsurface.MAX_MIPMAP_LEVEL and max(w, h) >= 512:
            mipmap_level += 1
            x, y, w, h = x/2, y/2, w/2, h/2

        pixbuf = self.render_as_pixbuf(x, y, w, h, mipmap_level=mipmap_level)
        assert pixbuf.get_width() == w and pixbuf.get_height() == h
        pixbuf = helpers.scale_proportionally(pixbuf, 256, 256)
        logger.info('Rendered thumbnail in %d seconds.',
                    time.time() - t0)
        return pixbuf

    def save_png(self, filename, alpha=False, multifile=False, **kwargs):
        doc_bbox = self.get_effective_bbox()
        if multifile:
            self.save_multifile_png(filename, **kwargs)
        else:
            if alpha:
                tmp_layer = layer.PaintingLayer()
                for l in self.layers:
                    l.merge_into(tmp_layer)
                tmp_layer.save_as_png(filename, *doc_bbox)
            else:
                pixbufsurface.save_as_png(self, filename, *doc_bbox, alpha=False, **kwargs)

    def save_multifile_png(self, filename, alpha=False, **kwargs):
        prefix, ext = os.path.splitext(filename)
        # if we have a number already, strip it
        l = prefix.rsplit('.', 1)
        if l[-1].isdigit():
            prefix = l[0]
        doc_bbox = self.get_effective_bbox()
        for i, l in enumerate(self.layers):
            filename = '%s.%03d%s' % (prefix, i+1, ext)
            l.save_as_png(filename, *doc_bbox, **kwargs)

    def load_png(self, filename, feedback_cb=None):
        self.clear()
        bbox = self.load_layer_from_png(filename, 0, 0, feedback_cb)
        self.set_frame(bbox, user_initiated=False)

    @staticmethod
    def _pixbuf_from_stream(fp, feedback_cb=None):
        loader = GdkPixbuf.PixbufLoader()
        while True:
            if feedback_cb is not None:
                feedback_cb()
            buf = fp.read(LOAD_CHUNK_SIZE)
            if buf == '':
                break
            loader.write(buf)
        loader.close()
        return loader.get_pixbuf()

    def load_from_pixbuf_file(self, filename, feedback_cb=None):
        fp = open(filename, 'rb')
        pixbuf = self._pixbuf_from_stream(fp, feedback_cb)
        fp.close()
        self.load_from_pixbuf(pixbuf)

    load_jpg = load_from_pixbuf_file
    load_jpeg = load_from_pixbuf_file

    def save_jpg(self, filename, quality=90, **kwargs):
        x, y, w, h = self.get_effective_bbox()
        if w == 0 or h == 0:
            x, y, w, h = 0, 0, N, N # allow to save empty documents
        pixbuf = self.render_as_pixbuf(x, y, w, h, **kwargs)
        options = {"quality": str(quality)}
        pixbuf.savev(filename, 'jpeg', options.keys(), options.values())

    save_jpeg = save_jpg

    def save_ora(self, filename, options=None, **kwargs):
        logger.info('save_ora: %r (%r, %r)', filename, options, kwargs)
        t0 = time.time()
        tempdir = tempfile.mkdtemp('mypaint')
        if not isinstance(tempdir, unicode):
            tempdir = tempdir.decode(sys.getfilesystemencoding())
        # use .tmp extension, so we don't overwrite a valid file if there is an exception
        z = zipfile.ZipFile(filename + '.tmpsave', 'w', compression=zipfile.ZIP_STORED)
        # work around a permission bug in the zipfile library: http://bugs.python.org/issue3394
        def write_file_str(filename, data):
            zi = zipfile.ZipInfo(filename)
            zi.external_attr = 0100644 << 16
            z.writestr(zi, data)
        write_file_str('mimetype', 'image/openraster') # must be the first file
        image = ET.Element('image')
        stack = ET.SubElement(image, 'stack')
        effective_bbox = self.get_effective_bbox()
        x0, y0, w0, h0 = effective_bbox
        a = image.attrib
        a['w'] = str(w0)
        a['h'] = str(h0)

        def store_pixbuf(pixbuf, name):
            tmp = join(tempdir, 'tmp.png')
            t1 = time.time()
            pixbuf.savev(tmp, 'png', [], [])
            logger.debug('%.3fs pixbuf saving %s', time.time() - t1, name)
            z.write(tmp, name)
            os.remove(tmp)

        # Save layers
        canvas_bbox = tuple(self.get_bbox())
        frame_bbox = tuple(effective_bbox)
        for idx, l in enumerate(reversed(self.layers)):
            if l.is_empty():
                continue
            selected = (idx == self.layer_idx)
            attrs = l.save_to_openraster(z, tempdir, idx, selected,
                                         canvas_bbox, frame_bbox, **kwargs)
            el = ET.Element('layer')
            stack.append(el)
            for k, v in attrs.iteritems():
                el.attrib[k] = str(v)

        # Save background
        bglayer = self.layer_stack.background_layer
        attrs = bglayer.save_to_openraster(z, tempdir, "background", False,
                                           canvas_bbox, frame_bbox, **kwargs)
        el = ET.Element('layer')
        stack.append(el)
        for k, v in attrs.iteritems():
            el.attrib[k] = str(v)

        # preview (256x256)
        t2 = time.time()
        logger.debug('starting to render full image for thumbnail...')

        thumbnail_pixbuf = self.render_thumbnail()
        store_pixbuf(thumbnail_pixbuf, 'Thumbnails/thumbnail.png')
        logger.debug('total %.3fs spent on thumbnail', time.time() - t2)

        helpers.indent_etree(image)
        xml = ET.tostring(image, encoding='UTF-8')

        write_file_str('stack.xml', xml)
        z.close()
        os.rmdir(tempdir)
        if os.path.exists(filename):
            os.remove(filename) # windows needs that
        os.rename(filename + '.tmpsave', filename)

        logger.info('%.3fs save_ora total', time.time() - t0)

        return thumbnail_pixbuf


    def _layer_new_from_openraster(self, orazip, attrs, feedback_cb):
        """Create a new layer from an ORA zipfile, and a dict of XML attrs

        :param orazip: OpenRaster zipfile, open for extraction.
        :param attrs: Dict of stack XML <layer/> attrs.
        :param feedback_cb: 0-arg callable used for providing loader feedback.
        :returns: ``(LAYER, SELECTED)``, New layer, and whether it's selected
        :rtype: tuple

        The LAYER return value will be None if data could not be loaded for
        some reason.
        """

        # Switch the class to instantiate bsed on the filename extension.
        src = attrs.get("src", None)
        if src is None:
            logger.warning('Ignoring layer with no src attrib %r', attrs)
            return None
        src_basename, src_ext = os.path.splitext(src)
        src_ext = src_ext.lower()
        t0 = time.time()
        if src_ext == '.png':
            new_layer = layer.PaintingLayer()
        elif src_ext == '.svg':
            new_layer = layer.ExternalLayer()
        else:
            logger.warning("Unknown extension %r for %r", src_ext, src_basename)
            return (None, False)
        selected = new_layer.load_from_openraster(orazip, attrs, self._tempdir,
                                                  feedback_cb)
        t1 = time.time()
        logger.debug('%.3fs loading and converting %r layer %r',
                     t1 - t0, src_ext, src_basename)
        return (new_layer, selected)


    @staticmethod
    def _load_ora_layers_list(root, x=0, y=0):
        """Builds and returns a layers list based on OpenRaster stack.xml

        :param root: ``<stack/>`` XML node (can be a sub-stack
        :type root: xml.etree.ElementTree.Element
        :param x: X offset of the (sub-)stack
        :param y: Y offset of the (sub-)stack
        :returns: Normalized list of ``Element``s (see description)
        :rtype: list

        Members of the returned list are the etree ``Element``s for each
        source ``<layer/>``, normalized. Returned layer x and y ``attrib``s
        are all relative to the same origin.
        """
        res = []
        for item in root:
            if item.tag == 'layer':
                if 'x' in item.attrib:
                    item.attrib['x'] = int(item.attrib['x']) + x
                if 'y' in item.attrib:
                    item.attrib['y'] = int(item.attrib['y']) + y
                res.append(item)
            elif item.tag == 'stack':
                stack_x = int( item.attrib.get('x', 0) )
                stack_y = int( item.attrib.get('y', 0) )
                res += _load_ora_layers_list(item, stack_x, stack_y)
            else:
                logger.warning('ignoring unsupported tag %r', item.tag)
        return res


    def load_ora(self, filename, feedback_cb=None):
        """Loads from an OpenRaster file"""
        # TODO: move the guts of this to RootLayerStack
        logger.info('load_ora: %r', filename)
        t0 = time.time()
        tempdir = self._tempdir
        z = zipfile.ZipFile(filename)
        logger.debug('mimetype: %r', z.read('mimetype').strip())
        xml = z.read('stack.xml')
        image = ET.fromstring(xml)
        stack = image.find('stack')

        image_w = int(image.attrib['w'])
        image_h = int(image.attrib['h'])

        def get_pixbuf(filename):
            t1 = time.time()

            try:
                fp = z.open(filename, mode='r')
            except KeyError:
                # support for bad zip files (saved by old versions of the GIMP ORA plugin)
                fp = z.open(filename.encode('utf-8'), mode='r')
                logger.warning('Bad OpenRaster ZIP file. There is an utf-8 '
                               'encoded filename that does not have the '
                               'utf-8 flag set: %r', filename)

            res = self._pixbuf_from_stream(fp, feedback_cb)
            fp.close()
            logger.debug('%.3fs loading pixbuf %s', time.time() - t1, filename)
            return res


        self.layer_stack.clear()
        no_background = True

        selected_layer = None
        nloaded = 0
        for el in self._load_ora_layers_list(stack):
            a = el.attrib

            if 'background_tile' in a:
                assert no_background
                try:
                    logger.debug("background tile: %r", a['background_tile'])
                    self.layer_stack.set_background(get_pixbuf(a['background_tile']))
                    no_background = False
                    continue
                except tiledsurface.BackgroundError, e:
                    logger.warning('ORA background tile not usable: %r', e)

            llayer, selected = self._layer_new_from_openraster(z, a, feedback_cb)
            if llayer is None:
                logger.warning("Skipping empty layer")
                continue
            self.layer_stack.deepinsert([0], llayer)
            if selected:
                selected_layer = llayer
            llayer.content_observers.append(self.layer_modified_cb)
            llayer.set_symmetry_axis(self.get_symmetry_axis())
            nloaded += 1

        if nloaded == 0:
            # no assertion (allow empty documents)
            logger.error('Could not load any layer, document is empty.')
            logger.info('Adding an empty painting layer')
            dlayer = layer.PaintingLayer()
            dlayer.content_observers.append(self.layer_modified_cb)
            dlayer.set_symmetry_axis(self.get_symmetry_axis())
            self.layer_stack.deepinsert([0], dlayer)

        assert len(self.layer_stack) > 0

        # Select the topmost layer
        self.layer_stack.set_current_path([len(self.layer_stack)-1])

        # Set the selected layer index
        if selected_layer is not None:
            for epath, elayer in self.layer_stack.deepenumerate():
                if elayer is selected_layer:
                    self.set_current_path(epath)
                    break

        # Set the frame size to that saved in the image.
        self.update_frame(x=0, y=0, width=image_w, height=image_h,
                          user_initiated=False)

        # Enable frame if the saved image size is something other than the
        # calculated bounding box. Goal: if the user saves an "infinite
        # canvas", it loads as an infinite canvas.
        bbox_c = helpers.Rect(x=0, y=0, w=image_w, h=image_h)
        bbox = self.get_bbox()
        frame_enab = not (bbox_c==bbox or bbox.empty() or bbox_c.empty())
        self.set_frame_enabled(frame_enab, user_initiated=False)

        z.close()

        logger.info('%.3fs load_ora total', time.time() - t0)


class _LayerStackMapping (object):
    """Temporary compatibility hack"""

    ## Construction

    def __init__(self, doc):
        super(_LayerStackMapping, self).__init__()
        self._doc = doc
        self._layer_paths = []
        self._layer_idx = 0
        doc.doc_observers.append(self._doc_structure_changed)

    ## Updates

    def _doc_structure_changed(self, doc):
        current_path = self._doc.layer_stack.get_current_path()
        self._layer_paths[:] = []
        self._layer_idx = 0
        i = 0
        for path, layer in self._doc.layer_stack.deepenumerate():
            self._layer_paths.append(path)
            if path == current_path:
                self._layer_idx = i
            i += 1


    ## Current-layer index


    @property
    def layer_idx(self):
        warn("Use doc.layer_stack.current_path instead",
             DeprecatedAPIWarning, stacklevel=3)
        return self._layer_idx

    @layer_idx.setter
    def layer_idx(self, i):
        warn("Use doc.layer_stack.current_path instead",
             DeprecatedAPIWarning, stacklevel=3)
        i = helpers.clamp(int(i), 0, max(0, len(self._layer_paths)-1))
        path = self._layer_paths[i]
        self._doc.layer_stack.current_path = path
        self._layer_idx = i


    ## Sequence emulation

    def __iter__(self): # "for l in doc.layers: ..." #
        warn("Use doc.layer_stack.deepiter() etc. instead",
             DeprecatedAPIWarning, stacklevel=2)
        for layer in self._doc.layer_stack.deepiter():
            yield layer

    def __len__(self): # len(doc.layers) #
        warn("Use doc.layer_stack.deepiter() etc. instead",
             DeprecatedAPIWarning, stacklevel=2)
        return len(self._layer_paths)

    def __getitem__(self, key): # doc.layers[int] #
        warn("Use doc.layer_stack instead",
             DeprecatedAPIWarning, stacklevel=2)
        path = self._getpath(key)
        layer = self._doc.layer_stack.deepget(path)
        assert layer is not None
        return layer

    def __setitem__(self, key, value):
        raise NotImplementedError


    def _getpath(self, key):
        try: key = int(key)
        except ValueError: raise TypeError, "keys must be ints"
        if key < 0:
            raise IndexError, "key out of range"
        if key >= len(self._layer_paths):
            raise IndexError, "key out of range"
        return self._layer_paths[key]


    def index(self, layer):
        warn("Use doc.layer_stack.deepindex() instead",
             DeprecatedAPIWarning, stacklevel=2)
        for i, ly in enumerate(self._doc.layer_stack.deepiter()):
            if ly is layer:
                return i
        raise ValueError, "Layer not found"


    def insert(self, index, layer):
        warn("Use doc.layer_stack.deepinsert() instead",
             DeprecatedAPIWarning, stacklevel=2)
        insert_path = self.get_insert_path(index)
        self._doc.layer_stack.deepinsert(insert_path, layer)
        self._doc_structure_changed(self._doc)

    def remove(self, layer):
        self._doc.layer_stack.deepremove(layer)
        self._doc_structure_changed(self._doc)

    def pop(self, i):
        warn("Use doc.layer_stack instead",
             DeprecatedAPIWarning, stacklevel=2)
        path = self._getpath(i)
        return self._doc.layer_stack.deeppop(path)


    def get_insert_path(self, insert_index):
        """Normalizes an insertion index to an insert path

        :param insert_index: insert index, as for `list.insert()`
        :type insert_index: int
        :return: a root-stack path suitable for deepinsert()
        :rtype: tuple
        """
        # Like list.insert(), indixes > the length always append items.
        # Let's take that to mean inserting at the top of the root stack.
        npaths = len(self._layer_paths)
        if insert_index >= npaths:
            return (len(self._doc.layer_stack),)
        # Otherwise, do the lookup thing to find a path for deepinsert().
        idx = insert_index
        if idx < 0:
            idx = max(idx, -npaths) # still negative, but now a valid index
        return self._layer_paths[idx]
