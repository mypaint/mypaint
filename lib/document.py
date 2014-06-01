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
from observable import event


## Module constants

DEFAULT_RESOLUTION = 72

# Sizes
N = tiledsurface.N
LOAD_CHUNK_SIZE = 64*1024


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

    This is the "model" in the Model-View-Controller design for the
    drawing canvas. The View mostly resides in `gui.tileddrawwidget`,
    and the Controller is mostly in `gui.document` and
    `gui.canvasevent`.

    The model contains everything that the user would want to save. It
    is possible to use the model without any GUI attached (see
    ``../tests/``).
    """
    # Please note the following difficulty with the undo stack:
    #
    #   Most of the time there is an unfinished (but already rendered)
    #   stroke pending, which has to be turned into a command.Action
    #   or discarded as empty before any other action is possible.

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
        self._layers.layer_content_changed += self._canvas_modified_cb
        self.brush = brush.Brush(brushinfo)
        self.brush.brushinfo.observers.append(self.brushsettings_changed_cb)
        self.stroke = None
        self.doc_observers = [] #: See `call_doc_observers()`
        self.frame_observers = []
        self.symmetry_observers = []  #: See `set_symmetry_axis()`
        self._symmetry_axis = None
        self.command_stack = command.CommandStack()
        self._painting_only = painting_only
        self._tempdir = None

        # Optional page area and resolution information
        self._frame = [0, 0, 0, 0]
        self._frame_enabled = False
        self._xres = None
        self._yres = None

        # Backgrounds for rendering
        blank_arr = numpy.zeros((N, N, 4), dtype='uint16')
        self._blank_bg_surface = tiledsurface.Background(blank_arr)

        # Compatibility
        self.layers = _LayerStackMapping(self)

        self.clear()

    def __repr__(self):
        bbox = self.get_bbox()
        nlayers = len(list(self.layer_stack.deepenumerate()))
        return ("<Document nlayers=%d bbox=%r paintonly=%r>" %
                (nlayers, bbox, self._painting_only))

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


    def get_resolution(self):
        """Returns the document model's nominal resolution

        The OpenRaster format saves resolution information in both vertical and
        horizontal resolutions, but MyPaint does not support this at present.
        This method returns the a unidirectional document resolution in pixels
        per inch; this is the user-chosen factor that UI controls should use
        when converting real-world measurements in frames, fonts, and other
        objects to document pixels.

        Note that the document resolution has no direct relation to screen
        pixels or printed dots.
        """
        if self._xres and self._yres:
            return max(1, max(self._xres, self._yres))
        else:
            return DEFAULT_RESOLUTION


    def set_resolution(self, res):
        """Sets the document model's nominal resolution

        The OpenRaster format saves resolution information in both vertical and
        horizontal resolutions, but MyPaint does not support this at present.
        This method sets the document resolution in pixels per inch in both
        directions.

        Note that the document resolution has no direct relation to screen
        pixels or printed dots.
        """
        if res is not None:
            res = int(res)
            res = max(1, res)
        # Maybe. Using 72 as a fake null would be pretty weird.
        #if res == DEFAULT_RESOLUTION:
        #    res = None
        self._xres = res
        self._yres = res


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
        current = self.layer_stack.current
        x, y, w, h = current.get_bbox()
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


    ## Misc actions

    def clear(self):
        """Clears everything, and resets the command stack

        This results in an empty layers stack, no undo history, and a new empty
        working-document temp directory.
        """
        self.flush_updates()
        self.set_symmetry_axis(None)
        prev_area = self.get_full_redraw_bbox()
        # Clean up any persistent state belonging to the last load
        if self._tempdir is not None:
            self._cleanup_tempdir()
        self._create_tempdir()
        # throw everything away, including undo stack
        self.command_stack.clear()
        self._layers.clear()
        self.add_layer((-1,))
        # disallow undo of the first layer
        self.command_stack.clear()
        self.unsaved_painting_time = 0.0
        # Reset frame
        self._frame = [0, 0, 0, 0]
        self._frame_enabled = False
        self._xres = None
        self._yres = None
        # Notify
        self.canvas_area_modified(*prev_area)
        self.call_frame_observers()

    def brushsettings_changed_cb(self, settings, lightweight_settings=set([
            'radius_logarithmic', 'color_h', 'color_s', 'color_v',
            'opaque', 'hardness', 'slow_tracking', 'slow_tracking_per_dab'
            ])):
        # The lightweight brush settings are expected to change often in
        # mid-stroke e.g. by heavy keyboard usage. If only those change, we
        # don't create a new undo step. (And thus also no separate pickable
        # stroke in the strokemap.)
        if settings - lightweight_settings:
            self.flush_updates()

    def select_layer(self, index=None, path=None, layer=None):
        """Selects a layer undoably"""
        layers = self.layer_stack
        sel_path = layers.canonpath(index=index, path=path, layer=layer,
                                    usecurrent=False, usefirst=True)
        self.do(command.SelectLayer(self, path=sel_path))


    ## Layer stack (z-order and grouping)

    def restack_layer(self, src_path, targ_path):
        """Moves a layer within the layer stack by path, undoably

        :param tuple src_path: path of the layer to be moved
        :param tuple targ_path: target insert path

        The source path must identify an existing layer. The target
        path must be a valid insertion path at the time this method is
        called.
        """
        logger.debug("Restack layer at %r to %r", src_path, targ_path)
        cmd = command.RestackLayer(self, src_path, targ_path)
        self.do(cmd)

    def bubble_current_layer_up(self):
        """Moves the current layer up in the stack (undoable)"""
        cmd = command.BubbleLayerUp(self)
        self.do(cmd)

    def bubble_current_layer_down(self):
        """Moves the current layer down in the stack (undoable)"""
        cmd = command.BubbleLayerDown(self)
        self.do(cmd)


    ## Misc layer command frontends

    def duplicate_current_layer(self):
        """Makes an exact copy of the current layer (undoable)"""
        self.do(command.DuplicateLayer(self))


    def clear_layer(self):
        """Clears the current layer (undoable)"""
        if not self.layer_stack.current.is_empty():
            self.do(command.ClearLayer(self))


    ## Drawing/painting strokes


    def stroke_to(self, dtime, x, y, pressure, xtilt, ytilt):
        """Draws a stroke to the current layer with the current brush.

        This is called by GUI code in response to motion events on the canvas -
        both with and without pressure. If enough time has elapsed, an input
        flush is requested (see `flush_updates()`).

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
        warn("Use a gui.canvasevent.BrushworkModeMixin's stroke_to() "
             "instead", DeprecatedAPIWarning, stacklevel=2)
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
            self.flush_updates()

    def redo_last_stroke_with_different_brush(self, brushinfo):
        cmd = self.get_last_command()
        if not isinstance(cmd, command.Brushwork):
            return
        cmd.update(brushinfo=brushinfo)


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
        :param sample_merged: Use all visible layers when sampling
        :type sample_merged: bool
        :param make_new_layer: Write output to a new layer on top
        :type make_new_layer: bool

        Filling an infinite canvas requires limits. If the frame is
        enabled, this limits the maximum size of the fill, and filling
        outside the frame is not possible.

        Otherwise, if the entire document is empty, the limits are
        dynamic.  Initially only a single tile will be filled. This can
        then form one corner for the next fill's limiting rectangle.
        This is a little quirky, but allows big areas to be filled
        rapidly as needed on blank layers.
        """
        bbox = helpers.Rect(*tuple(self.get_effective_bbox()))
        rootstack = self.layer_stack
        if not self.layer_stack.current.get_fillable():
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

    def _canvas_modified_cb(self, root, layer, x, y, w, h):
        """Internal callback: forwards redraw nofifications"""
        self.canvas_area_modified(x, y, w, h)

    @event
    def canvas_area_modified(self, x, y, w, h):
        """Event: canvas was updated, either within a rectangle or fully

        :param x: top-left x coordinate for the redraw bounding box
        :param y: top-left y coordinate for the redraw bounding box
        :param w: width of the redraw bounding box, or 0 for full redraw
        :param h: height of the redraw bounding box, or 0 for full redraw

        This event method is invoked to notify observers about needed redraws
        originating from within the model, e.g. painting, fills, or layer
        moves. It is also used to notify about the entire canvas needing to be
        redrawn. In the latter case, the `w` or `h` args forwarded to
        registered observers is zero.

        See also: `invalidate_all()`.
        """
        pass

    def invalidate_all(self):
        """Marks everything as invalid"""
        self.canvas_area_modified(0, 0, 0, 0)


    ## Undo/redo command stack

    @event
    def flush_updates(self):
        """Reqests flushing of all pending document updates

        This `lib.observable.event` is called whan pending updates
        should be flushed into the working document completely.
        Attached observers are expected to react by writing pending
        changes to the layers stack, and pushing an appropriate command
        onto the command stack using `do()`.
        """

    def undo(self):
        self.flush_updates()
        while 1:
            cmd = self.command_stack.undo()
            if not cmd or not cmd.automatic_undo:
                return cmd

    def redo(self):
        self.flush_updates()
        while 1:
            cmd = self.command_stack.redo()
            if not cmd or not cmd.automatic_undo:
                return cmd

    def do(self, cmd):
        self.flush_updates()
        self.command_stack.do(cmd)


    def update_last_command(self, **kwargs):
        self.flush_updates()
        return self.command_stack.update_last_command(**kwargs)


    def get_last_command(self):
        self.flush_updates()
        return self.command_stack.get_last_command()


    ## Utility methods

    def get_bbox(self):
        """Returns the data bounding box of the document

        This is currently the union of all the data bounding boxes of all of
        the layers. It disregards the user-chosen frame.

        """
        res = helpers.Rect()
        for layer in self.layer_stack.deepiter():
            # OPTIMIZE: only visible layers...
            # careful: currently saving assumes that all layers are included
            bbox = layer.get_bbox()
            res.expandToIncludeRect(bbox)
        return res

    def get_full_redraw_bbox(self):
        """Returns the full-redraw bounding box of the document

        This is the same concept as `layer.BaseLayer.get_full_redraw_bbox()`,
        and is built up from the full-redraw bounding boxes of all layers.
        """
        res = helpers.Rect()
        for layer in self.layer_stack.deepiter():
            bbox = layer.get_full_redraw_bbox()
            if bbox.w == 0 and bbox.h == 0: # infinite
                res = bbox
            else:
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


    ## More layer stack commands


    def add_layer(self, path):
        """Adds a new layer at a specified path"""
        self.do(command.AddLayer(self, path))

    def remove_layer(self,layer=None):
        """Delete a layer"""
        self.do(command.RemoveLayer(self, layer))

    def rename_layer(self, layer, name):
        """Rename a layer"""
        self.do(command.RenameLayer(self, name, layer))

    def normalize_layer_mode(self):
        """Normalize current layer's mode and opacity"""
        layers = self.layer_stack
        self.do(command.NormalizeLayerMode(self, layers.current))

    def merge_current_layer_down(self):
        """Merge the current layer into the one below"""
        rootstack = self.layer_stack
        cur_path = rootstack.current_path
        if cur_path is None:
            return False
        dst_path = rootstack.get_merge_down_target(cur_path)
        if dst_path is None:
            logger.info("Merge Down is not possible here")
            return False
        self.do(command.MergeLayerDown(self))
        return True

    def merge_visible_layers(self):
        self.do(command.MergeVisibleLayers(self))

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
        if layer is self.layer_stack:
            return
        cmd_class = command.SetLayerVisibility
        cmd = self.get_last_command()
        if isinstance(cmd, cmd_class) and cmd.layer is layer:
            self.update_last_command(visible=visible)
        else:
            cmd = cmd_class(self, visible, layer)
            self.do(cmd)

    def set_layer_locked(self, locked, layer):
        """Sets the input-locked status of a layer."""
        if layer is self.layer_stack:
            return
        cmd_class = command.SetLayerLocked
        cmd = self.get_last_command()
        if isinstance(cmd, cmd_class) and cmd.layer is layer:
            self.update_last_command(locked=locked)
        else:
            cmd = cmd_class(self, locked, layer)
            self.do(cmd)

    def set_layer_opacity(self, opacity):
        """Sets the opacity of the current layer

        :param float opacity: New layer opacity
        """
        current = self.layer_stack.current
        if current is self.layer_stack:
            return
        cmd_class = command.SetLayerOpacity
        cmd = self.get_last_command()
        if isinstance(cmd, cmd_class) and cmd.layer is current:
            logger.debug("Updating current layer opacity: %r", opacity)
            self.update_last_command(opacity=opacity)
        else:
            logger.debug("Setting current layer opacity: %r", opacity)
            cmd = cmd_class(self, opacity, layer=current)
            self.do(cmd)

    def set_layer_mode(self, mode):
        """Sets the combining mode for the current layer

        :param int mode: New layer combining mode to use
        """
        # To be honest, I'm not sure this command needs the full
        # update() mechanism. Modes aren't updated on a continuous
        # slider, like opacity, and setting a mode feels like a fairly
        # positive choice.
        current = self.layer_stack.current
        cmd_class = command.SetLayerMode
        cmd = self.get_last_command()
        if isinstance(cmd, cmd_class) and cmd.layer is current:
            logger.debug("Updating current layer mode: %r", mode)
            self.update_last_command(mode=mode)
        else:
            logger.debug("Setting current layer mode: %r", mode)
            cmd = cmd_class(self, mode, layer=current)
            self.do(cmd)

    def set_layer_stack_isolated(self, isolated, layer=None):
        """Sets the isolation flag for a layer stack, undoably

        :param isolated: State for the isolated flag
        :type isolated: bool
        :param layer: The layer to affect, or None for the current layer.
        :type layer: lib.layer.Layer
        """
        cmd = self.get_last_command()
        if layer is None:
            layer = self.layer_stack.current
        if layer is self.layer_stack:
            return
        cmd_class = command.SetLayerStackIsolated
        if isinstance(cmd, cmd_class) and cmd.layer is layer:
            self.update_last_command(isolated=isolated)
        else:
            self.do(cmd_class(self, isolated, layer=layer))

    ## Saving and loading


    def load_from_pixbuf(self, pixbuf):
        """Load a document from a pixbuf."""
        self.clear()
        bbox = self.load_layer_from_pixbuf(pixbuf)
        self.set_frame(bbox, user_initiated=False)


    def save(self, filename, **kwargs):
        """Save the document to a file.

        :param str filename: The filename to save to.
        :param dict kwargs: Passed on to the chosen save method.
        :raise SaveLoadError: The error string will be set to something
          descriptive and presentable to the user.
        :returns: A thumbnail pixbuf, or None if not supported
        :rtype: GdkPixbuf

        The filename's extension is used to determine the save format, and a
        ``save_*()`` method is chosen to perform the save.
        """
        self.flush_updates()
        junk, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        save = getattr(self, 'save_' + ext, self._unsupported)
        result = None
        try:
            result = save(filename, **kwargs)
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
        return result


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
        warn("Use doc.layer_stack.render_as_pixbuf() instead",
             DeprecatedAPIWarning, stacklevel=2)
        return self.layer_stack.render_as_pixbuf(*args, **kwargs)


    def render_thumbnail(self, **kwargs):
        """Renders a thumbnail for the effective (frame) bbox"""
        t0 = time.time()
        bbox = self.get_effective_bbox()
        pixbuf = self.layer_stack.render_thumbnail(bbox, **kwargs)
        logger.info('Rendered thumbnail in %d seconds.',
                    time.time() - t0)
        return pixbuf


    def save_png(self, filename, alpha=False, multifile=False, **kwargs):
        doc_bbox = self.get_effective_bbox()
        if multifile:
            self.save_multifile_png(filename, **kwargs)
        else:
            self.layer_stack.save_as_png(filename, *doc_bbox, alpha=alpha,
                                         background=(not alpha), **kwargs)

    def save_multifile_png(self, filename, alpha=False, **kwargs):
        prefix, ext = os.path.splitext(filename)
        # if we have a number already, strip it
        l = prefix.rsplit('.', 1)
        if l[-1].isdigit():
            prefix = l[0]
        doc_bbox = self.get_effective_bbox()
        for i, l in enumerate(self.layer_stack.deepiter()):
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
        pixbuf = self.layer_stack.render_as_pixbuf(x, y, w, h, **kwargs)
        options = {"quality": str(quality)}
        pixbuf.savev(filename, 'jpeg', options.keys(), options.values())


    save_jpeg = save_jpg


    def save_ora(self, filename, options=None, **kwargs):
        """Saves OpenRaster data to a file"""
        logger.info('save_ora: %r (%r, %r)', filename, options, kwargs)
        t0 = time.time()
        tempdir = tempfile.mkdtemp('mypaint')
        if not isinstance(tempdir, unicode):
            tempdir = tempdir.decode(sys.getfilesystemencoding())

        # Use .tmpsave extension, so we don't overwrite a valid file if there
        # is an exception
        orazip = zipfile.ZipFile(filename + '.tmpsave', 'w',
                                 compression=zipfile.ZIP_STORED)

        # work around a permission bug in the zipfile library:
        # http://bugs.python.org/issue3394
        def write_file_str(filename, data):
            zi = zipfile.ZipInfo(filename)
            zi.external_attr = 0100644 << 16
            orazip.writestr(zi, data)

        write_file_str('mimetype', 'image/openraster') # must be the first file
        image = ET.Element('image')
        effective_bbox = self.get_effective_bbox()
        x0, y0, w0, h0 = effective_bbox
        image.attrib['w'] = str(w0)
        image.attrib['h'] = str(h0)

        # Update the initially-selected flag on all layers
        layers = self.layer_stack
        for s_path, s_layer in layers.deepenumerate():
            selected = (s_path == layers.current_path)
            s_layer.initially_selected = selected

        # Save the layer stack
        canvas_bbox = tuple(self.get_bbox())
        frame_bbox = tuple(effective_bbox)
        root_stack_path = ()
        root_stack_elem = self.layer_stack.save_to_openraster(
                                orazip, tempdir, root_stack_path,
                                canvas_bbox, frame_bbox, **kwargs )
        image.append(root_stack_elem)

        # Resolution info
        if self._xres and self._yres:
            image.attrib["xres"] = str(self._xres)
            image.attrib["yres"] = str(self._yres)

        # Version declaration
        image.attrib["version"] = "0.0.4-pre.1"

        # Thumbnail preview (256x256)
        thumbnail = layers.render_thumbnail(frame_bbox)
        tmpfile = join(tempdir, 'tmp.png')
        thumbnail.savev(tmpfile, 'png', [], [])
        orazip.write(tmpfile, 'Thumbnails/thumbnail.png')
        os.remove(tmpfile)

        # Save fully rendered image too
        tmpfile = os.path.join(tempdir, "mergedimage.png")
        self.layer_stack.save_as_png( tmpfile, *frame_bbox,
                                      alpha=False, background=True,
                                      **kwargs )
        orazip.write(tmpfile, 'mergedimage.png')
        os.remove(tmpfile)

        # Prettification
        helpers.indent_etree(image)
        xml = ET.tostring(image, encoding='UTF-8')

        # Finalize
        write_file_str('stack.xml', xml)
        orazip.close()
        os.rmdir(tempdir)
        if os.path.exists(filename):
            os.remove(filename) # windows needs that
        os.rename(filename + '.tmpsave', filename)

        logger.info('%.3fs save_ora total', time.time() - t0)
        return thumbnail


    def load_ora(self, filename, feedback_cb=None):
        """Loads from an OpenRaster file"""
        logger.info('load_ora: %r', filename)
        t0 = time.time()
        tempdir = self._tempdir
        orazip = zipfile.ZipFile(filename)
        logger.debug('mimetype: %r', orazip.read('mimetype').strip())
        xml = orazip.read('stack.xml')
        image_elem = ET.fromstring(xml)
        root_stack_elem = image_elem.find('stack')
        image_width = max(0, int(image_elem.attrib.get('w', 0)))
        image_height = max(0, int(image_elem.attrib.get('h', 0)))
        # Resolution: false value, 0 specifically, means unspecified
        image_xres = max(0, int(image_elem.attrib.get('xres', 0)))
        image_yres = max(0, int(image_elem.attrib.get('yres', 0)))

        # Delegate loading of image data to the layers tree itself
        self.layer_stack.clear()
        self.layer_stack.load_from_openraster(orazip, root_stack_elem,
                                              tempdir, feedback_cb, x=0, y=0)
        assert len(self.layer_stack) > 0

        # Set up symmetry axes
        for path, descendent in self.layer_stack.deepenumerate():
            descendent.set_symmetry_axis(self.get_symmetry_axis())

        # Resolution information if specified
        # Before frame to benefit from its observer call
        if image_xres and image_yres:
            self._xres = image_xres
            self._yres = image_yres
        else:
            self._xres = None
            self._yres = None

        # Set the frame size to that saved in the image.
        self.update_frame(x=0, y=0, width=image_width, height=image_height,
                          user_initiated=False)

        # Enable frame if the saved image size is something other than the
        # calculated bounding box. Goal: if the user saves an "infinite
        # canvas", it loads as an infinite canvas.
        bbox_c = helpers.Rect(x=0, y=0, w=image_width, h=image_height)
        bbox = self.get_bbox()
        frame_enab = not (bbox_c==bbox or bbox.empty() or bbox_c.empty())
        self.set_frame_enabled(frame_enab, user_initiated=False)

        orazip.close()

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
        return self._doc.layer_stack.deepiter()

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
