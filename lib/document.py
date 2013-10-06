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


class Document():
    """
    This is the "model" in the Model-View-Controller design.
    (The "view" would be ../gui/tileddrawwidget.py.)
    It represents everything that the user would want to save.


    The "controller" mostly in drawwindow.py.
    It is possible to use it without any GUI attached (see ../tests/)
    """
    # Please note the following difficulty with the undo stack:
    #
    #   Most of the time there is an unfinished (but already rendered)
    #   stroke pending, which has to be turned into a command.Action
    #   or discarded as empty before any other action is possible.
    #   (split_stroke)

    def __init__(self, brushinfo=None):
        if not brushinfo:
            brushinfo = brush.BrushInfo()
            brushinfo.load_defaults()
        self.layers = []
        self.brush = brush.Brush(brushinfo)
        self.brush.brushinfo.observers.append(self.brushsettings_changed_cb)
        self.stroke = None
        self.canvas_observers = []  #: See `layer_modified_cb()`
        self.stroke_observers = [] #: See `split_stroke()`
        self.doc_observers = [] #: See `call_doc_observers()`
        self.frame_observers = []
        self.command_stack_observers = []
        self.symmetry_observers = []  #: See `set_symmetry_axis()`
        self.__symmetry_axis = None
        self.default_background = (255, 255, 255)
        self.clear(True)

        self._frame = [0, 0, 0, 0]
        self._frame_enabled = False


    ## Layer (x, y) position


    def move_current_layer(self, dx, dy):
        layer = self.layers[self.layer_idx]
        layer.translate(dx, dy)


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
        return self.__symmetry_axis


    def set_symmetry_axis(self, x):
        """Sets the active painting symmetry X axis value.

        A value of `None` inactivates symmetrical painting. After setting, all
        registered `symmetry_observers` are called without arguments.
        """
        # TODO: make this undoable?
        for layer in self.layers:
            layer.set_symmetry_axis(x)
        self.__symmetry_axis = x
        for func in self.symmetry_observers:
            func()


    def clear(self, init=False):
        self.split_stroke()
        self.set_symmetry_axis(None)
        if not init:
            bbox = self.get_bbox()
        # throw everything away, including undo stack

        self.command_stack = command.CommandStack()
        self.command_stack.stack_observers = self.command_stack_observers
        self.set_background(self.default_background)
        self.layers = []
        self.layer_idx = None
        self.add_layer(0)
        # disallow undo of the first layer
        self.command_stack.clear()
        self.unsaved_painting_time = 0.0

        if not init:
            for f in self.canvas_observers:
                f(*bbox)

        self.call_doc_observers()

    def get_current_layer(self):
        return self.layers[self.layer_idx]
    layer = property(get_current_layer)


    def split_stroke(self):
        """Splits the current stroke, announcing the newly stacked stroke

        The stroke being drawn is pushed onto to the command stack and the
        callbacks in the list `self.stroke_observers` are invoked with two
        arguments: the newly completed stroke, and the brush used. The brush
        argument is a temporary read-only convenience object.

        This is called every so often when drawing a single long brushstroke on
        input to allow parts of a long line to be undone.

        """
        if not self.stroke: return
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

    def select_layer(self, idx):
        self.do(command.SelectLayer(self, idx))

    def record_layer_move(self, layer, dx, dy):
        layer_idx = self.layers.index(layer)
        self.do(command.MoveLayer(self, layer_idx, dx, dy, True))

    def move_layer(self, was_idx, new_idx, select_new=False):
        self.do(command.ReorderSingleLayer(self, was_idx, new_idx, select_new))

    def duplicate_layer(self, insert_idx=None, name=''):
        self.do(command.DuplicateLayer(self, insert_idx, name))

    def reorder_layers(self, new_layers):
        self.do(command.ReorderLayers(self, new_layers))

    def clear_layer(self):
        if not self.layer.is_empty():
            self.do(command.ClearLayer(self))


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
        if not self.stroke:
            self.stroke = stroke.Stroke()
            self.stroke.start_recording(self.brush)
            self.snapshot_before_stroke = self.layer.save_snapshot()
        self.stroke.record_event(dtime, x, y, pressure, xtilt, ytilt)

        split = self.layer.stroke_to(self.brush, x, y,
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


    def get_bbox(self):
        """Returns the dynamic bounding box of the document.

        This is currently the union of all the bounding boxes of all of the
        layers. It disregards the user-chosen frame.

        """
        res = helpers.Rect()
        for layer in self.layers:
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


    def render_into(self, surface, tiles, mipmap_level=0, layers=None, background=None):

        # TODO: move this loop down in C/C++
        for tx, ty in tiles:
            with surface.tile_request(tx, ty, readonly=False) as dst:
                self.blit_tile_into(dst, False, tx, ty, mipmap_level, layers, background)

    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, mipmap_level=0, layers=None, background=None):
        assert dst_has_alpha is False
        if layers is None:
            layers = self.layers
        if background is None:
            background = self.background

        assert dst.shape[-1] == 4
        if dst.dtype == 'uint8':
            dst_8bit = dst
            dst = numpy.empty((N, N, 4), dtype='uint16')
        else:
            dst_8bit = None

        background.blit_tile_into(dst, dst_has_alpha, tx, ty, mipmap_level)

        for layer in layers:
            layer.composite_tile(dst, dst_has_alpha, tx, ty, mipmap_level)

        if dst_8bit is not None:
            mypaintlib.tile_convert_rgbu16_to_rgbu8(dst, dst_8bit)

    def get_rendered_image_behind_current_layer(self, tx, ty):
        dst = numpy.empty((N, N, 4), dtype='uint16')
        l = self.layers[0:self.layer_idx]
        self.blit_tile_into(dst, False, tx, ty, layers=l)
        return dst


    def add_layer(self, insert_idx=None, after=None, name=''):
        self.do(command.AddLayer(self, insert_idx, after, name))


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


    def set_background(self, obj, make_default=False):
        # This is not an undoable action. One reason is that dragging
        # on the color chooser would get tons of undo steps.
        if not isinstance(obj, tiledsurface.Background):
            if isinstance(obj, GdkPixbuf.Pixbuf):
                obj = helpers.gdkpixbuf2numpy(obj)
            obj = tiledsurface.Background(obj)
        self.background = obj
        if make_default:
            self.default_background = obj
        self.invalidate_all()


    def load_from_pixbuf(self, pixbuf):
        """Load a document from a pixbuf."""
        self.clear()
        bbox = self.load_layer_from_pixbuf(pixbuf)
        self.set_frame(bbox, user_initiated=False)


    def is_layered(self):
        """True if there are more than one nonempty layers."""
        count = 0
        for l in self.layers:
            if not l.is_empty():
                count += 1
        return count > 1

    def is_empty(self):
        """True if there is only one layer and it is empty."""
        return len(self.layers) == 1 and self.layer.is_empty()

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
                tmp_layer = layer.Layer()
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
        x0, y0, w0, h0 = self.get_effective_bbox()
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

        def store_surface(surface, name, rect=[]):
            tmp = join(tempdir, 'tmp.png')
            t1 = time.time()
            surface.save_as_png(tmp, *rect, **kwargs)
            logger.debug('%.3fs surface saving %s', time.time() - t1, name)
            z.write(tmp, name)
            os.remove(tmp)

        def add_layer(x, y, opac, surface, name, layer_name, visible=True,
                      locked=False, selected=False,
                      compositeop=DEFAULT_COMPOSITE_OP, rect=[]):
            layer = ET.Element('layer')
            stack.append(layer)
            store_surface(surface, name, rect)
            a = layer.attrib
            if layer_name:
                a['name'] = layer_name
            a['src'] = name
            a['x'] = str(x)
            a['y'] = str(y)
            a['opacity'] = str(opac)
            if compositeop not in VALID_COMPOSITE_OPS:
                compositeop = DEFAULT_COMPOSITE_OP
            a['composite-op'] = compositeop
            if visible:
                a['visibility'] = 'visible'
            else:
                a['visibility'] = 'hidden'
            if locked:
                a['edit-locked'] = 'true'
            if selected:
                a['selected'] = 'true'
            return layer

        for idx, l in enumerate(reversed(self.layers)):
            if l.is_empty():
                continue
            opac = l.opacity
            x, y, w, h = l.get_bbox()
            sel = (idx == self.layer_idx)
            el = add_layer(x-x0, y-y0, opac, l._surface,
                           'data/layer%03d.png' % idx, l.name, l.visible,
                           locked=l.locked, selected=sel,
                           compositeop=l.compositeop, rect=(x, y, w, h))
            # strokemap
            sio = StringIO()
            l.save_strokemap_to_file(sio, -x, -y)
            data = sio.getvalue(); sio.close()
            name = 'data/layer%03d_strokemap.dat' % idx
            el.attrib['mypaint_strokemap_v2'] = name
            write_file_str(name, data)

        # save background as layer (solid color or tiled)
        bg = self.background
        # save as fully rendered layer
        x, y, w, h = self.get_bbox()
        l = add_layer(x-x0, y-y0, 1.0, bg, 'data/background.png', 'background',
                      locked=True, selected=False,
                      compositeop=DEFAULT_COMPOSITE_OP,
                      rect=(x,y,w,h))
        x, y, w, h = bg.get_bbox()
        # save as single pattern (with corrected origin)
        store_surface(bg, 'data/background_tile.png', rect=(x+x0, y+y0, w, h))
        l.attrib['background_tile'] = 'data/background_tile.png'

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

    @staticmethod
    def __xsd2bool(v):
        v = str(v).lower()
        if v in ['true', '1']: return True
        else: return False

    def load_ora(self, filename, feedback_cb=None):
        """Loads from an OpenRaster file"""
        logger.info('load_ora: %r', filename)
        t0 = time.time()
        tempdir = tempfile.mkdtemp('mypaint')
        if not isinstance(tempdir, unicode):
            tempdir = tempdir.decode(sys.getfilesystemencoding())
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

        def get_layers_list(root, x=0,y=0):
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
                    res += get_layers_list(item, stack_x, stack_y)
                else:
                    logger.warning('ignoring unsupported tag %r', item.tag)
            return res

        self.clear() # this leaves one empty layer
        no_background = True

        selected_layer = None
        for layer in get_layers_list(stack):
            a = layer.attrib

            if 'background_tile' in a:
                assert no_background
                try:
                    logger.debug("background tile: %r", a['background_tile'])
                    self.set_background(get_pixbuf(a['background_tile']))
                    no_background = False
                    continue
                except tiledsurface.BackgroundError, e:
                    logger.warning('ORA background tile not usable: %r', e)

            src = a.get('src', '')
            if not src.lower().endswith('.png'):
                logger.warning('Ignoring non-png layer %r', src)
                continue
            name = a.get('name', '')
            x = int(a.get('x', '0'))
            y = int(a.get('y', '0'))
            opac = float(a.get('opacity', '1.0'))
            compositeop = str(a.get('composite-op', DEFAULT_COMPOSITE_OP))
            if compositeop not in VALID_COMPOSITE_OPS:
                compositeop = DEFAULT_COMPOSITE_OP
            selected = self.__xsd2bool(a.get("selected", 'false'))
            locked = self.__xsd2bool(a.get("edit-locked", 'false'))

            visible = not 'hidden' in a.get('visibility', 'visible')
            self.add_layer(insert_idx=0, name=name)
            t1 = time.time()

            # extract the png form the zip into a file first
            # the overhead for doing so seems to be neglegible (around 5%)
            z.extract(src, tempdir)
            tmp_filename = join(tempdir, src)
            self.load_layer_from_png(tmp_filename, x, y, feedback_cb)
            os.remove(tmp_filename)

            layer = self.layers[0]

            self.set_layer_opacity(helpers.clamp(opac, 0.0, 1.0), layer)
            self.set_layer_compositeop(compositeop, layer)
            self.set_layer_visibility(visible, layer)
            self.set_layer_locked(locked, layer)
            if selected:
                selected_layer = layer
            logger.debug('%.3fs loading and converting layer png',
                         time.time() - t1)
            # strokemap
            fname = a.get('mypaint_strokemap_v2', None)
            if fname:
                sio = StringIO(z.read(fname))
                layer.load_strokemap_from_file(sio, x, y)
                sio.close()

        if len(self.layers) == 1:
            # no assertion (allow empty documents)
            logger.error('Could not load any layer, document is empty.')

        if len(self.layers) > 1:
            # remove the still present initial empty top layer
            self.select_layer(len(self.layers)-1)
            self.remove_layer()
            # this leaves the topmost layer selected

        if selected_layer is not None:
            for i, layer in zip(range(len(self.layers)), self.layers):
                if layer is selected_layer:
                    self.select_layer(i)
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

        # remove empty directories created by zipfile's extract()
        for root, dirs, files in os.walk(tempdir, topdown=False):
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(tempdir)

        logger.info('%.3fs load_ora total', time.time() - t0)
