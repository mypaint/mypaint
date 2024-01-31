# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2010-2019 by the MyPaint Development Team.
# Copyright (C) 2007-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

from __future__ import division, print_function
from collections import deque
from warnings import warn
from copy import deepcopy
import weakref
from gettext import gettext as _
from logging import getLogger

import lib.layer
import lib.layer.data
from . import helpers
from lib.observable import event
import lib.stroke
from lib.pycompat import unicode

logger = getLogger(__name__)


## Command stack and action interface


class CommandStack (object):
    """Undo/redo stack"""

    def __init__(self, max_stack_size, **kwargs):
        super(CommandStack, self).__init__()
        assert isinstance(max_stack_size, int) and max_stack_size > 0
        self.max_stack_size = max_stack_size
        self.undo_stack = deque()
        self.redo_stack = deque()
        self.stack_updated()

    def __repr__(self):
        return ("<CommandStack undo_len=%d redo_len=%d>" %
                (len(self.undo_stack), len(self.redo_stack),))

    def clear(self):
        self._discard_undo()
        self._discard_redo()
        self.stack_updated()

    def _discard_undo(self):
        self.undo_stack = deque()

    def _discard_redo(self):
        self.redo_stack = deque()

    def do(self, command):
        """Performs a new command

        :param Command command: New action to perform and push

        This operation adds a new command to the undo stack after
        calling its redo() method to perform the work it represents.
        It also trims the undo stack.

        If the command is cancelled, indicated by it returning False,
        its undo() method is called and it is not added to the stack.
        """
        # Discard the redo stack regardless of cancellation
        # This follows the behavior of Krita, but might not be desirable
        self._discard_redo()
        completed = command.redo()
        # NOTE: The below is not equivalent to checking truthiness,
        # do not switch order and replace with "if completed:"
        if completed is False:
            command.undo()
        else:
            self.undo_stack.append(command)
            self.reduce_undo_history()
            self.stack_updated()

    def undo(self):
        """Un-performs the last performed command

        This operation undoes one command, moving it from the undo stack
        to the redo stack, and invoking its its undo() method.
        """
        if not self.undo_stack:
            return
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        self.stack_updated()
        return command

    def redo(self):
        """Re-performs the last command undone with undo()

        This operation re-does one command, moving it from the undo
        stack to the redo stack, and invoking its its redo() method.

        If the command is cancelled, as indicated by it returning False,
        the undo() method of the same command is run, and no stack changes
        are performed.

        Calls stack_updated() if the command is not cancelled
        """
        if not self.redo_stack:
            return
        command = self.redo_stack[-1]
        completed = command.redo()
        if completed is False:
            command.undo()
        else:
            self.redo_stack.pop()
            self.undo_stack.append(command)
            self.stack_updated()
        return command

    def reduce_undo_history(self):
        """Trims the undo stack"""
        while len(self.undo_stack) > self.max_stack_size:
            self.undo_stack.popleft()

    def get_last_command(self):
        """Returns the most recently performed command"""
        if not self.undo_stack:
            return None
        return self.undo_stack[-1]

    def update_last_command(self, **kwargs):
        """Updates the most recently performed command"""
        cmd = self.get_last_command()
        if cmd is None:
            return None
        old_name = cmd.display_name
        cmd.update(**kwargs)
        if old_name != cmd.display_name:
            self.stack_updated()
        return cmd

    @event
    def stack_updated(self):
        """Event: command stack was updated"""
        pass


class Command (object):
    """A reversible change to the document model

    Commands represent alterations made by the user to a document which
    they might wish to undo. They should in general be lightweight
    mementos of the of the work carried out, but may store snapshots of
    layer data provided those those don't create circular reference
    chains.

    Typical commands are constructed as a complete description of the
    work to be done by a simple action callback, and perform all the
    actual work in their `redo()` method.  Alternatively, a command can
    be constructed as an incomplete description of work to be performed
    interactively, and record the work as it proceeds. In this second
    case, the UI code *must* ensure that the work being done is
    committed to the document (via `lib.document.Document.redo()`) when
    it resuests that input is flushed.

    This class is the base for all commands.  Subclasses must implement
    at least the `redo()` and `undo()` methods, and may implement an
    update method if it makes sense for the data being changed.
    """

    ## Defaults for object properties

    automatic_undo = False
    display_name = _("Unknown Command")

    ## Method defs

    def __init__(self, doc, **kwargs):
        """Constructor

        :param lib.document.Document doc: the model to be changed
        :param **kwargs: Initial description of the work to be done
        """
        super(Command, self).__init__()
        object.__init__(self)
        #: The document model to alter (proxy, to permit clean gc)
        self.doc = weakref.proxy(doc)

    def __repr__(self):
        return "<%s>" % (self.display_name,)

    ## Main Command interface

    def redo(self):
        """Callback used to perform, or re-perform the work

        Initially, this is essentially a commit to the in-memory
        document. It should finalize any changes made at construction or
        subsequently, and make sure that notifications are issued
        correctly.  Redo may also be called after an undo if the user
        changes their mind about undoing something.
        """
        raise NotImplementedError

    def undo(self):
        """Callback used to roll back work

        This is the rollback to `redo()`'s commit action.
        """
        raise NotImplementedError

    def update(self, **kwargs):
        """In-place update on the tip of the undo stack.

        This method should update the model in the way specified in
        `**kwargs`.  The interpretation of arguments is left to the
        concrete implementation.

        The alternative to implementing this method is an undo()
        followed by a redo(). This can result in too many notifications
        being sent, however.  In general, this method should be
        implemented whenever only the final value of a change matters,
        for example a change to a layer's opacity or its locked status.
        """
        raise NotImplementedError

    ## Deprecated utility functions for subclasses

    def _notify_canvas_observers(self, layer_bboxes):
        """Notifies the document's redraw observers"""
        warn("Layers should issue their own canvas updates",
             PendingDeprecationWarning, stacklevel=2)
        redraw_bbox = helpers.Rect()
        for layer_bbox in layer_bboxes:
            if layer_bbox.w == 0 and layer_bbox.h == 0:
                redraw_bbox = layer_bbox
                break
            else:
                redraw_bbox.expand_to_include_rect(layer_bbox)
        self.doc.canvas_area_modified(*redraw_bbox)


class Brushwork (Command):
    """Some seconds of painting on a layer in a document."""

    def __init__(self, doc, layer_path=None, description=None,
                 abrupt_start=False, layer=None, **kwds):
        """Initializes as an active brushwork command

        :param doc: document being updated
        :type doc: lib.document.Document
        :param tuple layer_path: path of the layer to affect within doc
        :param unicode description: Descriptive name for this brushwork
        :param bool abrupt_start: Reset brush & dwell before starting
        :param lib.layer.data.SimplePaintingLayer layer: explicit target layer

        The Brushwork command is created as an active command which can
        be used for capturing brushstrokes. Recording must be stopped
        before the command is added to the CommandStack.

        If an explicit target layer is used, it must be one that's
        guaranteed to persist for the lifetime of the current document
        model to prevent leaks.

        If one is not used, the layer path must always refer to the same
        layer while the command is used for recording the stroke, and at
        the points in time where redo() or undo() might be called on it.

        """
        super(Brushwork, self).__init__(doc, **kwds)
        if not (layer_path or layer):
            raise ValueError("Either layer_path or layer must be set")
        elif (layer_path and layer):
            raise ValueError("Cannot set both layer_path and layer")
        self._layer = layer
        self._layer_path = layer_path
        self._abrupt_start = abrupt_start
        # Recording phase
        self._abrupt_start_done = False
        self._stroke_target_layer = None
        self._stroke_seq = None
        # When recorded, undo & redo switch the model between these states
        self._time_before = None
        self._sshot_before = None
        self._time_after = None
        self._sshot_after = None
        # For display
        self.description = description
        # State vars
        self._recording_started = False
        self._recording_finished = False
        self.split_due = False
        self._sshot_after_applied = False

    def __repr__(self):
        time = 0.0
        if self._stroke_seq is not None:
            time = self._stroke_seq.total_painting_time
        repstr = (
            "<{cls} {id:#x} {seconds:.03f}s "
            "{self.description!r}>"
        ).format(
            cls = self.__class__.__name__,
            id = id(self),
            seconds = time,
            self = self,
        )
        return repstr

    @property
    def display_name(self):
        """Dynamic property: string used for displaying the command"""
        if self.description is not None:
            return self.description
        if self._stroke_seq is None:
            time = 0.0
            brush_name = _("Undefined (command not started yet)")
        else:
            time = self._stroke_seq.total_painting_time
            brush_name = unicode(self._stroke_seq.brush_name)
        # TRANSLATORS: A short time spent painting / making brushwork.
        # TRANSLATORS: This can correspond to zero or more touches of
        # TRANSLATORS: the physical stylus to the tablet.
        return _(u"{seconds:.01f}s of painting with {brush_name}").format(
            seconds=time,
            brush_name=brush_name,
        )

    @property
    def _target_layer(self):
        """The command's target layer.

        This is either the explicit target layer from the constructor,
        or the layer accessed via its path.

        The _stroke_target_layer cache property is used during painting.

        """
        model = self.doc
        return self._layer or model.layer_stack.deepget(self._layer_path)

    def redo(self):
        """Performs, or re-performs after undo"""
        model = self.doc
        layer = self._target_layer
        assert self._recording_finished, "Call stop_recording() first"
        assert self._sshot_before is not None
        assert self._sshot_after is not None
        assert self._time_before is not None
        if not self._sshot_after_applied:
            layer.load_snapshot(self._sshot_after)
            self._sshot_after_applied = True
        # Update painting time
        assert self._time_after is not None
        model.unsaved_painting_time = self._time_after

    def undo(self):
        """Undoes the effects of redo()"""
        model = self.doc
        layer = self._target_layer
        assert self._recording_finished, "Call stop_recording() first"
        layer.load_snapshot(self._sshot_before)
        model.unsaved_painting_time = self._time_before
        self._sshot_after_applied = False

    def update(self, brushinfo):
        """Retrace the last stroke with a new brush"""
        layer = self._target_layer
        assert self._recording_finished, "Call stop_recording() first"
        assert self._sshot_after_applied, \
            "command.Brushwork must be applied before being updated"
        layer.load_snapshot(self._sshot_before)
        stroke = self._stroke_seq.copy_using_different_brush(brushinfo)
        layer.render_stroke(stroke)
        self._stroke_seq = stroke
        layer.add_stroke_shape(stroke, self._sshot_before)
        self._sshot_after = layer.save_snapshot()

    def _check_recording_started(self):
        """Ensure command is in the recording phase"""
        assert not self._recording_finished
        if self._recording_started:
            return
        # Cache the layer being painted to. This is accessed frequently
        # during the painting phase.
        model = self.doc
        layer = self._target_layer
        assert layer is not None, \
            "Layer with path %r not available" % (self._layer_path,)
        if not layer.get_paintable():
            logger.warning(
                "Brushwork: skipped non-paintable layer %r",
                layer,
            )
            return
        self._stroke_target_layer = layer

        assert self._sshot_before is None
        assert self._time_before is None
        assert self._stroke_seq is None
        self._sshot_before = layer.save_snapshot()
        self._time_before = model.unsaved_painting_time
        self._stroke_seq = lib.stroke.Stroke()
        self._stroke_seq.start_recording(model.brush)
        assert self._sshot_after is None
        self._recording_started = True

    def stroke_to(self, dtime, x, y, pressure, xtilt, ytilt,
                  viewzoom, viewrotation, barrel_rotation):
        """Painting: forward a stroke position update to the model

        :param float dtime: Seconds since the last call to this method
        :param float x: Document X position update
        :param float y: Document Y position update
        :param float pressure: Pressure, ranging from 0.0 to 1.0
        :param float xtilt: X-axis tilt, ranging from -1.0 to 1.0
        :param float ytilt: Y-axis tilt, ranging from -1.0 to 1.0
        :param float viewzoom: current view zoom level from 0 to 64
        :param float viewrotation; current view rotation from -180.0 to 180.0
        :param float barrel_rotation: Barrel Rotation of stylus, ranging from 0.0 to 1.0 

        Stroke data is recorded at this level, but strokes are not
        autosplit here because that would involve the creation of a new
        Brushwork command on the CommandStack. Instead, callers should
        check `split_due` and split appropriately.

        An example of a mode which does just this can be found in gui/.

        """
        self._check_recording_started()
        model = self.doc
        layer = self._stroke_target_layer
        if layer is None:
            return  # wasn't suitable for painting
        # Reset initial brush state if requested.
        brush = model.brush
        if self._abrupt_start and not self._abrupt_start_done:
            brush.reset()
            layer.stroke_to(
                brush, x, y,
                0.0,
                xtilt, ytilt,
                10.0,
                viewzoom, viewrotation, barrel_rotation,
            )
            self._abrupt_start_done = True
        # Record and paint this position
        self._stroke_seq.record_event(
            dtime,
            x, y, pressure,
            xtilt, ytilt, viewzoom, viewrotation, barrel_rotation,
        )
        self.split_due = layer.stroke_to(
            brush,
            x, y, pressure,
            xtilt, ytilt, dtime, viewzoom, viewrotation, barrel_rotation,
        )

    def stop_recording(self, revert=False):
        """Ends the recording phase

        :param bool revert: revert any changes to the model
        :rtype: bool
        :returns: whether any changes were made

        When called with default arguments,
        this method makes the command ready to add to the command stack
        using the document model's do() method.
        If no changes were made, you can (and should)
        just discard the command instead.

        If `revert` is true,
        all changes made to the layer during recording
        will be rolled back,
        so that the layer has its original appearance and state.
        Reverted commands should be discarded.

        After this method is called,
        the `stroke_to()` method must not be called again.

        """
        self._check_recording_started()
        layer = self._stroke_target_layer
        self._stroke_target_layer = None  # prevent potential leak
        self._recording_finished = True
        if self._stroke_seq is None:
            # Unclear circumstances, but I've seen it happen
            # (unpaintable layers and visibility state toggling).
            # Perhaps _recording_started should be made synonymous with this?
            logger.warning(
                "No recorded stroke, but recording was started? "
                "Please report this glitch if you can reliably reproduce it."
            )
            return False  # nothing recorded, so nothing changed
        self._stroke_seq.stop_recording()
        if layer is None:
            return False  # wasn't suitable for painting, thus nothing changed
        if revert:
            assert self._sshot_before is not None
            layer.load_snapshot(self._sshot_before)
            logger.debug("Reverted %r: tiles_changed=%r", self, False)
            return False  # nothing changed
        t0 = self._time_before
        self._time_after = t0 + self._stroke_seq.total_painting_time
        layer.add_stroke_shape(self._stroke_seq, self._sshot_before)
        self._sshot_after = layer.save_snapshot()
        self._sshot_after_applied = True  # changes happened before redo()
        tiles_changed = (not self._stroke_seq.empty)
        logger.debug(
            "Stopped recording %r: tiles_changed=%r",
            self, tiles_changed,
        )
        return tiles_changed


## Concrete command classes


class FloodFill (Command):
    """Flood-fill on the current layer"""

    display_name = _("Flood Fill")

    def __init__(
            self, doc, fill_args,
            sample_merged, src_path, make_new_layer, status_cb, **kwds):
        """
        Create a new fill command

        :param doc: the current active document
        :param fill_args: common fill parameters bundled up
        :type fill_args: lib.floodfill.FloodFillArguments
        :param sample_merged: fill w. visual reference to the full layer stack
        :param src_path: Path to the source layer, or None if the active layer
        :param make_new_layer: Whether a new layer should be created
        :param status_cb: GUI status/cancellation callback
        """
        super(FloodFill, self).__init__(doc, **kwds)
        self.fill_args = fill_args
        self.fill_args.framed = doc.get_frame_enabled()
        self.sample_merged = sample_merged
        self.src_path = src_path
        self.make_new_layer = make_new_layer
        self.new_layer = None
        self.new_layer_path = None
        self.snapshot = None
        self.status_cb = status_cb

    def redo(self):
        # Pick a source
        layers = self.doc.layer_stack
        if self.sample_merged:
            src_layer = layers
        elif self.src_path is not None:
            src_layer = layers.deepget(self.src_path)
        else:
            src_layer = layers.current
        # Choose a target
        if self.make_new_layer:
            # Write to a new layer
            assert self.new_layer is None
            nl = lib.layer.PaintingLayer()
            self.new_layer = nl
            path = layers.get_current_path()
            path = layers.path_above(path, insert=1)
            layers.deepinsert(path, nl)
            path = layers.deepindex(nl)
            self.new_layer_path = path
            layers.set_current_path(path)
            dst_layer = nl
            # With alpha locking and some comp modes, it makes
            # no sense to perform the actual fill on an empty layer
            if self.fill_args.skip_empty_dst():
                return
        else:
            # Overwrite current, but snapshot 1st
            assert self.snapshot is None
            self.snapshot = layers.current.save_snapshot()
            dst_layer = layers.current
        # Fill connected areas of the source into the destination
        handle = src_layer.flood_fill(self.fill_args, dst_layer=dst_layer)

        # Give the fill a second before starting a cancel dialog
        max_wait_time = 1.0
        handle.wait(max_wait_time)
        if handle.running() and self.status_cb:
            return self.status_cb(handle)
        # In case of no status/cancel cb, ensure fill finishes
        handle.wait()

    def undo(self):
        layers = self.doc.layer_stack
        if self.make_new_layer:
            assert self.new_layer is not None
            path = layers.get_current_path()
            layers.deepremove(self.new_layer)
            layers.set_current_path(path)  # or attempt to
            self.new_layer = None
            self.new_layer_path = None
        else:
            assert self.snapshot is not None
            layers.current.load_snapshot(self.snapshot)
            self.snapshot = None


class TrimLayer (Command):
    """Trim the current layer to the extent of the document frame"""

    display_name = _("Trim Layer")

    def __init__(self, doc, **kwds):
        super(TrimLayer, self).__init__(doc, **kwds)
        self.before = None

    def redo(self):
        layer = self.doc.layer_stack.current
        self.before = layer.save_snapshot()
        frame = self.doc.get_frame()
        layer.trim(frame)

    def undo(self):
        layer = self.doc.layer_stack.current
        layer.load_snapshot(self.before)


class UniqLayer (Command):
    """Remove areas from the current layer that don't alter the backdrop."""

    display_name = _(u"Uniquify Layer Pixels")

    def __init__(self, doc, pixels=False, **kwds):
        super(UniqLayer, self).__init__(doc, **kwds)
        self._before = None
        self._pixels = pixels

    def redo(self):
        root = self.doc.layer_stack
        layer = root.current
        self._before = layer.save_snapshot()
        path = root.current_path
        root.uniq_layer(path, pixels=self._pixels)

    def undo(self):
        root = self.doc.layer_stack
        layer = root.current
        layer.load_snapshot(self._before)


class RefactorGroup (Command):
    """Extract common parts of sublayers to a new layer, then delete them."""

    display_name = _(u"Refactor Group")

    def __init__(self, doc, pixels=False, **kwds):
        super(RefactorGroup, self).__init__(doc, **kwds)
        self._before = None
        self._pixels = pixels

    def redo(self):
        root = self.doc.layer_stack
        layer = root.current
        self._before = layer.save_snapshot()
        path = root.current_path
        root.refactor_layer_group(path, pixels=self._pixels)

    def undo(self):
        root = self.doc.layer_stack
        layer = root.current
        layer.load_snapshot(self._before)


class ClearLayer (Command):
    """Clears the current layer"""

    display_name = _("Clear Layer")

    def __init__(self, doc, **kwds):
        super(ClearLayer, self).__init__(doc, **kwds)
        self._before = None

    def redo(self):
        layer = self.doc.layer_stack.current
        self._before = layer.save_snapshot()
        layer.clear()

    def undo(self):
        layer = self.doc.layer_stack.current
        layer.load_snapshot(self._before)
        self._before = None


class LoadLayer (Command):
    """Loads a layer from a surface"""

    # This is used by Paste layer as well as when loading from a PNG
    # file. However in the latter case, the undo stack is reset
    # immediately afterward.

    display_name = _("Paste Layer")

    def __init__(self, doc, surface, to_new_layer=False, **kwds):
        super(LoadLayer, self).__init__(doc, **kwds)
        self.doc = doc
        self.surface = surface
        self.add_layer_cmd = None
        self.before = None
        if to_new_layer:
            insert_path = self.doc.layer_stack.current_path
            self.add_layer_cmd = AddLayer(doc, insert_path)

    def redo(self):
        if self.add_layer_cmd:
            self.add_layer_cmd.redo()
        layer = self.doc.layer_stack.current
        if not self.add_layer_cmd:
            self.before = layer.save_snapshot()
        layer.load_from_surface(self.surface)

    def undo(self):
        if self.add_layer_cmd:
            self.add_layer_cmd.undo()
        else:
            self.doc.layer_stack.current.load_snapshot(self.before)
            del self.before


class NewLayerMergedFromVisible (Command):
    """Create a new layer from the merge of all visible layers

    Performs a Merge Visible, and inserts the result into the layer
    stack just before the highest root of any visible layer.
    """

    display_name = _("New Layer from Visible")

    def __init__(self, doc, **kwds):
        super(NewLayerMergedFromVisible, self).__init__(doc, **kwds)
        self._old_current_path = doc.layer_stack.current_path
        self._result_insert_path = None
        self._result_layer = None
        self._result_final_path = None
        self._paths_merged = None

    def redo(self):
        rootstack = self.doc.layer_stack
        merged = self._result_layer
        if merged is None:
            self._result_insert_path = (len(rootstack),)
            self._paths_merged = []
            for path, layer in rootstack.walk(visible=True):
                if path[0] < self._result_insert_path[0]:
                    self._result_insert_path = (path[0],)
                self._paths_merged.append(path)
            merged = rootstack.layer_new_merge_visible()
            self._result_layer = merged
        assert self._result_insert_path is not None
        rootstack.deepinsert(self._result_insert_path, merged)
        self._result_final_path = rootstack.deepindex(merged)
        rootstack.current_path = self._result_final_path

    def undo(self):
        rootstack = self.doc.layer_stack
        rootstack.deeppop(self._result_final_path)
        rootstack.current_path = self._old_current_path


class MergeVisibleLayers (Command):
    """Consolidate all visible layers into one

    Deletes all visible layers, but inserts the result of merging them
    into the layer stack just before the highest root of any of the
    merged+deleted layers.

    """

    display_name = _("Merge Visible Layers")

    def __init__(self, doc, **kwds):
        super(MergeVisibleLayers, self).__init__(doc, **kwds)
        self._nothing_initially_visible = False
        self._old_current_path = doc.layer_stack.current_path
        self._result_layer = None
        self._result_insert_path = None
        self._result_final_path = None
        self._paths_merged = None    # paths to merge (and remove)
        self._layers_merged = None   # zip()s with _paths_merged

    def redo(self):
        rootstack = self.doc.layer_stack
        # First time, we calculate the merged layer and cache it once.
        # Also store the paths to remove,
        # and calculate where to put the result of the merge.
        merged = self._result_layer
        if merged is None:
            self._result_insert_path = (len(rootstack),)
            self._paths_merged = []
            for path, layer in rootstack.walk(visible=True):
                if path[0] < self._result_insert_path[0]:
                    self._result_insert_path = (path[0],)
                self._paths_merged.append(path)
            # If nothing was visible, our job is easy
            if len(self._paths_merged) == 0:
                self._nothing_initially_visible = True
                logger.debug("MergeVisibleLayers: no visible layers")
                return
            # Otherwise, calculate and store the result
            merged = rootstack.layer_new_merge_visible()
            self._result_layer = merged
        # Every time around, remove the layers which were visible,
        # keeping refs to them in _paths_merged order.
        assert self._result_insert_path is not None
        assert self._paths_merged is not None
        logger.debug(
            "MergeVisibleLayers: remove paths %r",
            self._paths_merged,
        )
        self._layers_merged = []
        for removed_layer_path in reversed(self._paths_merged):
            removed_layer = rootstack.deeppop(removed_layer_path)
            self._layers_merged.insert(0, removed_layer)
        # The insert path always lies before the removed layers.
        logger.debug(
            "MergeVisibleLayers: insert merge result at %r",
            self._result_insert_path,
        )
        rootstack.deepinsert(self._result_insert_path, merged)
        # Not sure we need to record the final path,
        # isn't it always the same as the insert path?
        self._result_final_path = rootstack.deepindex(merged)
        rootstack.current_path = self._result_final_path

    def undo(self):
        if self._nothing_initially_visible:
            return
        # Remove the merged path
        rootstack = self.doc.layer_stack
        rootstack.deeppop(self._result_final_path)
        # Restore the previously removed paths
        assert len(self._paths_merged) == len(self._layers_merged)
        for path, layer in zip(self._paths_merged, self._layers_merged):
            rootstack.deepinsert(path, layer)
        self._layers_merged = None
        # Restore previous path selection.
        rootstack.current_path = self._old_current_path


class MergeLayerDown (Command):
    """Merge the current layer and the one below it into a new layer"""

    display_name = _("Merge Down")

    def __init__(self, doc, **kwds):
        super(MergeLayerDown, self).__init__(doc, **kwds)
        rootstack = doc.layer_stack
        self._upper_path = tuple(rootstack.current_path)
        self._lower_path = rootstack.get_merge_down_target(self._upper_path)
        self._upper_layer = None
        self._lower_layer = None
        self._merged_layer = None

    def redo(self):
        rootstack = self.doc.layer_stack
        merged = self._merged_layer
        if merged is None:
            merged = rootstack.layer_new_merge_down(self._upper_path)
            assert merged is not None
            self._merged_layer = merged
        self._lower_layer = rootstack.deeppop(self._lower_path)
        self._upper_layer = rootstack.deeppop(self._upper_path)
        rootstack.deepinsert(self._upper_path, merged)
        assert rootstack.deepindex(merged) == self._upper_path
        assert self._lower_layer is not None
        assert self._upper_layer is not None
        assert rootstack.deepget(self._upper_path) is merged
        rootstack.current_path = self._upper_path

    def undo(self):
        rootstack = self.doc.layer_stack
        merged = self._merged_layer
        removed = rootstack.deeppop(self._upper_path)
        assert removed is merged
        rootstack.deepinsert(self._upper_path, self._lower_layer)
        rootstack.deepinsert(self._upper_path, self._upper_layer)
        assert rootstack.deepget(self._upper_path) is self._upper_layer
        assert rootstack.deepget(self._lower_path) is self._lower_layer
        self._upper_layer = None
        self._lower_layer = None
        rootstack.current_path = self._upper_path


class NormalizeLayerMode (Command):
    """Normalize a layer's mode & opacity, incorporating its backdrop

    If the layer has any non-zero-alpha pixels, they will take on a
    ghost image of the its current backdrop as a result of this
    operation.
    """

    display_name = _("Normalize Layer Mode")

    def __init__(self, doc, layer=None, path=None, index=None, **kwds):
        super(NormalizeLayerMode, self).__init__(doc, **kwds)
        layers = self.doc.layer_stack
        self._path = layers.canonpath(layer=layer, path=path,
                                      index=index, usecurrent=True)
        self._old_layer = None
        self._old_current_path = None

    def redo(self):
        layers = self.doc.layer_stack
        self._old_current_path = layers.current_path
        parent_path, idx = self._path[:-1], self._path[-1]
        parent = layers.deepget(parent_path)
        self._old_layer = parent[idx]
        normalized = layers.layer_new_normalized(self._path)
        parent[idx] = normalized
        layers.current_path = self._path

    def undo(self):
        layers = self.doc.layer_stack
        parent_path, idx = self._path[:-1], self._path[-1]
        parent = layers.deepget(parent_path)
        parent[idx] = self._old_layer
        self._old_layer = None
        layers.current_path = self._old_current_path


class AddLayer (Command):
    """Inserts a layer into the layer stack.

    The layer can be supplied at construction time. Alternatively a
    constructor function or class can be passed in, along with a name
    and any other **kwds you need. The default class if neither is
    specified is the normal painting layer type.

    In both cases, the command object takes ownership of the layer.

    """

    def __init__(self, doc, insert_path, name=None,
                 layer_class=lib.layer.PaintingLayer,
                 layer=None, is_import=False, **kwds):
        super(AddLayer, self).__init__(doc, **kwds)
        self._insert_path = insert_path
        self._prev_currentlayer_path = None
        self._layer = layer or layer_class(name=name, **kwds)
        self._is_import = bool(is_import)

    @property
    def display_name(self):
        if self._is_import:
            tmpl = _("Import Layers")
        else:
            tmpl = _("Add {layer_default_name}")
        return tmpl.format(
            layer_default_name=self._layer.DEFAULT_NAME,
        )

    def redo(self):
        layers = self.doc.layer_stack
        self._prev_currentlayer_path = layers.get_current_path()
        layers.deepinsert(self._insert_path, self._layer)
        assert self._layer.name is not None
        inserted_path = layers.deepindex(self._layer)
        assert inserted_path is not None
        layers.set_current_path(inserted_path)

    def undo(self):
        layers = self.doc.layer_stack
        layers.deepremove(self._layer)
        layers.set_current_path(self._prev_currentlayer_path)
        self._prev_currentlayer_path = None


class RemoveLayer (Command):
    """Removes the current layer"""

    display_name = _("Remove Layer")

    def __init__(self, doc, **kwds):
        super(RemoveLayer, self).__init__(doc, **kwds)
        layers = self.doc.layer_stack
        assert layers.current_path
        self._unwanted_path = layers.current_path
        self._removed_layer = None
        self._replacement_layer = None

    def redo(self):
        assert self._removed_layer is None, "double redo()?"
        layers = self.doc.layer_stack
        path = layers.get_current_path()
        path_above = layers.path_above(path)
        self._removed_layer = layers.deeppop(self._unwanted_path)
        if len(layers) == 0:
            logger.debug("Removed last layer")
            if self.doc.CREATE_PAINTING_LAYER_IF_EMPTY:
                logger.debug("Replacing removed layer")
                repl = self._replacement_layer
                if repl is None:
                    repl = lib.layer.PaintingLayer()
                    self._replacement_layer = repl
                    repl.name = layers.get_unique_name(repl)
                layers.append(repl)
                layers.set_current_path((0,))
            assert self._unwanted_path == (0,)
        else:
            if not layers.deepget(path):
                if layers.deepget(path_above):
                    layers.set_current_path(path_above)
                else:
                    layers.set_current_path((0,))

    def undo(self):
        layers = self.doc.layer_stack
        if self._replacement_layer is not None:
            layers.deepremove(self._replacement_layer)
        layers.deepinsert(self._unwanted_path, self._removed_layer)
        layers.set_current_path(self._unwanted_path)
        self._removed_layer = None


class SelectLayer (Command):
    """Select a layer"""

    display_name = _("Select Layer")
    automatic_undo = True

    def __init__(self, doc, index=None, path=None, layer=None, **kwds):
        super(SelectLayer, self).__init__(doc, **kwds)
        layers = self.doc.layer_stack
        self.path = layers.canonpath(index=index, path=path, layer=layer)
        self.prev_path = layers.canonpath(path=layers.get_current_path())

    def redo(self):
        layers = self.doc.layer_stack
        layers.set_current_path(self.path)

    def undo(self):
        layers = self.doc.layer_stack
        layers.set_current_path(self.prev_path)


class MoveLayer (Command):
    """Moves a layer around the canvas

    Layer move commands are intended to be manipulated by the UI after
    creation, and before being committed to the command stack.  During
    this initial active move phase, `move_to()` repositions the
    reference point, and `process_move()` handles the effects of doing
    this in chunks so that the screen can be updated smoothly.  After
    the layer is committed to the command stack, the active move phase
    methods can no longer be used.
    """

    # TRANSLATORS: Command to move a layer in the horizontal plane,
    # TRANSLATORS: preserving its position in the stack.
    # TRANSLATORS: Note "in the horizontal plane" - not "horizontally".
    display_name = _("Move Layer")

    def __init__(self, doc, layer_path, x0, y0, **kwds):
        """Initializes, as an active layer move command

        :param doc: document to be moved
        :type doc: lib.document.Document
        :param layer_path: path of the layer to affect within doc
        :param float x0: Reference point X coordinate
        :param float y0: Reference point Y coordinate
        """
        super(MoveLayer, self). __init__(doc, **kwds)
        self._layer_path = layer_path
        layer = self.doc.layer_stack.deepget(layer_path)
        y0 = int(y0)
        self._x0 = x0
        self._y0 = y0
        self._move = layer.get_move(x0, y0)
        self._x = 0
        self._y = 0
        self._processing_complete = True

    ## Active moving phase

    def move_to(self, x, y):
        """Move the reference point to a new position

        :param x: New reference point X coordinate
        :param y: New reference point Y coordinate

        This is a higher-level wrapper around the raw layer and surface
        moving API, tailored for use by GUI code.
        """
        assert self._move is not None
        x = int(x)
        y = int(y)
        if (x, y) == (self._x, self._y):
            return
        self._x = x
        self._y = y
        dx = self._x - self._x0
        dy = self._y - self._y0
        self._move.update(dx, dy)
        self._processing_complete = False

    def process_move(self):
        """Process chunks of the updated move

        :returns: True if there are remaining chunks of work to do
        :rtype: bool

        This is a higher-level wrapper around the raw layer and surface
        moving API, tailored for use by GUI code.
        """
        assert self._move is not None
        more_needed = self._move.process()
        self._processing_complete = not more_needed
        return more_needed

    ## Command stack callbacks

    def redo(self):
        """Updates the document as needed when do()/redo() is invoked"""
        # The first time this is called, finish up the active move.
        # Doc has already been updated, and notifications were sent.
        if self._move is not None:
            assert self._processing_complete
            self._move.cleanup()
            self._move = None
            return
        # Any second invocation is always reversing a previous undo().
        # Need to do doc updates and send notifications this time.
        if (self._x, self._y) == (self._x0, self._y0):
            return
        layer = self.doc.layer_stack.deepget(self._layer_path)
        dx = self._x - self._x0
        dy = self._y - self._y0
        redraw_bboxes = layer.translate(dx, dy)
        self._notify_canvas_observers(redraw_bboxes)

    def undo(self):
        """Updates the document as needed when undo() is invoked"""
        # When called, this is always reversing a previous redo().
        # Update the doc and send notifications.
        assert self._move is None
        if (self._x, self._y) == (self._x0, self._y0):
            return
        layer = self.doc.layer_stack.deepget(self._layer_path)
        dx = self._x - self._x0
        dy = self._y - self._y0
        redraw_bboxes = layer.translate(-dx, -dy)
        self._notify_canvas_observers(redraw_bboxes)


class DuplicateLayer (Command):
    """Make an exact copy of the current layer"""

    display_name = _("Duplicate Layer")

    def __init__(self, doc, **kwds):
        super(DuplicateLayer, self).__init__(doc, **kwds)
        self._path = self.doc.layer_stack.current_path

    def redo(self):
        layers = self.doc.layer_stack
        layer_copy = deepcopy(layers.current)
        layers.deepinsert(self._path, layer_copy)
        assert layers.deepindex(layer_copy) == self._path
        layers.set_current_path(self._path)
        self._notify_canvas_observers([layer_copy.get_full_redraw_bbox()])

    def undo(self):
        layers = self.doc.layer_stack
        layers.deeppop(self._path)
        orig_layer = layers.deepget(self._path)
        self._notify_canvas_observers([orig_layer.get_full_redraw_bbox()])


class BubbleLayerUp (Command):
    """Move the current layer up through the stack"""

    display_name = _("Move Layer Up")

    def redo(self):
        layers = self.doc.layer_stack
        layers.bubble_layer_up(layers.current_path)

    def undo(self):
        layers = self.doc.layer_stack
        layers.bubble_layer_down(layers.current_path)


class BubbleLayerDown (Command):
    """Move the current layer down through the stack"""

    display_name = _("Move Layer Down")

    def redo(self):
        layers = self.doc.layer_stack
        layers.bubble_layer_down(layers.current_path)

    def undo(self):
        layers = self.doc.layer_stack
        layers.bubble_layer_up(layers.current_path)


class RestackLayer (Command):
    """Move a layer from one position in the stack to another

    Layer restacking operations allow layers to be moved inside other
    layers even if the target layer type doesn't permit sub-layers. In
    this case, a new parent layer stack is created::

      layer1            layer1
      targetlayer       newparent
      layer2        →    ├─ movedlayer
      movedlayer         └─ targetlayer
                        layer2

    This shows a move of path ``(3,)`` to the path ``(1, 0)``.
    """

    display_name = _("Move Layer in Stack")

    def __init__(self, doc, src_path, targ_path, **kwds):
        """Initialize with source and target paths

        :param tuple src_path: Valid source path
        :param tuple targ_path: Valid target path for the move

        This style of move requires the source path to exist at the time
        of creation, and for the target path to be a valid insertion
        path at the point the command is created. The target's parent
        path must exist too.
        """
        super(RestackLayer, self).__init__(doc, **kwds)
        src_path = tuple(src_path)
        targ_path = tuple(targ_path)
        rootstack = self.doc.layer_stack
        if lib.layer.path_startswith(targ_path, src_path):
            raise ValueError("Target path %r is inside source path %r"
                             % (targ_path, src_path))
        if len(targ_path) == 0:
            raise ValueError("Cannot move a layer to path ()")
        if rootstack.deepget(src_path) is None:
            raise ValueError("Source path %r does not exist"
                             % (src_path,))
        if rootstack.deepget(targ_path[:-1]) is None:
            raise ValueError("Parent of target path %r doesn't exist"
                             % (targ_path,))
        self._src_path = src_path
        self._src_path_after = None
        self._targ_path = targ_path
        self._new_parent = None

    def redo(self):
        """Perform the move"""
        src_path = self._src_path
        targ_path = self._targ_path
        rootstack = self.doc.layer_stack
        affected = []
        oldcurrent = rootstack.current
        # Replace src with a placeholder
        placeholder = lib.layer.PlaceholderLayer(name="moving")
        src = rootstack.deepget(src_path)
        src_parent = rootstack.deepget(src_path[:-1])
        src_index = src_path[-1]
        src_parent[src_index] = placeholder
        affected.append(src)
        # Do the insert
        targ_index = targ_path[-1]
        targ_parent = rootstack.deepget(targ_path[:-1])
        if isinstance(targ_parent, lib.layer.LayerStack):
            targ_parent.insert(targ_index, src)
        else:
            # The target path is a nonexistent path one level deeper
            # than an existing data layer. Need to create a new parent
            # for both the moved layer and the existing data layer.
            assert len(targ_path) > 1
            targ_parent_index = targ_path[-2]
            targ_gparent = rootstack.deepget(targ_path[:-2])
            container = lib.layer.LayerStack()
            container.name = rootstack.get_unique_name(container)
            targ_gparent[targ_parent_index] = container
            container.append(src)
            container.append(targ_parent)
            self._new_parent = container
            affected.append(targ_parent)
        # Remove placeholder
        rootstack.deepremove(placeholder)
        assert rootstack.deepindex(placeholder) is None
        self._src_path_after = rootstack.deepindex(src)
        assert self._src_path_after is not None
        # Current index mgt
        if oldcurrent is None:
            rootstack.current_path = (0,)
        else:
            rootstack.current_path = rootstack.deepindex(oldcurrent)
        # Issue redraws
        redraw_bboxes = [a.get_full_redraw_bbox() for a in affected]
        self._notify_canvas_observers(redraw_bboxes)

    def undo(self):
        """Unperform the move"""
        rootstack = self.doc.layer_stack
        affected = []
        src_path = self._src_path
        src_path_after = self._src_path_after
        oldcurrent = rootstack.current
        # Remove the layer that was moved
        if self._new_parent:
            assert len(self._new_parent) == 2
            assert (rootstack.deepget(src_path_after[:-1])
                    is self._new_parent)
            src = self._new_parent[0]
            oldleaf = self._new_parent[1]
            oldleaf_parent = rootstack.deepget(src_path_after[:-2])
            oldleaf_index = src_path_after[-2]
            oldleaf_parent[oldleaf_index] = oldleaf
            assert rootstack.deepindex(self._new_parent) is None
            self._new_parent = None
            affected.append(oldleaf)
        else:
            src = rootstack.deeppop(src_path_after)
        self._src_path_after = None
        # Insert it back where it came from
        rootstack.deepinsert(src_path, src)
        affected.append(src)
        # Current index mgt
        if oldcurrent is None:
            rootstack.current_path = (0,)
        else:
            rootstack.current_path = rootstack.deepindex(oldcurrent)
        # Redraws
        redraw_bboxes = [a.get_full_redraw_bbox() for a in affected]
        self._notify_canvas_observers(redraw_bboxes)


class RenameLayer (Command):
    """Renames a layer."""

    display_name = _("Rename Layer")

    def __init__(self, doc, name, layer=None, path=None, index=None,
                 **kwds):
        super(RenameLayer, self).__init__(doc, **kwds)
        layers = self.doc.layer_stack
        assert layers.current_path
        self._path = layers.canonpath(layer=layer, path=path, index=index,
                                      usecurrent=True)
        self._new_name = name
        self._old_name = None

    @property
    def layer(self):
        return self.doc.layer_stack.deepget(self._path)

    def redo(self):
        self._old_name = self.layer.name
        self.layer.name = self._new_name

    def undo(self):
        self.layer.name = self._old_name

    def update(self, name):
        self.layer.name = name
        self._new_name = name


class SetLayerVisibility (Command):
    """Sets the visibility status of a layer"""

    def __init__(self, doc, visible, layer=None, path=None, index=None,
                 **kwds):
        super(SetLayerVisibility, self).__init__(doc, **kwds)
        layers = self.doc.layer_stack
        self._path = layers.canonpath(layer=layer, path=path, index=index,
                                      usecurrent=True)
        self._new_visibility = visible
        self._old_visibility = None

    @property
    def layer(self):
        return self.doc.layer_stack.deepget(self._path)

    def redo(self):
        self._old_visibility = self.layer.visible
        self.layer.visible = self._new_visibility

    def undo(self):
        self.layer.visible = self._old_visibility

    def update(self, visible):
        self.layer.visible = visible
        self._new_visibility = visible

    @property
    def display_name(self):
        if self._new_visibility:
            return _("Make Layer Visible")
        else:
            return _("Make Layer Invisible")


class SetLayerLocked (Command):
    """Sets the locking status of a layer"""

    def __init__(self, doc, locked, layer=None, path=None, index=None,
                 **kwds):
        super(SetLayerLocked, self).__init__(doc, **kwds)
        self.new_locked = locked
        layers = self.doc.layer_stack
        self._path = layers.canonpath(layer=layer, path=path, index=index,
                                      usecurrent=True)

    @property
    def layer(self):
        return self.doc.layer_stack.deepget(self._path)

    def redo(self):
        self.old_locked = self.layer.locked
        self.layer.locked = self.new_locked
        redraw_bboxes = [self.layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)

    def undo(self):
        self.layer.locked = self.old_locked
        redraw_bboxes = [self.layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)

    def update(self, locked):
        self.layer.locked = locked
        self.new_locked = locked
        redraw_bboxes = [self.layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)

    @property
    def display_name(self):
        if self.new_locked:
            return _("Lock Layer")
        else:
            return _("Unlock Layer")


class SetLayerOpacity (Command):
    """Sets the opacity of a layer"""

    def __init__(self, doc, opacity, layer=None, path=None, index=None,
                 **kwds):
        super(SetLayerOpacity, self).__init__(doc, **kwds)
        layers = doc.layer_stack
        self._path = layers.canonpath(layer=layer, path=path, index=index,
                                      usecurrent=True)
        self._new_opacity = opacity
        self._old_opacity = None

    @property
    def display_name(self):
        percent = self._new_opacity * 100.0
        return _(u"Set Layer Opacity: %0.1f%%") % (percent,)

    @property
    def layer(self):
        return self.doc.layer_stack.deepget(self._path)

    def redo(self):
        layer = self.layer
        self._old_opacity = layer.opacity
        layer.opacity = self._new_opacity

    def update(self, opacity):
        layer = self.layer
        if layer.opacity == opacity:
            return
        self._new_opacity = opacity
        layer.opacity = opacity

    def undo(self):
        layer = self.layer
        layer.opacity = self._old_opacity


class SetLayerMode (Command):
    """Sets the combining mode for a layer"""

    def __init__(self, doc, mode, layer=None, path=None, index=None,
                 **kwds):
        super(SetLayerMode, self).__init__(doc, **kwds)
        layers = self.doc.layer_stack
        self._path = layers.canonpath(layer=layer, path=path, index=index,
                                      usecurrent=True)
        self._new_mode = mode
        self._old_mode = None
        self._old_opacity = None

    @property
    def display_name(self):
        info = lib.modes.MODE_STRINGS.get(self._new_mode)
        name = info and info[0] or _(u"Unknown Mode")
        return _(u"Set Layer Mode: %s") % (name,)

    @property
    def layer(self):
        return self.doc.layer_stack.deepget(self._path)

    def redo(self):
        layer = self.layer
        self._old_mode = layer.mode
        self._old_opacity = layer.opacity
        layer.mode = self._new_mode

    def undo(self):
        layer = self.layer
        layer.mode = self._old_mode
        layer.opacity = self._old_opacity


class SetFrameEnabled (Command):
    """Enable or disable the document frame"""

    @property
    def display_name(self):
        if self.after:
            return _("Enable Frame")
        else:
            return _("Disable Frame")

    def __init__(self, doc, enable, **kwds):
        super(SetFrameEnabled, self).__init__(doc, **kwds)
        self.before = None
        self.after = enable

    def redo(self):
        self.before = self.doc.frame_enabled
        self.doc.set_frame_enabled(self.after, user_initiated=False)

    def undo(self):
        self.doc.set_frame_enabled(self.before, user_initiated=False)


class UpdateFrame (Command):
    """Update frame dimensions"""

    display_name = _("Update Frame")

    def __init__(self, doc, frame, **kwds):
        super(UpdateFrame, self).__init__(doc, **kwds)
        self.new_frame = frame
        self.old_frame = None
        self.old_enabled = doc.get_frame_enabled()

    def redo(self):
        if self.old_frame is None:
            self.old_frame = self.doc.frame[:]
        self.doc.update_frame(*self.new_frame, user_initiated=False)
        self.doc.set_frame_enabled(True, user_initiated=False)

    def update(self, frame):
        assert self.old_frame is not None
        self.new_frame = frame
        self.doc.update_frame(*self.new_frame, user_initiated=False)
        self.doc.set_frame_enabled(True, user_initiated=False)

    def undo(self):
        assert self.old_frame is not None
        self.doc.update_frame(*self.old_frame, user_initiated=False)
        self.doc.set_frame_enabled(self.old_enabled, user_initiated=False)


class ExternalLayerEdit (Command):
    """An edit made in a external application"""

    display_name = _("Edit Layer Externally")

    def __init__(self, doc, layer, tmpfile, **kwds):
        super(ExternalLayerEdit, self).__init__(doc, **kwds)
        self._tmpfile = tmpfile
        self._layer_path = self.doc.layer_stack.canonpath(layer=layer)
        self._before = None
        self._after = None

    def redo(self):
        layer = self.doc.layer_stack.deepget(self._layer_path)
        if not self._before:
            self._before = layer.save_snapshot()
        if self._after:
            layer.load_snapshot(self._after)
        else:
            layer.load_from_external_edit_tempfile(self._tmpfile)
            self._after = layer.save_snapshot()

    def undo(self):
        layer = self.doc.layer_stack.deepget(self._layer_path)
        layer.load_snapshot(self._before)
