# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

import lib.layer
import helpers
from observable import event
import tiledsurface

import weakref
from gettext import gettext as _
from logging import getLogger
logger = getLogger(__name__)


## Command stack and action interface


class CommandStack (object):
    """Undo/redo stack"""

    def __init__(self):
        object.__init__(self)
        self.undo_stack = []
        self.redo_stack = []
        self.stack_updated()

    def __repr__(self):
        return ("<CommandStack undo_len=%d redo_len=%d>" %
                ( len(self.undo_stack), len(self.redo_stack), ))

    def clear(self):
        self._discard_undo()
        self._discard_redo()
        self.stack_updated()

    def _discard_undo(self):
        self.undo_stack = []

    def _discard_redo(self):
        self.redo_stack = []

    def do(self, command):
        self._discard_redo()
        command.redo()
        self.undo_stack.append(command)
        self.reduce_undo_history()
        self.stack_updated()

    def undo(self):
        if not self.undo_stack: return
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        self.stack_updated()
        return command

    def redo(self):
        if not self.redo_stack: return
        command = self.redo_stack.pop()
        command.redo()
        self.undo_stack.append(command)
        self.stack_updated()
        return command

    def reduce_undo_history(self):
        stack = self.undo_stack
        self.undo_stack = []
        steps = 0
        for item in reversed(stack):
            self.undo_stack.insert(0, item)
            if not item.automatic_undo:
                steps += 1
            if steps == 30: # and memory > ...
                break

    def get_last_command(self):
        if not self.undo_stack: return None
        return self.undo_stack[-1]

    def update_last_command(self, **kwargs):
        cmd = self.get_last_command()
        if cmd is None:
            return None
        cmd.update(**kwargs)
        self.stack_updated() # the display_name may have changed
        return cmd

    @event
    def stack_updated(self):
        """Event: command stack was updated"""
        pass

class Action (object):
    """An undoable, redoable action.

    Base class for all undo/redoable actions. Subclasses must implement the
    undo and redo methods.
    """
    automatic_undo = False
    display_name = _("Unknown Action")

    def __init__(self, doc):
        object.__init__(self)
        self.doc = weakref.proxy(doc)

    def __repr__(self):
        return "<%s>" % (self.display_name,)

    def redo(self):
        """Callback used to perform, or re-perform the Action"""
        raise NotImplementedError


    def undo(self):
        """Callback used to un-perform an already performed Action"""
        raise NotImplementedError


    def update(self, **kwargs):
        """In-place update on the tip of the undo stack.

        This method should update the model in the way specified in `**kwargs`.
        The interpretation of arguments is left to the concrete implementation.

        Updating the top Action on the command stack is used to prevent
        situations where an undo() followed by a redo() would result in
        multiple sendings of GdkEvents by code designed to keep interface state
        in sync with the model.

        """

        # Updating is used in situations where only the user's final choice of
        # a state such as layer visibility matters in the command-stream.
        # Creating a nice workflow for the user by using `undo()` then `do()`
        # with a replacement Action can sometimes cause GtkAction and
        # command.Action flurries or loops across multiple GdkEvent callbacks.
        #
        # This can make coding difficult elsewhere. For example,
        # GtkToggleActions must be kept in in sync with undoable boolean model
        # state, but even when an interlock or check is coded, the fact that
        # processing happens in multiple GtkEvent handlers can result in,
        # essentially, a toggle action which turns itself off immediately after
        # being toggled on. See https://gna.org/bugs/?20096 for a concrete
        # example.

        raise NotImplementedError


    # Utility functions
    def _notify_canvas_observers(self, layer_bboxes):
        """Notifies the document's canvas_observers for redraws, etc."""
        redraw_bbox = helpers.Rect()
        for layer_bbox in layer_bboxes:
            if layer_bbox.w == 0 and layer_bbox.h == 0:
                redraw_bbox = layer_bbox
                break
            else:
                redraw_bbox.expandToIncludeRect(layer_bbox)
        for func in self.doc.canvas_observers:
            func(*redraw_bbox)

    def _notify_document_observers(self):
        self.doc.call_doc_observers()

class Stroke (Action):
    """Completed stroke, i.e. some seconds of painting

    Stroke actions are only initialized (and performed) after the stroke is
    just completed and fully rendered.
    """

    display_name = _("Painting")

    def __init__(self, doc, stroke, snapshot_before):
        Action.__init__(self, doc)
        assert stroke.finished
        self.stroke = stroke
        # immutable; not used for drawing any more, just for inspection
        self.before = snapshot_before
        layer = self.doc.layer_stack.current
        layer.add_stroke(stroke, snapshot_before)
        # this snapshot will include the updated stroke list (modified by the
        # line above)
        self.after = layer.save_snapshot()

    def redo(self):
        layer = self.doc.layer_stack.current
        layer.load_snapshot(self.after)

    def undo(self):
        layer = self.doc.layer_stack.current
        layer.load_snapshot(self.before)


## Concrete command classes


class FloodFill (Action):
    """Flood-fill on the current layer"""

    display_name = _("Flood Fill")

    def __init__(self, doc, x, y, color, bbox, tolerance,
                 sample_merged, make_new_layer):
        Action.__init__(self, doc)
        self.x = x
        self.y = y
        self.color = color
        self.bbox = bbox
        self.tolerance = tolerance
        self.sample_merged = sample_merged
        self.make_new_layer = make_new_layer
        self.new_layer = None
        self.new_layer_path = None
        self.snapshot = None

    def redo(self):
        # Pick a source
        layers = self.doc.layer_stack
        if self.sample_merged:
            src_layer = layers
        else:
            src_layer = layers.current
        # Choose a target
        if self.make_new_layer:
            # Write to a new layer
            assert self.new_layer is None
            nl = lib.layer.PaintingLayer(rootstack=layers)
            nl.content_observers.append(self.doc.layer_modified_cb)
            nl.set_symmetry_axis(self.doc.get_symmetry_axis())
            self.new_layer = nl
            insert_path = list(layers.get_current_path())
            insert_path[-1] += 1
            layers.deepinsert(insert_path, nl)
            path = layers.deepindex(nl)
            self.new_layer_path = path
            layers.set_current_path(path)
            self._notify_document_observers()
            dst_layer = nl
        else:
            # Overwrite current, but snapshot 1st
            assert self.snapshot is None
            self.snapshot = layers.current.save_snapshot()
            dst_layer = layers.current
        # Fill connected areas of the source into the destination
        src_layer.flood_fill(self.x, self.y, self.color, self.bbox,
                             self.tolerance, dst_layer=dst_layer)

    def undo(self):
        layers = self.doc.layer_stack
        if self.make_new_layer:
            assert self.new_layer is not None
            path = layers.path_below(layers.get_current_path())
            layers.set_current_path(path)
            layers.deepremove(self.new_layer)
            redraw_bboxes = [self.new_layer.get_full_redraw_bbox()]
            self._notify_canvas_observers(redraw_bboxes)
            self._notify_document_observers()
            self.new_layer = None
            self.new_layer_path = None
        else:
            assert self.snapshot is not None
            layers.current.load_snapshot(self.snapshot)
            self.snapshot = None


class TrimLayer (Action):
    """Trim the current layer to the extent of the document frame"""

    display_name = _("Trim Layer")

    def __init__(self, doc):
        Action.__init__(self, doc)
        self.before = None

    def redo(self):
        layer = self.doc.layer_stack.current
        self.before = layer.save_snapshot()
        frame = self.doc.get_frame()
        layer.trim(frame)

    def undo(self):
        layer = self.doc.layer_stack.current
        layer.load_snapshot(self.before)


class ClearLayer(Action):
    display_name = _("Clear Layer")
    def __init__(self, doc):
        Action.__init__(self, doc)
    def redo(self):
        self.before = self.doc.layer.save_snapshot()
        self.doc.layer.clear()
        self._notify_document_observers()
    def undo(self):
        self.doc.layer.load_snapshot(self.before)
        del self.before
        self._notify_document_observers()

class LoadLayer(Action):
    display_name = _("Load Layer")
    def __init__(self, doc, surface):
        Action.__init__(self, doc)
        self.surface = surface
    def redo(self):
        layer = self.doc.layer
        self.before = layer.save_snapshot()
        layer.load_from_surface(self.surface)
    def undo(self):
        self.doc.layer.load_snapshot(self.before)
        del self.before

class MergeLayer (Action):
    """Merge the current layer down onto the layer below it"""

    display_name = _("Merge Down")

    def __init__(self, doc):
        Action.__init__(self, doc)
        layers = doc.layer_stack
        src_path = layers.current_path
        dst_path = layers.get_merge_down_target_path()
        assert dst_path is not None
        self._src_path = src_path
        self._dst_path = dst_path
        self._src_layer = None  # Will be removed
        self._src_sshot = None  #   ... after being permuted.
        self._dst_sshot = None  # Will just be permuted.

    def redo(self):
        layers = self.doc.layer_stack
        src = layers.deepget(self._src_path)
        dst = layers.deepget(self._dst_path)
        redraw_bboxes = [dst.get_full_redraw_bbox(),
                         src.get_full_redraw_bbox()]
        assert layers.current == src
        assert src is not dst
        # Snapshot
        self._src_layer = src
        self._src_sshot = src.save_snapshot()
        self._dst_sshot = dst.save_snapshot()
        # Normalize mode and opacity before merging
        src_bg_func = layers.get_backdrop_func(self._src_path)
        src.normalize_mode(src_bg_func)
        dst_bg_func = layers.get_backdrop_func(self._dst_path)
        dst.normalize_mode(dst_bg_func)
        # Merge down
        dst.merge_down_from(src)
        # Remove and select layer below
        layers.deeppop(self._src_path)
        layers.set_current_path(self._dst_path)
        # Notify
        assert layers.current == dst
        self._notify_document_observers()
        self._notify_canvas_observers(redraw_bboxes)

    def undo(self):
        layers = self.doc.layer_stack
        # Reinsert and select removed layer
        src = self._src_layer
        dst = layers.deepget(self._dst_path)
        redraw_bboxes = [dst.get_full_redraw_bbox(),
                         src.get_full_redraw_bbox()]
        layers.deepinsert(self._src_path, src)
        layers.set_current_path(self._src_path)
        # Restore the prior states for the merged layers
        src.load_snapshot(self._src_sshot)
        dst.load_snapshot(self._dst_sshot)
        # Cleanup
        self._src_layer = None
        self._src_sshot = None
        self._dst_sshot = None
        # Notify
        assert layers.current == src
        self._notify_document_observers()
        self._notify_canvas_observers(redraw_bboxes)

class NormalizeLayerMode (Action):
    """Normalize a layer's mode & opacity, incorporating its backdrop

    If the layer has any non-zero-alpha pixels, they will take on a ghost image
    of the its current backdrop as a result of this operation.
    """

    display_name = _("Normalize Layer Mode")

    def __init__(self, doc, layer=None, path=None, index=None):
        Action.__init__(self, doc)
        layers = self.doc.layer_stack
        self._path = layers.canonpath(layer=layer, path=path, index=index,
                                      usecurrent=True)
    def redo(self):
        layers = self.doc.layer_stack
        layer = layers.deepget(self._path)
        self._sshot_before = layer.save_snapshot()
        bg_func = layers.get_backdrop_func(self._path)
        layer.normalize_mode(bg_func)
        self._notify_document_observers()
        self._notify_canvas_observers([layer.get_full_redraw_bbox()])

    def undo(self):
        layers = self.doc.layer_stack
        layer = layers.deepget(self._path)
        layer.load_snapshot(self._sshot_before)
        self._sshot_before = None
        self._notify_document_observers()
        self._notify_canvas_observers([layer.get_full_redraw_bbox()])

class AddLayer (Action):
    """Creates a new layer, and inserts it into the layer stack"""

    display_name = _("Add Layer")

    def __init__(self, doc, insert_path, name=''):
        Action.__init__(self, doc)
        layers = doc.layer_stack
        self.insert_path = insert_path
        self.prev_currentlayer_path = None
        self.layer = lib.layer.PaintingLayer(name=name, rootstack=layers)
        self.layer.content_observers.append(self.doc.layer_modified_cb)
        self.layer.set_symmetry_axis(self.doc.get_symmetry_axis())

    def redo(self):
        layers = self.doc.layer_stack
        self.prev_currentlayer_path = layers.get_current_path()
        layers.deepinsert(self.insert_path, self.layer)
        inserted_path = layers.deepindex(self.layer)
        assert inserted_path is not None
        layers.set_current_path(inserted_path)
        self._notify_canvas_observers([self.layer.get_full_redraw_bbox()])
        self._notify_document_observers()

    def undo(self):
        layers = self.doc.layer_stack
        layers.deepremove(self.layer)
        layers.set_current_path(self.prev_currentlayer_path)
        self._notify_canvas_observers([self.layer.get_full_redraw_bbox()])
        self._notify_document_observers()


class RemoveLayer (Action):
    """Removes a layer, replacing it with a new one if it was the last"""

    display_name = _("Remove Layer")

    def __init__(self, doc, layer=None):
        Action.__init__(self, doc)
        layers = self.doc.layer_stack
        self.unwanted_path = layers.canonpath(layer=layer, usecurrent=True)
        self.removed_layer = None
        self.replacement_layer = None

    def redo(self):
        assert self.removed_layer is None, "double redo()?"
        layers = self.doc.layer_stack
        path = layers.get_current_path()
        path_below = layers.path_below(path)
        self.removed_layer = layers.deeppop(self.unwanted_path)
        if len(layers) == 0:
            logger.debug("Removed last layer, replacing it")
            repl = self.replacement_layer
            if repl is None:
                repl = lib.layer.PaintingLayer(rootstack=layers)
                repl.content_observers.append(self.doc.layer_modified_cb)
                repl.set_symmetry_axis(self.doc.get_symmetry_axis())
                self.replacement_layer = repl
            layers.append(repl)
            layers.set_current_path((0,))
            assert self.unwanted_path == (0,)
        else:
            if not layers.deepget(path):
                if layers.deepget(path_below):
                    layers.set_current_path(path_below)
                else:
                    layers.set_current_path((0,))
        redraw_bboxes = [self.removed_layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()

    def undo(self):
        layers = self.doc.layer_stack
        if self.replacement_layer is not None:
            layers.deepremove(self.replacement_layer)
        layers.deepinsert(self.unwanted_path, self.removed_layer)
        layers.set_current_path(self.unwanted_path)
        redraw_bboxes = [self.removed_layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()
        self.removed_layer = None


class SelectLayer (Action):
    """Select a layer"""

    display_name = _("Select Layer")
    automatic_undo = True

    def __init__(self, doc, index=None, path=None, layer=None):
        Action.__init__(self, doc)
        layers = self.doc.layer_stack
        self.path = layers.canonpath(index=index, path=path, layer=layer)
        self.prev_path = layers.canonpath(path=layers.get_current_path())

    def redo(self):
        layers = self.doc.layer_stack
        layers.set_current_path(self.path)
        self._notify_document_observers()

    def undo(self):
        layers = self.doc.layer_stack
        layers.set_current_path(self.prev_path)
        self._notify_document_observers()

class MoveLayer(Action):
    """Moves a layer around the canvas"""

    display_name = _("Move Layer on Canvas")
    # NOT "Move Layer" for now - old translatable string with different sense

    def __init__(self, doc, layer_idx, dx, dy, ignore_first_redo=True):
        Action.__init__(self, doc)
        self.layer_idx = layer_idx
        self.dx = dx
        self.dy = dy
        self.ignore_first_redo = ignore_first_redo

    def redo(self):
        layer = self.doc.layers[self.layer_idx]
        redraw_bboxes = [layer.get_full_redraw_bbox()]
        if self.ignore_first_redo:
            # these are typically created interactively, after
            # the entire layer has been moved
            self.ignore_first_redo = False
        else:
            layer.translate(self.dx, self.dy)
            redraw_bboxes.append(layer.get_full_redraw_bbox())
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()

    def undo(self):
        layer = self.doc.layers[self.layer_idx]
        redraw_bboxes = [layer.get_full_redraw_bbox()]
        layer.translate(-self.dx, -self.dy)
        redraw_bboxes.append(layer.get_full_redraw_bbox())
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()


class DuplicateLayer (Action):
    """Make an exact copy of the current layer"""

    display_name = _("Duplicate Layer")

    def __init__(self, doc):
        Action.__init__(self, doc)
        self._path = self.doc.layer_stack.current_path

    def redo(self):
        layers = self.doc.layer_stack
        layer_copy = layers.current.copy()
        layer_copy.assign_unique_name(layers.get_names())
        layers.deepinsert(self._path, layer_copy)
        assert layers.deepindex(layer_copy) == self._path
        self._notify_canvas_observers([layer_copy.get_full_redraw_bbox()])
        self._notify_document_observers()

    def undo(self):
        layers = self.doc.layer_stack
        layer_copy = layers.deeppop(self._path)
        orig_layer = layers.deepget(self._path)
        self._notify_canvas_observers([orig_layer.get_full_redraw_bbox()])
        self._notify_document_observers()


class BubbleLayerUp (Action):
    """Move a layer up through the stack, preserving its tree structure"""

    display_name = _("Move Layer Up")

    def redo(self):
        layers = self.doc.layer_stack
        current_layer = layers.current
        if layers.bubble_layer_up(layers.current_path):
            self.doc.select_layer(layer=current_layer, user_initiated=False)
            redraw_bboxes = [current_layer.get_full_redraw_bbox()]
            self._notify_canvas_observers(redraw_bboxes)

    def undo(self):
        layers = self.doc.layer_stack
        current_layer = layers.current
        if layers.bubble_layer_down(layers.current_path):
            self.doc.select_layer(layer=current_layer, user_initiated=False)
            redraw_bboxes = [current_layer.get_full_redraw_bbox()]
            self._notify_canvas_observers(redraw_bboxes)


class BubbleLayerDown (Action):
    """Move a layer down through the stack, preserving its tree structure"""

    display_name = _("Move Layer Down")

    def redo(self):
        layers = self.doc.layer_stack
        current_layer = layers.current
        if layers.bubble_layer_down(layers.current_path):
            self.doc.select_layer(layer=current_layer, user_initiated=False)
            redraw_bboxes = [current_layer.get_full_redraw_bbox()]
            self._notify_canvas_observers(redraw_bboxes)

    def undo(self):
        layers = self.doc.layer_stack
        current_layer = layers.current
        if layers.bubble_layer_up(layers.current_path):
            self.doc.select_layer(layer=current_layer, user_initiated=False)
            redraw_bboxes = [current_layer.get_full_redraw_bbox()]
            self._notify_canvas_observers(redraw_bboxes)


class ReorderLayerInStack (Action):
    """Move a layer from one position in the layer stack to another"""

    display_name = _("Move Layer in Stack")

    def __init__(self, doc, old_path, new_path):
        Action.__init__(self, doc)
        self._old_path = old_path
        self._new_path = new_path
        self._new_parent = None

    def redo(self):
        layers = self.doc.layer_stack
        affected_layers = []
        current_layer = layers.current
        moved_layer = layers.deeppop(self._old_path)
        # There's a special case when reparenting into a target layer
        # which is not a sub-stack. The UI allows it, but we must make
        # a LayerStack to house both layers.
        if self._old_path[:-1] != self._new_path[:-1]:
            # Moving from one parent to another
            parent_path = self._new_path[:-1]
            parent = layers.deepget(parent_path)
            assert parent is not None, \
                    "parent path %r identifies nothing" % (parent_path,)
            if not isinstance(parent, lib.layer.LayerStack):
                # Make a new parent
                assert self._new_path[-1] == 0
                sibling = parent
                parent = lib.layer.LayerStack(rootstack=layers)
                layers.deepinsert(parent_path, parent)
                layers.deepremove(sibling)
                parent.append(sibling)
                parent.append(moved_layer)
                affected_layers = [parent, sibling, moved_layer]
                self._new_parent = parent
        # Otherwise, we're either reparenting into an existing LayerStack
        # or moving within the same layer.
        if not affected_layers:
            layers.deepinsert(self._new_path, moved_layer)
            affected_layers = [moved_layer]
        # Select the previously selected layer, and notify
        self.doc.select_layer(layer=current_layer, user_initiated=False)
        redraw_bboxes = [l.get_full_redraw_bbox() for l in affected_layers]
        self._notify_canvas_observers(redraw_bboxes)

    def undo(self):
        layers = self.doc.layer_stack
        affected_layers = []
        current_layer = self.doc.layer_stack.current
        if self._new_parent is not None:
            parent_path = layers.deepindex(self._new_parent)
            sibling, moved_layer = tuple(self._new_parent)
            self._new_parent.remove(sibling)
            self._new_parent.remove(moved_layer)
            layers.deepinsert(parent_path, sibling)
            layers.deepremove(self._new_parent)
            layers.deepinsert(self._old_path, moved_layer)
            if current_layer is self._new_parent:
                current_layer = moved_layer
            affected_layers = [sibling, moved_layer]
            assert self._new_parent not in affected_layers
            self._new_parent = None
        else:
            moved_layer = self.doc.layer_stack.deeppop(self._new_path)
            layers.deepinsert(self._old_path, moved_layer)
            affected_layers = [moved_layer]
        assert current_layer is not None
        self.doc.select_layer(layer=current_layer, user_initiated=False)
        redraw_bboxes = [l.get_full_redraw_bbox() for l in affected_layers]
        self._notify_canvas_observers(redraw_bboxes)


class RenameLayer (Action):
    """Renames a layer"""

    display_name = _("Rename Layer")

    def __init__(self, doc, name, layer=None, path=None, index=None):
        Action.__init__(self, doc)
        self.new_name = name
        layers = self.doc.layer_stack
        self._path = layers.canonpath(layer=layer, path=path, index=index,
                                      usecurrent=True)
    @property
    def layer(self):
        return self.doc.layer_stack.deepget(self._path)

    def redo(self):
        self.old_name = self.layer.name
        self.layer.name = self.new_name
        self._notify_document_observers()

    def undo(self):
        self.layer.name = self.old_name
        self._notify_document_observers()


class SetLayerVisibility (Action):
    """Sets the visibility status of a layer"""

    def __init__(self, doc, visible, layer=None, path=None, index=None):
        Action.__init__(self, doc)
        self.new_visibility = visible
        layers = self.doc.layer_stack
        self._path = layers.canonpath(layer=layer, path=path, index=index,
                                      usecurrent=True)
    @property
    def layer(self):
        return self.doc.layer_stack.deepget(self._path)

    def redo(self):
        self.old_visibility = self.layer.visible
        self.layer.visible = self.new_visibility
        redraw_bboxes = [self.layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()

    def undo(self):
        self.layer.visible = self.old_visibility
        redraw_bboxes = [self.layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()

    def update(self, visible):
        self.layer.visible = visible
        self.new_visibility = visible
        redraw_bboxes = [self.layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()

    @property
    def display_name(self):
        if self.new_visibility:
            return _("Make Layer Visible")
        else:
            return _("Make Layer Invisible")

class SetLayerLocked (Action):
    """Sets the locking status of a layer"""

    def __init__(self, doc, locked, layer=None, path=None, index=None):
        Action.__init__(self, doc)
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
        self._notify_document_observers()

    def undo(self):
        self.layer.locked = self.old_locked
        redraw_bboxes = [self.layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()

    def update(self, locked):
        self.layer.locked = locked
        self.new_locked = locked
        redraw_bboxes = [self.layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()

    @property
    def display_name(self):
        if self.new_locked:
            return _("Lock Layer")
        else:
            return _("Unlock Layer")


class SetLayerOpacity (Action):
    """Sets the opacity of a layer"""

    display_name = _("Change Layer Opacity")

    def __init__(self, doc, opacity, layer=None, path=None, index=None):
        Action.__init__(self, doc)
        self.new_opacity = opacity
        layers = doc.layer_stack
        self._path = layers.canonpath(layer=layer, path=path, index=index,
                                      usecurrent=True)
    @property
    def layer(self):
        return self.doc.layer_stack.deepget(self._path)

    def redo(self):
        previous_effective_opacity = self.layer.effective_opacity
        self.old_opacity = self.layer.opacity
        self.layer.opacity = self.new_opacity
        self._notify_document_observers()
        redraw_bboxes = [self.layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)

    def undo(self):
        previous_effective_opacity = self.layer.effective_opacity
        self.layer.opacity = self.old_opacity
        self._notify_document_observers()
        redraw_bboxes = [self.layer.get_full_redraw_bbox()]
        self._notify_canvas_observers(redraw_bboxes)


class SetLayerMode (Action):
    """Sets the combining mode for a layer"""

    display_name = _("Change Layer Mode")

    def __init__(self, doc, mode, layer=None, path=None, index=None):
        Action.__init__(self, doc)
        self.new_mode = mode
        layers = self.doc.layer_stack
        self._path = layers.canonpath(layer=layer, path=path, index=index,
                                      usecurrent=True)

    def redo(self):
        layer = self.doc.layer_stack.deepget(self._path)
        self.old_mode = layer.mode
        redraw_bboxes = [layer.get_full_redraw_bbox()]
        layer.mode = self.new_mode
        redraw_bboxes.append(layer.get_full_redraw_bbox())
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()

    def undo(self):
        layer = self.doc.layer_stack.deepget(self._path)
        redraw_bboxes = [layer.get_full_redraw_bbox()]
        layer.mode = self.old_mode
        redraw_bboxes.append(layer.get_full_redraw_bbox())
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()


class SetLayerStackIsolated (Action):
    """Sets a layer stack's isolated flag"""

    display_name = _("Alter Layer Group's Isolation")

    def __init__(self, doc, isolated, layer=None, path=None, index=None):
        Action.__init__(self, doc)
        self._new_state = isolated
        self._old_state = None
        layers = self.doc.layer_stack
        self._path = layers.canonpath(layer=layer, path=path, index=index,
                                      usecurrent=True)

    def redo(self):
        stack = self.doc.layer_stack.deepget(self._path)
        assert isinstance(stack, lib.layer.LayerStack)
        redraw_bboxes = [stack.get_full_redraw_bbox()]
        self._old_state = stack.isolated
        stack.isolated = self._new_state
        redraw_bboxes.append(stack.get_full_redraw_bbox())
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()

    def undo(self):
        stack = self.doc.layer_stack.deepget(self._path)
        assert isinstance(stack, lib.layer.LayerStack)
        redraw_bboxes = [stack.get_full_redraw_bbox()]
        stack.isolated = self._old_state
        self._old_state = None
        redraw_bboxes.append(stack.get_full_redraw_bbox())
        self._notify_canvas_observers(redraw_bboxes)
        self._notify_document_observers()

    def update(self, isolated):
        self._new_state = isolated
        self.redo()


class SetFrameEnabled (Action):
    """Enable or disable the document frame"""

    @property
    def display_name(self):
        if self.after:
            return _("Enable Frame")
        else:
            return _("Disable Frame")

    def __init__(self, doc, enable):
        Action.__init__(self, doc)
        self.before = None
        self.after = enable

    def redo(self):
        self.before = self.doc.frame_enabled
        self.doc.set_frame_enabled(self.after, user_initiated=False)

    def undo(self):
        self.doc.set_frame_enabled(self.before, user_initiated=False)


class UpdateFrame (Action):
    """Update frame dimensions"""

    display_name = _("Update Frame")

    def __init__(self, doc, frame):
        Action.__init__(self, doc)
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

