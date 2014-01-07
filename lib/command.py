# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

import layer
import helpers
from observable import event

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
        return "<CommandStack\n  <Undo len=%d last3=%r>\n" \
                "  <Redo len=%d last3=%r> >" % (
                    len(self.undo_stack), self.undo_stack[-3:],
                    len(self.redo_stack), self.redo_stack[:3],  )

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
    def _notify_canvas_observers(self, affected_layers):
        bbox = helpers.Rect()
        for layer in affected_layers:
            layer_bbox = layer.get_bbox()
            bbox.expandToIncludeRect(layer_bbox)
        for func in self.doc.canvas_observers:
            func(*bbox)

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
            src_layer = layer.PaintingLayer()
            for l in layers:
                l.merge_into(src_layer, strokemap=False)
        else:
            src_layer = layers.current
        # Choose a target
        if self.make_new_layer:
            # Write to a new layer
            assert self.new_layer is None
            nl = layer.PaintingLayer(rootstack=layers)
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
            self._notify_canvas_observers([self.new_layer])
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
        self.before = self.doc.layer.save_snapshot()
        frame = self.doc.get_frame()
        self.doc.layer.trim(frame)

    def undo(self):
        self.doc.layer.load_snapshot(self.before)


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
    def __init__(self, doc, tiledsurface):
        Action.__init__(self, doc)
        self.tiledsurface = tiledsurface
    def redo(self):
        layer = self.doc.layer
        self.before = layer.save_snapshot()
        layer.load_from_surface(self.tiledsurface)
    def undo(self):
        self.doc.layer.load_snapshot(self.before)
        del self.before

class MergeLayer(Action):
    """merge the current layer into dst"""
    display_name = _("Merge Layers")
    def __init__(self, doc, dst_idx):
        Action.__init__(self, doc)
        self.dst_layer = self.doc.layers[dst_idx]
        self.normalize_src = ConvertLayerToNormalMode(doc, doc.layer)
        self.normalize_dst = ConvertLayerToNormalMode(doc, self.dst_layer)
        self.select_dst = SelectLayer(doc, dst_idx)
        self.remove_src = RemoveLayer(doc)
    def redo(self):
        self.normalize_src.redo()
        self.normalize_dst.redo()
        self.dst_before = self.dst_layer.save_snapshot()
        assert self.doc.layer is not self.dst_layer
        self.doc.layer.merge_into(self.dst_layer)
        self.remove_src.redo()
        self.select_dst.redo()
        self._notify_document_observers()
    def undo(self):
        self.select_dst.undo()
        self.remove_src.undo()
        self.dst_layer.load_snapshot(self.dst_before)
        del self.dst_before
        self.normalize_dst.undo()
        self.normalize_src.undo()
        self._notify_document_observers()

class ConvertLayerToNormalMode(Action):
    display_name = _("Convert Layer Mode")
    def __init__(self, doc, layer):
        Action.__init__(self, doc)
        self.layer = layer
        self.set_normal_mode = SetLayerCompositeOp(doc, 'svg:src-over', layer)
        self.set_opacity = SetLayerOpacity(doc, 1.0, layer)
    def redo(self):
        self.before = self.layer.save_snapshot()
        prev_idx = self.doc.layer_idx
        self.doc.layer_idx = self.doc.layers.index(self.layer)
        get_bg = self.doc.get_rendered_image_behind_current_layer
        self.layer.convert_to_normal_mode(get_bg)
        self.doc.layer_idx = prev_idx
        self.set_normal_mode.redo()
        self.set_opacity.redo()
    def undo(self):
        self.set_opacity.undo()
        self.set_normal_mode.undo()
        self.layer.load_snapshot(self.before)
        del self.before

class AddLayer (Action):
    """Creates a new layer, and inserts it into the layer stack"""

    display_name = _("Add Layer")

    def __init__(self, doc, insert_path, name=''):
        Action.__init__(self, doc)
        layers = doc.layer_stack
        self.insert_path = insert_path
        self.prev_currentlayer_path = None
        self.layer = layer.PaintingLayer(name=name, rootstack=layers)
        self.layer.content_observers.append(self.doc.layer_modified_cb)
        self.layer.set_symmetry_axis(self.doc.get_symmetry_axis())

    def redo(self):
        layers = self.doc.layer_stack
        self.prev_currentlayer_path = layers.get_current_path()
        layers.deepinsert(self.insert_path, self.layer)
        inserted_path = layers.deepindex(self.layer)
        assert inserted_path is not None
        layers.set_current_path(inserted_path)
        self._notify_document_observers()

    def undo(self):
        layers = self.doc.layer_stack
        layers.deepremove(self.layer)
        layers.set_current_path(self.prev_currentlayer_path)
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
                repl = layer.PaintingLayer(rootstack=layers)
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
        self._notify_canvas_observers([self.removed_layer])
        self._notify_document_observers()

    def undo(self):
        layers = self.doc.layer_stack
        if self.replacement_layer is not None:
            layers.deepremove(self.replacement_layer)
        layers.deepinsert(self.unwanted_path, self.removed_layer)
        layers.set_current_path(self.unwanted_path)
        self._notify_canvas_observers([self.removed_layer])
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
        if self.ignore_first_redo:
            # these are typically created interactively, after
            # the entire layer has been moved
            self.ignore_first_redo = False
        else:
            layer.translate(self.dx, self.dy)
        self._notify_canvas_observers([layer])
        self._notify_document_observers()
    def undo(self):
        layer = self.doc.layers[self.layer_idx]
        layer.translate(-self.dx, -self.dy)
        self._notify_canvas_observers([layer])
        self._notify_document_observers()

class ReorderSingleLayer(Action):
    display_name = _("Reorder Layer in Stack")
    def __init__(self, doc, was_idx, new_idx, select_new=False):
        Action.__init__(self, doc)
        self.was_idx = was_idx
        self.new_idx = new_idx
        self.select_new = select_new
    def redo(self):
        moved_layer = self.doc.layers[self.was_idx]
        self.doc.layers.remove(moved_layer)
        self.doc.layers.insert(self.new_idx, moved_layer)
        if self.select_new:
            self.was_selected = self.doc.layer_idx
            self.doc.layer_idx = self.new_idx
        self._notify_canvas_observers([moved_layer])
        self._notify_document_observers()
    def undo(self):
        moved_layer = self.doc.layers[self.new_idx]
        self.doc.layers.remove(moved_layer)
        self.doc.layers.insert(self.was_idx, moved_layer)
        if self.select_new:
            self.doc.layer_idx = self.was_selected
            self.was_selected = None
        self._notify_canvas_observers([moved_layer])
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
        self._notify_canvas_observers([layer_copy])
        self._notify_document_observers()

    def undo(self):
        layers = self.doc.layer_stack
        layer_copy = layers.deeppop(self._path)
        original_layer = layers.deepget(self._path)
        self._notify_canvas_observers([original_layer])
        self._notify_document_observers()


class BubbleLayerUp (Action):
    """Move a layer up through the stack, preserving its tree structure"""

    display_name = _("Move Layer Up")

    def redo(self):
        layers = self.doc.layer_stack
        current_layer = layers.current
        if layers.bubble_layer_up(layers.current_path):
            self.doc.select_layer(layer=current_layer, user_initiated=False)
            self._notify_canvas_observers([current_layer])

    def undo(self):
        layers = self.doc.layer_stack
        current_layer = layers.current
        if layers.bubble_layer_down(layers.current_path):
            self.doc.select_layer(layer=current_layer, user_initiated=False)
            self._notify_canvas_observers([current_layer])


class BubbleLayerDown (Action):
    """Move a layer down through the stack, preserving its tree structure"""

    display_name = _("Move Layer Down")

    def redo(self):
        layers = self.doc.layer_stack
        current_layer = layers.current
        if layers.bubble_layer_down(layers.current_path):
            self.doc.select_layer(layer=current_layer, user_initiated=False)
            self._notify_canvas_observers([current_layer])

    def undo(self):
        layers = self.doc.layer_stack
        current_layer = layers.current
        if layers.bubble_layer_up(layers.current_path):
            self.doc.select_layer(layer=current_layer, user_initiated=False)
            self._notify_canvas_observers([current_layer])


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
            if not isinstance(parent, layer.LayerStack):
                # Make a new parent
                assert self._new_path[-1] == 0
                sibling = parent
                parent = layer.LayerStack(rootstack=layers)
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
        self._notify_canvas_observers(affected_layers)

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
        self._notify_canvas_observers(affected_layers)


class RenameLayer(Action):
    display_name = _("Rename Layer")
    def __init__(self, doc, name, layer):
        Action.__init__(self, doc)
        self.new_name = name
        self.layer = layer
    def redo(self):
        self.old_name = self.layer.name
        self.layer.name = self.new_name
        self._notify_document_observers()
    def undo(self):
        self.layer.name = self.old_name
        self._notify_document_observers()

class SetLayerVisibility(Action):
    def __init__(self, doc, visible, layer):
        Action.__init__(self, doc)
        self.new_visibility = visible
        self.layer = layer
    def redo(self):
        self.old_visibility = self.layer.visible
        self.layer.visible = self.new_visibility
        self._notify_canvas_observers([self.layer])
        self._notify_document_observers()
    def undo(self):
        self.layer.visible = self.old_visibility
        self._notify_canvas_observers([self.layer])
        self._notify_document_observers()
    def update(self, visible):
        self.layer.visible = visible
        self.new_visibility = visible
        self._notify_canvas_observers([self.layer])
        self._notify_document_observers()
    @property
    def display_name(self):
        if self.new_visibility:
            return _("Make Layer Visible")
        else:
            return _("Make Layer Invisible")

class SetLayerLocked (Action):
    def __init__(self, doc, locked, layer):
        Action.__init__(self, doc)
        self.new_locked = locked
        self.layer = layer
    def redo(self):
        self.old_locked = self.layer.locked
        self.layer.locked = self.new_locked
        self._notify_canvas_observers([self.layer])
        self._notify_document_observers()
    def undo(self):
        self.layer.locked = self.old_locked
        self._notify_canvas_observers([self.layer])
        self._notify_document_observers()
    def update(self, locked):
        self.layer.locked = locked
        self.new_locked = locked
        self._notify_canvas_observers([self.layer])
        self._notify_document_observers()
    @property
    def display_name(self):
        if self.new_locked:
            return _("Lock Layer")
        else:
            return _("Unlock Layer")

class SetLayerOpacity(Action):
    display_name = _("Change Layer Visibility")
    def __init__(self, doc, opacity, layer=None):
        Action.__init__(self, doc)
        self.new_opacity = opacity
        self.layer = layer
    def redo(self):
        if self.layer:
            l = self.layer
        else:
            l = self.doc.layer
        previous_effective_opacity = l.effective_opacity
        self.old_opacity = l.opacity
        l.opacity = self.new_opacity
        if l.effective_opacity != previous_effective_opacity:
            self._notify_canvas_observers([l])
        self._notify_document_observers()
    def undo(self):
        if self.layer:
            l = self.layer
        else:
            l = self.doc.layer
        previous_effective_opacity = l.effective_opacity
        l.opacity = self.old_opacity
        if l.effective_opacity != previous_effective_opacity:
            self._notify_canvas_observers([l])
        self._notify_document_observers()

class SetLayerCompositeOp(Action):
    display_name = _("Change Layer Blending Mode")
    def __init__(self, doc, compositeop, layer=None):
        Action.__init__(self, doc)
        self.new_compositeop = compositeop
        self.layer = layer
    def redo(self):
        if self.layer:
            l = self.layer
        else:
            l = self.doc.layer
        self.old_compositeop = l.compositeop
        l.compositeop = self.new_compositeop
        self._notify_canvas_observers([l])
        self._notify_document_observers()
    def undo(self):
        if self.layer:
            l = self.layer
        else:
            l = self.doc.layer
        l.compositeop = self.old_compositeop
        self._notify_canvas_observers([l])
        self._notify_document_observers()


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

