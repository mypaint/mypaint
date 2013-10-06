# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import layer
import helpers
from gettext import gettext as _

class CommandStack:
    def __init__(self):
        self.call_before_action = []
        self.stack_observers = []
        self.clear()

    def __repr__(self):
        return "<CommandStack\n  <Undo len=%d last3=%r>\n" \
                "  <Redo len=%d last3=%r> >" % (
                    len(self.undo_stack), self.undo_stack[-3:],
                    len(self.redo_stack), self.redo_stack[:3],  )

    def clear(self):
        self.undo_stack = []
        self.redo_stack = []
        self.notify_stack_observers()

    def do(self, command):
        for f in self.call_before_action: f()
        self.redo_stack = [] # discard
        command.redo()
        self.undo_stack.append(command)
        self.reduce_undo_history()
        self.notify_stack_observers()

    def undo(self):
        if not self.undo_stack: return
        for f in self.call_before_action: f()
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        self.notify_stack_observers()
        return command

    def redo(self):
        if not self.redo_stack: return
        for f in self.call_before_action: f()
        command = self.redo_stack.pop()
        command.redo()
        self.undo_stack.append(command)
        self.notify_stack_observers()
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
        self.notify_stack_observers() # the display_name may have changed
        return cmd

    def notify_stack_observers(self):
        for func in self.stack_observers:
            func(self)

class Action:
    """An undoable, redoable action.

    Base class for all undo/redoable actions. Subclasses must implement the
    undo and redo methods. They should have a reference to the document in 
    self.doc.

    """
    automatic_undo = False
    display_name = _("Unknown Action")

    def __repr__(self):
        return "<%s>" % (self.display_name,)


    def redo(self):
        """Callback used to perform, or re-perform the Action.
        """
        raise NotImplementedError


    def undo(self):
        """Callback used to un-perform an already performed Action.
        """
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

class Stroke(Action):
    display_name = _("Painting")
    def __init__(self, doc, stroke, snapshot_before):
        """called only when the stroke was just completed and is now fully rendered"""
        self.doc = doc
        assert stroke.finished
        self.stroke = stroke # immutable; not used for drawing any more, just for inspection
        self.before = snapshot_before
        self.doc.layer.add_stroke(stroke, snapshot_before)
        # this snapshot will include the updated stroke list (modified by the line above)
        self.after = self.doc.layer.save_snapshot()
    def undo(self):
        self.doc.layer.load_snapshot(self.before)
    def redo(self):
        self.doc.layer.load_snapshot(self.after)


class FloodFill (Action):
    """Flood-fill on the current layer"""

    display_name = _("Flood Fill")

    def __init__(self, doc, x, y, color, bbox, tolerance,
                 sample_merged, make_new_layer):
        self.doc = doc
        self.x = x
        self.y = y
        self.color = color
        self.bbox = bbox
        self.tolerance = tolerance
        self.sample_merged = sample_merged
        self.make_new_layer = make_new_layer
        self.new_layer = None
        self.new_layer_idx = None
        self.snapshot = None

    def redo(self):
        # Pick a source
        if self.sample_merged:
            src_layer = layer.Layer()
            for l in self.doc.layers:
                l.merge_into(src_layer, strokemap=False)
        else:
            src_layer = self.doc.layer
        # Choose a target
        if self.make_new_layer:
            # Write to a new layer
            assert self.new_layer is None
            nl = layer.Layer()
            nl.content_observers.append(self.doc.layer_modified_cb)
            nl.set_symmetry_axis(self.doc.get_symmetry_axis())
            self.new_layer = nl
            self.new_layer_idx = self.doc.layer_idx + 1
            self.doc.layers.insert(self.new_layer_idx, nl)
            self.doc.layer_idx = self.new_layer_idx
            self._notify_document_observers()
            dst_layer = nl
        else:
            # Overwrite current, but snapshot 1st
            assert self.snapshot is None
            self.snapshot = self.doc.layer.save_snapshot()
            dst_layer = self.doc.layer
        # Fill connected areas of the source into the destination
        src_layer.flood_fill(self.x, self.y, self.color, self.bbox,
                             self.tolerance, dst_layer=dst_layer)

    def undo(self):
        if self.make_new_layer:
            assert self.new_layer is not None
            self.doc.layer_idx = self.new_layer_idx - 1
            self.doc.layers.remove(self.new_layer)
            self._notify_canvas_observers([self.doc.layer])
            self._notify_document_observers()
            self.new_layer = None
            self.new_layer_idx = None
        else:
            assert self.snapshot is not None
            self.doc.layer.load_snapshot(self.snapshot)
            self.snapshot = None


class TrimLayer (Action):
    """Trim the current layer to the extent of the document frame"""

    display_name = _("Trim Layer")

    def __init__(self, doc):
        self.doc = doc
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
        self.doc = doc
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
        self.doc = doc
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
        self.doc = doc
        self.dst_layer = self.doc.layers[dst_idx]
        self.normalize_src = ConvertLayerToNormalMode(doc, doc.layer)
        self.normalize_dst = ConvertLayerToNormalMode(doc, self.dst_layer)
        self.remove_src = RemoveLayer(doc)
    def redo(self):
        self.normalize_src.redo()
        self.normalize_dst.redo()
        self.dst_before = self.dst_layer.save_snapshot()
        assert self.doc.layer is not self.dst_layer
        self.doc.layer.merge_into(self.dst_layer)
        self.remove_src.redo()
        self.select_dst = SelectLayer(self.doc, self.doc.layers.index(self.dst_layer))
        self.select_dst.redo()
        self._notify_document_observers()
    def undo(self):
        self.select_dst.undo()
        del self.select_dst
        self.remove_src.undo()
        self.dst_layer.load_snapshot(self.dst_before)
        del self.dst_before
        self.normalize_dst.undo()
        self.normalize_src.undo()
        self._notify_document_observers()

class ConvertLayerToNormalMode(Action):
    display_name = _("Convert Layer Mode")
    def __init__(self, doc, layer):
        self.doc = doc
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

class AddLayer(Action):
    display_name = _("Add Layer")
    def __init__(self, doc, insert_idx=None, after=None, name=''):
        self.doc = doc
        self.insert_idx = insert_idx
        if after:
            l_idx = self.doc.layers.index(after)
            self.insert_idx = l_idx + 1
        self.layer = layer.Layer(name)
        self.layer.content_observers.append(self.doc.layer_modified_cb)
        self.layer.set_symmetry_axis(self.doc.get_symmetry_axis())
    def redo(self):
        self.doc.layers.insert(self.insert_idx, self.layer)
        self.prev_idx = self.doc.layer_idx
        self.doc.layer_idx = self.insert_idx
        self._notify_document_observers()
    def undo(self):
        self.doc.layers.remove(self.layer)
        self.doc.layer_idx = self.prev_idx
        self._notify_document_observers()

class RemoveLayer(Action):
    """Removes a layer, replacing it with a new one if it was the last.
    """
    display_name = _("Remove Layer")
    def __init__(self, doc,layer=None):
        self.doc = doc
        self.layer = layer
        self.newlayer0 = None
    def redo(self):
        if self.layer:
            self.idx = self.doc.layers.index(self.layer)
            self.doc.layers.remove(self.layer)
        else:
            self.idx = self.doc.layer_idx
            self.layer = self.doc.layers.pop(self.doc.layer_idx)
        if len(self.doc.layers) == 0:
            if self.newlayer0 is None:
                ly = layer.Layer("")
                ly.content_observers.append(self.doc.layer_modified_cb)
                ly.set_symmetry_axis(self.doc.get_symmetry_axis())
                self.newlayer0 = ly
            self.doc.layers.append(self.newlayer0)
            self.doc.layer_idx = 0
            assert self.idx == 0
        else:
            if self.doc.layer_idx == len(self.doc.layers):
                self.doc.layer_idx -= 1
        self._notify_canvas_observers([self.layer])
        self._notify_document_observers()
    def undo(self):
        if self.newlayer0 is not None:
            self.doc.layers.remove(self.newlayer0)
        self.doc.layers.insert(self.idx, self.layer)
        self.doc.layer_idx = self.idx
        self._notify_canvas_observers([self.layer])
        self._notify_document_observers()

class SelectLayer(Action):
    display_name = _("Select Layer")
    automatic_undo = True
    def __init__(self, doc, idx):
        self.doc = doc
        self.idx = idx
    def redo(self):
        assert self.idx >= 0 and self.idx < len(self.doc.layers)
        self.prev_idx = self.doc.layer_idx
        self.doc.layer_idx = self.idx
        self._notify_document_observers()
    def undo(self):
        self.doc.layer_idx = self.prev_idx
        self._notify_document_observers()

class MoveLayer(Action):
    display_name = _("Move Layer on Canvas")
    # NOT "Move Layer" for now - old translatable string with different sense
    def __init__(self, doc, layer_idx, dx, dy, ignore_first_redo=True):
        self.doc = doc
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
        self.doc = doc
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

class DuplicateLayer(Action):
    display_name = _("Duplicate Layer")
    def __init__(self, doc, insert_idx=None, name=''):
        self.doc = doc
        self.insert_idx = insert_idx
        snapshot = self.doc.layers[self.insert_idx].save_snapshot()
        self.new_layer = layer.Layer(name)
        self.new_layer.load_snapshot(snapshot)
        self.new_layer.content_observers.append(self.doc.layer_modified_cb)
        self.new_layer.set_symmetry_axis(doc.get_symmetry_axis())
    def redo(self):
        self.doc.layers.insert(self.insert_idx+1, self.new_layer)
        self.duplicate_layer = self.doc.layers[self.insert_idx+1]
        self._notify_canvas_observers([self.duplicate_layer])
        self._notify_document_observers()
    def undo(self):
        self.doc.layers.remove(self.duplicate_layer)
        original_layer = self.doc.layers[self.insert_idx]
        self._notify_canvas_observers([original_layer])
        self._notify_document_observers()

class ReorderLayers(Action):
    display_name = _("Reorder Layer Stack")
    def __init__(self, doc, new_order):
        self.doc = doc
        self.old_order = doc.layers[:]
        self.selection = self.old_order[doc.layer_idx]
        self.new_order = new_order
        for layer in new_order:
            assert layer in self.old_order
        assert len(self.old_order) == len(new_order)
    def redo(self):
        self.doc.layers[:] = self.new_order
        self.doc.layer_idx = self.doc.layers.index(self.selection)
        self._notify_canvas_observers(self.doc.layers)
        self._notify_document_observers()
    def undo(self):
        self.doc.layers[:] = self.old_order
        self.doc.layer_idx = self.doc.layers.index(self.selection)
        self._notify_canvas_observers(self.doc.layers)
        self._notify_document_observers()

class RenameLayer(Action):
    display_name = _("Rename Layer")
    def __init__(self, doc, name, layer):
        self.doc = doc
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
        self.doc = doc
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
        self.doc = doc
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
        self.doc = doc
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
        self.doc = doc
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
        self.doc = doc
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
        self.doc = doc
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

