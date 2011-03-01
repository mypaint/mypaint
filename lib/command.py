# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import layer

class CommandStack:
    def __init__(self):
        self.call_before_action = []
        self.clear()
    
    def clear(self):
        self.undo_stack = []
        self.redo_stack = []

    def do(self, command):
        for f in self.call_before_action: f()
        self.redo_stack = [] # discard
        command.redo()
        self.undo_stack.append(command)
        self.reduce_undo_history()
    
    def undo(self):
        if not self.undo_stack: return
        for f in self.call_before_action: f()
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        return command
        
    def redo(self):
        if not self.redo_stack: return
        for f in self.call_before_action: f()
        command = self.redo_stack.pop()
        command.redo()
        self.undo_stack.append(command)
        return command

    def reduce_undo_history(self):
        stack = self.undo_stack
        self.undo_stack = []
        steps = 0
        for item in reversed(stack):
            self.undo_stack.insert(0, item)
            if not item.automatic_undo:
                steps += 1
            if steps == 20: # and memory > ...
                break

    def get_last_command(self):
        if not self.undo_stack: return None
        return self.undo_stack[-1]
        

class Action:
    '''Base class for all undo/redoable actions. Subclasses must implement the
    undo and redo methods. They should have a reference to the document in 
    self.doc'''
    automatic_undo = False

    def redo(self):
        raise NotImplementedError
    def undo(self):
        raise NotImplementedError

    # Utility functions
    def _notify_canvas_observers(self, affected_layer):
        bbox = affected_layer.surface.get_bbox()
        for f in self.doc.canvas_observers:
            f(*bbox)

    def _notify_document_observers(self):
        self.doc.call_doc_observers()

class Stroke(Action):
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

class ClearLayer(Action):
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
    def __init__(self, doc, data, x, y):
        self.doc = doc
        self.data = [x, y, data]
    def redo(self):
        layer = self.doc.layer
        self.before = layer.save_snapshot()
        layer.load_from_pixbuf(self.data)
    def undo(self):
        self.doc.layer.load_snapshot(self.before)
        del self.before

class MergeLayer(Action):
    """merge the current layer into dst"""
    def __init__(self, doc, dst_idx):
        self.doc = doc
        self.dst_layer = self.doc.layers[dst_idx]
        self.remove_src = RemoveLayer(doc)
    def redo(self):
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
        self._notify_document_observers()

class AddLayer(Action):
    def __init__(self, doc, insert_idx=None, after=None, name=''):
        self.doc = doc
        self.insert_idx = insert_idx
        if after:
            l_idx = self.doc.layers.index(after)
            self.insert_idx = l_idx + 1
        self.layer = layer.Layer(name)
        self.layer.surface.observers.append(self.doc.layer_modified_cb)
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
    def __init__(self, doc,layer=None):
        self.doc = doc
        self.layer = layer
    def redo(self):
        assert len(self.doc.layers) > 1
        if self.layer:
            self.idx = self.doc.layers.index(self.layer)
            self.doc.layers.remove(self.layer)
        else:
            self.idx = self.doc.layer_idx
            self.layer = self.doc.layers.pop(self.doc.layer_idx)
        if self.doc.layer_idx == len(self.doc.layers):
            self.doc.layer_idx -= 1
        self._notify_canvas_observers(self.layer)
        self._notify_document_observers()
    def undo(self):
        self.doc.layers.insert(self.idx, self.layer)
        self.doc.layer_idx = self.idx
        self._notify_canvas_observers(self.layer)
        self._notify_document_observers()

class SelectLayer(Action):
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
    def __init__(self, doc, was_idx, new_idx):
        self.doc = doc
        self.was_idx = was_idx
        self.new_idx = new_idx
    def redo(self):
        moved_layer = self.doc.layers[self.was_idx]
        self.doc.layers.remove(moved_layer)
        self.doc.layers.insert(self.new_idx, moved_layer)
        self._notify_canvas_observers(moved_layer)
        self._notify_document_observers()
    def undo(self):
        moved_layer = self.doc.layers[self.new_idx]
        self.doc.layers.remove(moved_layer)
        self.doc.layers.insert(self.was_idx, moved_layer)
        self._notify_canvas_observers(moved_layer)
        self._notify_document_observers()

class SetLayerVisibility(Action):
    def __init__(self, doc, visible, layer):
        self.doc = doc
        self.new_visibility = visible
        self.layer = layer
    def redo(self):
        self.old_visibility = self.layer.visible
        self.layer.visible = self.new_visibility
        self._notify_canvas_observers(self.layer)
        self._notify_document_observers()
    def undo(self):
        self.layer.visible = self.old_visibility
        self._notify_canvas_observers(self.layer)
        self._notify_document_observers()

class SetLayerLocked (Action):
    def __init__(self, doc, locked, layer):
        self.doc = doc
        self.new_locked = locked
        self.layer = layer
    def redo(self):
        self.old_locked = self.layer.locked
        self.layer.locked = self.new_locked
        self._notify_canvas_observers(self.layer)
        self._notify_document_observers()
    def undo(self):
        self.layer.locked = self.old_locked
        self._notify_canvas_observers(self.layer)
        self._notify_document_observers()

class SetLayerOpacity(Action):
    def __init__(self, doc, opacity, layer=None):
        self.doc = doc
        self.new_opacity = opacity
        self.layer = layer
    def redo(self):
        if self.layer:
            l = self.layer
        else:
            l = self.doc.layer
        self.old_opacity = l.opacity
        l.opacity = self.new_opacity
        self._notify_canvas_observers(l)
        self._notify_document_observers()
    def undo(self):
        if self.layer:
            l = self.layer
        else:
            l = self.doc.layer
        l.opacity = self.old_opacity
        self._notify_canvas_observers(l)
        self._notify_document_observers()

