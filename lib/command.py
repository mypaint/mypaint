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
    # children must support:
    # - redo
    # - undo
    automatic_undo = False

# FIXME: the code below looks horrible, there must be a less redundant way to implement this.
# - eg command_* special members on the document class?
# - and then auto-build wrapper methods?

class Stroke(Action):
    def __init__(self, doc, stroke, snapshot_before, snapshot_after):
        self.doc = doc
        assert stroke.finished
        self.stroke = stroke # immutable; not used for drawing any more, just for inspection
        self.before = snapshot_before
        self.after = snapshot_after
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
    def undo(self):
        self.doc.layer.load_snapshot(self.before)
        del self.before

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

class AddLayer(Action):
    def __init__(self, doc, insert_idx):
        self.doc = doc
        self.insert_idx = insert_idx
    def redo(self):
        l = layer.Layer()
        l.surface.observers.append(self.doc.layer_modified_cb)
        self.doc.layers.insert(self.insert_idx, l)
        self.prev_idx = self.doc.layer_idx
        self.doc.layer_idx = self.insert_idx
        for f in self.doc.layer_observers:
            f()
    def undo(self):
        self.doc.layers.pop(self.insert_idx)
        self.doc.layer_idx = self.prev_idx
        for f in self.doc.layer_observers:
            f()

class RemoveLayer(Action):
    def __init__(self, doc):
        self.doc = doc
    def redo(self):
        self.idx = self.doc.layer_idx
        self.layer = self.doc.layers.pop(self.doc.layer_idx)
        if self.doc.layer_idx == len(self.doc.layers):
            self.doc.layer_idx -= 1
        for f in self.doc.layer_observers:
            f()
    def undo(self):
        self.doc.layers.insert(self.idx, self.layer)
        self.doc.layer_idx = self.idx
        for f in self.doc.layer_observers:
            f()

class SelectLayer(Action):
    automatic_undo = True
    def __init__(self, doc, idx):
        self.doc = doc
        self.idx = idx
    def redo(self):
        assert self.idx >= 0 and self.idx < len(self.doc.layers)
        self.prev_idx = self.doc.layer_idx
        self.doc.layer_idx = self.idx
        for f in self.doc.layer_observers:
            f()
    def undo(self):
        self.doc.layer_idx = self.prev_idx
        for f in self.doc.layer_observers:
            f()

