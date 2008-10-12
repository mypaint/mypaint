# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

class CommandStack:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []
        self.call_before_action = []
    
    def add(self, command):
        for f in self.call_before_action: f()
        self.redo_stack = [] # discard
        command.execute()
        self.undo_stack.append(command)
    
    def undo(self):
        if not self.undo_stack: return
        for f in self.call_before_action: f()
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        
    def redo(self):
        if not self.redo_stack: return
        for f in self.call_before_action: f()
        command = self.redo_stack.pop()
        command.redo()
        self.undo_stack.append(command)

    def get_last_command(self):
        if not self.undo_stack: return None
        return self.undo_stack[-1]
        

class Action:
    # children must support:
    # - execute
    # - redo
    # - undo
    pass

class Stroke(Action):
    def __init__(self, layer, stroke):
        self.layer = layer
        assert stroke.finished
        self.stroke = stroke # immutable
    def execute(self):
        # this stroke has been rendered while recording
        self.layer.rendered.strokes.append(self.stroke)
        self.redo()
        self.layer.populate_cache()
    def undo(self):
        self.layer.strokes.remove(self.stroke)
        self.layer.rerender()
    def redo(self):
        self.layer.strokes.append(self.stroke)
        self.layer.rerender()

class ClearLayer(Action):
    def __init__(self, layer):
        self.layer = layer
    def execute(self):
        self.old_strokes = self.layer.strokes[:] # copy
        self.old_background = self.layer.background
        self.layer.strokes = []
        self.layer.background = None
        self.layer.rerender()
    def undo(self):
        self.layer.strokes = self.old_strokes
        self.layer.background = self.old_background
        self.layer.rerender()

        del self.old_strokes, self.old_background
    redo = execute

class LoadImage(ClearLayer):
    def __init__(self, layer, pixbuf):
        ClearLayer.__init__(self, layer)
        self.pixbuf = pixbuf
    def execute(self):
        ClearLayer.execute(self)
        self.layer.background = self.pixbuf
        self.layer.rerender()
    redo = execute

# class ModifyStrokes(Action):
#     def __init__(self, layer, strokes, new_brush):
#         self.layer = layer
#         for s in strokes:
#             assert s in layer.strokes
#         self.strokemap = [[s, None] for s in strokes]
#         self.set_new_brush(new_brush)
#     def set_new_brush(self, new_brush):
#         # only called when the action is not (yet/anymore) on the undo stack
#         for pair in self.strokemap:
#             pair[1] = pair[0].copy_using_different_brush(new_brush)
#     def execute(self, undo=False):
#         for old, new in self.strokemap:
#             if undo: old, new = new, old
#             i = self.layer.strokes.index(old)
#             self.layer.strokes[i] = new
#         self.layer.rerender()
#     def undo(self):
#         self.execute(undo=True)
#     redo = execute

