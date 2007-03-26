
# if it is allowed to delete a stroke (below other strokes)
# then Actions operate on the stroke stack,
# thus a separate stroke stack needs to be maintained (per layer?)


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
    def __init__(self, layer, stroke, z=-1):
        self.layer = layer
        self.stroke = stroke # immutable
        if z == -1:
            z = len(layer.strokes)
        self.z = z
    def execute(self):
        # this stroke has been rendered while recording
        self.layer.rendered.strokes.append(self.stroke)
        self.redo()
    def undo(self):
        self.layer.strokes.remove(self.stroke)
    def redo(self):
        assert self.z >= 0
        assert self.z <= len(self.layer.strokes)
        self.layer.strokes.insert(self.z, self.stroke)

class ClearLayer(Action):
    def __init__(self, layer):
        self.layer = layer
    def execute(self):
        self.old_strokes = self.layer.strokes[:] # copy
        self.old_background = self.layer.background
        self.layer.strokes = []
        self.layer.background = None
        mdw = self.layer.mdw
        self.viewport = mdw.get_viewport_orig()
        mdw.set_viewport_orig(0, 0)
    def undo(self):
        self.layer.strokes = self.old_strokes
        self.layer.background = self.old_background
        mdw = self.layer.mdw
        mdw.set_viewport_orig(*self.viewport)

        del self.old_strokes, self.old_background, self.viewport
    redo = execute

class LoadImage(ClearLayer):
    def __init__(self, layer, pixbuf):
        ClearLayer.__init__(self, layer)
        self.pixbuf = pixbuf
    def execute(self):
        ClearLayer.execute(self)
        self.layer.background = self.pixbuf
    redo = execute

class ModifyStrokes(Action):
    def __init__(self, layer, strokes, new_brush):
        self.layer = layer
        self.strokes = strokes
        self.old_strokes = None
        self.set_new_brush(new_brush)
    def set_new_brush(self, new_brush):
        assert not self.old_strokes
        self.new_brush_settings = new_brush.save_to_string()
    def execute(self):
        self.old_strokes = self.layer.strokes[:] # copy
        for s in self.strokes:
            i = self.layer.strokes.index(s)
            s = s.copy()
            s.change_brush_settings(self.new_brush_settings)
            self.layer.strokes[i] = s
    def undo(self):
        self.layer.strokes = self.old_strokes[:] # copy
        self.old_strokes = None
    redo = execute

