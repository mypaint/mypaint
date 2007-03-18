
# if it is allowed to delete a stroke (below other strokes)
# then Actions operate on the stroke stack,
# thus a separate stroke stack needs to be maintained (per layer?)


class CommandStack:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []
    
    def add(self, command):
        self.redo_stack = [] # discard
        command.execute()
        self.undo_stack.append(command)
    
    def undo(self):
        if not self.undo_stack: return
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        
    def redo(self):
        if not self.redo_stack: return
        command = self.redo_stack.pop()
        command.redo()
        self.undo_stack.append(command)

    def get_last_command(self):
        if not self.undo_stack: return None
        return self.undo_stack[-1]
        

class Action:
    def __init__(self, doc):
        self.doc = doc
    def execute(self):
        assert False, 'abstract method'
    def undo(self):
        assert False, 'abstract method'
    def redo(self):
        assert False, 'abstract method'

class Stroke(Action):
    def __init__(self, layer, stroke):
        self.layer = layer
        self.stroke = stroke # immutable
    def execute(self):
        # already rendered
        self.layer.strokes.append(self.stroke)
        self.layer.rendered_strokes.append(self.stroke)
    def undo(self):
        self.layer.strokes.remove(self.stroke)
    def redo(self):
        self.layer.strokes.append(self.stroke)

class ClearLayer(Action):
    def __init__(self, layer):
        self.layer = layer
    def execute(self):
        self.old_data = self.layer.clear()
    def undo(self):
        self.layer.unclear(self.old_data)
        del self.old_data
    redo = execute

class ModifyStrokes(Action):
    def __init__(self, layer, count, new_brush):
        self.layer = layer
        self.count = count
        self.old_strokes = None
        self.set_new_brush(new_brush)
    def set_new_brush(self, new_brush):
        assert not self.old_strokes
        self.new_brush_settings = new_brush.save_to_string()
    def execute(self):
        assert self.count > 0 
        assert self.count <= len(self.layer.strokes)
        self.old_strokes = self.layer.strokes[-self.count:]
        new_strokes = [s.copy() for s in self.old_strokes]
        for s in new_strokes:
            s.change_brush_settings(self.new_brush_settings)
        self.layer.strokes[-self.count:] = new_strokes
    def undo(self):
        self.layer.strokes[-self.count:] = self.old_strokes
        self.old_strokes = None
    redo = execute

