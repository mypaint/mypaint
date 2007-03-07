
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
        

class Action:
    def __init__(self, doc):
        self.doc = doc
    def execute(self):
        assert False, 'abstract method'
    def unto(self):
        assert False, 'abstract method'
    def redo(self):
        assert False, 'abstract method'

class Stroke(Action):
    def __init__(self, layer, stroke):
        self.layer = layer
        self.stroke = stroke # stroke immutable, otherwise need to copy here
    def execute(self):
        self.layer.add_stroke(self.stroke, must_render=False)
    def redo(self):
        self.layer.add_stroke(self.stroke)
    def undo(self):
        self.layer.remove_stroke(self.stroke)

class ClearLayer(Action):
    def __init__(self, layer):
        self.layer = layer
    def execute(self):
        self.old_data = self.layer.clear()
    def undo(self):
        self.layer.unclear(self.old_data)
        del self.old_data
    redo = execute
