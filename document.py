"""
Design thoughts:
A layer:
- is usually a container of several strokes (strokes can be removed)
- can be rendered onto a bitmap
- can contain cache bitmaps, so it doesn't have to rerender all the time

A stroke:
- is a list of motion events
- knows everything needed to draw itself (brush settings / initial brush state)
- has fixed brush settings (only brush states can change during a stroke)
"""

import brush, random

class Stroke:
    def __init__(self):
        self.finished = False

    def start_recording(self, mdw, brush):
        # FIXME: must store current zoom
        assert not self.finished
        self.mdw = mdw

        self.brush_settings = brush.save_to_string() # OPTIMIZE
        self.brush_state = brush.get_state()
        self.seed = random.randrange(0x10000)
        self.brush = brush
        brush.srandom(self.seed)
        brush.reset_stroke_bbox()

        self.mdw.start_recording()

    def stop_recording(self):
        assert not self.finished
        self.stroke_data = self.mdw.stop_recording()
        self.bbox = self.brush.get_stroke_bbox()
        self.total_painting_time = self.brush.get_stroke_total_painting_time()
        x, y, w, h = self.bbox
        self.empty = w <= 0 and h <= 0
        if not self.empty:
            print 'Recorded', len(self.stroke_data), 'bytes. (painting time: %.2fs)' % self.total_painting_time
            print self.bbox
        #print 'Compressed size:', len(zlib.compress(self.stroke_data)), 'bytes.'
        del self.mdw, self.brush
        self.finished = True
        
    def render(self, surface):
        assert self.finished
        mdw = surface # Currently the surface can only be a MyDrawWidget.

        b = brush.Brush_Lowlevel() # temporary brush
        b.load_from_string(self.brush_settings)
        b.set_state(self.brush_state)
        b.srandom(self.seed)
        #b.set_print_inputs(1)
        original_brush = mdw.set_brush(b)
        print 'replaying', len(self.stroke_data), 'bytes'
        mdw.replay(self.stroke_data, 1)
        mdw.set_brush(original_brush)

class Layer:
    def __init__(self, mdw):
        self.mdw = mdw # MyDrawWidget used as "surface" until real layers/surfaces are implemented
        self.strokes = []

    def add_stroke(self, stroke, must_render=True):
        self.strokes.append(stroke)
        if must_render:
            stroke.render(self.mdw)

    def remove_stroke(self, stroke):
        self.strokes.remove(stroke)
        self.rerender()

    def rerender(self):
        self.mdw.clear() # FIXME resizes the mdw, too small
        for stroke in self.strokes:
            stroke.render(self.mdw)

    def clear(self):
        data = self.strokes
        self.strokes = []
        self.rerender()
        return data

    def unclear(self, data):
        self.strokes = data
        self.rerender()
