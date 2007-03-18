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
    # A stroke is immutable, except a freshly created/copied one.
    def __init__(self):
        self.finished = False
        self.rendered = False # only used for assertions

    def start_recording(self, mdw, brush):
        # FIXME: must store current zoom
        assert not self.finished
        self.mdw = mdw

        self.brush_settings = brush.save_to_string() # fast (brush caches this string)
        self.brush_state = brush.get_state()
        self.seed = random.randrange(0x10000)
        self.brush = brush
        brush.srandom(self.seed)
        # assumptions: (no tragic consequences when violated, but...)
        # - brush.split_stroke() has just been called, i.e.
        #   - stroke bbox is empty
        #   - stroke idle and painting times are empty

        self.mdw.start_recording()
        self.rendered = True # being rendered right now, while recording

    def stop_recording(self):
        assert not self.finished
        self.stroke_data = self.mdw.stop_recording()
        self.bbox = self.brush.get_stroke_bbox()
        self.total_painting_time = self.brush.get_stroke_total_painting_time()
        x, y, w, h = self.bbox
        self.empty = w <= 0 and h <= 0
        if not self.empty:
            print 'Recorded', len(self.stroke_data), 'bytes. (painting time: %.2fs)' % self.total_painting_time
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

        self.rendered = True

    def copy(self):
        assert self.finished
        s = Stroke()
        s.__dict__.update(self.__dict__)
        s.rendered = False
        return s

    def change_brush_settings(self, brush_settings):
        assert self.finished 
        assert not self.rendered
        self.brush_settings = brush_settings
        # note: the new brush might have different meanings of the states
        # (another custom state, or speed inputs filtered differently)
        # too difficult to compensate this here, we just accept some glitches


class Layer:
    def __init__(self, mdw):
        self.mdw = mdw # MyDrawWidget used as "surface" until real layers/surfaces are implemented
        self.strokes = [] # gets manipulated directly from outside
        self.rendered_strokes = []

    def rerender(self):
        print 'rerender:',
        if self.rendered_strokes == self.strokes:
            print 'nothing changed'
            return

        # Only added some new strokes?
        if len(self.rendered_strokes) < len(self.strokes):
            n = len(self.rendered_strokes)
            if self.rendered_strokes == self.strokes[:n]:
                for new_stroke in self.strokes[n:]:
                    new_stroke.render(self.mdw)
                self.rendered_strokes = self.strokes[:] # copy
                print 'only add'
                return
                
        print 'full rerender'
        # TODO: check caches here
        self.mdw.clear() # FIXME resizes the mdw, too small
        for stroke in self.strokes:
            stroke.render(self.mdw)
        self.rendered_strokes = self.strokes[:] # copy

    def clear(self):
        data = self.strokes
        self.strokes = []
        self.rerender()
        return data

    def unclear(self, data):
        self.strokes = data
        self.rerender()
