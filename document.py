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

import brush, random, gc

class Stroke:
    # A stroke is immutable, except a freshly created/copied one.
    def __init__(self):
        self.finished = False
        self.rendered = False # only used for assertions

    def start_recording(self, mdw, brush):
        assert not self.finished
        self.mdw = mdw

        self.viewport_orig_x = mdw.viewport_x - mdw.original_canvas_x0
        self.viewport_orig_y = mdw.viewport_y - mdw.original_canvas_y0
        self.viewport_zoom = mdw.get_zoom()
        self.brush_settings = brush.save_to_string() # fast (brush caches this string)

        brush.translate_state(-mdw.original_canvas_x0, -mdw.original_canvas_y0)
        self.brush_state = brush.get_state()
        brush.translate_state(mdw.original_canvas_x0, mdw.original_canvas_y0)

        self.seed = random.randrange(0x10000)
        self.brush = brush
        brush.srandom(self.seed)
        # assumptions: (no tragic consequences when violated, but...)
        # - brush.split_stroke() has just been called, i.e.
        #   - stroke bbox is empty
        #   - stroke idle and painting times are empty

        self.mdw.start_recording()
        self.rendered = True # being rendered while recording

    def stop_recording(self):
        assert not self.finished
        self.stroke_data = self.mdw.stop_recording()
        x, y, w, h = self.brush.get_stroke_bbox()
        self.bbox = (x-self.mdw.original_canvas_x0, y-self.mdw.original_canvas_y0, w, h)
        self.total_painting_time = self.brush.get_stroke_total_painting_time()
        self.empty = w <= 0 and h <= 0
        if not self.empty:
            print 'Recorded', len(self.stroke_data), 'bytes. (painting time: %.2fs)' % self.total_painting_time
        #print 'Compressed size:', len(zlib.compress(self.stroke_data)), 'bytes.'
        del self.mdw, self.brush
        self.finished = True
        
    def render(self, surface):
        assert self.finished
        mdw = surface # Currently the surface can only be a MyDrawWidget.

        old_viewport_zoom = mdw.get_zoom()
        old_viewport_orig = mdw.get_viewport_orig()
        mdw.set_zoom(self.viewport_zoom)
        mdw.set_viewport_orig(self.viewport_orig_x, self.viewport_orig_y)

        x, y, w, h = self.bbox
        mdw.resize_if_needed(also_include_rect=(x+mdw.original_canvas_x0, y+mdw.original_canvas_y0, w, h))

        b = brush.Brush_Lowlevel() # temporary brush
        b.load_from_string(self.brush_settings)
        b.set_state(self.brush_state)
        b.translate_state(mdw.original_canvas_x0, mdw.original_canvas_y0)
        b.srandom(self.seed)
        #b.set_print_inputs(1)
        original_brush = mdw.set_brush(b)
        #print 'replaying', len(self.stroke_data), 'bytes'
        mdw.replay(self.stroke_data, 1)
        mdw.set_brush(original_brush)

        
        mdw.set_zoom(old_viewport_zoom)
        mdw.set_viewport_orig(*old_viewport_orig)

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
        self.caches = []
        self.strokes_to_cache = 4

    def cache_state(self):
        n = 2
        while len(self.caches) >= n:
            self.caches.pop(0)
        gc.collect()
        strokes = self.rendered_strokes[:]
        snapshot = self.mdw.save_snapshot()
        self.caches.append((strokes, snapshot))

    def rerender(self):
        print 'rerender:'
        if self.rendered_strokes == self.strokes:
            print 'nothing changed'
            return

        mdw = self.mdw

        def only_add(restore=None):
            # Only added some new strokes?
            n = len(self.rendered_strokes)
            if n <= len(self.strokes):
                if self.rendered_strokes == self.strokes[:n]:
                    if restore:
                        mdw.load_snapshot(restore)
                    new_strokes = self.strokes[n:]
                    print 'rendering', len(new_strokes), 'strokes'
                    for new_stroke in new_strokes:
                        new_stroke.render(mdw)
                    self.rendered_strokes = self.strokes[:] # copy
                    return True
            return False

        if only_add():
            print 'only add'
            return

        if self.caches:
            print 'there are', len(self.caches), 'caches'
        # Start from cached state?
        while self.caches:
            self.rendered_strokes, snapshot = self.caches[-1]
            if only_add(snapshot):
                print 'used cache'
                return
            self.caches.pop()
            gc.collect()
            print 'discarded cache'

        print 'full rerender'

        # TODO: check caches here

        old_viewport_orig = mdw.get_viewport_orig() # mdw.clear() will reset viewport

        mdw.clear()
        # TODO: load image, if any
        # TODO: call cache_state() at two good moments
        self.rendered_strokes = []

        cache_at = []
        n = self.strokes_to_cache
        if len(self.strokes) > n:
            cache_at.append(self.strokes[-n])
        if len(self.strokes) > 2*n:
            cache_at.append(self.strokes[-2*n])

        for stroke in self.strokes:
            stroke.render(mdw)
            self.rendered_strokes.append(stroke)
            if stroke in cache_at:
                self.cache_state()
        self.rendered_strokes = self.strokes[:] # copy

        mdw.set_viewport_orig(*old_viewport_orig)

    def clear(self):
        data = self.strokes
        self.strokes = []
        self.rerender()
        return data

    def unclear(self, data):
        self.strokes = data
        self.rerender()
