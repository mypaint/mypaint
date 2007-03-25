"""
Design thoughts:
A layer:
- is usually a container of several strokes (strokes can be removed)
- can be rendered onto a bitmap
- can contain cache bitmaps, so it doesn't have to rerender all strokes all the time

A stroke:
- is a list of motion events
- knows everything needed to draw itself (brush settings / initial brush state)
- has fixed brush settings (only brush states can change during a stroke)
"""

import brush, helpers, random, gc
from time import time
infinity = 99999999 

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
        self.bbox = helpers.Rect(x-self.mdw.original_canvas_x0, y-self.mdw.original_canvas_y0, w, h)
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

        x, y, w, h = self.bbox.tuple()
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

    def populate_cache(self):
        # decide whether the currently rendered pixbuf should be cached

        rendered = self.rendered_strokes
        if len(rendered) < self.strokes_to_cache:
            # no need to cache that
            return
        for cached, snapshot in self.caches:
            # does this cache contain the beginning of the currently rendered strokes?
            if cached == rendered[:len(cached)]:
                # would it be acceptable to go from this cache into the current state?
                if len(rendered) - len(cached) < self.strokes_to_cache:
                    # no need to add a new cache
                    return

        print 'adding cache (%d strokes)' % len(rendered)

        t = time()
        # the last one is the most recently used one
        max_caches = 2
        while len(self.caches) > max_caches-1:
            tmp = self.caches.pop(0)
            print 'dropping a cache with', len(tmp[0]), 'strokes'
            del tmp
        gc.collect()

        strokes = self.rendered_strokes[:]
        snapshot = self.mdw.save_snapshot()
        self.caches.append((strokes, snapshot))
        print 'caching the layer bitmap took %.3f seconds' % (time() - t)

    def rerender(self, only_estimate_cost=False):
        t1 = time()
        mdw = self.mdw

        def count_strokes_from(rendered):
            n = len(rendered)
            if rendered == self.strokes[:n]:
                new_strokes = self.strokes[n:]
                return len(new_strokes)
            return infinity

        def render_new_strokes():
            rendered = self.rendered_strokes
            n = len(rendered)
            assert rendered == self.strokes[:n]
            new_strokes = self.strokes[n:]

            print 'rendering', len(new_strokes), 'strokes'

            caching = True
            # when replaying a huge amount of strokes, only populate the cache towards the end
            if len(new_strokes) > 2*self.strokes_to_cache:
                caching = new_strokes[-2*self.strokes_to_cache]

            for new_stroke in new_strokes:
                new_stroke.render(mdw)
                rendered.append(new_stroke)
                if caching is new_stroke:
                    caching = True
                if caching is True:
                    self.populate_cache()

            assert rendered == self.strokes

        # will contain (cost, function) pairs of all possible actions
        options = []

        def render_from_current():
            print 'only add'
            render_new_strokes()

        cost = count_strokes_from(self.rendered_strokes)
        options.append((cost, render_from_current))

        if cost <= 1:
            # no need to evaluate more options
            if not only_estimate_cost:
                render_from_current()
            return cost

        for cache in self.caches:
            rendered, snapshot = cache
            print 'evaluating a cache containing %d strokes' % len(rendered)
            cost = count_strokes_from(rendered)
            cost += 3 # penalty for loading a pixbuf

            def render_cached(rendered=rendered, snapshot=snapshot, cache=cache):
                print 'using a cache containing %d strokes' % len(rendered)
                # least recently used caching strategy
                self.caches.remove(cache)
                self.caches.append(cache)
                mdw.load_snapshot(snapshot)
                self.rendered_strokes = rendered[:]
                render_new_strokes()

            options.append((cost, render_cached))

        def render_from_empty():
            print 'full rerender'
            old_viewport_orig = mdw.get_viewport_orig() # mdw.clear() will reset viewport
            mdw.clear()
            # TODO: load image, if any (maybe like a stroke?)
            self.rendered_strokes = []
            render_new_strokes()
            mdw.set_viewport_orig(*old_viewport_orig)
            mdw.resize_if_needed()

        cost = len(self.strokes)
        options.append((cost, render_from_empty))

        cost, render = min(options)
        del options # garbage collector might be called by render(), allow to free cache items

        t2 = time()
        if not only_estimate_cost:
            render()
        t3 = time()
        print 'rerender took', t3-t1, 'seconds, wasted', t2-t1, 'seconds for cost evaluation'
        return cost

    def clear(self):
        data = self.strokes[:] # copy
        self.strokes = []
        self.rerender()
        return data

    def unclear(self, data):
        self.strokes = data[:] # copy
        self.rerender()
