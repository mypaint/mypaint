# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

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
    serial_number = 0
    def __init__(self):
        self.finished = False
        self.rendered = False # only used for assertions
        Stroke.serial_number += 1
        self.serial_number = Stroke.serial_number

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
        #if not self.empty:
        #    print 'Recorded', len(self.stroke_data), 'bytes. (painting time: %.2fs)' % self.total_painting_time
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

class Struct:
    pass

def strokes_from_to(a, b):
    if a.background != b.background:
        return None
    n = len(a.strokes)
    if a.strokes == b.strokes[:n]:
        new_strokes = b.strokes[n:]
        return new_strokes
    return None


class Layer:
    def __init__(self, mdw):
        self.mdw = mdw # MyDrawWidget used as "surface" until real layers/surfaces are implemented

        self.strokes = [] # gets manipulated directly from outside
        self.background = None
        self.rendered = Struct()
        self.rendered.strokes = []
        self.rendered.background = None

        self.caches = []
        self.strokes_to_cache = 6

    def populate_cache(self):
        # too few strokes to be worth caching?
        if len(self.rendered.strokes) < self.strokes_to_cache:
            return
        # got a close-enough cache already?
        for cache in self.caches:
            new_strokes = strokes_from_to(cache, self.rendered)
            if new_strokes is None: continue
            if len(new_strokes) < self.strokes_to_cache:
                return

        #print 'adding cache (%d strokes)' % len(self.rendered.strokes)

        t = time()
        # the last one is the most recently used one
        max_caches = 3
        while len(self.caches) > max_caches-1:
            cache = self.caches.pop(0)
            #print 'dropping a cache with', len(cache.strokes), 'strokes'
            del cache
        gc.collect()

        cache = Struct()
        cache.strokes = self.rendered.strokes[:]
        cache.background = self.rendered.background
        cache.snapshot = self.mdw.save_snapshot()
        self.caches.append(cache)
        #print 'caching the layer bitmap took %.3f seconds' % (time() - t)

    def rerender(self, only_estimate_cost=False):
        #print 'rerender'
        t1 = time()
        mdw = self.mdw

        def count_strokes_from(rendered):
            strokes = strokes_from_to(rendered, self)
            if strokes is None:
                return infinity
            return len(strokes)

        def render_new_strokes():
            new_strokes = strokes_from_to(self.rendered, self)
            warning = len(new_strokes) > 20
            if warning:
                print 'rendering', len(new_strokes), 'strokes...'

            caching = True
            # when replaying a huge amount of strokes, only populate the cache towards the end
            if len(new_strokes) > 2*self.strokes_to_cache:
                caching = new_strokes[-2*self.strokes_to_cache]

            for new_stroke in new_strokes:
                new_stroke.render(mdw)
                self.rendered.strokes.append(new_stroke)
                if caching is new_stroke:
                    caching = True
                if caching is True:
                    self.populate_cache()

            assert self.rendered.strokes == self.strokes

            if warning:
                print 'done rendering.'

        # will contain (cost, function) pairs of all possible actions
        options = []

        cost = count_strokes_from(self.rendered)
        options.append((cost, render_new_strokes))

        if cost <= 1:
            # no need to evaluate other options
            if cost > 0 and not only_estimate_cost:
                render_new_strokes()
            return cost

        for cache in self.caches:
            #print 'evaluating a cache containing %d strokes' % len(cache.strokes)
            cost = count_strokes_from(cache)
            cost += 3 # penalty for loading a pixbuf

            def render_cached(cache=cache):
                #print 'using a cache containing %d strokes' % len(cache.strokes)
                # least recently used caching strategy
                self.caches.remove(cache)
                self.caches.append(cache)
                mdw.load_snapshot(cache.snapshot)
                self.rendered.strokes = cache.strokes[:]
                self.rendered.background = cache.background
                render_new_strokes()

            options.append((cost, render_cached))

        def render_from_empty():
            #print 'full rerender'
            old_viewport_orig = mdw.get_viewport_orig() # mdw.clear() will reset viewport
            if self.background:
                mdw.load(self.background)
            else:
                mdw.clear()
            self.rendered.strokes = []
            self.rendered.background = self.background
            render_new_strokes()
            mdw.set_viewport_orig(*old_viewport_orig)
            mdw.resize_if_needed()

        cost = len(self.strokes)
        if self.background:
            cost += 3 # penalty for loading a pixbuf
        options.append((cost, render_from_empty))

        cost, render = min(options)
        del options # garbage collector might be called by render(), allow to free cache items

        if only_estimate_cost:
            return cost

        t2 = time()
        render()
        t3 = time()
        #print 'rerender took %.3f seconds, wasted %.3f seconds for cost evaluation' % (t3-t1, t2-t1)
        return cost
