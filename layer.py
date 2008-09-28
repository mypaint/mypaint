# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

import tilelib
from time import time
import gc

infinity = 99999999

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
    # A layer contains a list of strokes, and possibly a background pixmap.
    # There is also a surface with those strokes rendered.
    #
    # The stroke list can be manipulated, but for decent performance
    # only the end of the list should be altered. There is a cache
    # system to speed this up (for undo/redo).

    def __init__(self):
        # The following three members get manipulated from outside;
        # the code manipulating them is responsible to keep the
        # strokes in sync with the surface, eg. by calling rerender()
        # FIXME: find a better design? also, rerender can take too long
        self.surface = tilelib.TiledSurface()
        self.strokes = []
        self.background = None

        self.rendered = Struct()
        self.rendered.strokes = [] # note that stroke objects are immutable
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
        cache.snapshot = self.surface.save_snapshot()
        self.caches.append(cache)
        #print 'caching the layer bitmap took %.3f seconds' % (time() - t)

    def rerender(self, only_estimate_cost=False):
        print 'rerender'
        t1 = time()
        surface = self.surface

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

            caching = True # FIXME: shouldn't this be False?
            # when replaying a huge amount of strokes, only populate the cache towards the end
            if len(new_strokes) > 2*self.strokes_to_cache:
                caching = new_strokes[-2*self.strokes_to_cache]

            for new_stroke in new_strokes:
                new_stroke.render(surface)
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
                surface.load_snapshot(cache.snapshot)
                self.rendered.strokes = cache.strokes[:]
                self.rendered.background = cache.background
                render_new_strokes()

            options.append((cost, render_cached))

        def render_from_empty():
            #print 'full rerender'
            if self.background:
                surface.load(self.background)
            else:
                surface.clear()
            self.rendered.strokes = []
            self.rendered.background = self.background
            render_new_strokes()

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

