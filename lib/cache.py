# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


from collections import OrderedDict


class LRUCache (object):
    """Least-recently-used cache with dict-like usage"""
    # The idea for using an OrderedDict comes from Kun Xi:
    # http://www.kunxi.org/blog/2014/05/lru-cache-in-python/

    _SENTINEL = object()

    def __init__(self, capacity=2048):
        self._capacity = capacity
        self._cache = OrderedDict()
        self._hits = 0
        self._misses = 0

    def __repr__(self):
        hitrate = 1.0
        missrate = 0.0
        accesses = float(self._hits + self._misses)
        if accesses > 0:
            hitrate = self._hits / accesses
            missrate = self._misses / accesses
        return "<LRUCache c: %d/%d h: %.0f%% m: %.0f%%>" % (
                len(self._cache),
                self._capacity,
                hitrate * 100,
                missrate * 100,
            )

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def __len__(self):
        return len(self._cache)

    def __contains__(self, key):
        return key in self._cache

    def __getitem__(self, key):
        item = self.get(item, self._SENTINEL)
        if item is self._SENTINEL:
            raise KeyError
        return item

    def get(self, key, default=None):
        try:
            item = self._cache.pop(key)
            self._cache[key] = item
            self._hits += 1
            return item
        except KeyError:
            self._misses += 1
            return default

    def __setitem__(self, key, item):
        try:
            self._cache.pop(key)
        except KeyError:
            while len(self._cache) >= self._capacity:
                self._cache.popitem(last=False)
        self._cache[key] = item

