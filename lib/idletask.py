# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gobject

class Processor:
    """
    A queue of low priority tasks that are automatically processed
    when gtk is idle, or on demand.
    """
    def __init__(self, max_pending):
        self._queue = []
        self.max_pending = max_pending

    def add_work(self, func, weight=1.0):
        if not self._queue:
            gobject.idle_add(self._idle_cb)
        func.__weight = weight
        self._queue.append(func)
        self.finish_downto(self.max_pending)

    def finish_one(self):
        print '.'
        func = self._queue.pop(0)
        func()

    def finish_downto(self, max_pending):
        while 1:
            pending_weight = sum([func.__weight for func in self._queue])
            if pending_weight <= max_pending:
                return
            self.finish_one()

    def finish_all(self):
        self.finish_downto(0)

    def _idle_cb(self):
        if not self._queue:
            return False
        self.finish_one()
        return True

