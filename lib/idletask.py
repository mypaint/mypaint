# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gi.repository import GObject


class Processor:
    """Queue of low priority tasks for background processing

    Queued tasks are automatically processed when gtk is idle, or on demand.

    """

    def __init__(self, max_pending):
        """Initialize, specifying maximum overhead.

        :param max_pending: maximum queue weight before `add_work()` starts
            doing immediate work.
        """
        self._queue = []
        self.max_pending = float(max_pending)


    def add_work(self, func, weight=1.0):
        """Adds work, possibly doing some of it if there's too much overhead.

        :param func: a callable of no arguments; return is ignored.
        :param weight: weight estimate for `func`.

        The queue will be processed until the sum of its functions' weight
        estimates is smaller than the processor's ``max_pending`` setting.
        Further processing happens automatically in the background.

        """
        if not self._queue:
            GObject.idle_add(self._idle_cb)
        func.__weight = float(weight)
        self._queue.append(func)
        self._finish_downto(self.max_pending)


    def _finish_one(self):
        func = self._queue.pop(0)
        func()
        return func.__weight


    def _finish_downto(self, max_pending):
        pending = sum([func.__weight for func in self._queue])
        while self._queue and pending > float(max_pending):
            pending -= self._finish_one()


    def finish_all(self):
        """Finishes all queued tasks."""
        while self._queue:
            self._finish_one()
        assert len(self._queue) == 0


    def _idle_cb(self):
        if not self._queue:
            return False
        self._finish_one()
        return True

