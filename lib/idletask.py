# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Non-threaded, prioritizable background processing."""


import collections

from gi.repository import GLib


class Processor (object):
    """Queue of low priority tasks for background processing

    Queued tasks are automatically processed in the main thread.
    They run when GTK is idle, or on demand.

    The default priority is much lower than gui event processing.

    """

    def __init__(self, priority=GLib.PRIORITY_LOW):
        """Initialize, specifying a priority"""
        object.__init__(self)
        self._queue = collections.deque()
        self._priority = priority
        self._idle_id = None

    def add_work(self, func, *args, **kwargs):
        """Adds work

        :param func: a task callable.
        :param *args: passed to func
        :param **kwargs: passed to func

        This starts the queue running if it isn't already.
        Each callable will be called with the given parameters
        until it returns false, at which point it's discarded.

        """
        if not self._idle_id:
            self._idle_id = GLib.idle_add(
                self._process,
                priority=self._priority,
            )
        self._queue.append((func, args, kwargs))

    def finish_all(self):
        """Complete processing: finishes all queued tasks."""
        while self._process():
            pass
        assert self._idle_id is None
        assert len(self._queue) == 0

    def stop(self):
        """Immediately stop processing and clear the queue."""
        if self._idle_id:
            GLib.source_remove(self._idle_id)
            self._idle_id = None
        self._queue.clear()
        assert self._idle_id is None
        assert len(self._queue) == 0

    def _process(self):
        if not self._idle_id:
            return False
        if not self._queue:
            self._idle_id = None
            return False
        func, args, kwargs = self._queue[0]
        run_again = bool(func(*args, **kwargs))
        if not run_again:
            self._queue.popleft()
        return True
