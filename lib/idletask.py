# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gi.repository import GObject


class Processor (object):
    """Queue of low priority tasks for background processing

    Queued tasks are automatically processed when gtk is idle, or on demand.

    """

    def __init__(self, priority=GObject.PRIORITY_LOW):
        """Initialize, specifying a priority"""
        object.__init__(self)
        self._queue = []


    def add_work(self, func, *args, **kwargs):
        """Adds work

        :param func: a callable. The return value is ignored
        :param *args: passed to ``func``
        :param **kwargs: passed to ``func``
        """
        if not self._queue:
            GObject.idle_add(self._idle_cb, priority=GObject.PRIORITY_LOW)
        self._queue.append((func, args, kwargs))


    def _finish_one(self):
        func, args, kwargs = self._queue.pop(0)
        func(*args, **kwargs)


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

