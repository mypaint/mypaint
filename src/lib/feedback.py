# This file is part of MyPaint.
# Copyright (C) 2017 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Interfaces and objects for providing UI feedback."""


# Imports:

from __future__ import division, print_function

from lib.observable import event
import lib.helpers


# Class defs:

class Progress (object):
    """Itemized progress report with hierarchy.

    A top-level GUI process can create one of these objects, and connect
    to its .changed event to get real-time reports from calls it passes
    the Progress down to.

    >>> prog0 = Progress()
    >>> feedback = []
    >>> prog0.changed += lambda p: feedback.append(p.fraction())

    If the number of items is unset, the fraction is None.  Code that
    doesn't know how much work it has to do can call changed() to notify
    observers, however. This may cause a pulse in GUI progress bars, for
    example.

    >>> prog0.fraction() is None
    True
    >>> prog0.changed()
    >>> prog0.changed()
    >>> prog0.fraction() is None
    True
    >>> feedback
    [None, None]

    Progress report objects are only really useful if each stage
    declares how many items it expects to be filling in. You can't set
    the item count to less than its current value, and each change to ie
    emits the changed() event.

    >>> prog0.items = 10
    >>> prog0.fraction()
    0.0
    >>> feedback
    [None, None, 0.0]

    You can increment the progress report to mark items complete,
    or explicitly declare how many items are now complete.

    >>> prog0 += 1
    >>> prog0 += 2
    >>> feedback
    [None, None, 0.0, 0.1, 0.3]
    >>> prog0.completed(5)
    >>> feedback
    [None, None, 0.0, 0.1, 0.3, 0.5]

    If a stage is more fine-grained, you can open() a sub-progress
    report and pass that around or fill it in.

    >>> p1 = prog0.open()
    >>> p2 = prog0.open()
    >>> p1.items = 2
    >>> p2.items = 4
    >>> feedback
    [None, None, 0.0, 0.1, 0.3, 0.5, 0.5, 0.5]
    >>> prog0.fraction()
    0.5
    >>> p1 += 1
    >>> p2 += 1
    >>> prog0.fraction()
    0.575
    >>> feedback
    [None, None, 0.0, 0.1, 0.3, 0.5, 0.5, 0.5, 0.55, 0.575]

    """

    # Initialization and basic fields:

    def __init__(self):
        """Create a new, unsized Progress freedback object.

        It is reasonable for functions supporting this API to demand
        either unsized or sized Progress objects.

        """
        super(Progress, self).__init__()
        self._items = None
        self._completed = 0
        self._open = {}  # {Progress: weight_in_items}

    @property
    def items(self):
        """The number of open items (read, write once).

        An attempt to set the number of items after it has already been
        set results in an exception.

        >>> p = Progress()
        >>> p.items = 10
        >>> p.items = 15   # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: ...

        """
        return self._items

    @items.setter
    def items(self, n):
        if self._items is not None:
            raise ValueError("Progress.items has already been set.")
        n = max(0, int(n))
        if (self._items is None) or (n > self._items):
            self._items = n
            self.changed()

    # Observable events and monitoring:

    @event
    def changed(self):
        """Event: fires when "fraction" changes."""

    def _child_changed_cb(self, sprog):
        """Handles changes to an open child Progress."""
        if sprog not in self._open:
            return
        if self._items is None:
            self.changed()
            return
        if sprog.fraction() >= 1:
            sprog_weight = self._open.pop(sprog)
            self._completed += sprog_weight
            sprog.changed -= self._child_changed_cb
        self.changed()

    # Casts:

    def __bool__(self):
        """Progress objects always test true.

        >>> p = Progress()
        >>> bool(p)
        True

        """
        return True

    def __int__(self):
        """The integer value is the number of completed items.

        >>> p = Progress()
        >>> p.items = 10
        >>> p += 3
        >>> p1 = p.open()
        >>> p1.items = 97
        >>> p1.close()
        >>> int(p)
        4

        """
        return self._completed

    # Updating of open WIP objs:

    def __iadd__(self, n):
        """Mark this many items complete.

        Internally uses completed(), so it is subject to the same
        constraints.

        """
        n = max(0, int(n))
        c = self._completed + n
        self.completed(c)
        return self

    def completed(self, c):
        """Try to mark up to item c as completed.

        You cannot mark as complete more than ``items`` minus the
        combined weight of all the currently open child Progress
        objects.

        """
        if self._items is None:
            self.changed()
            return
        c = lib.helpers.clamp(c, 0, (self._items - self._open_items_weight()))
        if c > self._completed:
            self._completed = c
            self.changed()

    def close(self):
        """Mark this Progress as complete.

        If you set the size of a Progress object in a given function,
        and start marking its items as complete, it's normal to close()
        the Progress object later on.

        Or to put it another way, parent function calls male new,
        unsized Progress objects with open(), and pass those to
        functions they invoke. The child functions set sizes, declare
        that work is done in an incremental fashion, and finally call
        close() on what they're passed.

        """
        if self._items is not None:
            self._completed = self._items
        for sprog in self._open.keys():
            sprog.changed -= self._child_changed_cb
        self._open.clear()
        self.changed()

    def __repr__(self):
        """String representation of a Progress object.

        >>> p = Progress()
        >>> p
        <Progress 0.0/None>
        >>> p.items = 10
        >>> p += 1
        >>> p
        <Progress 1.0/10>

        """
        repr_str = "<Progress %0.1f/%r>" % (
            self._completed + self._open_items_completion(),
            self._items,
        )
        return repr_str

    def _open_items_weight(self):
        """Total weight of all open items (monitored child objs)."""
        total = 0
        for p, w in self._open.items():
            total += w
        return total

    def _open_items_completion(self):
        """Weighted completion sum over all open items."""
        total = 0.0
        for p, w in self._open.items():
            f = p.fraction()
            if f is None:
                f = 0.0
            total += f * float(w)
        return total

    def fraction(self):
        """Read-only completeness fraction."""
        if self._items is None:
            return None
        if self._items <= 0:
            return 1.0
        f1 = float(self._completed)
        f1 += self._open_items_completion()
        f1 /= float(self._items)
        return lib.helpers.clamp(f1, 0.0, 1.0)

    def open(self, weight=1):
        """Open a new monitored child Progress.

        :param int weight: Number of items represented by the child.
        :returns: A new, unsized Progress object.
        :rtype: Progress

        A child Progress object cannot represent more items than are
        currently available. If you try, the returned Progress object
        will work, but it won't be monitored.

        >>> p = Progress()
        >>> p.items = 10
        >>> p1 = p.open(4)
        >>> p1.items = 2
        >>> p1 += 1
        >>> p
        <Progress 2.0/10>
        >>> p1.close()
        >>> p
        <Progress 4.0/10>

        """
        sprog_weight = max(1, int(weight))
        sprog = Progress()
        if self._items is not None:
            c = self._completed + self._open_items_weight() + sprog_weight
            if c < self._items:
                sprog.changed += self._child_changed_cb
                self._open[sprog] = sprog_weight
        return sprog


def _test():
    """Run doctests"""
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _test()
