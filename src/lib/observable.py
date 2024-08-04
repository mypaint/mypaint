# This file is part of MyPaint.
# Copyright (C) 2013-2017 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Observable method calls and C#-like syntactic sugar for events."""

from __future__ import division, print_function

import weakref
import logging

logger = logging.getLogger(__name__)


class observable (object):  # noqa: N801
    """Decorator for methods which notify their observers after being called.

    To use, mark methods intended to be called on instances of a class with
    the ``@observable`` decorator:

    >>> class Tester (object):
    ...     @observable
    ...     def foo(self, a, b):
    ...         return a + b

    This allows it to be subscribed to with a "+= callable" syntax that's
    similar to the way C# does events.  In this implementation, you can hook up
    any Python callable which has the same signature as the observable method.
    Observer callables are passed a reference to the observed instance as their
    first positional argument, just like the observable method itself.

    >>> tester = Tester()
    >>> arr = []
    >>> tester.foo += lambda t, a, b: arr.extend([a, b])
    >>> tester.foo += lambda t, a, b: arr.extend([a+1, b+1])

    When the decorated method is called, the original definition of the method
    is invoked first, then each registered observer callback in turn. Observer
    callbacks are invoked before the observable function returns, and their
    return values are ignored.

    >>> tester.foo(41, 1)
    42
    >>> arr
    [41, 1, 42, 2]

    Only instance methods can be appended to. Trying to do this via the class
    results in an exception.

    >>> Tester.foo += lambda t, a: a+1
    Traceback (most recent call last):
    ...
    TypeError: unsupported operand type(s) for +=: 'observable' and 'function'

    Observable methods do not keep any references to the objects behind
    observers which happen to be bound methods, to avoid circular reference
    chains which could prevent garbage collection.  Instead, they store
    weakrefs to the bound methods' objects.  Any dead weakrefs are removed
    silently when the observed method is called.  Be cautious when writing
    something like:

    >>> class TesterObserver (object):
    ...     def obs(self, tester, a, b):
    ...         print ("Obsr: %r, %r" % (a, b))
    >>> tester = Tester()
    >>> tester.foo += TesterObserver().obs
    >>> tester.foo(2, 1)
    3

    Nothing is printed here because the ``TesterObserver`` instance had no
    permanent refs and was garbage collected before the call to ``foo()``.
    However, observable methods *do* retain strong references to simple
    functions like lambda expressions, so if you absolutely must observe with
    methods on purely throwaway objects, you can do

    >>> tester.foo += lambda t, a, b: TesterObserver().obs(t, a, b)
    >>> tester.foo(2, 1)
    Obsr: 2, 1
    3

    The more normal case involves observer objects which still have remaining
    strong references at the times their observed functions get called:

    >>> tester = Tester()
    >>> obsr = TesterObserver();
    >>> tester.foo += obsr.obs
    >>> tester.foo(6, 2)
    Obsr: 6, 2
    8

    You can remove bound methods and static functions using in-place
    notation too, and test for their presence using "in":

    >>> fn = lambda t, a: a+1
    >>> tester.foo += fn
    >>> fn in tester.foo
    True
    >>> obsr.obs in tester.foo
    True
    >>> tester.foo -= obsr.obs
    >>> tester.foo -= fn
    >>> fn in tester.foo
    False
    >>> obsr.obs in tester.foo
    False

    If you remove the last strong ref to such an observer, the observable
    method cleans up its internal weakref to it without any fuss the next time
    it's called:

    >>> tester.foo += obsr.obs
    >>> del obsr
    >>> tester.foo(6, 2)
    8

    The rationale for this is that observer objects are likely to have quite
    different lifetimes than the things they observe.  Sometimes the observer
    will have a longer lifetime, sometimes the observed object will live
    longer.  Avoiding a live reference hidden away in an observable method
    allows the observer to perish when it (or its owner) expects.
    Additionally, observers quite often construct and own the things they
    observe.  It's a natural style that's also a recipe for an accidental
    circular reference chain, so special-casing observables for bound methods
    makes sense.
    """

    def __init__(self, func):
        """Initialize as a descriptor supporting the decorator protocol"""
        super(observable, self).__init__()
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__

    def __get__(self, instance, owner):
        """Creation of the wrapper callable.

        The descriptor protocol is used for distinguishing between being
        accessed by class and being accessed by instance.  For the purposes of
        the decorator interface, we return a callable object, which is cached
        privately within `instance` so that the callable is associated
        permanently with the method.

        """
        # Return the decorator callable when accessed via the class: normal
        # descriptor protocol behaviour for instance things.
        if instance is None:
            return self
        # For second and subsequent calls, use a cache stored in the observable
        # object using this class's name mangling.
        try:
            wrappers_dict = instance.__wrappers
        except AttributeError:
            wrappers_dict = dict()
            instance.__wrappers = wrappers_dict
        wrapper = wrappers_dict.get(self.func)
        if wrapper is None:
            wrapper = _MethodWithObservers(instance, self.func)
            wrappers_dict[self.func] = wrapper
        elif wrapper.instance_weakref() is not instance:
            # Okay, change of identity. Happens with the standard copy().
            self._update_observers(instance)
            wrappers_dict = instance.__wrappers
            old_wrapper = wrapper
            wrapper = wrappers_dict.get(self.func)
            assert wrapper is not old_wrapper
            assert wrapper.instance_weakref() == instance
        assert callable(wrapper)
        return wrapper

    def __set__(self, obj, value):
        """Ignored (only defined to create a data descriptor)

        Without this, a shallow copy() of an object with this descriptor
        results in an entry in the class dict which shadows any non-data
        descriptor with the same name.
        """
        pass

    @classmethod
    def _update_observers(cls, instance):
        """Internal: required updates after observable instances are copied

        :param instance: The object to update

        This is called on the first access via a descriptor on the copy, and
        replaces the private __wrappers dict with one whose values refer back
        to the copy, not the original.

        Given an observer function that requires a particular identity for the
        object being observed,

          >>> class ListMunger (object):
          ...     @observable
          ...     def append_sum(self, items):
          ...         items.append(sum(items))
          >>> m1 = ListMunger()
          >>> class ListMungerExtras (object):
          ...     def bump_last(self, munger, items):
          ...         if munger is m1: items.append("invoked on m1")
          ...         else:            items.append("not invoked on m1")
          >>> mx = ListMungerExtras()
          >>> m1.append_sum += mx.bump_last
          >>> nums = [1, 1, 2]; m1.append_sum(nums); nums
          [1, 1, 2, 4, 'invoked on m1']

        this hack allows both deep and shallow copies to work as expected.

          >>> from copy import copy, deepcopy
          >>> m2 = deepcopy(m1)
          >>> m1.append_sum is m2.append_sum
          False
          >>> nums = [0, 1, 2]; m2.append_sum(nums); nums
          [0, 1, 2, 3, 'not invoked on m1']

          >>> m3 = copy(m1)
          >>> m1.append_sum is m3.append_sum
          False
          >>> nums = [3, 2, 1]; m3.append_sum(nums); nums
          [3, 2, 1, 6, 'not invoked on m1']

        """
        logger.debug("Updating wrappers for %r", instance)
        updated_wrappers = {}
        for func, old_wrapper in instance.__wrappers.items():
            new_wrapper = _MethodWithObservers(instance, func)
            new_wrapper.observers = old_wrapper.observers[:]
            updated_wrappers[func] = new_wrapper
        instance.__wrappers = updated_wrappers


class _MethodWithObservers (object):
    """Callable wrapper: calls the decorated method, then its observers

    This is what a __get__ on the observed object's @observable descriptor
    actually returns. Instances are stashed in the ``_<mangling>_wrappers``
    member of the observed object itself. Each `_MethodWithObservers` instance
    is callable, and when called invokes all the registered observers in
    turn.
    """

    def __init__(self, instance, func):
        """Constructed on demand, when the @observable method is looked up.

        :param instance: The object with the @observable method.
        :param func: the function being wrapped.

        """
        super(_MethodWithObservers, self).__init__()
        self.observers = []
        self.func = func
        self.instance_weakref = weakref.ref(instance)

        #: True while __call__() is off notifying the observers.
        self.calling_observers = False

        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self._func_repr = _method_repr(instance=instance, func=func)

    def __call__(self, *args, **kwargs):
        """Call the wrapped function, and call/manage its observers

        Those registered observers which are `_BoundObserverMethod`s signal to
        be removed when they realize their underlying instance has been
        garbage collected by raising an internal exception, which is caught
        here and handled. Observers which do this are removed (and logged at
        priority `logging.DEBUG`).

        Observers which are plain callables are assumed to be static, and
        don't get removed.

        """
        observed = self.instance_weakref()

        result = self.func(observed, *args, **kwargs)
        if self.calling_observers:
            logger.debug("Recursive call to %r detected and skipped",
                         self)
            return result
        self.calling_observers = True
        try:
            for observer in self.observers[:]:
                try:
                    observer(observed, *args, **kwargs)
                except _BoundObserverMethod._ReferenceError:
                    logger.debug('Removing %r' % (observer,))
                    self.observers.remove(observer)
                except:
                    # Exceptions raised before the observer's stack frame
                    # is entered (e.g. incorrect-parameter-number
                    # TypeErrors) don't reveal the full names.
                    # Workaround is to log the repr() of the failing item.
                    logger.error("Failed to call observer %r", observer)
                    raise
        finally:
            del observed
            self.calling_observers = False
        return result

    def __iadd__(self, observer):
        """Registers an observer with the method to be invoked after it

        :param observer: The method or function to register
        :type observer: callable

        The `observer` parameter can be a bound method or any other sort of
        callable. Bound methods are wrapped in a _BoundObserverMethod object
        internally, to avoid keeping a hard reference to the object the
        method is bound to.
        """
        self.observers.append(_wrap_observer(observer))
        return self

    def __isub__(self, observer):
        """Deregisters an observer"""
        self.observers.remove(_wrap_observer(observer))
        return self

    def __iter__(self):
        """Iterate over the list of observers"""
        return iter(self.observers)

    def __repr__(self):
        """Pretty-printing"""
        return ("<_MethodWithObservers %s>" % (self._func_repr))


class event (observable):  # noqa: N801
    """Alias for observable methods with no predefined function body.

    This allows C#-style event declarations using an alternative shorthand
    syntax, but for events forming part of a public API it's clearest to to
    use Python's decorator notation and a method body consisting of just a
    docstring.

    >>> class Popcorn (object):
    ...     @event
    ...     def popped(self):
    ...        '''Popped event, called when...'''
    ...     salted = event()
    ...     buttered = event()
    >>> popcorn = Popcorn()
    >>> pops = []
    >>> popcorn.popped += lambda p: pops.append("pop")
    >>> popcorn.buttered += lambda p: pops.append("yum")
    >>> popcorn.popped()
    >>> popcorn.salted()
    >>> popcorn.buttered()
    >>> popcorn.popped()
    >>> popcorn.popped()
    >>> pops
    ['pop', 'yum', 'pop', 'pop']

    """

    def __init__(self, func=None):
        """Construct, allowing shorthand event(), or regular @event notation

        If `func` is not given, an anonymous no-op callback is used. It's
        unique to allow the parent class's cache to work.

        """
        if func is None:
            def func(*a):
                pass
            func.__name__ = "<event>"
        super(event, self).__init__(func)


def _wrap_observer(observer):
    """Factory function for the observers in a _MethodWithObservers."""
    if _is_bound_method(observer):
        return _BoundObserverMethod(observer)
    else:
        return observer


def _is_bound_method(func):
    """True if a callable is a bound method"""
    assert callable(func)
    if hasattr(func, "__self__") and hasattr(func, "__func__"):
        # Python2 needs this test, Python3 doesn't:
        if func.__self__ is not None:
            return True
    return False


def _method_repr(bound=None, instance=None, func=None):
    """Terse, useful, hopefully permanent string repr() for a method

    Names only, given that object repr()s may change over time and this
    is cached inside some internal objects.
    """
    if bound is not None:
        assert(_is_bound_method(bound))
        func = bound.__func__
        instance = bound.__self__
    funcname = func.__name__
    clsname = instance.__class__.__name__
    modname = instance.__class__.__module__
    return "%s.%s.%s" % (modname, clsname, funcname)


class _BoundObserverMethod (object):
    """Wrapper for observer callbacks which are bound methods of some object.

    To allow short-lived objects to observe long-lived objects with bound
    methods and still be gc'able, we need weakrefs.  However it's not possible
    to take a weakref to a bound method and have that be the only thing
    referring to it.  Therefore, wrap it up as a weakref to the object the
    method is bound to (which can then die naturally), and its implementing
    function (which is always a statically allocated thing belonging to the
    class definition: those are eternal and we don't care about them).

    """

    class _ReferenceError (ReferenceError):
        """Raised when calling if the observing object is now dead."""
        pass

    def __init__(self, method):
        """Initialize for a bound method

        :param method: a bound method, or another _BoundObserverMethod to copy
        """
        super(_BoundObserverMethod, self).__init__()
        if isinstance(method, _BoundObserverMethod):
            obs_ref = method._observer_ref
            obs_func = method._observer_func
            orig_repr = method._orig_repr
        elif _is_bound_method(method):
            obs_ref = weakref.ref(method.__self__)
            obs_func = method.__func__
            orig_repr = _method_repr(bound=method)
        else:
            raise ValueError("Unknown bound method type for %r"
                             % (method,))
        self._observer_ref = obs_ref
        self._observer_func = obs_func
        self._orig_repr = orig_repr

    def __copy__(self):
        """Standard shallow copy implementation"""
        return _BoundObserverMethod(self)

    def __repr__(self):
        """String representation of a bound observer method

        >>> class C (object):
        ...    def m(self):
        ...        return 42
        >>> c = C()
        >>> bom = _BoundObserverMethod(c.m)
        >>> repr(bom)  #doctest: +ELLIPSIS
        '<_BoundObserverMethod ....C.m>'
        >>> del c
        >>> repr(bom)  #doctest: +ELLIPSIS
        '<_BoundObserverMethod ....C.m (dead)>'
        """
        dead = self._observer_ref() is None
        suff = " (dead)" if dead else ""
        return ("<_BoundObserverMethod %s%s>" % (self._orig_repr, suff))

    def __call__(self, observed, *args, **kwargs):
        """Call the bound method, or raise _ReferenceError"""
        observer = self._observer_ref()
        if observer is None:
            raise self._ReferenceError
        self._observer_func(observer, observed, *args, **kwargs)
        del observer

    def __eq__(self, other):
        """Tests for equality

        Can test against BoundObserverMethods, plain bound methods, or
        callables generally.
        """
        if _is_bound_method(other):
            return self._observer_func == other.__func__
        elif isinstance(other, _BoundObserverMethod):
            return self._observer_func == other._observer_func
        elif callable(other):
            return self._observer_func == other
        else:
            return False


class _WasAbsent:
    """The prior absence of a key in modification announcements.

    This class is only here to provide a predictable repr() for the
    doctests: use the "is" operator for any actual tests in code.

    See: ObservableDict.ABSENT (this class's only instance).

    """

    def __repr__(self):
        return "<WasAbsent>"


class ObservableDict (dict):
    """A dict whose modify ops can be observed.

    ObservableDict objects work just like the builtin dict class, but
    operations which modify the dict's contents using the modified()
    event.

    >>> od = ObservableDict({"a": 199})
    >>> od
    ObservableDict({'a': 199})
    >>> changes = []
    >>> od.modified += lambda dic, old: changes.extend(list(old))
    >>> od["b"] = 41
    >>> od["a"] += 1
    >>> changes
    ['b', 'a']
    >>> od["c"] = 101
    >>> od["d"] = 202
    >>> od.pop("a")
    200
    >>> changes
    ['b', 'a', 'c', 'd', 'a']
    >>> od.update({"b": 99, "c": 999})
    >>> len(changes)
    7
    >>> od.clear()
    >>> len(changes)  # previous 7, plus the number of keys just removed
    10
    >>> od
    ObservableDict({})

    Limitations: you need to monitor values in the dict separately.
    However, key insertions and deletions, and assignments of values to
    keys can be monitored quite nicely.

    """

    # Class constants:

    #: Value used by operations that invoke modified() to indicated
    #: the absence of a key.
    ABSENT = _WasAbsent()

    # Observable interface:

    @event
    def modified(self, old_values):
        """Event: one or more data keys were modified.

        :param dict old_values: modified keys, and the old values.

        You should not modify the ObservableDict in any function you
        attach to this event.  Also, note that the keys listed may no
        longer exist in the dict.

        The old_values argument is a dict mapping the keys that were
        changed by an operation to their *previous* values. Event
        observers can look up the new values (or their absences) in the
        OrderedDict itself.

        When a key was created by an operation, it will be listed in
        old_values. However, it will be mapped to the unique value
        ObservableDict.ABSENT.

        This event is always announced after the changes are complete.
        Batch operations like clear() or update() call it exactly once.

        """

    # Minor overrides:

    def __repr__(self):
        dict_repr = dict.__repr__(self)
        return "%s(%s)" % (type(self).__name__, dict_repr)

    # Same as the builtin dict type, but announcing changes:

    def clear(self):
        keys = list(self.keys())
        result = dict.clear(self)
        self.modified(keys)
        return result

    def __setitem__(self, key, value):
        old_value = self.get(key, self.ABSENT)
        result = dict.__setitem__(self, key, value)
        self.modified({key: old_value})
        return result

    def __delitem__(self, key):
        old_value = self.get(key, self.ABSENT)
        result = dict.__delitem__(self, key)
        self.modified({key: old_value})
        return result

    def update(self, *args, **kwargs):
        """Update from a dict, or one built from the args.

        >>> od = ObservableDict({"a": 101})
        >>> hist = []
        >>> od.modified += lambda d, o: hist.append(o)
        >>> od.update({"b": 202, "a": 303})
        >>> isinstance(hist[0], dict)
        True
        >>> sorted(list(hist[0].items()))  # first hist, predictable order
        [('a', 101), ('b', <WasAbsent>)]
        >>> hist[0]["b"] is ObservableDict.ABSENT
        True

        """
        updates = dict(*args, **kwargs)
        old_values = {k: self.get(k, self.ABSENT) for k in updates}
        result = dict.update(self, updates)
        self.modified(old_values)
        return result

    def pop(self, key, *args, **kwargs):
        old_value = self.get(key, self.ABSENT)
        result = dict.pop(self, key, *args, **kwargs)
        self.modified({key: old_value})
        return result

    def setdefault(self, key, *args, **kwargs):
        old_value = self.get(key, self.ABSENT)
        result = dict.setdefault(self, key, *args, **kwargs)
        self.modified({key: old_value})
        return result

    def popitem(self):
        (key, old_value) = dict.popitem(self)
        self.modified({key: old_value})
        return (key, old_value)

    def copy(self):
        """Make a shallow copy of the ObservableDict.

        :returns: An unobserved shallow clone.
        :rtype: ObservableDict

        """
        return self.__class__(self)

    @event
    def sync_pending_changes(self, flush=True, **kwargs):
        """Ask for pending changes to be synchronized (updated/flushed)

        This event is triggered to signal objects which have their own
        internal state that need to be be reflected in the per-doc
        settings to write their changes to the settings dict. By
        default, the request to flush changes is non-optional.

        :param bool flush: if this is False, the flush is optional too
        :param \*\*kwargs: passed through to observers

        See: `lib.observable.event` for details of the signalling
        mechanism.
        See also: lib.document.Document.sync_pending_changes().

        """

def _test():
    """Run doctest strings"""
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _test()
