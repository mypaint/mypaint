# This file is part of MyPaint.
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Observable method calls and C#-like syntactic sugar for events."""

import weakref
import sys
from warnings import warn
import logging

logger = logging.getLogger(__name__)


class observable (object):
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

    If you remove the last strong ref to such an observer, the observable
    method cleans up its internal weakref to it without any fuss the next time
    it's called:

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
            return instance.__wrappers[self.func]
        except AttributeError:
            instance.__wrappers = {}
        except KeyError:
            pass
        wrapper = observable._MethodWithObservers(instance, self.func)
        instance.__wrappers[self.func] = wrapper
        return wrapper


    class _MethodWithObservers (object):
        """Calls the decorated method, then its observers."""

        def __init__(self, instance, func):
            """Constructed on demand, when the @observable method is looked up.

            :param instance: The object with the @observable method.
            :param func: the function being wrapped.

            """
            super(observable._MethodWithObservers, self).__init__()
            self.observers = []
            self.func = func
            self.instance_weakref = weakref.ref(instance)
            self.__name__ = func.__name__
            self.__doc__ = func.__doc__

        def __call__(self, *args, **kwargs):
            """Call the wrapped function, and call/manage its observers."""
            observed = self.instance_weakref()
            result = self.func(observed, *args, **kwargs)
            for observer in self.observers[:]:
                try:
                    observer(observed, *args, **kwargs)
                except _BoundObserverMethod._ReferenceError, ex:
                    logger.debug('Removing %r' % (observer,))
                    self.observers.remove(observer)
            del observed
            return result

        def __iadd__(self, observer):
            self.observers.append(_wrap_observer(observer))
            return self

        def __isub__(self, observer):
            self.observers.remove(_wrap_observer(observer))
            return self


class event (observable):
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
            func = lambda *a: None
            func.__name__ = "<event>"
        super(event, self).__init__(func)


def _wrap_observer(observer):
    """Factory function for the observers in a _BoundObserverMethod"""
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


class _BoundObserverMethod (object):
    """Wrapper for observer callbacks which are bound methods of some object.

    To allow short-lived objects to observe long-lived objects with bound
    methods and still be gc'able, we need weakrefs.  However it's not possible
    to take a weakref to a bound method and have that be the only thing
    referring to it.  Therefore, wrapp it up as a weakref to the object the
    method is bound to (which can then die naturally), and its implementing
    function (which is always a staticly allocated thing belonging to the class
    definition: those are eternal and we don't care about them).

    """

    class _ReferenceError (weakref.ReferenceError):
        """Raised when calling if the observer object is now dead."""
        pass

    def __init__(self, observer):
        """Initialize for a bound method."""
        super(_BoundObserverMethod, self).__init__()
        self._instance_weakref = weakref.ref(observer.__self__)
        self._observer_method = observer.__func__
        self._orig_repr = "%r of %r" % (observer.__func__.__name__,
                                        observer.__self__)

    def __repr__(self):
        if self._instance_weakref() is not None:
            return self._orig_repr
        else:
            return ("<dead _BoundObserverMethod, formerly {%s}>"
                    % (self._orig_repr,))


    def __call__(self, observed, *args, **kwargs):
        """'Rebind' and call, or raise _ReferenceError."""
        observer = self._instance_weakref()
        if observer is None:
            raise self._ReferenceError
        self._observer_method(observer, observed, *args, **kwargs)
        del observer


def _test():
    import doctest
    doctest.testmod()
    #class Hive(object):
    #    @event
    #    def poked(self):
    #        pass
    #def _fury(h):
    #    raise RuntimeError
    #h = Hive()
    #h.poked += _fury
    #h.poked()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _test()

