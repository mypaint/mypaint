# This file is part of MyPaint.
# Copyright (C) 2013-2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""String and tuple-based construction and reconstruction of objects."""

## Imports

from __future__ import division, print_function
import logging
from warnings import warn

from lib.gibindings import GObject

from lib.observable import event

logger = logging.getLogger(__name__)


## Class definitions

class ConstructError (Exception):
    """Errors encountered when constructing objects.

    Raised when an object cannot be looked up by GType name:

        >>> import gi
        >>> from lib.gibindings import Gtk
        >>> make_widget = ObjFactory(gtype=Gtk.Entry)
        >>> make_widget("NonExist12345")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ConstructError: Cannot construct a 'NonExist12345': module not imp[...]

    Just importing a module defining a class
    with a "__gtype_name__" defined for it into the running Python interpreter
    is sufficient to clue GObject's type system
    into the existence of the class, so the error message refers to that.
    This exception is also raised when construction
    fails because the type subclassing requirements are not met:

        >>> make_widget("GtkLabel")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ConstructError: GtkLabel is not a subclass of GtkEntry

    """


class ObjFactory (object):
    """Pythonic cached factory for GObjects.

    Objects are constructable from their GObject type name and a simple tuple
    containing any construction parameters needed.

      >>> import gi
      >>> from lib.gibindings import Gtk
      >>> make_widget = ObjFactory(gtype=Gtk.Widget)
      >>> w1 = make_widget.get("GtkLabel", "Hello, World",)
      >>> w1 is not None
      True

    Factories can be used as functions, given that they basically have only a
    single job to do.

      >>> w2 = make_widget("GtkLabel", "Hello, World")

    The combination of GObject type name and parameters provides a meaningful
    identity for a UI element, e.g. a particular window with its own button
    launcher.  The identifiers are used as a cache key internally.

      >>> w1 is w2
      True

    Identities can be extracted from objects built by the factory:

      >>> make_widget.identify("constructed elsewhere") is None
      True
      >>> saved_ident = make_widget.identify(w1)
      >>> saved_ident
      ('GtkLabel', 'Hello, World')

    and allow their reconstruction in future app sessions:

      >>> w3 = make_widget(*saved_ident)
      >>> w3 is w1
      True

    """

    def __init__(self, gtype=None):
        """Constructs, with an optional required type.

        :param gtype: a required type
        :type gtype: Python GI class representation

        If `gtype` is defined, the factory will be limited to producing objects
        of that type (or its subclasses) only.

        """
        super(ObjFactory, self).__init__()
        self._required_type = gtype
        self._cache = {}

    def get(self, gtype_name, *params):
        """Fetch an object by identity, via an internal cache.

        A cache is used, to avoid overconstruction.  If construction is needed,
        the type name is used to obtain the Python class representing the
        GObject type, which is then instantiated by passing its Python
        constructor the supplied parameters as its ``*args``.

        Construction parameters are assumed to qualify and specialize objects
        sufficiently for `params` plus the type name to form a meaningful
        identity for the object.

        This is the same concept of identity the cache uses.  If the
        construction parameters need to change during the lifetime of the
        object to maintain this identity, the `rebadge()` method can be used to
        update them and allow the object to be reconstructed correctly for the
        next session.

        :param gtype_name: a registered name (cf. __gtype_name__)
        :type gtype_name: str
        :param params: parameters for the Python constructor
        :type params: tuple
        :returns: the newly constructed object
        :rtype: GObject
        :raises ConstructError: when construction fails.

        Fires `object_created()` after an object has been successfully created.

        """
        key = self._make_key(gtype_name, params)
        if key in self._cache:
            return self._cache[key]
        logger.debug("Creating %r via factory", key)
        try:
            gtype = GObject.type_from_name(gtype_name)
        except RuntimeError:
            raise ConstructError(
                "Cannot construct a '%s': module not imported?"
                % gtype_name
            )
        if self._required_type:
            if not gtype.is_a(self._required_type):
                raise ConstructError(
                    "%s is not a subclass of %s"
                    % (gtype_name, self._required_type.__gtype__.name)
                )
        try:
            product = gtype.pytype(*params)
        except Exception:
            warn("Failed to construct a %s (pytype=%r, params=%r)"
                 % (gtype_name, gtype.pytype, params),
                 RuntimeWarning)
            raise
        product.__key = key
        self._cache[key] = product
        self.object_created(product)
        return product

    def __call__(self, gtype_name, *params):
        """Shorthand allowing use as as a factory pseudo-method."""
        return self.get(gtype_name, *params)

    @event
    def object_created(self, product):
        """Event: an object was created by `get()`

        :param product: The newly constructed object.
        """

    def cache_has(self, gtype_name, *params):
        """Returns whether an object with the given key is in the cache.

        :param gtype_name: gtype-system name for the object's class.
        :param params: Sequence of construction params.
        :returns: Whether the object with this identity exists in the cache.
        :rtype: bool
        """
        key = self._make_key(gtype_name, params)
        return key in self._cache

    def identify(self, product):
        """Gets the typename & params of an object created by this factory.

        :param product: An object created by this factory
        :returns: identity tuple, or `None`
        :rtype: None, or a tuple, ``(GTYPENAME, PARAMS...)``
        """
        try:
            key = product.__key
        except AttributeError:
            return None
        return key

    @staticmethod
    def _make_key(gtype_name, params):
        """Internal cache key creation function.

        >>> ObjFactory._make_key("GtkLabel", ["test test"])
        ('GtkLabel', 'test test')

        """
        return tuple([gtype_name] + list(params))

    def rebadge(self, product, new_params):
        """Changes the construct params of an object.

        Use this when a constructed object has had something intrinsic changed
        that's encoded as a construction parameter.

        :params product: An object created by this factory.
        :params new_params: A new sequence of identifying parameters.
        :rtype: bool
        :returns: Whether the rebadge succeeded.

        Rebadging will fail if another object exists in the cache with the same
        identity.  If successful, this updates the factory cache, and the
        embedded identifier in the object itself.

        Fires `object_rebadged()` if the parameters were actually changed.
        Changing the params to their current values has no effect, and does not
        fire the @event.

        """
        old_key = self.identify(product)
        gtype_name = old_key[0]
        old_params = old_key[1:]
        new_key = self._make_key(gtype_name, new_params)
        if old_key == new_key:
            return True
        if new_key in self._cache:
            return False
        product.__key = new_key
        self._cache[new_key] = product
        self._cache.pop(old_key)
        self.object_rebadged(product, old_params, new_params)
        return True

    @event
    def object_rebadged(self, product, old_params, new_params):
        """Event: object's construct params were updated by `rebadge()`"""


if __name__ == '__main__':
    logging.basicConfig()
    import doctest
    doctest.testmod()
