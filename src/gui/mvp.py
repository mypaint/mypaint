# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2017 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Mixin and decorator functionality for the MVP UI pattern."""

# Imports:

import os
import inspect
import abc
import functools
import logging

from lib.gibindings import Gtk


logger = logging.getLogger(__name__)


# Main API classes:

class Presenter:
    """Abstract interface for standardized MVP presenters.

    What's MVP?
    -----------

    "Model-View-Presenter" or "MVP" is a UI pattern where a presenter
    object mediates between a View and a Model. The presenter observes
    changes to both sides, and for each change, updates the other side.
    Presenters encapsulate presentation logic and formatting for a set
    of displayable objects, termed the "view". They also turn user
    actions into internal state changes of obejcts which cannot by
    themselves be shown to the user, called the "model".

    Ref: https://en.wikipedia.org/wiki/Model-view-presenter

    MyPaint MVP conventions
    -----------------------

    In MyPaint, a Presenter's View is a hierarchy of basic GTK objects
    that are typically laid out with the Glade UI designer and stored as
    an XML file. View observation is done with GTK signal callback
    methods, which are named in the XML UI definition, and which are
    expected to be implemented by the corresponding Presenter object.

    In MyPaint, a Presenter's Model can be anything. It's normally one
    or more objects from lib, for example a layer or a whole layers
    stack.  Observation of model objects happens by attaching callbacks
    to their "@event" methods (see lib.observable).

    In the suggested implementation, presenters are tightly bound to
    their view, and often own and instantiate it on demand. See
    BuiltUIPresenter for a mixin class that makes this easy.  They may
    be quite loosely bound to their model if needed, and are typically
    introduced to it during construction. The rest of the code,
    including other presenters, should keep a ref to the presenter for
    as long as the presentation logic needs to happen.

    It is conventional here to name concrete presenter classes like
    "<Model>UI" or "<Role>UI", noting that a presenter provides the
    behavioural aspects of the user interface.

    MyPaint encourages tight-ish coupling of a View to its Presenter(s)
    but looser coupling of Presenters to their Model objects. Presenters
    can own their View. Conversely, presenters almost never own very
    complicated models: most use their view to present aspects of a
    model owned by something else.

    For a presenter observing its model hierarchy, connect methods to
    "@event"s exposed by the highest level model object you can find.
    See "lib.observable" for how this makes garbage collection nicer.
    Conversely, when a presenter needs to observe its view herarchy, use
    standard GTK signals and individual widgets' connect() method.

    Decorators
    ----------

    Use the @model_updater and @view_updater decorators on the
    Presenter's callback methods to save on having to write tons of
    fiddly value tests or implementing other ways of preventing loops.

    Decorate each model observer callback and each view signal handler
    callback to make sure you don't end up with a cascade of calls.
    The callbacks can be specified with or without args.

    >>> class SomeUI (Presenter):
    ...
    ...     @view_updater
    ...     def model_field_A_updated_cb(self, *args, **kwargs):
    ...         pass
    ...
    ...     @view_updater(default=42)
    ...     def model_field_B_updated_cb(self, *args, **kwargs):
    ...         pass
    ...
    ...     @view_updater
    ...     def view_widget_1_updated_cb(self, *args, **kwargs):
    ...         pass
    ...
    ...     @model_updater(default=False)
    ...     def view_widget_2_updated_cb(self, *args, **kwargs):
    ...         pass

    Why do this? It follows from the MVP pattern that if you're not
    careful about the update flow, you can end up with a loop.

    These decorators fix that potential problem by completely skipping
    the wrapped methods if the Presenter is currently doing the *other*
    type of update.  You can return a default value you specify if
    needed: this can keep GTK+ signals happier.

    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def widget(self):
        """The primary widget of the view.

        :returns: the main view widget
        :rtype: Gtk.Widget

        All presenters must expose a single widget to be the primary
        entry point to the view. This property should be overridden to
        provide, for example, the widget to use for the contents of a
        fancy dialog, or the toplevel window in an application.

        For "adoptive" presenters (ones where the Presenter adopts a
        given widget hierarchy as its view, and connects signals to it),
        make the "widget" property *read-write* and do the extra hookup
        work in your "@widget.setter". Your presenter's owner then has a
        clean interface for attaching bundles of additional behaviour to
        parts of its own view.

        """

    @staticmethod
    def _updater(func=None, default=None):
        """Decorates methods that must only be called once at any one time.

        :param callable func: The method to be wrapped.
        :param default: Return value for when the the method is skipped.
        :returns: the decorated method.

        The wrapped function is skipped if it is currently being called
        on the presenter.

        Use this decorator (as model_updater or view_updater) for all
        model or view observer callbacks which update the other side.
        Doing this prevents potential loops.

        See also: lib.observable.event, Gtk.Widget.connect.

        """

        # Allow the @pie syntax to be used with or without args.
        if func is None:
            return functools.partial(Presenter._updater, default=default)

        @functools.wraps(func)
        def method_wrapper(inst, *args, **kwargs):
            try:
                in_call = func.__in_call
            except AttributeError:
                in_call = False

            if in_call:
                logger.debug("@_updater: suppressed a call to %r", func)
                return default

            func.__in_call = True
            try:
                return func(inst, *args, **kwargs)
            except:
                raise
            finally:
                func.__in_call = False

        return method_wrapper


class BuiltUIPresenter (Presenter):
    """Mixin providing Pythonic access to views built from GtkBuilder XML.

    This style of presenter has its view defined entirely by a
    corresponding GTK+ UI XML file. The view is constructed on demand by
    methods here, and its signals will be automatically bound to methods
    of the Presenter when the view objects are constructed. The
    Presenter mixin interface provides overridable hooks for setting the
    initial state of the view.

    Subclasses should define `primary_widget` so as to access and return
    a widget object from `view`.

    """

    _UI_FILE_EXTENSIONS = [".ui", ".glade", ".xml"]

    @property
    def view(self):
        """On-demand access the built View objects by attribute lookup.

        :returns: A wrapper for accessing UI objects.
        :rtype: _ViewWrapper

        When accessed for the first time, this property method
        constructs an internal GtkBuilder, and uses it to instantiate
        GTK objects from a UI XML file.

        The file is expected to reside in the same directory as
        self.__class__.__file___, and to be named after it (*.glade).
        However, if self.__ui_xml__ is defined, that will be used for
        the basename instead.

        Upon construction, the builder connects the new objects' signals
        to self.

        """

        # Cache the loaded UI wrapper.
        try:
            return self.__ui_wrapper
        except AttributeError:
            pass

        # Select a .glade file
        class_defn_file = inspect.getfile(self.__class__)
        mod_dirname = os.path.dirname(class_defn_file)
        try:
            ui_file_basenames = [self.__ui_xml__]
        except AttributeError:
            py_basename = os.path.basename(class_defn_file)
            py_basename, _oldext = os.path.splitext(py_basename)
            ui_file_basenames = [
                py_basename + e
                for e in self._UI_FILE_EXTENSIONS
            ]

        # Load objects, bind, cache and return.
        for basename in ui_file_basenames:
            ui_path = os.path.join(mod_dirname, basename)
            logger.debug("BuiltUIPresenter: trying to load %r", ui_path)
            if not os.path.isfile(ui_path):
                continue
            logger.debug(
                "BuiltUIPresenter: found UI definition in %r; loading",
                ui_path,
            )
            wrapper = _ViewWrapper(ui_path)
            self.__ui_wrapper = wrapper
            self.init_view()
            wrapper._builder.connect_signals(self)
            self.init_view_post_connect()
            return wrapper

        raise RuntimeError(
            "BuiltUIPresenter: could not load "
            "any UI definition XML file (tried %r, in %r)"
            % (ui_file_basenames, mod_dirname),
        )

    def init_view(self):
        """Hook: initialize the view objects (before signal connection).

        Implementations should set the initial state of all relevant UI
        object to reflect the state of the model. It is called before
        signals are connected, for convenience.

        Note that this method is called the first time the view()
        property is accessed.

        This base implementation does nothing. You'll typically want to
        use this rather than init_view_post_connect().

        """

    def init_view_post_connect(self):
        """Hook: initialize the view objects (AFTER signal connection).

        This method is called after connection of signals, but is
        otherwise identical to init_view(). Override that method
        instead, unless you have specific needs.

        """


# Convenience/selfdoc aliases for the updater decorator:

model_updater = Presenter._updater
view_updater = Presenter._updater


# Helper classes:

class _ViewWrapper:
    """Private GTK+ view object abstraction.

    This can be accessed by the ID of object IDs inside the
    corresponding UI XML file.

    """

    _TRANSLATION_DOMAIN = "mypaint"

    def __init__(self, filename):
        self._filename = filename
        builder = Gtk.Builder()
        builder.set_translation_domain(self._TRANSLATION_DOMAIN)
        builder.add_from_file(filename)
        self._builder = builder
        self._cache = {}

    def __getattr__(self, attr_name):
        try:
            return self._cache[attr_name]
        except KeyError:
            pass

        for name in (attr_name, attr_name.replace("_", "-")):
            obj = self._builder.get_object(name)
            if obj is not None:
                self._cache[attr_name] = obj
                return obj

        raise AttributeError(
            "No object with name %r (incl. \"_\" substs) in %r"
            % (attr_name, self._filename),
        )
