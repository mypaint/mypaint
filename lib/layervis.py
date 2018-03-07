# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Bulk management of layer visibility: backend and model.

The use case is a user preparing an artwork with multiple variations,
possibly with a lot shared between them.  For example, they may want one
file to store different expressions and poses for a character,
or different times of day and lighting for a landscape.

A user needing this should be able to store the set of layers they see
right now in the form of a viewing context, and assign a name to to the
context. They should be able to switch between these different contexts
at will. On each such transition, the set of layers which are visible in
the document changes accordingly. These transitions are undoable.

"""

# Imports:

from __future__ import print_function, division

import logging

from lib.gettext import C_
import lib.naming
from lib.observable import event
from lib.command import Command
from lib.pycompat import unicode


# Module vars:

logger = logging.getLogger(__name__)

NEW_VIEW_IDENT = C_(
    "layer visibility sets: default name for a user-managed view",
    u"View",
)

UNSAVED_VIEW_DISPLAY_NAME = C_(
    "layer visibility sets: text shown when no user-managed view is active",
    u"No active view",
)


# Data model classes:

class _View (object):
    """Lightweight representation of a layer viewing context.

    Views are represented as essentially just a tag.  They intentionally
    do not contain references to their layers to avoid circular refs.
    Instead, layers contain refs to their views in a private set that
    the LayerViewManager knows about.

    """

    def __init__(self, name, locked=True, **kwargs):
        super(_View, self).__init__()
        self._name = unicode(name)
        self._locked = bool(locked)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = unicode(value)

    @property
    def locked(self):
        return self._locked

    @locked.setter
    def locked(self, state):
        self._locked = bool(state)

    def __repr__(self):
        return "<{cls} {id:#016x} {name!r} {locked}>".format(
            cls=self.__class__.__name__,
            id=id(self),
            name=self._name,
            locked="locked" if self._locked else "unlocked",
        )

    def __str__(self):
        return self._name.encode("unicode_escape")

    def __unicode__(self):
        return self._name

    def __eq__(self, other):
        """Equality tests.

        >>> _View("name1") == _View("name1")
        False
        >>> t1 = t2 = _View("name3")
        >>> t1 is t2
        True
        >>> t1 == t2
        True

        Views can be compared to other views (uses ``is``),
        or by (unicode) string equality.

        """
        return self is other

    def __hash__(self):
        return id(self)

    def __bool__(self):
        return True

    def to_jsf(self):
        """Convert to a form serializable by the json module."""
        return {
            "name": self.name,
            "locked": self.locked,
        }

    @classmethod
    def new_from_jsf(cls, jsf):
        """Returns a new _View built from the json-serializable form."""
        return cls(**jsf)


class _NamedViewsSet (object):
    """A set of _View objects that enforces unique naming."""

    def __init__(self):
        super(_NamedViewsSet, self).__init__()
        self.objs = set()  # {_View}
        self.names = dict()  # {str: _View}

    def add(self, view):
        if view in self.objs:
            return
        name = lib.naming.make_unique_name(
            view.name, self.names,
            always_number=NEW_VIEW_IDENT,
        )
        assert name not in self.names
        if view.name != name:
            view.name = name
        self.names[name] = view
        self.objs.add(view)

    def remove(self, view):
        if view not in self.objs:
            return
        self.remove_by_name(view.name)
        assert view not in self.objs

    def remove_by_name(self, name):
        if name not in self.names:
            return
        view = self.names.pop(name)
        self.objs.remove(view)

    def clear(self):
        self.objs.clear()
        self.names.clear()


class LayerViewManager (object):
    """Controls which layers are visible in a document with named views."""

    _SETTINGS_KEY = "layervis"
    _SETTINGS_VIEWS_SUBKEY = "views"
    _SETTINGS_ACTIVE_VIEW_SUBKEY = "active_view"
    _SETTINGS_LAYER_VIEWS_SUBKEY = "layer_views"

    # Construction:

    def __init__(self, docmodel):
        super(LayerViewManager, self).__init__()
        self._docmodel = docmodel
        root = docmodel.layer_stack
        self._stack = docmodel.layer_stack

        # The currently active view.
        # A value of None means that no named view is active.
        self._current_view = None

        # The set of views managed by the user.
        # There may be more view objects in existence,
        # for example in the command stack.
        self._views = _NamedViewsSet()

        # Layer stack monitoring.
        root.layer_properties_changed += self._stack_layer_props_changed_cb
        root.layer_inserted += self._stack_layer_inserted_cb

        # Observe myself to change layer views when it's safe.
        self.current_view_changed += self._current_view_changed_cb

        # Save and load.
        docmodel.settings.sync_pending_changes \
            += self._doc_settings_sync_pending_changes_cb
        docmodel.settings.modified += self._doc_settings_modified_cb

    # Loading and saving via the doc settings dictionary:

    def _doc_settings_sync_pending_changes_cb(self, settings, flush=False):
        """Save to the doc settings when needed (e.g. before ORA save)

        This is called before the document is saved (and at many other
        times, including autosave), so use it to store a serializable
        copy of the running state into the settings dict.

        See also: _doc_settings_modified_cb().

        """
        # Don't do anything while the doc settings are being updated.
        if self._docmodel.settings.modified.calling_observers:
            return
        # Only serious changes.
        if not flush:
            return
        # TODO: use a dirty flag to avoid too many walks?

        # The user-managed list of views
        views_list = [v.to_jsf() for v in self._views.objs]

        # The currently active view
        current = None
        if self._current_view is not None:
            current = self._current_view.name

        # Each layer's views
        by_path = []
        for path, layer in self._docmodel.layer_stack.walk():
            vset = self._get_vset_for_layer(layer)
            names = [v.name for v in vset if v in self._views.objs]
            by_path.append([list(path), names])

        self._docmodel.settings[self._SETTINGS_KEY] = {
            self._SETTINGS_VIEWS_SUBKEY: views_list,
            self._SETTINGS_ACTIVE_VIEW_SUBKEY: current,
            self._SETTINGS_LAYER_VIEWS_SUBKEY: by_path,
        }

    def _doc_settings_modified_cb(self, settings, oldvalues):
        """Update state when the doc settings change (e.g. ORA load).

        This clears and completely rebuilds the internal state of the
        manager object to match the serializable form in the settings
        dict. An important use is to restore the state after the
        document settings have been loaded from a .ora file.

        See also: _doc_sync_pending_changes_cb().

        """
        # Don't run when storing.
        if self._docmodel.settings.sync_pending_changes.calling_observers:
            return
        # Only when the settings key changes.
        if self._SETTINGS_KEY not in oldvalues:
            return

        sdict = settings.get(self._SETTINGS_KEY, {})

        # Reconstruct the user-managed list of views from a list of
        # stored names.
        self._views.clear()
        try:
            views_list = sdict.get(self._SETTINGS_VIEWS_SUBKEY, [])
            for jsf in views_list:
                view = _View.new_from_jsf(jsf)
                self._views.add(view)
        except Exception:
            logger.exception(
                "settings-modified: failed to load settings[%r][%r]",
                self._SETTINGS_KEY,
                self._SETTINGS_VIEWS_SUBKEY,
            )

        # Reconstruct the current view from the stored name.
        self._current_view = None
        try:
            current_name = sdict.get(self._SETTINGS_ACTIVE_VIEW_SUBKEY, None)
            if current_name is not None:
                self._current_view = self._views.names.get(current_name, None)
        except Exception:
            logger.exception(
                "settings-modified: failed to load settings[%r][%r]",
                self._SETTINGS_KEY,
                self._SETTINGS_ACTIVE_VIEW_SUBKEY,
            )

        # Reconstruct layer vsets from he stored names.
        path_to_layer = {}
        for path, layer in self._docmodel.layer_stack.walk():
            path_to_layer[path] = layer
            vset = self._get_vset_for_layer(layer)
            vset.clear()
        try:
            vset_rec = sdict.get(self._SETTINGS_LAYER_VIEWS_SUBKEY, [])
            for path, tagnames in vset_rec:
                path = tuple(path)
                layer = path_to_layer.get(path)
                if layer is None:
                    return
                layer_vset = self._get_vset_for_layer(layer)
                for tagname in tagnames:
                    view = self._views.names.get(tagname, None)
                    if tagname is None:
                        continue
                    layer_vset.add(view)
        except Exception:
            logger.exception(
                "settings-modified: failed to load [%r]['layer-visibs']",
                self._SETTINGS_KEY,
            )

        # Announce the changes, to update any connected UI.
        self.view_names_changed()
        self.current_view_changed()

    # Public properties

    @property
    def current_view_name(self):
        """RO property: the current view's name.

        :returns: The name of the current view, or None
        :rtype: unicode

        If the current view name is None, the current view is the
        built-in unnamed and unsaved view.

        """
        if self._current_view is None:
            return None
        return self._current_view.name

    @property
    def current_view_locked(self):
        """RO property: the current view's lock state.

        :returns: Whether the current view is locked.
        :rtype: bool

        The built-in, unsaved view is always unlocked.

        """
        if self._current_view is None:
            return False
        return self._current_view.locked

    @property
    def view_names(self):
        """RO property: list of the current set of managed view names."""
        return list(self._views.names.keys())

    # Public events:

    @event
    def current_view_changed(self):
        """Event: the current view was changed."""

    @event
    def view_names_changed(self):
        """Event: view_names is different."""

    # Self-observation:

    def _current_view_changed_cb(self, _lvm):
        """Respond to a change of the current view: set layer visibilities.

        Doing this as a self-observer callback method allows
        _stack_layer_props_changed_cb to safely ignore the changes to
        layer visibilities that originate here.

        """
        view = self._current_view
        if view is None:
            return

        for path, layer in self._stack.walk():
            vset = self._get_vset_for_layer(layer)
            if (view in vset) and not layer.visible:
                layer.visible = True
            elif (view not in vset) and layer.visible:
                layer.visible = False

    # Document observation:

    def _stack_layer_props_changed_cb(self, root, path, layer, changed):
        """Respond to any outside change of layer "visible" properties."""
        view = self._current_view
        if view is None:
            return
        if self.current_view_changed.calling_observers:
            return
        if "visible" not in changed:
            return
        vset = self._get_vset_for_layer(layer)
        if layer.visible:
            vset.add(view)
        elif view in vset:
            vset.remove(view)

    def _stack_layer_inserted_cb(self, root, path):
        view = self._current_view
        if view is None:
            return
        layer = root.deepget(path)
        if layer is None:
            return
        vset = self._get_vset_for_layer(layer)
        if layer.visible:
            vset.add(view)
        elif view in vset:
            vset.remove(view)

    def _get_vset_for_layer(self, layer):
        """Gets the "visible in views" set for a layer."""
        try:
            vset = layer.__visible_in_views
        except AttributeError:
            vset = set()
            layer.__visible_in_views = vset
        return vset

    # Higher-level API:

    def clear(self):
        self._current_view = None
        self.current_view_changed()
        self._views.clear()
        self.view_names_changed()

    def add_new_view(self, name=None):
        """Adds a new named view capturing the currently visible layers.

        :param unicode name: Base name for a new named view, or None.
        :rtype: _View
        :returns: the added view.

        If name=None or name="" is passed, the new view will be named
        uniquely after NEW_VIEW_IDENT. The None value is reserved for
        representing the default working view.

        """
        if name is None or name == "":
            name = NEW_VIEW_IDENT
        name = unicode(name)

        # All currently visible layers are tagged as visible in the new view.
        view = _View(name)
        for path, layer in self._stack.walk():
            if layer.visible:
                vset = self._get_vset_for_layer(layer)
                vset.add(view)

        self._views.add(view)
        self.view_names_changed()
        self.activate_view(view)
        return view

    def add_view(self, view):
        """Adds a view, but does not activate it."""
        if view is None:
            raise ValueError("Cannot add None")
        self._views.add(view)
        self.view_names_changed()

    def activate_view(self, view):
        """Activates a view."""
        if view is not None:
            if view not in self._views.objs:
                raise ValueError("view not in views list")
        self._current_view = view
        self.current_view_changed()

    def activate_view_by_name(self, name):
        """Activates a view by name."""
        view = None
        if name is not None:
            view = self._views.names.get(name, None)
            if view is None:
                raise ValueError("no view named %r in views list" % (name,))
        return self.activate_view(view)

    def remove_active_view(self, restore=None):
        """Removes the currently active view.

        :param _View restore: Optional view to restore.

        """
        view = self._current_view
        if view is None:
            raise ValueError("Cannot delete the default view.")

        self.activate_view(restore)

        self._views.remove(view)
        self.view_names_changed()

        return view

    def rename_active_view(self, name):
        """Renames the currently active view.

        :param unicode name: Base name for the new named view.
        :rtype: tuple
        :returns: The old and new names, as (old_name, new_unique_name)

        """
        view = self._current_view
        if view is None:
            raise ValueError("Cannot rename the default view.")
        if name == "" or name is None:
            name = NEW_VIEW_IDENT
        old_name = view.name
        popped_tag = self._views.names.pop(old_name)
        assert popped_tag is view
        new_name = lib.naming.make_unique_name(
            name, self._views.names,
            always_number=NEW_VIEW_IDENT,
        )
        self._views.names[new_name] = popped_tag
        view.name = new_name

        self.view_names_changed()
        self.current_view_changed()

        return (old_name, new_name)

    def set_active_view_locked(self, locked):
        view = self._current_view
        if view is None:
            raise ValueError("Cannot lock or unlock the default view.")
        locked = bool(locked)
        if locked != bool(view.locked):
            view.locked = locked
            self.current_view_changed()


# Command classes:

class AddLayerView (Command):
    """Adds a new layer visibility set, capturing what's visible now."""

    def __init__(self, doc, name=None, **kwds):
        super(AddLayerView, self).__init__(doc, name=name, **kwds)
        self._lvm = doc.layer_view_manager
        self._prev_active_view = None
        self._new_view_name_orig = name

    def redo(self):
        assert self._prev_active_view is None
        self._prev_active_view = self._lvm._current_view
        name = self._new_view_name_orig
        self._lvm.add_new_view(name=name)
        assert self._lvm._current_view is not self._prev_active_view
        assert self._lvm._current_view is not None

    def undo(self):
        assert self._lvm._current_view is not None
        self._lvm.remove_active_view(restore=self._prev_active_view)
        assert self._lvm._current_view is self._prev_active_view
        self._prev_active_view = None

    @property
    def display_name(self):
        return C_(
            "layer views: commands: add",
            u"Add Layer View",
        )


class RemoveActiveLayerView (Command):
    """Removes the active layer view."""

    def __init__(self, doc, **kwds):
        super(RemoveActiveLayerView, self).__init__(doc, **kwds)
        self._lvm = doc.layer_view_manager
        self._removed_view = None

    def redo(self):
        removed = self._lvm.remove_active_view()
        self._removed_view = removed

    def undo(self):
        self._lvm.add_view(self._removed_view)
        self._lvm.activate_view(self._removed_view)
        self._removed_view = None

    @property
    def display_name(self):
        return C_(
            "layer views: commands: remove",
            u"Remove Layer View",
        )


class RenameActiveLayerView (Command):
    """Renames the active layer view."""

    def __init__(self, doc, name, **kwds):
        super(RenameActiveLayerView, self).__init__(doc, name=name, **kwds)
        self._lvm = doc.layer_view_manager

        self._old_name = None
        self._new_name = name

    def redo(self):
        assert self._old_name is None
        (old_name, new_name) = self._lvm.rename_active_view(self._new_name)
        self._old_name = old_name

    def undo(self):
        assert self._old_name is not None
        old_name = self._old_name
        self._lvm.rename_active_view(old_name)
        self._old_name = None

    @property
    def display_name(self):
        return C_(
            "layer views: commands: remove",
            u"Remove Layer View",
        )


class ActivateLayerView (Command):
    """Makes a different layer view active."""

    def __init__(self, doc, name, **kwds):
        super(ActivateLayerView, self).__init__(doc, name=name, **kwds)
        self._lvm = doc.layer_view_manager
        assert name in self._lvm.view_names or name is None
        assert name is not self._lvm.current_view_name
        self._prev_view = None
        self._next_view_name = name

    def redo(self):
        prev = self._lvm._current_view
        self._lvm.activate_view_by_name(self._next_view_name)
        self._prev_view = prev

    def undo(self):
        prev = self._prev_view
        self._lvm.activate_view(prev)
        assert prev is self._lvm._current_view
        self._prev_view = None

    @property
    def display_name(self):
        return C_(
            "layer views: commands: activate",
            u"Activate Layer View “{name}”",
        ).format(
            name=self._next_view_name,
        )


class SetActiveLayerViewLocked (Command):
    """Sets the locked state of the active layer view."""

    def __init__(self, doc, locked, **kwds):
        super(SetActiveLayerViewLocked, self) \
            .__init__(doc, locked=locked, **kwds)
        self._lvm = doc.layer_view_manager
        self._old_locked = None
        self._new_locked = locked

    def redo(self):
        self._old_locked = self._lvm.current_view_locked
        self._lvm.set_active_view_locked(self._new_locked)

    def undo(self):
        self._lvm.set_active_view_locked(self._old_locked)
        self._old_locked = None

    @property
    def display_name(self):
        if self._new_locked:
            return C_(
                "layer views: commands: set active view's lock flag",
                u"Lock Layer View",
            )
        else:
            return C_(
                "layer views: commands: set active view's lock flag",
                u"Unlock Layer View",
            )
