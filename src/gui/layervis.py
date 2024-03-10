# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2017 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Bulk management of layer visibility: presentation layer and view."""

# Imports:

from __future__ import print_function, division

import gui.mvp
import logging
import gui.dialogs
import gui.application
import lib.layervis
from lib.gettext import C_
from lib.xml import escape

from lib.gibindings import Gtk


# Module vars:

logger = logging.getLogger(__name__)


# Class definitions:

class LayerViewUI (gui.mvp.BuiltUIPresenter, object):
    """Presents the document's named layer views in a row of controls."""

    def __init__(self, docmodel):
        super(LayerViewUI, self).__init__()
        self._docmodel = docmodel
        self._lvm = docmodel.layer_view_manager
        self._app = gui.application.get_app()

    def _init_model(self):
        lvm = self._lvm
        lvm.current_view_changed += self._current_view_changed_cb
        lvm.view_names_changed += self._view_names_changed_cb

    def init_view(self):
        self._init_model()
        store = Gtk.ListStore(str, str)   # columns: <our_id, display_markup>
        markup = C_(
            "view controls: dropdown: item markup",
            u"<i>{builtin_view_name}</i>",
        ).format(
            builtin_view_name=escape(lib.layervis.UNSAVED_VIEW_DISPLAY_NAME),
        )
        # None has a special meaning for GTK combo ID columns,
        # so we substitute the empty string. Corrolory: you can't name
        # views to the empty string.
        store.append([u"", markup])
        store.set_sort_column_id(0, Gtk.SortType.ASCENDING)
        self._store = store

        combo = self.view.current_view_combo
        combo.set_model(store)
        combo.set_id_column(0)

        cell = self.view.layer_text_cell
        combo.add_attribute(cell, "markup", 1)

        self._refresh_ui()

    # MVP implementation:

    @property
    def widget(self):
        return self.view.layout_grid

    # View updater callbacks:

    @gui.mvp.view_updater
    def _current_view_changed_cb(self, lvm):
        self._update_current_view_name()
        self._update_buttons()

    @gui.mvp.view_updater
    def _view_names_changed_cb(self, lvm):
        self._update_view_combo_names()

    @gui.mvp.view_updater
    def _refresh_ui(self):
        self._update_view_combo_names()
        self._update_current_view_name()
        self._update_buttons()

    @gui.mvp.view_updater
    def _update_view_combo_names(self):
        """Update the combo box to reflect a changed list names."""
        model_names = set(self._lvm.view_names)
        store = self._store

        kept_names = set()
        it = store.get_iter_first()
        while it is not None:
            name, = store.get(it, 0)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            # Skip over our representation of the builtin
            if name == u"":
                it = store.iter_next(it)
                continue
            # Drop or keep
            if name not in model_names:
                it_still_valid = store.remove(it)  # advances it
                if not it_still_valid:
                    it = None
            else:
                it = store.iter_next(it)
                kept_names.add(name)

        for name in model_names:
            if name not in kept_names:
                markup = C_(
                    "view controls: dropdown: item markup",
                    u"{user_chosen_view_name}",
                ).format(
                    user_chosen_view_name=escape(name),
                )
                store.append([name, markup])

    @gui.mvp.view_updater
    def _update_buttons(self):
        current = self._lvm.current_view_name
        op_buttons = [
            self.view.rename_button,
            self.view.remove_button,
            self.view.locked_button,
        ]
        for b in op_buttons:
            b.set_sensitive(current is not None)

        icon_name = "mypaint-object-unlocked-symbolic"
        active = self._lvm.current_view_locked
        if self._lvm.current_view_locked:
            icon_name = "mypaint-object-locked-symbolic"
        self.view.locked_image.set_from_icon_name(
            icon_name,
            Gtk.IconSize.BUTTON,
        )
        btn = self.view.locked_button
        if bool(active) != bool(btn.get_active()):
            btn.set_active(active)

    @gui.mvp.view_updater
    def _update_current_view_name(self):
        combo = self.view.current_view_combo
        id_ = self._lvm.current_view_name
        if id_ is None:
            id_ = ""
        combo.set_active_id(id_)

    # Model updaters:

    @gui.mvp.model_updater
    def _add_button_clicked_cb(self, button):
        doc = self._docmodel
        name = self._lvm.current_view_name
        if name is None:
            name = lib.layervis.NEW_VIEW_IDENT
        cmd = lib.layervis.AddLayerView(doc, name=name)
        doc.do(cmd)

    @gui.mvp.model_updater
    def _remove_button_clicked_cb(self, button):
        doc = self._docmodel
        cmd = lib.layervis.RemoveActiveLayerView(doc)
        doc.do(cmd)

    @gui.mvp.model_updater
    def _rename_button_clicked_cb(self, button):
        old_name = self._lvm.current_view_name
        if old_name is None:
            return
        new_name = gui.dialogs.ask_for_name(
            title=C_(
                "view controls: rename view dialog",
                u"Rename View",
            ),
            widget=self._app.drawWindow,
            default=old_name,
        )
        if new_name is None:
            return
        new_name = new_name.strip()
        if new_name == old_name:
            return
        doc = self._docmodel
        cmd = lib.layervis.RenameActiveLayerView(doc, new_name)
        doc.do(cmd)

    @gui.mvp.model_updater
    def _locked_button_toggled_cb(self, button):
        if self._lvm.current_view_name is None:
            return
        new_locked = button.get_active()
        doc = self._docmodel
        cmd = lib.layervis.SetActiveLayerViewLocked(doc, locked=new_locked)
        doc.do(cmd)

    @gui.mvp.model_updater
    def _current_view_combo_changed_cb(self, combo):
        name = combo.get_active_id()
        if name is None:
            return  # specifically means "nothing active" in GTK
        if name == "":
            name = None
        if name == self._lvm.current_view_name:
            return  # nothing to do
        doc = self._docmodel
        cmd = lib.layervis.ActivateLayerView(doc, name)
        doc.do(cmd)
