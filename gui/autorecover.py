# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Autorecovery UI"""

import weakref
import os.path
from datetime import datetime
from gettext import gettext as _
import logging
logger = logging.getLogger(__name__)

from gi.repository import Gtk
from gi.repository import GLib

import lib.document
import lib.helpers
import lib.errors


class Presenter (object):
    """Shows and runs a dialog, allowing the user to resume autosaves.

    See also: lib.document.Document.resume_from_autosave().

    """

    _THUMBNAIL_SIZE = 128

    _RESPONSE_CANCEL = 0
    _RESPONSE_CONTINUE = 1

    _LISTSTORE_THUMBNAIL_COLUMN = 0
    _LISTSTORE_DESCRIPTION_COLUMN = 1
    _LISTSTORE_PATH_COLUMN = 2

    def __init__(self, app, **kwargs):
        """Initialize for the main app and its working doc."""
        super(Presenter, self).__init__(**kwargs)
        self._app = weakref.proxy(app)
        builder_xml = os.path.splitext(__file__)[0] + ".glade"
        builder = Gtk.Builder()
        builder.set_translation_domain("mypaint")
        builder.add_from_file(builder_xml)
        builder.connect_signals(self)
        self._dialog = builder.get_object("recovery_dialog")
        self._treeview = builder.get_object("recovery_treeview")
        self._liststore = builder.get_object("recovery_liststore")
        self._recover_button = builder.get_object("recover_autosave_button")

    def run(self, no_autosaves_dialog=False):
        """Show and run the dialog, and possibly resume an autosave.

        :param bool no_autosaves_dialog: Show an error if no backups.
        """
        # Load the available autosaves
        self._liststore.clear()
        doc = self._app.doc
        autosaves = []
        for asav in lib.document.get_available_autosaves():
            # Another instance may be working in there.
            if asav.cache_in_use:
                continue
            # Nor autosaves inside the current doc's cache folder.
            asav_cachedir = os.path.dirname(asav.path)
            if os.path.samefile(doc.model.cache_dir, asav_cachedir):
                continue
            autosaves.append(asav)
        if not autosaves:
            if no_autosaves_dialog:
                cache_root = lib.document.get_app_cache_root()
                self._app.message_dialog(
                    _(u"No backups were found in the cache."),
                    title = _(u"No Available Backups"),
                    type = Gtk.MessageType.ERROR,
                    investigate_dir = cache_root,
                    investigate_str = _(u"Open the Cache Folder…")
                )
            return
        s = self._THUMBNAIL_SIZE
        if len(autosaves) >= 3:
            s /= 2
        autosaves.sort(key=(lambda i: i.last_modified))
        for autosave in reversed(autosaves):
            thumb = autosave.thumbnail
            if thumb is None:
                flags = 0
                icon_theme = Gtk.IconTheme.get_default()
                thumb = icon_theme.load_icon("image-missing", s, flags)
            if thumb is not None:
                thumb = lib.helpers.scale_proportionally(thumb, s, s)
                thumb = lib.helpers.pixbuf_thumbnail(thumb, s, s, alpha=True)
            desc = autosave.get_description()
            assert isinstance(desc, unicode)
            assert isinstance(autosave.path, unicode)
            self._liststore.append((thumb, desc, autosave.path))
        # Get the user to pick an autosave to recover
        autosave = None
        error = None
        try:
            self._dialog.set_transient_for(self._app.drawWindow)
            self._dialog.show_all()
            result = self._dialog.run()
            if result == self._RESPONSE_CONTINUE:
                sel = self._treeview.get_selection()
                model, iter = sel.get_selected()
                path = model.get_value(iter, self._LISTSTORE_PATH_COLUMN)
                path = path.decode("utf-8")
                autosave = lib.document.AutosaveInfo.new_for_path(path)
                logger.info("Recovering %r...", autosave)
                try:
                    doc.model.resume_from_autosave(path)
                except lib.errors.FileHandlingError as e:
                    error = e
                else:
                    doc.reset_view(True, True, True)
        finally:
            self._dialog.set_transient_for(None)
            self._dialog.hide()
        # If an error was detected, tell the user about it.
        # They'll be given a new working doc & cache automatically.
        if error:
            self._app.message_dialog(
                unicode(error),
                title = _(u"Backup Recovery Failed"),
                type = Gtk.MessageType.ERROR,
                investigate_dir = error.investigate_dir,
                investigate_str = _(u"Open the Backup’s Folder…")
            )
        # If it loaded OK, get the user to save the recovered file ASAP.
        elif autosave:
            fh = self._app.filehandler
            fh.set_filename(None)
            lastmod = autosave.last_modified
            strftime_tmpl = "%Y-%m-%d %H%M%S"
            sugg_name_tmpl = _(u"Recovered file from {iso_datetime}.ora")
            sugg_name = sugg_name_tmpl.format(
                iso_datetime = lastmod.strftime(strftime_tmpl),
            )
            fh.save_as_dialog(fh.save_file, suggested_filename=sugg_name)

    def _recovery_treeview_row_activated_cb(self, treeview, treepath, column):
        """When a row's double-clicked, resume work on that doc."""
        self._dialog.response(self._RESPONSE_CONTINUE)

    def _recovery_tree_selection_changed_cb(self, sel):
        """When a row's clicked, make the continue button clickable."""
        model, iter = sel.get_selected()
        path = model.get_value(iter, self._LISTSTORE_PATH_COLUMN)
        path = path.decode("utf-8")
        assert isinstance(path, unicode)
        autosave = lib.document.AutosaveInfo.new_for_path(path)
        sensitive = not autosave.cache_in_use
        self._recover_button.set_sensitive(sensitive)
