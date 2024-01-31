# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2015-2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Autorecovery UI"""

from __future__ import division, print_function
import weakref
import os.path
from gettext import gettext as _
import shutil
import logging

from lib.gibindings import Gtk

import lib.document
import lib.helpers
import lib.errors
from lib.pycompat import unicode

logger = logging.getLogger(__name__)


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

    _CHECK_AT_STARTUP_PREFS_KEY = "autorecovery.check_at_startup"

    def __init__(self, app, **kwargs):
        """Initialize for the main app and its working doc."""
        super(Presenter, self).__init__(**kwargs)
        self._app = weakref.proxy(app)
        builder_xml = os.path.splitext(__file__)[0] + ".glade"
        builder = Gtk.Builder()
        builder.set_translation_domain("mypaint")
        builder.add_from_file(builder_xml)
        self._dialog = builder.get_object("recovery_dialog")
        self._treeview = builder.get_object("recovery_treeview")
        self._liststore = builder.get_object("recovery_liststore")
        self._recover_button = builder.get_object("recover_autosave_button")
        self._delete_button = builder.get_object("delete_autosave_button")
        at_start_checkbutton = builder.get_object("at_startup_checkbutton")
        at_start_checkbutton.set_active(self.check_at_startup)
        builder.connect_signals(self)

    @property
    def check_at_startup(self):
        prefs = self._app.preferences
        return bool(prefs.get(self._CHECK_AT_STARTUP_PREFS_KEY, True))

    @check_at_startup.setter
    def check_at_startup(self, value):
        prefs = self._app.preferences
        prefs[self._CHECK_AT_STARTUP_PREFS_KEY] = bool(value)

    def _reload_liststore(self):
        """Load the available autosaves"""
        self._liststore.clear()
        doc = self._app.doc
        autosaves = []
        for asav in lib.document.get_available_autosaves():
            # Another instance may be working in there.
            if asav.cache_in_use:
                logger.debug(
                    "Ignoring %r: in use",
                    asav.path,
                )
                continue
            # Skip autosaves inside the current doc's cache folder.
            asav_cachedir = os.path.dirname(asav.path)
            asav_cachedir = os.path.normpath(os.path.realpath(asav_cachedir))
            doc_cachedir = doc.model.cache_dir
            doc_cachedir = os.path.normpath(os.path.realpath(doc_cachedir))
            if doc_cachedir == asav_cachedir:
                logger.debug(
                    "Ignoring %r: belongs to current working doc.",
                    asav.path,
                )
                continue
            logger.debug(
                "Making %r available for autosave recovery.",
                asav.path,
            )
            autosaves.append(asav)
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
        return autosaves

    def run(self, startup=False):
        """Show and run the dialog, and possibly resume an autosave.

        :param bool startup: indicates that MyPaint is starting up.

        """
        # Don't run at startup if asked not to.
        if startup:
            if not self.check_at_startup:
                return
        # Only run if there are autosaves which can be recovered.
        autosaves = self._reload_liststore()
        if not autosaves:
            if not startup:
                cache_root = lib.document.get_app_cache_root()
                self._app.message_dialog(
                    _(u"No backups were found in the cache."),
                    title = _(u"No Available Backups"),
                    investigate_dir = cache_root,
                    investigate_str = _(u"Open the Cache Folder…"),
                    message_type = Gtk.MessageType.ERROR,
                )
            return
        doc = self._app.doc
        # Get the user to pick an autosave to recover
        autosave = None
        error = None
        try:
            self._dialog.set_transient_for(self._app.drawWindow)
            self._dialog.show_all()
            result = self._dialog.run()
            if result == self._RESPONSE_CONTINUE:
                autosave = self._get_selected_autosave()
                if autosave is not None:
                    logger.info("Recovering %r...", autosave)
                    try:
                        doc.model.resume_from_autosave(autosave.path)
                    except lib.errors.FileHandlingError as e:
                        error = e
                    else:
                        doc.reset_view(True, True, True)
        finally:
            self._dialog.hide()
        # If an error was detected, tell the user about it.
        # They'll be given a new working doc & cache automatically.
        if error:
            self._app.message_dialog(
                unicode(error),
                title = _(u"Backup Recovery Failed"),
                investigate_dir = error.investigate_dir,
                investigate_str = _(u"Open the Backup’s Folder…"),
                message_type = Gtk.MessageType.ERROR,
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
        """When a row's clicked, update button sensitivities etc."""
        self._update_buttons()

    def _update_buttons(self):
        autosave = self._get_selected_autosave()
        sensitive = False
        if autosave is not None:
            sensitive = not autosave.cache_in_use
        self._recover_button.set_sensitive(sensitive)
        self._delete_button.set_sensitive(sensitive)

    def _get_selected_autosave(self):
        sel = self._treeview.get_selection()
        model, iter = sel.get_selected()
        if iter is None:
            return None
        path = model.get_value(iter, self._LISTSTORE_PATH_COLUMN)
        if not isinstance(path, unicode):
            path = path.decode("utf-8")
        assert isinstance(path, unicode)
        if not os.path.isdir(path):
            return None
        return lib.document.AutosaveInfo.new_for_path(path)

    def _delete_autosave_button_clicked_cb(self, button):
        autosave = self._get_selected_autosave()
        if autosave is None or autosave.cache_in_use \
                or not os.path.isdir(autosave.path):
            return
        logger.info("Recursively deleting %r...", autosave.path)
        try:
            shutil.rmtree(autosave.path, ignore_errors=True)
        except Exception:
            logger.exception("Deleting %r failed.", autosave.path)
        else:
            logger.info("Deleted %r successfully.", autosave.path)
        self._reload_liststore()

    def _at_startup_checkbutton_toggled_cb(self, toggle):
        self.check_at_startup = bool(toggle.get_active())
