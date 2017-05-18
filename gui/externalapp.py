# -*- encoding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2014-2015 by the MyPaint Development Team
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""External application launching and monitoring"""


## Imports
from __future__ import division, print_function

import logging
logger = logging.getLogger(__name__)
import weakref
import os.path
import os

from lib.gettext import gettext as _
from lib.gettext import C_

from gi.repository import Gio
from gi.repository import Pango
from gi.repository import Gdk
from gi.repository import Gtk

import lib.xml


## UI string consts

_LAUNCH_SUCCESS_MSG = _(u"Launched application to edit layer “{layer_name}”")
_LAUNCH_FAILED_MSG = _(u"Error: failed to launch application to edit "
                       u"layer “{layer_name}”")
_LAYER_UPDATED_MSG = _(u"Updated layer “{layer_name}” with external edits")

# XXX: Stolen from gui.filehandling during string freeze for v1.2.0.
# TRANSLATORS: This is a pretty gross abuse of context, but the content
# TRANSLATORS: is hopefully sufficiently similar to excuse it.
_LAYER_UPDATE_FAILED_MSG = C_(
    "file handling: open failed (statusbar)",
    u"Could not load “{file_basename}”.",
)
# TODO: This string should be updated with better context & content after
# TODO: the release, and similar contexts added to the ones above too.


## Class definitions

class LayerEditManager (object):
    """Launch external apps to edit layers, monitoring file changes"""

    def __init__(self, doc):
        """Initialize, attached to a document controller

        :param gui.document.Document doc: Owning controller

        """
        super(LayerEditManager, self).__init__()
        self._doc = doc
        self._active_edits = []

    def begin(self, layer):
        """Begin editing a layer in an external application

        :param lib.layer.LayerBase layer: Layer to start editing

        This starts the edit procedure by launching a chosen
        application for a tempfile requested from the layer. The file is
        monitored for changes, which are loaded back into the associated
        layer automatically.

        Each invocation of this callback from ``EditLayerExternally``
        creates a new tempfile for editing the layer, and launches a new
        instance of the external app. Previous tempfiles are removed
        from monitoring in favour of the new one.

        """
        logger.info("Starting external edit for %r...", layer.name)
        try:
            new_edit_tempfile = layer.new_external_edit_tempfile
        except AttributeError:
            return
        file_path = new_edit_tempfile()
        if os.name == 'nt':
            self._begin_file_edit_using_startfile(file_path, layer)
            # Avoid segfault: https://github.com/mypaint/mypaint/issues/531
            # Upstream: https://bugzilla.gnome.org/show_bug.cgi?id=758248
        else:
            self._begin_file_edit_using_gio(file_path, layer)
        self._begin_file_monitoring_using_gio(file_path, layer)

    def _begin_file_edit_using_startfile(self, file_path, layer):
        logger.info("Using os.startfile() to edit %r", file_path)
        os.startfile(file_path, "edit")
        self._doc.app.show_transient_message(
            _LAUNCH_SUCCESS_MSG.format(
                layer_name = layer.name,
            ))

    def _begin_file_edit_using_gio(self, file_path, layer):
        import application
        file_url = "file://" + file_path
        app = application.get_app()
        logger.debug("Using show_uri_on_window to open %r", file_url)
        success = Gtk.show_uri_on_window(app.drawWindow, file_url, Gdk.CURRENT_TIME)
        if not success:
            self._doc.app.show_transient_message(
                _LAUNCH_FAILED_MSG.format(
                    layer_name=layer.name,
                ))
            logger.error(
                "Failed to launch application to edit %r",
                file_path,
                )
            return
        self._doc.app.show_transient_message(
            _LAUNCH_SUCCESS_MSG.format(
                layer_name=layer.name,
            ))

    def _begin_file_monitoring_using_gio(self, file_path, layer):
        self._cleanup_stale_monitors(added_layer=layer)
        logger.debug("Begin monitoring %r for changes (layer=%r)",
                     file_path, layer)
        file = Gio.File.new_for_path(file_path)
        file_mon = file.monitor_file(Gio.FileMonitorFlags.NONE, None)
        file_mon.connect("changed", self._file_changed_cb)
        edit_info = (file_mon, weakref.ref(layer), file, file_path)
        self._active_edits.append(edit_info)

    def commit(self, layer):
        """Commit a layer's ongoing external edit"""
        logger.debug("Commit %r's current tempfile", layer)
        self._cleanup_stale_monitors()
        for mon, layer_ref, file, file_path in self._active_edits:
            if layer_ref() is not layer:
                continue
            model = self._doc.model
            file_basename = os.path.basename(file_path)
            try:
                model.update_layer_from_external_edit_tempfile(
                    layer,
                    file_path,
                )
            except Exception as ex:
                logger.error(
                    "Loading tempfile for %r (%r) failed: %r",
                    layer,
                    file_path,
                    str(ex),
                )
                status_msg = _LAYER_UPDATE_FAILED_MSG.format(
                    file_basename = file_basename,
                    layer_name = layer.name,
                )
            else:
                status_msg = _LAYER_UPDATED_MSG.format(
                    file_basename = file_basename,
                    layer_name = layer.name,
                )
            self._doc.app.show_transient_message(status_msg)
            return

    def _file_changed_cb(self, mon, file1, file2, event_type):
        self._cleanup_stale_monitors()
        if event_type == Gio.FileMonitorEvent.DELETED:
            logger.debug("File %r was deleted", file1.get_path())
            self._cleanup_stale_monitors(deleted_file=file1)
            return
        if event_type == Gio.FileMonitorEvent.CHANGES_DONE_HINT:
            logger.debug("File %r was changed", file1.get_path())
            for a_mon, layer_ref, file, file_path in self._active_edits:
                if a_mon is mon:
                    layer = layer_ref()
                    self.commit(layer)
                    return

    def _cleanup_stale_monitors(self, added_layer=None, deleted_file=None):
        for i in reversed(range(len(self._active_edits))):
            mon, layer_ref, file, file_path = self._active_edits[i]
            layer = layer_ref()
            stale = False
            if layer is None:
                logger.info("Removing monitor for garbage-collected layer")
                stale = True
            elif layer is added_layer:
                logger.info("Replacing monitor for already-tracked layer")
                stale = True
            if file is deleted_file:
                logger.info("Removing monitor for deleted file")
                stale = True
            if stale:
                mon.cancel()
                logger.info("File %r is no longer monitored", file.get_path())
                self._active_edits[i:i+1] = []

