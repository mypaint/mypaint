# -*- encoding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""External application launching and monitoring"""


## Imports

from logging import getLogger
logger = getLogger(__name__)
import weakref

from gettext import gettext as _

from gi.repository import Gio


## UI string consts

_LAUNCH_SUCCESS_MSG = _(u"Launched {app_name} to edit layer “{layer_name}”")
_LAUNCH_FAILED_MSG = _(u"Error: failed to launch {app_name} to edit "
                       u"layer “{layer_name}”")
_LAYER_UPDATED_MSG = _(u"Updated layer “{layer_name}” with external edits")


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

        This starts the edit procedure by launching the default
        application for a tempfile requested from the layer. The file is
        monitored for changes, which are loaded back into the associated
        layer automatically.

        Each invocation of this callback from ``EditLayerExternally``
        creates a new tempfile for editing the layer, and launches a new
        instance of the external app. Previous tempfiles are removed
        from monitoring in favour of the new one.

        """

        try:
            new_edit_tempfile = layer.new_external_edit_tempfile
        except AttributeError:
            return
        file_path = new_edit_tempfile()
        file = Gio.File.new_for_path(file_path)
        flags = Gio.FileQueryInfoFlags.NONE
        attr = Gio.FILE_ATTRIBUTE_STANDARD_FAST_CONTENT_TYPE
        file_info = file.query_info(attr, flags, None)
        file_type = file_info.get_attribute_string(attr)
        appinfo = Gio.AppInfo.get_default_for_type(file_type, False)
        if not appinfo:
            logger.error("No default app registered for %r", file_type)
            return

        disp = self._doc.tdw.get_display()
        launch_ctx = disp.get_app_launch_context()

        if not appinfo.supports_files():
            logger.error(
                "The default handler for %r, %r, only supports "
                "opening files by URI",
                appinfo.get_name(),
                file_path,
                )
            return

        logger.debug(
            "Launching %r with %r (default app for %r)",
            appinfo.get_name(),
            file_path,
            file_type,
            )
        launched_app = appinfo.launch([file], launch_ctx)
        if not launched_app:
            self._doc.app.show_transient_message(
                _LAUNCH_FAILED_MSG.format(
                    app_name=appinfo.get_name(),
                    layer_name=layer.name,
                ))
            logger.error(
                "Failed to launch %r with %r",
                appinfo.get_name(),
                file_path,
                )
            return
        self._doc.app.show_transient_message(
            _LAUNCH_SUCCESS_MSG.format(
                app_name=appinfo.get_name(),
                layer_name=layer.name,
            ))
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
        logger.debug("Commit %r's current tempfile",
                     layer)
        self._cleanup_stale_monitors()
        for mon, layer_ref, file, file_path in self._active_edits:
            if layer_ref() is not layer:
                continue
            model = self._doc.model
            self._doc.app.show_transient_message(
                _LAYER_UPDATED_MSG.format(
                    layer_name=layer.name,
                ))
            model.update_layer_from_external_edit_tempfile(layer, file_path)
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
