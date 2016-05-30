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

_LAUNCH_SUCCESS_MSG = _(u"Launched {app_name} to edit layer “{layer_name}”")
_LAUNCH_FAILED_MSG = _(u"Error: failed to launch {app_name} to edit "
                       u"layer “{layer_name}”")
_LAUNCH_CANCELLED_MSG = _(u"Editing cancelled. You can still edit "
                          u"“{layer_name}” from the Layers menu.")
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

class OpenWithDialog (Gtk.Dialog):
    """Choose an app from those recommended for a type"""

    ICON_SIZE = Gtk.IconSize.DIALOG
    SPECIFIC_FILE_MSG = _(
        u"MyPaint needs to edit a file of type \u201c{type_name}\u201d "
        u"({content_type}). What application should it use?"
        )
    GENERIC_MSG = _(
        u"What application should MyPaint use for editing files of "
        u"type \u201c{type_name}\u201d ({content_type})?"
        )

    def __init__(self, content_type, specific_file=False):
        Gtk.Dialog.__init__(self)
        self.set_title(_("Open With..."))
        self.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
        self.add_button(Gtk.STOCK_OK, Gtk.ResponseType.OK)
        self.set_default_response(Gtk.ResponseType.CANCEL)
        self.set_response_sensitive(Gtk.ResponseType.OK, False)
        self.connect("show", self._show_cb)

        content_box = self.get_content_area()
        content_box.set_border_width(12)
        content_box.set_spacing(12)

        msg_template = self.GENERIC_MSG
        if specific_file:
            msg_template = self.SPECIFIC_FILE_MSG
        msg_text = msg_template.format(
            content_type=content_type,
            type_name=Gio.content_type_get_description(content_type),
            )
        msg_label = Gtk.Label(label=msg_text)
        msg_label.set_single_line_mode(False)
        msg_label.set_line_wrap(True)
        msg_label.set_alignment(0.0, 0.5)
        content_box.pack_start(msg_label, False, False, 0)

        default_app = Gio.AppInfo.get_default_for_type(content_type, False)
        default_iter = None
        app_list_store = Gtk.ListStore(object)
        apps = Gio.AppInfo.get_all_for_type(content_type)
        for app in apps:
            if not app.should_show():
                continue
            row_iter = app_list_store.append([app])
            if default_iter is not None:
                continue
            if default_app and Gio.AppInfo.equal(app, default_app):
                default_iter = row_iter

        # TreeView to show available apps for this content type
        view = Gtk.TreeView()
        view.set_model(app_list_store)
        view.set_headers_clickable(False)
        view.set_headers_visible(False)
        view.connect("row-activated", self._row_activated_cb)

        view_scrolls = Gtk.ScrolledWindow()
        view_scrolls.set_shadow_type(Gtk.ShadowType.IN)
        view_scrolls.add(view)
        view_scrolls.set_size_request(375, 225)
        content_box.pack_start(view_scrolls, True, True, 0)

        # Column 0: application icon
        cell = Gtk.CellRendererPixbuf()
        col = Gtk.TreeViewColumn(_("Icon"), cell)
        col.set_cell_data_func(cell, self._app_icon_datafunc)
        icon_size_ok, icon_w, icon_h = Gtk.icon_size_lookup(self.ICON_SIZE)
        if icon_size_ok:
            col.set_min_width(icon_w)
        col.set_expand(False)
        col.set_resizable(False)
        view.append_column(col)

        # Column 1: application name
        cell = Gtk.CellRendererText()
        cell.set_property("ellipsize", Pango.EllipsizeMode.END)
        cell.set_property("editable", False)
        col = Gtk.TreeViewColumn(_("Name"), cell)
        col.set_cell_data_func(cell, self._app_name_datafunc)
        col.set_expand(True)
        col.set_min_width(150)
        view.append_column(col)

        # Selection: mode and initial value
        selection = view.get_selection()
        selection.set_mode(Gtk.SelectionMode.SINGLE)
        if default_iter:
            selection.select_iter(default_iter)
            self.set_default_response(Gtk.ResponseType.OK)
            self.set_response_sensitive(Gtk.ResponseType.OK, True)
        selection.connect("changed", self._selection_changed_cb)

        # Results go here
        self.selected_appinfo = default_app   #: The app the user chose

    def _show_cb(self, dialog):
        content_box = self.get_content_area()
        content_box.show_all()

    def _app_name_datafunc(self, col, cell, model, it, data):
        app = model.get_value(it, 0)
        name = app.get_display_name()
        desc = app.get_description()
        if desc is not None:
            markup_template = "<b>{name}</b>\n{description}"
        else:
            markup_template = "<b>{name}</b>\n<i>({description})</i>"
            desc = _("no description")
        markup = markup_template.format(
            name=lib.xml.escape(name),
            description=lib.xml.escape(desc),
            )
        cell.set_property("markup", markup)

    def _app_icon_datafunc(self, col, cell, model, it, data):
        app = model.get_value(it, 0)
        icon = app.get_icon()
        cell.set_property("gicon", icon)
        cell.set_property("stock-size", self.ICON_SIZE)

    def _row_activated_cb(self, view, treepath, column):
        model = view.get_model()
        treeiter = model.get_iter(treepath)
        if treeiter:
            appinfo = model.get_value(treeiter, 0)
            self.selected_appinfo = appinfo
            self.response(Gtk.ResponseType.OK)

    def _selection_changed_cb(self, selection):
        model, selected_iter = selection.get_selected()
        if selected_iter:
            appinfo = model.get_value(selected_iter, 0)
            self.selected_appinfo = appinfo
            self.set_response_sensitive(Gtk.ResponseType.OK, True)
            self.set_default_response(Gtk.ResponseType.OK)
        else:
            self.selected_appinfo = None
            self.set_response_sensitive(Gtk.ResponseType.OK, False)
            self.set_default_response(Gtk.ResponseType.CANCEL)


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
                app_name = "(unknown Win32 app)",  # FIXME: needs i18n
                layer_name = layer.name,
            ))

    def _begin_file_edit_using_gio(self, file_path, layer):
        logger.info("Using OpenWithDialog and GIO to open %r", file_path)
        logger.debug("Querying file path for info")
        file = Gio.File.new_for_path(file_path)
        flags = Gio.FileQueryInfoFlags.NONE
        attr = Gio.FILE_ATTRIBUTE_STANDARD_FAST_CONTENT_TYPE
        file_info = file.query_info(attr, flags, None)
        file_type = file_info.get_attribute_string(attr)

        logger.debug("Creating and launching external layer edit dialog")
        dialog = OpenWithDialog(file_type, specific_file=True)
        dialog.set_modal(True)
        dialog.set_transient_for(self._doc.app.drawWindow)
        dialog.set_position(Gtk.WindowPosition.CENTER)
        response = dialog.run()
        dialog.destroy()
        if response != Gtk.ResponseType.OK:
            self._doc.app.show_transient_message(
                _LAUNCH_CANCELLED_MSG.format(
                    layer_name=layer.name,
                ))
            return
        appinfo = dialog.selected_appinfo
        assert appinfo is not None

        disp = self._doc.tdw.get_display()
        launch_ctx = disp.get_app_launch_context()

        logger.debug(
            "Launching %r with %r (user-chosen app for %r)",
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    dialog = OpenWithDialog("image/svg+xml")
    #dialog = OpenWithDialog("text/plain")
    #dialog = OpenWithDialog("image/jpeg")
    #dialog = OpenWithDialog("application/xml")
    response = dialog.run()
    if response == Gtk.ResponseType.OK:
        app_name = dialog.selected_appinfo.get_name()
        logger.debug("AppInfo chosen: %r", app_name)
    else:
        logger.debug("Dialog was cancelled")
