# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2007-2014 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2009-2015 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os
import re
from glob import glob
import sys
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict

import glib
import gtk

from lib import document, helpers, tiledsurface
from lib import fileutils
from lib.errors import FileHandlingError
from lib.errors import AllocationError
import drawwindow
from lib import mypaintlib
from lib.gettext import gettext as _
from lib.gettext import ngettext
from lib.gettext import C_
import lib.glib

SAVE_FORMAT_ANY = 0
SAVE_FORMAT_ORA = 1
SAVE_FORMAT_PNGSOLID = 2
SAVE_FORMAT_PNGTRANS = 3
SAVE_FORMAT_PNGMULTI = 4
SAVE_FORMAT_JPEG = 5
SAVE_FORMAT_PNGAUTO = 6


# Utility function to work around the fact that gtk FileChooser/FileFilter
# does not have an easy way to use case insensitive filters

def get_case_insensitive_glob(string):
    '''Ex: '*.ora' => '*.[oO][rR][aA]' '''
    ext = string.split('.')[1]
    globlist = ["[%s%s]" % (c.lower(), c.upper()) for c in ext]
    return '*.%s' % ''.join(globlist)


def add_filters_to_dialog(filters, dialog):
    for name, patterns in filters:
        f = gtk.FileFilter()
        f.set_name(name)
        for p in patterns:
            f.add_pattern(get_case_insensitive_glob(p))
        dialog.add_filter(f)


def dialog_set_filename(dialog, s):
    # According to pygtk docu we should use set_filename(),
    # however doing so removes the selected filefilter.
    path, name = os.path.split(s)
    dialog.set_current_folder(path)
    dialog.set_current_name(name)


class FileHandler(object):
    def __init__(self, app):
        self.app = app
        #NOTE: filehandling and drawwindow are very tightly coupled
        self.save_dialog = None

        # File filters definitions, for dialogs
        self.file_filters = [
            # (name, patterns)
            (_("All Recognized Formats"), ("*.ora", "*.png", "*.jpg", "*.jpeg")),
            (_("OpenRaster (*.ora)"), ("*.ora",)),
            (_("PNG (*.png)"), ("*.png",)),
            (_("JPEG (*.jpg; *.jpeg)"), ("*.jpg", "*.jpeg")),
        ]

        # Recent filter, for the menu.
        # Better to use a regex with re.IGNORECASE than
        # .upper()==.upper() hacks since internally, filenames are
        # Unicode and capitalization rules like Turkish's dotless "i"
        # exist. One day we want all the formats GdkPixbuf can load to
        # be supported in the dialog.

        file_regex_exts = set()
        for name, patts in self.file_filters:
            for p in patts:
                e = p.replace("*.", "", 1)
                file_regex_exts.add(re.escape(e))
        file_re = r'[.](?:' + ('|'.join(file_regex_exts)) + r')$'
        logger.debug("Using regex /%s/i for filtering recent files", file_re)
        self._file_extension_regex = re.compile(file_re, re.IGNORECASE)
        rf = gtk.RecentFilter()
        rf.add_pattern('')
        # The blank-string pattern is eeded so the custom func will
        # get URIs at all, despite the needed flags below.
        rf.add_custom(
            func = self._recentfilter_func,
            needed = (
                gtk.RecentFilterFlags.APPLICATION |
                gtk.RecentFilterFlags.URI
            )
        )
        ra = app.find_action("OpenRecent")
        ra.add_filter(rf)

        ag = app.builder.get_object('FileActions')
        for action in ag.list_actions():
            self.app.kbm.takeover_action(action)

        self._filename = None
        self.current_file_observers = []
        self.file_opened_observers = []
        self.active_scrap_filename = None
        self.lastsavefailed = False
        self._update_recent_items()

        saveformat_keys = [
            SAVE_FORMAT_ANY,
            SAVE_FORMAT_ORA,
            SAVE_FORMAT_PNGSOLID,
            SAVE_FORMAT_PNGTRANS,
            SAVE_FORMAT_PNGMULTI,
            SAVE_FORMAT_JPEG,
        ]
        saveformat_values = [
            # (name, extension, options)
            (_("By extension (prefer default format)"), None, {}),
            (_("OpenRaster (*.ora)"), '.ora', {}),
            (_("PNG solid with background (*.png)"), '.png', {'alpha': False}),
            (_("PNG transparent (*.png)"), '.png', {'alpha': True}),
            (_("Multiple PNG transparent (*.XXX.png)"), '.png', {'multifile': True}),
            (_("JPEG 90% quality (*.jpg; *.jpeg)"), '.jpg', {'quality': 90}),
        ]
        self.saveformats = OrderedDict(zip(saveformat_keys, saveformat_values))
        self.ext2saveformat = {
            ".ora": (SAVE_FORMAT_ORA, "image/openraster"),
            ".png": (SAVE_FORMAT_PNGAUTO, "image/png"),
            ".jpeg": (SAVE_FORMAT_JPEG, "image/jpeg"),
            ".jpg": (SAVE_FORMAT_JPEG, "image/jpeg"),
        }
        self.config2saveformat = {
            'openraster': SAVE_FORMAT_ORA,
            'jpeg-90%': SAVE_FORMAT_JPEG,
            'png-solid': SAVE_FORMAT_PNGSOLID,
        }

        self.__statusbar_context_id = None

    @property
    def _statusbar_context_id(self):
        cid = self.__statusbar_context_id
        if not cid:
            cid = self.app.statusbar.get_context_id("filehandling-message")
            self.__statusbar_context_id = cid
        return cid

    def _update_recent_items(self):
        """Updates self._recent_items from the GTK RecentManager.

        This list is consumed in open_last_cb.

        """
        # Note: i.exists() does not work on Windows if the pathname
        # contains utf-8 characters. Since GIMP also saves its URIs
        # with utf-8 characters into this list, I assume this is a
        # gtk bug.  So we use our own test instead of i.exists().

        recent_items = []
        rm = gtk.RecentManager.get_default()
        for i in rm.get_items():
            if not i:
                continue
            apps = i.get_applications()
            if not (apps and "mypaint" in apps):
                continue
            if self._uri_is_loadable(i.get_uri()):
                recent_items.append(i)
        # This test should be kept in sync with _recentfilter_func.
        recent_items.reverse()
        self._recent_items = recent_items

    def get_filename(self):
        return self._filename

    def set_filename(self, value):
        self._filename = value
        for f in self.current_file_observers:
            f(self.filename)

        if self.filename:
            if self.filename.startswith(self.get_scrap_prefix()):
                self.active_scrap_filename = self.filename

    filename = property(get_filename, set_filename)

    def init_save_dialog(self):
        dialog = gtk.FileChooserDialog(_("Save..."), self.app.drawWindow,
                                       gtk.FILE_CHOOSER_ACTION_SAVE,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_SAVE, gtk.RESPONSE_OK))
        self.save_dialog = dialog
        dialog.set_default_response(gtk.RESPONSE_OK)
        dialog.set_do_overwrite_confirmation(True)
        add_filters_to_dialog(self.file_filters, dialog)

        # Add widget for selecting save format
        box = gtk.HBox()
        label = gtk.Label(_('Format to save as:'))
        label.set_alignment(0.0, 0.0)
        combo = self.saveformat_combo = gtk.ComboBoxText()
        for name, ext, opt in self.saveformats.itervalues():
            combo.append_text(name)
        combo.set_active(0)
        combo.connect('changed', self.selected_save_format_changed_cb)
        box.pack_start(label)
        box.pack_start(combo, expand=False)
        dialog.set_extra_widget(box)
        dialog.show_all()

    def selected_save_format_changed_cb(self, widget):
        """When the user changes the selected format to save as in the dialog,
        change the extension of the filename (if existing) immediately."""
        dialog = self.save_dialog
        filename = dialog.get_filename()
        if filename:
            filename = filename.decode('utf-8')
            filename, ext = os.path.splitext(filename)
            if ext:
                saveformat = self.saveformat_combo.get_active()
                ext = self.saveformats[saveformat][1]
                if ext is not None:
                    dialog_set_filename(dialog, filename+ext)

    def confirm_destructive_action(self, title=_('Confirm'),
                                   question=_('Really continue?')):
        """Asks the user to confirm an action that might lose work

        :param title: Dialog title
        :param question: Question to ask the user
        :rtype bool:
        :returns: true if the user chose to destroy their work

        This method doesn't bother asking if there are fewer than a handful of
        seconds of unsaved work. By default, that's 1 second, but the
        build-time and runtime debugging flags make this period longer to allow
        faster develop and test cycles.
        """
        self.doc.model.sync_pending_changes()
        t = self.doc.model.unsaved_painting_time

        t_bother = 1
        if mypaintlib.heavy_debug:
            t_bother += 7
        if os.environ.get("MYPAINT_DEBUG", False):
            t_bother += 7
        # This used to be 30, but see https://gna.org/bugs/?17955
        # Then 8 by default, but Twitter users hate that too.
        logger.debug("Destructive action don't-bother period is %ds", t_bother)
        if t < t_bother:
            return True

        #TRANSLATORS: Abbreviated string is an already translated time
        #TRANSLATORS: period abbreviation, like "6m42s", or "103s".
        unsaved_str = _(
            u"This will discard {abbreviated_time} of unsaved painting"
        ).format(
            abbreviated_time = helpers.fmt_time_period_abbr(t),
        )
        d = gtk.Dialog(title, self.app.drawWindow, gtk.DIALOG_MODAL)

        b = d.add_button(gtk.STOCK_DISCARD, gtk.RESPONSE_OK)
        b.set_image(gtk.image_new_from_stock(gtk.STOCK_DELETE, gtk.ICON_SIZE_BUTTON))
        d.add_button(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL)
        b = d.add_button(_("_Save as Scrap"), gtk.RESPONSE_APPLY)
        b.set_image(gtk.image_new_from_stock(gtk.STOCK_SAVE, gtk.ICON_SIZE_BUTTON))

        d.set_default_response(gtk.RESPONSE_CANCEL)
        l = gtk.Label()
        l.set_markup("<b>%s</b>\n\n%s" % (question, unsaved_str))
        l.set_padding(10, 10)
        l.show()
        d.vbox.pack_start(l)
        response = d.run()
        d.destroy()
        if response == gtk.RESPONSE_APPLY:
            self.save_scrap_cb(None)
            return True
        return response == gtk.RESPONSE_OK

    def new_cb(self, action):
        if not self.confirm_destructive_action():
            return
        self.doc.model.clear()
        self.filename = None
        self._update_recent_items()
        self.app.doc.reset_view(True, True, True)

    @staticmethod
    def gtk_main_tick():
        while gtk.events_pending():
            gtk.main_iteration()

    @drawwindow.with_wait_cursor
    def open_file(self, filename):
        prefs = self.app.preferences
        display_colorspace_setting = prefs["display.colorspace"]
        statusbar = self.app.statusbar
        statusbar_cid = self._statusbar_context_id
        statusbar.remove_all(statusbar_cid)
        file_basename = os.path.basename(filename)
        statusbar.push(statusbar_cid, C_(
            "file handling: open: during load (statusbar)",
            u"Loading “{file_basename}”…"
        ).format(
            file_basename = file_basename,
        ))
        try:
            self.doc.model.load(
                filename,
                feedback_cb=self.gtk_main_tick,
                convert_to_srgb=(display_colorspace_setting == "srgb"),
            )
        except (FileHandlingError, AllocationError, MemoryError) as e:
            statusbar.remove_all(statusbar_cid)
            self.app.show_transient_message(C_(
                "file handling: open failed (statusbar)",
                u"Could not load “{file_basename}”.",
            ).format(
                file_basename = file_basename,
            ))
            self.app.message_dialog(unicode(e), type=gtk.MESSAGE_ERROR)
        else:
            statusbar.remove_all(statusbar_cid)
            self.filename = os.path.abspath(filename)
            for func in self.file_opened_observers:
                func(self.filename)
            logger.info('Loaded from %r', self.filename)
            self.app.doc.reset_view(True, True, True)
            # try to restore the last used brush and color
            layers = self.doc.model.layer_stack
            search_layers = []
            if layers.current is not None:
                search_layers.append(layers.current)
            search_layers.extend(layers.deepiter())
            for layer in search_layers:
                si = layer.get_last_stroke_info()
                if si:
                    self.app.restore_brush_from_stroke_info(si)
                    break
            self.app.show_transient_message(C_(
                "file handling: open success (statusbar)",
                u"Loaded “{file_basename}”.",
            ).format(
                file_basename = file_basename,
            ))

    def open_scratchpad(self, filename):
        try:
            self.app.scratchpad_doc.model.load(filename, feedback_cb=self.gtk_main_tick)
            self.app.scratchpad_filename = os.path.abspath(filename)
            self.app.preferences["scratchpad.last_opened_scratchpad"] = self.app.scratchpad_filename
        except (FileHandlingError, AllocationError, MemoryError) as e:
            self.app.message_dialog(unicode(e), type=gtk.MESSAGE_ERROR)
        else:
            self.app.scratchpad_filename = os.path.abspath(filename)
            self.app.preferences["scratchpad.last_opened_scratchpad"] = self.app.scratchpad_filename
            logger.info('Loaded scratchpad from %r',
                        self.app.scratchpad_filename)
            self.app.scratchpad_doc.reset_view(True, True, True)

    @drawwindow.with_wait_cursor
    def save_file(self, filename, export=False, **options):
        """Saves the main document to one or more files (app/toplevel)

        :param filename: The base filename to save
        :param bool export: True if exporting
        :param **options: Pass-through options

        This method invokes `_save_doc_to_file()` with the main working
        doc, but also attempts to save thumbnails and perform recent
        files list management, when appropriate.

        See `_save_doc_to_file()`
        """
        thumbnail_pixbuf = self._save_doc_to_file(
            filename,
            self.doc,
            export=export,
            statusmsg=True,
            **options
        )
        if "multifile" in options:  # thumbs & recents are inappropriate
            return
        if not os.path.isfile(filename):  # failed to save
            return
        if not export:
            self.filename = os.path.abspath(filename)
            basename, ext = os.path.splitext(self.filename)
            recent_mgr = gtk.RecentManager.get_default()
            uri = lib.glib.filename_to_uri(self.filename)
            recent_data = gtk.RecentData()
            recent_data.app_name = "mypaint"
            recent_data.app_exec = sys.argv_unicode[0].encode("utf-8")
            mime_default = "application/octet-stream"
            fmt, mime_type = self.ext2saveformat.get(ext, (None, mime_default))
            recent_data.mime_type = mime_type
            recent_mgr.add_full(uri, recent_data)
        if not thumbnail_pixbuf:
            options["render_background"] = not options.get("alpha", False)
            thumbnail_pixbuf = self.doc.model.render_thumbnail(**options)
        helpers.freedesktop_thumbnail(filename, thumbnail_pixbuf)

    @drawwindow.with_wait_cursor
    def save_scratchpad(self, filename, export=False, **options):
        if self.app.scratchpad_doc.model.unsaved_painting_time or export or not os.path.exists(filename):
            self._save_doc_to_file(
                filename,
                self.app.scratchpad_doc,
                export=export,
                statusmsg=False,
                **options
            )
        if not export:
            self.app.scratchpad_filename = os.path.abspath(filename)
            self.app.preferences["scratchpad.last_opened_scratchpad"] = self.app.scratchpad_filename

    def _save_doc_to_file(self, filename, doc, export=False, statusmsg=True,
                          **options):
        """Saves a document to one or more files

        :param filename: The base filename to save
        :param gui.document.Document doc: Controller for the document to save
        :param bool export: True if exporting
        :param **options: Pass-through options

        This method handles logging, statusbar messages,
        and alerting the user to when the save failed.

        See also: `lib.document.Document.save()`.
        """
        thumbnail_pixbuf = None
        prefs = self.app.preferences
        display_colorspace_setting = prefs["display.colorspace"]
        options['save_srgb_chunks'] = (display_colorspace_setting == "srgb")
        if statusmsg:
            statusbar = self.app.statusbar
            statusbar_cid = self._statusbar_context_id
            statusbar.remove_all(statusbar_cid)
            file_basename = os.path.basename(filename)
            if export:
                during_tmpl = C_(
                    "file handling: during export (statusbar)",
                    u"Exporting to “{file_basename}”…"
                )
            else:
                during_tmpl = C_(
                    "file handling: during save (statusbar)",
                    u"Saving “{file_basename}”…"
                )
            statusbar.push(statusbar_cid, during_tmpl.format(
                file_basename = file_basename,
            ))
        try:
            x, y, w, h = doc.model.get_bbox()
            if w == 0 and h == 0:
                w, h = tiledsurface.N, tiledsurface.N
                # TODO: Add support for other sizes
            thumbnail_pixbuf = doc.model.save(
                filename,
                feedback_cb=self.gtk_main_tick,
                **options
            )
            self.lastsavefailed = False
        except (FileHandlingError, AllocationError, MemoryError) as e:
            if statusmsg:
                statusbar.remove_all(statusbar_cid)
                if export:
                    failed_tmpl = C_(
                        "file handling: export failure (statusbar)",
                        u"Failed to export to “{file_basename}”.",
                    )
                else:
                    failed_tmpl = C_(
                        "file handling: save failure (statusbar)",
                        u"Failed to save “{file_basename}”.",
                    )
                self.app.show_transient_message(failed_tmpl.format(
                    file_basename = file_basename,
                ))
            self.lastsavefailed = True
            self.app.message_dialog(unicode(e), type=gtk.MESSAGE_ERROR)
        else:
            if statusmsg:
                statusbar.remove_all(statusbar_cid)
            file_location = os.path.abspath(filename)
            multifile_info = ''
            if "multifile" in options:
                multifile_info = " (basis; used multiple .XXX.ext names)"
            if not export:
                logger.info('Saved to %r%s', file_location, multifile_info)
            else:
                logger.info('Exported to %r%s', file_location, multifile_info)
            if statusmsg:
                if export:
                    success_tmpl = C_(
                        "file handling: export success (statusbar)",
                        u"Exported to “{file_basename}” successfully.",
                    )
                else:
                    success_tmpl = C_(
                        "file handling: save success (statusbar)",
                        u"Saved “{file_basename}” successfully.",
                    )
                self.app.show_transient_message(success_tmpl.format(
                    file_basename = file_basename,
                ))
        return thumbnail_pixbuf

    def update_preview_cb(self, file_chooser, preview):
        filename = file_chooser.get_preview_filename()
        if filename:
            filename = filename.decode('utf-8')
            pixbuf = helpers.freedesktop_thumbnail(filename)
            if pixbuf:
                # if pixbuf is smaller than 256px in width, copy it onto a transparent 256x256 pixbuf
                pixbuf = helpers.pixbuf_thumbnail(pixbuf, 256, 256, True)
                preview.set_from_pixbuf(pixbuf)
                file_chooser.set_preview_widget_active(True)
            else:
                #TODO display "no preview available" image?
                file_chooser.set_preview_widget_active(False)

    def get_open_dialog(self, filename=None, start_in_folder=None, file_filters=[]):
        dialog = gtk.FileChooserDialog(_("Open..."), self.app.drawWindow,
                                       gtk.FILE_CHOOSER_ACTION_OPEN,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)
        add_filters_to_dialog(file_filters, dialog)

        if filename:
            dialog.set_filename(filename)
        elif start_in_folder and os.path.isdir(start_in_folder):
            dialog.set_current_folder(start_in_folder)

        return dialog

    def open_cb(self, action):
        if not self.confirm_destructive_action():
            return
        dialog = gtk.FileChooserDialog(_("Open..."), self.app.drawWindow,
                                       gtk.FILE_CHOOSER_ACTION_OPEN,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)

        preview = gtk.Image()
        dialog.set_preview_widget(preview)
        dialog.connect("update-preview", self.update_preview_cb, preview)

        add_filters_to_dialog(self.file_filters, dialog)

        if self.filename:
            dialog.set_filename(self.filename)
        else:
            # choose the most recent save folder
            self._update_recent_items()
            for item in reversed(self._recent_items):
                uri = item.get_uri()
                fn, _h = lib.glib.filename_from_uri(uri)
                dn = os.path.dirname(fn)
                if os.path.isdir(dn):
                    dialog.set_current_folder(dn)
                    break
        try:
            if dialog.run() == gtk.RESPONSE_OK:
                dialog.hide()
                self.open_file(dialog.get_filename().decode('utf-8'))
        finally:
            dialog.destroy()

    def open_scratchpad_dialog(self):
        dialog = gtk.FileChooserDialog(_("Open Scratchpad..."), self.app.drawWindow,
                                       gtk.FILE_CHOOSER_ACTION_OPEN,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)

        preview = gtk.Image()
        dialog.set_preview_widget(preview)
        dialog.connect("update-preview", self.update_preview_cb, preview)

        add_filters_to_dialog(self.file_filters, dialog)

        if self.app.scratchpad_filename:
            dialog.set_filename(self.app.scratchpad_filename)
        else:
            # choose the most recent save folder
            self._update_recent_items()
            for item in reversed(self._recent_items):
                uri = item.get_uri()
                fn, _h = lib.glib.filename_from_uri(uri)
                dn = os.path.dirname(fn)
                if os.path.isdir(dn):
                    dialog.set_current_folder(dn)
                    break
        try:
            if dialog.run() == gtk.RESPONSE_OK:
                dialog.hide()
                self.app.scratchpad_filename = dialog.get_filename().decode('utf-8')
                self.open_scratchpad(self.app.scratchpad_filename)
        finally:
            dialog.destroy()

    def save_cb(self, action):
        if not self.filename:
            self.save_as_cb(action)
        else:
            self.save_file(self.filename)

    def save_as_cb(self, action):
        if self.filename:
            current_filename = self.filename
        else:
            current_filename = ''
            # choose the most recent save folder
            self._update_recent_items()
            for item in reversed(self._recent_items):
                uri = item.get_uri()
                fn, _h = lib.glib.filename_from_uri(uri)
                dn = os.path.dirname(fn)
                if os.path.isdir(dn):
                    break

        if action.get_name() == 'Export':
            # Do not change working file
            self.save_as_dialog(self.save_file, suggested_filename=current_filename, export=True)
        else:
            self.save_as_dialog(self.save_file, suggested_filename=current_filename)

    def save_scratchpad_as_dialog(self, export=False):
        if self.app.scratchpad_filename:
            current_filename = self.app.scratchpad_filename
        else:
            current_filename = ''

        self.save_as_dialog(self.save_scratchpad, suggested_filename=current_filename, export=export)

    def save_as_dialog(self, save_method_reference, suggested_filename=None, start_in_folder=None, export=False, **options):
        if not self.save_dialog:
            self.init_save_dialog()
        dialog = self.save_dialog
        # Set the filename in the dialog
        if suggested_filename:
            dialog_set_filename(dialog, suggested_filename)
        else:
            dialog_set_filename(dialog, '')
            # Recent directory?
            if start_in_folder:
                dialog.set_current_folder(start_in_folder)

        try:
            # Loop until we have filename with an extension
            while dialog.run() == gtk.RESPONSE_OK:
                filename = dialog.get_filename()
                if filename is None:
                    continue
                filename = filename.decode('utf-8')
                name, ext = os.path.splitext(filename)
                saveformat = self.saveformat_combo.get_active()

                # If no explicitly selected format, use the extension to figure it out
                if saveformat == SAVE_FORMAT_ANY:
                    cfg = self.app.preferences['saving.default_format']
                    default_saveformat = self.config2saveformat[cfg]
                    if ext:
                        try:
                            saveformat, mime = self.ext2saveformat[ext]
                        except KeyError:
                            saveformat = default_saveformat
                    else:
                        saveformat = default_saveformat

                # if saveformat isn't a key, it must be SAVE_FORMAT_PNGAUTO.
                desc, ext_format, options = self.saveformats.get(saveformat,
                    ("", ext, {'alpha': None}))
                #
                if ext:
                    if ext_format != ext:
                        # Minor ugliness: if the user types '.png' but
                        # leaves the default .ora filter selected, we
                        # use the default options instead of those
                        # above. However, they are the same at the moment.
                        options = {}
                    assert(filename)
                    dialog.hide()
                    if export:
                        # Do not change working file
                        save_method_reference(filename, True, **options)
                    else:
                        save_method_reference(filename, **options)
                    break

                filename = name + ext_format

                # trigger overwrite confirmation for the modified filename
                dialog_set_filename(dialog, filename)
                dialog.response(gtk.RESPONSE_OK)

        finally:
            dialog.hide()
            dialog.destroy()  # avoid GTK crash: https://gna.org/bugs/?17902
            self.save_dialog = None

    def save_scrap_cb(self, action):
        filename = self.filename
        prefix = self.get_scrap_prefix()
        self.app.filename = self.save_autoincrement_file(filename, prefix, main_doc=True)

    def save_scratchpad_cb(self, action):
        filename = self.app.scratchpad_filename
        prefix = self.get_scratchpad_prefix()
        self.app.scratchpad_filename = self.save_autoincrement_file(filename, prefix, main_doc=False)

    def save_autoincrement_file(self, filename, prefix, main_doc=True):
        # If necessary, create the folder(s) the scraps are stored under
        prefix_dir = os.path.dirname(prefix)
        if not os.path.exists(prefix_dir):
            os.makedirs(prefix_dir)

        number = None
        if filename:
            junk, file_fragment = os.path.split(filename)
            if file_fragment.startswith("_md5"):
                #store direct, don't attempt to increment
                if main_doc:
                    self.save_file(filename)
                else:
                    self.save_scratchpad(filename)
                return filename

            l = re.findall(re.escape(prefix) + '([0-9]+)', filename)
            if l:
                number = l[0]

        if number:
            # reuse the number, find the next character
            char = 'a'
            for filename in glob(prefix + number + '_*'):
                c = filename[len(prefix + number + '_')]
                if c >= 'a' and c <= 'z' and c >= char:
                    char = chr(ord(c)+1)
            if char > 'z':
                # out of characters, increase the number
                filename = None
                return self.save_autoincrement_file(filename, prefix, main_doc)
            filename = '%s%s_%c' % (prefix, number, char)
        else:
            # we don't have a scrap filename yet, find the next number
            maximum = 0
            for filename in glob(prefix + '[0-9][0-9][0-9]*'):
                filename = filename[len(prefix):]
                res = re.findall(r'[0-9]*', filename)
                if not res:
                    continue
                number = int(res[0])
                if number > maximum:
                    maximum = number
            filename = '%s%03d_a' % (prefix, maximum+1)

        # Add extension
        cfg = self.app.preferences['saving.default_format']
        default_saveformat = self.config2saveformat[cfg]
        filename += self.saveformats[default_saveformat][1]

        assert not os.path.exists(filename)
        if main_doc:
            self.save_file(filename)
        else:
            self.save_scratchpad(filename)
        return filename

    def get_scrap_prefix(self):
        prefix = self.app.preferences['saving.scrap_prefix']
        # This should really use two separate settings, not one.
        # https://github.com/mypaint/mypaint/issues/375
        prefix = fileutils.expanduser_unicode(prefix)
        prefix = os.path.abspath(prefix)
        if os.path.isdir(prefix):
            if not prefix.endswith(os.path.sep):
                prefix += os.path.sep
        return prefix

    def get_scratchpad_prefix(self):
        # TODO allow override via prefs, maybe
        prefix = os.path.join(self.app.user_datapath, 'scratchpads')
        prefix = os.path.abspath(prefix)
        if os.path.isdir(prefix):
            if not prefix.endswith(os.path.sep):
                prefix += os.path.sep
        return prefix

    def get_scratchpad_default(self):
        # TODO get the default name from preferences
        prefix = self.get_scratchpad_prefix()
        return os.path.join(prefix, "scratchpad_default.ora")

    def get_scratchpad_autosave(self):
        prefix = self.get_scratchpad_prefix()
        return os.path.join(prefix, "autosave.ora")

    def list_scraps(self):
        prefix = self.get_scrap_prefix()
        return self.list_prefixed_dir(prefix)

    def list_scratchpads(self):
        prefix = self.get_scratchpad_prefix()
        files = self.list_prefixed_dir(prefix)
        if os.path.isdir(os.path.join(prefix, "special")):
            files += self.list_prefixed_dir(os.path.join(prefix, "special") + os.path.sep)
        return files

    def list_prefixed_dir(self, prefix):
        filenames = []
        for ext in ['png', 'ora', 'jpg', 'jpeg']:
            filenames += glob(prefix + '[0-9]*.' + ext)
            filenames += glob(prefix + '[0-9]*.' + ext.upper())
            # For the special linked scratchpads
            filenames += glob(prefix + '_md5[0-9a-f]*.' + ext)
        filenames.sort()
        return filenames

    def list_scraps_grouped(self):
        filenames = self.list_scraps()
        return self.list_files_grouped(filenames)

    def list_scratchpads_grouped(self):
        filenames = self.list_scratchpads()
        return self.list_files_grouped(filenames)

    def list_files_grouped(self, filenames):
        """return scraps grouped by their major number"""
        def scrap_id(filename):
            s = os.path.basename(filename)
            if s.startswith("_md5"):
                return s
            return re.findall('([0-9]+)', s)[0]
        groups = []
        while filenames:
            group = []
            sid = scrap_id(filenames[0])
            while filenames and scrap_id(filenames[0]) == sid:
                group.append(filenames.pop(0))
            groups.append(group)
        return groups

    def open_recent_cb(self, action):
        """Callback for RecentAction"""
        if not self.confirm_destructive_action():
            return
        uri = action.get_current_uri()
        fn, _h = lib.glib.filename_from_uri(uri)
        self.open_file(fn)

    def open_last_cb(self, action):
        """Callback to open the last file"""
        if not self._recent_items:
            return
        if not self.confirm_destructive_action():
            return
        uri = self._recent_items.pop().get_uri()
        fn, _h = lib.glib.filename_from_uri(uri)
        self.open_file(fn)

    def open_scrap_cb(self, action):
        groups = self.list_scraps_grouped()
        if not groups:
            msg = _('There are no scrap files named "%s" yet.') % \
                (self.get_scrap_prefix() + '[0-9]*')
            self.app.message_dialog(msg, gtk.MESSAGE_WARNING)
            return
        if not self.confirm_destructive_action():
            return
        next = action.get_name() == 'NextScrap'

        if next:
            idx = 0
        else:
            idx = -1
        for i, group in enumerate(groups):
            if self.active_scrap_filename in group:
                if next:
                    idx = i + 1
                else:
                    idx = i - 1
        filename = groups[idx % len(groups)][-1]
        self.open_file(filename)

    def reload_cb(self, action):
        if self.filename and self.confirm_destructive_action():
            self.open_file(self.filename)

    def delete_scratchpads(self, filenames):
        prefix = self.get_scratchpad_prefix()
        prefix = os.path.abspath(prefix)
        for filename in filenames:
            if os.path.isfile(filename) and os.path.abspath(filename).startswith(prefix):
                os.remove(filename)
                logger.info("Removed %s", filename)

    def delete_default_scratchpad(self):
        if os.path.isfile(self.get_scratchpad_default()):
            os.remove(self.get_scratchpad_default())
            logger.info("Removed the scratchpad default file")

    def delete_autosave_scratchpad(self):
        if os.path.isfile(self.get_scratchpad_autosave()):
            os.remove(self.get_scratchpad_autosave())
            logger.info("Removed the scratchpad autosave file")

    def _recentfilter_func(self, rfinfo):
        """Recent-file filter function.

        This does a filename extension check, and also verifies that the
        file actually exists.

        """
        if not rfinfo:
            return False
        apps = rfinfo.applications
        if not (apps and "mypaint" in apps):
            return False
        return self._uri_is_loadable(rfinfo.uri)
        # Keep this test in sync with _update_recent_items().

    def _uri_is_loadable(self, file_uri):
        """True if a URI is valid to be loaded by MyPaint."""
        if file_uri is None:
            return False
        if not file_uri.startswith("file://"):
            return False
        file_path, _host = lib.glib.filename_from_uri(file_uri)
        if not os.path.exists(file_path):
            return False
        if not self._file_extension_regex.search(file_path):
            return False
        return True
