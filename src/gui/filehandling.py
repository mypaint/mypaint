# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2009-2019 by the MyPaint Development Team
# Copyright (C) 2007-2014 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""File opening/saving."""


## Imports

from __future__ import division, print_function

import os
import re
from glob import glob
import sys
import logging
from collections import OrderedDict
import time

from lib.gibindings import Gtk
from lib.gibindings import Pango

from lib import helpers
from lib import fileutils
from lib.errors import FileHandlingError
from lib.errors import AllocationError
import gui.compatibility as compat
from gui.widgets import with_wait_cursor
from lib import mypaintlib
from lib.gettext import ngettext
from lib.gettext import C_
import lib.glib
from lib.glib import filename_to_unicode
import lib.xml
import lib.feedback
from lib.pycompat import unicode, PY3

logger = logging.getLogger(__name__)


## Save format consts

class _SaveFormat:
    """Safe format consts."""
    ANY = 0
    ORA = 1
    PNG_AUTO = 2
    PNG_SOLID = 3
    PNG_TRANS = 4
    PNGS_BY_LAYER = 5
    PNGS_BY_VIEW = 6
    JPEG = 7


## Internal helper funcs


def _get_case_insensitive_glob(string):
    """Converts a glob pattern into a case-insensitive glob pattern.

    >>> _get_case_insensitive_glob('*.ora')
    '*.[oO][rR][aA]'

    This utility function is a workaround for the GTK
    FileChooser/FileFilter not having an easy way to use case
    insensitive filters

    """
    ext = string.split('.')[1]
    globlist = ["[%s%s]" % (c.lower(), c.upper()) for c in ext]
    return '*.%s' % ''.join(globlist)


def _add_filters_to_dialog(filters, dialog):
    """Adds Gtk.FileFilter objs for patterns to a dialog."""
    for name, patterns in filters:
        f = Gtk.FileFilter()
        f.set_name(name)
        for p in patterns:
            f.add_pattern(_get_case_insensitive_glob(p))
        dialog.add_filter(f)


def _dialog_set_filename(dialog, s):
    """Sets the filename and folder visible in a dialog.

    According to the PyGTK documentation we should use set_filename();
    however, doing so removes the selected file filter.

    TODO: verify whether this is still needed with GTK3+PyGI.

    """
    path, name = os.path.split(s)
    dialog.set_current_folder(path)
    dialog.set_current_name(name)


## Class definitions

class _IOProgressUI:
    """Wraps IO activity calls to show progress to the user.

    Code about to do a potentially lengthy save or load operation
    constructs one one of these temporary state manager objects, and
    uses it to call their supplied IO callable.  The _IOProgressUI
    supplies the IO callable with a lib.feedback.Progress object which
    deeper levels will need to call regularly to keep the UI updated.
    Statusbar messages and error or progress dialogs may be shown via
    the main application.

    Yes, this sounds a lot like context managers and IO coroutines,
    and maybe one day it all will be just that.

    """

    # Message templating consts:

    _OP_DURATION_TEMPLATES = {
        "load": C_(
            "Document I/O: message shown while working",
            u"Loading {files_summary}…",
        ),
        "import": C_(
            "Document I/O: message shown while working",
            u"Importing layers from {files_summary}…",
        ),
        "save": C_(
            "Document I/O: message shown while working",
            u"Saving {files_summary}…",
        ),
        "export": C_(
            "Document I/O: message shown while working",
            u"Exporting to {files_summary}…",
        ),
    }

    _OP_FAILED_TEMPLATES = {
        "export": C_(
            "Document I/O: fail message",
            u"Failed to export to {files_summary}.",
        ),
        "save": C_(
            "Document I/O: fail message",
            u"Failed to save {files_summary}.",
        ),
        "import": C_(
            "Document I/O: fail message",
            u"Could not import layers from {files_summary}.",
        ),
        "load": C_(
            "Document I/O: fail message",
            u"Could not load {files_summary}.",
        ),
    }

    _OP_FAIL_DIALOG_TITLES = {
        "save": C_(
            "Document I/O: fail dialog title",
            u"Save failed",
        ),
        "export": C_(
            "Document I/O: fail dialog title",
            u"Export failed",
        ),
        "import": C_(
            "Document I/O: fail dialog title",
            u"Import Layers failed",
        ),
        "load": C_(
            "Document I/O: fail dialog title",
            u"Open failed",
        ),
    }

    _OP_SUCCEEDED_TEMPLATES = {
        "export": C_(
            "Document I/O: success",
            u"Exported to {files_summary} successfully.",
        ),
        "save": C_(
            "Document I/O: success",
            u"Saved {files_summary} successfully.",
        ),
        "import": C_(
            "Document I/O: success",
            u"Imported layers from {files_summary}.",
        ),
        "load": C_(
            "Document I/O: success",
            u"Loaded {files_summary}.",
        ),
    }

    # Message templating:

    @staticmethod
    def format_files_summary(f):
        """The suggested way of formatting 1+ filenames for display.

        :param f: A list of filenames, or a single filename.
        :returns: A files_summary value for the constructor.
        :rtype: unicode|str

        """
        if isinstance(f, tuple) or isinstance(f, list):
            nfiles = len(f)
            # TRANSLATORS: formatting for {files_summary} for multiple files.
            # TRANSLATORS: corresponding msgid for single files: "“{basename}”"
            return ngettext(u"{n} file", u"{n} files", nfiles).format(
                n=nfiles,
            )
        elif isinstance(f, bytes) or isinstance(f, unicode):
            if isinstance(f, bytes):
                f = f.decode("utf-8")
            return C_(
                "Document I/O: the {files_summary} for a single file",
                u"“{basename}”",
            ).format(basename=os.path.basename(f))
        else:
            raise TypeError("Expected a string, or a sequence of strings.")

    # Method defs:

    def __init__(self, app, op_type, files_summary,
                 use_statusbar=True, use_dialogs=True):
        """Construct, describing what UI messages to show.

        :param app: The top-level MyPaint application object.
        :param str op_type: What kind of operation is about to happen.
        :param unicode files-summary: User-visible descripion of files.
        :param bool use_statusbar: Show statusbar messages for feedback.
        :param bool use_dialogs: Whether to use dialogs for feedback.

        """
        self._app = app
        self.clock_func = time.perf_counter if PY3 else time.clock

        files_summary = unicode(files_summary)
        op_type = str(op_type)
        if op_type not in self._OP_DURATION_TEMPLATES:
            raise ValueError("Unknown operation type %r" % (op_type,))

        msg = self._OP_DURATION_TEMPLATES[op_type].format(
            files_summary = files_summary,
        )
        self._duration_msg = msg

        msg = self._OP_SUCCEEDED_TEMPLATES[op_type].format(
            files_summary = files_summary,
        )
        self._success_msg = msg

        msg = self._OP_FAILED_TEMPLATES[op_type].format(
            files_summary = files_summary,
        )
        self._fail_msg = msg

        msg = self._OP_FAIL_DIALOG_TITLES[op_type]
        self._fail_dialog_title = msg

        self._is_write = (op_type in ["save", "export"])

        cid = self._app.statusbar.get_context_id("filehandling-message")
        self._statusbar_context_id = cid

        self._use_statusbar = bool(use_statusbar)
        self._use_dialogs = bool(use_dialogs)

        #: True only if the IO function run by call() succeeded.
        self.success = False

        self._progress_dialog = None
        self._progress_bar = None
        self._start_time = None
        self._last_pulse = None

    @with_wait_cursor
    def call(self, func, *args, **kwargs):
        """Call a save or load callable and watch its progress.

        :param callable func: The IO function to be called.
        :param \*args: Passed to func.
        :param \*\*kwargs: Passed to func.
        :returns: The return value of func.

        Messages about the operation in progress may be shown to the
        user according to the object's op_type and files_summary.  The
        supplied callable is called with a *args and **kwargs, plus a
        "progress" keyword argument that when updated will keep the UI
        managed by this object updated.

        If the callable returned, self.success is set to True. If it
        raised an exception, it will remain False.

        See also: lib.feedback.Progress.

        """
        statusbar = self._app.statusbar
        progress = lib.feedback.Progress()
        progress.changed += self._progress_changed_cb
        kwargs = kwargs.copy()
        kwargs["progress"] = progress

        cid = self._statusbar_context_id
        if self._use_statusbar:
            statusbar.remove_all(cid)
            statusbar.push(cid, self._duration_msg)

        self._start_time = self.clock_func()
        self._last_pulse = None
        result = None
        try:
            result = func(*args, **kwargs)
        except (FileHandlingError, AllocationError, MemoryError) as e:
            # Catch predictable exceptions here, and don't re-raise
            # them. Dialogs may be shown, but they will use
            # understandable language.
            logger.exception(
                u"IO failed (user-facing explanations: %s / %s)",
                self._fail_msg,
                unicode(e),
            )
            if self._use_statusbar:
                statusbar.remove_all(cid)
                self._app.show_transient_message(self._fail_msg)
            if self._use_dialogs:
                self._app.message_dialog(
                    title=self._fail_dialog_title,
                    text=self._fail_msg,
                    secondary_text=unicode(e),
                    message_type=Gtk.MessageType.ERROR,
                )
            self.success = False
        else:
            if result is False:
                logger.info("IO operation was cancelled by the user")
            else:
                logger.info("IO succeeded: %s", self._success_msg)
            if self._use_statusbar:
                statusbar.remove_all(cid)
                if result is not False:
                    self._app.show_transient_message(self._success_msg)
            self.success = result is not False
        finally:
            if self._progress_bar is not None:
                self._progress_dialog.destroy()
                self._progress_dialog = None
                self._progress_bar = None
        return result

    def _progress_changed_cb(self, progress):
        if self._progress_bar is None:
            now = self.clock_func()
            if (now - self._start_time) > 0.25:
                dialog = Gtk.Dialog(
                    title=self._duration_msg,
                    transient_for=self._app.drawWindow,
                    modal=True,
                    destroy_with_parent=True,
                )
                dialog.set_position(Gtk.WindowPosition.CENTER_ON_PARENT)
                dialog.set_decorated(False)
                style = dialog.get_style_context()
                style.add_class(Gtk.STYLE_CLASS_OSD)

                label = Gtk.Label()
                label.set_text(self._duration_msg)
                label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)

                progress_bar = Gtk.ProgressBar()
                progress_bar.set_size_request(400, -1)

                dialog.vbox.set_border_width(16)
                dialog.vbox.set_spacing(8)
                dialog.vbox.pack_start(label, True, True, 0)
                dialog.vbox.pack_start(progress_bar, True, True, 0)

                progress_bar.show()
                dialog.show_all()
                self._progress_dialog = dialog
                self._progress_bar = progress_bar
                self._last_pulse = now

        self._update_progress_bar(progress)
        self._process_gtk_events()

    def _update_progress_bar(self, progress):
        if not self._progress_bar:
            return
        fraction = progress.fraction()
        if fraction is None:
            now = self.clock_func()
            if (now - self._last_pulse) > 0.1:
                self._progress_bar.pulse()
                self._last_pulse = now
        else:
            self._progress_bar.set_fraction(fraction)

    def _process_gtk_events(self):
        while Gtk.events_pending():
            Gtk.main_iteration()


class FileHandler (object):
    """File handling object, part of the central app object.

    A single app-wide instance of this object is accessible from the
    central gui.application.Application instance as as app.filehandler.
    Several GTK action callbacks for opening and saving files reside
    here, and the object's public methods may be called from other parts
    of the application.

    NOTE: filehandling and drawwindow are very tightly coupled.

    """

    def __init__(self, app):
        self.app = app
        self.save_dialog = None

        # File filters definitions, for dialogs
        # (name, patterns)
        self.file_filters = [(
            C_(
                "save/load dialogs: filter patterns",
                u"All Recognized Formats",
            ), ["*.ora", "*.png", "*.jpg", "*.jpeg"],
        ), (
            C_(
                "save/load dialogs: filter patterns",
                u"OpenRaster (*.ora)",
            ), ["*.ora"],
        ), (
            C_(
                "save/load dialogs: filter patterns",
                u"PNG (*.png)",
            ), ["*.png"],
        ), (
            C_(
                "save/load dialogs: filter patterns",
                u"JPEG (*.jpg; *.jpeg)",
            ), ["*.jpg", "*.jpeg"],
        )]

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
        rf = Gtk.RecentFilter()
        rf.add_pattern('')
        # The blank-string pattern is eeded so the custom func will
        # get URIs at all, despite the needed flags below.
        rf.add_custom(
            func = self._recentfilter_func,
            needed = (
                Gtk.RecentFilterFlags.APPLICATION |
                Gtk.RecentFilterFlags.URI
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

        # { FORMAT: (name, extension, options) }
        self.saveformats = OrderedDict([
            (_SaveFormat.ANY, (C_(
                "save dialogs: save formats and options",
                u"By extension (prefer default format)",
            ), None, {})),
            (_SaveFormat.ORA, (C_(
                "save dialogs: save formats and options",
                u"OpenRaster (*.ora)",
            ), '.ora', {})),
            (_SaveFormat.PNG_AUTO, (C_(
                "save dialogs: save formats and options",
                u"PNG, respecting “Show Background” (*.png)"
            ), '.png', {})),
            (_SaveFormat.PNG_SOLID, (C_(
                "save dialogs: save formats and options",
                u"PNG, solid RGB (*.png)",
            ), '.png', {'alpha': False})),
            (_SaveFormat.PNG_TRANS, (C_(
                "save dialogs: save formats and options",
                u"PNG, transparent RGBA (*.png)",
            ), '.png', {'alpha': True})),
            (_SaveFormat.PNGS_BY_LAYER, (C_(
                "save dialogs: save formats and options",
                u"Multiple PNGs, by layer (*.NUM.png)",
            ), '.png', {'multifile': 'layers'})),
            (_SaveFormat.PNGS_BY_VIEW, (C_(
                "save dialogs: save formats and options",
                u"Multiple PNGs, by view (*.NAME.png)",
            ), '.png', {'multifile': 'views'})),
            (_SaveFormat.JPEG, (C_(
                "save dialogs: save formats and options",
                u"JPEG 90% quality (*.jpg; *.jpeg)",
            ), '.jpg', {'quality': 90})),
        ])
        self.ext2saveformat = {
            ".ora": (_SaveFormat.ORA, "image/openraster"),
            ".png": (_SaveFormat.PNG_AUTO, "image/png"),
            ".jpeg": (_SaveFormat.JPEG, "image/jpeg"),
            ".jpg": (_SaveFormat.JPEG, "image/jpeg"),
        }
        self.config2saveformat = {
            'openraster': _SaveFormat.ORA,
            'jpeg-90%': _SaveFormat.JPEG,
            'png-solid': _SaveFormat.PNG_SOLID,
        }

    def _update_recent_items(self):
        """Updates self._recent_items from the GTK RecentManager.

        This list is consumed in open_last_cb.

        """
        # Note: i.exists() does not work on Windows if the pathname
        # contains utf-8 characters. Since GIMP also saves its URIs
        # with utf-8 characters into this list, I assume this is a
        # gtk bug.  So we use our own test instead of i.exists().

        recent_items = []
        rm = Gtk.RecentManager.get_default()
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

    def init_save_dialog(self, export):
        if export:
            save_dialog_name = C_(
                "Dialogs (window title): File→Export…",
                u"Export"
            )
        else:
            save_dialog_name = C_(
                "Dialogs (window title): File→Save As…",
                u"Save As"
            )
        dialog = Gtk.FileChooserDialog(
            save_dialog_name,
            self.app.drawWindow,
            Gtk.FileChooserAction.SAVE,
            (
                Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                Gtk.STOCK_SAVE, Gtk.ResponseType.OK,
            ),
        )
        dialog.set_default_response(Gtk.ResponseType.OK)
        dialog.set_do_overwrite_confirmation(True)
        _add_filters_to_dialog(self.file_filters, dialog)

        # Add widget for selecting save format
        box = Gtk.HBox()
        box.set_spacing(12)
        label = Gtk.Label(label=C_(
            "save dialogs: formats and options: (label)",
            u"Format to save as:",
        ))
        label.set_alignment(0.0, 0.5)
        combo = Gtk.ComboBoxText()
        for (name, ext, opt) in self.saveformats.values():
            combo.append_text(name)
        combo.set_active(0)
        combo.connect('changed', self.selected_save_format_changed_cb)
        self.saveformat_combo = combo

        box.pack_start(label, True, True, 0)
        box.pack_start(combo, False, True, 0)
        dialog.set_extra_widget(box)
        dialog.show_all()
        return dialog

    def selected_save_format_changed_cb(self, widget):
        """When the user changes the selected format to save as in the dialog,
        change the extension of the filename (if existing) immediately."""
        dialog = self.save_dialog
        filename = dialog.get_filename()
        if filename:
            filename = filename_to_unicode(filename)
            filename, ext = os.path.splitext(filename)
            if ext:
                saveformat = self.saveformat_combo.get_active()
                ext = self.saveformats[saveformat][1]
                if ext is not None:
                    _dialog_set_filename(dialog, filename + ext)

    def confirm_destructive_action(self, title=None, confirm=None,
                                   offer_save=True):
        """Asks the user to confirm an action that might lose work.

        :param unicode title: Short question to ask the user.
        :param unicode confirm: Imperative verb for the "do it" button.
        :param bool offer_save: Set False to turn off the save checkbox.
        :rtype: bool
        :returns: True if the user allows the destructive action

        Phrase the title question tersely.
        In English/source, use title case for it, and with a question mark.
        Good examples are “Really Quit?”,
        or “Delete Everything?”.
        The title should always tell the user
        what destructive action is about to take place.
        If it is not specified, a default title is used.

        Use a single, specific, imperative verb for the confirm string.
        It should reflect the title question.
        This is used for the primary confirmation button, if specified.
        See the GNOME HIG for further guidelines on what to use here.

        This method doesn't bother asking
        if there's less than a handful of seconds of unsaved work.
        By default, that's 1 second.
        The build-time and runtime debugging flags
        make this period longer
        to allow more convenient development and testing.

        Ref: https://developer.gnome.org/hig/stable/dialogs.html.en

        """
        if title is None:
            title = C_(
                "Destructive action confirm dialog: "
                "fallback title (normally overridden)",
                "Really Continue?"
            )

        # Get an accurate assessment of how much change is unsaved.
        self.doc.model.sync_pending_changes()
        t = self.doc.model.unsaved_painting_time

        # This used to be 30, but see https://gna.org/bugs/?17955
        # Then 8 by default, but Twitter users hate that too.
        t_bother = 1
        if mypaintlib.heavy_debug:
            t_bother += 7
        if os.environ.get("MYPAINT_DEBUG", False):
            t_bother += 7
        logger.debug("Destructive action don't-bother period is %ds", t_bother)
        if t < t_bother:
            return True

        # Custom response codes.
        # The default ones are all negative ints.
        continue_response_code = 1

        # Dialog setup.
        d = Gtk.MessageDialog(
            title=title,
            transient_for=self.app.drawWindow,
            message_type=Gtk.MessageType.QUESTION,
            modal=True
        )

        # Translated strings for things
        cancel_btn_text = C_(
            "Destructive action confirm dialog: cancel button",
            u"_Cancel",
        )
        save_to_scraps_first_text = C_(
            "Destructive action confirm dialog: save checkbox",
            u"_Save to Scraps first",
        )
        if not confirm:
            continue_btn_text = C_(
                "Destructive action confirm dialog: "
                "fallback continue button (normally overridden)",
                u"Co_ntinue",
            )
        else:
            continue_btn_text = confirm

        # Button setup. Cancel first, continue at end.
        buttons = [
            (cancel_btn_text, Gtk.ResponseType.CANCEL, False),
            (continue_btn_text, continue_response_code, True),
        ]
        for txt, code, destructive in buttons:
            button = d.add_button(txt, code)
            styles = button.get_style_context()
            if destructive:
                styles.add_class(Gtk.STYLE_CLASS_DESTRUCTIVE_ACTION)

        # Explanatory message.
        if self.filename:
            file_basename = os.path.basename(self.filename)
        else:
            file_basename = None
        warning_msg_tmpl = C_(
            "Destructive action confirm dialog: warning message",
            u"You risk losing {abbreviated_time} of unsaved painting."
        )
        markup_tmpl = warning_msg_tmpl
        d.set_markup(markup_tmpl.format(
            abbreviated_time = lib.xml.escape(helpers.fmt_time_period_abbr(t)),
            current_file_name = lib.xml.escape(file_basename),
        ))

        # Checkbox for saving
        if offer_save:
            save1st_text = save_to_scraps_first_text
            save1st_cb = Gtk.CheckButton.new_with_mnemonic(save1st_text)
            save1st_cb.set_hexpand(False)
            save1st_cb.set_halign(Gtk.Align.END)
            save1st_cb.set_vexpand(False)
            save1st_cb.set_margin_top(12)
            save1st_cb.set_margin_bottom(12)
            save1st_cb.set_margin_start(12)
            save1st_cb.set_margin_end(12)
            save1st_cb.set_can_focus(False)  # set back again in show handler
            d.connect(
                "show",
                self._destructive_action_dialog_show_cb,
                save1st_cb,
            )
            save1st_cb.connect(
                "toggled",
                self._destructive_action_dialog_save1st_toggled_cb,
                d,
            )
            vbox = d.get_content_area()
            vbox.set_spacing(0)
            vbox.set_margin_top(12)
            vbox.pack_start(save1st_cb, False, True, 0)

        # Get a response and handle it.
        d.set_default_response(Gtk.ResponseType.CANCEL)
        response_code = d.run()
        d.destroy()
        if response_code == continue_response_code:
            logger.debug("Destructive action confirmed")
            if offer_save and save1st_cb.get_active():
                logger.info("Saving current canvas as a new scrap")
                self.save_scrap_cb(None)
            return True
        else:
            logger.debug("Destructive action cancelled")
            return False

    def _destructive_action_dialog_show_cb(self, dialog, checkbox):
        checkbox.show_all()
        checkbox.set_can_focus(True)

    def _destructive_action_dialog_save1st_toggled_cb(self, checkbox, dialog):
        # Choosing to save locks you into a particular course of action.
        # Hopefully this isn't too strange.
        # Escape will still work.
        cancel_allowed = not checkbox.get_active()
        cancel_btn = dialog.get_widget_for_response(Gtk.ResponseType.CANCEL)
        cancel_btn.set_sensitive(cancel_allowed)

    def new_cb(self, action):
        ok_to_start_new_doc = self.confirm_destructive_action(
            title = C_(
                u'File→New: confirm dialog: title question',
                u"New Canvas?",
            ),
            confirm = C_(
                u'File→New: confirm dialog: continue button',
                u"_New Canvas",
            ),
        )
        if not ok_to_start_new_doc:
            return
        self.app.reset_compat_mode()
        self.doc.reset_background()
        self.doc.model.clear()
        self.filename = None
        self._update_recent_items()
        self.app.doc.reset_view(True, True, True)

    @staticmethod
    def gtk_main_tick(*args, **kwargs):
        while Gtk.events_pending():
            Gtk.main_iteration()

    def open_file(self, filename, **kwargs):
        """Load a file, replacing the current working document."""
        if not self._call_doc_load_method(
                self.doc.model.load, filename, False, **kwargs):
            # Without knowledge of _when_ the process failed, clear
            # the document to make sure we're not in an inconsistent state.
            # TODO: Improve the control flow to permit a less draconian
            # approach, for exceptions occurring prior to any doc-changes.
            self.filename = None
            self.app.reset_compat_mode()
            self.doc.model.clear()
            return

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

    def import_layers(self, filenames):
        """Load a file, replacing the current working document."""

        if not self._call_doc_load_method(self.doc.model.import_layers,
                                          filenames, True):
            return
        logger.info('Imported layers from %r', filenames)

    def _call_doc_load_method(
            self, method, arg, is_import, compat_handler=None):
        """Internal: common GUI aspects of loading or importing files.

        Calls a document model loader method (on lib.document.Document)
        with the given argument. Catches common loading exceptions and
        shows appropriate error messages.

        """
        if not compat_handler:
            compat_handler = compat.ora_compat_handler(self.app)
        prefs = self.app.preferences
        display_colorspace_setting = prefs["display.colorspace"]

        op_type = is_import and "import" or "load"

        files_summary = _IOProgressUI.format_files_summary(arg)
        ioui = _IOProgressUI(self.app, op_type, files_summary)
        result = ioui.call(
            method, arg,
            convert_to_srgb=(display_colorspace_setting == "srgb"),
            compat_handler=compat_handler,
            incompatible_ora_cb=compat.incompatible_ora_cb(self.app)
        )
        return (result is not False) and ioui.success

    def open_scratchpad(self, filename):
        no_ui_progress = lib.feedback.Progress()
        no_ui_progress.changed += self.gtk_main_tick
        try:
            self.app.scratchpad_doc.model.load(
                filename,
                progress=no_ui_progress,
            )
            self.app.scratchpad_filename = os.path.abspath(filename)
            self.app.preferences["scratchpad.last_opened_scratchpad"] \
                = self.app.scratchpad_filename
        except (FileHandlingError, AllocationError, MemoryError) as e:
            self.app.message_dialog(
                unicode(e),
                message_type=Gtk.MessageType.ERROR
            )
        else:
            self.app.scratchpad_filename = os.path.abspath(filename)
            self.app.preferences["scratchpad.last_opened_scratchpad"] \
                = self.app.scratchpad_filename
            logger.info('Loaded scratchpad from %r',
                        self.app.scratchpad_filename)
            self.app.scratchpad_doc.reset_view(True, True, True)

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
            use_statusbar=True,
            **options
        )
        if "multifile" in options:  # thumbs & recents are inappropriate
            return
        if not os.path.isfile(filename):  # failed to save
            return
        if not export:
            self.filename = os.path.abspath(filename)
            basename, ext = os.path.splitext(self.filename)
            recent_mgr = Gtk.RecentManager.get_default()
            uri = lib.glib.filename_to_uri(self.filename)
            recent_data = Gtk.RecentData()
            recent_data.app_name = "mypaint"
            app_exec = sys.argv_unicode[0]
            assert isinstance(app_exec, unicode)
            recent_data.app_exec = app_exec
            mime_default = "application/octet-stream"
            fmt, mime_type = self.ext2saveformat.get(ext, (None, mime_default))
            recent_data.mime_type = mime_type
            recent_mgr.add_full(uri, recent_data)
        if not thumbnail_pixbuf:
            options["render_background"] = not options.get("alpha", False)
            thumbnail_pixbuf = self.doc.model.render_thumbnail(**options)
        helpers.freedesktop_thumbnail(filename, thumbnail_pixbuf)

    @with_wait_cursor
    def save_scratchpad(self, filename, export=False, **options):
        save_needed = (
            self.app.scratchpad_doc.model.unsaved_painting_time
            or export
            or not os.path.exists(filename)
        )
        if save_needed:
            self._save_doc_to_file(
                filename,
                self.app.scratchpad_doc,
                export=export,
                use_statusbar=False,
                **options
            )
        if not export:
            self.app.scratchpad_filename = os.path.abspath(filename)
            self.app.preferences["scratchpad.last_opened_scratchpad"] \
                = self.app.scratchpad_filename

    def _save_doc_to_file(self, filename, doc, export=False,
                          use_statusbar=True,
                          **options):
        """Saves a document to one or more files

        :param filename: The base filename to save
        :param Document doc: Controller for the document to save
        :param bool export: True if exporting
        :param **options: Pass-through options

        This method handles logging, statusbar messages,
        and alerting the user to when the save failed.

        See also: lib.document.Document.save(), _IOProgressUI.
        """
        thumbnail_pixbuf = None
        prefs = self.app.preferences
        display_colorspace_setting = prefs["display.colorspace"]
        options['save_srgb_chunks'] = (display_colorspace_setting == "srgb")

        files_summary = _IOProgressUI.format_files_summary(filename)
        op_type = export and "export" or "save"
        ioui = _IOProgressUI(self.app, op_type, files_summary,
                             use_statusbar=use_statusbar)

        thumbnail_pixbuf = ioui.call(doc.model.save, filename, **options)
        self.lastsavefailed = not ioui.success
        return thumbnail_pixbuf

    def update_preview_cb(self, file_chooser, preview):
        filename = file_chooser.get_preview_filename()
        if filename:
            filename = filename_to_unicode(filename)
            pixbuf = helpers.freedesktop_thumbnail(filename)
            if pixbuf:
                # if pixbuf is smaller than 256px in width, copy it onto
                # a transparent 256x256 pixbuf
                pixbuf = helpers.pixbuf_thumbnail(pixbuf, 256, 256, True)
                preview.set_from_pixbuf(pixbuf)
                file_chooser.set_preview_widget_active(True)
            else:
                # TODO: display "no preview available" image?
                file_chooser.set_preview_widget_active(False)

    def open_cb(self, action):
        ok_to_open = self.app.filehandler.confirm_destructive_action(
            title = C_(
                u'File→Open: confirm dialog: title question',
                u"Open File?",
            ),
            confirm = C_(
                u'File→Open: confirm dialog: continue button',
                u"_Open…",
            ),
        )
        if not ok_to_open:
            return
        dialog = Gtk.FileChooserDialog(
            title=C_(
                u'File→Open: file chooser dialog: title',
                u"Open File",
            ),
            transient_for=self.app.drawWindow,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
        dialog.add_button(Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        dialog.set_default_response(Gtk.ResponseType.OK)

        # Compatibility override options for .ora files
        selector = compat.CompatSelector(self.app)
        dialog.connect('selection-changed', selector.file_selection_changed_cb)
        dialog.set_extra_widget(selector.widget)

        preview = Gtk.Image()
        dialog.set_preview_widget(preview)
        dialog.connect("update-preview", self.update_preview_cb, preview)

        _add_filters_to_dialog(self.file_filters, dialog)

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
            if dialog.run() == Gtk.ResponseType.OK:
                dialog.hide()
                filename = dialog.get_filename()
                filename = filename_to_unicode(filename)
                self.open_file(
                    filename,
                    compat_handler=selector.compat_function
                )
        finally:
            dialog.destroy()

    def open_scratchpad_dialog(self):
        dialog = Gtk.FileChooserDialog(
            C_(
                "load dialogs: title",
                u"Open Scratchpad…",
            ),
            self.app.drawWindow,
            Gtk.FileChooserAction.OPEN,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
             Gtk.STOCK_OPEN, Gtk.ResponseType.OK),
        )
        dialog.set_default_response(Gtk.ResponseType.OK)

        preview = Gtk.Image()
        dialog.set_preview_widget(preview)
        dialog.connect("update-preview", self.update_preview_cb, preview)

        _add_filters_to_dialog(self.file_filters, dialog)

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
            if dialog.run() == Gtk.ResponseType.OK:
                dialog.hide()
                filename = dialog.get_filename()
                filename = filename_to_unicode(filename)
                self.app.scratchpad_filename = filename
                self.open_scratchpad(filename)
        finally:
            dialog.destroy()

    def import_layers_cb(self, action):
        """Action callback: import layers from multiple files."""
        dialog = Gtk.FileChooserDialog(
            title = C_(
                u'Layers→Import Layers: files-chooser dialog: title',
                u"Import Layers",
            ),
            parent = self.app.drawWindow,
            action = Gtk.FileChooserAction.OPEN,
        )
        dialog.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
        dialog.add_button(Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        dialog.set_default_response(Gtk.ResponseType.OK)

        dialog.set_select_multiple(True)

        # TODO: decide how well the preview plays with multiple-select.
        preview = Gtk.Image()
        dialog.set_preview_widget(preview)
        dialog.connect("update-preview", self.update_preview_cb, preview)

        _add_filters_to_dialog(self.file_filters, dialog)

        # Choose the most recent save folder.
        self._update_recent_items()
        for item in reversed(self._recent_items):
            uri = item.get_uri()
            fn, _h = lib.glib.filename_from_uri(uri)
            dn = os.path.dirname(fn)
            if os.path.isdir(dn):
                dialog.set_current_folder(dn)
                break

        filenames = []
        try:
            if dialog.run() == Gtk.ResponseType.OK:
                dialog.hide()
                filenames = dialog.get_filenames()
        finally:
            dialog.destroy()

        if filenames:
            filenames = [filename_to_unicode(f) for f in filenames]
            self.import_layers(filenames)

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

        self.save_as_dialog(
            self.save_file,
            suggested_filename=current_filename,
            export = (action.get_name() == 'Export'),
        )

    def save_scratchpad_as_dialog(self, export=False):
        if self.app.scratchpad_filename:
            current_filename = self.app.scratchpad_filename
        else:
            current_filename = ''

        self.save_as_dialog(
            self.save_scratchpad,
            suggested_filename=current_filename,
            export=export,
        )

    def save_as_dialog(self, save_method_reference, suggested_filename=None,
                       start_in_folder=None, export=False,
                       **options):
        if not self.save_dialog:
            self.save_dialog = self.init_save_dialog(export)
        dialog = self.save_dialog
        # Set the filename in the dialog
        if suggested_filename:
            _dialog_set_filename(dialog, suggested_filename)
        else:
            _dialog_set_filename(dialog, '')
            # Recent directory?
            if start_in_folder:
                dialog.set_current_folder(start_in_folder)

        try:
            # Loop until we have filename with an extension
            while dialog.run() == Gtk.ResponseType.OK:
                filename = dialog.get_filename()
                if filename is None:
                    continue
                filename = filename_to_unicode(filename)
                name, ext = os.path.splitext(filename)
                saveformat = self.saveformat_combo.get_active()

                # If no explicitly selected format, use the extension to
                # figure it out
                if saveformat == _SaveFormat.ANY:
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
                desc, ext_format, options = self.saveformats.get(
                    saveformat,
                    ("", ext, {'alpha': None}),
                )

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
                _dialog_set_filename(dialog, filename)
                dialog.response(Gtk.ResponseType.OK)

        finally:
            dialog.hide()
            dialog.destroy()  # avoid GTK crash: https://gna.org/bugs/?17902
            self.save_dialog = None

    def save_scrap_cb(self, action):
        filename = self.filename
        prefix = self.get_scrap_prefix()
        self.app.filename = self.save_autoincrement_file(
            filename,
            prefix,
            main_doc=True,
        )

    def save_scratchpad_cb(self, action):
        filename = self.app.scratchpad_filename
        prefix = self.get_scratchpad_prefix()
        self.app.scratchpad_filename = self.save_autoincrement_file(
            filename,
            prefix,
            main_doc=False,
        )

    def save_autoincrement_file(self, filename, prefix, main_doc=True):
        # If necessary, create the folder(s) the scraps are stored under
        prefix_dir = os.path.dirname(prefix)
        if not os.path.exists(prefix_dir):
            os.makedirs(prefix_dir)

        number = None
        if filename:
            junk, file_fragment = os.path.split(filename)
            if file_fragment.startswith("_md5"):
                # store direct, don't attempt to increment
                if main_doc:
                    self.save_file(filename)
                else:
                    self.save_scratchpad(filename)
                return filename

            found_nums = re.findall(re.escape(prefix) + '([0-9]+)', filename)
            if found_nums:
                number = found_nums[0]

        if number:
            # reuse the number, find the next character
            char = 'a'
            for filename in glob(prefix + number + '_*'):
                c = filename[len(prefix + number + '_')]
                if c >= 'a' and c <= 'z' and c >= char:
                    char = chr(ord(c) + 1)
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
            filename = '%s%03d_a' % (prefix, maximum + 1)

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
        return self._list_prefixed_dir(prefix)

    def list_scratchpads(self):
        prefix = self.get_scratchpad_prefix()
        files = self._list_prefixed_dir(prefix)
        special_prefix = os.path.join(prefix, "special")
        if os.path.isdir(special_prefix):
            files += self._list_prefixed_dir(special_prefix + os.path.sep)
        return files

    def _list_prefixed_dir(self, prefix):
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
        uri = action.get_current_uri()
        fn, _h = lib.glib.filename_from_uri(uri)
        ok_to_open = self.app.filehandler.confirm_destructive_action(
            title = C_(
                u'File→Open Recent→* confirm dialog: title',
                u"Open File?"
            ),
            confirm = C_(
                u'File→Open Recent→* confirm dialog: continue button',
                u"_Open"
            ),
        )
        if not ok_to_open:
            return
        self.open_file(fn)

    def open_last_cb(self, action):
        """Callback to open the last file"""
        if not self._recent_items:
            return
        ok_to_open = self.app.filehandler.confirm_destructive_action(
            title = C_(
                u'File→Open Most Recent confirm dialog: '
                u'title',
                u"Open Most Recent File?",
            ),
            confirm = C_(
                u'File→Open Most Recent→* confirm dialog: '
                u'continue button',
                u"_Open"
            ),
        )
        if not ok_to_open:
            return
        uri = self._recent_items.pop().get_uri()
        fn, _h = lib.glib.filename_from_uri(uri)
        self.open_file(fn)

    def open_scrap_cb(self, action):
        groups = self.list_scraps_grouped()
        if not groups:
            msg = C_(
                'File→Open Next/Prev Scrap: error message',
                u"There are no scrap files yet. Try saving one first.",
            )
            self.app.message_dialog(msg, message_type=Gtk.MessageType.WARNING)
            return
        next = action.get_name() == 'NextScrap'
        if next:
            dialog_title = C_(
                u'File→Open Next/Prev Scrap confirm dialog: '
                u'title',
                u"Open Next Scrap?"
            )
            idx = 0
            delta = 1
        else:
            dialog_title = C_(
                u'File→Open Next/Prev Scrap confirm dialog: '
                u'title',
                u"Open Previous Scrap?"
            )
            idx = -1
            delta = -1
        ok_to_open = self.app.filehandler.confirm_destructive_action(
            title = dialog_title,
            confirm = C_(
                u'File→Open Next/Prev Scrap confirm dialog: '
                u'continue button',
                u"_Open"
            ),
        )
        if not ok_to_open:
            return
        for i, group in enumerate(groups):
            if self.active_scrap_filename in group:
                idx = i + delta
        filename = groups[idx % len(groups)][-1]
        self.open_file(filename)

    def reload_cb(self, action):
        if not self.filename:
            self.app.show_transient_message(C_(
                u'File→Revert: status message: canvas has no filename yet',
                u"Cannot revert: canvas has not been saved to a file yet.",
            ))
            return
        ok_to_reload = self.app.filehandler.confirm_destructive_action(
            title = C_(
                u'File→Revert confirm dialog: '
                u'title',
                u"Revert Changes?",
            ),
            confirm = C_(
                u'File→Revert confirm dialog: '
                u'continue button',
                u"_Revert"
            ),
        )
        if ok_to_reload:
            self.open_file(self.filename)

    def delete_scratchpads(self, filenames):
        prefix = self.get_scratchpad_prefix()
        prefix = os.path.abspath(prefix)
        for filename in filenames:
            if not (os.path.isfile(filename) and
                    os.path.abspath(filename).startswith(prefix)):
                continue
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
