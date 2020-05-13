# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2010-2018 by the MyPaint Development Team.
# Copyright (C) 2009-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Common dialog functions"""

## Imports
from __future__ import division, print_function

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GdkPixbuf

from gettext import gettext as _
from fnmatch import fnmatch

from . import widgets
from lib.color import RGBColor
from . import uicolor


## Module constants

OVERWRITE_THIS = 1
OVERWRITE_ALL = 2
DONT_OVERWRITE_THIS = 3
DONT_OVERWRITE_ANYTHING = 4
CANCEL = 5


## Function defs

def confirm(widget, question):
    window = widget.get_toplevel()
    d = Gtk.MessageDialog(
        window,
        Gtk.DialogFlags.MODAL,
        Gtk.MessageType.QUESTION,
        Gtk.ButtonsType.NONE,
        question)
    d.add_button(Gtk.STOCK_NO, Gtk.ResponseType.REJECT)
    d.add_button(Gtk.STOCK_YES, Gtk.ResponseType.ACCEPT)
    d.set_default_response(Gtk.ResponseType.ACCEPT)
    response = d.run()
    d.destroy()
    return response == Gtk.ResponseType.ACCEPT


def _entry_activate_dialog_response_cb(entry, dialog,
                                       response=Gtk.ResponseType.ACCEPT):
    dialog.response(response)


def ask_for_name(widget, title, default):
    window = widget.get_toplevel()
    d = Gtk.Dialog(title,
                   window,
                   Gtk.DialogFlags.MODAL,
                   (Gtk.STOCK_CANCEL, Gtk.ResponseType.REJECT,
                    Gtk.STOCK_OK, Gtk.ResponseType.ACCEPT))
    d.set_position(Gtk.WindowPosition.CENTER_ON_PARENT)

    hbox = Gtk.HBox()
    hbox.set_property("spacing", widgets.SPACING)
    hbox.set_border_width(widgets.SPACING)

    d.vbox.pack_start(hbox, True, True, 0)
    hbox.pack_start(Gtk.Label(label=_('Name')), False, False, 0)

    if default is None:
        default = ""

    d.e = e = Gtk.Entry()
    e.set_size_request(250, -1)
    e.set_text(default)
    e.select_region(0, len(default))
    e.set_input_hints(Gtk.InputHints.UPPERCASE_WORDS)
    e.set_input_purpose(Gtk.InputPurpose.FREE_FORM)

    e.connect("activate", _entry_activate_dialog_response_cb, d)

    hbox.pack_start(e, True, True, 0)
    d.vbox.show_all()
    if d.run() == Gtk.ResponseType.ACCEPT:
        result = d.e.get_text()
        if isinstance(result, bytes):
            result = result.decode('utf-8')
    else:
        result = None
    d.destroy()
    return result


def error(widget, message):
    window = widget.get_toplevel()
    d = Gtk.MessageDialog(
        window,
        Gtk.DialogFlags.MODAL,
        Gtk.MessageType.ERROR,
        Gtk.ButtonsType.OK,
        message,
    )
    d.run()
    d.destroy()


def image_new_from_png_data(data):
    loader = GdkPixbuf.PixbufLoader.new_with_type("png")
    loader.write(data)
    loader.close()
    pixbuf = loader.get_pixbuf()
    image = Gtk.Image()
    image.set_from_pixbuf(pixbuf)
    return image


def confirm_rewrite_brush(window, brushname, existing_preview_pixbuf,
                          imported_preview_data):
    flags = Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT
    dialog = Gtk.Dialog(_("Overwrite brush?"), window, flags)

    cancel = Gtk.Button(stock=Gtk.STOCK_CANCEL)
    cancel.show_all()
    img_yes = Gtk.Image()
    img_yes.set_from_stock(Gtk.STOCK_YES, Gtk.IconSize.BUTTON)
    img_no = Gtk.Image()
    img_no.set_from_stock(Gtk.STOCK_NO, Gtk.IconSize.BUTTON)
    overwrite_this = Gtk.Button(label=_("Replace"))
    overwrite_this.set_image(img_yes)
    overwrite_this.show_all()
    skip_this = Gtk.Button(label=_("Rename"))
    skip_this.set_image(img_no)
    skip_this.show_all()
    overwrite_all = Gtk.Button(label=_("Replace all"))
    overwrite_all.show_all()
    skip_all = Gtk.Button(label=_("Rename all"))
    skip_all.show_all()

    buttons = [
        (cancel, CANCEL),
        (skip_all, DONT_OVERWRITE_ANYTHING),
        (overwrite_all, OVERWRITE_ALL),
        (skip_this, DONT_OVERWRITE_THIS),
        (overwrite_this, OVERWRITE_THIS),
    ]
    for button, code in buttons:
        dialog.add_action_widget(button, code)

    hbox = Gtk.HBox()
    vbox_l = Gtk.VBox()
    vbox_r = Gtk.VBox()
    try:
        preview_r = Gtk.image_new_from_pixbuf(existing_preview_pixbuf)
    except AttributeError:
        preview_r = Gtk.Image.new_from_pixbuf(existing_preview_pixbuf)
    label_l = Gtk.Label(label=_("Imported brush"))
    label_r = Gtk.Label(label=_("Existing brush"))

    question = Gtk.Label(label=_(
        u"<b>A brush named “{brush_name}” already exists.</b>\n"
        u"Do you want to replace it, "
        u"or should the new brush be renamed?"
    ).format(
        brush_name = brushname,
    ))
    question.set_use_markup(True)

    preview_l = image_new_from_png_data(imported_preview_data)

    vbox_l.pack_start(preview_l, True, True, 0)
    vbox_l.pack_start(label_l, False, True, 0)

    vbox_r.pack_start(preview_r, True, True, 0)
    vbox_r.pack_start(label_r, False, True, 0)

    hbox.pack_start(vbox_l, False, True, 0)
    hbox.pack_start(question, True, True, 0)
    hbox.pack_start(vbox_r, False, True, 0)
    hbox.show_all()

    dialog.vbox.pack_start(hbox, True, True, 0)

    answer = dialog.run()
    dialog.destroy()
    return answer


def confirm_rewrite_group(window, groupname, deleted_groupname):
    flags = Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT
    dialog = Gtk.Dialog(_("Overwrite brush group?"), window, flags)

    cancel = Gtk.Button(stock=Gtk.STOCK_CANCEL)
    cancel.show_all()
    img_yes = Gtk.Image()
    img_yes.set_from_stock(Gtk.STOCK_YES, Gtk.IconSize.BUTTON)
    img_no = Gtk.Image()
    img_no.set_from_stock(Gtk.STOCK_NO, Gtk.IconSize.BUTTON)
    overwrite_this = Gtk.Button(label=_("Replace"))
    overwrite_this.set_image(img_yes)
    overwrite_this.show_all()
    skip_this = Gtk.Button(label=_("Rename"))
    skip_this.set_image(img_no)
    skip_this.show_all()

    buttons = [
        (cancel, CANCEL),
        (skip_this, DONT_OVERWRITE_THIS),
        (overwrite_this, OVERWRITE_THIS),
    ]
    for button, code in buttons:
        dialog.add_action_widget(button, code)

    question = Gtk.Label(label=_(
        u"<b>A group named “{groupname}” already exists.</b>\n"
        u"Do you want to replace it, or should the new group be renamed?\n"
        u"If you replace it, the brushes may be moved to a group called"
        u" “{deleted_groupname}”."
    ).format(
        groupname=groupname,
        deleted_groupname=deleted_groupname,
    ))
    question.set_use_markup(True)

    dialog.vbox.pack_start(question, True, True, 0)
    dialog.vbox.show_all()

    answer = dialog.run()
    dialog.destroy()
    return answer


def open_dialog(title, window, filters):
    """Show a file chooser dialog.

    Filters should be a list of tuples: (filtertitle, globpattern).

    Returns a tuple of the form (fileformat, filename). Here
    "fileformat" is the index of the filter that matched filename, or
    None if there were no matches).  "filename" is None if no file was
    selected.

    """
    dialog = Gtk.FileChooserDialog(title, window,
                                   Gtk.FileChooserAction.OPEN,
                                   (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                    Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
    dialog.set_default_response(Gtk.ResponseType.OK)
    for filter_title, pattern in filters:
        f = Gtk.FileFilter()
        f.set_name(filter_title)
        f.add_pattern(pattern)
        dialog.add_filter(f)

    result = (None, None)
    if dialog.run() == Gtk.ResponseType.OK:
        filename = dialog.get_filename()
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        file_format = None
        for i, (_junk, pattern) in enumerate(filters):
            if fnmatch(filename, pattern):
                file_format = i
                break
        result = (file_format, filename)
    dialog.hide()
    return result


def save_dialog(title, window, filters, default_format=None):
    """Shows a file save dialog.

    "filters" should be a list of tuples: (filter title, glob pattern).

    "default_format" may be a pair (format id, suffix).
    That suffix will be added to filename if it does not match any of filters.

    Returns a tuple of the form (fileformat, filename).  Here
    "fileformat" is index of filter that matches filename, or None if no
    matches).  "filename" is None if no file was selected.

    """
    dialog = Gtk.FileChooserDialog(title, window,
                                   Gtk.FileChooserAction.SAVE,
                                   (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                    Gtk.STOCK_SAVE, Gtk.ResponseType.OK))
    dialog.set_default_response(Gtk.ResponseType.OK)
    dialog.set_do_overwrite_confirmation(True)

    for filter_title, pattern in filters:
        f = Gtk.FileFilter()
        f.set_name(filter_title)
        f.add_pattern(pattern)
        dialog.add_filter(f)

    result = (None, None)
    while dialog.run() == Gtk.ResponseType.OK:
        filename = dialog.get_filename()
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        file_format = None
        for i, (_junk, pattern) in enumerate(filters):
            if fnmatch(filename, pattern):
                file_format = i
                break
        if file_format is None and default_format is not None:
            file_format, suffix = default_format
            filename += suffix
            dialog.set_current_name(filename)
            dialog.response(Gtk.ResponseType.OK)
        else:
            result = (file_format, filename)
            break
    dialog.hide()
    return result


def confirm_brushpack_import(packname, window=None, readme=None):

    dialog = Gtk.Dialog(
        _("Import brush package?"),
        window,
        Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT,
        (
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.REJECT,
            Gtk.STOCK_OK,
            Gtk.ResponseType.ACCEPT
        )
    )

    dialog.vbox.set_spacing(12)

    if readme:
        tv = Gtk.TextView()
        tv.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        tv.get_buffer().set_text(readme)
        tv.set_editable(False)
        tv.set_left_margin(12)
        tv.set_right_margin(12)
        try:  # methods introduced in GTK 3.18
            tv.set_top_margin(6)
            tv.set_bottom_margin(6)
        except AttributeError:
            pass
        scrolls = Gtk.ScrolledWindow()
        scrolls.set_size_request(640, 480)
        scrolls.add(tv)
        dialog.vbox.pack_start(scrolls, True, True, 0)

    question = Gtk.Label(label=_(
        "<b>Do you really want to import package “{brushpack_name}”?</b>"
    ).format(
        brushpack_name=packname,
    ))
    question.set_use_markup(True)
    dialog.vbox.pack_start(question, True, True, 0)
    dialog.vbox.show_all()
    answer = dialog.run()
    dialog.destroy()
    return answer


def ask_for_color(title, color=None, previous_color=None, parent=None):
    """Returns a color chosen by the user via a modal dialog.

    The dialog is a standard `Gtk.ColorSelectionDialog`.
    The returned value may be `None`,
    which means that the user pressed Cancel in the dialog.

    """
    if color is None:
        color = RGBColor(0.5, 0.5, 0.5)
    if previous_color is None:
        previous_color = RGBColor(0.5, 0.5, 0.5)
    dialog = Gtk.ColorSelectionDialog(title)
    sel = dialog.get_color_selection()
    sel.set_current_color(uicolor.to_gdk_color(color))
    sel.set_previous_color(uicolor.to_gdk_color(previous_color))
    dialog.set_position(Gtk.WindowPosition.MOUSE)
    dialog.set_modal(True)
    dialog.set_resizable(False)
    if parent is not None:
        dialog.set_transient_for(parent)
    # GNOME likes to darken the main window
    # when it is set as the transient-for parent window.
    # The setting is "Attached Modal Dialogs", which defaultss to ON.
    # See https://github.com/mypaint/mypaint/issues/325 .
    # This is unhelpful for art programs,
    # but advertising the dialog
    # as a utility window restores sensible behaviour.
    dialog.set_type_hint(Gdk.WindowTypeHint.UTILITY)
    dialog.set_default_response(Gtk.ResponseType.OK)
    response_id = dialog.run()
    result = None
    if response_id == Gtk.ResponseType.OK:
        col_gdk = sel.get_current_color()
        result = uicolor.from_gdk_color(col_gdk)
    dialog.destroy()
    return result
