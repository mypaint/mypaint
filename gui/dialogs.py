# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
from gtk import gdk
from gettext import gettext as _
from fnmatch import fnmatch
import brushmanager
from pixbuflist import PixbufList
import widgets
import spinbox
from colors import HSVColor
import windowing


OVERWRITE_THIS = 1
OVERWRITE_ALL  = 2
DONT_OVERWRITE_THIS = 3
DONT_OVERWRITE_ANYTHING = 4
CANCEL = 5

def confirm(widget, question):
    window = widget.get_toplevel()
    d = gtk.MessageDialog(
        window,
        gtk.DIALOG_MODAL,
        gtk.MESSAGE_QUESTION,
        gtk.BUTTONS_NONE,
        question)
    d.add_button(gtk.STOCK_NO, gtk.RESPONSE_REJECT)
    d.add_button(gtk.STOCK_YES, gtk.RESPONSE_ACCEPT)
    d.set_default_response(gtk.RESPONSE_ACCEPT)
    response = d.run()
    d.destroy()
    return response == gtk.RESPONSE_ACCEPT

def ask_for_name(widget, title, default):
    window = widget.get_toplevel()
    d = gtk.Dialog(title,
                   window,
                   gtk.DIALOG_MODAL,
                   (gtk.STOCK_CANCEL, gtk.RESPONSE_REJECT,
                    gtk.STOCK_OK, gtk.RESPONSE_ACCEPT))
    d.set_position(gtk.WIN_POS_MOUSE)

    hbox = gtk.HBox()
    hbox.set_property("spacing", widgets.SPACING)
    hbox.set_border_width(widgets.SPACING)

    d.vbox.pack_start(hbox)
    hbox.pack_start(gtk.Label(_('Name')), False, False)

    d.e = e = gtk.Entry()
    e.set_size_request(250, -1)
    e.set_text(default)
    e.select_region(0, len(default))
    def responseToDialog(entry, dialog, response):  
        dialog.response(response)  
    e.connect("activate", responseToDialog, d, gtk.RESPONSE_ACCEPT)  

    hbox.pack_start(e, True, True)
    d.vbox.show_all()
    if d.run() == gtk.RESPONSE_ACCEPT:
        result = d.e.get_text().decode('utf-8')
    else:
        result = None
    d.destroy()
    return result

def error(widget, message):
    window = widget.get_toplevel()
    d = gtk.MessageDialog(window, gtk.DIALOG_MODAL, gtk.MESSAGE_ERROR, gtk.BUTTONS_OK, message)
    d.run()
    d.destroy()

def image_new_from_png_data(data):
    loader = gtk.gdk.PixbufLoader("png")
    loader.write(data)
    loader.close()
    pixbuf = loader.get_pixbuf()
    image = gtk.Image()
    image.set_from_pixbuf(pixbuf)
    return image

def confirm_rewrite_brush(window, brushname, existing_preview_pixbuf, imported_preview_data):
    dialog = gtk.Dialog(_("Overwrite brush?"),
                        window, gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT)

    cancel         = gtk.Button(stock=gtk.STOCK_CANCEL)
    cancel.show_all()
    img_yes        = gtk.Image()
    img_yes.set_from_stock(gtk.STOCK_YES, gtk.ICON_SIZE_BUTTON)
    img_no         = gtk.Image()
    img_no.set_from_stock(gtk.STOCK_NO, gtk.ICON_SIZE_BUTTON)
    overwrite_this = gtk.Button(_("Replace"))
    overwrite_this.set_image(img_yes)
    overwrite_this.show_all()
    skip_this      = gtk.Button(_("Rename"))
    skip_this.set_image(img_no)
    skip_this.show_all()
    overwrite_all  = gtk.Button(_("Replace all"))
    overwrite_all.show_all()
    skip_all       = gtk.Button(_("Rename all"))
    skip_all.show_all()

    buttons = [(cancel,         CANCEL),
               (skip_all,       DONT_OVERWRITE_ANYTHING),
               (overwrite_all,  OVERWRITE_ALL),
               (skip_this,      DONT_OVERWRITE_THIS),
               (overwrite_this, OVERWRITE_THIS)]
    for button, code in buttons:
        dialog.add_action_widget(button, code)

    hbox   = gtk.HBox()
    vbox_l = gtk.VBox()
    vbox_r = gtk.VBox()
    preview_r = gtk.image_new_from_pixbuf(existing_preview_pixbuf)
    label_l = gtk.Label(_("Imported brush"))
    label_r = gtk.Label(_("Existing brush"))

    question = gtk.Label(_("<b>A brush named `%s' already exists.</b>\nDo you want to replace it, or should the new brush be renamed?") % brushname)
    question.set_use_markup(True)

    preview_l = image_new_from_png_data(imported_preview_data)

    vbox_l.pack_start(preview_l, expand=True)
    vbox_l.pack_start(label_l, expand=False)

    vbox_r.pack_start(preview_r, expand=True)
    vbox_r.pack_start(label_r, expand=False)

    hbox.pack_start(vbox_l, expand=False)
    hbox.pack_start(question, expand=True)
    hbox.pack_start(vbox_r, expand=False)
    hbox.show_all()

    dialog.vbox.pack_start(hbox)

    answer = dialog.run()
    dialog.destroy()
    return answer

def confirm_rewrite_group(window, groupname, deleted_groupname):
    dialog = gtk.Dialog(_("Overwrite brush group?"),
                        window, gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT)

    cancel         = gtk.Button(stock=gtk.STOCK_CANCEL)
    cancel.show_all()
    img_yes        = gtk.Image()
    img_yes.set_from_stock(gtk.STOCK_YES, gtk.ICON_SIZE_BUTTON)
    img_no         = gtk.Image()
    img_no.set_from_stock(gtk.STOCK_NO, gtk.ICON_SIZE_BUTTON)
    overwrite_this = gtk.Button(_("Replace"))
    overwrite_this.set_image(img_yes)
    overwrite_this.show_all()
    skip_this      = gtk.Button(_("Rename"))
    skip_this.set_image(img_no)
    skip_this.show_all()

    buttons = [(cancel,         CANCEL),
               (skip_this,      DONT_OVERWRITE_THIS),
               (overwrite_this, OVERWRITE_THIS)]
    for button, code in buttons:
        dialog.add_action_widget(button, code)

    question = gtk.Label(_("<b>A group named `{groupname}' already exists.</b>\nDo you want to replace it, or should the new group be renamed?\nIf you replace it, the brushes may be moved to a group called `{deleted_groupname}'.").format(groupname=groupname, deleted_groupname=deleted_groupname))
    question.set_use_markup(True)

    dialog.vbox.pack_start(question)
    dialog.vbox.show_all()

    answer = dialog.run()
    dialog.destroy()
    return answer

def open_dialog(title, window, filters):
    """
    filters should be a list of tuples: (filter title, glob pattern).
    Returns a tuple: (file format, filename).
    Here "file format" is index of filter that matches filename (None if no matches).
    filename is None if no file was selected.
    """
    dialog = gtk.FileChooserDialog(title, window,
                                   gtk.FILE_CHOOSER_ACTION_OPEN,
                                   (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                    gtk.STOCK_OPEN, gtk.RESPONSE_OK))
    dialog.set_default_response(gtk.RESPONSE_OK)
    for filter_title, pattern in filters:
        f = gtk.FileFilter()
        f.set_name(filter_title)
        f.add_pattern(pattern)
        dialog.add_filter(f)

    result = (None, None)
    if dialog.run() == gtk.RESPONSE_OK:
        filename = dialog.get_filename().decode('utf-8')
        file_format = None
        for i, (_, pattern) in enumerate(filters):
            if fnmatch(filename, pattern):
                file_format = i
                break
        result = (file_format, filename)
    dialog.hide()
    return result

def save_dialog(title, window, filters, default_format=None):
    """
    filters should be a list of tuples: (filter title, glob pattern).
    default_format may be a pair (format id, suffix). That suffix will be added to filename if
    it does not match any of filters.
    Returns a tuple: (file format, filename).
    Here "file format" is index of filter that matches filename (None if no matches).
    filename is None if no file was selected.
    """
    dialog = gtk.FileChooserDialog(title, window,
                                   gtk.FILE_CHOOSER_ACTION_SAVE,
                                   (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                    gtk.STOCK_SAVE, gtk.RESPONSE_OK))
    dialog.set_default_response(gtk.RESPONSE_OK)
    dialog.set_do_overwrite_confirmation(True)

    for filter_title, pattern in filters:
        f = gtk.FileFilter()
        f.set_name(filter_title)
        f.add_pattern(pattern)
        dialog.add_filter(f)

    result = (None, None)
    while dialog.run() == gtk.RESPONSE_OK:
        filename = dialog.get_filename().decode('utf-8')
        file_format = None
        for i, (_, pattern) in enumerate(filters):
            if fnmatch(filename, pattern):
                file_format = i
                break
        if file_format is None and default_format is not None:
            file_format, suffix = default_format
            filename += suffix
            dialog.set_current_name(filename)
            dialog.response(gtk.RESPONSE_OK)
        else:
            result = (file_format, filename)
            break
    dialog.hide()
    return result

def confirm_brushpack_import(packname, window=None, readme=None):
    def show_text(text):
        tv = gtk.TextView()
        tv.set_wrap_mode(gtk.WRAP_WORD)
        tv.get_buffer().set_text(text)
        return tv

    dialog = gtk.Dialog(_("Import brush package?"),
                       window,
                       gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
                       (gtk.STOCK_CANCEL, gtk.RESPONSE_REJECT,
                        gtk.STOCK_OK, gtk.RESPONSE_ACCEPT))

    if readme:
        #readme_label = gtk.Label(_("readme.txt") % packname)
        #dialog.vbox.pack_start(readme_label)
        readme_tv = show_text(readme)
        dialog.vbox.pack_start(readme_tv)

    question = gtk.Label(_("<b>Do you really want to import package `%s'?</b>") % packname)
    question.set_use_markup(True)
    dialog.vbox.pack_start(question)
    dialog.vbox.show_all()
    answer = dialog.run()
    dialog.destroy()
    return answer


class QuickBrushChooser (gtk.VBox):
    PREFS_KEY = 'widgets.brush_chooser.selected_group'
    ICON_SIZE = 48

    class _BrushList (PixbufList):
        def __init__(self, chooser, brushes):
            s = QuickBrushChooser.ICON_SIZE
            PixbufList.__init__(self, brushes, s, s,
                                namefunc = lambda x: x.name,
                                pixbuffunc = lambda x: x.preview)
            self.chooser = chooser

        def on_select(self, brush):
            self.chooser.on_select(brush)

    def __init__(self, app, on_select):
        gtk.VBox.__init__(self)
        self.app = app
        self.on_select = on_select
        self.bm = app.brushmanager

        active_group_name = app.preferences.get(self.PREFS_KEY, None)

        model = self._make_groups_sb_model()
        self.groups_sb = spinbox.ItemSpinBox(model, self.on_groups_sb_changed,
                                             active_group_name)
        active_group_name = self.groups_sb.get_value()

        brushes = self.bm.groups[active_group_name][:]
        self.brushlist = self._BrushList(self, brushes)
        self.brushlist.dragging_allowed = False

        scrolledwin = gtk.ScrolledWindow()
        scrolledwin.set_policy(gtk.POLICY_NEVER, gtk.POLICY_ALWAYS)
        scrolledwin.add_with_viewport(self.brushlist)
        icon_size = self.ICON_SIZE
        w = icon_size * 4
        h = icon_size * 4
        scrolledwin.set_size_request(-1, h)
        self.brushlist.set_size_request(w, -1)
        scrolledwin.get_child().set_size_request(w, -1)

        self.pack_start(self.groups_sb, False, False)
        self.pack_start(scrolledwin, True, True)
        self.set_spacing(widgets.SPACING_TIGHT)

    def _make_groups_sb_model(self):
        group_names = self.bm.groups.keys()
        group_names.sort()
        model = []
        for name in group_names:
            label_text = brushmanager.translate_group_name(name)
            model.append((name, label_text))
        return model

    def update_groups_sb(self):
        model = self._make_groups_sb_model()
        self.groups_sb.set_model(model)

    def on_groups_sb_changed(self, group_name):
        self.app.preferences[self.PREFS_KEY] = group_name
        self.brushlist.itemlist[:] = self.bm.groups[group_name][:]
        self.brushlist.update()


class BrushChooserDialog (windowing.ChooserDialog):
    """Speedy brush chooser dialog.
    """

    def __init__(self, app):
        windowing.ChooserDialog.__init__(self,
          app=app, title=_("Change Brush"),
          actions=['BrushChooserPopup'],
          config_name="brushchooser")
        self._response_brush = None
        self._chooser = QuickBrushChooser(app, self._select_cb)

        # Only send the response (and close the dialog) on button release to
        # avoid accidental dabs with the stylus.
        bl = self._chooser.brushlist
        bl.connect("button-release-event", self._brushlist_button_release_cb)

        vbox = self.get_content_area()
        vbox.pack_start(self._chooser, True, True)

        self.connect("response", self._response_cb)


    def _select_cb(self, brush):
        self._response_brush = brush


    def _brushlist_button_release_cb(self, *junk):
        if self._response_brush is not None:
            self.response(gtk.RESPONSE_ACCEPT)


    def _response_cb(self, dialog, response_id):
        if response_id == gtk.RESPONSE_ACCEPT:
            bm = self.app.brushmanager
            bm.select_brush(dialog._response_brush)

