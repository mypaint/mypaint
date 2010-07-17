# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os
import gtk
from gettext import gettext as _

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

    hbox = gtk.HBox()
    d.vbox.pack_start(hbox)
    hbox.pack_start(gtk.Label(_('Name')))

    d.e = e = gtk.Entry()
    e.set_text(default)
    e.select_region(0, len(default))
    def responseToDialog(entry, dialog, response):  
        dialog.response(response)  
    e.connect("activate", responseToDialog, d, gtk.RESPONSE_ACCEPT)  

    hbox.pack_start(e)
    d.vbox.show_all()
    if d.run() == gtk.RESPONSE_ACCEPT:
        result = d.e.get_text()
    else:
        result = None
    d.destroy()
    return result

def error(widget, message):
    window = widget.get_toplevel()
    d = gtk.MessageDialog(window, gtk.DIALOG_MODAL, gtk.MESSAGE_ERROR, gtk.BUTTONS_OK, message)
    d.run()
    d.destroy()

def confirm_rewrite_brush(window, brushname, existing_preview_file, imported_preview_data):
    dialog = gtk.Dialog(_("Overwrite brush?"),
                        window, gtk.DIALOG_MODAL)

    cancel         = gtk.Button(stock=gtk.STOCK_CANCEL)
    cancel.show_all()
    img_yes        = gtk.Image()
    img_yes.set_from_stock(gtk.STOCK_YES, gtk.ICON_SIZE_BUTTON)
    img_no         = gtk.Image()
    img_no.set_from_stock(gtk.STOCK_NO, gtk.ICON_SIZE_BUTTON)
    overwrite_this = gtk.Button(_("Replace"))
    overwrite_this.set_image(img_yes)
    overwrite_this.show_all()
    skip_this      = gtk.Button(_("Don't replace"))
    skip_this.set_image(img_no)
    skip_this.show_all()
    overwrite_all  = gtk.Button(_("Replace all"))
    overwrite_all.show_all()
    skip_all       = gtk.Button(_("Don't replace anything"))
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
#         preview_l = gtk.Image()
    preview_r = gtk.image_new_from_file(existing_preview_file)
    label_l = gtk.Label(_("Imported brush"))
    label_r = gtk.Label(_("Existing brush"))

    question = gtk.Label(_("""<b>Brush named `%s' already exists in your collection.</b>
Are you really want to replace your brush with imported one?""" % brushname))
    question.set_use_markup(True)

    tmp_name =os.tmpnam() + '.png'
    tmp = open(tmp_name, 'w')
    tmp.write(imported_preview_data)
    tmp.close()
    preview_l = gtk.image_new_from_file(tmp_name)
    os.remove(tmp_name)

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

