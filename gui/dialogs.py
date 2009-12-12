# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
from gettext import gettext as _

def confirm(window, title):
    if not isinstance(window, gtk.Window):
        window = window.get_toplevel()
    d = gtk.Dialog(title,
         window,
         gtk.DIALOG_MODAL,
         (gtk.STOCK_YES, gtk.RESPONSE_ACCEPT,
          gtk.STOCK_NO, gtk.RESPONSE_REJECT))
    response = d.run()
    d.destroy()
    return response == gtk.RESPONSE_ACCEPT


def ask_for_name(window, title, default):
    if not isinstance(window, gtk.Window):
        window = window.get_toplevel()
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

def error(window, message):
    if not isinstance(window, gtk.Window):
        window = window.get_toplevel()
    d = gtk.MessageDialog(window, gtk.DIALOG_MODAL, gtk.MESSAGE_ERROR, gtk.BUTTONS_OK, message)
    d.run()
    d.destroy()

