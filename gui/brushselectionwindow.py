# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"select brush window"
import os
import gtk
gdk = gtk.gdk
from math import floor
import zipfile
import tempfile
from lib import brush, document
from lib.helpers import to_unicode
import tileddrawwidget, pixbuflist
from gettext import gettext as _

# not translatable for now (this string is saved into a file and would screw up between language switches)
DEFAULT_BRUSH_GROUP = 'default'
DELETED_BRUSH_GROUP = 'deleted'

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.app.selected_brush_observers.append(self.brush_selected_cb)
        self.app.brush.settings_observers.append(self.brush_modified_cb)
        self.app.kbm.add_window(self)
        self.last_selected_brush = None

        self.disable_selection_callback = True
        self.selected_ok = False
        self.new_brush = None

        self.brushgroups = BrushGroupsList(self.app, self)
        self.set_title(_('Brush selection'))
        self.set_role('Brush selector')
        self.connect('delete-event', self.app.hide_window_cb)

        #main container
        vbox = gtk.VBox()
        self.add(vbox)
        
        self.scroll = scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        scroll.add_with_viewport(self.brushgroups)
        #self.connect('configure-event', self.on_configure)
        expander = self.expander = gtk.Expander(label=_('Edit'))
        expander.set_expanded(False)

        vbox.pack_start(scroll)
        vbox.pack_start(gtk.HSeparator(), expand=False)
        vbox.pack_start(expander, expand=False, fill=False)

        #expanded part
        vbox2 = gtk.VBox()
        hbox = gtk.HBox()
        hbox.set_border_width(8)
        vbox2.pack_start(hbox, expand=True)
        update_button = gtk.Button(stock=gtk.STOCK_REFRESH) # FIXME: remove?
        update_button.connect('clicked', self.update_cb)
        vbox2.pack_start(update_button, expand=False)
        expander.add(vbox2)

        left_vbox = gtk.VBox()
        right_vbox = gtk.VBox()
        hbox.pack_start(left_vbox, expand=False, fill=False)
        hbox.pack_end(right_vbox, expand=False, fill=False)

        #expanded part, left side
        doc = document.Document()
        self.tdw = tileddrawwidget.TiledDrawWidget(doc)
        self.tdw.set_size_request(brush.preview_w, brush.preview_h)
        left_vbox.pack_start(self.tdw, expand=False, fill=False)

        b = gtk.Button(_('Clear'))
        def clear_cb(window):
            self.tdw.doc.clear_layer()
        b.connect('clicked', clear_cb)
        left_vbox.pack_start(b, expand=False, padding=5)

        #vbox2a = gtk.VBox()
        #hbox.pack_end(vbox2a, expand=True, fill=True, padding=5)
        #l = self.brush_name_label = gtk.Label()
        #l.set_justify(gtk.JUSTIFY_LEFT)
        #vbox2a.pack_start(l, expand=False)
        #tv = self.brush_info_textview = gtk.TextView()
        #vbox2a.pack_start(tv, expand=True)

        #expanded part, right side
        l = self.brush_name_label = gtk.Label()
        l.set_justify(gtk.JUSTIFY_LEFT)
        l.set_text(_('(no name)'))
        right_vbox.pack_start(l, expand=False)

        right_vbox_buttons = [
        (_('add as new'), self.add_as_new_cb),
        (_('rename...'), self.rename_cb),
        (_('remove...'), self.delete_selected_cb),
        (_('settings...'), self.brush_settings_cb),
        (_('save settings'), self.update_settings_cb),
        (_('save preview'), self.update_preview_cb),
        ]

        for title, clicked_cb in right_vbox_buttons:
            b = gtk.Button(title)
            b.connect('clicked', clicked_cb)
            right_vbox.pack_start(b, expand=False)

    def set_preview_pixbuf(self, pixbuf):
        if pixbuf is None:
            self.tdw.doc.clear()
        else:
            self.tdw.doc.load_from_pixbuf(pixbuf)

    def get_preview_pixbuf(self):
        pixbuf = self.tdw.doc.render_as_pixbuf(0, 0, brush.preview_w, brush.preview_h)
        return pixbuf

    def brush_settings_cb(self, window):
        w = self.app.brushSettingsWindow
        w.show_all() # might be for the first time
        w.present()

    def add_as_new_cb(self, window):
        b = brush.Brush(self.app)
        if self.app.brush:
            b.copy_settings_from(self.app.brush)
        b.preview = self.get_preview_pixbuf()
        b.save()
        group = self.brushgroups.active_group or DEFAULT_BRUSH_GROUP
        self.app.brushgroups.setdefault(group, []) # create default group if needed
        self.app.brushgroups[group].insert(0, b)
        self.app.select_brush(b)
        self.app.save_brushorder()
        self.brushgroups.update()

    def rename_cb(self, window):
        b = self.app.selected_brush
        if b is None or not b.name:
            display = gtk.gdk.display_get_default()
            display.beep()
            return

        name = ask_for_name(self, _("Rename Brush"), b.name.replace('_', ' '))
        if not name:
            return
        name = name.replace(' ', '_')
        print 'renaming brush', repr(b.name), '-->', repr(name)
        # ensure we don't overwrite an existing brush by accident
        for groupname, brushes in self.app.brushgroups.iteritems():
            if groupname == DELETED_BRUSH_GROUP:
                continue
            for b2 in brushes:
                if b2.name == name:
                    print 'Target already exists!'
                    display = gtk.gdk.display_get_default()
                    display.beep()
                    return
        success = b.delete_from_disk()
        old_name = b.name
        b.name = name
        b.save()
        if not success:
            # we are renaming a stock brush
            # we can't delete the original; instead we put it away so it doesn't reappear
            old_brush = brush.Brush(self.app)
            old_brush.load(old_name)
            deleted_brushes = self.app.brushgroups.setdefault(DELETED_BRUSH_GROUP, [])
            deleted_brushes.insert(0, old_brush)

        self.app.select_brush(b)
        self.app.save_brushorder()
        self.brushgroups.update()

    def update_preview_cb(self, window):
        pixbuf = self.get_preview_pixbuf()
        b = self.app.selected_brush
        if b is None or not b.name:
            # no brush selected
            display = gtk.gdk.display_get_default()
            display.beep()
            return
        b.preview = pixbuf
        b.save()
        self.brushgroups.update()

    def update_settings_cb(self, window):
        b = self.app.selected_brush
        if b is None or not b.name:
            # no brush selected
            display = gtk.gdk.display_get_default()
            display.beep()
            return
        b.copy_settings_from(self.app.brush)
        b.save()

    def delete_selected_cb(self, window):
        b = self.app.selected_brush
        if b is None or not b.name: return
        if not run_confirm_dialog(self, _("Really delete brush from disk?")):
            return

        self.app.select_brush(None)

        for brushes in self.app.brushgroups.itervalues():
            if b in brushes:
                brushes.remove(b)
        if not b.delete_from_disk():
            # stock brush can't be deleted
            deleted_brushes = self.app.brushgroups.setdefault(DELETED_BRUSH_GROUP, [])
            deleted_brushes.insert(0, b)

        self.app.save_brushorder()
        self.brushgroups.update()

    def brush_selected_cb(self, brush):
        if brush is None: return
        name = brush.name
        if name is None:
            name = _('(no name)')
        else:
            name = name.replace('_', ' ')
        self.brush_name_label.set_text(name)

    def update_brush_preview(self, brush):
        self.set_preview_pixbuf(brush.preview)
        self.last_selected_brush = brush

    def brush_modified_cb(self):
        self.tdw.doc.set_brush(self.app.brush)

    def update_cb(self, button):
        self.update()

    def update(self):
        brush = self.app.brush
        callbacks = self.app.selected_brush_observers
        self.app.selected_brush_observers = callbacks
        self.brushgroups.update()

def ask_for_name(window, title, default):
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

def run_confirm_dialog(window, title):
    d = gtk.Dialog(title,
         window,
         gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
         (gtk.STOCK_YES, gtk.RESPONSE_ACCEPT,
          gtk.STOCK_NO, gtk.RESPONSE_REJECT))
    response = d.run()
    d.destroy()
    return response == gtk.RESPONSE_ACCEPT

class BrushList(pixbuflist.PixbufList):
    def __init__(self, app, win, groupname, grouplist):
        self.app = app
        self.win = win
        self.app.selected_brush_observers.append(self.brush_selected_cb)
        self.group = groupname
        self.grouplist = grouplist
        self.device = None
        self.brushes = self.app.brushgroups[self.group]
        pixbuflist.PixbufList.__init__(self, self.brushes, 48, 48,
                                       namefunc = lambda x: x.name,
                                       pixbuffunc = lambda x: x.preview)
        #self.connect('item-activated', self.on_item_activated)
        self.app.drawWindow.tdw.device_observers.append(self.on_device_change)

        self.connect('drag-data-get', self.drag_data_get)
        self.connect('drag-data-received', self.drag_data_received)
        self.connect('motion-notify-event', self.motion_notify)
        self.connect('button-press-event', self.on_button_press)
        self.connect('button-release-event', self.on_button_release)

    def remove_brush(self, brush):
        self.brushes.remove(brush)
        self.update()

    def insert_brush(self, idx, brush):
        self.brushes.insert(idx, brush)
        self.update()

    def on_drag_data(self, copy, source_widget, brush_name, target_idx):
        assert source_widget, 'cannot handle drag data from another app'
        print 'Dragging brush', brush_name, 'between', source_widget.group, '-->', self.group
        b, = [b for b in source_widget.brushes if b.name == brush_name]
        if source_widget is self:                  # If brush dragged from same widget
            copy = False
        else:
            if b in self.brushes:
                source_widget.remove_brush(b)
                return True
        if not copy:
            source_widget.remove_brush(b)
        self.grouplist.active_group = self.group # hm... could use some self.select() method somewhere?
        self.insert_brush(target_idx, b)
        self.app.save_brushorder()
        return True

    def on_device_change(self, old_device, new_device):
        self.device = new_device.source

    def on_select(self, brush):
        if self.win.disable_selection_callback:
            return
        self.win.disable_selection_callback = True
        self.win.selected_ok = False
        # keep the color setting
        color = self.app.brush.get_color_hsv()
        brush.set_color_hsv(color)

        # brush changed on harddisk?
        changed = brush.reload_if_changed()
        if changed:
            self.update()

        if not self.win.selected_ok:
            self.app.eraser_mode = False
            if self.device is not None:
                self.app.brush_by_device[self.device] = brush
            self.app.select_brush(brush)
        self.win.disable_selection_callback = False
        self.grouplist.set_active_group(self.group, brush)

    def brush_selected_cb(self, brush):
        if self.win.selected_ok:
            return
        self.win.selected_ok = self.set_selected(brush)
def child_brushlist(expander):
    return expander.child

class Expander(gtk.Expander):
    def __init__(self,label, groupname):
        gtk.Expander.__init__(self, label)
        self.add_events(gdk.BUTTON_PRESS_MASK)
        self.connect('button-press-event', self.on_button_press)
        self.groupname = groupname
        self.popup_menu = None # gets overwritten

    def on_button_press(self, w, event):
        if event.button == 3:
            menu = self.popup_menu(self.groupname)
            if not menu:
                return
            menu.popup(None,None,None, 3, event.time)

class BrushGroupsList(gtk.VBox):
    def __init__(self, app, window):
        gtk.VBox.__init__(self)
        self.app = app
        self.parent_window = window
        #self.active_group = to_unicode( app.get_config('State', 'active_group'))
        self.active_group = DEFAULT_BRUSH_GROUP
        self.update(init=True)

    def update(self,init=False):
        if not init:
            for group, expander in self.group_list.iteritems():
                self.app.selected_brush_observers.remove(child_brushlist(expander).brush_selected_cb)
            self.foreach(self.remove)
        self.group_list = {}            # group name -> list of Expander's
        b = self.app.selected_brush
        # Set DEFAULT_BRUSH_GROUP to be first in the list
        tmp = list(sorted(self.app.brushgroups.keys()))
        if DEFAULT_BRUSH_GROUP in tmp:
            tmp.remove(DEFAULT_BRUSH_GROUP)
            tmp.insert(0, DEFAULT_BRUSH_GROUP)

        for groupname in tmp:
            expander = Expander(groupname, groupname)
            brushlist = BrushList(self.app, self.parent_window, groupname, self)
            #if self.active_group == group:
            #    brushlist.set_selected(b)
            expander.add(brushlist)
            expander.set_expanded(False)
            expander.popup_menu = self.expander_menu
            self.pack_start(expander, expand=False)
            self.group_list[groupname] = expander
        if self.active_group in self.group_list:
            self.group_list[self.active_group].set_expanded(True)
        elif DEFAULT_BRUSH_GROUP in self.group_list:
            self.group_list[DEFAULT_BRUSH_GROUP].set_expanded(True)
        self.show_all()

    def expander_menu(self, groupname):
        m = gtk.Menu()
        menu = [ (_("New group..."), self.create_group),
                 (_("Rename group..."), self.rename_group),
                 (_("Delete group..."), self.delete_group)]
        for label, callback in menu:
            mi = gtk.MenuItem(label)
            mi.connect('activate', callback, groupname)
            m.append(mi)
        m.show_all()
        return m

    def create_group(self, w, active_group):
        new_group = ask_for_name(self.parent_window, _('Create group'), '')
        if new_group and new_group not in self.app.brushgroups:
            self.app.brushgroups[new_group] = []
            self.app.save_brushorder()
            self.update()

    def rename_group(self, w, old_group):
        new_group = ask_for_name(self.parent_window, _('Rename group'), old_group)
        if new_group and new_group not in self.app.brushgroups:
            self.app.brushgroups[new_group] = self.app.brushgroups[old_group]
            del self.app.brushgroups[old_group]
            self.app.save_brushorder()
            self.update()

    def delete_group(self,w, group):
        if run_confirm_dialog(self.parent_window, _('Delete group %s') % group):
            homeless_brushes = self.app.brushgroups[group]
            del self.app.brushgroups[group]

            for groupname, brushes in self.app.brushgroups.iteritems():
                for b2 in brushes:
                    if b2 in homeless_brushes:
                        homeless_brushes.remove(b2)

            deleted_brushes = self.app.brushgroups.setdefault(DELETED_BRUSH_GROUP, [])
            for b in homeless_brushes:
                deleted_brushes.insert(0, b)

            self.app.save_brushorder()
            self.update()

    def set_active_group(self, name, brush):
        def expand(e):
            if not e.get_expanded():
                e.set_expanded(True)
        self.active_group = name
        for group, expander in self.group_list.iteritems():
            if group!=name:
                child_brushlist(expander).set_selected(None)
            else:
                child_brushlist(expander).set_selected(brush)
#                expand(expander)
            child_brushlist(expander).queue_draw()
