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

        # TODO: evaluate glade/gazpacho, the code below is getting scary

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
        b.copy_settings_from(self.app.brush)
        b.groups = [DEFAULT_BRUSH_GROUP]
        b.preview = self.get_preview_pixbuf()
        TODO
        self.app.brushes.insert(0, b)
        #still needed?:
        self.brushlist.update()
        self.app.select_brush(b)
        b.save()
        self.brushgroups.add_brush_to_group(b, DEFAULT_BRUSH_GROUP)
        self.app.save_brushorder()
        self.brushgroups.update()

    def rename_cb(self, window):
        b = self.app.selected_brush
        if b is None or not b.name:
            display = gtk.gdk.display_get_default()
            display.beep()
            return

        d = gtk.Dialog(_("Rename Brush"),
                       self,
                       gtk.DIALOG_MODAL,
                       (gtk.STOCK_CANCEL, gtk.RESPONSE_REJECT,
                        gtk.STOCK_OK, gtk.RESPONSE_ACCEPT))

        hbox = gtk.HBox()
        d.vbox.pack_start(hbox)
        hbox.pack_start(gtk.Label('Name'))

        e = gtk.Entry()
        e.set_text(b.name.replace('_', ' '))
        e.select_region(0, len(b.name))
        def responseToDialog(entry, dialog, response):  
            dialog.response(response)  
        e.connect("activate", responseToDialog, d, gtk.RESPONSE_ACCEPT)  

        hbox.pack_start(e)
        d.vbox.show_all()
        if d.run() == gtk.RESPONSE_ACCEPT:
            new_name = e.get_text().replace(' ', '_')
            print 'renaming brush', repr(b.name), '-->', repr(new_name)
            TODO
            if [True for x in self.app.brushes if x.name == new_name]:
                print 'Target already exists!'
                display = gtk.gdk.display_get_default()
                display.beep()
                d.destroy()
                return
            b.delete_from_disk()
            b.name = new_name
            b.save()
            self.app.select_brush(b)
            self.app.save_brushorder()
        d.destroy()

    def update_preview_cb(self, window):
        pixbuf = self.get_preview_pixbuf()
        b = self.app.selected_brush
        if b is None:
            # no brush selected
            display = gtk.gdk.display_get_default()
            display.beep()
            return
        b.preview = pixbuf
        b.save()
        self.brushgroups.update()

    def update_settings_cb(self, window):
        b = self.app.selected_brush
        if b is None:
            # no brush selected
            display = gtk.gdk.display_get_default()
            display.beep()
            return
        b.copy_settings_from(self.app.brush)
        b.save()
        # TODO: call this somewhere, but not here: self.app.save_brushorder()

    def delete_selected_cb(self, window):
        b = self.app.selected_brush
        if b is None: return
        d = confirm_dialog(self, _("Really delete this brush?"))
        response = d.run()
        d.destroy()
        if response != gtk.RESPONSE_ACCEPT: return

        self.app.select_brush(None)
        TODO
        self.app.brushes.remove(b)
        b.delete_from_disk()
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

    #def get_brushes_dict(self):
    #    return self.brushgroups.groups
#
#    def get_brushes(self):
#        res = []
#        bg = self.brushgroups
#        for group in sorted(bg.groups):
#            brushes = bg.groups[group]
#            for b in brushes:
#                if b not in res:
#                    res.append(b)
#        return res

    def index_of(self, brush):
        return self.get_brushes().index(brush)

def rename_dialog(window, title, default):
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
    return d

def confirm_dialog(window, title):
    d = gtk.Dialog(title,
         window,
         gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
         (gtk.STOCK_YES, gtk.RESPONSE_ACCEPT,
          gtk.STOCK_NO, gtk.RESPONSE_REJECT))
    return d

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
    def __init__(self,label, data):
        gtk.Expander.__init__(self, label)
        self.add_events(gdk.BUTTON_PRESS_MASK | gdk.BUTTON_RELEASE_MASK)
        self.connect('button-press-event', self.on_button_press)
        self.connect('button-release-event', self.on_button_release)
        self.data = data
        self.button_pressed = False

    def popup_menu(self, data):
        return None

    def on_button_press(self, w, event):
        if event.button == 3:
            self.button_pressed = True

    def on_button_release(self, w, event):
        if self.button_pressed and event.button == 3:
            menu = self.popup_menu(self.data)
            if not menu:
                return
            menu.popup(None,None,None, 3, event.time)
        self.button_pressed = False

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
        tmp.remove(DEFAULT_BRUSH_GROUP)
        tmp.insert(0, DEFAULT_BRUSH_GROUP)

        for groupname in tmp:
            expander = Expander(groupname, None) # huh?
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
        else:
            self.group_list[DEFAULT_BRUSH_GROUP].set_expanded(True)
        self.show_all()

    def expander_menu(self, groupname):
        m = gtk.Menu()
        menu = [(_("Rename group..."), self.rename_group),
                (_("Delete group..."), self.delete_group)]
        for label, callback in menu:
            mi = gtk.MenuItem(label)
            mi.connect('activate', callback, groupname)
            m.append(mi)
        m.show_all()
        return m

    def rename_group(self, w, old_group):
        d = rename_dialog(self.parent_window, _('Rename group'), old_group)
        if d.run() == gtk.RESPONSE_ACCEPT:
            new_group = d.e.get_text()
            if not new_group or new_group==old_group:
                return
            for b in self.app.brushes:
                if old_group in b.groups:
                    self.del_brush_from_group(b, old_group)
                    self.add_brush_to_group(b, new_group)
                    b.save()
            try:
                del self.app.brushgroups[old_group]
            except KeyError:
                pass
            self.update()
        d.destroy()

    def delete_group(self,w, group):
        d = confirm_dialog(self.parent_window, _('Delete group %s') % group)
        if d.run() == gtk.RESPONSE_ACCEPT:
            for b in self.app.brushes:
                if group in b.groups:
                    self.del_brush_from_group(b, group)
                    b.save()
            self.update()
        d.destroy()

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
