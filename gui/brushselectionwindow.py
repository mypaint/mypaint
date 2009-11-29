# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"select brush window"
import gtk, pango
gdk = gtk.gdk
from lib import brush, document
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

        self.set_title(_('Brush selection'))
        self.set_role('Brush selector')
        self.connect('delete-event', self.app.hide_window_cb)

        self.brushgroups = BrushGroupsList(self.app, self)
        self.groupselector = GroupSelector(self.app, self.brushgroups)

        #main container
        vbox = gtk.VBox()
        self.add(vbox)
        
        self.scroll = scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        scroll.add_with_viewport(self.brushgroups)
        #self.connect('configure-event', self.on_configure)
        expander = self.expander = gtk.Expander(label=_('Edit'))
        expander.set_expanded(False)

        vbox.pack_start(self.groupselector, expand=False)
        vbox.pack_start(gtk.HSeparator(), expand=False)
        vbox.pack_start(scroll, expand=True)
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

        #expanded part, right side
        l = self.brush_name_label = gtk.Label()
        l.set_justify(gtk.JUSTIFY_LEFT)
        l.set_text(_('(no name)'))
        right_vbox.pack_start(l, expand=False)

        right_vbox_buttons = [
        (_('add as new'), self.create_brush_cb),
        (_('rename...'), self.rename_brush_cb),
        (_('remove...'), self.delete_brush_cb),
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

    def create_brush_cb(self, window):
        b = brush.Brush(self.app)
        if self.app.brush:
            b.copy_settings_from(self.app.brush)
        b.preview = self.get_preview_pixbuf()
        b.save()
        active_groups = self.brushgroups.active_groups
        if active_groups:
            group = active_groups[0]
        else:
            group = DEFAULT_BRUSH_GROUP
        if group not in active_groups:
            active_groups.insert(0, group)
        self.app.brushgroups.setdefault(group, []) # create default group if needed
        self.app.brushgroups[group].insert(0, b)
        self.app.save_brushorder()
        self.brushgroups.update()
        self.groupselector.queue_draw()
        self.app.select_brush(b)

    def rename_brush_cb(self, window):
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

    def delete_brush_cb(self, window):
        # XXXX brushgroup update? Better idea: make "deleted" group mandatory! (and undeletable)
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
        self.group = groupname
        self.grouplist = grouplist
        self.brushes = self.app.brushgroups[self.group]
        pixbuflist.PixbufList.__init__(self, self.brushes, 48, 48,
                                       namefunc = lambda x: x.name,
                                       pixbuffunc = lambda x: x.preview)
    def remove_brush(self, brush):
        self.brushes.remove(brush)
        self.update()

    def insert_brush(self, idx, brush):
        self.brushes.insert(idx, brush)
        self.update()

    def on_drag_data(self, copy, source_widget, brush_name, target_idx):
        assert source_widget, 'cannot handle drag data from another app'
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

    def on_select(self, brush):
        # keep the color setting
        color = self.app.brush.get_color_hsv()
        brush.set_color_hsv(color)

        # brush changed on harddisk?
        changed = brush.reload_if_changed()
        if changed:
            self.update()
        self.app.select_brush(brush)
        #self.grouplist.set_active_group(self.group, brush)

class BrushGroupsList(gtk.VBox):
    def __init__(self, app, window):
        gtk.VBox.__init__(self)
        self.app = app
        self.parent_window = window
        self.app.brushgroups.setdefault(DEFAULT_BRUSH_GROUP, [])
        self.active_groups = [DEFAULT_BRUSH_GROUP]
        self.group_widgets = {}
        self.update()
        self.app.selected_brush_observers.append(self.brush_selected_cb)

    def update(self):
        old_widgets = self.group_widgets
        self.group_widgets = {}

        self.foreach(self.remove)

        for group in self.active_groups:
            if group in old_widgets:
                w = old_widgets[group]
            else:
                w = BrushList(self.app, self.parent_window, group, self)
            self.group_widgets[group] = w
            self.pack_start(w, expand=False, fill=False, padding=3)
            # FIXME: are we leaking memory by not calling .destroy() on unused widgets? probably not...

        self.show_all()

    def brush_selected_cb(self, brush):
        for w in self.group_widgets.itervalues():
            w.set_selected(brush)

class GroupSelector(gtk.DrawingArea):

    class GroupData:
        pass

    def __init__(self, app, brushgroups):
        gtk.DrawingArea.__init__(self)

        self.app = app
        self.brushgroups = brushgroups

        self.active_groups = brushgroups.active_groups

        self.connect("expose-event", self.expose_cb)
        self.connect("button-press-event", self.button_press_cb)
	self.set_events(gdk.EXPOSURE_MASK |
                        gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.POINTER_MOTION_MASK
                        )
        self.idx2group = {}
        self.layout = None

    def expose_cb(self, widget, event):
        cr = self.window.cairo_create()
        width, height = self.window.get_size()

        # Fill the background with gray (FIXME: gtk theme colors please)
        cr.set_source_rgb(0.7, 0.7, 0.7)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        cr.set_source_rgb(0.0, 0.0, 0.2)
        layout = cr.create_layout()
        layout.set_width(width*pango.SCALE)

        #attr = pango.AttrList()
        #attr.insert(pango.AttrBackground(0x5555, 0x5555, 0xffff, 5, 7))

        all_groups = list(sorted(self.app.brushgroups.keys()))

        idx = 0
        text = ''
        #attr = pango.AttrList()
        self.idx2group = {}
        for group in all_groups:
            s = group.encode('utf8')
            for c in s:
                self.idx2group[idx] = group
                idx += 1
            if group in self.active_groups:
                text += '<b>' + group + '</b>'
            else:
                text += group
            text += ' '
            idx += 1

        #layout.set_text(text)
        layout.set_markup(text)
        #layout.set_attributes(attr)
        cr.show_layout(layout)

        w, h = layout.get_pixel_size()
        self.set_size_request(-1, h)

        self.layout = layout

    def group_at(self, x, y):
        x, y = int(x), int(y) # avoid warning
        i, d = self.layout.xy_to_index(x*pango.SCALE, y*pango.SCALE)
        return self.idx2group.get(i)
        
    def button_press_cb(self, widget, event):
        if event.type != gdk.BUTTON_PRESS:
            return
            # ignore the extra double-click event
        group = self.group_at(event.x, event.y)
        if event.button == 1:
            if not group:
                return
            if group in self.active_groups:
                self.active_groups.remove(group)
            else:
                self.active_groups.insert(0, group)
            self.brushgroups.update()
            self.queue_draw()
        elif event.button == 3:
            menu = self.context_menu(group)
            menu.popup(None,None,None, event.button, event.time, group)

    def context_menu(self, group):
        m = gtk.Menu()
        menu = []
        menu = [ (_("New group..."), self.create_group) ]
        if group:
            menu += [ (_("Rename group..."), self.rename_group),
                      (_("Delete group..."), self.delete_group)]
        for label, callback in menu:
            mi = gtk.MenuItem(label)
            mi.connect('activate', callback, group)
            m.append(mi)
        m.show_all()
        return m

    def create_group(self, w, group):
        new_group = ask_for_name(self.get_toplevel(), _('Create group'), '')
        if new_group and new_group not in self.app.brushgroups:
            self.app.brushgroups[new_group] = []
            self.app.save_brushorder()
            self.active_groups.insert(0, new_group)
            self.brushgroups.update()
            self.queue_draw()

    def rename_group(self, w, old_group):
        new_group = ask_for_name(self.get_toplevel(), _('Rename group'), old_group)
        if new_group and new_group not in self.app.brushgroups:
            self.app.brushgroups[new_group] = self.app.brushgroups[old_group]
            del self.app.brushgroups[old_group]
            self.app.save_brushorder()
            self.brushgroups.update()
            self.queue_draw()

    def delete_group(self,w, group):
        if run_confirm_dialog(self.get_toplevel(), _('Delete group %s') % group):
            homeless_brushes = self.app.brushgroups[group]
            del self.app.brushgroups[group]
            if group in self.active_groups:
                self.active_groups.remove(group)

            for brushes in self.app.brushgroups.itervalues():
                for b2 in brushes:
                    if b2 in homeless_brushes:
                        homeless_brushes.remove(b2)

            # if the user has deleted the "deleted" group, we recreate it...?
            deleted_brushes = self.app.brushgroups.setdefault(DELETED_BRUSH_GROUP, [])
            for b in homeless_brushes:
                deleted_brushes.insert(0, b)

            self.app.save_brushorder()
            self.brushgroups.update()
            self.queue_draw()

