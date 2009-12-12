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
from lib import document
import pixbuflist, brushcreationwidget, dialogs, brushmanager
from gettext import gettext as _

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
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
        expander.add(brushcreationwidget.Widget(app))

        vbox.pack_start(self.groupselector, expand=False)
        vbox.pack_start(gtk.HSeparator(), expand=False)
        vbox.pack_start(scroll, expand=True)
        vbox.pack_start(gtk.HSeparator(), expand=False)
        vbox.pack_start(expander, expand=False, fill=False)


class BrushList(pixbuflist.PixbufList):
    def __init__(self, app, win, groupname, grouplist):
        self.app = app
        self.bm = app.brushmanager
        self.win = win
        self.group = groupname
        self.grouplist = grouplist
        self.brushes = self.bm.groups[self.group]
        pixbuflist.PixbufList.__init__(self, self.brushes, 48, 48,
                                       namefunc = lambda x: x.name,
                                       pixbuffunc = lambda x: x.preview)
        self.bm.brushes_observers.append(self.brushes_modified_cb)

    def brushes_modified_cb(self, brushes):
        if brushes is self.brushes:
            self.update()

    def remove_brush(self, brush):
        self.brushes.remove(brush)
        for f in self.bm.brushes_observers: f(self.brushes)

    def insert_brush(self, idx, brush):
        self.brushes.insert(idx, brush)
        for f in self.bm.brushes_observers: f(self.brushes)

    def on_drag_data(self, copy, source_widget, brush_name, target_idx):
        assert source_widget, 'cannot handle drag data from another app'
        b, = [b for b in source_widget.brushes if b.name == brush_name]
        if source_widget is self:
            copy = False
        else:
            if b in self.brushes:
                source_widget.remove_brush(b)
                return True
        if not copy:
            source_widget.remove_brush(b)
        self.insert_brush(target_idx, b)
        for f in self.bm.brushes_observers: f(self.brushes)
        return True

    def on_select(self, brush):
        # keep the color setting
        color = self.app.brush.get_color_hsv()
        brush.set_color_hsv(color)

        # brush changed on harddisk?
        changed = brush.reload_if_changed()
        if changed:
            self.update()
        self.bm.select_brush(brush)

class BrushGroupsList(gtk.VBox):
    def __init__(self, app, window):
        gtk.VBox.__init__(self)
        self.app = app
        self.bm = app.brushmanager
        self.parent_window = window
        self.group_widgets = {}
        self.update()
        self.bm.selected_brush_observers.append(self.brush_selected_cb)
        self.bm.groups_observers.append(self.brushes_modified_cb)

    def brushes_modified_cb(self):
        self.update()

    def update(self):
        old_widgets = self.group_widgets
        self.group_widgets = {}

        self.foreach(self.remove)

        for group in self.bm.active_groups:
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
        self.bm = app.brushmanager
        self.bm.groups_observers.append(self.active_groups_changed_cb)
        self.brushgroups = brushgroups

        self.connect("expose-event", self.expose_cb)
        self.connect("button-press-event", self.button_press_cb)
	self.set_events(gdk.EXPOSURE_MASK |
                        gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.POINTER_MOTION_MASK
                        )
        self.idx2group = {}
        self.layout = None

    def active_groups_changed_cb(self):
        self.queue_draw()

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

        all_groups = list(sorted(self.bm.groups.keys()))

        idx = 0
        text = ''
        #attr = pango.AttrList()
        self.idx2group = {}
        for group in all_groups:
            s = group.encode('utf8')
            for c in s:
                self.idx2group[idx] = group
                idx += 1
            if group in self.bm.active_groups:
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
            if group in self.bm.active_groups:
                self.bm.active_groups.remove(group)
            else:
                self.bm.active_groups.insert(0, group)
            self.brushgroups.update()
            self.queue_draw()
        elif event.button == 3:
            menu = self.context_menu(group)
            menu.popup(None,None,None, event.button, event.time, group)

    def context_menu(self, group):
        m = gtk.Menu()
        menu = []
        menu = [ (_("New group..."), self.create_group_cb) ]
        if group:
            menu += [ (_("Rename group..."), self.rename_group_cb),
                      (_("Delete group..."), self.delete_group_cb)]
        for label, callback in menu:
            mi = gtk.MenuItem(label)
            mi.connect('activate', callback, group)
            m.append(mi)
        m.show_all()
        return m

    def create_group_cb(self, w, group):
        new_group = dialogs.ask_for_name(self.get_toplevel(), _('Create group'), '')
        if new_group:
            self.bm.create_group(new_group)

    def rename_group_cb(self, w, old_group):
        new_group = dialogs.ask_for_name(self.get_toplevel(), _('Rename group'), old_group)
        # TODO: complain if target exists 
        if new_group and new_group not in self.bm.groups:
            self.bm.rename_group(old_group, new_group)

    def delete_group_cb(self, w, group):
        # TODO: complain if group == DELETED_BRUSH_GROUP
        if dialogs.confirm(self.get_toplevel(), _('Delete group %s') % group):
            self.bm.delete_group(group)
