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
import pixbuflist, brushcreationwidget, dialogs
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

        self.groupselector = GroupSelector(self.app)
        self.brushgroups = BrushGroupsList(self.app)

        #main container
        vbox = gtk.VBox()
        self.add(vbox)
        
        self.scroll = scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        scroll.add_with_viewport(self.brushgroups)
        expander = self.expander = gtk.Expander(label=_('Edit'))
        expander.set_expanded(False)
        expander.add(brushcreationwidget.Widget(app))

        vbox.pack_start(self.groupselector, expand=False)
        vbox.pack_start(gtk.HSeparator(), expand=False)
        vbox.pack_start(scroll, expand=True)
        vbox.pack_start(gtk.HSeparator(), expand=False)
        vbox.pack_start(expander, expand=False, fill=False)


class BrushList(pixbuflist.PixbufList):
    def __init__(self, app, group):
        self.app = app
        self.bm = app.brushmanager
        self.brushes = self.bm.groups[group]
        pixbuflist.PixbufList.__init__(self, self.brushes, 48, 48,
                                       namefunc = lambda x: x.name,
                                       pixbuffunc = lambda x: x.preview)
        self.set_selected(self.bm.selected_brush)
        self.bm.brushes_observers.append(self.brushes_modified_cb)
        self.bm.selected_brush_observers.append(self.brush_selected_cb)

    def brushes_modified_cb(self, brushes):
        if brushes is self.brushes:
            self.update()

    def brush_selected_cb(self, brush):
        self.set_selected(brush)

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
        if brush.reload_if_changed():
            for brushes in self.bm.groups.itervalues():
                for f in self.bm.brushes_observers: f(brushes)

        self.bm.select_brush(brush)

class BrushGroupsList(gtk.VBox):
    def __init__(self, app):
        gtk.VBox.__init__(self)
        self.app = app
        self.bm = app.brushmanager
        self.group_widgets = {}
        self.update()
        self.bm.groups_observers.append(self.groups_modified_cb)

    def groups_modified_cb(self):
        self.update()

    def update(self):
        for group in self.group_widgets.keys():
            if group not in self.bm.groups:
                # this leaks around 2MB of memory, not sure why
                # (no real problem, this is only when deleting/renaming groups)
                del self.group_widgets[group]

        self.foreach(self.remove)

        for group in self.bm.active_groups:
            if group in self.group_widgets:
                w = self.group_widgets[group]
            else:
                w = BrushList(self.app, group)
            self.group_widgets[group] = w
            self.pack_start(w, expand=False, fill=False, padding=3)

        self.show_all()

class GroupSelector(gtk.DrawingArea):
    def __init__(self, app):
        gtk.DrawingArea.__init__(self)

        self.app = app
        self.bm = app.brushmanager
        self.bm.groups_observers.append(self.active_groups_changed_cb)

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

        style = self.get_style()

        c = style.bg[gtk.STATE_NORMAL]
        c_floats = [float(c.red)/65535, float(c.green)/65535, float(c.blue)/65535]
        cr.set_source_rgb(*c_floats)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        c = style.text[gtk.STATE_NORMAL]
        c_floats = [float(c.red)/65535, float(c.green)/65535, float(c.blue)/65535]
        cr.set_source_rgb(*c_floats)
        layout = cr.create_layout()
        layout.set_width(width*pango.SCALE)

        all_groups = list(sorted(self.bm.groups.keys()))

        idx = 0
        text = ''
        attr = pango.AttrList()
        self.idx2group = {}
        for group in all_groups:
            s = group.encode('utf8')
            idx_start = idx
            for c in s:
                self.idx2group[idx] = group
                idx += 1
            if group in self.bm.active_groups:
                # those colors create too much distraction:
                #c = style.bg[gtk.STATE_SELECTED]
                #attr.insert(pango.AttrBackground(c.red, c.green, c.blue, idx_start, idx))
                #c = style.text[gtk.STATE_SELECTED]
                #attr.insert(pango.AttrForeground(c.red, c.green, c.blue, idx_start, idx))
                attr.insert(pango.AttrWeight(pango.WEIGHT_BOLD, idx_start, idx))

            text += group + ' '
            idx += 1

        layout.set_text(text)
        layout.set_attributes(attr)
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
            return # ignore the extra double-click event
        group = self.group_at(event.x, event.y)
        if event.button == 1:
            if not group:
                return
            if group in self.bm.active_groups:
                self.bm.active_groups.remove(group)
            else:
                self.bm.active_groups.insert(0, group)
            for f in self.bm.groups_observers: f()
        elif event.button == 3:
            menu = self.context_menu(group)
            menu.popup(None,None,None, event.button, event.time, group)

    def context_menu(self, group):
        m = gtk.Menu()
        menu = []
        menu += [ (_("New group..."), self.create_group_cb) ]
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
        new_group = dialogs.ask_for_name(self, _('Create group'), '')
        if new_group:
            self.bm.create_group(new_group)

    def rename_group_cb(self, w, old_group):
        new_group = dialogs.ask_for_name(self, _('Rename group'), old_group)
        if not new_group:
            return
        if new_group not in self.bm.groups:
            self.bm.rename_group(old_group, new_group)
        else:
            dialogs.error(self, _('A group with this name already exists!'))

    def delete_group_cb(self, w, group):
        if dialogs.confirm(self, _('Really delete group %s?') % group):
            self.bm.delete_group(group)
            if group in self.bm.groups:
                dialogs.error(self, _('This group can not be deleted (try to make it empty first).'))
