# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

"tune brush window"
import gtk
import functionwindow
import brush, brushsettings

class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app
        self.app.brush_selected_callbacks.append(self.brush_selected_cb)

        self.set_title('Brush settings')
        self.connect('delete-event', self.app.hide_window_cb)

        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        self.add(scroll)

        table = gtk.Table(4, len(brushsettings.settings_visible))
        #table.set_border_width(4)
        #table.set_col_spacings(15)
        scroll.add_with_viewport(table)

        self.adj = {}
        self.app.brush_adjustment = {}
        for i, s in enumerate(brushsettings.settings_visible):
            l = gtk.Label(s.name)
            l.set_alignment(0, 0.5)
            l.set_tooltip_text(s.tooltip)

            adj = gtk.Adjustment(value=s.default, lower=s.min, upper=s.max, step_incr=0.01, page_incr=0.1)
            adj.connect('value-changed', self.value_changed_cb, s.index, self.app)
            self.adj[s] = adj
            self.app.brush_adjustment[s.cname] = adj
            h = gtk.HScale(adj)
            h.set_digits(2)
            h.set_draw_value(True)
            h.set_value_pos(gtk.POS_LEFT)

            #sb = gtk.SpinButton(adj, climb_rate=0.1, digits=2)
            b = gtk.Button("%.1f" % s.default)
            b.connect('clicked', self.set_fixed_value_clicked_cb, adj, s.default)

            if s.constant:
                b2 = gtk.Label("=")
                b2.set_alignment(0.5, 0.5)
                adj.three_dots_button = None
            else:
                b2 = gtk.Button("...")
                b2.connect('clicked', self.details_clicked_cb, adj, s)
                adj.three_dots_button = b2

            table.attach(l, 0, 1, i, i+1, gtk.FILL, gtk.FILL, 5, 0)
            table.attach(h, 1, 2, i, i+1, gtk.EXPAND | gtk.FILL, gtk.EXPAND | gtk.FILL)
            table.attach(b, 2, 3, i, i+1, gtk.FILL, gtk.FILL)
            table.attach(b2, 3, 4, i, i+1, gtk.FILL, gtk.FILL)

        self.functionWindows = {}

        self.set_size_request(450, 500)

        self.relabel_buttons()

    def set_fixed_value_clicked_cb(self, widget, adj, value):
        adj.set_value(value);

    def details_clicked_cb(self, window, adj, setting):
        # FIXME: should the old window get closed automatically?
        #        Hm... probably not.
        w = self.functionWindows.get(setting)
        if w is None:
            w = functionwindow.Window(self.app, setting, adj)
            self.functionWindows[setting] = w
            w.show_all()
        w.present() # get to the front

    def value_changed_cb(self, adj, index, app):
        app.brush.settings[index].set_base_value(adj.get_value())

    def relabel_buttons(self):
        for s in brushsettings.settings_visible:
            adj = self.adj[s]
            s = self.app.brush.settings[s.index]
            adj.set_value(s.base_value)
            if adj.three_dots_button:
                def set_label(s):
                    if adj.three_dots_button.get_label() == s: return
                    adj.three_dots_button.set_label(s)
                if s.has_only_base_value():
                    set_label("...")
                else:
                    set_label("X")

    def brush_selected_cb(self, brush_selected):
        self.relabel_buttons()


