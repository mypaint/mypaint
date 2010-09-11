# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"tune brush window"
from gettext import gettext as _
import gtk
import functionwindow, brushcreationwidget
import windowing
from brushlib import brushsettings
from lib import command

class Window(windowing.SubWindow):
    def __init__(self, app):
        windowing.SubWindow.__init__(self, app)
        self.app.brushmanager.selected_brush_observers.append(self.brush_selected_cb)

        self.set_title(_('Brush settings'))
        self.connect('delete-event', self.app.hide_window_cb)

        vbox = gtk.VBox()
        self.add(vbox)

        # Expander with brushcreation widget under it
        expander = self.expander = gtk.Expander(label=_('Edit and save brush'))
        expander.set_expanded(False)
        expander.add(brushcreationwidget.Widget(app))

        vbox.pack_end(expander, expand=False, fill=False)

        cb = self.live_update = gtk.CheckButton(_('Live update the last canvas stroke'))
        vbox.pack_start(cb, expand=False, fill=True, padding=5)
        cb.connect('toggled', self.live_update_cb)
        self.app.brush.settings_observers.append(self.live_update_cb)

        scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        vbox.pack_start(scroll, expand=True, fill=True)

        table = gtk.Table(4, len(brushsettings.settings_visible))
        #table.set_border_width(4)
        #table.set_col_spacings(15)
        scroll.add_with_viewport(table)

        self.adj = {}
        for i, s in enumerate(brushsettings.settings_visible):
            l = gtk.Label(s.name)
            l.set_alignment(0, 0.5)
            l.set_tooltip_text(s.tooltip)

            adj = self.app.brush_adjustment[s.cname]
            adj.connect('value-changed', self.value_changed_cb, s.index, self.app)
            self.adj[s] = adj
            h = gtk.HScale(adj)
            h.set_digits(2)
            h.set_draw_value(True)
            h.set_value_pos(gtk.POS_LEFT)

            b = gtk.Button("%.2f" % s.default)
            b.connect('clicked', self.set_fixed_value_clicked_cb, adj, s.default)
            b.set_tooltip_text(_('Reset to default value'))
            adj.default_value_button = b

            if s.constant:
                b2 = gtk.Label("")
                b2.set_tooltip_text(_("No additional configuration"))
                b2.set_alignment(0.5, 0.5)
                adj.three_dots_button = None
            else:
                b2 = gtk.Button("...")
                b2.set_tooltip_text(_("Add input values mapping"))
                b2.connect('clicked', self.details_clicked_cb, adj, s)
                adj.three_dots_button = b2

            table.attach(l, 0, 1, i, i+1, gtk.FILL, gtk.FILL, 5, 0)
            table.attach(h, 1, 2, i, i+1, gtk.EXPAND | gtk.FILL, gtk.EXPAND | gtk.FILL)
            table.attach(b, 2, 3, i, i+1, gtk.FILL, gtk.FILL)
            table.attach(b2, 3, 4, i, i+1, gtk.FILL, gtk.FILL)

        self.functionWindows = {}

        self.set_default_size(450, 500)

        self.update_settings()

    def set_fixed_value_clicked_cb(self, widget, adj, value):
        adj.set_value(value)

    def details_clicked_cb(self, window, adj, setting):
        # FIXME: should the old window get closed automatically?
        #        Hm... probably not.
        w = self.functionWindows.get(setting)
        if w is None:
            w = functionwindow.Window(self.app, setting, adj)
            self.functionWindows[setting] = w
            w.show_all()
            bsw_pos = self.get_position()
            if bsw_pos:
                x, y = bsw_pos
                x += 16
                y = max(0, y-50)
                w.move(x, y)
        w.present() # get to the front

    def value_changed_cb(self, adj, index, app):
        setting = [k for k, v in self.adj.items() if v == adj][0]
        s = app.brush.settings[index]
        s.set_base_value(adj.get_value())
        self.relabel_setting_buttons(adj, setting, s)

    def update_settings(self):
        """Update all settings; their value and button labels"""
        for setting in brushsettings.settings_visible:
            adj = self.adj[setting]
            s = self.app.brush.settings[setting.index]
            self.relabel_setting_buttons(adj, setting, s)
            adj.set_value(s.base_value)

    def relabel_setting_buttons(self, adj, setting, brushsetting):
        """Relabel the buttons of a setting"""
        s = brushsetting

        # Make "input value mapping" button reflect if this brush
        # allready has a mapping or not
        if adj.three_dots_button:
            def set_label(s, t):
                if adj.three_dots_button.get_label() == s: return
                adj.three_dots_button.set_label(s)
                adj.three_dots_button.set_tooltip_text(t)
            if s.has_only_base_value():
                set_label("...", _("Add input values mapping"))
            else:
                set_label("X", _("Modify input values mapping"))

        # Make "reset to default value" button insensitive
        # if the value is already the default (the button will have no effect)
        if adj.get_value() == setting.default:
            adj.default_value_button.set_sensitive(False)
        else:
            adj.default_value_button.set_sensitive(True)

    def brush_selected_cb(self, brush_selected):
        self.update_settings()

    def live_update_cb(self, *trash):
        if self.live_update.get_active():
            doc = self.app.doc.model
            cmd = 'something'
            while cmd:
                cmd = doc.undo()
                if isinstance(cmd, command.Stroke):
                    # found it
                    # bad design that we need to touch internals document.py here...
                    new_stroke = cmd.stroke.copy_using_different_brush(self.app.brush)
                    snapshot_before = doc.layer.save_snapshot()
                    new_stroke.render(doc.layer.surface)
                    doc.do(command.Stroke(doc, new_stroke, snapshot_before))
                    break

