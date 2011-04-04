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

    PAGE_BRUSHSETTINGS = 0
    PAGE_BRUSHINPUTS = 1

    def __init__(self, app):
        windowing.SubWindow.__init__(self, app, key_input=True)
        self.app.brushmanager.selected_brush_observers.append(self.brush_selected_cb)

        self.adj = {}
        self.functionWindows = {}
        # A list of all brushsettings (cname) which are to be displayed
        self.visible_settings = []

        self.set_title(_('Brush Editor'))
        self.init_ui()
        self.set_default_size(450, 500)

        self.update_settings()

    def init_ui(self):
        """Construct and pack widgets."""
        vbox = gtk.VBox()
        self.add(vbox)

        # Expander with brushcreation widget under it
        expander = self.expander = gtk.Expander(label=_('Edit brush icon'))
        expander.set_expanded(False)
        brushicon_editor = brushcreationwidget.BrushIconEditorWidget(self.app)
        expander.add(brushicon_editor)
        vbox.pack_end(expander, expand=False, fill=False)

        # Header with brush name and actions
        brush_actions = brushcreationwidget.BrushManipulationWidget(self.app, brushicon_editor)
        vbox.pack_start(brush_actions, expand=False)

        # Live update
        cb = self.live_update = gtk.CheckButton(_('Live update the last canvas stroke'))
        vbox.pack_start(cb, expand=False, fill=True)
        cb.connect('toggled', self.live_update_cb)
        self.app.brush.settings_observers.append(self.live_update_cb)

        # ScrolledWindow for brushsetting-expanders
        scroll = self.brushsettings_widget = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_ALWAYS)

        self.brushinputs_widget = functionwindow.BrushInputsWidget(self.app)
        nb = self.settings_notebook = gtk.Notebook()
        nb.set_show_tabs(False)
        nb.insert_page(self.brushsettings_widget, position=self.PAGE_BRUSHSETTINGS)
        nb.insert_page(self.brushinputs_widget, position=self.PAGE_BRUSHINPUTS)
        nb.set_current_page(self.PAGE_BRUSHSETTINGS)

        def activate_brushsettings_page(*ignore):
            nb.set_current_page(self.PAGE_BRUSHSETTINGS)
        self.brushinputs_widget.back_button.set_label(_('All settings'))
        self.brushinputs_widget.back_button.connect('clicked', activate_brushsettings_page)

        vbox.pack_start(nb, expand=True, fill=True)

        brushsetting_vbox = gtk.VBox()
        scroll.add_with_viewport(brushsetting_vbox)

        header_label = gtk.Label()
        header_label.set_markup('<b><span size="large">%s</span></b>' % ('Brush Settings',))
        header_label.set_alignment(0.0, 0.0)
        brushsetting_vbox.pack_start(header_label, expand=False)

        groups = [
            {'id' : 'basic',    'title' : _('Basic'),   'settings' : [ 'radius_logarithmic', 'radius_by_random', 'hardness', 'eraser', 'offset_by_random', 'elliptical_dab_angle', 'elliptical_dab_ratio', 'direction_filter' ]},
            {'id' : 'opacity',  'title' : _('Opacity'), 'settings' : [ 'opaque', 'opaque_multiply', 'opaque_linearize' ]},
            {'id' : 'dabs',     'title' : _('Dabs'),    'settings' : [ 'dabs_per_basic_radius', 'dabs_per_actual_radius', 'dabs_per_second' ]},
            {'id' : 'smudge',   'title' : _('Smudge'),  'settings' : [ 'smudge', 'smudge_length', 'smudge_radius_log' ]},
            {'id' : 'speed',    'title' : _('Speed'),   'settings' : [ 'speed1_slowness', 'speed2_slowness', 'speed1_gamma', 'speed2_gamma', 'offset_by_speed', 'offset_by_speed_slowness' ]},
            {'id' : 'tracking', 'title' : _('Tracking'),'settings' : [ 'slow_tracking', 'slow_tracking_per_dab', 'tracking_noise' ]},
            {'id' : 'stroke',   'title' : _('Stroke'),  'settings' : [ 'stroke_threshold', 'stroke_duration_logarithmic', 'stroke_holdtime' ]},
            {'id' : 'color',    'title' : _('Color'),   'settings' : [ 'change_color_h', 'change_color_l', 'change_color_hsl_s', 'change_color_v', 'change_color_hsv_s' ]},
            {'id' : 'custom',   'title' : _('Custom'),  'settings' : [ 'custom_input', 'custom_input_slowness' ]}
            ]

        for group in groups:
            self.visible_settings = self.visible_settings + group['settings']
            bold_title = '<b>%s</b>' % (group['title'])
            group_expander = gtk.Expander(label=bold_title)
            group_expander.set_use_markup(True)
            table = gtk.Table(4, len(group['settings']))

            if group['id'] == 'basic':
                group_expander.set_expanded(True)

            for i, cname in enumerate(group['settings']):
                s = brushsettings.settings_dict[cname]
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

            group_expander.add(table)
            brushsetting_vbox.pack_start(group_expander, expand=False)

    def set_fixed_value_clicked_cb(self, widget, adj, value):
        adj.set_value(value)

    def details_clicked_cb(self, window, adj, setting):
        """Go to brush input/dynamics page."""
        self.brushinputs_widget.set_brushsetting(setting, adj)
        self.settings_notebook.set_current_page(self.PAGE_BRUSHINPUTS)

    def value_changed_cb(self, adj, index, app):
        setting = [k for k, v in self.adj.items() if v == adj][0]
        s = app.brush.settings[index]
        s.set_base_value(adj.get_value())
        self.relabel_setting_buttons(adj, setting, s)

    def update_settings(self):
        """Update all settings; their value and button labels"""
        for s in self.visible_settings:
            setting = brushsettings.settings_dict[s]
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

    def brush_selected_cb(self, brush):
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

