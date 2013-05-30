# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"tune brush window"
from gettext import gettext as _
import gtk, gobject
import functionwindow, brushcreationwidget
import windowing
from brushlib import brushsettings


class BrushSettingsWindow (windowing.SubWindow):

    PAGE_BRUSHSETTINGS = 0
    PAGE_BRUSHINPUTS = 1
    PAGE_BRUSHPROPERTIES = 2

    def __init__(self):
        import application
        app = application.get_app()
        assert app is not None
        windowing.SubWindow.__init__(self, app, key_input=True)

        self.adj = {}
        self.functionWindows = {}
        # A list of all brushsettings (cname) which are to be displayed
        self.visible_settings = []
        self.live_update_queued = False

        self.set_title(_('Brush Settings Editor'))
        self.init_ui()
        self.set_default_size(450, 500)

        self.app.brush.observers.append(self.brush_modified_cb)

    def init_ui(self):
        """Construct and pack widgets."""
        vbox = gtk.VBox()
        self.add(vbox)
        self.set_border_width(5)

        brushicon_editor = brushcreationwidget.BrushIconEditorWidget(self.app)
        self.brushinputs_widget = functionwindow.BrushInputsWidget(self.app)

        # Header with brush name and actions
        brush_actions = brushcreationwidget.BrushManipulationWidget(self.app, brushicon_editor)
        vbox.pack_start(brush_actions, expand=False, padding=8)

        # Header with current page name
        header_hbox = gtk.HBox()
        self.header_label = gtk.Label()
        self.header_label.set_markup('<b><span size="large">%s</span></b>' % ('Brush Settings',))
        self.header_label.set_alignment(0.0, 0.0)

        self.header_button = gtk.Button(_('Back to settings'))
        self.header_button.set_no_show_all(True)

        header_hbox.pack_start(self.header_label, expand=False)
        header_hbox.pack_end(self.header_button, expand=False)
        vbox.pack_start(header_hbox, expand=False)

        # Live update (goes to the end)
        cb = self.live_update = gtk.CheckButton(_('Update the last canvas stroke in realtime'))
        vbox.pack_end(cb, expand=False, fill=True)
        cb.connect('toggled', self.live_update_cb)
        cb.set_no_show_all(True)

        # ScrolledWindow for brushsetting-expanders
        scroll = self.brushsettings_widget = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_ALWAYS)

        nb = self.settings_notebook = gtk.Notebook()
        nb.set_show_tabs(False)
        nb.insert_page(self.brushsettings_widget, None, self.PAGE_BRUSHSETTINGS)
        nb.insert_page(self.brushinputs_widget, None, self.PAGE_BRUSHINPUTS)
        nb.insert_page(brushicon_editor, None, self.PAGE_BRUSHPROPERTIES)

        vbox.pack_start(nb, expand=True, fill=True)

        def activate_brushsettings_page(*ignore):
            nb.set_current_page(self.PAGE_BRUSHSETTINGS)
            self.header_label.set_markup('<b><span size="large">%s</span></b>' % ('Brush Settings',))
            self.header_button.hide()
            self.live_update.show()
        def activate_brushproperties_page(*ignore):
            nb.set_current_page(self.PAGE_BRUSHPROPERTIES)
            self.header_label.set_markup('<b><span size="large">%s</span></b>' % ('Brush Icon',))
            self.header_button.show()
            self.live_update.hide()

        activate_brushsettings_page() # Default page

        brush_actions.edit_brush_properties_cb = activate_brushproperties_page
        self.header_button.connect('clicked', activate_brushsettings_page)

        brushsetting_vbox = gtk.VBox()
        scroll.add_with_viewport(brushsetting_vbox)

        groups = [
            {'id' : 'experimental', 'title' : _('Experimental'), 'settings' : []},
            {'id' : 'basic',        'title' : _('Basic'),        'settings' : [ 'radius_logarithmic', 'radius_by_random', 'hardness', 'snap_to_pixel', 'anti_aliasing', 'eraser', 'offset_by_random', 'elliptical_dab_angle', 'elliptical_dab_ratio', 'direction_filter' ]},
            {'id' : 'opacity',      'title' : _('Opacity'),      'settings' : [ 'opaque', 'opaque_multiply', 'opaque_linearize', 'lock_alpha' ]},
            {'id' : 'dabs',         'title' : _('Dabs'),         'settings' : [ 'dabs_per_basic_radius', 'dabs_per_actual_radius', 'dabs_per_second' ]},
            {'id' : 'smudge',       'title' : _('Smudge'),       'settings' : [ 'smudge', 'smudge_length', 'smudge_radius_log' ]},
            {'id' : 'speed',        'title' : _('Speed'),        'settings' : [ 'speed1_slowness', 'speed2_slowness', 'speed1_gamma', 'speed2_gamma', 'offset_by_speed', 'offset_by_speed_slowness' ]},
            {'id' : 'tracking',     'title' : _('Tracking'),     'settings' : [ 'slow_tracking', 'slow_tracking_per_dab', 'tracking_noise' ]},
            {'id' : 'stroke',       'title' : _('Stroke'),       'settings' : [ 'stroke_threshold', 'stroke_duration_logarithmic', 'stroke_holdtime' ]},
            {'id' : 'color',        'title' : _('Color'),        'settings' : [ 'change_color_h', 'change_color_l', 'change_color_hsl_s', 'change_color_v', 'change_color_hsv_s', 'restore_color', 'colorize' ]},
            {'id' : 'custom',       'title' : _('Custom'),       'settings' : [ 'custom_input', 'custom_input_slowness' ]}
            ]
        hidden_settings = ['color_h', 'color_s', 'color_v']

        # add new settings to the "experimental" group
        grouped_settings = hidden_settings[:]
        for g in groups:
            grouped_settings.extend(g['settings'])
        for s in brushsettings.settings:
            if s.cname not in grouped_settings:
                groups[0]['settings'].append(s.cname)
                print 'Warning: setting "%r" should be added to a group in brushsettingswindow.py' % s.cname
        # hide experimental group if empty
        if not groups[0]['settings']:
            groups.pop(0)

        for group in groups:
            self.visible_settings = self.visible_settings + group['settings']
            bold_title = '<b>%s</b>' % (group['title'])
            group_expander = gtk.Expander(label=bold_title)
            group_expander.set_use_markup(True)
            table = gtk.Table(4, len(group['settings']))

            if group['id'] in ['basic', 'experimental']:
                group_expander.set_expanded(True)

            for i, cname in enumerate(group['settings']):
                s = brushsettings.settings_dict[cname]
                l = gtk.Label(s.name)
                l.set_alignment(0, 0.5)
                l.set_tooltip_text(s.tooltip)

                adj = self.app.brush_adjustment[s.cname]
                adj.connect('value-changed', self.value_changed_cb, cname)
                self.adj[cname] = adj
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
                    b2.set_tooltip_text(_("Add input value mapping"))
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
        self.header_label.set_markup('<b><span size="large">%s</span></b>' % setting.name)
        self.header_label.set_tooltip_text(setting.tooltip)
        self.header_button.show()
        self.live_update.show()

    def value_changed_cb(self, adj, cname):
        value = adj.get_value()
        self.app.brush.set_base_value(cname, value)

    def update_settings(self, settings):
        """Update adjustment and button labels"""
        for cname in settings.intersection(self.visible_settings):
            # Update slider
            adj = self.adj[cname]
            adj.set_value(self.app.brush.get_base_value(cname))

            # Make the "input value mapping" button reflect whether
            # this brush already has a mapping or not
            if adj.three_dots_button:
                def set_label(s, t):
                    if adj.three_dots_button.get_label() == s: return
                    adj.three_dots_button.set_label(s)
                    adj.three_dots_button.set_tooltip_text(t)
                if self.app.brush.has_only_base_value(cname):
                    set_label("...", _("Add input value mapping"))
                else:
                    set_label("X", _("Modify input value mapping"))

            # Make "reset to default value" button insensitive
            # if the value is already the default (the button will have no effect)
            if adj.get_value() == brushsettings.settings_dict[cname].default:
                adj.default_value_button.set_sensitive(False)
            else:
                adj.default_value_button.set_sensitive(True)

    def brush_modified_cb(self, settings):
        self.update_settings(settings)
        self.live_update_cb()

    def live_update_cb(self, *junk):
        if not self.app.doc.modes.top.is_live_updateable:
            # formerly: "not if in a drag"
            # currrently: only FreehandMode supports it (rest untested)
            return
        if not self.live_update.get_active() or not self.get_visible() or self.live_update_queued:
            return
        self.live_update_queued = True
        def do_update():
            self.live_update_queued = False
            doc = self.app.doc.model
            doc.redo_last_stroke_with_different_brush(self.app.brush)
        gobject.idle_add(do_update)

