# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Brush selection/organizer window

Can be used for selecting brushes, and can be docked into the sidebar.
Responsible for ordering, loading and saving brush lists.

"""

import platform

import gtk2compat

from gettext import gettext as _
if gtk2compat.USE_GTK3:
    import gi
    from gi.repository import PangoCairo
import pango
import gtk
from gtk import gdk

import pixbuflist
import dialogs
import brushmanager
from brushlib import brushsettings
from lib.helpers import escape
from colors import RGBColor


class ToolWidget (gtk.VBox):

    EXPANDER_PREFS_KEY = "brushmanager.common_settings_expanded"

    stock_id = "mypaint-tool-brush"
    tool_widget_title = _("Brushes")

    def __init__(self, app):
        self.app = app
        gtk.VBox.__init__(self)

        self.last_selected_brush = None

        self.groupselector = GroupSelector()
        self.brushgroups = BrushGroupsList()

        self.scroll = scroll = gtk.ScrolledWindow()
        scroll.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        scroll.add_with_viewport(self.brushgroups)

        expander = gtk.Expander(label=None)
        expander.add(get_common_settings_widget(app))
        expander.connect("notify::expanded", self.common_settings_expanded_cb)
        self.expander = expander
        self.expander_prefs_loaded = False
        self.connect("show", self.show_cb)

        self.pack_start(self.groupselector, expand=False)
        self.pack_start(gtk.HSeparator(), expand=False)
        self.pack_start(scroll, expand=True)
        self.pack_start(gtk.HSeparator(), expand=False)
        self.pack_start(expander, expand=False, fill=False)


    def show_cb(self, widget):
        assert not self.expander_prefs_loaded
        is_expanded = bool(self.app.preferences.get(self.EXPANDER_PREFS_KEY, False))
        self.expander.set_expanded(is_expanded)
        self.expander_prefs_loaded = True


    def common_settings_expanded_cb(self, expander, *junk):
        if not self.expander_prefs_loaded:
            return
        is_expanded = bool(expander.get_expanded())
        self.app.preferences[self.EXPANDER_PREFS_KEY] = is_expanded


def get_common_settings_widget(app):
    """Return a widget with controls for manipulating common settings"""

    cmn = ['radius_logarithmic', 'opaque', 'hardness']
    common_settings = [s for s in brushsettings.settings_visible if s.cname in cmn]
    settings_box = gtk.VBox()

    def value_changed_cb(adj, cname, app):
        app.brush.set_base_value(cname, adj.get_value())

    def get_setting_widget(setting):
        """Return a widget to control a single setting"""
        adj = app.brush_adjustment[s.cname]
        adj.connect('value-changed', value_changed_cb, s.cname, app)

        l = gtk.Label(s.name)
        l.set_alignment(0, 0.5)

        h = gtk.HScale(adj)
        h.set_digits(2)
        h.set_draw_value(True)
        h.set_value_pos(gtk.POS_LEFT)

        box = gtk.HBox()
        box.pack_start(l)
        box.pack_start(h)
        return box

    for s in common_settings:
        settings_box.pack_start(get_setting_widget(s))

    return settings_box

class BrushList(pixbuflist.PixbufList):

    ICON_SIZE = 48
    MIN_WIDTH_NICONS = 1
    NATURAL_WIDTH_NICONS = 4
    MIN_HEIGHT_NICONS = 1

    def __init__(self, app, group):
        self.app = app
        self.bm = app.brushmanager
        self.brushes = self.bm.groups[group]
        self.group = group
        s = self.ICON_SIZE
        pixbuflist.PixbufList.__init__(self, self.brushes, s, s,
                                       namefunc = lambda x: x.name,
                                       pixbuffunc = lambda x: x.preview)
        # Support device changing with the same event as that used
        # for brush choice:
        if not gtk2compat.USE_GTK3:
            self.set_extension_events(gdk.EXTENSION_EVENTS_ALL)

        self.set_selected(self.bm.selected_brush)
        self.bm.brushes_observers.append(self.brushes_modified_cb)
        self.bm.selected_brush_observers.append(self.brush_selected_cb)


    def do_get_request_mode(self):
        return gtk.SizeRequestMode.HEIGHT_FOR_WIDTH


    def do_get_preferred_width(self):
        return (self.MIN_WIDTH_NICONS * self.ICON_SIZE,
                self.NATURAL_WIDTH_NICONS * self.ICON_SIZE)


    def do_get_preferred_height_for_width(self, width):
        icons_wide = max(1, int(width / self.ICON_SIZE))
        num_brushes = len(self.brushes)
        icons_tall = max(int(num_brushes / icons_wide),
                         max(self.MIN_HEIGHT_NICONS, 1))
        if icons_tall * icons_wide  < num_brushes:
            icons_tall += 1
        return (icons_tall * self.ICON_SIZE,
                icons_tall * self.ICON_SIZE)


    def brushes_modified_cb(self, brushes):
        if brushes is self.brushes:
            self.update()

    def brush_selected_cb(self, managed_brush, brushinfo):
        self.set_selected(managed_brush)

    def remove_brush(self, brush):
        self.brushes.remove(brush)
        for f in self.bm.brushes_observers: f(self.brushes)

    def insert_brush(self, idx, brush):
        self.brushes.insert(idx, brush)
        for f in self.bm.brushes_observers: f(self.brushes)

    def button_press_cb(self, widget, event):
        if gtk2compat.USE_GTK3:
            device = event.get_source_device()
        else:
            device = event.device
        self.app.device_monitor.device_used(device)
        return pixbuflist.PixbufList.button_press_cb(self, widget, event)

    def drag_begin_cb(self, widget, context):
        preview = self.bm.selected_brush.preview
        preview = preview.scale_simple(preview.get_width()//2, preview.get_height()//2, gtk.gdk.INTERP_BILINEAR)
        self.drag_source_set_icon_pixbuf(preview)
        pixbuflist.PixbufList.drag_begin_cb(self, widget, context)

    #def drag_end_cb(self, widget, context):
    #    pixbuflist.PixbufList.drag_end_cb(self, widget, context)

    def on_drag_data(self, copy, source_widget, brush_name, target_idx):
        assert source_widget, 'cannot handle drag data from another app'
        b, = [b for b in source_widget.brushes if b.name == brush_name]
        if source_widget is self:
            copy = False
        else:
            if b in self.brushes:
                source_widget.remove_brush(b)
                self.remove_brush(b)
                self.insert_brush(target_idx, b)
                return True
        if not copy:
            source_widget.remove_brush(b)
        self.insert_brush(target_idx, b)
        return True

    def on_select(self, brush):
        # brush changed on harddisk?
        if brush.reload_if_changed():
            for brushes in self.bm.groups.itervalues():
                for f in self.bm.brushes_observers: f(brushes)

        self.bm.select_brush(brush)



class BrushGroupsList(gtk.VBox):
    """A vertical stack of brush lists.
    """

    def __init__(self):
        gtk.VBox.__init__(self)
        from application import get_app
        app = get_app()
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

        for child in list(self.get_children()):
            self.remove(child)

        for group in self.bm.active_groups:
            if group in self.group_widgets:
                w = self.group_widgets[group]
            else:
                w = BrushList(self.app, group)
            self.group_widgets[group] = w
            self.pack_start(w, expand=False, fill=False, padding=3)

        self.show_all()



class GroupSelector (gtk.DrawingArea):
    """Brush group selector widget.

    Shows a line-wrapped list of group names, any or all of which may be
    selected or unselected with the pointer device. Toggling groups on or off
    this way updates the associated BrushGroupsList to show different groups.

    """


    VERTICAL_MARGIN = 2


    def __init__(self):
        gtk.DrawingArea.__init__(self)
        from application import get_app
        app = get_app()
        self.app = app
        self.bm = app.brushmanager
        self.bm.groups_observers.append(self.active_groups_changed_cb)

        if not gtk2compat.USE_GTK3:
            self.drag_dest_set(gtk.DEST_DEFAULT_MOTION | gtk.DEST_DEFAULT_DROP,
                [('LIST_ITEM', gtk.TARGET_SAME_APP, pixbuflist.DRAG_ITEM_NAME)],
                gdk.ACTION_COPY|gdk.ACTION_MOVE)

        self.connect('drag-motion', self.drag_motion_cb)
        self.connect('drag-data-received', self.drag_data_received_cb)
        self.connect('drag-leave', self.drag_clear_cb)
        self.connect('drag-begin', self.drag_clear_cb)
        self.connect('drag-end', self.drag_clear_cb)

        if gtk2compat.USE_GTK3:
            self.connect("draw", self.draw_cb)
        else:
            self.connect("expose-event", self.expose_cb)

        self.connect("button-press-event", self.button_press_cb)
        self.connect("motion-notify-event", self.motion_notify_cb)
        self.connect("leave-notify-event", self.leave_notify_cb)
        self.set_events(gdk.EXPOSURE_MASK |
                        gdk.BUTTON_PRESS_MASK |
                        gdk.BUTTON_RELEASE_MASK |
                        gdk.POINTER_MOTION_MASK
                        )
        self.idx2group = {}
        self.layout = None
        self.gtkstate_prelight_group = None
        self.gtkstate_active_group = None
        self.drag_target_group = None
        self.set_tooltip_text(_('Brush Groups: click to select\n'
                                'Ctrl: select multiple\n'
                                'Middle-click: toggle group\n'
                                'Right-click: groups menu'))

        # Style change detection, and default styles.
        # Layout colors, represented independently of GTK/GDK.
        self._text_colors = {
            "normal":     (RGBColor(0., 0., 0.), RGBColor(.8, .8, .8)),
            "normal_pre": (RGBColor(0., 0., 0.), RGBColor(.9, .9, .9)),
            "selected":     (RGBColor(1., 1., 1.), RGBColor(0., .5, .8)),
            "selected_pre": (RGBColor(1., 1., 1.), RGBColor(.1, .6, .9)),
            "active":     (RGBColor(.8, .8, .8), RGBColor(0., 0., 0.)),
            "active_pre":   (RGBColor(.9, .9, .9), RGBColor(0., 0., 0.)),
            }
        self._leading = 2 * pango.SCALE
        if gtk2compat.USE_GTK3:
            self.connect("style-updated", self._style_updated_cb)
            # Fake a style
            style_context = self.get_style_context()
            style_context.add_class(gtk.STYLE_CLASS_VIEW)
        else:
            # No new PyGTK code.
            pass


    def active_groups_changed_cb(self):
        self.queue_draw()


    def lay_out_group_names(self, width):
        """Typeset the group names into a new Pango layout at a given width.
        """
        layout = pango.Layout(self.get_pango_context())
        layout.set_width(width*pango.SCALE)

        all_groups = list(sorted(self.bm.groups.keys()))
        idx = 0
        attr = pango.AttrList()
        self.idx2group = {}

        # Pick separator chars. Platform-dependent, but assume Unicode
        # for modern OSes first.
        pad_s = u"\u202f"  # NARROW NO-BREAK SPACE
        sp_s = pad_s + u"\u200b"  # ZERO WIDTH SPACE

        if platform.system() == 'Windows' or platform.system() == 'Darwin':
            # workaround for https://gna.org/bugs/?15192
            pad_s = ''
            sp_s = ' '

        markup = ''
        for group in all_groups:
            group_label = brushmanager.translate_group_name(group)
            u = pad_s + group_label + pad_s
            idx_start = idx
            for c in u.encode('utf-8'):
                self.idx2group[idx] = group
                idx += 1

            # Note the difference in terminology here
            colors_name = "normal"
            set_bgcolor = False
            if group == self.gtkstate_active_group: # activated the menu
                colors_name = "active"
                set_bgcolor = True
            elif group in self.bm.active_groups: # those groups visible
                colors_name = "selected"
                set_bgcolor = True
            if group == self.gtkstate_prelight_group:
                colors_name += "_pre"
                set_bgcolor = True
            fg, bg = self._text_colors[colors_name]

            if group == self.drag_target_group:
                # Invert colours
                fg, bg = bg, fg

            c_fg = fg.to_hex_str()
            m = escape(u)
            if set_bgcolor:
                c_bg = bg.to_hex_str()
                bgcolor = " bgcolor='%s'" % (c_bg,)
            else:
                bgcolor = ""
            m = "<span fgcolor='%s'%s>%s</span>" % (c_fg, bgcolor, m)
            markup += m + sp_s
            idx += len(sp_s.encode("utf-8"))

        layout.set_markup(markup)
        layout.set_spacing(self._leading)
        return layout


    def do_get_request_mode(self):
        return gtk.SizeRequestMode.HEIGHT_FOR_WIDTH

    def do_get_preferred_width(self):
        return (100, 300)

    def do_get_preferred_height_for_width(self, width):
        layout = self.lay_out_group_names(width)
        w, h = layout.get_pixel_size()
        h += 2 * self.VERTICAL_MARGIN
        return (h, h)


    def _style_updated_cb(self, widget, *a):
        """Callback: updates colors in response to the style changing.
        """
        style_context = widget.get_style_context()

        text_colors_new = self._text_colors.copy()
        f_norm = gtk.StateFlags.NORMAL
        f_pre = gtk.StateFlags.PRELIGHT
        f_active = gtk.StateFlags.ACTIVE
        f_sel = gtk.StateFlags.SELECTED
        style_info = [ ("normal", f_norm, False),
                       ("normal_pre", f_norm|f_pre, False),
                       ("active", f_active, True),
                       ("active_pre", f_active|f_pre, True),
                       ("selected", f_sel, False),
                       ("selected_pre", f_sel|f_pre, False), ]
        for key, flags, inverse_video in style_info:
            fg_rgba = style_context.get_color(flags)
            bg_rgba = style_context.get_background_color(flags)
            fg = RGBColor.new_from_gdk_rgba(fg_rgba)
            bg = RGBColor.new_from_gdk_rgba(bg_rgba)
            if inverse_video:
                fg, bg = bg, fg
            text_colors_new[key] = (fg, bg)

        leading_new = self._leading
        font_description = style_context.get_font(gtk.StateFlags.NORMAL)
        if font_description:
            leading_new = font_description.get_size() / 6

        style_changed = (self._leading != leading_new or
                         self._text_colors != text_colors_new)
        if style_changed:
            self._leading = leading_new
            self._text_colors = text_colors_new
            self.queue_draw()


    def expose_cb(self, widget, event):
        cr = self.get_window().cairo_create()
        return self.draw_cb(widget, cr)


    def draw_cb(self, widget, cr):
        alloc = self.get_allocation()
        width = alloc.width
        height = alloc.height

        fg, bg = self._text_colors["normal"]
        #cr.set_source_rgb(*bg.get_rgb())
        #cr.paint()

        cr.set_source_rgb(*fg.get_rgb())
        layout = self.lay_out_group_names(width)

        vmargin = self._leading // pango.SCALE
        layout.set_spacing(self._leading)

        cr.move_to(0, self.VERTICAL_MARGIN)
        if gtk2compat.USE_GTK3:
            PangoCairo.show_layout(cr, layout)
        else:
            cr.show_layout(layout)

        self.layout = layout


    def group_at(self, x, y):
        x, y = int(x), int(y) # avoid warning
        if self.layout is None:
            return None
        index_tup = self.layout.xy_to_index(x*pango.SCALE, y*pango.SCALE)
        if gtk2compat.USE_GTK3:
            inside, i, trailing = index_tup
        else:
            i, trailing = index_tup
        return self.idx2group.get(i)


    def button_press_cb(self, widget, event):
        if event.type != gdk.BUTTON_PRESS:
            return # double or tripple click
        group = self.group_at(event.x, event.y)

        if event.button in [1, 2]:
            if not group:
                return
            active_groups = self.bm.active_groups[:]
            is_toggle_request = (event.state & gdk.CONTROL_MASK or
                                 event.state & gdk.SHIFT_MASK or
                                 event.button == 2)
            if is_toggle_request:
                # toggle group visibility
                if group in active_groups:
                    active_groups.remove(group)
                else:
                    active_groups += [group]
            else:
                # group solo
                active_groups = [group]
            self.bm.set_active_groups(active_groups)

        elif event.button == 3:
            # context menu
            self.gtkstate_active_group = group
            self.queue_draw()
            menu = self.context_menu(group)
            data = None
            def _pos(*a):
                return int(event.x_root), int(event.y_root), True
            # GTK3: arguments have a different order, and "data" is required.
            # GTK3: Use keyword arguments for max compatibility.
            menu.popup(parent_menu_shell=None, parent_menu_item=None,
                       func=_pos, button=event.button,
                       activate_time=event.time, data=data)


    def motion_notify_cb(self, widget, event):
        old_prelight_group = self.gtkstate_prelight_group
        self.gtkstate_prelight_group = self.group_at(event.x, event.y)
        if self.gtkstate_prelight_group != old_prelight_group:
            self.queue_draw()


    def leave_notify_cb(self, widget, event):
        old_prelight_group = self.gtkstate_prelight_group
        self.gtkstate_prelight_group = None
        if self.gtkstate_prelight_group != old_prelight_group:
            self.gtkstate_prelight_group = None
            self.queue_draw()


    def context_menu(self, group):
        m = gtk.Menu()
        menu = []
        menu += [ (_("New Group..."), self.create_group_cb) ]
        if group:
            menu += [ (_("Rename Group..."), self.rename_group_cb),
                      (_("Delete Group"), self.delete_group_cb),
                      (_("Export Group..."), self.export_group_cb), ]
        menu += [ None ]
        menu += [ (_("Import Brushes..."),
                   self.app.drawWindow.import_brush_pack_cb) ]
        menu += [ (_("Get More Brushes..."),
                   self.app.drawWindow.download_brush_pack_cb) ]
        for entry in menu:
            if entry is None:
                item = gtk.SeparatorMenuItem()
            else:
                label, callback = entry
                item = gtk.MenuItem(label)
                item.connect('activate', callback, group)
            m.append(item)
        m.connect('selection-done', self.menu_finished_cb)
        m.show_all()
        return m


    def menu_finished_cb(self, menushell):
        self.gtkstate_active_group = None
        self.queue_draw()


    def create_group_cb(self, w, group):
        new_group = dialogs.ask_for_name(self, _('Create Group'), '')
        if new_group:
            self.bm.create_group(new_group)


    def rename_group_cb(self, w, old_group):
        new_group = dialogs.ask_for_name(self, _('Rename Group'), old_group)
        if not new_group:
            return
        if new_group not in self.bm.groups:
            self.bm.rename_group(old_group, new_group)
        else:
            dialogs.error(self, _('A group with this name already exists!'))


    def export_group_cb(self, w, group):
        format_id, filename = dialogs.save_dialog(_("Export Brushes"), None,
                                 [(_("MyPaint brush package (*.zip)"), "*.zip")],
                                 default_format = (0, ".zip"))
        if filename is not None:
            self.bm.export_group(group, filename)


    def delete_group_cb(self, w, group):
        msg = _('Really delete group %s?') \
                % brushmanager.translate_group_name(group)
        if dialogs.confirm(self, msg):
            self.bm.delete_group(group)
            if group in self.bm.groups:
                msg = _('This group can not be deleted '
                        '(try to empty it first).')
                dialogs.error(self, msg)


    def drag_data_received_cb(self, widget, context, x, y, selection,
                              targetType, time):
        """
        Respond to some data being dropped via dnd onto the list of groups.
        This typically results in a brush being copied or moved into the group
        being dragged onto.
        """
        group = self.group_at(x,y)
        source = context.get_source_widget()
        # ensure, that drop comes from BrushList and targets some group
        if not group or not isinstance(source, BrushList):
            context.finish(False, False, time)
            return
        target = self.bm.groups[group]
        brush = self.bm.get_brush_by_name(selection.data)
        changed = []
        if context.action == gdk.ACTION_MOVE:
            source.brushes.remove(brush)
            if brush not in target:
                target.append(brush)
            changed = source.brushes
        elif context.action == gdk.ACTION_COPY:
            if brush not in target:
                target.append(brush)
                changed = target
        else:
            context.finish(False, False, time)
        for f in self.bm.brushes_observers: f(changed)
        context.finish(True, False, time)


    def drag_motion_cb(self, widget, context, x, y, time):
        """
        During dragging a brush from an open BrushList, select the action to
        take when the drop happens. Provide feedback by changing the mouse
        cursor and highlighting the group label under the cursor, if
        applicable.
        """
        group = self.group_at(x,y)
        source = context.get_source_widget()
        if group is None or not isinstance(source, BrushList):
            # Unknown action if not dragging from a BrushList or onto
            # a group label.
            action = gdk.ACTION_DEFAULT
        else:
            # Default action is to copy the brush
            action = gdk.ACTION_COPY
            dragged_brush = source.selected
            target = self.bm.groups[group]
            if group == source.group:
                # Dragging to the current group label should behave
                # like dragging between visible BrushLists, i.e. a no-op.
                action = gdk.ACTION_DEFAULT
            elif dragged_brush in target:
                # If the brush is already in the target group, move it instead
                action = gdk.ACTION_MOVE
            else:
                # The user can force a move by pressing shift during the drag
                px, py, kbmods = self.get_window().get_pointer()
                if kbmods & gdk.SHIFT_MASK:
                    action = gdk.ACTION_MOVE
        context.drag_status(action, time)
        if action == gdk.ACTION_DEFAULT:
            group = None
        if group != self.drag_target_group:
            self.drag_target_group = group
            self.queue_draw()


    def drag_clear_cb(self, widget, context, time):
        """
        Remove any UI features showing the drag-and-drop target. Call when the
        cursor goes out of the widget during a drag, or when the drag would
        have no effect for any other reason.
        """
        if self.drag_target_group is  None: return
        self.drag_target_group = None
        self.queue_draw()
