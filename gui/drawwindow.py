# -*- coding: utf-8 -*-
#
# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""
This is the main drawing window, containing menu actions.
Painting is done in tileddrawwidget.py.
"""

MYPAINT_VERSION="0.7.1+git"

import os, math
from gettext import gettext as _

import gtk
from gtk import gdk, keysyms

import tileddrawwidget, colorselectionwindow, historypopup, \
       stategroup, keyboard, colorpicker, filehandling
from lib import document, helpers, backgroundsurface, command, layer

#TODO: make generic by taking the windows as arguments and put in a helper file?
def with_wait_cursor(func):
    """python decorator that adds a wait cursor around a function"""
    def wrapper(self, *args, **kwargs):
        # process events which might include cursor changes
        while gtk.events_pending():
            gtk.main_iteration(False)
        self.app.drawWindow.window.set_cursor(gdk.Cursor(gdk.WATCH))
        self.app.drawWindow.tdw.window.set_cursor(None)
        # make sure it is actually changed before we return
        while gtk.events_pending():
            gtk.main_iteration(False)
        try:
            func(self, *args, **kwargs)
        finally:
            self.app.drawWindow.window.set_cursor(None)
            self.app.drawWindow.tdw.update_cursor()
    return wrapper


class Window(gtk.Window):
    def __init__(self, app):
        gtk.Window.__init__(self)
        self.app = app

        self.connect('delete-event', self.quit_cb)
        self.connect('key-press-event', self.key_press_event_cb_before)
        self.connect('key-release-event', self.key_release_event_cb_before)
        self.connect_after('key-press-event', self.key_press_event_cb_after)
        self.connect_after('key-release-event', self.key_release_event_cb_after)
        self.connect("drag-data-received", self.drag_data_received)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("button-release-event", self.button_release_cb)
        self.connect("scroll-event", self.scroll_cb)
        self.set_default_size(600, 400)
        vbox = gtk.VBox()
        self.add(vbox)

        #TODO: move self.doc into application.py?
        self.doc = document.Document()
        self.doc.set_brush(self.app.brush)

        self.create_ui()
        self.menubar = self.app.ui_manager.get_widget('/Menubar')
        vbox.pack_start(self.menubar, expand=False)

        self.tdw = tileddrawwidget.TiledDrawWidget(self.doc)
        vbox.pack_start(self.tdw)

        # FIXME: hack, to be removed
        fname = os.path.join(self.app.datapath, 'backgrounds', '03_check1.png')
        pixbuf = gdk.pixbuf_new_from_file(fname)
        self.tdw.neutral_background_pixbuf = backgroundsurface.Background(pixbuf)

        self.zoomlevel_values = [1.0/8, 2.0/11, 0.25, 1.0/3, 0.50, 2.0/3, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 8.0]
        self.zoomlevel = self.zoomlevel_values.index(1.0)
        self.tdw.zoom_min = min(self.zoomlevel_values)
        self.tdw.zoom_max = max(self.zoomlevel_values)
        self.fullscreen = False

        self.app.brush.settings_observers.append(self.brush_modified_cb)
        self.tdw.device_observers.append(self.device_changed_cb)

        self.eraser_mode_radius_change = 3*(0.3) # can go back to exact original with brush_smaller_cb()
        self.eraser_mode_original_radius = None

        # enable drag & drop
        self.drag_dest_set(gtk.DEST_DEFAULT_MOTION | gtk.DEST_DEFAULT_HIGHLIGHT | gtk.DEST_DEFAULT_DROP, [("text/uri-list", 0, 1)], gtk.gdk.ACTION_DEFAULT|gtk.gdk.ACTION_COPY)

    def create_ui(self):
        actions = [
            # name, stock id, label, accelerator, tooltip, callback
            ('FileMenu',     None, _('File')),
            ('Quit',         gtk.STOCK_QUIT, _('Quit'), '<control>q', None, self.quit_cb),

            ('EditMenu',           None, _('Edit')),
            ('Undo',               gtk.STOCK_UNDO, _('Undo'), 'Z', None, self.undo_cb),
            ('Redo',               gtk.STOCK_REDO, _('Redo'), 'Y', None, self.redo_cb),

            ('BrushMenu',    None, _('Brush')),
            ('Brighter',     None, _('Brighter'), None, None, self.brighter_cb),
            ('Smaller',      None, _('Smaller'), 'd', None, self.brush_smaller_cb),
            ('MoreOpaque',   None, _('More Opaque'), 's', None, self.more_opaque_cb),
            ('LessOpaque',   None, _('Less Opaque'), 'a', None, self.less_opaque_cb),
            ('Eraser',       None, _('Toggle Eraser Mode'), 'e', None, self.eraser_cb),
            ('PickContext',  None, _('Pick Context (layer, brush and color)'), 'w', None, self.pick_context_cb),

            ('ColorMenu',    None, _('Color')),
            ('Darker',       None, _('Darker'), None, None, self.darker_cb),
            ('Bigger',       None, _('Bigger'), 'f', None, self.brush_bigger_cb),
            ('ColorPickerPopup',    None, _('Pick Color'), 'r', None, self.popup_cb),
            ('ColorHistoryPopup',  None, _('Color History'), 'x', None, self.popup_cb),
            ('ColorChangerPopup', None, _('Color Changer'), 'v', None, self.popup_cb),
            ('ColorRingPopup',  None, _('Color Ring'), None, None, self.popup_cb),

            ('ContextMenu',  None, _('Brushkeys')),
            #each of the context actions are generated and added below
            ('ContextStore', None, _('Save to Most Recently Restored'), 'q', None, self.context_cb),
            ('ContextHelp',  gtk.STOCK_HELP, _('Help!'), None, None, self.show_infodialog_cb),

            ('LayerMenu',    None, _('Layers')),

            ('BackgroundWindow', None, _('Background...'), None, None, self.toggleWindow_cb),
            ('ClearLayer',   None, _('Clear'), 'Delete', None, self.clear_layer_cb),
            ('CopyLayer',          None, _('Copy to Clipboard'), '<control>C', None, self.copy_cb),
            ('PasteLayer',         None, _('Paste Clipboard (Replace Layer)'), '<control>V', None, self.paste_cb),
            ('PickLayer',    None, _('Select Layer at Cursor'), 'h', None, self.pick_layer_cb),
            ('LayerFG',      None, _('Next (above current)'),  'Page_Up', None, self.layer_fg_cb),
            ('LayerBG',      None, _('Next (below current)'), 'Page_Down', None, self.layer_bg_cb),
            ('NewLayerFG',   None, _('New (above current)'), '<control>Page_Up', None, self.new_layer_cb),
            ('NewLayerBG',   None, _('New (below current)'), '<control>Page_Down', None, self.new_layer_cb),
            ('MergeLayer',   None, _('Merge Down'), '<control>Delete', None, self.merge_layer_cb),
            ('RemoveLayer',  None, _('Remove'), '<shift>Delete', None, self.remove_layer_cb),
            ('IncreaseLayerOpacity', None, _('Increase Layer Opacity'),  'p', None, self.layer_increase_opacity),
            ('DecreaseLayerOpacity', None, _('Decrease Layer Opacity'),  'o', None, self.layer_decrease_opacity),

            ('BrushSelectionWindow',  None, _('Brush List...'), 'b', None, self.toggleWindow_cb),
            ('BrushSettingsWindow',   None, _('Brush Settings...'), '<control>b', None, self.toggleWindow_cb),
            ('ColorSelectionWindow',  None, _('Color Triangle...'), 'g', None, self.toggleWindow_cb),
            ('ColorSamplerWindow',  gtk.STOCK_COLOR_PICKER, _('Color Sampler...'), 't', None, self.toggleWindow_cb),
            ('SettingsWindow',        gtk.STOCK_PREFERENCES, _('Settings...'), None, None, self.toggleWindow_cb),

            ('HelpMenu',     None, _('Help')),
            ('Docu', None, _('Where is the Documentation?'), None, None, self.show_infodialog_cb),
            ('ShortcutHelp',  None, _('Change the Keyboard Shortcuts?'), None, None, self.show_infodialog_cb),
            ('About', gtk.STOCK_ABOUT, _('About MyPaint'), None, None, self.about_cb),

            ('DebugMenu',    None, _('Debug')),


            ('ShortcutsMenu', None, _('Shortcuts')),

            ('ViewMenu', None, _('View')),
            ('Fullscreen',   gtk.STOCK_FULLSCREEN, _('Fullscreen'), 'F11', None, self.fullscreen_cb),
            ('ResetView',   gtk.STOCK_ZOOM_100, _('Reset (Zoom, Rotation, Mirror)'), None, None, self.reset_view_cb),
            ('ZoomIn',       gtk.STOCK_ZOOM_IN, _('Zoom In (at cursor)'), 'period', None, self.zoom_cb),
            ('ZoomOut',      gtk.STOCK_ZOOM_OUT, _('Zoom Out'), 'comma', None, self.zoom_cb),
            ('RotateLeft',   None, _('Rotate Counterclockwise'), None, None, self.rotate_cb),
            ('RotateRight',  None, _('Rotate Clockwise'), None, None, self.rotate_cb),
            ('SoloLayer',    None, _('Layer Solo'), 'Home', None, self.solo_layer_cb),
            ('ToggleAbove',  None, _('Hide Layers Above Current'), 'End', None, self.toggle_layers_above_cb), # TODO: make toggle action
            ('ViewHelp',  gtk.STOCK_HELP, _('Help'), None, None, self.show_infodialog_cb),
            ]
        ag = self.action_group = gtk.ActionGroup('WindowActions')
        ag.add_actions(actions)
        context_actions = []
        for x in range(10):
            r = ('Context0%d' % x,    None, _('Restore Brush %d') % x, 
                    '%d' % x, None, self.context_cb)
            s = ('Context0%ds' % x,   None, _('Save to Brush %d') % x, 
                    '<control>%d' % x, None, self.context_cb)
            context_actions.append(s)
            context_actions.append(r)
        ag.add_actions(context_actions)
        toggle_actions = [
            # name, stock id, label, accelerator, tooltip, callback, default toggle status
            ('PrintInputs', None, _('Print Brush Input Values to stdout'), None, None, self.print_inputs_cb),
            ('VisualizeRendering', None, _('Visualize Rendering'), None, None, self.visualize_rendering_cb),
            ('NoDoubleBuffereing', None, _('Disable GTK Double Buffering'), None, None, self.no_double_buffering_cb),
            ('Flip', None, _('Mirror Image'), 'i', None, self.flip_cb),
            ]
        ag.add_toggle_actions(toggle_actions)
        self.app.ui_manager.insert_action_group(ag, -1)
        menupath = os.path.join(self.app.datapath, 'gui/menu.xml')
        self.app.ui_manager.add_ui_from_file(menupath)
        #self.app.accel_group = self.app.ui_manager.get_accel_group()

        kbm = self.app.kbm
        kbm.add_window(self)

        for action in ag.list_actions():
            self.app.kbm.takeover_action(action)

        kbm.add_extra_key('<control>z', 'Undo')
        kbm.add_extra_key('<control>y', 'Redo')
        kbm.add_extra_key('KP_Add', 'ZoomIn')
        kbm.add_extra_key('KP_Subtract', 'ZoomOut')
        kbm.add_extra_key('plus', 'ZoomIn')
        kbm.add_extra_key('minus', 'ZoomOut')

        kbm.add_extra_key('Left', lambda(action): self.move('MoveLeft'))
        kbm.add_extra_key('Right', lambda(action): self.move('MoveRight'))
        kbm.add_extra_key('Down', lambda(action): self.move('MoveDown'))
        kbm.add_extra_key('Up', lambda(action): self.move('MoveUp'))

        kbm.add_extra_key('<control>Left', 'RotateLeft')
        kbm.add_extra_key('<control>Right', 'RotateRight')

        sg = stategroup.StateGroup()
        self.layerblink_state = sg.create_state(self.layerblink_state_enter, self.layerblink_state_leave)

        sg = stategroup.StateGroup()
        self.strokeblink_state = sg.create_state(self.strokeblink_state_enter, self.strokeblink_state_leave)
        self.strokeblink_state.autoleave_timeout = 0.3

        # separate stategroup...
        sg2 = stategroup.StateGroup()
        self.layersolo_state = sg2.create_state(self.layersolo_state_enter, self.layersolo_state_leave)
        self.layersolo_state.autoleave_timeout = None

        p2s = sg.create_popup_state
        changer = p2s(colorselectionwindow.ColorChangerPopup(self.app))
        ring = p2s(colorselectionwindow.ColorRingPopup(self.app))
        hist = p2s(historypopup.HistoryPopup(self.app, self.doc))
        pick = self.colorpick_state = p2s(colorpicker.ColorPicker(self.app, self.doc))

        self.popup_states = {
            'ColorChangerPopup': changer,
            'ColorRingPopup': ring,
            'ColorHistoryPopup': hist,
            'ColorPickerPopup': pick,
            }
        changer.next_state = ring
        ring.next_state = changer
        changer.autoleave_timeout = None
        ring.autoleave_timeout = None

        pick.max_key_hit_duration = 0.0
        pick.autoleave_timeout = None

        hist.autoleave_timeout = 0.600
        self.history_popup_state = hist

    def drag_data_received(self, widget, context, x, y, selection, info, t):
        if selection.data:
            uri = selection.data.split("\r\n")[0]
            fn = helpers.get_file_path_from_dnd_dropped_uri(uri)
            if os.path.exists(fn):
                if self.app.filehandler.confirm_destructive_action():
                    self.app.filehandler.open_file(fn)

    def toggleWindow_cb(self, action):
        s = action.get_name()
        s = s[0].lower() + s[1:]
        w = getattr(self.app, s)
        if w.window and w.window.is_visible():
            w.hide()
        else:
            w.show_all() # might be for the first time
            w.present()

    def print_inputs_cb(self, action):
        self.doc.brush.print_inputs = action.get_active()

    def visualize_rendering_cb(self, action):
        self.tdw.visualize_rendering = action.get_active()
    def no_double_buffering_cb(self, action):
        self.tdw.set_double_buffered(not action.get_active())

    def undo_cb(self, action):
        cmd = self.doc.undo()
        if isinstance(cmd, command.MergeLayer):
            # show otherwise invisible change (hack...)
            self.layerblink_state.activate()

    def redo_cb(self, action):
        cmd = self.doc.redo()
        if isinstance(cmd, command.MergeLayer):
            # show otherwise invisible change (hack...)
            self.layerblink_state.activate()

    def copy_cb(self, action):
        # use the full document bbox, so we can past layers back to the correct position
        bbox = self.doc.get_bbox()
        pixbuf = self.doc.layer.surface.render_as_pixbuf(*bbox)
        cb = gtk.Clipboard()
        cb.set_image(pixbuf)

    def paste_cb(self, action):
        cb = gtk.Clipboard()
        def callback(clipboard, pixbuf, trash):
            if not pixbuf:
                print 'The clipboard doeas not contain any image to paste!'
                return
            # paste to the upper left of our doc bbox (see above)
            x, y, w, h = self.doc.get_bbox()
            self.doc.load_layer_from_pixbuf(pixbuf, x, y)
        cb.request_image(callback)

    def brush_modified_cb(self):
        # called at every brush setting modification, should return fast
        self.doc.set_brush(self.app.brush)

    def key_press_event_cb_before(self, win, event):
        key = event.keyval 
        ctrl = event.state & gdk.CONTROL_MASK
        #ANY_MODIFIER = gdk.SHIFT_MASK | gdk.MOD1_MASK | gdk.CONTROL_MASK
        #if event.state & ANY_MODIFIER:
        #    # allow user shortcuts with modifiers
        #    return False
        if key == keysyms.space:
            if ctrl:
                self.tdw.start_drag(self.dragfunc_rotate)
            else:
                self.tdw.start_drag(self.dragfunc_translate)
        else: return False
        return True
    def key_release_event_cb_before(self, win, event):
        if event.keyval == keysyms.space:
            self.tdw.stop_drag(self.dragfunc_translate)
            self.tdw.stop_drag(self.dragfunc_rotate)
            return True
        return False

    def key_press_event_cb_after(self, win, event):
        key = event.keyval
        if self.fullscreen and key == keysyms.Escape: self.fullscreen_cb()
        else: return False
        return True
    def key_release_event_cb_after(self, win, event):
        return False

    def dragfunc_translate(self, dx, dy):
        self.tdw.scroll(-dx, -dy)

    def dragfunc_rotate(self, dx, dy):
        self.tdw.scroll(-dx, -dy, False)
        self.tdw.rotate(2*math.pi*dx/500.0)

    #def dragfunc_rotozoom(self, dx, dy):
    #    self.tdw.scroll(-dx, -dy, False)
    #    self.tdw.zoom(math.exp(-dy/100.0))
    #    self.tdw.rotate(2*math.pi*dx/500.0)

    def button_press_cb(self, win, event):
        #print event.device, event.button
        if event.type != gdk.BUTTON_PRESS:
            # ignore the extra double-click event
            return
        if event.button == 2:
            # check whether we are painting (accidental)
            pressure = event.get_axis(gdk.AXIS_PRESSURE)
            if (event.state & gdk.BUTTON1_MASK) or pressure:
                # do not allow dragging while painting (often happens accidentally)
                pass
            else:
                self.tdw.start_drag(self.dragfunc_translate)
        elif event.button == 1:
            if event.state & gdk.CONTROL_MASK:
                self.end_eraser_mode()
                self.colorpick_state.activate(event)
        elif event.button == 3:
            self.history_popup_state.activate(event)

    def button_release_cb(self, win, event):
        #print event.device, event.button
        if event.button == 2:
            self.tdw.stop_drag(self.dragfunc_translate)
        # too slow to be useful:
        #elif event.button == 3:
        #    self.tdw.stop_drag(self.dragfunc_rotate)

    def scroll_cb(self, win, event):
        d = event.direction
        if d == gdk.SCROLL_UP:
            if event.state & gdk.SHIFT_MASK:
                self.rotate('RotateLeft')
            else:
                self.zoom('ZoomIn')
        elif d == gdk.SCROLL_DOWN:
            if event.state & gdk.SHIFT_MASK:
                self.rotate('RotateRight')
            else:
                self.zoom('ZoomOut')
        elif d == gdk.SCROLL_LEFT:
            self.rotate('RotateRight')
        elif d == gdk.SCROLL_LEFT:
            self.rotate('RotateLeft')

    def clear_layer_cb(self, action):
        self.doc.clear_layer()
        if len(self.doc.layers) == 1:
            # this is like creating a new document:
            # make "save next" use a new file name
            self.app.filehandler.filename = None

    def remove_layer_cb(self, action):
        if len(self.doc.layers) == 1:
            self.doc.clear_layer()
        else:
            self.doc.remove_layer()
            self.layerblink_state.activate(action)

    def layer_bg_cb(self, action):
        idx = self.doc.layer_idx - 1
        if idx < 0:
            return
        self.doc.select_layer(idx)
        self.layerblink_state.activate(action)

    def layer_fg_cb(self, action):
        idx = self.doc.layer_idx + 1
        if idx >= len(self.doc.layers):
            return
        self.doc.select_layer(idx)
        self.layerblink_state.activate(action)

    def pick_layer_cb(self, action):
        x, y = self.tdw.get_cursor_in_model_coordinates()
        for idx, layer in reversed(list(enumerate(self.doc.layers))):
            alpha = layer.surface.get_alpha (x, y, 5) * layer.opacity
            if alpha > 0.1:
                self.doc.select_layer(idx)
                self.layerblink_state.activate(action)
                return
        self.doc.select_layer(0)
        self.layerblink_state.activate(action)

    def pick_context_cb(self, action):
        x, y = self.tdw.get_cursor_in_model_coordinates()
        for idx, layer in reversed(list(enumerate(self.doc.layers))):
            alpha = layer.surface.get_alpha (x, y, 5) * layer.opacity
            if alpha > 0.1:
                old_layer = self.doc.layer
                self.doc.select_layer(idx)
                #if self.doc.layer != old_layer:
                #    self.layerblink_state.activate(action)

                # find the most recent (last) stroke that touches our picking point
                si = self.doc.layer.get_stroke_info_at(x, y)

                if si:
                    self.app.brush.load_from_string(si.brush_string)
                    self.app.select_brush(None)
                    self.si = si # FIXME: should be a method parameter?
                    self.strokeblink_state.activate(action)
                return

    def strokeblink_state_enter(self):
        l = layer.Layer()
        self.si.render_overlay(l.surface)
        self.tdw.overlay_layer = l
        self.tdw.queue_draw() # OPTIMIZE: excess
    def strokeblink_state_leave(self, reason):
        self.tdw.overlay_layer = None
        self.tdw.queue_draw() # OPTIMIZE: excess

    def layerblink_state_enter(self):
        self.tdw.current_layer_solo = True
        self.tdw.queue_draw()
    def layerblink_state_leave(self, reason):
        if self.layersolo_state.active:
            # FIXME: use state machine concept, maybe?
            return
        self.tdw.current_layer_solo = False
        self.tdw.queue_draw()
    def layersolo_state_enter(self):
        s = self.layerblink_state
        if s.active:
            s.leave()
        self.tdw.current_layer_solo = True
        self.tdw.queue_draw()
    def layersolo_state_leave(self, reason):
        self.tdw.current_layer_solo = False
        self.tdw.queue_draw()

    #def blink_layer_cb(self, action):
    #    self.layerblink_state.activate(action)

    def solo_layer_cb(self, action):
        self.layersolo_state.toggle(action)

    def new_layer_cb(self, action):
        insert_idx = self.doc.layer_idx
        if action.get_name() == 'NewLayerFG':
            insert_idx += 1
        self.doc.add_layer(insert_idx)
        self.layerblink_state.activate(action)

    @with_wait_cursor
    def merge_layer_cb(self, action):
        dst_idx = self.doc.layer_idx - 1
        if dst_idx < 0:
            return
        self.doc.merge_layer(dst_idx)
        self.layerblink_state.activate(action)

    def toggle_layers_above_cb(self, action):
        self.tdw.toggle_show_layers_above()

    def popup_cb(self, action):
        # This doesn't really belong here...
        # just because all popups are color popups now...
        # ...maybe should eraser_mode be a GUI state too?
        self.end_eraser_mode()

        state = self.popup_states[action.get_name()]
        state.activate(action)

    def eraser_cb(self, action):
        adj = self.app.brush_adjustment['eraser']
        if adj.get_value() > 0.9:
            self.end_eraser_mode()
        else:
            # enter eraser mode
            adj.set_value(1.0)
            adj2 = self.app.brush_adjustment['radius_logarithmic']
            r = adj2.get_value()
            self.eraser_mode_original_radius = r
            adj2.set_value(r + self.eraser_mode_radius_change)

    def end_eraser_mode(self):
        adj = self.app.brush_adjustment['eraser']
        if not adj.get_value() > 0.9:
            return
        adj.set_value(0.0)
        if self.eraser_mode_original_radius:
            # save eraser radius, restore old radius
            adj2 = self.app.brush_adjustment['radius_logarithmic']
            r = adj2.get_value()
            self.eraser_mode_radius_change = r - self.eraser_mode_original_radius
            adj2.set_value(self.eraser_mode_original_radius)
            self.eraser_mode_original_radius = None

    def device_changed_cb(self, old_device, new_device):
        # just enable eraser mode for now (TODO: remember full tool settings)
        # small problem with this code: it doesn't work well with brushes that have (eraser not in [1.0, 0.0])
        def is_eraser(device):
            if device is None: return False
            return device.source == gdk.SOURCE_ERASER or 'eraser' in device.name.lower()
        if old_device is None and not is_eraser(new_device):
            # keep whatever startup brush was choosen
            return

        print 'device change:', new_device.name, new_device.source

        self.app.brush_by_device[old_device.name] = (self.app.selected_brush, self.app.brush.save_to_string())

        if new_device.name in self.app.brush_by_device:
            brush_to_select, brush_settings = self.app.brush_by_device[new_device.name]
            # mark as selected in brushlist
            self.app.select_brush(brush_to_select)
            # restore modifications (radius / color change the user made)
            self.app.brush.load_from_string(brush_settings)
        else:
            # first time using that device
            adj = self.app.brush_adjustment['eraser']
            if is_eraser(new_device):
                # enter eraser mode
                adj.set_value(1.0)
            elif not is_eraser(new_device) and is_eraser(old_device):
                # leave eraser mode
                adj.set_value(0.0)

    def brush_bigger_cb(self, action):
        adj = self.app.brush_adjustment['radius_logarithmic']
        adj.set_value(adj.get_value() + 0.3)
    def brush_smaller_cb(self, action):
        adj = self.app.brush_adjustment['radius_logarithmic']
        adj.set_value(adj.get_value() - 0.3)

    def more_opaque_cb(self, action):
        # FIXME: hm, looks this slider should be logarithmic?
        adj = self.app.brush_adjustment['opaque']
        adj.set_value(adj.get_value() * 1.8)
    def less_opaque_cb(self, action):
        adj = self.app.brush_adjustment['opaque']
        adj.set_value(adj.get_value() / 1.8)

    def brighter_cb(self, action):
        self.end_eraser_mode()
        h, s, v = self.app.brush.get_color_hsv()
        v += 0.08
        if v > 1.0: v = 1.0
        self.app.brush.set_color_hsv((h, s, v))
    def darker_cb(self, action):
        self.end_eraser_mode()
        h, s, v = self.app.brush.get_color_hsv()
        v -= 0.08
        if v < 0.0: v = 0.0
        self.app.brush.set_color_hsv((h, s, v))

    def layer_increase_opacity(self, action):
        opa = helpers.clamp(self.doc.layer.opacity + 0.08, 0.0, 1.0)
        self.doc.set_layer_opacity(opa)

    def layer_decrease_opacity(self, action):
        opa = helpers.clamp(self.doc.layer.opacity - 0.08, 0.0, 1.0)
        self.doc.set_layer_opacity(opa)

    def quit_cb(self, *trash):
        self.doc.split_stroke()
        self.app.save_gui_config() # FIXME: should do this periodically, not only on quit

        if not self.app.filehandler.confirm_destructive_action(title=_('Quit'), question=_('Really Quit?')):
            return True

        gtk.main_quit()
        return False

    def zoom_cb(self, action):
        self.zoom(action.get_name())
    def rotate_cb(self, action):
        self.rotate(action.get_name())
    def flip_cb(self, action):
        self.tdw.set_flipped(action.get_active())

    def move(self, command):
        self.doc.split_stroke()
        step = min(self.tdw.window.get_size()) / 5
        if   command == 'MoveLeft' : self.tdw.scroll(-step, 0)
        elif command == 'MoveRight': self.tdw.scroll(+step, 0)
        elif command == 'MoveUp'   : self.tdw.scroll(0, -step)
        elif command == 'MoveDown' : self.tdw.scroll(0, +step)
        else: assert 0

    def zoom(self, command):
        if   command == 'ZoomIn' : self.zoomlevel += 1
        elif command == 'ZoomOut': self.zoomlevel -= 1
        else: assert 0
        if self.zoomlevel < 0: self.zoomlevel = 0
        if self.zoomlevel >= len(self.zoomlevel_values): self.zoomlevel = len(self.zoomlevel_values) - 1
        z = self.zoomlevel_values[self.zoomlevel]
        self.tdw.set_zoom(z)

    def rotate(self, command):
        if   command == 'RotateRight': self.tdw.rotate(+2*math.pi/14)
        elif command == 'RotateLeft' : self.tdw.rotate(-2*math.pi/14)
        else: assert 0

    def reset_view_cb(self, command):
        self.tdw.set_rotation(0.0)
        self.zoomlevel = self.zoomlevel_values.index(1.0)
        self.tdw.set_zoom(1.0)
        self.tdw.set_flipped(False)
        self.action_group.get_action('Flip').set_active(False)

    def fullscreen_cb(self, *trash):
        # note: there is some ugly flickering when toggling fullscreen
        #       self.window.begin_paint/end_paint does not help against it
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            x, y = self.get_position()
            w, h = self.get_size()
            self.geometry_before_fullscreen = (x, y, w, h)
            self.menubar.hide()
            self.window.fullscreen()
            #self.tdw.set_scroll_at_edges(True)
        else:
            self.window.unfullscreen()
            self.menubar.show()
            #self.tdw.set_scroll_at_edges(False)
            del self.geometry_before_fullscreen

    def context_cb(self, action):
        name = action.get_name()
        store = False
        if name == 'ContextStore':
            context = self.app.selected_context
            if not context:
                print 'No context was selected, ignoring store command.'
                return
            store = True
        else:
            if name.endswith('s'):
                store = True
                name = name[:-1]
            i = int(name[-2:])
            context = self.app.contexts[i]
        self.app.selected_context = context
        if store:
            context.copy_settings_from(self.app.brush)
            preview = self.app.brushSelectionWindow.get_preview_pixbuf()
            context.update_preview(preview)
            context.save()
        else:
            # restore (but keep color)
            color = self.app.brush.get_color_hsv()
            context.set_color_hsv(color)
            self.app.select_brush(context)
            self.app.brushSelectionWindow.set_preview_pixbuf(context.preview)

    def about_cb(self, action):
        d = gtk.AboutDialog()
        d.set_transient_for(self)
        d.set_program_name("MyPaint")
        d.set_version(MYPAINT_VERSION)
        d.set_copyright(_("Copyright (C) 2005-2009\nMartin Renold and the MyPaint Development Team"))
        d.set_website("http://mypaint.info/")
        d.set_logo(self.app.pixmaps.mypaint_logo)
        d.set_license(
            _(u"This program is free software; you can redistribute it and/or modify "
              u"it under the terms of the GNU General Public License as published by "
              u"the Free Software Foundation; either version 2 of the License, or "
              u"(at your option) any later version.\n"
              u"\n"
              u"This program is distributed in the hope that it will be useful, "
              u"but WITHOUT ANY WARRANTY. See the COPYING file for more details.")
            )
        d.set_wrap_license(True)
        d.set_authors([
            u"Martin Renold (%s)" % _('programming'),
            u'Artis Rozentāls (%s)' % _('brushes'),
            u'Yves Combe (%s)' % _('portability'),
            u'Popolon (%s)' % _('brushes, programming'),
            u'Clement Skau (%s)' % _('programming'),
            u"Marcelo 'Tanda' Cerviño (%s)" % _('patterns, brushes'),
            u'Jon Nordby (%s)' % _('programming'),
            u'Álinson Santos (%s)' % _('programming'),
            u'Tumagonx (%s)' % _('portability'),
            u'Ilya Portnov (%s)' % _('programming'),
            ])
        d.set_artists([
            u'Sebastian Kraft (%s)' % _('desktop icon'),
            ])
        # list all translators, not only those of the current language
        d.set_translator_credits(
            u'Ilya Portnov (ru)\n'
            u'Popolon (fr, zh_CN)\n'
            u'Jon Nordby (nb)\n'
            u'Griatch (sv)\n'
            u'Tobias Jakobs (de)\n'
            u'Martin Tabačan (cs)\n'
            u'Tumagonx (id)\n'
            u'Manuel Quiñones (es)\n'
            u'Gergely Aradszki (hu)\n'
            )
        
        d.run()
        d.destroy()

    def show_infodialog_cb(self, action):
        text = {
        'ShortcutHelp': 
                _("Move your mouse over a menu entry, then press the key to assign."),
        'ViewHelp': 
                _("You can also drag the canvas with the mouse while holding the middle "
                "mouse button or spacebar. Or with the arrow keys."
                "\n\n"
                "In contrast to earlier versions, scrolling and zooming are harmless now and "
                "will not make you run out of memory. But you still require a lot of memory "
                "if you paint all over while fully zoomed out."),
        'ContextHelp':
                _("This is used to quickly save/restore brush settings "
                 "using keyboard shortcuts. You can paint with one hand and "
                 "change brushes with the other without interrupting."
                 "\n\n"
                 "There are 10 memory slots to hold brush settings.\n"
                 "Those are annonymous "
                 "brushes, they are not visible in the brush selector list. "
                 "But they will stay even if you quit. "
                 "They will also remember the selected color. In contrast, selecting a "
                 "normal brush never changes the color. "),
        'Docu':
                _("There is a tutorial available "
                 "on the MyPaint homepage. It explains some features which are "
                 "hard to discover yourself.\n\n"
                 "Comments about the brush settings (opaque, hardness, etc.) and "
                 "inputs (pressure, speed, etc.) are available as tooltips. "
                 "Put your mouse over a label to see them. "
                 "\n"),
        }
        self.app.message_dialog(text[action.get_name()])
