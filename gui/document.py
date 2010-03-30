# -*- coding: utf-8 -*-
#
# This file is part of MyPaint.
# Copyright (C) 2007-2010 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os, math

import gtk
from gtk import gdk
from gettext import gettext as _

import lib.document
from lib import backgroundsurface, command, helpers, layer
import tileddrawwidget, stategroup


class Document(object):
    def __init__(self, app):
        self.app = app
        self.model = lib.document.Document()
        self.model.set_brush(self.app.brush)

        # View
        self.tdw = tileddrawwidget.TiledDrawWidget(self.model)

        # FIXME: hack, to be removed
        fname = os.path.join(self.app.datapath, 'backgrounds', '03_check1.png')
        pixbuf = gdk.pixbuf_new_from_file(fname)
        self.tdw.neutral_background_pixbuf = backgroundsurface.Background(pixbuf)

        self.zoomlevel_values = [1.0/8, 2.0/11, 0.25, 1.0/3, 0.50, 2.0/3, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 8.0]
        self.zoomlevel = self.zoomlevel_values.index(1.0)
        self.tdw.zoom_min = min(self.zoomlevel_values)
        self.tdw.zoom_max = max(self.zoomlevel_values)

        # Brush
        self.app.brush.settings_observers.append(self.brush_modified_cb)
        self.tdw.device_observers.append(self.device_changed_cb)

        self.last_pen_device = None
        self.eraser_mode_radius_change = 3*(0.3) # can go back to exact original with brush_smaller_cb()
        self.eraser_mode_original_radius = None

        self.init_actions()
        self.init_context_actions()
        self.app.ui_manager.insert_action_group(self.action_group, -1)
        for action in self.action_group.list_actions():
            self.app.kbm.takeover_action(action)
        self.init_stategroups()
        self.init_extra_keys()

    def init_actions(self):
        # name, stock id, label, accelerator, tooltip, callback
        actions = [
            ('Undo',               gtk.STOCK_UNDO, _('Undo'), 'Z', None, self.undo_cb),
            ('Redo',               gtk.STOCK_REDO, _('Redo'), 'Y', None, self.redo_cb),

            ('Brighter',     None, _('Brighter'), None, None, self.brighter_cb),
            ('Smaller',      None, _('Smaller'), 'd', None, self.brush_smaller_cb),
            ('MoreOpaque',   None, _('More Opaque'), 's', None, self.more_opaque_cb),
            ('LessOpaque',   None, _('Less Opaque'), 'a', None, self.less_opaque_cb),
            ('Eraser',       None, _('Toggle Eraser Mode'), 'e', None, self.eraser_cb), # TODO: make toggle action
            ('PickContext',  None, _('Pick Context (layer, brush and color)'), 'w', None, self.pick_context_cb),

            ('Darker',       None, _('Darker'), None, None, self.darker_cb),
            ('Bigger',       None, _('Bigger'), 'f', None, self.brush_bigger_cb),

            # Context actions are also added in init_context_actions
            ('ContextStore', None, _('Save to Most Recently Restored'), 'q', None, self.context_cb),

            ('ClearLayer',   gtk.STOCK_CLEAR, _('Clear'), 'Delete', None, self.clear_layer_cb),
            ('CopyLayer',          gtk.STOCK_COPY, _('Copy to Clipboard'), '<control>C', None, self.copy_cb),
            ('PasteLayer',         gtk.STOCK_PASTE, _('Paste Clipboard (Replace Layer)'), '<control>V', None, self.paste_cb),
            ('PickLayer',    None, _('Select Layer at Cursor'), 'h', None, self.pick_layer_cb),
            ('LayerFG',      None, _('Next (above current)'),  'Page_Up', None, self.layer_fg_cb),
            ('LayerBG',      None, _('Next (below current)'), 'Page_Down', None, self.layer_bg_cb),
            ('NewLayerFG',   None, _('New (above current)'), '<control>Page_Up', None, self.new_layer_cb),
            ('NewLayerBG',   None, _('New (below current)'), '<control>Page_Down', None, self.new_layer_cb),
            ('MergeLayer',   None, _('Merge Down'), '<control>Delete', None, self.merge_layer_cb),
            ('RemoveLayer',  gtk.STOCK_DELETE, _('Remove'), '<shift>Delete', None, self.remove_layer_cb),
            ('IncreaseLayerOpacity', None, _('Increase Layer Opacity'),  'p', None, self.layer_increase_opacity),
            ('DecreaseLayerOpacity', None, _('Decrease Layer Opacity'),  'o', None, self.layer_decrease_opacity),

            ('ShortcutsMenu', None, _('Shortcuts')),

            ('ResetView',   gtk.STOCK_ZOOM_100, _('Reset (Zoom, Rotation, Mirror)'), 'F12', None, self.reset_view_cb),
            ('ZoomIn',       gtk.STOCK_ZOOM_IN, _('Zoom In (at cursor)'), 'period', None, self.zoom_cb),
            ('ZoomOut',      gtk.STOCK_ZOOM_OUT, _('Zoom Out'), 'comma', None, self.zoom_cb),
            ('RotateLeft',   None, _('Rotate Counterclockwise'), None, None, self.rotate_cb),
            ('RotateRight',  None, _('Rotate Clockwise'), None, None, self.rotate_cb),
            ('SoloLayer',    None, _('Layer Solo'), 'Home', None, self.solo_layer_cb), # TODO: make toggle action
            ('ToggleAbove',  None, _('Hide Layers Above Current'), 'End', None, self.toggle_layers_above_cb), # TODO: make toggle action
        ]
        ag = self.action_group = gtk.ActionGroup('DocumentActions')
        ag.add_actions(actions)

        toggle_actions = [
            # name, stock id, label, accelerator, tooltip, callback, default toggle status
            ('PrintInputs', None, _('Print Brush Input Values to stdout'), None, None, self.print_inputs_cb),
            ('VisualizeRendering', None, _('Visualize Rendering'), None, None, self.visualize_rendering_cb),
            ('NoDoubleBuffereing', None, _('Disable GTK Double Buffering'), None, None, self.no_double_buffering_cb),
            ('Flip', None, _('Mirror Image'), 'i', None, self.flip_cb),
            ]
        ag.add_toggle_actions(toggle_actions)

    def init_context_actions(self):
        ag = self.action_group
        context_actions = []
        for x in range(10):
            r = ('Context0%d' % x,    None, _('Restore Brush %d') % x, 
                    '%d' % x, None, self.context_cb)
            s = ('Context0%ds' % x,   None, _('Save to Brush %d') % x, 
                    '<control>%d' % x, None, self.context_cb)
            context_actions.append(s)
            context_actions.append(r)
        ag.add_actions(context_actions)

    def init_stategroups(self):
        sg = stategroup.StateGroup()
        self.layerblink_state = sg.create_state(self.layerblink_state_enter, self.layerblink_state_leave)

        sg = stategroup.StateGroup()
        self.strokeblink_state = sg.create_state(self.strokeblink_state_enter, self.strokeblink_state_leave)
        self.strokeblink_state.autoleave_timeout = 0.3

        # separate stategroup...
        sg2 = stategroup.StateGroup()
        self.layersolo_state = sg2.create_state(self.layersolo_state_enter, self.layersolo_state_leave)
        self.layersolo_state.autoleave_timeout = None

    def init_extra_keys(self):
        kbm = self.app.kbm
        kbm.add_extra_key('<control>z', 'Undo')
        kbm.add_extra_key('<control>y', 'Redo')
        kbm.add_extra_key('<control><shift>z', 'Redo')
        kbm.add_extra_key('KP_Add', 'ZoomIn')
        kbm.add_extra_key('KP_Subtract', 'ZoomOut')
        kbm.add_extra_key('plus', 'ZoomIn')
        kbm.add_extra_key('minus', 'ZoomOut')

        kbm.add_extra_key('Left', lambda(action): self.pan('PanLeft'))
        kbm.add_extra_key('Right', lambda(action): self.pan('PanRight'))
        kbm.add_extra_key('Down', lambda(action): self.pan('PanDown'))
        kbm.add_extra_key('Up', lambda(action): self.pan('PanUp'))

        kbm.add_extra_key('<control>Left', 'RotateLeft')
        kbm.add_extra_key('<control>Right', 'RotateRight')

    # GENERIC
    def undo_cb(self, action):
        cmd = self.model.undo()
        if isinstance(cmd, command.MergeLayer):
            # show otherwise invisible change (hack...)
            self.layerblink_state.activate()

    def redo_cb(self, action):
        cmd = self.model.redo()
        if isinstance(cmd, command.MergeLayer):
            # show otherwise invisible change (hack...)
            self.layerblink_state.activate()

    def copy_cb(self, action):
        # use the full document bbox, so we can past layers back to the correct position
        bbox = self.model.get_bbox()
        if bbox.w == 0 or bbox.h == 0:
            print "WARNING: empty document, nothing copied"
            return
        else:
            pixbuf = self.model.layer.surface.render_as_pixbuf(*bbox)
        cb = gtk.Clipboard()
        cb.set_image(pixbuf)

    def paste_cb(self, action):
        cb = gtk.Clipboard()
        def callback(clipboard, pixbuf, trash):
            if not pixbuf:
                print 'The clipboard doeas not contain any image to paste!'
                return
            # paste to the upper left of our doc bbox (see above)
            x, y, w, h = self.model.get_bbox()
            self.model.load_layer_from_pixbuf(pixbuf, x, y)
        cb.request_image(callback)

    def brush_modified_cb(self):
        # called at every brush setting modification, should return fast
        self.model.set_brush(self.app.brush)


    def pick_context_cb(self, action):
        x, y = self.tdw.get_cursor_in_model_coordinates()
        for idx, layer in reversed(list(enumerate(self.model.layers))):
            alpha = layer.surface.get_alpha (x, y, 5) * layer.effective_opacity
            if alpha > 0.1:
                old_layer = self.model.layer
                self.model.select_layer(idx)
                if self.model.layer != old_layer:
                    self.layerblink_state.activate()

                # find the most recent (last) stroke that touches our picking point
                si = self.model.layer.get_stroke_info_at(x, y)

                if si:
                    self.app.brushmanager.select_brush(None) # FIXME: restore the selected brush
                    self.app.brush.load_from_string(si.brush_string)
                    self.si = si # FIXME: should be a method parameter?
                    self.strokeblink_state.activate(action)
                return

    # LAYER
    def clear_layer_cb(self, action):
        self.model.clear_layer()
        if self.model.is_empty():
            # the user started a new painting
            self.app.filehandler.filename = None

    def remove_layer_cb(self, action):
        self.model.remove_layer()
        if self.model.is_empty():
            # the user started a new painting
            self.app.filehandler.filename = None

    def layer_bg_cb(self, action):
        idx = self.model.layer_idx - 1
        if idx < 0:
            return
        self.model.select_layer(idx)
        self.layerblink_state.activate(action)

    def layer_fg_cb(self, action):
        idx = self.model.layer_idx + 1
        if idx >= len(self.model.layers):
            return
        self.model.select_layer(idx)
        self.layerblink_state.activate(action)

    def layer_increase_opacity(self, action):
        opa = helpers.clamp(self.model.layer.opacity + 0.08, 0.0, 1.0)
        self.model.set_layer_opacity(opa)

    def layer_decrease_opacity(self, action):
        opa = helpers.clamp(self.model.layer.opacity - 0.08, 0.0, 1.0)
        self.model.set_layer_opacity(opa)

    def solo_layer_cb(self, action):
        self.layersolo_state.toggle(action)

    def new_layer_cb(self, action):
        insert_idx = self.model.layer_idx
        if action.get_name() == 'NewLayerFG':
            insert_idx += 1
        self.model.add_layer(insert_idx)
        self.layerblink_state.activate(action)

#     @with_wait_cursor
    def merge_layer_cb(self, action):
        if self.model.merge_layer_down():
            self.layerblink_state.activate(action)

    def toggle_layers_above_cb(self, action):
        self.tdw.toggle_show_layers_above()

    def pick_layer_cb(self, action):
        x, y = self.tdw.get_cursor_in_model_coordinates()
        for idx, layer in reversed(list(enumerate(self.model.layers))):
            alpha = layer.surface.get_alpha (x, y, 5) * layer.effective_opacity
            if alpha > 0.1:
                self.model.select_layer(idx)
                self.layerblink_state.activate(action)
                return
        self.model.select_layer(0)
        self.layerblink_state.activate(action)

    # BRUSH
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

    def context_cb(self, action):
        name = action.get_name()
        store = False
        bm = self.app.brushmanager
        if name == 'ContextStore':
            context = bm.selected_context
            if not context:
                print 'No context was selected, ignoring store command.'
                return
            store = True
        else:
            if name.endswith('s'):
                store = True
                name = name[:-1]
            i = int(name[-2:])
            context = bm.contexts[i]
        bm.selected_context = context
        if store:
            context.copy_settings_from(self.app.brush)
            context.preview = bm.selected_brush.preview
            context.save()
        else:
            # restore (but keep color)
            color = self.app.brush.get_color_hsv()
            context.set_color_hsv(color)
            bm.select_brush(context)

    # TDW view manipulation
    def dragfunc_translate(self, dx, dy):
        self.tdw.scroll(-dx*3, -dy*3)

    def dragfunc_rotate(self, dx, dy):
        self.tdw.scroll(-dx, -dy)
        self.tdw.rotate(2*math.pi*dx/500.0)

    #def dragfunc_rotozoom(self, dx, dy):
    #    self.tdw.scroll(-dx, -dy)
    #    self.tdw.zoom(math.exp(-dy/100.0))
    #    self.tdw.rotate(2*math.pi*dx/500.0)

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

    def pan(self, command):
        self.model.split_stroke()
        step = min(self.tdw.window.get_size()) / 5
        if   command == 'PanLeft' : self.tdw.scroll(-step, 0)
        elif command == 'PanRight': self.tdw.scroll(+step, 0)
        elif command == 'PanUp'   : self.tdw.scroll(0, -step)
        elif command == 'PanDown' : self.tdw.scroll(0, +step)
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

    def zoom_cb(self, action):
        self.zoom(action.get_name())
    def rotate_cb(self, action):
        self.rotate(action.get_name())
    def flip_cb(self, action):
        self.tdw.set_flipped(action.get_active())

    def reset_view_cb(self, command):
        self.tdw.set_rotation(0.0)
        self.zoomlevel = self.zoomlevel_values.index(1.0)
        self.tdw.set_zoom(1.0)
        self.tdw.set_flipped(False)
        self.action_group.get_action('Flip').set_active(False)
        self.tdw.recenter_document()

    # DEBUGGING
    def print_inputs_cb(self, action):
        self.model.brush.print_inputs = action.get_active()
    def visualize_rendering_cb(self, action):
        self.tdw.visualize_rendering = action.get_active()
    def no_double_buffering_cb(self, action):
        self.tdw.set_double_buffered(not action.get_active())


    # ERASER
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

        print 'device change:', new_device.name, new_device.source

        # When editing brush settings, it is often more convenient to use the mouse.
        # Because of this, we don't restore brushsettings when switching to/from the mouse.
        # We act as if the mouse was identical to the last active pen device.
        if new_device.source == gdk.SOURCE_MOUSE and self.last_pen_device:
            new_device = self.last_pen_device
        if new_device.source == gdk.SOURCE_PEN:
            self.last_pen_device = new_device
        if old_device and old_device.source == gdk.SOURCE_MOUSE and self.last_pen_device:
            old_device = self.last_pen_device

        bm = self.app.brushmanager
        if old_device:
            bm.brush_by_device[old_device.name] = (bm.selected_brush, self.app.brush.save_to_string())


        if new_device.name in bm.brush_by_device:
            brush_to_select, brush_settings = bm.brush_by_device[new_device.name]
            # mark as selected in brushlist
            bm.select_brush(brush_to_select)
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
