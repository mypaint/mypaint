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
from bisect import bisect_left

import lib.document
from lib import backgroundsurface, command, helpers, layer
import tileddrawwidget, stategroup
from brushmanager import ManagedBrush

# This value allows the user to go back to the exact original size with brush_smaller_cb().
ERASER_MODE_RADIUS_CHANGE_DEFAULT = 3*(0.3)
ERASER_MODE_RADIUS_CHANGE_PREF = 'document.eraser_mode_radius_change'

class Document(object):
    def __init__(self, app):
        self.app = app
        self.model = lib.document.Document()
        self.model.set_brush(self.app.brush)

        # View
        self.tdw = tileddrawwidget.TiledDrawWidget(self.model)
        self.model.frame_observers.append(self.frame_changed_cb)

        # FIXME: hack, to be removed
        fname = os.path.join(self.app.datapath, 'backgrounds', '03_check1.png')
        pixbuf = gdk.pixbuf_new_from_file(fname)
        self.tdw.neutral_background_pixbuf = backgroundsurface.Background(pixbuf)

        self.zoomlevel_values = [1.0/8, 2.0/11, 0.25, 1.0/3, 0.50, 2.0/3,  # micro
                                 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 8.0,        # normal
                                 11.0, 16.0, 23.0, 32.0, 45.0, 64.0]       # macro
                                 # keep sorted for bisect

        default_zoom = self.app.preferences['view.default_zoom']
        self.zoomlevel = min(bisect_left(self.zoomlevel_values, default_zoom),
                             len(self.zoomlevel_values) - 1)
        default_zoom = self.zoomlevel_values[self.zoomlevel]
        self.tdw.scale = default_zoom
        self.tdw.zoom_min = min(self.zoomlevel_values)
        self.tdw.zoom_max = max(self.zoomlevel_values)

        # Brush
        self.app.brush.settings_observers.append(self.brush_modified_cb)
        self.app.brushmanager.selected_brush_observers.append(self.brush_selected_cb)

        # Device change management & pen-stroke watching
        self.tdw.device_observers.append(self.device_changed_cb)
        self.input_stroke_ended_observers.append(self.input_stroke_ended_cb)
        self.last_pen_device = None
        self.last_used_devbrush_was_eraser = None

        # Eraser mode
        self.eraser_mode_original_settings = None

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
            ('Warmer',       None, _('Warmer'), None, None, self.warmer_cb),
            ('Cooler',       None, _('Cooler'), None, None, self.cooler_cb),
            ('Purer',        None, _('Purer'), None, None, self.purer_cb),
            ('Grayer',       None, _('Grayer'), None, None, self.grayer_cb),
            ('Bigger',       None, _('Bigger'), 'f', None, self.brush_bigger_cb),

            # Context actions are also added in init_context_actions
            ('ContextStore', None, _('Save to Most Recently Restored'), 'q', None, self.context_cb),

            ('ClearLayer',   gtk.STOCK_CLEAR, _('Clear'), 'Delete', None, self.clear_layer_cb),
            ('CopyLayer',          gtk.STOCK_COPY, _('Copy to Clipboard'), '<control>C', None, self.copy_cb),
            ('PasteLayer',         gtk.STOCK_PASTE, _('Paste Clipboard (Replace Layer)'), '<control>V', None, self.paste_cb),
            ('PickLayer',    gtk.STOCK_JUMP_TO, _('Select Layer at Cursor'), 'h', None, self.pick_layer_cb),
            ('LayerFG',      gtk.STOCK_GO_UP, _('Next (above current)'),  'Page_Up', None, self.layer_fg_cb),
            ('LayerBG',      gtk.STOCK_GO_DOWN, _('Next (below current)'), 'Page_Down', None, self.layer_bg_cb),
            ('NewLayerFG',   gtk.STOCK_ADD, _('New (above current)'), '<control>Page_Up', None, self.new_layer_cb),
            ('NewLayerBG',   None, _('New (below current)'), '<control>Page_Down', None, self.new_layer_cb),
            ('MergeLayer',   gtk.STOCK_DND_MULTIPLE, # XXX need a batter one, but stay consistent with layerswindow for now
                             _('Merge Down'), '<control>Delete', None, self.merge_layer_cb),
            ('RemoveLayer',  gtk.STOCK_DELETE, _('Remove'), '<shift>Delete', None, self.remove_layer_cb),
            ('IncreaseLayerOpacity', None, _('Increase Layer Opacity'),  'p', None, self.layer_increase_opacity),
            ('DecreaseLayerOpacity', None, _('Decrease Layer Opacity'),  'o', None, self.layer_decrease_opacity),

            ('ShortcutsMenu', None, _('Shortcuts')),

            ('ResetView',   gtk.STOCK_ZOOM_FIT, _('Reset and Center'), 'F12', None, self.reset_view_cb),
            ('ResetMenu',   None, _('Reset')),
                ('ResetZoom',   gtk.STOCK_ZOOM_100, _('Zoom'), None, None, self.reset_view_cb),
                ('ResetRotation',   None, _('Rotation'), None, None, self.reset_view_cb),
                ('ResetMirror', None, _('Mirror'), None, None, self.reset_view_cb),
            ('ZoomIn',       gtk.STOCK_ZOOM_IN, _('Zoom In (at cursor)'), 'period', None, self.zoom_cb),
            ('ZoomOut',      gtk.STOCK_ZOOM_OUT, _('Zoom Out'), 'comma', None, self.zoom_cb),
            ('RotateLeft',   None, _('Rotate Counterclockwise'), '<control>Left', None, self.rotate_cb),
            ('RotateRight',  None, _('Rotate Clockwise'), '<control>Right', None, self.rotate_cb),
            ('MirrorHorizontal', None, _('Mirror Horizontal'), 'i', None, self.mirror_horizontal_cb),
            ('MirrorVertical', None, _('Mirror Vertical'), 'u', None, self.mirror_vertical_cb),
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
        # The keyboard shortcuts below are not visible in the menu.
        # Shortcuts assigned through the menu will take precedence.
        # If we assign the same key twice, the last one will work.

        kbm.add_extra_key('bracketleft', 'Smaller') # GIMP, Photoshop, Painter
        kbm.add_extra_key('bracketright', 'Bigger') # GIMP, Photoshop, Painter
        kbm.add_extra_key('equal', 'ZoomIn') # (on US keyboard next to minus)
        kbm.add_extra_key('comma', 'Smaller') # Krita
        kbm.add_extra_key('period', 'Bigger') # Krita

        kbm.add_extra_key('BackSpace', 'ClearLayer')

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

    @property
    def input_stroke_ended_observers(self):
        """Array of callbacks interested in the end of an input stroke.

        Observers are called with the GTK event as their only argument. This
        is a good place to listen for "just painted something" events;
        app.brush will contain everything needed about the input stroke which
        just ended, in the state in which it ended.

        An input stroke is a single pen-down, draw, pen-up action. This sort of
        stroke is not the same as a brush engine stroke (see ``lib.document``).
        """
        return self.tdw._input_stroke_ended_observers

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

    def brush_selected_cb(self, brush):
        self.forget_eraser_mode()

    def pick_context_cb(self, action):
        x, y = self.tdw.get_cursor_in_model_coordinates()
        for idx, layer in reversed(list(enumerate(self.model.layers))):
            if layer.locked:
                continue
            if not layer.visible:
                continue
            alpha = layer.surface.get_alpha (x, y, 5) * layer.effective_opacity
            if alpha > 0.1:
                old_layer = self.model.layer
                self.model.select_layer(idx)
                if self.model.layer != old_layer:
                    self.layerblink_state.activate()

                # find the most recent (last) stroke that touches our picking point
                si = self.model.layer.get_stroke_info_at(x, y)

                if si:
                    picked_brush = ManagedBrush(self.app.brushmanager)
                    picked_brush.brushinfo.parse(si.brush_string)
                    self.app.brushmanager.select_brush(picked_brush)
                    self.forget_eraser_mode()
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
            if layer.locked:
                continue
            if not layer.visible:
                continue
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
        # stop a little higher than 0.0, to avoid resetting hue to 0
        if v < 0.005: v = 0.005
        self.app.brush.set_color_hsv((h, s, v))

    def warmer_cb(self,action):
        self.end_eraser_mode()
        h, s, v = self.app.brush.get_color_hsv()
        # using about 40 degrees (0.111 +- e) as warmest and
        # 130 deg (0.611 +- e) as coolest
        e = 0.015
        if 0.111 + e < h < 0.611:
            h -= 0.015
        elif 0.0 <= h < 0.111 - e or 0.611<= h <= 1.0:
            h += 0.015
        if h > 1.0: h -= 1.0
        if h < 0.0: h += 1.0
        self.app.brush.set_color_hsv((h, s, v))

    def cooler_cb(self,action):
        self.end_eraser_mode()
        h, s, v = self.app.brush.get_color_hsv()
        e = 0.015
        if 0.111 < h < 0.611 - e:
            h += 0.015
        elif 0.0 <= h <= 0.111 or 0.611 + e < h <= 1.0:
            h -= 0.015
        if h > 1.0: h -= 1.0
        if h < 0.0: h += 1.0
        self.app.brush.set_color_hsv((h, s, v))

    def purer_cb(self,action):
        self.end_eraser_mode()
        h, s, v = self.app.brush.get_color_hsv()
        s += 0.08
        if s > 1.0: s = 1.0
        self.app.brush.set_color_hsv((h, s, v))

    def grayer_cb(self,action):
        self.end_eraser_mode()
        h, s, v = self.app.brush.get_color_hsv()
        s -= 0.08
        # stop a little higher than 0.0, to avoid resetting hue to 0
        if s < 0.005: s = 0.005
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
            context.brushinfo = self.app.brush.brushinfo.clone()
            context.preview = bm.selected_brush.preview
            context.save()
        else:
            # restore (but keep color, see https://gna.org/bugs/index.php?16977)
            color = self.app.brush.get_color_hsv()
            bm.select_brush(context)
            self.app.brush.set_color_hsv(color)
            # XXX
            self.forget_eraser_mode()

    # TDW view manipulation
    def dragfunc_translate(self, dx, dy, x, y):
        self.tdw.scroll(-dx, -dy)
        # Accelerated scrolling: Highly comfortable to me, but lots of
        # negative feedback from users.  Certainly should be disabled
        # by default. Maybe add it back as preference one day.
        # https://gna.org/bugs/?16232
        #
        #self.tdw.scroll(-dx*3, -dy*3)

    def dragfunc_rotate(self, dx, dy, x, y):
        # calculate angular velocity from viewport center
        cx, cy = self.tdw.get_center()
        x, y = x-cx, y-cy
        phi2 = math.atan2(y, x)
        x, y = x-dx, y-dy
        phi1 = math.atan2(y, x)
        self.tdw.rotate(phi2-phi1)
        #self.tdw.rotate(2*math.pi*dx/300.0)

    def dragfunc_zoom(self, dx, dy, x, y):
        # workaround (should zoom at x=(first click point).x instead of cursor)
        self.tdw.scroll(-dx, -dy)
        # The meaning of drag direction up/down is a convention from
        # Blender and probably other 3D tools (Google Earth at least).
        self.tdw.zoom(math.exp(-dy/100.0))

    #def dragfunc_rotozoom(self, dx, dy, x, y):
    #    self.tdw.scroll(-dx, -dy)
    #    self.tdw.zoom(math.exp(-dy/100.0))
    #    self.tdw.rotate(2*math.pi*dx/500.0)

    def dragfunc_frame(self, dx, dy, x, y):
        if not self.model.frame_enabled:
            return

        x, y, w, h = self.model.get_frame()

        # Find the difference in document coordinates
        cr = self.tdw.get_model_coordinates_cairo_context()
        x0, y0 = cr.device_to_user(x, y)
        x1, y1 = cr.device_to_user(x+dx, y+dy)

        self.model.move_frame(dx=x1-x0, dy=y1-y0)

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
        # Allows easy and quick rotation to 45/90/180 degrees
        rotation_step = 2*math.pi/16

        if   command == 'RotateRight': self.tdw.rotate(+rotation_step)
        elif command == 'RotateLeft' : self.tdw.rotate(-rotation_step)
        else: assert 0

    def zoom_cb(self, action):
        self.zoom(action.get_name())
    def rotate_cb(self, action):
        self.rotate(action.get_name())
    def mirror_horizontal_cb(self, action):
        self.tdw.mirror()
    def mirror_vertical_cb(self, action):
        self.tdw.rotate(math.pi)
        self.tdw.mirror()

    def reset_view_cb(self, command):
        if command is None:
            command_name = None
            reset_all = True
        else:
            command_name = command.get_name()
            reset_all = (command_name is None) or ('View' in command_name)
        if reset_all or ('Rotation' in command_name):
            self.tdw.set_rotation(0.0)
        if reset_all or ('Zoom' in command_name):
            default_zoom = self.app.preferences['view.default_zoom']
            self.zoomlevel = self.zoomlevel_values.index(default_zoom)
            self.tdw.set_zoom(default_zoom)
        if reset_all or ('Mirror' in command_name):
            self.tdw.set_mirrored(False)
        if reset_all:
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
        if self.eraser_mode_original_settings:
            self.end_eraser_mode()
            return
        adj = self.app.brush_adjustment['eraser']
        e = adj.get_value()
        if e > 0.9:
            print "can't enter eraser mode if the current brush is an eraser"
            self.tdw.window.beep()  # TODO: better feedback
            return
        # enter eraser mode
        adj.set_value(1.0)
        adj2 = self.app.brush_adjustment['radius_logarithmic']
        r = adj2.get_value()
        self.eraser_mode_original_settings = (e, r)
        dr = self.app.preferences.get(ERASER_MODE_RADIUS_CHANGE_PREF, ERASER_MODE_RADIUS_CHANGE_DEFAULT)
        adj2.set_value(r + dr)
        # TODO: feedback to the user: "just shrink the brush three steps if this is too big"?

    def end_eraser_mode(self):
        if not self.eraser_mode_original_settings:
            return
        orig_e, orig_r = self.eraser_mode_original_settings
        adj = self.app.brush_adjustment['eraser']
        adj.set_value(orig_e)
        # save eraser mode radius change, restore old radius
        adj2 = self.app.brush_adjustment['radius_logarithmic']
        r = adj2.get_value()
        self.app.preferences[ERASER_MODE_RADIUS_CHANGE_PREF] = r - orig_r
        adj2.set_value(orig_r)
        self.forget_eraser_mode()

    def forget_eraser_mode(self):
        # Forget eraser mode states without altering the current brush.
        # Do this always after restoring something from a saved brush.
        self.eraser_mode_original_settings = None

    def clone_selected_brush_for_saving(self):
        # Clones the current brush as a new, nameless brush with all eraser
        # mode settings de-applied.
        settings_override = {}
        clone = self.app.brushmanager.clone_selected_brush(name=None)
        if self.eraser_mode_original_settings:
            orig_e, orig_r = self.eraser_mode_original_settings
            i = clone.brushinfo
            i['eraser']             = (orig_e, i.get('eraser',             (None, {}))[1])
            i['radius_logarithmic'] = (orig_r, i.get('radius_logarithmic', (None, {}))[1])
        return clone

    def device_is_eraser(self, device):
        if device is None: return False
        return device.source == gdk.SOURCE_ERASER or 'eraser' in device.name.lower()

    def device_changed_cb(self, old_device, new_device):
        # small problem with this code: it doesn't work well with brushes that have (eraser not in [1.0, 0.0])
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
            if self.device_is_eraser(old_device):
                old_brush = bm.clone_selected_brush(name=None)  # preserve user-chosen eraser mode settings
            else:
                old_brush = self.clone_selected_brush_for_saving()  # discard eraser mode settings
            bm.store_brush_for_device(old_device.name, old_brush)

        if new_device.source == gdk.SOURCE_MOUSE:
            # Avoid fouling up unrelated devbrushes at stroke end
            self.app.preferences.pop('devbrush.last_used', None)
            self.last_used_devbrush_was_eraser = None
        else:
            # Select the brush and update the UI.
            # Use a sane default if there's nothing associated
            # with the device yet.
            brush = bm.fetch_brush_for_device(new_device.name)
            if brush is None:
                if self.device_is_eraser(new_device):
                    brush = bm.get_default_eraser()
                else:
                    brush = bm.get_default_brush()
            self.app.preferences['devbrush.last_used'] = new_device.name
            self.last_used_devbrush_was_eraser = self.device_is_eraser(new_device)
            bm.select_brush(brush)
        self.forget_eraser_mode()

    def input_stroke_ended_cb(self, event):
        # Store device-specific brush settings at the end of the stroke, not
        # when the device changes because the user can change brush radii etc.
        # in the middle of a stroke, and because device_changed_cb won't
        # respond when the user fiddles with colours, opacity and sizes via the
        # dialogs.
        device_name = self.app.preferences.get('devbrush.last_used', None)
        if device_name is None:
            return
        if self.last_used_devbrush_was_eraser:
            bm = self.app.brushmanager
            selected_brush = bm.clone_selected_brush(name=None)  # preserve user-chosen eraser mode settings
        else:
            selected_brush = self.clone_selected_brush_for_saving()  # discard eraser mode settings
        self.app.brushmanager.store_brush_for_device(device_name, selected_brush)
        # However it may be better to reflect any brush settings change into
        # the last-used devbrush immediately. The UI idea here is that the
        # pointer (when you're holding the pen) is special, it's the point of a
        # real-world tool that you're dipping into a palette, or modifying
        # using the sliders.

    def frame_changed_cb(self):
        self.tdw.queue_draw()
