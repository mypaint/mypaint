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
from warnings import warn

import pygtkcompat
import gobject
import gtk
from gtk import gdk
from gettext import gettext as _

import lib.document
from lib import command, helpers, layer, tiledsurface
import tileddrawwidget, stategroup
from brushmanager import ManagedBrush
import dialogs
import canvasevent
import colorpicker   # purely for registration
import linemode


class CanvasController (object):
    """Minimal canvas controller using a stack of modes.

    Basic CanvasController objects can be set up to handle scroll events like
    zooming or rotation only, pointer events like drawing only, or both.

    The actual interpretation of each event is delegated to the top item on the
    controller's modes stack: see `gui.canvasevent.CanvasInteractionMode` for
    details. Simpler modes may assume the basic CanvasController interface,
    more complex ones 

    """

    # NOTE: if muliple views of a single model are required, this interface
    # will have to be revised.


    def __init__(self, tdw):
        """Initialize.

        :param tdw: The view widget to attach handlers onto.
        :type tdw: gui.tileddrawwidget.TiledDrawWidget

        """
        object.__init__(self)
        self.tdw = tdw     #: the TiledDrawWidget being controlled.
        self.modes = canvasevent.ModeStack(self)  #: stack of delegates


    def init_pointer_events(self):
        """Establish TDW event listeners for pointer button presses & drags.
        """
        self.tdw.connect("button-press-event", self.button_press_cb)
        self.tdw.connect("motion-notify-event", self.motion_notify_cb)
        self.tdw.connect("button-release-event", self.button_release_cb)


    def init_scroll_events(self):
        """Establish TDW event listeners for scroll-wheel actions.
        """
        self.tdw.connect("scroll-event", self.scroll_cb)
        self.tdw.add_events(gdk.SCROLL_MASK)


    def button_press_cb(self, tdw, event):
        """Delegates a ``button-press-event`` to the top mode in the stack.
        """
        result = self.modes.top.button_press_cb(tdw, event)
        self.__update_last_event_info(tdw, event)
        return result


    def button_release_cb(self, tdw, event):
        """Delegates a ``button-release-event`` to the top mode in the stack.
        """
        result = self.modes.top.button_release_cb(tdw, event)
        self.__update_last_event_info(tdw, event)
        return result


    def motion_notify_cb(self, tdw, event):
        """Delegates a ``motion-notify-event`` to the top mode in the stack.
        """
        result = self.modes.top.motion_notify_cb(tdw, event)
        self.__update_last_event_info(tdw, event)
        return result


    def scroll_cb(self, tdw, event):
        """Delegates a ``scroll-event`` to the top mode in the stack.
        """
        result = self.modes.top.scroll_cb(tdw, event)
        self.__update_last_event_info(tdw, event)
        return result


    def __update_last_event_info(self, tdw, event):
        # Update the stored details of the last event delegated.
        tdw.__last_event_x = event.x
        tdw.__last_event_y = event.y
        tdw.__last_event_time = event.time


    def get_last_event_info(self, tdw):
        """Get details of the last event delegated to a mode in the stack.

        :rtype tuple: ``(time, x, y)``

        """
        t, x, y = 0, None, None
        try:
            t = tdw.__last_event_time
            x = tdw.__last_event_x
            y = tdw.__last_event_y
        except AttributeError:
            pass
        return (t, x, y)



class Document (CanvasController):
    """Manipulation of a loaded document via the the GUI.

    A `gui.Document` is something like a Controller in the MVC sense: it
    translates GtkAction activations and keypresses for changing the view into
    View (`gui.tileddrawwidget`) manipulations. It is also responsible for
    directly manipulating the Model (`lib.document`) in response to actions
    and keypresses, for example manipulating the layer stack.

    Some per-application state can be manipulated through this object too: for
    example the drawing brush which is owned by the main application
    singleton.

    """

    # Layers have this attr set temporarily if they don't have a name yet
    _NONAME_LAYER_REFNUM_ATTR = "_document_noname_ref_number"


    def __init__(self, app, leader=None):
        self.app = app
        self.model = lib.document.Document(self.app.brush)
        tdw = tileddrawwidget.TiledDrawWidget(self.app, self.model)
        CanvasController.__init__(self, tdw)
        self.modes.observers.append(self.mode_stack_changed_cb)

        # Pass on certain actions to other gui.documents.
        self.followers = []

        self.model.frame_observers.append(self.frame_changed_cb)
        self.model.symmetry_observers.append(self.update_symmetry_toolitem)

        # Deferred until after the app starts (runs in the first idle-
        # processing phase) as a workaround for https://gna.org/bugs/?14372
        # ([Windows] crash when moving the pen during startup)
        gobject.idle_add(self.init_pointer_events)
        gobject.idle_add(self.init_scroll_events)

        # Input stroke observers
        self.input_stroke_ended_observers = []
        """Callbacks interested in the end of an input stroke.

        Observers are called with the GTK event as their only argument. This
        is a good place to listen for "just painted something" events in some
        cases; app.brush will contain everything needed about the input stroke
        which is ending.

        An input stroke is a single button-down, move, button-up
        action. This sort of stroke is not the same as a brush engine
        stroke (see ``lib.document``). It is possible that the visible
        stroke starts earlier and ends later, depending on how the
        operating system maps pressure to button up/down events.
        """

        self.input_stroke_started_observers = []
        """See `self.input_stroke_ended_observers`"""

        # FIXME: hack, to be removed
        fname = os.path.join(self.app.datapath, 'backgrounds', '03_check1.png')
        pixbuf = gdk.pixbuf_new_from_file(fname)
        self.tdw.neutral_background_pixbuf = tiledsurface.Background(helpers.gdkpixbuf2numpy(pixbuf))

        self.zoomlevel_values = [1.0/16, 1.0/8, 2.0/11, 0.25, 1.0/3, 0.50, 2.0/3,  # micro
                                 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 8.0,        # normal
                                 11.0, 16.0, 23.0, 32.0, 45.0, 64.0]       # macro

        default_zoom = self.app.preferences['view.default_zoom']
        self.tdw.scale = default_zoom
        self.tdw.zoom_min = min(self.zoomlevel_values)
        self.tdw.zoom_max = max(self.zoomlevel_values)

        # Device-specific brushes: save at end of stroke
        self.input_stroke_ended_observers.append(self.input_stroke_ended_cb)

        self.init_stategroups()
        if leader is not None:
            # This is a side controller (e.g. the scratchpad) which plays
            # follow-the- leader for some events.
            assert isinstance(leader, Document)
            leader.followers.append(self)
            self.action_group = leader.action_group # hack, but needed by tdw
        else:
            # This doc owns the Actions which are (sometimes) passed on to
            # followers to perform. It's model is also the main 'document'
            # being worked on by the user.
            self.init_actions()
            self.init_context_actions()
            for action in self.action_group.list_actions():
                self.app.kbm.takeover_action(action)
            for action in self.modes_action_group.list_actions():
                self.app.kbm.takeover_action(action)
            self.init_extra_keys()

            toggle_action = self.app.builder.get_object('ContextRestoreColor')
            toggle_action.set_active(self.app.preferences['misc.context_restores_color'])

        #: Saved transformation to allow FitView to be toggled.
        self.saved_view = None


    def init_actions(self):
        # Actions are defined in mypaint.xml, just grab a ref to the groups
        self.action_group = self.app.builder.get_object('DocumentActions')
        self.modes_action_group = self.app.builder.get_object("ModeStackActions")

        # Set up certain actions to reflect model state changes
        self.model.command_stack_observers.append(
                self.update_command_stack_toolitems)
        self.update_command_stack_toolitems(self.model.command_stack)
        self.model.doc_observers.append(self.model_structure_changed_cb)
        self.model_structure_changed_cb(self.model)


    def init_context_actions(self):
        ag = self.action_group
        context_actions = []
        for x in range(10):
            r = ('Context0%d' % x, None, _('Restore Brush %d') % x,
                 '%d' % x, None, self.context_cb)
            s = ('Context0%ds' % x, None, _('Save to Brush %d') % x,
                 '<control>%d' % x, None, self.context_cb)
            context_actions.append(s)
            context_actions.append(r)
        ag.add_actions(context_actions)


    def init_stategroups(self):
        sg = stategroup.StateGroup()
        self.layerblink_state = sg.create_state(self.layerblink_state_enter,
                                                self.layerblink_state_leave)
        sg = stategroup.StateGroup()
        self.strokeblink_state = sg.create_state(self.strokeblink_state_enter,
                                                 self.strokeblink_state_leave)
        self.strokeblink_state.autoleave_timeout = 0.3

        # separate stategroup...
        sg2 = stategroup.StateGroup()
        self.layersolo_state = sg2.create_state(self.layersolo_state_enter,
                                                self.layersolo_state_leave)
        self.layersolo_state.autoleave_timeout = None


    def init_extra_keys(self):
        # The keyboard shortcuts below are not visible in the menu.
        # Shortcuts assigned through the menu will take precedence.
        # If we assign the same key twice, the last one will work.
        k = self.app.kbm.add_extra_key

        k('bracketleft', 'Smaller') # GIMP, Photoshop, Painter
        k('bracketright', 'Bigger') # GIMP, Photoshop, Painter
        k('<control>bracketleft', 'RotateLeft') # Krita
        k('<control>bracketright', 'RotateRight') # Krita
        k('less', 'LessOpaque') # GIMP
        k('greater', 'MoreOpaque') # GIMP
        k('equal', 'ZoomIn') # (on US keyboard next to minus)
        k('comma', 'Smaller') # Krita
        k('period', 'Bigger') # Krita

        k('BackSpace', 'ClearLayer')

        k('<control>z', 'Undo')
        k('<control>y', 'Redo')
        k('<control><shift>z', 'Redo')
        k('<control>w', lambda(action): self.app.drawWindow.quit_cb())
        k('KP_Add', 'ZoomIn')
        k('KP_Subtract', 'ZoomOut')
        k('KP_4', 'RotateLeft') # Blender
        k('KP_6', 'RotateRight') # Blender
        k('KP_5', 'ResetRotation')
        k('plus', 'ZoomIn')
        k('minus', 'ZoomOut')
        k('<control>plus', 'ZoomIn') # Krita
        k('<control>minus', 'ZoomOut') # Krita
        k('bar', 'Symmetry')

        k('Left', lambda(action): self.pan('PanLeft'))
        k('Right', lambda(action): self.pan('PanRight'))
        k('Down', lambda(action): self.pan('PanDown'))
        k('Up', lambda(action): self.pan('PanUp'))

        k('<control>Left', 'RotateLeft')
        k('<control>Right', 'RotateRight')


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
        # use the full document bbox, so we can paste layers back to the correct position
        bbox = self.model.get_bbox()
        if bbox.w == 0 or bbox.h == 0:
            print "WARNING: empty document, nothing copied"
            return
        else:
            pixbuf = self.model.layer.render_as_pixbuf(*bbox)
        cb = gtk.Clipboard()
        cb.set_image(pixbuf)


    def paste_cb(self, action):
        cb = gtk.Clipboard()
        def callback(clipboard, pixbuf, junk):
            if not pixbuf:
                print 'The clipboard doeas not contain any image to paste!'
                return
            # paste to the upper left of our doc bbox (see above)
            x, y, w, h = self.model.get_bbox()
            self.model.load_layer_from_pixbuf(pixbuf, x, y)
        cb.request_image(callback)


    def pick_context_cb(self, action):
        active_tdw = self.tdw.__class__.get_active_tdw()
        if not self.tdw is active_tdw:
            for follower in self.followers:
                if follower.tdw is active_tdw:
                    print "passing %s action to %s" % (action.get_name(), follower)
                    follower.pick_context_cb(action)
                    return
            return
        x, y = self.tdw.get_cursor_in_model_coordinates()
        for idx, layer in reversed(list(enumerate(self.model.layers))):
            if layer.locked:
                continue
            if not layer.visible:
                continue
            alpha = layer.get_alpha (x, y, 5) * layer.effective_opacity
            if alpha > 0.1:
                old_layer = self.model.layer
                self.model.select_layer(idx)
                if self.model.layer != old_layer:
                    self.layerblink_state.activate()

                # find the most recent (last) stroke that touches our picking point
                si = self.model.layer.get_stroke_info_at(x, y)
                if si:
                    self.restore_brush_from_stroke_info(si)
                    self.si = si # FIXME: should be a method parameter?
                    self.strokeblink_state.activate(action)
                return


    def restore_brush_from_stroke_info(self, strokeinfo):
        mb = ManagedBrush(self.app.brushmanager)
        mb.brushinfo.load_from_string(strokeinfo.brush_string)
        self.app.brushmanager.select_brush(mb)
        self.app.brushmodifier.restore_context_of_selected_brush()


    # LAYER
    def clear_layer_cb(self, action):
        self.model.clear_layer()


    def remove_layer_cb(self, action):
        self.model.remove_layer()

    def convert_layer_to_normal_mode_cb(self, action):
        self.model.convert_layer_to_normal_mode()

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
            alpha = layer.get_alpha (x, y, 5) * layer.effective_opacity
            if alpha > 0.1:
                self.model.select_layer(idx)
                self.layerblink_state.activate(action)
                return
        self.model.select_layer(0)
        self.layerblink_state.activate(action)


    def move_layer_in_stack_cb(self, action):
        """Moves the current layer up or down one slot (action callback)

        The direction the layer moves depends on the action name:
        "RaiseLayerInStack" or "LowerLayerInStack".

        """
        current_layer_pos = self.model.layer_idx
        if action.get_name() == 'RaiseLayerInStack':
            new_layer_pos = current_layer_pos + 1
        elif action.get_name() == 'LowerLayerInStack':
            new_layer_pos = current_layer_pos - 1
        else:
            return
        if new_layer_pos < len(self.model.layers) and new_layer_pos >= 0:
            self.model.move_layer(current_layer_pos, new_layer_pos,
                                  select_new=True)


    def duplicate_layer_cb(self, action):
        """Duplicates the current layer (action callback)"""
        layer = self.model.get_current_layer()
        name = layer.name
        if name:
            name = _("Copy of %s") % name
        else:
            layer_num = self.get_number_for_nameless_layer(layer)
            name = _("Copy of Untitled layer #%d") % layer_num
        self.model.duplicate_layer(self.model.layer_idx, name)


    def rename_layer_cb(self, action):
        """Prompts for a new name for the current layer (action callback)"""
        layer = self.model.get_current_layer()
        new_name = dialogs.ask_for_name(self.app.drawWindow, _("Layer Name"), layer.name)
        if new_name:
            self.model.rename_layer(layer, new_name)


    def layer_lock_toggle_cb(self, action):
        layer = self.model.layer
        if bool(layer.locked) != bool(action.get_active()):
            self.model.set_layer_locked(action.get_active(), layer)


    def layer_visible_toggle_cb(self, action):
        layer = self.model.layer
        if bool(layer.visible) != bool(action.get_active()):
            self.model.set_layer_visibility(action.get_active(), layer)


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
        h, s, v = self.app.brush.get_color_hsv()
        v += 0.08
        if v > 1.0: v = 1.0
        self.app.brush.set_color_hsv((h, s, v))


    def darker_cb(self, action):
        h, s, v = self.app.brush.get_color_hsv()
        v -= 0.08
        # stop a little higher than 0.0, to avoid resetting hue to 0
        if v < 0.005: v = 0.005
        self.app.brush.set_color_hsv((h, s, v))


    def increase_hue_cb(self,action):
        h, s, v = self.app.brush.get_color_hsv()
        e = 0.015
        h = (h + e) % 1.0
        self.app.brush.set_color_hsv((h, s, v))


    def decrease_hue_cb(self,action):
        h, s, v = self.app.brush.get_color_hsv()
        e = 0.015
        h = (h - e) % 1.0
        self.app.brush.set_color_hsv((h, s, v))


    def purer_cb(self,action):
        h, s, v = self.app.brush.get_color_hsv()
        s += 0.08
        if s > 1.0: s = 1.0
        self.app.brush.set_color_hsv((h, s, v))


    def grayer_cb(self,action):
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
            context.brushinfo = self.app.brush.clone()
            context.preview = bm.selected_brush.preview
            context.save()
        else:
            if self.app.preferences['misc.context_restores_color']:
                bm.select_brush(context) # restore brush
                self.app.brushmodifier.restore_context_of_selected_brush() # restore color
            else:
                bm.select_brush(context)

    def context_toggle_color_cb(self, action):
        self.app.preferences['misc.context_restores_color'] = bool(action.get_active())

    def strokeblink_state_enter(self):
        l = layer.Layer()
        self.si.render_overlay(l)
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
        allocation = self.tdw.get_allocation()
        step = min((allocation.width, allocation.height)) / 5
        if   command == 'PanLeft' : self.tdw.scroll(-step, 0)
        elif command == 'PanRight': self.tdw.scroll(+step, 0)
        elif command == 'PanUp'   : self.tdw.scroll(0, -step)
        elif command == 'PanDown' : self.tdw.scroll(0, +step)
        else: assert 0
        self.clear_saved_view()


    def zoom(self, command):
        try:
            zoom_index = self.zoomlevel_values.index(self.tdw.scale)
        except ValueError:
            zoom_levels = self.zoomlevel_values[:]
            zoom_levels.append(self.tdw.scale)
            zoom_levels.sort()
            zoom_index = zoom_levels.index(self.tdw.scale)

        if   command == 'ZoomIn' : zoom_index += 1
        elif command == 'ZoomOut': zoom_index -= 1
        else: assert 0
        if zoom_index < 0: zoom_index = 0
        if zoom_index >= len(self.zoomlevel_values):
            zoom_index = len(self.zoomlevel_values) - 1

        z = self.zoomlevel_values[zoom_index]
        etime, ex, ey = self.get_last_event_info(self.tdw)
        self.tdw.set_zoom(z, (ex, ey))
        self.clear_saved_view()


    def rotate(self, command):
        # Allows easy and quick rotation to 45/90/180 degrees
        # (Around the window center, not pointer center, seems to be the better default.)
        rotation_step = 2*math.pi/16
        if command == 'RotateRight':
            self.tdw.rotate(+rotation_step)
        else:   # command == 'RotateLeft'
            self.tdw.rotate(-rotation_step)
        self.clear_saved_view()


    def zoom_cb(self, action):
        self.zoom(action.get_name())


    def rotate_cb(self, action):
        self.rotate(action.get_name())


    def symmetry_action_toggled_cb(self, action):
        """Change the model's symmetry state in response to UI events.
        """
        alloc = self.tdw.get_allocation()
        if action.get_active():
            xmid_d, ymid_d = alloc.width/2.0, alloc.height/2.0
            xmid_m, ymid_m = self.tdw.display_to_model(xmid_d, ymid_d)
            if self.model.get_symmetry_axis() != xmid_m:
                self.model.set_symmetry_axis(xmid_m)
        else:
            if self.model.get_symmetry_axis() is not None:
                self.model.set_symmetry_axis(None)


    def update_symmetry_toolitem(self):
        """Updates the UI to reflect changes to the model's symmetry state.
        """
        ag = self.action_group
        action = ag.get_action("Symmetry")
        new_xmid = self.model.get_symmetry_axis()
        if new_xmid is None and action.get_active():
            action.set_active(False)
        elif (new_xmid is not None) and (not action.get_active()):
            action.set_active(True)


    def mirror_horizontal_cb(self, action):
        self.tdw.mirror()
        self.clear_saved_view()


    def mirror_vertical_cb(self, action):
        self.tdw.rotate(math.pi)
        self.tdw.mirror()
        self.clear_saved_view()


    def reset_view_cb(self, command):
        """Action callback: resets various aspects of the view.

        The reset chosen depends on the action's name.

        """
        if command is None:
            command_name = None
        else:
            command_name = command.get_name()
        zoom = mirror = rotation = False
        if command_name is None or 'View' in command_name:
            zoom = mirror = rotation = True
        elif 'Rotation' in command_name:
            rotation = True
        elif 'Zoom' in command_name:
            zoom = True
        elif 'Mirror' in command_name:
            mirror = True
        if rotation or zoom or mirror:
            self.reset_view(rotation, zoom, mirror)
            self.clear_saved_view()


    def reset_view(self, rotation=False, zoom=False, mirror=False):
        """Programatically resets the view to the defaults.
        """
        if rotation:
            self.tdw.set_rotation(0.0)
        if zoom:
            default_zoom = self.app.preferences['view.default_zoom']
            self.tdw.set_zoom(default_zoom)
        if mirror:
            self.tdw.set_mirrored(False)
        if rotation and zoom and mirror:
            self.tdw.recenter_document()


    def fit_view_toggled_cb(self, action):
        """Callback: toggles between fit-document and the current view.

        This callback saves to and restores from the saved view. If the action
        is toggled off when there is a saved view, the saved view will be
        restored.

        """

        # View>Fit: fits image within window's borders.
        if action.get_active():
            self.saved_view = self.tdw.get_transformation()
            self.fit_view()
        else:
            if self.saved_view is not None:
                self.tdw.set_transformation(self.saved_view)
            self.saved_view = None


    def fit_view(self):
        """Programatically fits the view to the document.
        """
        bbox = tuple(self.tdw.doc.get_effective_bbox())
        w, h = bbox[2:4]
        if w == 0:
            # When there is nothing on the canvas reset zoom to default.
            self.reset_view(True, True, True)
            return

        allocation = self.tdw.get_allocation()
        w1, h1 = allocation.width, allocation.height
        # Store radians and reset rotation to zero.
        radians = self.tdw.rotation
        self.tdw.set_rotation(0.0)
        # Store mirror and temporarily it turn off mirror.
        mirror = self.tdw.mirrored
        self.tdw.set_mirrored(False)
        # Using w h as the unrotated bbox, calculate the bbox of the
        # rotated doc.
        cos = math.cos(radians)
        sin = math.sin(radians)
        wcos = w * cos
        hsin = h * sin
        wsin = w * sin
        hcos = h * cos
        # We only need to calculate the positions of two corners of the
        # bbox since it is centered and symetrical, but take the max
        # value since during rotation one corner's distance along the
        # x axis shortens while the other lengthens. Same for the y axis.
        x = max(abs(wcos - hsin), abs(wcos + hsin))
        y = max(abs(wsin + hcos), abs(wsin - hcos))
        # Compare the doc and window dimensions and take the best fit
        zoom = min((w1-20)/x, (h1-20)/y)
        # Reapply all transformations
        self.tdw.recenter_document() # Center image
        self.tdw.set_rotation(radians) # reapply canvas rotation
        self.tdw.set_mirrored(mirror) #reapply mirror
        self.tdw.set_zoom(zoom) # Set new zoom level


    def clear_saved_view(self):
        """Discards the saved view, and deactivates any associated view toggle

        This should be called after the user has changed the view
        interactively, i.e. by dragging or by a simple user Action. Associated
        view ToggleActions like FitView are made inactive.

        """
        self.saved_view = None
        fit_view = self.app.find_action("FitView")
        if fit_view.get_active():
            fit_view.set_active(False)



    # DEBUGGING

    def print_inputs_cb(self, action):
        self.model.brush.set_print_inputs(action.get_active())


    def visualize_rendering_cb(self, action):
        self.tdw.renderer.visualize_rendering = action.get_active()


    def no_double_buffering_cb(self, action):
        self.tdw.renderer.set_double_buffered(not action.get_active())


    # LAST-USED BRUSH STATE

    def input_stroke_ended_cb(self, event):
        # Store device-specific brush settings at the end of the stroke, not
        # when the device changes because the user can change brush radii etc.
        # in the middle of a stroke, and because device_changed_cb won't
        # respond when the user fiddles with colours, opacity and sizes via the
        # dialogs.
        device_name = self.app.preferences.get('devbrush.last_used', None)
        if device_name is None:
            return
        bm = self.app.brushmanager
        selected_brush = bm.clone_selected_brush(name=None) # for saving
        bm.store_brush_for_device(device_name, selected_brush)
        # However it may be better to reflect any brush settings change into
        # the last-used devbrush immediately. The UI idea here is that the
        # pointer (when you're holding the pen) is special, it's the point of a
        # real-world tool that you're dipping into a palette, or modifying
        # using the sliders.


    # MODEL STATE REFLECTION

    def update_command_stack_toolitems(self, stack):
        # Undo and Redo are shown and hidden, and have their labels updated
        # in response to user commands.
        ag = self.action_group
        undo_action = ag.get_action("Undo")
        undo_action.set_sensitive(len(stack.undo_stack) > 0)
        if len(stack.undo_stack) > 0:
            cmd = stack.undo_stack[-1]
            desc = _("Undo %s") % cmd.display_name
        else:
            desc = _("Undo")  # Used when initializing the prefs dialog
        undo_action.set_label(desc)
        undo_action.set_tooltip(desc)
        redo_action = ag.get_action("Redo")
        redo_action.set_sensitive(len(stack.redo_stack) > 0)
        if len(stack.redo_stack) > 0:
            cmd = stack.redo_stack[-1]
            desc = _("Redo %s") % cmd.display_name
        else:
            desc = _("Redo")  # Used when initializing the prefs dialog
        redo_action.set_label(desc)
        redo_action.set_tooltip(desc)


    def model_structure_changed_cb(self, doc):
        # Handle model structural changes.
        ag = self.action_group

        # Reflect position of current layer in the list.
        sel_is_top = sel_is_bottom = False
        sel_is_bottom = doc.layer_idx == 0
        sel_is_top = doc.layer_idx == len(doc.layers)-1
        ag.get_action("RaiseLayerInStack").set_sensitive(not sel_is_top)
        ag.get_action("LowerLayerInStack").set_sensitive(not sel_is_bottom)
        ag.get_action("LayerFG").set_sensitive(not sel_is_top)
        ag.get_action("LayerBG").set_sensitive(not sel_is_bottom)
        ag.get_action("MergeLayer").set_sensitive(not sel_is_bottom)
        ag.get_action("PickLayer").set_sensitive(len(doc.layers) > 1)

        # The current layer's status
        layer = doc.layer
        action = ag.get_action("LayerLockedToggle")
        if bool(action.get_active()) != bool(layer.locked):
            action.set_active(bool(layer.locked))
        action = ag.get_action("LayerVisibleToggle")
        if bool(action.get_active()) != bool(layer.visible):
            action.set_active(bool(layer.visible))

        # Active modes.
        self.modes.top.model_structure_changed_cb(doc)


    def frame_changed_cb(self):
        self.tdw.queue_draw()


    def get_number_for_nameless_layer(self, layer):
        """Assigns a unique integer for otherwise nameless layers

        For use by the layers window, mainly: when presenting the layers stack
        we need a unique number to make it distinguishable from other layers.

        """
        assert not layer.name
        num = getattr(layer, self._NONAME_LAYER_REFNUM_ATTR, None)
        if num is None:
            seen_nums = set([0])
            for l in self.model.layers:
                if l.name:
                    continue
                n = getattr(l, self._NONAME_LAYER_REFNUM_ATTR, None)
                if n is not None:
                    seen_nums.add(n)
            # Hmm. Which method is best?
            if True:
                # High water mark
                num = max(seen_nums) + 1
            else:
                # Reuse former IDs
                num = len(self.model.layers)
                for i in xrange(1, num):
                    if i not in seen_nums:
                        num = i
                        break
            setattr(layer, self._NONAME_LAYER_REFNUM_ATTR, num)
        return num


    def mode_flip_action_activated_cb(self, flip_action):
        """Callback: mode "flip" action activated.

        :param flip_action: the gtk.Action which was activated

        Mode classes are looked up via `canvasevent.ModeRegistry` based on the
        name of the action: flip actions are named after the RadioActions they
        nominally control, with "Flip" prepended.  Activating a FlipAction has
        the effect of flipping a mode off if it is currently active, or on if
        another mode is active. Mode flip actions are the usual actions bound
        to keypresses since being able to toggle off with the same key is
        useful.

        Because these modes are intended for keyboard activation, they are
        instructed to ignore the initial keyboard modifier state when entered.
        See also: `canvasevent.SpringLoadedModeMixin`.

        """
        flip_action_name = flip_action.get_name()
        assert flip_action_name.startswith("Flip")
        # Find the corresponding gtk.RadioAction
        action_name = flip_action_name.replace("Flip", "", 1)
        mode_class = canvasevent.ModeRegistry.get_mode_class(action_name)
        if mode_class is None:
            warn('"%s" not registered: check imports' % action_name, Warning)
            return

        # If a mode object of this exact class is active, pop the stack.
        # Otherwise, instantiate and enter.
        if self.modes.top.__class__ is mode_class:
            self.modes.pop()
            flip_action.keyup_callback = lambda *a: None  # suppress repeats
        else:
            mode = mode_class(ignore_modifiers=True)
            if flip_action.keydown:
                flip_action.__pressed = True
                # Change what happens on a key-up after a short while.
                # Distinguishes long presses from short.
                timeout = getattr(mode, "keyup_timeout", 500)
                cb = self._modeflip_change_keyup_callback
                ev = gtk.get_current_event()
                if ev is not None:
                    ev = ev.copy()
                if timeout > 0:
                    # Queue a change of key-up callback after the timeout
                    gobject.timeout_add(timeout, cb, mode, flip_action, ev)
                    def _continue_mode_early_keyup_cb(*a):
                        # Record early keyup, but otherwise keep in mode
                        flip_action.__pressed = False
                    flip_action.keyup_callback = _continue_mode_early_keyup_cb
                else:
                    # Key-up exits immediately
                    def _exit_mode_early_keyup_cb(*a):
                        if mode is self.modes.top:
                            self.modes.pop()
                    flip_action.keyup_callback = _exit_mode_early_keyup_cb
            self.modes.context_push(mode)


    def _modeflip_change_keyup_callback(self, mode, flip_action, ev):
        # Changes the keyup handler to one which will pop the mode stack
        # if the mode instance is still at the top.
        if not flip_action.__pressed:
            return False

        if mode is self.modes.top:
            def _exit_mode_late_keyup_cb(*a):
                if mode is self.modes.top:
                    self.modes.pop()
            flip_action.keyup_callback = _exit_mode_late_keyup_cb

        ## Could make long-presses start the drag+grab somehow, e.g.
        #if hasattr(mode, '_start_drag'):
        #    mode._start_drag(mode.doc.tdw, ev)
        return False


    def mode_radioaction_changed_cb(self, action, current_action):
        """Callback: GtkRadioAction controlling the modes stack activated.

        :param action: the lead gtk.RadioAction
        :param current_action: the newly active gtk.RadioAction

        Mode classes are looked up via `canvasevent.ModeRegistry` based on the
        name of the action. This action instantiates the mode and pushes it
        onto the mode stack unless the active mode is already an instance of
        the mode class.

        """
        # Update the mode stack so that its top element matches the newly
        # chosen action.
        action_name = current_action.get_name()
        mode_class = canvasevent.ModeRegistry.get_mode_class(action_name)
        if mode_class is None:
            warn('"%s" not registered: check imports' % action_name, Warning)
            return

        if self.modes.top.__class__ is not mode_class:
            mode = mode_class()
            self.modes.context_push(mode)


    def mode_stack_changed_cb(self, mode):
        """Callback: mode stack has changed structure.
        """
        # Activate the action corresponding to the current top mode.
        print "DEBUG: mode stack updated:", self.modes
        action_name = getattr(mode, '__action_name__', None)
        if action_name is None:
            return None
        action = self.app.builder.get_object(action_name)
        if action is not None:
            # Not every mode has a corresponding action
            if not action.get_active():
                action.set_active(True)

