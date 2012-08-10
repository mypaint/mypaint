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

import pygtkcompat
import gobject
import gtk
from gtk import gdk
from gettext import gettext as _

import lib.document
from lib import backgroundsurface, command, helpers, layer
import tileddrawwidget, stategroup
from brushmanager import ManagedBrush
import dialogs
import canvasevent


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
        """Array of callbacks interested in the end of an input stroke.

        Observers are called with the GTK event as their only argument. This
        is a good place to listen for "just painted something" events;
        app.brush will contain everything needed about the input stroke which
        just ended, in the state in which it ended.

        An input stroke is a single pen-down, draw, pen-up action. This sort of
        stroke is not the same as a brush engine stroke (see ``lib.document``).
        """

        self.input_stroke_started_observers = []
        """See `self.input_stroke_ended_observers`"""

        # FIXME: hack, to be removed
        fname = os.path.join(self.app.datapath, 'backgrounds', '03_check1.png')
        pixbuf = gdk.pixbuf_new_from_file(fname)
        self.tdw.neutral_background_pixbuf = backgroundsurface.Background(pixbuf)

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
            # This is a side conteoller (e.g. the scratchpad) which plays
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
            # restore (but keep color, see https://gna.org/bugs/index.php?16977)
            color = self.app.brush.get_color_hsv()
            bm.select_brush(context)
            self.app.brush.set_color_hsv(color)


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
        self._set_zoom(z)


    def _set_zoom(self, z, at_pointer=True):
        if at_pointer:
            etime, ex, ey = self.get_last_event_info(self.tdw)
            self.tdw.set_zoom(z, (ex, ey))
        else:
            self.tdw.set_zoom(z)


    def rotate(self, command):
        # Allows easy and quick rotation to 45/90/180 degrees
        rotation_step = 2*math.pi/16
        etime, ex, ey = self.get_last_event_info(self.tdw)
        center = ex, ey
        if command == 'RotateRight':
            self.tdw.rotate(+rotation_step, center)
        else:   # command == 'RotateLeft'
            self.tdw.rotate(-rotation_step, center)


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
            self._set_zoom(default_zoom)
        if reset_all or ('Mirror' in command_name):
            self.tdw.set_mirrored(False)
        if reset_all:
            self.tdw.recenter_document()
        elif 'Fit' in command_name:
            # View>Fit: fits image within window's borders.
            junk, junk, w, h = self.tdw.doc.get_effective_bbox()
            if w == 0:
                # When there is nothing on the canvas reset zoom to default.
                self.reset_view_cb(None)
            else:
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
                self._set_zoom(zoom, at_pointer=False) # Set new zoom level


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
            desc = _("Undo: nothing to undo")
        undo_action.set_label(desc)
        undo_action.set_tooltip(desc)
        redo_action = ag.get_action("Redo")
        redo_action.set_sensitive(len(stack.redo_stack) > 0)
        if len(stack.redo_stack) > 0:
            cmd = stack.redo_stack[-1]
            desc = _("Redo %s") % cmd.display_name
        else:
            desc = _("Redo: nothing to redo")
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


    def mode_radioaction_changed_cb(self, action, current_action):
        """Callback: GtkRadioAction controlling the modes stack activated.
        """
        # Update the mode stack so that its top element matches the newly
        # chosen action.
        action_name = current_action.get_name()
        mode_class = canvasevent.ModeRegistry.get_mode_class(action_name)
        assert mode_class is not None
        if self.modes.top.__class__ is not mode_class:
            mode = mode_class()
            print "DEBUG: activated", mode
            self.modes.reset(replacement=mode)
            # TODO: perhaps mode classes should list modes they can be
            # stacked on top of. That would allow things like picker modes
            # or drags to be invoked in the middle of fancy line modes,
            # for example.


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

