# -*- coding: utf-8 -*-
#
# This file is part of MyPaint.
# Copyright (C) 2007-2010 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

## Imports

import os, math
from warnings import warn
import logging
logger = logging.getLogger(__name__)

import gtk2compat
import gobject
import gtk
from gtk import gdk
from gettext import gettext as _

import lib.layer
from lib.helpers import clamp
from lib.observable import event
import stategroup
from brushmanager import ManagedBrush
import dialogs
import canvasevent
import colorpicker   # purely for registration


## Class definitions


class CanvasController (object):
    """Minimal canvas controller using a stack of modes.

    Basic CanvasController objects can be set up to handle scroll events
    like zooming or rotation only, pointer events like drawing only, or
    both.

    The actual interpretation of each event is delegated to the top item
    on the controller's modes stack: see `gui.canvasevent` for details.
    Simpler modes may assume the basic CanvasController interface, more
    complex ones may require the full Document interface.

    """

    # NOTE: If muliple, editable, views of a single model are required,
    # NOTE: then this interface will have to be revised.


    ## Initialization

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


    ## Low-level GTK event handlers: delgated to the current mode

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
        return False   #XXX don't consume motions to allow workspace autohide


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


    ## High-level event observing interface

    @event
    def input_stroke_ended(self, event):
        """Event: input stroke just ended

        An input stroke is a single button-down, move, button-up action. This
        sort of stroke is not the same as a brush engine stroke (see
        ``lib.document``). It is possible that the visible stroke starts
        earlier and ends later, depending on how the operating system maps
        pressure to button up/down events.

        :param self: Passed on to registered observers
        :param event: The button release event which ended the input stroke

        Observer functions and methods are called with the originating Document
        Controler and the GTK event as arguments. This is a good place to
        listen for "just painted something" events in some cases; ``app.brush``
        will contain everything needed about the input stroke which is ending.
        """
        pass

    @event
    def input_stroke_started(self, event):
        """Event: input stroke just started

        Callbacks interested in the start of an input stroke should be attached
        here. See `input_stroke_ended()`.
        """
        pass


class Document (CanvasController): #TODO: rename to "DocumentController"#
    """Manipulation of a loaded document via the the GUI.

    A `gui.Document` is something like a Controller in the MVC sense: it
    translates GtkAction activations and keypresses for changing the
    view into View (`gui.tileddrawwidget`) manipulations. It is also
    responsible for directly manipulating the Model (`lib.document` and
    some of its internals) in response to actions and keypresses.

    Some per-application state can be manipulated through this object
    too: for example the drawing brush which is owned by the main
    application singleton.
    """

    ## Class constants

    # Layers have this attr set temporarily if they don't have a name yet
    _NONAME_LAYER_REFNUM_ATTR = "_document_noname_ref_number"

    #: Rotation step amount for single-shot commands.
    #: Allows easy and quick rotation to 45/90/180 degrees.
    ROTATION_STEP = 2*math.pi/16

    # Constants for rotating and zooming by increments
    ROTATE_ANTICLOCKWISE = 4  #: Rotation step direction: RotateLeft
    ROTATE_CLOCKWISE = 8   #: Rotation step direction: RotateRight
    ZOOM_INWARDS = 16  #: Zoom step direction: into the canvas
    ZOOM_OUTWARDS = 32  #: Zoom step direction: out of the canvas

    # Step zoom and rotations can happen at specified locations, or these.
    CENTER_ON_VIEWPORT = 1  #: Zoom or rotate at the canvas center
    CENTER_ON_POINTER = 2  #: Zoom/rotate at the last observed pointer pos

    # Constants for panning (movement) by increments
    PAN_STEP = 0.2 #: Stepwise panning amount: proportion of the canvas size
    PAN_LEFT = 1   #: Stepwise panning direction: left
    PAN_RIGHT = 2   #: Stepwise panning direction: right
    PAN_UP = 3   #: Stepwise panning direction: up
    PAN_DOWN = 4   #: Stepwise panning direction: down

    # Picking
    MIN_PICKING_OPACITY = 0.1
    PICKING_RADIUS = 5


    ## Construction

    def __init__(self, app, tdw, model, leader=None):
        """Constructor for a document controller

        :param app: main application instance
        :type app: gui.application.Application
        :param tdw: primary canvas widget for this controller
        :type tdw: gui.tileddrawwidget.TiledDrawWidget
        :param model: model document to be controlled and reflected
        :type model: lib.document.Document
        :param leader: controller to receive certain reported actions from
        :type leader: gui.document.Document

        The leader/follower setup is there to allow the main document
        controller's "Pick" actions to be passed on to the scratchpad.
        """
        self.app = app
        self.model = model
        CanvasController.__init__(self, tdw)

        # Current mode observation
        self.modes.observers.append(self.mode_stack_changed_cb)

        # Pass on certain actions to other gui.documents.
        self.followers = []

        self.model.frame_observers.append(self._frame_changed_cb)
        self.model.symmetry_observers.append(self.update_symmetry_toolitem)

        # Deferred until after the app starts (runs in the first idle-
        # processing phase) as a workaround for https://gna.org/bugs/?14372
        # ([Windows] crash when moving the pen during startup)
        gobject.idle_add(self.init_pointer_events)
        gobject.idle_add(self.init_scroll_events)

        self.zoomlevel_values = [1.0/16, 1.0/8, 2.0/11, 0.25, 1.0/3, 0.50, 2.0/3,  # micro
                                 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 8.0,        # normal
                                 11.0, 16.0, 23.0, 32.0, 45.0, 64.0]       # macro

        default_zoom = self.app.preferences['view.default_zoom']
        self.tdw.scale = default_zoom
        self.tdw.zoom_min = min(self.zoomlevel_values)
        self.tdw.zoom_max = max(self.zoomlevel_values)

        # Device-specific brushes: save at end of stroke
        self.input_stroke_ended += self._input_stroke_ended_cb

        self._init_stategroups()
        if leader is not None:
            # This is a side controller (e.g. the scratchpad) which plays
            # follow-the- leader for some events.
            assert isinstance(leader, Document)
            leader.followers.append(self)
            self.action_group = leader.action_group # hack, but needed by tdw
        else:
            # This doc owns the Actions which are (sometimes) passed on to
            # followers to perform. Its model is also the main 'document'
            # being worked on by the user.
            self._init_actions()
            self._init_context_actions()
            for action in self.action_group.list_actions():
                self.app.kbm.takeover_action(action)
            for action in self.modes_action_group.list_actions():
                self.app.kbm.takeover_action(action)
            self._init_extra_keys()

            toggle_action = self.app.builder.get_object('ContextRestoreColor')
            toggle_action.set_active(self.app.preferences['misc.context_restores_color'])

        #: Saved transformation to allow FitView to be toggled.
        self.saved_view = None

        #: Viewport change/manipulation observers.
        self.view_changed_observers = []
        self.view_changed_observers.append(self._view_changed_cb)
        self._view_changed_notification_srcid = None
        do_notify = lambda *a: self.notify_view_changed()
        self.tdw.connect_after("size-allocate", do_notify)

        # Brush settings observers
        self.app.brush.observers.append(self._brush_settings_changed_cb)


    def _init_actions(self):
        """Internal: initializes action groups & state reflection"""
        # Actions are defined in mypaint.xml, just grab a ref to
        # the groups.
        builder = self.app.builder
        self.action_group = builder.get_object('DocumentActions')
        self.modes_action_group = builder.get_object("ModeStackActions")

        # Fine-grained observation of various model objects
        cmdstack = self.model.command_stack
        layerstack = self.model.layer_stack
        observed_events = {
            self._update_command_stack_actions: [
                cmdstack.stack_updated,
            ],
            self._update_merge_layer_down_action: [
                # Depends on this layer and the layer beneath it
                # being compatible.
                layerstack.layer_properties_changed,
                layerstack.current_path_updated,
                layerstack.layer_inserted,
                layerstack.layer_deleted,
            ],
            self._update_normalize_layer_action: [
                layerstack.current_path_updated,
                layerstack.layer_properties_changed,
            ],
            self._update_layer_bubble_actions: [
                # Depends on where this layer lies in the stack
                layerstack.current_path_updated,
                layerstack.layer_inserted,
                layerstack.layer_deleted,
            ],
            self._update_layer_select_actions: [
                # Depends on where this layer lies in the stack
                layerstack.current_path_updated,
                layerstack.layer_inserted,
                layerstack.layer_deleted,
            ],
            self._update_trim_layer_action: [
                layerstack.current_path_updated,
            ],
            self._update_layer_pick_action: [
                # Can't pick if there are no layers
                layerstack.layer_inserted,
                layerstack.layer_deleted,
            ],
            self._update_show_background_toggle: [
                layerstack.background_visible_changed,
                layerstack.current_layer_solo_changed,
            ],
            self._update_layer_solo_toggle: [
                layerstack.current_layer_solo_changed,
            ],
            self._update_layer_flag_toggles: [
                # Visible and Locked
                layerstack.current_path_updated,
                layerstack.layer_properties_changed,
            ],
            self._update_layer_stack_isolated_toggle: [
                layerstack.current_path_updated,
                layerstack.layer_properties_changed,
            ],
        }
        for observer_method, events in observed_events.items():
            for event in events:
                event += observer_method
            observer_method()

    def _init_context_actions(self):
        """Internal: initializes several context actions"""
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

    def _init_stategroups(self):
        """Internal: initializes internal StateGroups"""
        sg = stategroup.StateGroup()
        self.layerblink_state = sg.create_state(self.layerblink_state_enter,
                                                self.layerblink_state_leave)
        sg = stategroup.StateGroup()
        self.strokeblink_state = sg.create_state(self.strokeblink_state_enter,
                                                 self.strokeblink_state_leave)
        self.strokeblink_state.autoleave_timeout = 0.3

    def _init_extra_keys(self):
        """Internal: initializes secondary keyboard shortcuts

        The keyboard shortcuts here are not visible in the menu.
        Shortcuts assigned through the menu will take precedence.
        If we assign the same key twice, the last one will work.
        """
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

        k('Left', lambda(action): self.pan(self.PAN_LEFT))
        k('Right', lambda(action): self.pan(self.PAN_RIGHT))
        k('Down', lambda(action): self.pan(self.PAN_DOWN))
        k('Up', lambda(action): self.pan(self.PAN_UP))

        k('<control>Left', 'RotateLeft')
        k('<control>Right', 'RotateRight')

    ## Command history traversal actions

    def undo_cb(self, action):
        """Undo action callback"""
        cmd = self.model.undo()

    def redo_cb(self, action):
        """Redo action callback"""
        cmd = self.model.redo()


    def _update_command_stack_actions(self, *_ignored):
        """Update the undo and redo actions"""
        stack = self.model.command_stack
        draw_window = self.app.drawWindow
        ag = self.action_group

        # Icon names
        style_state = draw_window.get_style_context().get_state()
        try: # GTK 3.8+
            if style_state & gtk.StateFlags.DIR_LTR:
                direction = 'ltr'
            else:
                direction = 'rtl'
        except AttributeError:
            # Deprecated in 3.8
            if draw_window.get_direction() == gtk.TextDirection.LTR:
                direction = 'ltr'
            else:
                direction = 'rtl'
        undo_icon_name = "mypaint-undo-%s-symbolic" % (direction,)
        redo_icon_name = "mypaint-redo-%s-symbolic" % (direction,)

        # Undo
        undo_action = ag.get_action("Undo")
        undo_action.set_sensitive(len(stack.undo_stack) > 0)
        undo_action.set_icon_name(undo_icon_name)
        if len(stack.undo_stack) > 0:
            cmd = stack.undo_stack[-1]
            desc = _("Undo %s") % cmd.display_name
        else:
            desc = _("Undo")  # Used when initializing the prefs dialog
        undo_action.set_label(desc)
        undo_action.set_tooltip(desc)

        # Redo
        redo_action = ag.get_action("Redo")
        redo_action.set_sensitive(len(stack.redo_stack) > 0)
        redo_action.set_icon_name(redo_icon_name)
        if len(stack.redo_stack) > 0:
            cmd = stack.redo_stack[-1]
            desc = _("Redo %s") % cmd.display_name
        else:
            desc = _("Redo")  # Used when initializing the prefs dialog
        redo_action.set_label(desc)
        redo_action.set_tooltip(desc)


    ## Copy/Paste

    def _get_clipboard(self):
        """Internal: return the GtkClipboard for the current display"""
        display = self.tdw.get_display()
        cb = gtk.Clipboard.get_for_display(display, gdk.SELECTION_CLIPBOARD)
        return cb

    def copy_cb(self, action):
        """``CopyLayer`` GtkAction callback: copy layer to clipboard"""
        # use the full document bbox, so we can paste layers back to the
        # correct position
        bbox = self.model.get_bbox()
        if bbox.w == 0 or bbox.h == 0:
            logger.error("Empty document, nothing copied")
            return
        rootstack = self.model.layer_stack
        pixbuf = rootstack.current.render_as_pixbuf(*bbox, alpha=True)
        cb = self._get_clipboard()
        cb.set_image(pixbuf)

    def paste_cb(self, action):
        """``PasteLayer`` GtkAction callback: replace layer with clipboard"""
        cb = self._get_clipboard()
        def callback(clipboard, pixbuf, junk):
            if not pixbuf:
                logger.error("The clipboard does not contain "
                             "any image to paste!")
                return
            # paste to the upper left of our doc bbox (see above)
            x, y, w, h = self.model.get_bbox()
            self.model.load_layer_from_pixbuf(pixbuf, x, y)
        cb.request_image(callback, None)

    ## Frame manipulation actions

    def trim_layer_cb(self, action):
        """Trim Layer action: discard tiles outside the frame"""
        self.model.trim_layer()

    def _update_trim_layer_action(self, *_ignored):
        """Updates the Trim Layer action's sensitivity"""
        app = self.app
        root = self.model.layer_stack
        can_trim = root.current.get_trimmable()
        app.find_action("TrimLayer").set_sensitive(can_trim)

    def toggle_frame_cb(self, action):
        """Frame Enabled toggle callback"""
        model = self.model
        enabled = bool(model.frame_enabled)
        desired = bool(action.get_active())
        if enabled != desired:
            model.set_frame_enabled(desired, user_initiated=True)

    def _frame_changed_cb(self):
        """Invoked when the frame changes"""
        action = self.app.find_action("FrameToggle")
        state = bool(self.model.frame_enabled)
        if bool(action.get_active()) != state:
            action.set_active(state)
        # XXX: redraws should be a concern of the tdw only:
        self.tdw.queue_draw()


    ## Layer and stroke picking

    def pick_context_cb(self, action):
        """Pick Context action: select layer and brush from stroke"""
        active_tdw = self.tdw.__class__.get_active_tdw()
        if not self.tdw is active_tdw:
            for follower in self.followers:
                if follower.tdw is active_tdw:
                    logger.debug("passing %s action to %s",
                                 action.get_name(), follower)
                    follower.pick_context_cb(action)
                    return
            return
        x, y = self.tdw.get_cursor_in_model_coordinates()
        layers = self.model.layer_stack
        old_path = layers.current_path
        for c_path, c_layer in self._layer_picking_iter():
            if not self._layer_is_pickable(c_path, (x, y)):
                continue
            self.model.select_layer(path=c_path)
            if c_path is not old_path:
                self.layerblink_state.activate()
            # Find the most recent (last) stroke at the pick point
            si = layers.current.get_stroke_info_at(x, y)
            if si:
                self.restore_brush_from_stroke_info(si)
                self.si = si # FIXME: should be a method parameter?
                self.strokeblink_state.activate(action)
            return

    def restore_brush_from_stroke_info(self, strokeinfo):
        """Restores the app brush from a stroke

        :param strokeinfo: Stroke details from the stroke map
        :type strokeinfo: lib.strokemap.StrokeShape
        """
        mb = ManagedBrush(self.app.brushmanager)
        mb.brushinfo.load_from_string(strokeinfo.brush_string)
        self.app.brushmanager.select_brush(mb)
        self.app.brushmodifier.restore_context_of_selected_brush()

    def pick_layer_cb(self, action):
        """Pick Layer action: select the layer under the pointer"""
        x, y = self.tdw.get_cursor_in_model_coordinates()
        for p_path, p_layer in self._layer_picking_iter():
            if not self._layer_is_pickable(p_path, (x, y)):
                continue
            self.model.select_layer(path=p_path)
            self.layerblink_state.activate(action)
            return
        self.model.select_layer(path=(0,))
        self.layerblink_state.activate(action)

    def _update_layer_pick_action(self, *_ignored):
        """Updates the Layer Picking action's sensitivity"""
        # PickContext is always sensitive, however
        app = self.app
        root = self.model.layer_stack
        pickable = len(root) > 1
        app.find_action("PickLayer").set_sensitive(pickable)

    def _layer_is_pickable(self, path, pos=None):
        """True if a (leaf) layer can be picked

        :param path: Layer path to the layer to be tested.
        :param pos: Optional (x,y) position to test for opacity.
        """
        stack = self.model.layer_stack
        while len(path) > 0:
            layer = stack.deepget(path, None)
            if layer is None:
                return False
            if layer.locked or not layer.visible:
                return False
            # Opacity cutoff. Opacity of the stroke is relevant if this
            # is a leaf layer.
            opacity = layer.effective_opacity
            if pos is not None:
                x, y = pos
                opacity *= layer.get_alpha(x, y, self.PICKING_RADIUS)
                pos = None
            # However the parent chain's opacity must be sufficiently
            # high all the way through for picking to work.
            if opacity < self.MIN_PICKING_OPACITY:
                return False
            path = path[:-1]
        return True

    def _layer_picking_iter(self):
        """Enumerates leaf layers in picking order, with paths"""
        layer_stack = self.model.layer_stack
        layers_enum = layer_stack.deepenumerate()
        parents = set()
        for path, layer in layers_enum:
            if path in parents:
                continue
            parent_path = path[:-1]
            parents.add(parent_path)
            yield (path, layer)


    ## Layer action callbacks

    def clear_layer_cb(self, action):
        """``ClearLayer`` GtkAction callback"""
        self.model.clear_layer()

    def remove_layer_cb(self, action):
        """``RemoveLayer`` GtkAction callback"""
        self.model.remove_layer()

    def normalize_layer_mode_cb(self, action):
        """``NormalizeLayerMode`` GtkAction callback"""
        self.model.normalize_layer_mode()

    ## Layer selection (current layer path in the tree)

    def select_layer_below_cb(self, action):
        """``SelectLayerBelow`` GtkAction callback"""
        layers = self.model.layer_stack
        path = layers.get_current_path()
        path = layers.path_below(path)
        if path:
            self.model.select_layer(path=path)
        self.layerblink_state.activate(action)

    def select_layer_above_cb(self, action):
        """``SelectLayerAbove`` GtkAction callback"""
        layers = self.model.layer_stack
        path = layers.get_current_path()
        path = layers.path_above(path)
        if path:
            self.model.select_layer(path=path)
        self.layerblink_state.activate(action)

    def _update_layer_select_actions(self, *_ignored):
        """Updates the Select Layer Above/Below actions"""
        app = self.app
        root = self.model.layer_stack
        current_path = root.current_path
        if current_path:
            has_predecessor = bool(root.path_above(current_path))
            has_successor = bool(root.path_below(current_path))
        else:
            has_predecessor = False
            has_successor = False
        app.find_action("SelectLayerAbove").set_sensitive(has_predecessor)
        app.find_action("SelectLayerBelow").set_sensitive(has_successor)

    ## Current layer's opacity

    def layer_increase_opacity(self, action):
        """``IncreaseLayerOpacity`` GtkAction callback"""
        opa = clamp(self.model.layer.opacity + 0.08, 0.0, 1.0)
        self.model.set_layer_opacity(opa)

    def layer_decrease_opacity(self, action):
        """``DecreaseLayerOpacity`` GtkAction callback"""
        opa = clamp(self.model.layer.opacity - 0.08, 0.0, 1.0)
        self.model.set_layer_opacity(opa)


    ## Global layer stack toggles

    def current_layer_solo_toggled_cb(self, action):
        """``SoloLayer`` GtkToggleAction callback

        Toggles between showing just the current layer (regardless of its
        visibility flag) and all visible layers.
        """
        self.model.layer_stack.current_layer_solo = action.get_active()

    def _update_layer_solo_toggle(self, *_ignored):
        """Updates the Layer Solo toggle action from the model"""
        root = self.model.layer_stack
        action = self.app.find_action("SoloLayer")
        state = root.current_layer_solo
        if bool(action.get_active()) != state:
            action.set_active(state)

    def show_background_toggle_cb(self, action):
        """``ShowBackgroundToggle`` GtkToggleAction callback"""
        layers = self.model.layer_stack
        if bool(layers.background_visible) != bool(action.get_active()):
            layers.background_visible = action.get_active()

    def _update_show_background_toggle(self, *_ignored):
        """Updates the Show Background toggle action from the model"""
        root = self.model.layer_stack
        action = self.app.find_action("ShowBackgroundToggle")
        state = root.background_visible
        if bool(action.get_active()) != state:
            action.set_active(state)
        action.set_sensitive(not root.current_layer_solo)

    def layer_stack_isolated_toggled_cb(self, action):
        """Group Isolation toggle callback: updates the current layer"""
        model = self.model
        rootstack = model.layer_stack
        layer = rootstack.current
        if not isinstance(layer, lib.layer.LayerStack):
            return
        if layer is rootstack:
            return
        if bool(layer.isolated) != bool(action.get_active()):
            model.set_layer_stack_isolated(action.get_active(), layer)

    def _update_layer_stack_isolated_toggle(self, *_ignored):
        """Updates the Group Isolation toggle from the current layer"""
        action = self.app.find_action("LayerStackIsolated")
        rootstack = self.model.layer_stack
        layer = rootstack.current
        if isinstance(layer, lib.layer.LayerStack):
            isolated_flag = bool(layer.isolated)
            auto_isolation = bool(layer.get_auto_isolation())
            isolated = isolated_flag or auto_isolation
            if bool(action.get_active()) != isolated:
                action.set_active(isolated)
            action.set_sensitive(not auto_isolation)
        else:
            action.set_active(False)
            action.set_sensitive(False)

    ## Layer stack order (bubbling)

    def reorder_layer_cb(self, action):
        """Changes the z-order of a layer (GtkAction callback)

        The direction the layer moves depends on the action name:
        "RaiseLayerInStack" or "LowerLayerInStack".
        """
        layers = self.model.layer_stack
        if action.get_name() == 'RaiseLayerInStack':
            self.model.bubble_current_layer_up()
        elif action.get_name() == 'LowerLayerInStack':
            self.model.bubble_current_layer_down()

    def _update_layer_bubble_actions(self, *_ignored):
        """Update bubble up/down actions from the model"""
        app = self.app
        root = self.model.layer_stack
        current_path = root.current_path
        if current_path:
            deep = len(current_path) > 1
            down_poss = deep or current_path[0] < len(root)-1
            up_poss = deep or current_path[0] > 0
        else:
            down_poss = False
            up_poss = False
        app.find_action("RaiseLayerInStack").set_sensitive(up_poss)
        app.find_action("LowerLayerInStack").set_sensitive(down_poss)

    ## Simple (non-toggle) layer commands

    def new_layer_cb(self, action):
        """New layer GtkAction callback

        Invoked for ``NewLayerFG`` and ``NewLayerBG``: where the new
        layer is created depends on the action's name.
        """
        layers = self.model.layer_stack
        path = layers.current_path
        if not path:
            return
        if action.get_name() == 'NewLayerFG':
            path = layers.path_above(path, insert=True)
        else:
            path = layers.path_below(path, insert=True)
        assert path is not None
        self.model.add_layer(path)
        self.layerblink_state.activate(action)

    def merge_layer_down_cb(self, action):
        """Merge Down: merge current layer with the one below it"""
        if self.model.merge_current_layer_down():
            self.layerblink_state.activate(action)

    def merge_visible_layers_cb(self, action):
        """Merge Visible: merge all visible layers into new one"""
        self.model.merge_visible_layers()
        self.layerblink_state.activate(action)

    def _update_merge_layer_down_action(self, *_ignored):
        """Updates the layer Merge Down action's sensitivity"""
        # This may change in response to the path changing *or* the
        # mode property of the current or underlying layer changing.
        app = self.app
        rootstack = self.model.layer_stack
        current = rootstack.current_path
        can_merge = bool(current and rootstack.get_merge_down_target(current))
        app.find_action("MergeLayerDown").set_sensitive(can_merge)

    def duplicate_layer_cb(self, action):
        """``DuplicateLayer`` GtkAction callback: clone the current layer"""
        self.model.duplicate_current_layer()

    def rename_layer_cb(self, action):
        """``RenameLayer`` GtkAction callback: layer rename dialog"""
        layer = self.model.layer_stack.get_current()
        if layer is self.model.layer_stack:
            return
        old_name = layer.name
        if old_name is None:
            old_name = layer.DEFAULT_NAME
        win = self.app.drawWindow
        new_name = dialogs.ask_for_name(win, _("Layer Name"), old_name)
        if new_name:
            self.model.rename_layer(layer, new_name)

    ## Per-layer flag toggles

    def layer_lock_toggle_cb(self, action):
        """``LayerLockedToggle`` GtkAction callback"""
        layer = self.model.layer_stack.get_current()
        if bool(layer.locked) != bool(action.get_active()):
            self.model.set_layer_locked(action.get_active(), layer)

    def layer_visible_toggle_cb(self, action):
        """``LayerVisibleToggle`` GtkAction callback"""
        layer = self.model.layer_stack.get_current()
        if bool(layer.visible) != bool(action.get_active()):
            self.model.set_layer_visibility(action.get_active(), layer)

    def _update_layer_flag_toggles(self, *_ignored):
        """Updates ToggleActions reflecting the current layer's flags"""
        rootstack = self.model.layer_stack
        current_layer = rootstack.current
        action_updates = [
                ("LayerLockedToggle", current_layer.locked),
                ("LayerVisibleToggle", current_layer.visible),
            ]
        for action_name, model_state in action_updates:
            action = self.app.find_action(action_name)
            if bool(action.get_active()) != bool(model_state):
                action.set_active(model_state)

    ## Brush settings callbacks

    def brush_bigger_cb(self, action):
        """``Bigger`` GtkAction callback"""
        self.model.flush_updates()
        adj = self.app.brush_adjustment['radius_logarithmic']
        adj.set_value(adj.get_value() + 0.3)

    def brush_smaller_cb(self, action):
        """``Smaller`` GtkAction callback"""
        self.model.flush_updates()
        adj = self.app.brush_adjustment['radius_logarithmic']
        adj.set_value(adj.get_value() - 0.3)

    def more_opaque_cb(self, action):
        """``MoreOpaque`` GtkAction callback"""
        # FIXME: hm, looks this slider should be logarithmic?
        self.model.flush_updates()
        adj = self.app.brush_adjustment['opaque']
        adj.set_value(adj.get_value() * 1.8)

    def less_opaque_cb(self, action):
        """``MoreOpaque`` GtkAction callback"""
        self.model.flush_updates()
        adj = self.app.brush_adjustment['opaque']
        adj.set_value(adj.get_value() / 1.8)

    def brighter_cb(self, action):
        """``Brighter`` GtkAction callback: lighten the brush colour"""
        # TODO: use HCY?
        self.model.flush_updates()
        h, s, v = self.app.brush.get_color_hsv()
        v += 0.08
        if v > 1.0: v = 1.0
        self.app.brush.set_color_hsv((h, s, v))

    def darker_cb(self, action):
        """``Darker`` GtkAction callback: darken the brush colour"""
        # TODO: use HCY?
        self.model.flush_updates()
        h, s, v = self.app.brush.get_color_hsv()
        v -= 0.08
        # stop a little higher than 0.0, to avoid resetting hue to 0
        if v < 0.005: v = 0.005
        self.app.brush.set_color_hsv((h, s, v))

    def increase_hue_cb(self,action):
        """``IncreaseHue`` GtkAction callback: anticlockwise hue rotation"""
        self.model.flush_updates()
        # TODO: use HCY?
        h, s, v = self.app.brush.get_color_hsv()
        e = 0.015
        h = (h + e) % 1.0
        self.app.brush.set_color_hsv((h, s, v))

    def decrease_hue_cb(self,action):
        """``DecreaseHue`` GtkAction callback: clockwise hue rotation"""
        self.model.flush_updates()
        # TODO: use HCY?
        h, s, v = self.app.brush.get_color_hsv()
        e = 0.015
        h = (h - e) % 1.0
        self.app.brush.set_color_hsv((h, s, v))

    def purer_cb(self,action):
        """``Purer`` GtkAction callback: make the brush colour less grey"""
        self.model.flush_updates()
        # TODO: use HCY?
        h, s, v = self.app.brush.get_color_hsv()
        s += 0.08
        if s > 1.0: s = 1.0
        self.app.brush.set_color_hsv((h, s, v))

    def grayer_cb(self,action):
        """``Grayer`` GtkAction callback: make the brush colour more grey"""
        # TODO: use HCY?
        h, s, v = self.app.brush.get_color_hsv()
        s -= 0.08
        # stop a little higher than 0.0, to avoid resetting hue to 0
        if s < 0.005: s = 0.005
        self.app.brush.set_color_hsv((h, s, v))


    ## Brush settings

    def brush_reload_settings(self, cnames=None):
        """Reset some or all brush settings to their saved values

        :param cname: Setting names to reset; default is all settings
        :type cname: Iterable of setting cnames.
        """
        app = self.app
        bm = app.brushmanager
        parent_brush = bm.get_parent_brush(brushinfo=app.brush)
        if parent_brush is None:
            return
        if cnames is None:
            bm.select_brush(parent_brush)
        else:
            parent_binfo = parent_brush.get_brushinfo()
            for cname in cnames:
                parent_value = parent_binfo.get_base_value(cname)
                adj = app.brush_adjustment[cname]
                adj.set_value(parent_value)
        app.brushmodifier.normal_mode.activate()

    def brush_reload_cb(self, action):
        """``BrushReload`` GtkAction callback. Reload all brush settings."""
        self.brush_reload_settings()

    def brush_is_modified(self):
        """True if the brush was modified from its saved state"""
        current_bi = self.app.brush
        parent_b = self.app.brushmanager.get_parent_brush(brushinfo=current_bi)
        if parent_b is None:
            return True
        return not parent_b.brushinfo.matches(current_bi)

    def _brush_settings_changed_cb(self, *a):
        """Internal callback: updates the UI when brush settings change"""
        reset_action = self.app.find_action("BrushReload")
        if self.brush_is_modified():
            if not reset_action.get_sensitive():
                reset_action.set_sensitive(True)
        else:
            if reset_action.get_sensitive():
                reset_action.set_sensitive(False)


    ## Brushkey callbacks

    def context_cb(self, action):
        """GtkAction callback for various brushkey operations"""
        name = action.get_name()
        store = False
        bm = self.app.brushmanager
        if name == 'ContextStore':
            context = bm.selected_context
            if not context:
                logger.error('No context was selected, '
                             'ignoring store command.')
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
            # restore brush
            bm.select_brush(context)
            if self.app.preferences['misc.context_restores_color']:
                # restore color
                self.app.brushmodifier.restore_context_of_selected_brush()

    def context_toggle_color_cb(self, action):
        """GtkToggleAction callback for whether brushkeys restore colour"""
        value = bool(action.get_active())
        self.app.preferences['misc.context_restores_color'] = value


    ## UI feedback for current layer/stroke

    def strokeblink_state_enter(self):
        """`gui.stategroup.State` entry callback for blinking a stroke"""
        overlay = lib.layer.SurfaceBackedLayer()
        overlay.load_from_strokeshape(self.si)
        self.tdw.overlay_layer = overlay
        bbox = tuple(overlay.get_bbox())
        self.model.canvas_area_modified(*bbox)

    def strokeblink_state_leave(self, reason):
        """`gui.stategroup.State` leave callback for blinking a stroke"""
        if self.tdw.overlay_layer is None:
            return
        bbox = self.tdw.overlay_layer.get_bbox()
        self.tdw.overlay_layer = None
        self.model.canvas_area_modified(*bbox)

    def layerblink_state_enter(self):
        """`gui.stategroup.State` entry callback for blinking a layer"""
        layers = self.model.layer_stack
        layers.current_layer_previewing = True

    def layerblink_state_leave(self, reason):
        """`gui.stategroup.State` leave callback for blinking a layer"""
        layers = self.model.layer_stack
        layers.current_layer_previewing = False


    ## Viewport manipulation

    def pan(self, direction):
        """Handles panning (scrolling) in increments.

        :param direction: direction of panning
        :type direction: `PAN_LEFT`, `PAN_RIGHT`, `PAN_UP`, or `PAN_DOWN`
        """
        allocation = self.tdw.get_allocation()
        step = min((allocation.width, allocation.height)) * self.PAN_STEP
        if direction == self.PAN_LEFT: self.tdw.scroll(-step, 0)
        elif direction == self.PAN_RIGHT: self.tdw.scroll(+step, 0)
        elif direction == self.PAN_UP: self.tdw.scroll(0, -step)
        elif direction == self.PAN_DOWN: self.tdw.scroll(0, +step)
        else: raise TypeError, 'unsupported pan() direction=%s' % (direction,)
        self.notify_view_changed()

    def zoom(self, direction, center=CENTER_ON_POINTER):
        """Handles zoom in increments.

        Zooms the doc's TDW by a set amount, either in or out.

        :param direction: direction of zoom
        :type direction: `ZOOM_INWARDS` or `ZOOM_OUTWARDS`
        :param center: zoom center
        :type center: tuple ``(x, y)`` in model coords, or `CENTER_ON_POINTER`
            or `CENTER_ON_VIEWPORT`
        """
        self.model.flush_updates()

        if center == self.CENTER_ON_POINTER:
            etime, ex, ey = self.get_last_event_info(self.tdw)
            center = (ex, ey)
        elif center == self.CENTER_ON_VIEWPORT:
            center = self.tdw.get_center()

        try:
            zoom_index = self.zoomlevel_values.index(self.tdw.scale)
        except ValueError:
            zoom_levels = self.zoomlevel_values[:]
            zoom_levels.append(self.tdw.scale)
            zoom_levels.sort()
            zoom_index = zoom_levels.index(self.tdw.scale)

        if direction == self.ZOOM_INWARDS:
            zoom_index += 1
        elif direction == self.ZOOM_OUTWARDS:
            zoom_index -= 1
        else:
            raise TypeError, 'unsupported zoom() direction=%s' % (direction,)

        if zoom_index < 0:
            zoom_index = 0
        if zoom_index >= len(self.zoomlevel_values):
            zoom_index = len(self.zoomlevel_values) - 1

        z = self.zoomlevel_values[zoom_index]
        self.tdw.set_zoom(z, center=center)
        self.notify_view_changed()

    def rotate(self, direction, center=CENTER_ON_POINTER):
        """Handles rotation in increments.

        Rotates the doc's TDW by a set amount, either left or right.

        :param direction: direction of rotation
        :type direction: `ROTATE_CLOCKWISE` or `ROTATE_ANTICLOCKWISE`
        :param center: rotation center
        :type center: tuple ``(x, y)`` in model coords, or `CENTER_ON_POINTER`
            or `CENTER_ON_VIEWPORT`
        """
        self.model.flush_updates()
        if center == self.CENTER_ON_POINTER:
            etime, ex, ey = self.get_last_event_info(self.tdw)
            center = (ex, ey)
        elif center == self.CENTER_ON_VIEWPORT:
            center = self.tdw.get_center()

        if direction == self.ROTATE_CLOCKWISE:
            self.tdw.rotate(+self.ROTATION_STEP, center=center)
        elif direction == self.ROTATE_ANTICLOCKWISE:
            self.tdw.rotate(-self.ROTATION_STEP, center=center)
        else:
            raise TypeError, 'unsupported direction=%s' % (direction,)

        self.notify_view_changed()

    def zoom_cb(self, action):
        """Callback for Zoom{In,Out} GtkActions"""
        direction = self.ZOOM_INWARDS
        if action.get_name() == 'ZoomOut':
            direction = self.ZOOM_OUTWARDS
        self.zoom(direction)

    def rotate_cb(self, action):
        """Callback for Rotate{Left,Right} GtkActions"""
        direction = self.ROTATE_CLOCKWISE
        if action.get_name() == 'RotateRight':
            direction = self.ROTATE_ANTICLOCKWISE
        self.rotate(direction)


    ## Symmetry

    def symmetry_action_toggled_cb(self, action):
        """Change the model's symmetry state in response to UI events"""
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
        """Updates the UI to reflect changes to the model's symmetry state"""
        ag = self.action_group
        action = ag.get_action("Symmetry")
        new_xmid = self.model.get_symmetry_axis()
        if new_xmid is None and action.get_active():
            action.set_active(False)
        elif (new_xmid is not None) and (not action.get_active()):
            action.set_active(True)


    ## More viewport manipulation

    def mirror_horizontal_cb(self, action):
        """Flips the viewport from left to right"""
        self.tdw.mirror()
        self.notify_view_changed()

    def mirror_vertical_cb(self, action):
        """Flips the viewport from top to bottom"""
        self.tdw.rotate(math.pi)
        self.tdw.mirror()
        self.notify_view_changed()

    def reset_view_cb(self, action):
        """Action callback: resets various aspects of the view.

        The reset chosen depends on the action's name.
        """
        if action is None:
            action_name = None
        else:
            action_name = action.get_name()
        zoom = mirror = rotation = False
        if action_name is None or 'View' in action_name:
            zoom = mirror = rotation = True
        elif 'Rotation' in action_name:
            rotation = True
        elif 'Zoom' in action_name:
            zoom = True
        elif 'Mirror' in action_name:
            mirror = True
        if rotation or zoom or mirror:
            self.reset_view(rotation, zoom, mirror)

    def reset_view(self, rotation=False, zoom=False, mirror=False):
        """Programatically resets the view to the defaults.

        :param rotation: Reset rotation to zero.
        :param zoom: Reset rotation to the prefs default zoom.
        :param mirror: Turn mirroring off
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
        if rotation or zoom or mirror:
            self.notify_view_changed()

    def fit_view_toggled_cb(self, action):
        """Callback: toggles between fit-document and the current view.

        This callback saves to and restores from the saved view. If the action
        is toggled off when there is a saved view, the saved view will be
        restored.
        """
        # Note: saved_view must be set to None before notify_view_changed() is
        # called by anything - we use it as an interlock.
        if action.get_active():
            view = self.tdw.get_transformation()
            self.saved_view = None
            self.fit_view()
            self.saved_view = view
        elif self.saved_view is not None:
            view = self.saved_view
            self.tdw.set_transformation(self.saved_view)
            self.saved_view = None
            self.notify_view_changed(immediate=True)

    def fit_view(self):
        """Programatically fits the view to the document"""
        bbox = tuple(self.tdw.doc.get_effective_bbox())
        w, h = bbox[2:4]
        if w == 0:
            # When there is nothing on the canvas reset zoom to default.
            self.reset_view(True, True, True)
            return
        # Otherwise, zoom and transform to fit the bounding box, while
        # preserving the user's rotation and mirroring settings.
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
        # Notify interested parties
        self.notify_view_changed(immediate=True)

    def notify_view_changed(self, prioritize=False, immediate=False):
        """Notifies all parties interested in the view having changed.

        These can be slightly expensive, so the callbacks are rate limited
        using an idle routine when called with default args. All callbacks in
        `self.view_changed_observers` are guaranteed to be called shortly after
        this method is called, with a ref to this Document.

        The default idle priority is intentionally very low. To raise it, set
        `prioritize` to true. This is designed to be used only when this
        notification indirectly updates a graphical element which is directly
        under the pointer, or otherwise where the user is looking.
        """
        if immediate:
            if self._view_changed_notification_srcid:
                gobject.source_remove(self._view_changed_notification_srcid)
                self._view_changed_notification_srcid = None
            self._view_changed_notification_idle_cb()
            return
        if self._view_changed_notification_srcid:
            return
        cb = self._view_changed_notification_idle_cb
        priority = gobject.PRIORITY_LOW
        if prioritize:
            priority = gobject.PRIORITY_HIGH_IDLE
        srcid = gobject.idle_add(cb, priority=priority)
        self._view_changed_notification_srcid = srcid

    def _view_changed_notification_idle_cb(self):
        """Background notifier callback used by `notify_view_changed()`"""
        for cb in self.view_changed_observers:
            cb(self)
        self._view_changed_notification_srcid = None
        return False

    def _view_changed_cb(self, doc):
        """Callback: clear saved view and reset toggles on viewport changes"""
        if not self.saved_view:
            return
        # Clear saved view first...
        self.saved_view = None
        # ... it's used as an interlock by toggle callbacks which use it.
        view_toggle_actions = ["FitView"]
        for action_name in view_toggle_actions:
            action = self.app.find_action(action_name)
            if action.get_active():
                action.set_active(False)


    ## Debugging

    def print_inputs_cb(self, action):
        """Toggles brush input printing"""
        self.model.brush.set_print_inputs(action.get_active())

    def visualize_rendering_cb(self, action):
        """Toggles highlighting of each redraw"""
        self.tdw.renderer.visualize_rendering = action.get_active()

    def no_double_buffering_cb(self, action):
        """Toggles double buffering"""
        self.tdw.renderer.set_double_buffered(not action.get_active())


    ## Model state reflection

    def _input_stroke_ended_cb(self, self_again, event):
        """Invoked after a pen-down, draw, pen-up 'input stroke'"""
        # Store device-specific brush settings at the end of the stroke,
        # not when the device changes because the user can change brush
        # radii etc. in the middle of a stroke, and because
        # device_changed_cb won't respond when the user fiddles with
        # colours, opacity and sizes via the dialogs.
        device_name = self.app.preferences.get('devbrush.last_used', None)
        if device_name is None:
            return
        bm = self.app.brushmanager
        selected_brush = bm.clone_selected_brush(name=None) # for saving
        bm.store_brush_for_device(device_name, selected_brush)
        # However it may be better to reflect any brush settings change
        # into the last-used devbrush immediately. The UI idea here is
        # that the pointer (when you're holding the pen) is special,
        # it's the point of a real-world tool that you're dipping into a
        # palette, or modifying using the sliders.

    def _update_normalize_layer_action(self, *_ignored):
        """Updates the Normalize Layer Mode action's sensitivity"""
        app = self.app
        current = self.model.layer_stack.current
        can_norm = current.get_mode_normalizable()
        app.find_action("NormalizeLayerMode").set_sensitive(can_norm)


    ## Mode flipping

    def mode_flip_action_activated_cb(self, flip_action):
        """Callback: a mode "flip" action was activated.

        :param flip_action: the gtk.Action which was activated

        Mode classes are looked up via `canvasevent.ModeRegistry` based
        on the name of the action: flip actions are named after the
        RadioActions they nominally control, with "Flip" prepended.
        Activating a FlipAction has the effect of flipping a mode off if
        it is currently active, or on if another mode is active. Mode
        flip actions are the usual actions bound to keypresses since
        being able to toggle off with the same key is useful.

        Because these modes are intended for keyboard activation, they
        are instructed to ignore the initial keyboard modifier state
        when entered.  See also: `canvasevent.SpringLoadedModeMixin`.

        """
        flip_action_name = flip_action.get_name()
        assert flip_action_name.startswith("Flip")
        # Find the corresponding gtk.RadioAction
        action_name = flip_action_name.replace("Flip", "", 1)
        mode_class = canvasevent.ModeRegistry.get_mode_class(action_name)
        if mode_class is None:
            warn("%r not registered: check imports" % (action_name,), Warning)
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
        """Changes what happens when a flip-action key is released"""
        # Changes the keyup handler to one which will pop the mode stack
        # if the mode instance is still at the top.
        if not flip_action.__pressed:
            return False

        if mode is self.modes.top:
            def _exit_mode_late_keyup_cb(*a):
                if mode is self.modes.top:
                    self.modes.pop()
            flip_action.keyup_callback = _exit_mode_late_keyup_cb

        # Could make long-presses start the drag+grab somehow, e.g.
        #if hasattr(mode, '_start_drag'):
        #    mode._start_drag(mode.doc.tdw, ev)
        return False


    ## Mode stack reflection

    def mode_radioaction_changed_cb(self, action, current_action):
        """Callback: radio action controlling the modes stack activated

        :param action: the lead gtk.RadioAction
        :param current_action: the newly active gtk.RadioAction

        Mode classes are looked up via `canvasevent.ModeRegistry` based
        on the name of the action. This action instantiates the mode and
        pushes it onto the mode stack unless the active mode is already
        an instance of the mode class.

        """
        # Update the mode stack so that its top element matches the
        # newly chosen action.
        action_name = current_action.get_name()
        mode_class = canvasevent.ModeRegistry.get_mode_class(action_name)
        if mode_class is None:
            warn("%r not registered: check imports" % (action_name,), Warning)
            return

        if self.modes.top.__class__ is not mode_class:
            mode = mode_class()
            self.modes.context_push(mode)

    def mode_stack_changed_cb(self, mode):
        """Callback: make actions follow changes to the mode stack"""
        # Activate the action corresponding to the current top mode.
        logger.debug("Mode changed: %r", self.modes)
        action_name = getattr(mode, '__action_name__', None)
        if action_name is None:
            return None
        action = self.app.builder.get_object(action_name)
        if action is not None:
            # Not every mode has a corresponding action
            if not action.get_active():
                action.set_active(True)

