# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2010-2019 by the MyPaint Development Team.
# Copyright (C) 2007-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Top-level document controller classes

The classes defined here oparate as controllers in the MVC sense,
i.e. they convert user input into updates to the document model.
"""

## Imports

from __future__ import division, print_function

import os
import os.path
import math
from warnings import warn
import weakref
import logging

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib

import lib.document
import lib.layer
import lib.helpers
from lib.helpers import clamp
import lib.observable
from . import stategroup
import gui.application
import gui.mode
import gui.colorpicker   # purely for registration
import gui.symmetry   # registration only
import gui.freehand
import gui.inktool   # registration only
import gui.layerprops
import gui.buttonmap
import gui.externalapp
import gui.device
import gui.backgroundwindow
import gui.tileddrawwidget
from gui.widgets import with_wait_cursor
from lib.gettext import gettext as _
from lib.gettext import C_
from lib.modes import PASS_THROUGH_MODE

logger = logging.getLogger(__name__)


## Class definitions

class CanvasController (object):
    """Minimal canvas controller using a stack of modes.

    Basic CanvasController objects can be set up to handle scroll events
    like zooming or rotation only, pointer events like drawing only, or
    both.

    The actual interpretation of each event is delegated to the top item
    on the controller's modes stack: see `gui.mode` for details.
    Simpler modes may assume the basic CanvasController interface, more
    complex ones may require the full Document interface.

    """

    # NOTE: If multiple, editable, views of a single model are required,
    # NOTE: then this interface will have to be revised.

    ## Initialization

    def __init__(self, tdw):
        """Initialize.

        :param tdw: The view widget to attach handlers onto.
        :type tdw: gui.tileddrawwidget.TiledDrawWidget

        """
        object.__init__(self)
        self.tdw = tdw     #: the TiledDrawWidget being controlled.
        self.modes = gui.mode.ModeStack(self)  #: stack of delegates
        self.modes.default_mode_class = gui.freehand.FreehandMode

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
        self.tdw.add_events(Gdk.EventMask.SCROLL_MASK)

    ## Low-level GTK event handlers: delegated to the current mode

    def button_press_cb(self, tdw, event):
        """Delegate a button-press-event to the current mode"""
        mode = self.modes.top
        result = mode.button_press_cb(tdw, event)
        self._update_last_event_info(tdw, event)
        return result

    def button_release_cb(self, tdw, event):
        """Delegate a button-release-event to the current mode"""
        mode = self.modes.top
        result = mode.button_release_cb(tdw, event)
        self._update_last_event_info(tdw, event)
        return result

    def motion_notify_cb(self, tdw, event, mode=None):
        """Delegate a motion-notify-event to the current mode"""
        mode = mode or self.modes.top
        result = mode.motion_notify_cb(tdw, event)
        self._update_last_event_info(tdw, event)
        return result

    def scroll_cb(self, tdw, event):
        """Delegate a scroll-event to the current mode"""
        mode = self.modes.top
        result = mode.scroll_cb(tdw, event)
        self._update_last_event_info(tdw, event)
        return result

    def key_press_cb(self, win, tdw, event):
        """Delegate a key-press-event to the current mode"""
        mode = self.modes.top
        return mode.key_press_cb(win, tdw, event)

    def key_release_cb(self, win, tdw, event):
        """Delegate a key-release-event to the current mode"""
        mode = self.modes.top
        return mode.key_release_cb(win, tdw, event)

    def _update_last_event_info(self, tdw, event):
        # Update the stored details of the last event delegated.
        tdw.last_tdw_event_info = event.time, event.x, event.y

    def get_last_event_info(self, tdw):
        """Get details of the last event delegated to a mode in the stack.

        :rtype: tuple
        :returns: event details: ``(time, x, y)``

        """
        info = tdw.last_tdw_event_info
        if not info:
            return 0, None, None
        else:
            return info

    ## High-level event observing interface

    @lib.observable.event
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
        Controller and the GTK event as arguments. This is a good place to
        listen for "just painted something" events in some cases; ``app.brush``
        will contain everything needed about the input stroke which is ending.
        """
        pass

    @lib.observable.event
    def input_stroke_started(self, event):
        """Event: input stroke just started

        Callbacks interested in the start of an input stroke should be attached
        here. See `input_stroke_ended()`.
        """
        pass


REDO_CMD = _("Redo %s")
REDO_PLAIN = _("Redo")
UNDO_CMD = _("Undo %s")
UNDO_PLAIN = _("Undo")


class Document (CanvasController):  # TODO: rename to "DocumentController"
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
    ROTATION_STEP = 2 * math.pi / 16

    # Constants for rotating and zooming by increments
    ROTATE_ANTICLOCKWISE = 4  #: Rotation step direction: RotateLeft
    ROTATE_CLOCKWISE = 8  #: Rotation step direction: RotateRight
    ZOOM_INWARDS = 16  #: Zoom step direction: into the canvas
    ZOOM_OUTWARDS = 32  #: Zoom step direction: out of the canvas

    # Step zoom and rotations can happen at specified locations, or these.
    CENTER_ON_VIEWPORT = 1  #: Zoom or rotate at the canvas center
    CENTER_ON_POINTER = 2  #: Zoom/rotate at the last observed pointer pos

    # Constants for panning (movement) by increments
    PAN_STEP = 0.2  #: Stepwise panning amount: proportion of the canvas size
    PAN_LEFT = 1  #: Stepwise panning direction: left
    PAN_RIGHT = 2  #: Stepwise panning direction: right
    PAN_UP = 3  #: Stepwise panning direction: up
    PAN_DOWN = 4  #: Stepwise panning direction: down

    # Picking
    MIN_PICKING_OPACITY = 0.1
    PICKING_RADIUS = 5

    # Opacity changing
    OPACITY_STEP = 0.08

    # Registration
    _INSTANCE_REFS = []

    ## Registry of controller instances

    @classmethod
    def get_instances(cls):
        """Iterates across all Document instances

        :returns: All Document instances registered
        :rtype: iterable
        """
        for instance_ref in cls._INSTANCE_REFS:
            instance = instance_ref()
            if not instance:
                continue
            yield instance

    @classmethod
    def get_primary_instance(cls):
        """Return the main application working doc"""
        primary_instance = None
        for instance in cls.get_instances():
            primary_instance = instance
            break
        return primary_instance

    @classmethod
    def get_active_instance(cls):
        """Get the Document instance which has the active tdw."""
        active_tdw = gui.tileddrawwidget.TiledDrawWidget.get_active_tdw()
        for instance in cls.get_instances():
            if instance.tdw is active_tdw:
                return instance
        return None

    ## Construction

    def __init__(self, app, tdw, model):
        """Constructor for a document controller

        :param app: main application instance
        :type app: gui.application.Application
        :param tdw: primary canvas widget for this controller
        :type tdw: gui.tileddrawwidget.TiledDrawWidget
        :param model: model document to be controlled and reflected
        :type model: lib.document.Document

        Document controllers initialized here are registered
        automatically with the class. See get_instances().

        """
        self.app = app
        self.model = model
        CanvasController.__init__(self, tdw)

        # Current mode observation
        self._top_mode = self.modes.top
        self.modes.changed += self._modestack_changed_cb

        self.model.frame_enabled_changed += self._frame_enabled_changed_cb
        layerstack = self.model.layer_stack
        layerstack.symmetry_state_changed += self._symmetry_state_changed_cb

        # Deferred until after the app starts (runs in the first idle-
        # processing phase) as a workaround for https://gna.org/bugs/?14372
        # ([Windows] crash when moving the pen during startup)
        GLib.idle_add(self.init_pointer_events)
        GLib.idle_add(self.init_scroll_events)

        self.zoomlevel_values = [
            # micro
            1.0 / 16, 1.0 / 8, 2.0 / 11, 0.25, 1.0 / 3, 0.50, 2.0 / 3,
            # normal
            1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 8.0,
            # macro
            11.0, 16.0, 23.0, 32.0, 45.0, 64.0,
        ]

        default_zoom = self.app.preferences['view.default_zoom']
        self.tdw.scale = default_zoom
        self.tdw.zoom_min = min(self.zoomlevel_values)
        self.tdw.zoom_max = max(self.zoomlevel_values)

        # Device-specific brushes: save at end of stroke
        self.input_stroke_ended += self._input_stroke_ended_cb

        self._init_stategroups()

        leader = self.get_primary_instance()
        if leader is not None:
            # This is a secondary controller (e.g. the scratchpad)
            # which plays follow-the-leader for some events.
            assert isinstance(leader, Document)
            self.action_group = leader.action_group  # hack, but needed by tdw
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
            toggle_action.set_active(
                self.app.preferences['misc.context_restores_color']
            )

        #: Saved transformation to allow FitView to be toggled.
        self.saved_view = None

        #: Viewport change/manipulation observers.
        self.view_changed_observers = []
        self.view_changed_observers.append(self._view_changed_cb)
        self._view_changed_notification_srcid = None
        self.tdw.connect_after(
            "size-allocate",
            lambda *a: self.notify_view_changed(),
        )

        # Brush settings observers
        self.app.brush.observers.append(self._brush_settings_changed_cb)

        # External file edit requests
        self._layer_edit_manager = gui.externalapp.LayerEditManager(self)

        # Registration
        cls = self.__class__
        cls._INSTANCE_REFS.append(weakref.ref(self))

    def _init_actions(self):
        """Internal: initializes action groups & state reflection"""
        # Actions are defined in resources.xml, just grab a ref to
        # the groups.
        builder = self.app.builder
        self.action_group = builder.get_object('DocumentActions')
        self.modes_action_group = builder.get_object("ModeStackActions")

        # Fine-grained observation of various model objects
        cmdstack = self.model.command_stack
        layerstack = self.model.layer_stack
        self._init_undo_redo_actions()
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
            self._update_layer_slice_actions: [
                layerstack.current_path_updated,
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
            self._update_current_layer_actions: [
                layerstack.current_path_updated,
            ],
            self._update_external_layer_edit_actions: [
                layerstack.current_path_updated,
            ],
            self._update_layer_visible_toggle_from_current_view: [
                self.model.layer_view_manager.current_view_changed,
            ],
        }
        for observer_method, events in observed_events.items():
            for event in events:
                event += observer_method
            observer_method()

    def _init_context_actions(self):
        """Internal: initializes several brush shortcut-key actions"""
        ag = self.action_group
        context_actions = []
        for x in range(10):
            rt = _("Load brush settings from shortcut slot %d") % x
            st = _("Store brush settings in shortcut slot %d") % x
            r = ('Context0%d' % x, None, _('Restore Brush %d') % x,
                 '%d' % x, rt, self.context_cb)
            s = ('Context0%ds' % x, None, _('Save to Brush %d') % x,
                 '<control>%d' % x, st, self.context_cb)
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

        k('bracketleft', 'Smaller')  # GIMP, Photoshop, Painter
        k('bracketright', 'Bigger')  # GIMP, Photoshop, Painter
        k('<control>bracketleft', 'RotateLeft')  # Krita
        k('<control>bracketright', 'RotateRight')  # Krita
        k('less', 'LessOpaque')  # GIMP
        k('greater', 'MoreOpaque')  # GIMP
        k('equal', 'ZoomIn')  # (on US keyboard next to minus)
        k('comma', 'Smaller')  # Krita
        k('period', 'Bigger')  # Krita

        k('BackSpace', 'ClearLayer')

        k('z', 'Undo')  # Old-style MyPaint Shortcut
        k('<control>y', 'Redo')
        k('y', 'Redo')  # Old-style MyPaint Shortcut
        k('<control>w', lambda action: self.app.drawWindow.quit_cb())
        k('KP_Add', 'ZoomIn')
        k('KP_Subtract', 'ZoomOut')
        k('KP_4', 'RotateLeft')  # Blender
        k('KP_6', 'RotateRight')  # Blender
        k('KP_5', 'ResetRotation')
        k('plus', 'ZoomIn')
        k('minus', 'ZoomOut')
        k('<control>plus', 'ZoomIn')  # Krita
        k('<control>minus', 'ZoomOut')  # Krita
        k('bar', 'SymmetryActive')

        k('Left', lambda action: self.pan(self.PAN_LEFT))
        k('Right', lambda action: self.pan(self.PAN_RIGHT))
        k('Down', lambda action: self.pan(self.PAN_DOWN))
        k('Up', lambda action: self.pan(self.PAN_UP))

        k('<control>Left', 'RotateLeft')
        k('<control>Right', 'RotateRight')

    ## Command history traversal actions

    def undo_cb(self, action):
        """Undo action callback"""
        self.model.undo()

    def redo_cb(self, action):
        """Redo action callback"""
        self.model.redo()

    def _init_undo_redo_actions(self):
        ag = self.action_group

        # Icon names
        style_state = self.app.drawWindow.get_style_context().get_state()
        if style_state & Gtk.StateFlags.DIR_LTR:
            direction = 'ltr'
        else:
            direction = 'rtl'
        undo_icon_name = "mypaint-undo-%s-symbolic" % direction
        redo_icon_name = "mypaint-redo-%s-symbolic" % direction

        # Undo
        undo_action = ag.get_action("Undo")
        undo_action.set_icon_name(undo_icon_name)
        self._undo_action = undo_action

        # Redo
        redo_action = ag.get_action("Redo")
        redo_action.set_icon_name(redo_icon_name)
        self._redo_action = redo_action

    def _update_undo_redo(self, action, stack, cmd_str, plain_str):
        """Set label, tooltip and sensitivity"""
        if len(stack) > 0:
            cmd = stack[-1]
            desc = cmd_str % cmd.display_name
        else:
            desc = plain_str
        action.set_label(desc)
        action.set_tooltip(desc)
        action.set_sensitive(len(stack) > 0)

    def _update_command_stack_actions(self, *_ignored):
        """Update the undo and redo actions"""
        stack = self.model.command_stack
        self._update_undo_redo(
            self._undo_action, stack.undo_stack, UNDO_CMD, UNDO_PLAIN)
        self._update_undo_redo(
            self._redo_action, stack.redo_stack, REDO_CMD, REDO_PLAIN)

    ## Event handling

    def button_press_cb(self, tdw, event):
        """Handles button press events received on a canvas"""
        # User-configurable switching between modes, menu popups etc.
        mode = self.modes.top
        consider_mode_switch = (
            mode.supports_button_switching
            and not getattr(mode, 'in_drag', False)
            and (
                event.button == 1
                or not (event.state & Gdk.ModifierType.BUTTON1_MASK)
            ))

        # Look up per-device user settings
        mon = self.app.device_monitor
        dev = event.get_source_device()
        dev_settings = mon.get_device_settings(dev)

        if consider_mode_switch:
            buttonmap = self.app.button_mapping
            modifiers = event.state & Gtk.accelerator_get_default_mod_mask()
            button = event.button
            action_names = [buttonmap.lookup(modifiers, button)]

            # Allow button 1 to initiate switches of mode as button 2 if
            # the device is navigation-only. This allows single-finger
            # panning with specially configured touchscreens while we're
            # not handling touch separately. Remove this when we
            # implement real touch event support.
            if dev_settings and (button == 1):
                if dev_settings.usage == gui.device.AllowedUsage.NAVONLY:
                    action_names.insert(0, buttonmap.lookup(modifiers, 2))

            # Limit to actions in the whitelist, unless it's empty
            limited_to = mode.permitted_switch_actions
            for name in action_names:
                if name and (not limited_to or name in limited_to):
                    return self._dispatch_named_action(None, tdw, event, name)

        # User-configurable forbidding of particular devices
        if dev_settings:
            if not (dev_settings.usage_mask & mode.pointer_behavior):
                return True

        # Normal event dispatch to the top mode on the mode stack
        return CanvasController.button_press_cb(self, tdw, event)

    def button_release_cb(self, tdw, event):
        """Handles button release events received on a canvas"""
        # User-configurable forbidding of particular devices
        mode = self.modes.top
        mon = self.app.device_monitor
        dev = event.get_source_device()
        dev_settings = mon.get_device_settings(dev)
        if dev_settings:
            if not (dev_settings.usage_mask & mode.pointer_behavior):
                return True
        # Normal event dispatch
        return CanvasController.button_release_cb(self, tdw, event)

    def motion_notify_cb(self, tdw, event):
        """Handles motion-notify events received on a canvas"""
        mode = self._top_mode
        mon = self.app.device_monitor
        dev = event.get_source_device()
        dev_settings = mon.get_device_settings(dev)
        if dev_settings:
            if not (dev_settings.usage_mask & mode.pointer_behavior):
                return True
        # Normal event dispatch
        CanvasController.motion_notify_cb(self, tdw, event, mode)
        return False  # XXX don't consume motions to allow workspace autohide

    def scroll_cb(self, tdw, event):
        """Handles scroll events received on a canvas"""
        mode = self.modes.top
        mon = self.app.device_monitor
        dev = event.get_source_device()
        dev_settings = mon.get_device_settings(dev)
        if dev_settings:
            if not (dev_settings.usage_mask & mode.scroll_behavior):
                return True
        CanvasController.scroll_cb(self, tdw, event)

    def key_press_cb(self, win, tdw, event):
        """Handles key-press events received on the main window"""
        # User-configurable switching between modes, menu popups etc.
        mode = self.modes.top
        consider_mode_switch = (
            mode.supports_button_switching
            and not getattr(mode, 'in_drag', False)
        )
        if consider_mode_switch:
            # Naively pick an action based on the button map
            buttonmap = self.app.button_mapping
            action_name = None
            mods = self.get_current_modifiers()
            is_modifier = (
                event.is_modifier
                or (mods != 0 and event.keyval != Gdk.KEY_space)
            )
            if is_modifier:
                # If the keypress is a modifier only, determine the
                # modifier mask a subsequent Button1 press event would
                # get. This is used for early spring-loaded mode
                # switching.
                action_name = buttonmap.get_unique_action_for_modifiers(mods)
                # Only mode-based immediate dispatch is allowed, however.
                # Might relax this later.
                if action_name is not None:
                    if not action_name.endswith("Mode"):
                        action_name = None
            else:
                # Strategy 2: pretend that the space bar is really button 2.
                if event.keyval == Gdk.KEY_space:
                    action_name = buttonmap.lookup(mods, 2)

            # Limit to actions in the whitelist, unless it's empty
            limited_to = mode.permitted_switch_actions
            if action_name and (not limited_to or action_name in limited_to):
                # If we found something to do, dispatch;
                return self._dispatch_named_action(
                    win,
                    tdw,
                    event,
                    action_name,
                )

            # Explain what's possible from here with some extra
            # modifiers and button presses.
            self._update_key_pressed_status_message()

            # TODO: Maybe display the inactive cursor belonging to the
            # TODO:   button1 binding for these modifiers. Blocker: need
            # TODO:   to do it without instantiating the handler class.
            # btn1_action_name = buttonmap.lookup(mods, 1)
            # btn1_handler_type, btn1_handler = gui.buttonmap\
            #    .get_handler_object(
            #       self.app,
            #       btn1_action_name,
            #    )
            # if btn1_handler_type == 'mode_class':
            #    assert issubclass(btn1_handler, gui.mode.DragMode)
            #    btn1_cursor = btn1_handler.inactive_cursor    # fails.
            #    if btn1_cursor:
            #        self.tdw.set_override_cursor(btn1_cursor)

        # Normal event dispatch
        return CanvasController.key_press_cb(self, win, tdw, event)

    def key_release_cb(self, win, tdw, event):
        self._update_key_pressed_status_message()
        return CanvasController.key_release_cb(self, win, tdw, event)

    def _dispatch_named_action(self, win, tdw, event, action_name):
        """Dispatch an action looked up via the buttonmap"""
        app = self.app
        drawwindow = app.drawWindow
        if action_name == 'ShowPopupMenu':
            # Unfortunately still a special case.
            # Just firing the action doesn't work well with pads which fire a
            # button-release event immediately after the button-press.
            # Name it after the action however, in case we find a fix.
            drawwindow.show_popupmenu(event=event)
            return True
        handler_type, handler = gui.buttonmap.get_handler_object(
            app, action_name,
        )
        if handler_type == 'mode_class':
            # Transfer control to another mode temporarily.
            assert issubclass(handler, gui.mode.DragMode)
            if issubclass(handler, gui.mode.OneshotDragMode):
                mode = handler(temporary_activation=True)
            else:
                mode = handler()
            self.modes.push(mode)
            if win is not None:
                return mode.key_press_cb(win, tdw, event)
            else:
                return mode.button_press_cb(tdw, event)
        elif handler_type == 'popup_state':
            # Still needed. The code is more tailored to MyPaint's
            # purposes. The names are action names, but have the more
            # tailored popup states code shadow generic action activation.
            if win is not None:
                # WORKAROUND: dispatch keypress events via the kbm so it can
                # keep track of pressed-down keys. Popup states become upset if
                # this doesn't happen: https://gna.org/bugs/index.php?20325
                action = app.find_action(action_name)
                return app.kbm.activate_keydown_event(action, event)
            else:
                # Pointer: popup states handle these themselves sanely.
                handler.activate(event)
                return True
        elif handler_type == 'gtk_action':
            # Generic named action activation. GtkActions trigger without
            # event details, so they're less flexible.
            # Hack: Firing the action in an idle handler helps with
            # actions that are sensitive to immediate button-release
            # events. But not ShowPopupMenu, sadly: we'd break button
            # hold behaviour for more reasonable devices if we used
            # this trick.
            GLib.idle_add(handler.activate)
            return True
        else:
            return False

    def _get_key_pressed_status_message_context_id(self):
        statusbar = self.app.statusbar
        try:
            context_id = self.__key_pressed_msg_statusbar_context
        except AttributeError:
            context_id = statusbar.get_context_id("key-pressed-msg")
            self.__key_pressed_msg_statusbar_context = context_id
        return context_id

    def get_current_modifiers(self):
        """Returns the current set of modifier keys as a Gdk bitmask.

        :returns: The current set of modifier keys.
        :rtype: Gdk.ModifierType

        This method should only be used in

        * Handlers for keypress events
          when the key in question is itself a modifier,
        * Handlers of multiple types of event (both key and pointer),
        * When the triggering event simply isn't available.

        Normal pointer button event handling should use
        ``event.state & Gtk.accelerator_get_default_mod_mask()``
        instead.

        """
        win = self.tdw.get_window()
        display = self.tdw.get_display()
        devmgr = display and display.get_device_manager() or None
        coredev = devmgr and devmgr.get_client_pointer() or None
        if coredev and win:
            win_, x, y, mask = win.get_device_position(coredev)
            return mask & Gtk.accelerator_get_default_mod_mask()
        return Gdk.ModifierType(0)

    def _update_key_pressed_status_message(self):
        """Update app statusbar to explain what modes are reachable"""
        context_id = self._get_key_pressed_status_message_context_id()
        statusbar = self.app.statusbar
        statusbar.remove_all(context_id)

        btn_map = self.app.button_mapping
        mods = self.get_current_modifiers()
        if mods == 0:
            return
        poss_list = btn_map.lookup_possibilities(mods)
        if not poss_list:
            return
        poss_list.sort()
        poss_msgs = []
        current_mode = self.modes.top
        limited_to = current_mode.permitted_switch_actions
        for pmods, button, action_name in poss_list:
            # Limit to actions in the white list, unless it's empty
            if limited_to and action_name not in limited_to:
                continue
            # Don't repeat what's currently held
            pmods = pmods & ~mods
            label = gui.buttonmap.button_press_displayname(button, pmods, True)
            mode_class = gui.mode.ModeRegistry.get_mode_class(action_name)
            mode_desc = None
            if mode_class:
                mode_desc = mode_class.get_name()
            else:
                action = self.app.find_action(action_name)
                if action:
                    mode_desc = action.get_label()
            if mode_desc:
                # TRANSLATORS: Statusbar message explaining button and modifier
                # TRANSLATORS: combinations used to access modes/tools/actions.
                # TRANSLATORS: "With <current-modifiers> held down: <list>"
                msg = _(u"{button_combination} is {resultant_action}").format(
                    button_combination=label,
                    resultant_action=mode_desc,
                )
                poss_msgs.append(msg)
        if not poss_msgs:
            return
        # TRANSLATORS: This is a separator for the list of button actions
        # TRANSLATORS: that appears when a modifier key is held down.
        # TRANSLATORS: search for " held down: " (incl spaces) for the context
        sep = _(";  ")
        # TRANSLATORS: "With <current-modifiers> held down: <separated-list>"
        # TRANSLATORS: Action names may contain coordinating conjunctions such
        # TRANSLATORS: as the English "and", so use appropriate punctuation or
        # TRANSLATORS: wording for the separator. Also a little more spacing
        # TRANSLATORS: than normal looks good here.
        msg = _("With {modifiers} held down:  {button_actions}.").format(
            modifiers=Gtk.accelerator_get_label(0, mods),
            button_actions=sep.join(poss_msgs),
        )
        self.app.statusbar.push(context_id, msg)

    ## Copy/Paste

    def _get_clipboard(self):
        """Internal: return the GtkClipboard for the current display"""
        display = self.tdw.get_display()
        cb = Gtk.Clipboard.get_for_display(display, Gdk.SELECTION_CLIPBOARD)
        return cb

    def copy_cb(self, action):
        """``CopyLayer`` GtkAction callback: copy layer to clipboard"""
        # use the full document bbox, so we can paste layers back to the
        # correct position
        rootstack = self.model.layer_stack
        if self.app.preferences.get("ui.legacy-copy-paste", False):
            bbox = self.model.get_bbox()
        else:
            bbox = rootstack.current.get_bbox()
        if bbox.w == 0 or bbox.h == 0:
            self.app.show_transient_message(C_(
                "Statusbar message: copy result",
                u"Empty document, nothing copied."
            ))
            return
        pixbuf = rootstack.render_layer_as_pixbuf(
            rootstack.current, bbox,
            alpha=True,
        )
        cb = self._get_clipboard()
        cb.set_image(pixbuf)
        self.app.show_transient_message(C_(
            "Statusbar message: copy result",
            u"Copied layer as {w}×{h} image."
        ).format(
            w=pixbuf.get_width(),
            h=pixbuf.get_height(),
        ))

    def paste_cb(self, action):
        """``PasteLayer`` GtkAction callback: replace layer with clipboard"""
        clipboard = self._get_clipboard()
        # Windows requires the available targets to be polled first.
        # Ensure that happens fully before polling for image data.
        # If we don't do this, nothing other than the 1st pasted image
        # can be pasted: https://github.com/mypaint/mypaint/issues/595
        targs_avail, targets = clipboard.wait_for_targets()
        if not targs_avail:
            self.app.show_transient_message(C_(
                "Statusbar message: paste result",
                u"Nothing on clipboard.",
            ))
            return
        logger.debug("Paste: available targets: %r", [str(a) for a in targets])
        # Then grab any available image, also synchronously
        pixbuf = clipboard.wait_for_image()
        if not pixbuf:
            self.app.show_transient_message(C_(
                "Statusbar message: paste result",
                u"Clipboard does not contain an image.",
            ))
            return

        # Supports old copy-paste, useful if moving a layer from one document
        # to another with the same bounding box.
        if self.app.preferences.get("ui.legacy-copy-paste", False):
            x, y, __, __ = self.model.get_bbox()
        # If pasting with a shortcut, the upper left corner of the content
        # is aligned with the cursor location, otherwise it is centered.
        elif action.keydown:
            x, y = self.tdw.display_to_model(
                *self.get_last_event_info(self.tdw)[1:])
        else:
            x, y = self.tdw.get_center_model_coords()
            x -= pixbuf.get_width()/2.0
            y -= pixbuf.get_height()/2.0
        try:
            self.model.load_layer_from_pixbuf(
                pixbuf, int(x), int(y), to_new_layer=True)
        except Exception:
            logger.exception("Paste failed")
            self.app.show_transient_message(C_(
                "Statusbar message: paste result",
                u"Cannot paste into this type of layer."
            ))
            return
        self.app.show_transient_message(C_(
            "Statusbar message: paste result",
            u"Pasted {w}×{h} image.",
        ).format(
            w = pixbuf.get_width(),
            h = pixbuf.get_height(),
        ))

    ## Frame manipulation actions

    def trim_layer_cb(self, action):
        """Trim Layer action: discard tiles outside the frame"""
        self.model.trim_current_layer()

    def _update_trim_layer_action(self, *_ignored):
        """Updates the Trim Layer action's sensitivity"""
        app = self.app
        rootstack = self.model.layer_stack
        current = rootstack.current
        can_trim = current is not rootstack and current.get_trimmable()
        app.find_action("TrimLayer").set_sensitive(can_trim)

    ## Layer tile manipulation commands

    @with_wait_cursor
    def uniq_layer_tiles_cb(self, action):
        """Discard tiles that don't change the backdrop"""
        self.model.uniq_current_layer(pixels=False)

    @with_wait_cursor
    def uniq_layer_pixels_cb(self, action):
        """Discard tiles and pixels that don't change the backdrop"""
        self.model.uniq_current_layer(pixels=True)

    @with_wait_cursor
    def refactor_layer_group_tiles_cb(self, action):
        """Extract common tiles to a new sublayer & delete from all others."""
        self.model.refactor_current_layer_group(pixels=False)

    @with_wait_cursor
    def refactor_layer_group_pixels_cb(self, action):
        """Extract common pixels to a new sublayer & delete from all others."""
        self.model.refactor_current_layer_group(pixels=True)

    def _update_layer_slice_actions(self, *_ignored):
        """Updates the layer-slice actions' sensitivities."""
        app = self.app
        rootstack = self.model.layer_stack
        current = rootstack.current

        can_uniq = (current is not None)
        can_uniq &= isinstance(current, lib.layer.PaintingLayer)
        uniq_acts = [
            "UniqLayerTiles",
            "UniqLayerPixels",
        ]
        for act in uniq_acts:
            app.find_action(act).set_sensitive(can_uniq)

        can_refactor = (current is not None)
        can_refactor &= isinstance(current, lib.layer.LayerStack)
        can_refactor &= (current.mode != PASS_THROUGH_MODE)
        refactor_acts = [
            "RefactorLayerGroupTiles",
            "RefactorLayerGroupPixels",
        ]
        for act in refactor_acts:
            app.find_action(act).set_sensitive(can_refactor)

    def toggle_frame_cb(self, action):
        """Frame Enabled toggle callback"""
        model = self.model
        enabled = bool(model.frame_enabled)
        desired = bool(action.get_active())
        if enabled != desired:
            model.set_frame_enabled(desired, user_initiated=True)

    def _frame_enabled_changed_cb(self, model, enabled):
        """Invoked when the frame changes"""
        action = self.app.find_action("FrameToggle")
        if bool(action.get_active()) != bool(enabled):
            action.set_active(enabled)

    ## Layer and stroke picking

    def blink_layer(self, action=None):
        if self.app.preferences.get("ui.blink_layers", True):
            self.layerblink_state.activate(action)
        elif self.model.layer_stack.current_layer_solo:
            self.tdw.queue_draw()

    def pick_context(self, x, y, action=None):
        """Picks layer and brush

        :param int x: X coord for pick, in the model's coordinate space
        :param int y: Y coord for pick, in the model's coordinate space
        :param Gdk.Action action: initiating action

        If the document has a pickable layer which has a brushstroke
        under the pick position, that layer is selected, and the
        brushstroke's settings are assigned to the current brush.

        The initiating action is used for coordinating keyboard releases
        ending the state. See gui.stategroup.

        """
        layers = self.model.layer_stack
        old_path = layers.current_path
        for c_path, c_layer in self._layer_picking_iter():
            if not self._layer_is_pickable(c_path, (x, y)):
                continue
            self.model.select_layer(path=c_path)
            if c_path != old_path:
                self.blink_layer()
            # Find the most recent (last) stroke at the pick point
            si = layers.current.get_stroke_info_at(x, y)
            if si:
                self.app.restore_brush_from_stroke_info(si)
                corners = self.tdw.get_corners_model_coords()
                bbox = lib.helpers.rotated_rectangle_bbox(corners)
                self.strokeblink_state.activate(
                    action,
                    strokeshape=si,
                    bbox=bbox,
                    center=(x, y),
                )
            return

    def pick_layer(self, x, y, action=None):
        """Picks layer only

        :param int x: X coord for pick, in the model's coordinate space
        :param int y: Y coord for pick, in the model's coordinate space
        :param Gdk.Action action: initiating action

        If the document has a pickable layer under the pick position,
        that layer is selected. Fn no layer is pickable there, the
        bottom layer is selected instead.

        The initiating action is used for coordinating keyboard releases
        ending the state. See gui.stategroup if you dare.

        """
        for p_path, p_layer in self._layer_picking_iter():
            if not self._layer_is_pickable(p_path, (x, y)):
                continue
            self.model.select_layer(path=p_path)
            self.blink_layer(action)
            return
        self.model.select_layer(path=(0,))
        self.blink_layer(action)

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
        parents = set()
        for path, layer in layer_stack.walk():
            if path in parents:
                continue
            parent_path = path[:-1]
            parents.add(parent_path)
            yield (path, layer)

    ## Layer action callbacks

    def clear_layer_cb(self, action):
        """``ClearLayer`` GtkAction callback"""
        self.model.clear_current_layer()

    def remove_layer_cb(self, action):
        """``RemoveLayer`` GtkAction callback"""
        self.model.remove_current_layer()

    def _update_current_layer_actions(self, *_ignored):
        """Update sensitivity of actions working on the current layer"""
        app = self.app
        rootstack = self.model.layer_stack
        have_current = bool(rootstack.current_path)
        current_layer_action_names = [
            "RemoveLayer",
            "ClearLayer",
            "DuplicateLayer",
            "NewPaintingLayerAbove",  # but not below so the button still works
            "LayerMode",  # the modes submenu
            "LayerProperties",
            "LayerVisibleToggle",
            "LayerLockedToggle",
            "LayerOpacityMenu",
            "IncreaseLayerOpacity",
            "DecreaseLayerOpacity",
            "CopyLayer",
        ]
        for name in current_layer_action_names:
            app.find_action(name).set_sensitive(have_current)

    def normalize_layer_mode_cb(self, action):
        """``NormalizeLayerMode`` GtkAction callback"""
        self.model.normalize_layer_mode()

    def _update_normalize_layer_action(self, *_ignored):
        """Updates the Normalize Layer Mode action's sensitivity"""
        app = self.app
        rootstack = self.model.layer_stack
        current = rootstack.current
        can_normalize = (current is not rootstack
                         and current.get_mode_normalizable())
        app.find_action("NormalizeLayerMode").set_sensitive(can_normalize)

    ## Layer selection (current layer path in the tree)

    def select_layer_below_cb(self, action):
        """``SelectLayerBelow`` GtkAction callback"""
        layers = self.model.layer_stack
        path = layers.get_current_path()
        path = layers.path_below(path)
        if path:
            self.model.select_layer(path=path)

        if self.model.layer_stack.current_layer_solo:
            self.tdw.queue_draw()
        else:
            self.blink_layer(action)

    def select_layer_above_cb(self, action):
        """``SelectLayerAbove`` GtkAction callback"""
        layers = self.model.layer_stack
        path = layers.get_current_path()
        path = layers.path_above(path)
        if path:
            self.model.select_layer(path=path)

        if self.model.layer_stack.current_layer_solo:
            self.tdw.queue_draw()
        else:
            self.blink_layer(action)

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
        rootstack = self.model.layer_stack
        if rootstack.current is rootstack:
            return
        opacity = rootstack.current.opacity
        opacity = clamp(opacity + self.OPACITY_STEP, 0.0, 1.0)
        self.model.set_current_layer_opacity(opacity)

    def layer_decrease_opacity(self, action):
        """``DecreaseLayerOpacity`` GtkAction callback"""
        rootstack = self.model.layer_stack
        if rootstack.current is rootstack:
            return
        opacity = rootstack.current.opacity
        opacity = clamp(opacity - self.OPACITY_STEP, 0.0, 1.0)
        self.model.set_current_layer_opacity(opacity)

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

    ## Background layer

    def reset_background(self):
        """Loads the default background layer."""

        # Load the default background image if one exists
        layer_stack = self.model.layer_stack
        for datapath in [self.app.user_datapath, self.app.datapath]:
            bg_path = os.path.join(
                datapath,
                gui.backgroundwindow.BACKGROUNDS_SUBDIR,
                gui.backgroundwindow.DEFAULT_BACKGROUND,
            )
            if not os.path.exists(bg_path):
                continue
            bg, errors = gui.backgroundwindow.load_background(bg_path)
            if bg:
                layer_stack.set_background(bg, make_default=True)
                logger.info("Initialized background from %r", bg_path)
                return
            else:
                logger.warning(
                    "Failed to load saved default background image %r",
                    bg_path,
                )
                if errors:
                    for error in errors:
                        logger.warning("warning: %r", error)

        # Otherwise, try to use a sensible fallback background image.
        bg_path = os.path.join(
            self.app.datapath,
            gui.backgroundwindow.BACKGROUNDS_SUBDIR,
            gui.backgroundwindow.FALLBACK_BACKGROUND,
        )
        bg, errors = gui.backgroundwindow.load_background(bg_path)
        if bg:
            layer_stack.set_background(bg, make_default=True)
            logger.info("Initialized background from %r", bg_path)
            return
        else:
            logger.warning(
                "Failed to load fallback background image %r",
                bg_path,
            )
            if errors:
                for error in errors:
                    logger.warning("warning: %r", error)

        # Double fallback. Just use a color.
        bg_color = (0xa8, 0xa4, 0x98)
        layer_stack.set_background(bg_color, make_default=True)
        logger.info("Initialized background to %r", bg_color)

    ## Layer stack order (bubbling)

    def reorder_layer_cb(self, action):
        """Changes the z-order of a layer (GtkAction callback)

        The direction the layer moves depends on the action name:
        "RaiseLayerInStack" or "LowerLayerInStack".
        """
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
            down_poss = deep or current_path[0] < len(root) - 1
            up_poss = deep or current_path[0] > 0
        else:
            down_poss = False
            up_poss = False
        app.find_action("RaiseLayerInStack").set_sensitive(up_poss)
        app.find_action("LowerLayerInStack").set_sensitive(down_poss)

    ## Simple (non-toggle) layer commands

    def new_layer_cb(self, action):
        """Callback: new layer

        Where the new layer is created, and the layer's type, depends on
        the action's name.

        """
        layers = self.model.layer_stack

        layer_class = lib.layer.PaintingLayer
        layer_kwds = {}
        edit_externally = False
        if "Vector" in action.get_name():
            layer_class = lib.layer.VectorLayer
            edit_externally = True
            # The new layer will be created with an outline of a random
            # color showing the position of the view at the time it was
            # created. Its bbox encloses this outline.
            corners = self.tdw.get_corners_model_coords()
            x, y, w, h = lib.helpers.rotated_rectangle_bbox(corners)
            layer_kwds["outline"] = corners
            layer_kwds["x"] = x
            layer_kwds["y"] = y
            layer_kwds["w"] = w
            layer_kwds["h"] = h
        elif "Group" in action.get_name():
            layer_class = lib.layer.LayerStack

        path = layers.current_path
        if not path:
            path = (-1,)
        elif 'Above' in action.get_name():
            path = layers.path_above(path, insert=True)
        else:
            path = layers.path_below(path, insert=True)
        assert path is not None

        self.model.add_layer(path, layer_class=layer_class, **layer_kwds)
        self.blink_layer(action)
        if edit_externally:
            self._begin_external_layer_edit()

    def merge_layer_down_cb(self, action):
        """Action callback: squash current layer into the one below it"""
        if self.model.merge_current_layer_down():
            self.blink_layer(action)

    def merge_visible_layers_cb(self, action):
        """Action callback: squash all visible layers into one"""
        self.model.merge_visible_layers()
        self.blink_layer(action)

    def new_layer_merged_from_visible_cb(self, action):
        """Action callback: combine all visible layers into a new one"""
        self.model.new_layer_merged_from_visible()
        self.blink_layer(action)

    def _update_merge_layer_down_action(self, *_ignored):
        """Updates the layer Merge Down action's sensitivity"""
        # This may change in response to the path changing *or* the
        # mode property of the current or underlying layer changing.
        app = self.app
        rootstack = self.model.layer_stack
        current = rootstack.current_path
        can_merge = (current is not rootstack
                     and bool(rootstack.get_merge_down_target(current)))
        app.find_action("MergeLayerDown").set_sensitive(can_merge)

    def duplicate_layer_cb(self, action):
        """``DuplicateLayer`` GtkAction callback: clone the current layer"""
        self.model.duplicate_current_layer()

    def layer_properties_cb(self, action):
        """LayerProperties GtkAction callback: layer properties dialog"""
        layer = self.model.layer_stack.get_current()
        if layer is self.model.layer_stack:
            return
        dialog = gui.layerprops.LayerPropertiesDialog(
            self.app.drawWindow,
            self.model,
        )
        dialog.run()
        dialog.destroy()

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

    def fakepressure_increase_cb(self, action):
        """``Fake Pressure Harder`` GtkAction callback"""
        fp = self.app.fakepressure
        self.app.fakepressure = fp * 1.1
        current_mode = self.modes.top
        widget = current_mode.get_options_widget()
        try:
            widget.fakepressure_modified_cb(self.app.fakepressure)
        except AttributeError:
            logger.info("Mode doesn't have fakepressure")
            return

    def fakepressure_decrease_cb(self, action):
        """``Fake Pressure Softer`` GtkAction callback"""
        fp = self.app.fakepressure
        self.app.fakepressure = fp / 1.1
        current_mode = self.modes.top
        widget = current_mode.get_options_widget()
        try:
            widget.fakepressure_modified_cb(self.app.fakepressure)
        except AttributeError:
            logger.info("Mode doesn't have fakepressure")
            return

    def fakerotation_increase_cb(self, action):
        """``Fake Rotation Twist Right`` GtkAction callback"""
        fr = self.app.fakerotation
        self.app.fakerotation = (fr + 0.0625) % 1.0
        current_mode = self.modes.top
        widget = current_mode.get_options_widget()
        try:
            widget.fakerotation_modified_cb(self.app.fakerotation)
        except AttributeError:
            logger.info("Mode doesn't have fakerotation")
            return

    def fakerotation_decrease_cb(self, action):
        """``Fake Rotation Twist Left`` GtkAction callback"""
        fr = self.app.fakerotation
        self.app.fakerotation = (fr - 0.0625) % 1.0
        current_mode = self.modes.top
        widget = current_mode.get_options_widget()
        try:
            widget.fakerotation_modified_cb(self.app.fakerotation)
        except AttributeError:
            logger.info("Mode doesn't have fakerotation")
            return

    def brush_bigger_cb(self, action):
        """``Bigger`` GtkAction callback"""
        adj = self.app.brush_adjustment['radius_logarithmic']
        adj.set_value(adj.get_value() + 0.3)

    def brush_smaller_cb(self, action):
        """``Smaller`` GtkAction callback"""
        adj = self.app.brush_adjustment['radius_logarithmic']
        adj.set_value(adj.get_value() - 0.3)

    def more_opaque_cb(self, action):
        """``MoreOpaque`` GtkAction callback"""
        # FIXME: hm, looks this slider should be logarithmic?
        adj = self.app.brush_adjustment['opaque']
        adj.set_value(adj.get_value() * 1.8)

    def less_opaque_cb(self, action):
        """``MoreOpaque`` GtkAction callback"""
        adj = self.app.brush_adjustment['opaque']
        adj.set_value(adj.get_value() / 1.8)

    def brighter_cb(self, action):
        """``Brighter`` GtkAction callback: lighten the brush color"""
        # TODO: use HCY?
        h, s, v = self.app.brush.get_color_hsv()
        v += 0.08
        if v > 1.0:
            v = 1.0
        self.app.brush.set_color_hsv((h, s, v))

    def darker_cb(self, action):
        """``Darker`` GtkAction callback: darken the brush color"""
        # TODO: use HCY?
        h, s, v = self.app.brush.get_color_hsv()
        v -= 0.08
        # stop a little higher than 0.0, to avoid resetting hue to 0
        if v < 0.005:
            v = 0.005
        self.app.brush.set_color_hsv((h, s, v))

    def increase_hue_cb(self, action):
        """Clockwise hue rotation ("IncreaseHue" action)."""
        # TODO: use HCY?
        h, s, v = self.app.brush.get_color_hsv()
        e = 0.015
        h = (h + e) % 1.0
        self.app.brush.set_color_hsv((h, s, v))

    def decrease_hue_cb(self, action):
        """Anticlockwise hue rotation ("DecreaseHue" action)."""
        # TODO: use HCY?
        h, s, v = self.app.brush.get_color_hsv()
        e = 0.015
        h = (h - e) % 1.0
        self.app.brush.set_color_hsv((h, s, v))

    def purer_cb(self, action):
        """``Purer`` GtkAction callback: make the brush color less grey"""
        # TODO: use HCY?
        h, s, v = self.app.brush.get_color_hsv()
        s += 0.08
        if s > 1.0:
            s = 1.0
        self.app.brush.set_color_hsv((h, s, v))

    def grayer_cb(self, action):
        """``Grayer`` GtkAction callback: make the brush color more grey"""
        # TODO: use HCY?
        h, s, v = self.app.brush.get_color_hsv()
        s -= 0.08
        # stop a little higher than 0.0, to avoid resetting hue to 0
        if s < 0.005:
            s = 0.005
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
        """GtkToggleAction callback for whether brushkeys restore color"""
        value = bool(action.get_active())
        self.app.preferences['misc.context_restores_color'] = value

    ## UI feedback for current layer/stroke

    def strokeblink_state_enter(self, strokeshape, bbox, center):
        """`gui.stategroup.State` entry callback for blinking a stroke"""
        overlay = lib.layer.SurfaceBackedLayer()
        overlay.load_from_strokeshape(strokeshape, bbox=bbox, center=center)
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
        if direction == self.PAN_LEFT:
            self.tdw.scroll(-step, 0, ongoing=False)
        elif direction == self.PAN_RIGHT:
            self.tdw.scroll(+step, 0, ongoing=False)
        elif direction == self.PAN_UP:
            self.tdw.scroll(0, -step, ongoing=False)
        elif direction == self.PAN_DOWN:
            self.tdw.scroll(0, +step, ongoing=False)
        else:
            raise TypeError('unsupported pan() direction=%s' % direction)
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
            raise TypeError('unsupported zoom() direction=%s' % direction)

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
        if center == self.CENTER_ON_POINTER:
            etime, ex, ey = self.get_last_event_info(self.tdw)
            center = (ex, ey)
        elif center == self.CENTER_ON_VIEWPORT:
            center = self.tdw.get_center()

        if direction == self.ROTATE_CLOCKWISE:
            step = +self.ROTATION_STEP
        elif direction == self.ROTATE_ANTICLOCKWISE:
            step = -self.ROTATION_STEP
        else:
            raise TypeError('unsupported direction=%s' % direction)
        self.tdw.rotate(
            step,
            center=center,
            ongoing=False,
        )
        self.notify_view_changed()

    def zoom_cb(self, action):
        """Callback for Zoom{In,Out} GtkActions"""
        direction = self.ZOOM_INWARDS
        if action.get_name() == 'ZoomOut':
            direction = self.ZOOM_OUTWARDS
        self.zoom(direction)

    def zoom_centered_cb(self, action):
        """Callback for Zoom{In,Out}Centered GtkActions"""
        direction = self.ZOOM_INWARDS
        if action.get_name() == 'ZoomOutCentered':
            direction = self.ZOOM_OUTWARDS
        self.zoom(direction, center=self.CENTER_ON_VIEWPORT)

    def pan_cb(self, action):
        """Callback for Pan{Left,Right,Up,Down} GtkActions"""
        direction = self.PAN_LEFT
        if action.get_name() == 'PanRight':
            direction = self.PAN_RIGHT
        elif action.get_name() == 'PanUp':
            direction = self.PAN_UP
        elif action.get_name() == 'PanDown':
            direction = self.PAN_DOWN
        self.pan(direction)

    def rotate_cb(self, action):
        """Callback for Rotate{Left,Right} GtkActions"""
        direction = self.ROTATE_CLOCKWISE
        if action.get_name() == 'RotateRight':
            direction = self.ROTATE_ANTICLOCKWISE
        self.rotate(direction)

    def rotate_centered_cb(self, action, *test):
        """Callback for Rotate{Left,Right}Centered GtkActions"""
        direction = self.ROTATE_CLOCKWISE
        if action.get_name() == 'RotateRightCentered':
            direction = self.ROTATE_ANTICLOCKWISE
        self.rotate(direction, center=self.CENTER_ON_VIEWPORT)

    ## Symmetry

    def symmetry_active_toggled_cb(self, action):
        """Handle changes to the SymmetryActive toggle"""
        stack = self.model.layer_stack
        center = None
        want_active = bool(action.get_active())

        # When going from an unset state to an active state, the symmetry
        # center (model coordinates) is set based on the center of the viewport
        if stack.symmetry_unset and want_active:
            stack.symmetry_unset = False
            alloc = self.tdw.get_allocation()
            dx, dy = alloc.width / 2.0, alloc.height / 2.0
            center = self.tdw.display_to_model(dx, dy)

        already_active = stack.symmetry_active
        if want_active != already_active or center is not None:
            stack.set_symmetry_state(want_active, center=center)

    def _symmetry_state_changed_cb(
            self, stack, active, center, sym_type, sym_lines, sym_angle):
        """Update the SymmetryActive toggle on model state changes"""
        if active is not None:
            symm_toggle = self.action_group.get_action("SymmetryActive")
            symm_toggle_active = bool(symm_toggle.get_active())
            model_symm_active = bool(active)
            if symm_toggle_active != model_symm_active:
                symm_toggle.set_active(model_symm_active)

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
        """Programmatically resets the view to the defaults.

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
        """Programmatically fits the view to the document"""
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
        # bbox since it is centered and symmetrical, but take the max
        # value since during rotation one corner's distance along the
        # x axis shortens while the other lengthens. Same for the y axis.
        x = max(abs(wcos - hsin), abs(wcos + hsin))
        y = max(abs(wsin + hcos), abs(wsin - hcos))
        # Compare the doc and window dimensions and take the best fit
        zoom = min((w1 - 20) / x, (h1 - 20) / y)
        # Reapply all transformations
        self.tdw.recenter_document()  # Center image
        self.tdw.set_rotation(radians)  # reapply canvas rotation
        self.tdw.set_mirrored(mirror)  # reapply mirror
        self.tdw.set_zoom(zoom)  # Set new zoom level
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
                GLib.source_remove(self._view_changed_notification_srcid)
                self._view_changed_notification_srcid = None
            self._view_changed_notification_idle_cb()
            return
        if self._view_changed_notification_srcid:
            return
        cb = self._view_changed_notification_idle_cb
        priority = GLib.PRIORITY_LOW
        if prioritize:
            priority = GLib.PRIORITY_HIGH_IDLE
        srcid = GLib.idle_add(cb, priority=priority)
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

    def vacuum_document_cb(self, action):
        """Discards empty (all-zeros) tiles."""
        r, t = self.model.layer_stack.remove_empty_tiles()
        self.app.show_transient_message(C_(
            "Statusbar message: vacuum document",
            u"Vacuum: discarded {removed} of {total} tiles.",
        ).format(
            removed=r,
            total=t,
        ))

    ## Model state reflection

    def _input_stroke_ended_cb(self, self_again, event):
        """Invoked after a pen-down, draw, pen-up 'input stroke'"""
        # Store device-specific brush settings at the end of the stroke,
        # not when the device changes because the user can change brush
        # radii etc. in the middle of a stroke, and because
        # device_changed_cb won't respond when the user fiddles with
        # colors, opacity and sizes via the dialogs.
        device_name = self.app.preferences.get('devbrush.last_used', None)
        if device_name is None:
            return
        bm = self.app.brushmanager
        selected_brush = bm.clone_selected_brush(name=None)  # for saving
        bm.store_brush_for_device(device_name, selected_brush)
        # However it may be better to reflect any brush settings change
        # into the last-used devbrush immediately. The UI idea here is
        # that the pointer (when you're holding the pen) is special,
        # it's the point of a real-world tool that you're dipping into a
        # palette, or modifying using the sliders.

    ## Mode flipping

    def mode_flip_action_activated_cb(self, flip_action):
        """Callback: a mode "flip" action was activated.

        :param flip_action: the Gtk.Action which was activated

        Mode classes are looked up via `gui.mode.ModeRegistry` based
        on the name of the action: flip actions are named after the
        RadioActions they nominally control, with "Flip" prepended.
        Activating a FlipAction has the effect of flipping a mode off if
        it is currently active, or on if another mode is active. Mode
        flip actions are the usual actions bound to keypresses since
        being able to toggle off with the same key is useful.

        Because these modes are intended for keyboard activation, they
        are instructed to ignore the initial keyboard modifier state
        when entered.  See also: `gui.mode.DragMode`.

        """
        # If this is not the active document, dispatch the action to it.
        active_doc = Document.get_active_instance()
        if (active_doc is not None) and (active_doc is not self):
            return active_doc.mode_flip_action_activated_cb(flip_action)

        flip_action_name = flip_action.get_name()
        assert flip_action_name.startswith("Flip")
        # Find the corresponding Gtk.RadioAction
        action_name = flip_action_name.replace("Flip", "", 1)
        mode_class = gui.mode.ModeRegistry.get_mode_class(action_name)
        if mode_class is None:
            warn("%r not registered: check imports" % (action_name,), Warning)
            return

        # If a mode object of this exact class is active, pop the stack.
        # Otherwise, instantiate and enter.
        if self.modes.top.__class__ is mode_class:
            self.modes.pop()
            flip_action.keyup_callback = lambda *a: None  # suppress repeats
        else:
            if issubclass(mode_class, gui.mode.OneshotDragMode):
                mode = mode_class(
                    ignore_modifiers=True,
                    temporary_activation=False,
                )
            else:
                mode = mode_class(ignore_modifiers=True)
            if flip_action.keydown:
                flip_action.__pressed = True
                # Change what happens on a key-up after a short while.
                # Distinguishes long presses from short.
                timeout = getattr(mode, "keyup_timeout", 500)
                cb = self._modeflip_change_keyup_callback
                ev = Gtk.get_current_event()
                if ev is not None:
                    ev = ev.copy()
                if timeout > 0:
                    # Queue a change of key-up callback after the timeout
                    GLib.timeout_add(timeout, cb, mode, flip_action, ev)

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
        # if hasattr(mode, '_start_drag'):
        #    mode._start_drag(mode.doc.tdw, ev)
        return False

    ## Mode stack reflection

    def mode_radioaction_changed_cb(self, action, current_action):
        """Callback: radio action controlling the modes stack activated

        :param action: the lead Gtk.RadioAction
        :param current_action: the newly active Gtk.RadioAction

        Mode classes are looked up via `gui.mode.ModeRegistry` based
        on the name of the action. This action instantiates the mode and
        pushes it onto the mode stack unless the active mode is already
        an instance of the mode class.

        """
        # Update the mode stack so that its top element matches the
        # newly chosen action.
        action_name = current_action.get_name()
        mode_class = gui.mode.ModeRegistry.get_mode_class(action_name)
        if mode_class is None:
            warn("%r not registered: check imports" % (action_name,), Warning)
            return

        if self.modes.top.__class__ is not mode_class:
            if issubclass(mode_class, gui.mode.OneshotDragMode):
                mode = mode_class(temporary_activation=False)
            else:
                mode = mode_class()
            self.modes.context_push(mode)

    def _modestack_changed_cb(self, modestack, old, new):
        """Callback: make actions follow changes to the mode stack"""
        # Activate the action corresponding to the current top mode.
        logger.debug("Mode changed: %r", self.modes)
        self._top_mode = new
        action_name = new.ACTION_NAME
        if not action_name:
            return None
        action = self.app.builder.get_object(action_name)
        if action is not None:
            # Not every mode has a corresponding action
            if not action.get_active():
                action.set_active(True)

    ## External layer editing support

    def begin_external_layer_edit_cb(self, action):
        """Callback: edit the current layer in an external app"""
        self._begin_external_layer_edit()

    def _begin_external_layer_edit(self):
        layer = self.model.layer_stack.current
        self._layer_edit_manager.begin(layer)

    def commit_external_layer_edit_cb(self, action):
        """Callback: Commit the current layer's ongoing external edit

        Exposed as an extra action just in case automatic monitoring
        fails on a particular platform. Normally the manager commits
        saved changes automatically.

        """
        layer = self.model.layer_stack.current
        self._layer_edit_manager.commit(layer)

    def _update_external_layer_edit_actions(self, *_ignored):
        """Update the External Layer Edit actions' sensitivities"""
        app = self.app
        rootstack = self.model.layer_stack
        current = rootstack.current
        can_commit = hasattr(current, "load_from_external_edit_tempfile")
        app.find_action("BeginExternalLayerEdit").set_sensitive(can_commit)
        app.find_action("CommitExternalLayerEdit").set_sensitive(can_commit)

    ## Layer views, and their locked flag

    def _update_layer_visible_toggle_from_current_view(self, *_ignored):
        app = self.app
        lvm = self.model.layer_view_manager
        view_locked = lvm.current_view_locked
        app.find_action("LayerVisibleToggle").set_sensitive(not view_locked)

    ## Inking tool node manipulation

    def insert_current_node_cb(self, action):
        """Insert a node before the currently selected node (keyboard)"""
        mode = self.modes.top
        if getattr(mode, 'insert_current_node', False):
            mode.insert_current_node()

    def delete_current_node_cb(self, action):
        """Delete the currently selected node (from keyboard)"""
        mode = self.modes.top
        if getattr(mode, 'delete_current_node', False):
            mode.delete_current_node()

    def simplify_nodes_cb(self, action):
        """Simplify the current inktool stroke (from keyboard)"""
        mode = self.modes.top
        if getattr(mode, 'simplify_nodes', False):
            mode.simplify_nodes()

    def cull_nodes_cb(self, action):
        """Callback: cull current inktool nodes (from keyboard)"""
        mode = self.modes.top
        if getattr(mode, 'cull_nodes', False):
            mode.cull_nodes()
