# This file is part of MyPaint.
# Copyright (C) 2008-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Canvas input event handling.
"""

import gtk2compat
from buttonmap import get_handler_object

import math
from numpy import isfinite

import gobject
import gtk
from gtk import gdk
from gtk import keysyms


# Actions it makes sense to bind to a button.
# Notably, tablet pads tend to offer many more buttons than the usual 3...

extra_actions = ["ShowPopupMenu",
                 "Undo", "Redo",
                 "Bigger", "Smaller",
                 "MoreOpaque", "LessOpaque",
                 "PickContext",
                 "Fullscreen",
                 "ToggleSubwindows",
                 "BrushChooserPopup",
                 "ColorRingPopup",
                 "ColorDetailsDialog",
                 "ColorChangerWashPopup",
                 "ColorChangerCrossedBowlPopup",
                 "ColorHistoryPopup",
                 "PalettePrev",
                 "PaletteNext",
                 ]


class ModeRegistry (type):
    """Lookup table for interaction modes and their associated actions

    Operates as the metaclass for `InteractionMode`, so all you need to do to
    create the association for a mode subclass is to define an
    ``__action_name__`` entry in the class's namespace containing the name of
    the associated `gtk.Action` defined in ``mypaint.xml``.

    """

    action_name_to_mode_class = {}


    # (Special-cased @staticmethod)
    def __new__(cls, name, bases, dict):
        """Creates and records a new (InteractionMode) class.

        :param cls: this metaclass
        :param name: name of the class under construction
        :param bases: immediate base classes of the class under construction
        :param dict: class dict for the class under construction
        :rtype: the constructed class, a regular InteractionMode class object

        If it exists, the ``__action_name__`` entry in `dict` is recorded,
        and can be used as a key for lookup of the returned class via the
        ``@classmethod``s defined on `ModeRegistry`.

        """
        action_name = dict.get("__action_name__", None)
        mode_class = super(ModeRegistry, cls).__new__(cls, name, bases, dict)
        if action_name is not None:
            action_name = str(action_name)
            cls.action_name_to_mode_class[action_name] = mode_class
        return mode_class


    @classmethod
    def get_mode_class(cls, action_name):
        """Looks up a registered mode class by its associated action's name.

        :param action_name: a string containing an action name (see this
           metaclass's docs regarding the ``__action_name__`` class variable)
        :rtype: an InteractionMode class object, or `None`.

        """
        return cls.action_name_to_mode_class.get(action_name, None)


    @classmethod
    def get_action_names(cls):
        """Returns all action names associated with interaction.

        :rtype: an iterable of action name strings.

        """
        return cls.action_name_to_mode_class.keys()


class InteractionMode (object):
    """Required base class for temporary interaction modes.

    Active interaction mode objects process input events, and can manipulate
    document views (TiledDrawWidget), the document model data (lib.document),
    and the mode stack they sit on. Interactions encapsulate state about their
    particular kind of interaction; for example a drag interaction typically
    contains the starting position for the drag.

    Event handler methods can create new sub-modes and push them to the stack.
    It is conventional to pass the current event to the equivalent method on
    the new object when this transfer of control happens.

    Subclasses may nominate a related `GtkAction` instance in the UI by setting
    the class-level variable ``__action_name__``: this should be the name of an
    action defined in `gui.app.Application.builder`'s XML file.

    """


    ## Class configuration

    #: All InteractionMode subclasses register themselves.
    __metaclass__ = ModeRegistry

    #: See the docs for `gui.canvasevent.ModeRegistry`.
    __action_name__ = None

    is_live_updateable = False # CHECK: what's this for?

    #: Timeout for Document.mode_flip_action_activated_cb(). How long, in
    #: milliseconds, it takes for the controller to change the key-up action
    #: when activated with a keyboard "Flip<ModeName>" action. Set to zero
    #: for modes where key-up should exit the mode at any time, and to a larger
    #: number for modes where the behaviour changes.
    keyup_timeout = 500

    ## Defaults for instances (sue me, I'm lazy)

    #: The `gui.document.Document` this mode affects: see enter()
    doc = None


    def stackable_on(self, mode):
        """Tests whether the mode can usefully stack onto an active mode.

        :param mode: another mode object
        :rtype: bool

        This method should return True if this mode can usefully be stacked
        onto another mode when switching via toolbars buttons or other actions.

        """
        return False


    def enter(self, doc):
        """Enters the mode: called by `ModeStack.push()` etc.

        :param doc: the `gui.document.Document` this mode should affect.
            A reference is kept in `self.doc`.

        This is called when the mode becomes active, i.e. when it becomes the
        top mode on a ModeStack, and before input is sent to it. Note that a
        mode may be entered only to be left immediately: mode stacks are
        cleared by repeated pop()ing.

        """
        self.doc = doc
        assert not hasattr(super(InteractionMode, self), "enter")


    def leave(self):
        """Leaves the mode: called by `ModeStack.pop()` etc.

        This is called when an active mode becomes inactive, i.e. when it is
        no longer the top mode on its ModeStack.

        """
        self.doc = None
        assert not hasattr(super(InteractionMode, self), "leave")


    def button_press_cb(self, tdw, event):
        """Handler for ``button-press-event``s."""
        assert not hasattr(super(InteractionMode, self), "button_press_cb")


    def motion_notify_cb(self, tdw, event):
        """Handler for ``motion-notify-event``s."""
        assert not hasattr(super(InteractionMode, self), "motion_notify_cb")


    def button_release_cb(self, tdw, event):
        """Handler for ``button-release-event``s."""
        assert not hasattr(super(InteractionMode, self), "button_release_cb")


    def scroll_cb(self, tdw, event):
        """Handler for ``scroll-event``s.
        """
        assert not hasattr(super(InteractionMode, self), "scroll_cb")


    def key_press_cb(self, win, tdw, event):
        """Handler for ``key-press-event``s.

        The base class implementation does nothing.
        Keypresses are received by the main window only, but at this point it
        has applied some heuristics to determine the active doc and view.
        These are passed through to the active mode and are accessible to
        keypress handlers via `self.doc` and the `tdw` argument.

        """
        assert not hasattr(super(InteractionMode, self), "key_press_cb")
        return True


    def key_release_cb(self, win, tdw, event):
        """Handler for ``key-release-event``s.

        The base class implementation does nothing. See `key_press_cb` for
        details of the additional arguments.

        """
        assert not hasattr(super(InteractionMode, self), "key_release_cb")
        return True


    def model_structure_changed_cb(self, doc):
        """Handler for model structural changes.

        Only called for the top mode on the stack, when the document model
        structure changes, so if a mode depends on the model structure, check
        in enter() too.

        """
        assert not hasattr(super(InteractionMode, self),
                           "model_structure_changed_cb")


    ## Drag sub-API (FIXME: this is in the wrong place)
    # Defined here to allow mixins to provide behaviour for both both drags and
    # regular events without having to derive from DragMode. Really these
    # buck-stops-here definitions belong in DragMode, so consider moving them
    # somewhere more sensible.

    def drag_start_cb(self, tdw, event):
        assert not hasattr(super(InteractionMode, self), "drag_start_cb")

    def drag_update_cb(self, tdw, event, dx, dy):
        assert not hasattr(super(InteractionMode, self), "drag_update_cb")

    def drag_stop_cb(self):
        assert not hasattr(super(InteractionMode, self), "drag_stop_cb")


    ## Internal utility functions

    def current_modifiers(self):
        """Returns the current set of modifier keys as a Gdk bitmask.

        For use in handlers for keypress events when the key in question is
        itself a modifier, handlers of multiple types of event, and when the
        triggering event isn't available. Pointer button event handling should
        use ``event.state & gtk.accelerator_get_default_mod_mask()``.

        """
        if gtk2compat.USE_GTK3:
            display = gdk.Display.get_default()
        else:
            display = gdk.display_get_default()
        screen, x, y, modifiers = display.get_pointer()
        modifiers &= gtk.accelerator_get_default_mod_mask()
        return modifiers



class ScrollableModeMixin (InteractionMode):
    """Mixin for scrollable modes.

    Implements some immediate rotation and zoom commands for the scroll wheel.
    These should be useful in many modes, but perhaps not all.

    """

    def scroll_cb(self, tdw, event):
        """Handles scroll-wheel events.

        Normal scroll wheel events: zoom.
        Shift+scroll, or left/right scroll: rotation.

        """
        doc = self.doc
        d = event.direction
        if d == gdk.SCROLL_UP:
            if event.state & gdk.SHIFT_MASK:
                doc.rotate(doc.ROTATE_CLOCKWISE)
            else:
                doc.zoom(doc.ZOOM_INWARDS)
        elif d == gdk.SCROLL_DOWN:
            if event.state & gdk.SHIFT_MASK:
                doc.rotate(doc.ROTATE_ANTICLOCKWISE)
            else:
                doc.zoom(doc.ZOOM_OUTWARDS)
        elif d == gdk.SCROLL_RIGHT:
            doc.rotate(doc.ROTATE_ANTICLOCKWISE)
        elif d == gdk.SCROLL_LEFT:
            doc.rotate(doc.ROTATE_CLOCKWISE)
        else:
            super(ScrollableModeMixin, self).scroll_cb(tdw, event)
        return True


class FreehandOnlyMode (InteractionMode):
    """A freehand-only drawing mode, which cannot be switched with modifiers.

    This mode can be used with the basic CanvasController, and in the absence
    of the main application.

    """

    is_live_updateable = True


    def enter(self, **kwds):
        super(FreehandOnlyMode, self).enter(**kwds)
        self.reset_drawing_state()


    def leave(self, **kwds):
        self.reset_drawing_state()
        super(FreehandOnlyMode, self).leave(**kwds)


    def reset_drawing_state(self):
        self.last_event_had_pressure_info = False
        # Windows stuff
        self.motions = []


    def button_press_cb(self, tdw, event):
        result = False
        if event.button == 1 and event.type == gdk.BUTTON_PRESS:
            # Single button press
            # Stroke started, notify observers
            try:
                observers = self.doc.input_stroke_started_observers
            except AttributeError:
                pass
            else:
                for func in observers:
                    func(event)
            # Mouse button pressed (while painting without pressure
            # information)
            if not self.last_event_had_pressure_info:
                # For the mouse we don't get a motion event for "pressure"
                # changes, so we simulate it. (Note: we can't use the
                # event's button state because it carries the old state.)
                self.motion_notify_cb(tdw, event, button1_pressed=True)
            result = True

        # Collaborate, but likely with nothing
        result |= bool(super(FreehandOnlyMode, self).button_press_cb(tdw, event))
        return result


    def button_release_cb(self, tdw, event):
        # (see comment above in button_press_cb)
        result = False
        if event.button == 1:
            if not self.last_event_had_pressure_info:
                self.motion_notify_cb(tdw, event, button1_pressed=False)
            # Notify observers after processing the event
            try:
                observers = self.doc.input_stroke_ended_observers
            except AttributeError:
                pass
            else:
                for func in observers:
                    func(event)
            result = True
        result |= bool(super(FreehandOnlyMode, self).button_release_cb(tdw, event))
        return result


    def motion_notify_cb(self, tdw, event, button1_pressed=None):
        # Purely responsible for drawing.
        if not tdw.is_sensitive:
            return super(FreehandOnlyMode, self).motion_notify_cb(tdw, event)

        model = tdw.doc
        app = tdw.app

        last_event_time, last_x, last_y = self.doc.get_last_event_info(tdw)
        if last_event_time:
            dtime = (event.time - last_event_time)/1000.0
            dx = event.x - last_x
            dy = event.y - last_y
        else:
            dtime = None
        if dtime is None:
            return super(FreehandOnlyMode, self).motion_notify_cb(tdw, event)

        same_device = True
        device = None
        if app is not None:
            if gtk2compat.USE_GTK3:
                device = event.get_source_device()
            else:
                device = event.device
            same_device = app.device_monitor.device_used(device)

        # Refuse drawing if the layer is locked or hidden
        if model.layer.locked or not model.layer.visible:
            return super(FreehandOnlyMode, self).motion_notify_cb(tdw, event)
            # TODO: some feedback, maybe

        x, y = tdw.display_to_model(event.x, event.y)

        pressure = event.get_axis(gdk.AXIS_PRESSURE)

        if pressure is not None and (   pressure > 1.0
                                     or pressure < 0.0
                                     or not isfinite(pressure)):
            if not hasattr(self, 'bad_devices'):
                self.bad_devices = []
            if device.name not in self.bad_devices:
                print 'WARNING: device "%s" is reporting bad pressure %+f' \
                    % (device.name, pressure)
                self.bad_devices.append(device.name)
            if not isfinite(pressure):
                # infinity/nan: use button state (instead of clamping in
                # brush.hpp) https://gna.org/bugs/?14709
                pressure = None

        # Fake pressure if we have none based on the extra argument passed if
        # this is a fake.
        if pressure is None:
            self.last_event_had_pressure_info = False
            if button1_pressed is None:
                button1_pressed = event.state & gdk.BUTTON1_MASK
            if button1_pressed:
                pressure = 0.5
            else:
                pressure = 0.0
        else:
            self.last_event_had_pressure_info = True

        xtilt = event.get_axis(gdk.AXIS_XTILT)
        ytilt = event.get_axis(gdk.AXIS_YTILT)
        # Check whether tilt is present.  For some tablets without
        # tilt support GTK reports a tilt axis with value nan, instead
        # of None.  https://gna.org/bugs/?17084
        if xtilt is None or ytilt is None or not isfinite(xtilt+ytilt):
            xtilt = 0.0
            ytilt = 0.0

        # Tilt inputs are assumed to be relative to the viewport, but the
        # canvas may be rotated or mirrored, or both. Compensate before
        # passing them to the brush engine. https://gna.org/bugs/?19988
        if not (xtilt == 0 and ytilt == 0):
            if tdw.mirrored:
                xtilt *= -1.0
            if tdw.rotation != 0:
                tilt_angle = math.atan2(ytilt, xtilt) - tdw.rotation
                tilt_magnitude = math.sqrt((xtilt**2) + (ytilt**2))
                xtilt = tilt_magnitude * math.cos(tilt_angle)
                ytilt = tilt_magnitude * math.sin(tilt_angle)

        if event.state & gdk.CONTROL_MASK or event.state & gdk.MOD1_MASK:
            # HACK: color picking, do not paint
            # Don't simply return; this is a workaround for unwanted lines
            # in https://gna.org/bugs/?16169
            pressure = 0.0

        if app is not None and app.pressure_mapping:
            pressure = app.pressure_mapping(pressure)

        if event.state & gdk.SHIFT_MASK:
            pressure = 0.0

        if pressure:
            tdw.set_last_painting_pos((x, y))

        # If the device has changed and the last pressure value from the
        # previous device is not equal to 0.0, this can leave a visible stroke
        # on the layer even if the 'new' device is not pressed on the tablet
        # and has a pressure axis == 0.0.  Reseting the brush when the device
        # changes fixes this issue, but there may be a much more elegant
        # solution that only resets the brush on this edge-case.

        if not same_device:
            model.brush.reset()

        # On Windows, GTK timestamps have a resolution around
        # 15ms, but tablet events arrive every 8ms.
        # https://gna.org/bugs/index.php?16569
        # TODO: proper fix in the brush engine, using only smooth,
        #       filtered speed inputs, will make this unneccessary
        if dtime < 0.0:
            print 'Time is running backwards, dtime=%f' % dtime
            dtime = 0.0
        data = (x, y, pressure, xtilt, ytilt)
        if dtime == 0.0:
            self.motions.append(data)
        elif dtime > 0.0:
            if self.motions:
                # replay previous events that had identical timestamp
                if dtime > 0.1:
                    # really old events, don't associate them with the new one
                    step = 0.1
                else:
                    step = dtime
                step /= len(self.motions)+1
                for data_old in self.motions:
                    model.stroke_to(step, *data_old)
                    dtime -= step
                self.motions = []
            model.stroke_to(dtime, *data)

        super(FreehandOnlyMode, self).motion_notify_cb(tdw, event)
        return True


class SwitchableModeMixin (InteractionMode):
    """Adds functionality for performing actions via modifiers & ptr buttons

    Mode switching happens in response to button- or key-press events, using
    the main app's ``button_mapping`` to look up action names. These actions
    can switch control to other modes by pushing them onto the mode stack;
    they can invoke popup States; or they can trigger regular GtkActions.

    Not every switchable mode can perform any such action. Subclasses should
    name actions they can invoke in ``permitted_switch_actions``. If this set
    is left empty, any action can be performed

    """

    permitted_switch_actions = set()  #: Optional whitelist for mode switching


    def button_press_cb(self, tdw, event):
        """Button-press event handler. Permits switching."""

        # Never switch in the middle of an active drag (see DragMode)
        if getattr(self, 'in_drag', False):
            return super(SwitchableModeMixin, self).button_press_cb(tdw, event)

        # Ignore accidental presses
        if event.type != gdk.BUTTON_PRESS:
            # Single button-presses only, not 2ble/3ple
            return super(SwitchableModeMixin, self).button_press_cb(tdw, event)
        if event.button != 1:
            # check whether we are painting (accidental)
            if event.state & gdk.BUTTON1_MASK:
                # Do not allow mode switching in the middle of
                # painting. This often happens by accident with wacom
                # tablet's stylus button.
                #
                # However we allow dragging if the user's pressure is
                # still below the click threshold.  This is because
                # some tablet PCs are not able to produce a
                # middle-mouse click without reporting pressure.
                # https://gna.org/bugs/index.php?15907
                return super(SwitchableModeMixin,
                             self).button_press_cb(tdw, event)

        # Look up action
        btn_map = self.doc.app.button_mapping
        modifiers = event.state & gtk.accelerator_get_default_mod_mask()
        action_name = btn_map.lookup(modifiers, event.button)

        # Forbid actions not named in the whitelist, if it's defined
        if len(self.permitted_switch_actions) > 0:
            if action_name not in self.permitted_switch_actions:
                action_name = None

        # Perform allowed action if one was looked up
        if action_name is not None:
            return self._dispatch_named_action(None, tdw, event, action_name)

        # Otherwise fall through to the next behaviour
        return super(SwitchableModeMixin, self).button_press_cb(tdw, event)


    def key_press_cb(self, win, tdw, event):
        """Keypress event handler. Permits switching."""

        # Never switch in the middle of an active drag (see DragMode)
        if getattr(self, 'in_drag', False):
            return super(SwitchableModeMixin,self).key_press_cb(win,tdw,event)

        # Naively pick an action based on the button map
        btn_map = self.doc.app.button_mapping
        action_name = None
        if event.is_modifier:
            # If the keypress is a modifier only, determine the modifier mask a
            # subsequent Button1 press event would get. This is used for early
            # spring-loaded mode switching.
            mods = self.current_modifiers()
            action_name = btn_map.get_unique_action_for_modifiers(mods)
            # Only mode-based immediate dispatch is allowed, however.
            # Might relax this later.
            if action_name is not None:
                if not action_name.endswith("Mode"):
                    action_name = None
        else:
            # Strategy 2: pretend that the space bar is really button 2.
            if event.keyval == keysyms.space:
                mods = event.state & gtk.accelerator_get_default_mod_mask()
                action_name = btn_map.lookup(mods, 2)

        # Forbid actions not named in the whitelist, if it's defined
        if len(self.permitted_switch_actions) > 0:
            if action_name not in self.permitted_switch_actions:
                action_name = None

        # If we found something to do, dispatch
        if action_name is not None:
            return self._dispatch_named_action(win, tdw, event, action_name)

        # Otherwise, fall through to the next behaviour
        return super(SwitchableModeMixin, self).key_press_cb(win, tdw, event)


    def _dispatch_named_action(self, win, tdw, event, action_name):
        # Send a named action from the button map to some handler code
        app = tdw.app
        drawwindow = app.drawWindow
        tdw.doc.split_stroke()
        if action_name == 'ShowPopupMenu':
            # Unfortunately still a special case.
            # Just firing the action doesn't work well with pads which fire a
            # button-release event immediately after the button-press.
            # Name it after the action however, in case we find a fix.
            drawwindow.show_popupmenu(event=event)
            return True
        handler_type, handler = get_handler_object(app, action_name)
        if handler_type == 'mode_class':
            # Transfer control to another mode temporarily.
            assert issubclass(handler, SpringLoadedModeMixin)
            mode = handler()
            self.doc.modes.push(mode)
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
            gobject.idle_add(handler.activate)
            return True
        else:
            return False


class SwitchableFreehandMode (SwitchableModeMixin, ScrollableModeMixin,
                              FreehandOnlyMode):
    """The default mode: freehand drawing, accepting modifiers to switch modes.
    """

    __action_name__ = 'SwitchableFreehandMode'
    permitted_switch_actions = set()   # Any action is permitted

    def __init__(self, ignore_modifiers=True, **args):
        # Ignore the additional arg that flip actions feed us
        super(SwitchableFreehandMode, self).__init__(**args)



class ModeStack (object):
    """A stack of InteractionModes. The top of the stack is the active mode.

    The stack can never be empty: if the final element is popped, it will be
    replaced with a new instance of its `default_mode_class`.

    """

    #: Class to instantiate if stack is empty: callable with 0 args.
    default_mode_class = SwitchableFreehandMode


    def __init__(self, doc):
        """Initialize, associated with a particular CanvasController (doc)

        :param doc: Controller instance: the main MyPaint app uses
            an instance of `gui.document.Document`. Simpler drawing
            surfaces can use a basic CanvasController and a
            simpler `default_mode_class`.
        :type doc: CanvasController
        """
        object.__init__(self)
        self._stack = []
        self._doc = doc
        self.observers = []


    def _notify_observers(self):
        top_mode = self._stack[-1]
        for func in self.observers:
            func(top_mode)


    @property
    def top(self):
        """The top node on the stack.
        """
        # Perhaps rename to "active()"?
        new_mode = self._check()
        if new_mode is not None:
            new_mode.enter(doc=self._doc)
            self._notify_observers()
        return self._stack[-1]


    def context_push(self, mode):
        """Context-aware push.

        :param mode: mode to be stacked and made active
        :type mode: `InteractionMode`

        Stacks a mode onto the topmost element in the stack it is compatible
        with, as determined by its ``stackable_on()`` method. Incompatible
        top modes are popped one by one until either a compatible mode is
        found, or the stack is emptied, then the new mode is pushed.

        """
        # Pop until the stack is empty, or the top mode is compatible
        while len(self._stack) > 0:
            if mode.stackable_on(self._stack[-1]):
                break
            self._stack.pop(-1).leave()
            if len(self._stack) > 0:
                self._stack[-1].enter(doc=self._doc)
        # Stack on top of any remaining compatible mode
        if len(self._stack) > 0:
            self._stack[-1].leave()
        self._doc.model.split_stroke()
        self._stack.append(mode)
        mode.enter(doc=self._doc)
        self._notify_observers()


    def pop(self):
        """Pops a mode, leaving the old top mode and entering the exposed top.
        """
        if len(self._stack) > 0:
            old_mode = self._stack.pop(-1)
            old_mode.leave()
        top_mode = self._check()
        if top_mode is None:
            top_mode = self._stack[-1]
        self._doc.model.split_stroke()
        top_mode.enter(doc=self._doc)
        self._notify_observers()


    def push(self, mode):
        """Pushes a mode, and enters it.
        """
        if len(self._stack) > 0:
            self._stack[-1].leave()
        self._doc.model.split_stroke()
        self._stack.append(mode)
        mode.enter(doc=self._doc)
        self._notify_observers()


    def reset(self, replacement=None):
        """Clears the stack, popping the final element and replacing it.

        :param replacement: Optional mode to go on top of the cleared stack.
        :type replacement: `InteractionMode`.

        """
        while len(self._stack) > 0:
            old_mode = self._stack.pop(-1)
            old_mode.leave()
            if len(self._stack) > 0:
                self._stack[-1].enter(doc=self._doc)
        top_mode = self._check(replacement)
        assert top_mode is not None
        self._notify_observers()


    def _check(self, replacement=None):
        # Ensures that the stack is non-empty.
        # Returns the new top mode if one was pushed.
        if len(self._stack) > 0:
            return None
        if replacement is not None:
            mode = replacement
        else:
            mode = self.default_mode_class()
        self._stack.append(mode)
        mode.enter(doc=self._doc)
        return mode


    def __repr__(self):
        s = '<ModeStack ['
        s += ", ".join([m.__class__.__name__ for m in self._stack])
        s += ']>'
        return s


    def __len__(self):
        return len(self._stack)


class SpringLoadedModeMixin (InteractionMode):
    """Behavioural add-ons for modes which last as long as modifiers are held.

    When a spring-loaded mode is first entered, it remembers which modifier
    keys were held down at that time. When keys are released, if the held
    modifiers are no longer held down, the mode stack is popped and the mode
    exits.

    """


    def __init__(self, ignore_modifiers=False, **kwds):
        """Construct, possibly ignoring initial modifiers.

        :param ignore_modifiers: If True, ignore the initial set of modifiers.

        Springloaded modes can be instructed to ignore the initial set of
        modifiers when they're entered. This is appropriate when the mode is
        being entered in response to a keyboard shortcut. Modifiers don't mean
        the same thing for keyboard shortcuts. Conversely, toolbar buttons and
        mode-switching via pointer buttons should use the default behaviour.

        In practice, it's not quite so clear cut. Instead we have keyboard-
        friendly "Flip*" actions (which allow the mode to be toggled off with a
        second press) that use the ``ignore_modifiers`` behaviour, and a
        secondary layer of radioactions which don't (but which reflect the
        state prettily).

        """
        super(SpringLoadedModeMixin, self).__init__(**kwds)
        self.ignore_modifiers = ignore_modifiers


    def enter(self, **kwds):
        """Enter the mode, recording the held modifier keys the first time.

        The attribute `self.initial_modifiers` is set the first time the mode
        is entered.

        """

        super(SpringLoadedModeMixin, self).enter(**kwds)
        assert self.doc is not None

        if self.ignore_modifiers:
            self.initial_modifiers = 0
            return

        old_modifiers = getattr(self, "initial_modifiers", None)
        if old_modifiers is not None:
            # Re-entering due to an overlying mode being popped from the stack.
            if old_modifiers != 0:
                # This mode started with modifiers held the first time round,
                modifiers = self.current_modifiers()
                if (modifiers & old_modifiers) == 0:
                    # But they're not held any more, so queue a further pop.
                    gobject.idle_add(self.__pop_modestack_idle_cb)
        else:
            # This mode is being entered for the first time; record modifiers
            modifiers = self.current_modifiers()
            self.initial_modifiers = self.current_modifiers()


    def __pop_modestack_idle_cb(self):
        # Pop the mode stack when this mode is re-entered but has to leave
        # straight away because its modifiers are no longer held. Doing it in
        # an idle function avoids confusing the derived class's enter() method:
        # a leave() during an enter() would be strange.
        if self.initial_modifiers is not None:
            self.doc.modes.pop()
        return False


    def key_release_cb(self, win, tdw, event):
        """Leave the mode if the initial modifier keys are no longer held.

        If the spring-loaded mode leaves because the modifiers keys held down
        when it was entered are no longer held, this method returns True, and
        so should the supercaller.

        """
        if self.initial_modifiers:
            modifiers = self.current_modifiers()
            if modifiers & self.initial_modifiers == 0:
                self.doc.modes.pop()
                return True
        return super(SpringLoadedModeMixin,self).key_release_cb(win,tdw,event)


class DragMode (InteractionMode):
    """Base class for drag activities.

    The drag can be entered when the pen is up or down: if the pen is down, the
    initial position will be determined from the first motion event.

    """

    inactive_cursor = gdk.Cursor(gdk.BOGOSITY)
    active_cursor = None

    def __init__(self, **kwds):
        super(DragMode, self).__init__(**kwds)
        self._grab_broken_conninfo = None
        self._reset_drag_state()


    def _reset_drag_state(self):
        self.last_x = None
        self.last_y = None
        self.start_x = None
        self.start_y = None
        self._start_keyval = None
        self._start_button = None
        self._grab_widget = None
        if self._grab_broken_conninfo is not None:
            tdw, connid = self._grab_broken_conninfo
            tdw.disconnect(connid)
            self._grab_broken_conninfo = None


    def _stop_drag(self, t=gdk.CURRENT_TIME):
        # Stops any active drag, calls drag_stop_cb(), and cleans up.
        if not self.in_drag:
            return
        tdw = self._grab_widget
        tdw.grab_remove()
        gdk.keyboard_ungrab(t)
        gdk.pointer_ungrab(t)
        self._grab_widget = None
        self.drag_stop_cb()
        self._reset_drag_state()


    def _start_drag(self, tdw, event):
        # Attempt to start a new drag, calling drag_start_cb() if successful.
        if self.in_drag:
            return
        if hasattr(event, "x"):
            self.start_x = event.x
            self.start_y = event.y
        else:
            #last_x, last_y = tdw.get_pointer()
            last_t, last_x, last_y = self.doc.get_last_event_info(tdw)
            self.start_x = last_x
            self.start_y = last_y
        tdw_window = tdw.get_window()
        event_mask = gdk.BUTTON_PRESS_MASK | gdk.BUTTON_RELEASE_MASK \
                   | gdk.POINTER_MOTION_MASK
        cursor = self.active_cursor
        if cursor is None:
            cursor = self.inactive_cursor

        # Grab the pointer
        grab_status = gdk.pointer_grab(tdw_window, False, event_mask, None,
                                       cursor, event.time)
        if grab_status != gdk.GRAB_SUCCESS:
            print "DEBUG: pointer grab failed:", grab_status
            print "DEBUG:  gdk_pointer_is_grabbed():", gdk.pointer_is_grabbed()
            # There seems to be a race condition between this grab under
            # PyGTK/GTK2 and some other grab - possibly just the implicit grabs
            # on colour selectors: https://gna.org/bugs/?20068 Only pointer
            # events are affected, and PyGI+GTK3 is unaffected.
            #
            # It's probably safest to exit the mode and not start the drag.
            # This condition should be rare enough for this to be a valid
            # approach: the irritation of having to click again to do something
            # should be far less than that of getting "stuck" in a drag.
            print "DEBUG: exiting mode"
            self.doc.modes.pop()

            # Sometimes a pointer ungrab is needed even though the grab
            # apparently failed to avoid the UI partially "locking up" with the
            # stylus (and only the stylus). Happens when WMs like Xfwm
            # intercept an <Alt>Button combination for window management
            # purposes. Results in gdk.GRAB_ALREADY_GRABBED, but this line is
            # necessary to avoid the rest of the UI becoming unresponsive even
            # though the canvas can be drawn on with the stylus. Are we
            # cancelling an implicit grab here, and why is it device specific?
            gdk.pointer_ungrab(event.time)
            return

        # We managed to establish a grab, so watch for it being broken.
        # This signal is disconnected when the mode leaves.
        connid = tdw.connect("grab-broken-event", self.tdw_grab_broken_cb)
        self._grab_broken_conninfo = (tdw, connid)

        # Grab the keyboard too, to be certain of getting the key release event
        # for a spacebar drag.
        grab_status = gdk.keyboard_grab(tdw_window, False, event.time)
        if grab_status != gdk.GRAB_SUCCESS:
            print "DEBUG: keyboard grab failed:", grab_status
            gdk.pointer_ungrab(event.time)
            self.doc.modes.pop()
            return

        # GTK too...
        tdw.grab_add()
        self._grab_widget = tdw

        # Drag has started, perform whatever action the mode needs.
        self.drag_start_cb(tdw, event)

        ## Break the grab after a while for debugging purposes
        #gobject.timeout_add_seconds(5, self.__break_own_grab_cb, tdw, False)


    def __break_own_grab_cb(self, tdw, fake=False):
        if fake:
            ev = gdk.Event(gdk.GRAB_BROKEN)
            ev.window = tdw.get_window()
            ev.send_event = True
            ev.put()
        else:
            import os
            os.system("wmctrl -s 0")
        return False


    def tdw_grab_broken_cb(self, tdw, event):
        # Cede control as cleanly as possible if something else grabs either
        # the keyboard or the pointer while a grab is active.
        # One possible cause for https://gna.org/bugs/?20333
        print "DEBUG: grab-broken-event on", tdw
        print "DEBUG:   send_event  :", event.send_event
        print "DEBUG:   keyboard    :", event.keyboard
        print "DEBUG:   implicit    :", event.implicit
        print "DEBUG:   grab_window :", event.grab_window
        print "DEBUG: exiting", self
        self.doc.modes.pop()
        return True


    @property
    def in_drag(self):
        return self._grab_widget is not None


    def enter(self, **kwds):
        super(DragMode, self).enter(**kwds)
        assert self.doc is not None
        self.doc.tdw.set_override_cursor(self.inactive_cursor)


    def leave(self, **kwds):
        self._stop_drag()
        if self.doc is not None:
            self.doc.tdw.set_override_cursor(None)
        super(DragMode, self).leave(**kwds)


    def button_press_cb(self, tdw, event):
        if event.type == gdk.BUTTON_PRESS:
            if self.in_drag:
                if self._start_button is None:
                    # Doing this allows single clicks to exit keyboard
                    # initiated drags, e.g. those forced when handling a
                    # keyboard event somewhere else.
                    self._start_button = event.button
            else:
                self._start_drag(tdw, event)
                if self.in_drag:
                    # Grab succeeded
                    self.last_x = event.x
                    self.last_y = event.y
                    self._start_button = event.button
        return super(DragMode, self).button_press_cb(tdw, event)


    def button_release_cb(self, tdw, event):
        if self.in_drag:
            if event.button == self._start_button:
                self._stop_drag()
        return super(DragMode, self).button_release_cb(tdw, event)


    def motion_notify_cb(self, tdw, event):
        # We might be here because an Action manipulated the modes stack
        # but if that's the case then we should wait for a button or
        # a keypress to initiate the drag.
        if self.in_drag:
            if self.last_x is not None:
                dx = event.x - self.last_x
                dy = event.y - self.last_y
                self.drag_update_cb(tdw, event, dx, dy)
            self.last_x = event.x
            self.last_y = event.y
            return True
        # Fall through to other behavioral mixins, just in case
        return super(DragMode, self).motion_notify_cb(tdw, event)


    def key_press_cb(self, win, tdw, event):
        if self.in_drag:
            # Eat keypresses in the middle of a drag no matter how
            # it was started.
            return True
        elif event.keyval == keysyms.space:
            # Start drags on space
            if event.keyval != self._start_keyval:
                self._start_keyval = event.keyval
                self._start_drag(tdw, event)
            return True
        # Fall through to other behavioral mixins
        return super(DragMode, self).key_press_cb(win, tdw, event)


    def key_release_cb(self, win, tdw, event):
        if self.in_drag:
            if event.keyval == self._start_keyval:
                self._stop_drag()
                self._start_keyval = None
            return True
        # Fall through to other behavioral mixins
        return super(DragMode, self).key_release_cb(win, tdw, event)


    def _force_drag_start(self):
        # Attempt to force a drag to start, using the current event.

        # XXX: This is only used by the picker mode, which needs to begin
        # picking straight away in enter even if this is in response to an
        # action activation.
        event = gtk.get_current_event()
        if event is None:
            print "no event"
            return
        if self.in_drag:
            return
        tdw = self.doc.tdw
        # Duck-profile the starting event. If it's a keypress event or a
        # button-press event, or anything that quacks like those, we can
        # attempt the grab and start the drag if it succeeded.
        if hasattr(event, "keyval"):
            if event.keyval != self._start_keyval:
                self._start_keyval = event.keyval
                self._start_drag(tdw, event)
        elif ( hasattr(event, "x") and hasattr(event, "y")
               and hasattr(event, "button") ):
            self._start_drag(tdw, event)
            if self.in_drag:
                # Grab succeeded
                self.last_x = event.x
                self.last_y = event.y
                # For the toolbar button, and for menus, it's a button-release
                # event.
                # Record which button is being pressed at start
                if type(event.button) == type(0):
                    # PyGTK supplies the actual button number ...
                    self._start_button = event.button
                else:
                    # ... but GI+GTK3 supplies a <void at 0xNNNNNNNN> object
                    # when the event comes from gtk.get_current_event() :(
                    self._start_button = None


class SpringLoadedDragMode (SpringLoadedModeMixin, DragMode):
    """Spring-loaded drag mode convenience base, with a key-release refinement

    If modifier keys were held when the mode was entered, a normal
    spring-loaded mode exits whenever those keys are all released. We don't
    want that to happen during drags however, so add this little refinement.

    """
    # XXX: refactor: could this just be merged into SpringLoadedModeMixin?

    def key_release_cb(self, win, tdw, event):
        if event.is_modifier and self.in_drag:
            return False
        return super(SpringLoadedDragMode, self).key_release_cb(win,tdw,event)


class OneshotDragModeMixin (InteractionMode):
    """Drag modes that can exit immediately when the drag stops.

    If SpringLoadedModeMixin is not also part of the mode object's class
    hierarchy, it will always exit at the end of a drag.

    If the mode object does inherit SpringLoadedModeMixin behaviour, what
    happens at the end of a drag is controlled by a class variable setting.

    """

    unmodified_persist = False
    #: If true, and spring-loaded, stay active if no modifiers held initially.


    def drag_stop_cb(self):
        if not hasattr(self, "initial_modifiers"):
            # Always exit at the end of a drag if not spring-loaded.
            self.doc.modes.pop()
        elif self.initial_modifiers != 0:
            # If started with modifiers, keeping the modifiers held keeps
            # spring-loaded modes active. If not, exit the mode.
            if (self.initial_modifiers & self.current_modifiers()) == 0:
                self.doc.modes.pop()
        else:
            # No modifiers were held when this mode was entered.
            if not self.unmodified_persist:
                self.doc.modes.pop()
        return super(OneshotDragModeMixin, self).drag_stop_cb()


class PanViewMode (SpringLoadedDragMode, OneshotDragModeMixin):
    """A oneshot mode for translating the viewport by dragging."""

    __action_name__ = 'PanViewMode'

    @property
    def inactive_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
                self.__action_name__)
    @property
    def active_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
                self.__action_name__)

    def stackable_on(self, mode):
        return isinstance(mode, SwitchableModeMixin)

    def drag_update_cb(self, tdw, event, dx, dy):
        tdw.scroll(-dx, -dy)
        self.doc.notify_view_changed()
        super(PanViewMode, self).drag_update_cb(tdw, event, dx, dy)


class ZoomViewMode (SpringLoadedDragMode, OneshotDragModeMixin):
    """A oneshot mode for zooming the viewport by dragging."""

    __action_name__ = 'ZoomViewMode'

    @property
    def active_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
                self.__action_name__)
    @property
    def inactive_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
                self.__action_name__)

    def stackable_on(self, mode):
        return isinstance(mode, SwitchableModeMixin)

    def drag_update_cb(self, tdw, event, dx, dy):
        tdw.scroll(-dx, -dy)
        tdw.zoom(math.exp(dy/100.0), center=(event.x, event.y))
        # TODO: Let modifiers constrain the zoom amount to 
        #       the defined steps.
        self.doc.notify_view_changed()
        super(ZoomViewMode, self).drag_update_cb(tdw, event, dx, dy)


class RotateViewMode (SpringLoadedDragMode, OneshotDragModeMixin):
    """A oneshot mode for rotating the viewport by dragging."""

    __action_name__ = 'RotateViewMode'

    @property
    def active_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
                self.__action_name__)
    @property
    def inactive_cursor(self):
        return self.doc.app.cursors.get_action_cursor(
                self.__action_name__)

    def stackable_on(self, mode):
        return isinstance(mode, SwitchableModeMixin)

    def drag_update_cb(self, tdw, event, dx, dy):
        # calculate angular velocity from the rotation center
        x, y = event.x, event.y
        cx, cy = tdw.get_center()
        x, y = x-cx, y-cy
        phi2 = math.atan2(y, x)
        x, y = x-dx, y-dy
        phi1 = math.atan2(y, x)
        tdw.rotate(phi2-phi1, center=(cx, cy))
        self.doc.notify_view_changed()
        # TODO: Allow modifiers to constrain the transformation angle
        #       to 22.5 degree steps.
        super(RotateViewMode, self).drag_update_cb(tdw, event, dx, dy)



class LayerMoveMode (SwitchableModeMixin,
                     ScrollableModeMixin,
                     SpringLoadedDragMode):
    """Moving a layer interactively.

    MyPaint is tile-based, and tiles must align between layers, so moving
    layers involves copying data around. This is slow for very large layers, so
    the work is broken into chunks and processed in the idle phase of the GUI
    for greater responsivity.

    """

    __action_name__ = 'LayerMoveMode'


    @property
    def active_cursor(self):
        cursor_name = "cursor_hand_closed"
        if not self._move_possible:
            cursor_name = "cursor_forbidden_everywhere"
        return self.doc.app.cursors.get_action_cursor(
                self.__action_name__, cursor_name)

    @property
    def inactive_cursor(self):
        cursor_name = "cursor_hand_open"
        if not self._move_possible:
            cursor_name = "cursor_forbidden_everywhere"
        return self.doc.app.cursors.get_action_cursor(
                self.__action_name__, cursor_name)

    unmodified_persist = True
    permitted_switch_actions = set([
            'RotateViewMode', 'ZoomViewMode', 'PanViewMode',
        ] + extra_actions)


    def stackable_on(self, mode):
        # Any drawing mode
        import linemode
        return isinstance(mode, linemode.LineModeBase) \
            or isinstance(mode, SwitchableFreehandMode)


    def __init__(self, **kwds):
        super(LayerMoveMode, self).__init__(**kwds)
        self.model_x0 = None
        self.model_y0 = None
        self.final_model_dx = None
        self.final_model_dy = None
        self._drag_update_idler_srcid = None
        self.layer = None
        self.move = None
        self.final_modifiers = 0
        self._move_possible = False


    def enter(self, **kwds):
        super(LayerMoveMode, self).enter(**kwds)
        self.final_modifiers = self.initial_modifiers
        self._update_cursors()
        self.doc.tdw.set_override_cursor(self.inactive_cursor)


    def leave(self, **kwds):
        # Force any remaining moves to a completed state while we still
        # have a self.doc. It's not enough for _finalize_move_idler() alone
        # to do this due to a race condition with leave()
        # https://gna.org/bugs/?20397
        if self.move is not None:
            while self._finalize_move_idler():
                pass
        return super(LayerMoveMode, self).leave(**kwds)


    def model_structure_changed_cb(self, doc):
        super(LayerMoveMode, self).model_structure_changed_cb(doc)
        if self.move is not None:
            # Cursor update is deferred to the end of the drag
            return
        self._update_cursors()


    def _update_cursors(self):
        layer = self.doc.model.get_current_layer()
        self._move_possible = layer.visible and not layer.locked
        self.doc.tdw.set_override_cursor(self.inactive_cursor)


    def drag_start_cb(self, tdw, event):
        if self.layer is None:
            self.layer = self.doc.model.get_current_layer()
            model_x, model_y = tdw.display_to_model(self.start_x, self.start_y)
            self.model_x0 = model_x
            self.model_y0 = model_y
            self.drag_start_tdw = tdw
            self.move = None
        return super(LayerMoveMode, self).drag_start_cb(tdw, event)


    def drag_update_cb(self, tdw, event, dx, dy):
        assert self.layer is not None

        # Begin moving, if we're not already
        if self.move is None and self._move_possible:
            self.move = self.layer.get_move(self.model_x0, self.model_y0)

        # Update the active move 
        model_x, model_y = tdw.display_to_model(event.x, event.y)
        model_dx = model_x - self.model_x0
        model_dy = model_y - self.model_y0
        self.final_model_dx = model_dx
        self.final_model_dy = model_dy

        if self.move is not None:
            self.move.update(model_dx, model_dy)
            # Keep showing updates in the background for feedback.
            if self._drag_update_idler_srcid is None:
                idler = self._drag_update_idler
                self._drag_update_idler_srcid = gobject.idle_add(idler)

        return super(LayerMoveMode, self).drag_update_cb(tdw, event, dx, dy)


    def _drag_update_idler(self):
        # Process tile moves in chunks in a background idler
        if self.move is None:
            # Might have exited, in which case leave() will have cleaned up
            self._drag_update_idler_srcid = None
            return False
        # Terminate if asked
        if self._drag_update_idler_srcid is None:
            self.move.cleanup()
            return False
        # Process some tile moves, and carry on if there's more to do
        if self.move.process():
            return True
        # Nothing more to do for this move
        self.move.cleanup()
        self._drag_update_idler_srcid = None
        return False


    def drag_stop_cb(self):
        self._drag_update_idler_srcid = None   # ask it to finish
        if self.move is not None:
            # Arrange for the background work to be done, and look busy
            tdw = self.drag_start_tdw
            tdw.set_sensitive(False)
            tdw.set_override_cursor(gdk.Cursor(gdk.WATCH))
            self.final_modifiers = self.current_modifiers()
            gobject.idle_add(self._finalize_move_idler)
        else:
            # Still need cleanup for tracking state, cursors etc.
            self._drag_cleanup()

        return super(LayerMoveMode, self).drag_stop_cb()


    def _drag_cleanup(self):
        # Reset drag tracking state
        self.model_x0 = self.model_y0 = None
        self.drag_start_tdw = self.move = None
        self.final_model_dx = self.final_model_dy = None
        self.layer = None

        # Cursor setting:
        # Reset busy cursor after drag which performed a move,
        # catch doc structure changes that happen during a drag
        self._update_cursors()

        # Leave mode if started with modifiers held and the user had released
        # them all at the end of the drag.
        if self.initial_modifiers:
            if (self.final_modifiers & self.initial_modifiers) == 0:
                self.doc.modes.pop()
        else:
            self.doc.modes.pop()

    def _finalize_move_idler(self):
        # Finalize everything once the drag's finished.
        if self.move is None:
            # Something else cleaned up. That's fine; both mode-leave and
            # drag-stop can call this. Just exit gracefully.
            return False
        assert self.doc is not None

        # Keep processing until the move queue is done.
        if self.move.process():
            return True

        # Cleanup tile moves
        self.move.cleanup()
        tdw = self.drag_start_tdw
        dx = self.final_model_dx
        dy = self.final_model_dy

        # Arrange for the strokemap to be moved too;
        # this happens in its own background idler.
        for stroke in self.layer.strokes:
            stroke.translate(dx, dy)
            # Minor problem: huge strokemaps take a long time to move, and the
            # translate must be forced to completion before drawing or any
            # further layer moves. This can cause apparent hangs for no
            # reason later on. Perhaps it would be better to process them
            # fully in this hourglass-cursor phase after all?

        # Record move so it can be undone
        self.doc.model.record_layer_move(self.layer, dx, dy)

        # Restore sensitivity
        tdw.set_sensitive(True)

        # Post-drag cleanup: cursor etc.
        self._drag_cleanup()

        # All done, stop idle processing
        return False
