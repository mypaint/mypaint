# This file is part of MyPaint.
# Copyright (C) 2008-2014 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Canvas input modes API: stack, and base classes for modes."""


## Imports

import logging
logger = logging.getLogger(__name__)

import gtk2compat
import buttonmap
import lib.command
from lib.observable import event

import math

import gobject
import gtk
from gtk import gdk
from gtk import keysyms
from gettext import gettext as _


## Module constants

# Actions it makes sense to bind to a button. (HACK).
# Notably, tablet pads tend to offer many more buttons than the usual 3...

BUTTON_BINDING_ACTIONS = [
    "ShowPopupMenu",
    "Undo",
    "Redo",
    "Bigger",
    "Smaller",
    "MoreOpaque",
    "LessOpaque",
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


## Behaviour flags

class Behavior:
    """Broad classification of what a mode's handler methods do

    These flags are assigned to devices in `gui.device` to allow the
    user to limit what devices are allowed to do. Mode instances expose
    their behaviour by defining their pointer_behavior and
    scroll_behavior properties.

    """
    NONE = 0x00  #: the mode does not perform any action
    PAINT_FREEHAND = 0x01  #: paint freehand brushstrokes
    PAINT_CONSTRAINED = 0x02  #: non-freehand painting: lines, or filling
    EDIT_OBJECTS = 0x04  #: move and adjust objects on screen
    CHANGE_VIEW = 0x08  #: move the viewport around
    # Useful masks
    NON_PAINTING = EDIT_OBJECTS | CHANGE_VIEW
    ALL_PAINTING = PAINT_FREEHAND | PAINT_CONSTRAINED
    ALL = NON_PAINTING | ALL_PAINTING


## Metaclass for all modes

class ModeRegistry (type):
    """Lookup table for interaction modes and their associated actions

    Operates as the metaclass for `InteractionMode`, so all you need to do to
    create the association for a mode subclass is to define an
    ``ACTION_NAME`` entry in the class's namespace containing the name of
    the associated `gtk.Action` defined in ``mypaint.xml``.

    """

    action_name_to_mode_class = {}
    mode_classes = set()

    # (Special-cased @staticmethod)
    def __new__(cls, name, bases, dict):
        """Creates and records a new (InteractionMode) class.

        :param cls: this metaclass
        :param name: name of the class under construction
        :param bases: immediate base classes of the class under construction
        :param dict: class dict for the class under construction
        :rtype: the constructed class, a regular InteractionMode class object

        If it exists, the ``ACTION_NAME`` entry in `dict` is recorded,
        and can be used as a key for lookup of the returned class via the
        ``@classmethod``s defined on `ModeRegistry`.

        """
        action_name = dict.get("ACTION_NAME", None)
        mode_class = super(ModeRegistry, cls).__new__(cls, name, bases, dict)
        if action_name is not None:
            action_name = str(action_name)
            cls.action_name_to_mode_class[action_name] = mode_class
        cls.mode_classes.add(mode_class)
        return mode_class

    @classmethod
    def get_mode_class(cls, action_name):
        """Looks up a registered mode class by its associated action's name.

        :param action_name: a string containing an action name (see this
           metaclass's docs regarding the ``ACTION_NAME`` class variable)
        :rtype: an InteractionMode class object, or `None`.

        """
        return cls.action_name_to_mode_class.get(action_name, None)

    @classmethod
    def get_action_names(cls):
        """Returns all action names associated with interaction.

        :rtype: an iterable of action name strings.

        """
        return cls.action_name_to_mode_class.keys()


## Mode base classes

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
    the class-level variable ``ACTION_NAME``: this should be the name of an
    action defined in `gui.app.Application.builder`'s XML file.

    """

    ## Class configuration

    #: All InteractionMode subclasses register themselves.
    __metaclass__ = ModeRegistry

    #: See the docs for `gui.mode.ModeRegistry`.
    ACTION_NAME = None

    #: True if the mode supports live update from the brush editor
    IS_LIVE_UPDATEABLE = False

    #: Timeout for Document.mode_flip_action_activated_cb(). How long, in
    #: milliseconds, it takes for the controller to change the key-up action
    #: when activated with a keyboard "Flip<ModeName>" action. Set to zero
    #: for modes where key-up should exit the mode at any time, and to a larger
    #: number for modes where the behaviour changes.
    keyup_timeout = 500

    ## Defaults for instances (sue me, I'm lazy)

    #: The `gui.document.Document` this mode affects: see enter()
    doc = None

    #: Broad description of what result clicking, moving the pointer,
    #: or dragging has in this mode. See `Behavior`.
    pointer_behavior = Behavior.NONE

    #: Broad description of what result scrolling a mouse scroll-wheel
    #: does in this mode. See `Behavior`.
    scroll_behavior = Behavior.NONE

    #: True if the mode supports switching to another mode based on
    #: combinations of pointer buttons and modifier keys.
    supports_button_switching = True

    #: Optional whitelist of the names of the modes which this mode can
    #: switch to. If the iterable is empty, all modes are possible.
    permitted_switch_actions = ()

    ## Status message info

    @classmethod
    def get_name(cls):
        """Returns a short human-readable description of the mode.

        :rtype: unicode

        This is used for status bar messages, and potentially elsewhere before
        the mode has been instantiated.  All concrete subclasses should
        override this.  By default the (non-localized) class name is returned.

        When capitalizing, use whatever style the GNOME HIG specifies for menu
        items.  In English, this is currently "header", or title case (first
        word capitalized, all other words capitalized except closed-class
        words).  Do not use trailing punctuation.

        """
        return unicode(cls.__name__)

    def get_usage(self):
        """Returns a medium-length usage message for the mode.

        :rtype: unicode

        This is used for status bar messages.  All concrete subclasses should
        override this.  The default return value is an empty string.

        The usage message should be a short, natural-sounding explanation to
        the user detailing what the current mode is for.  Note that the usage
        message is typically displayed after the mode's name or explanatory
        icon, so there is no need to repeat that.  Brevity is important because
        space is limited.

        When capitalizing, use whatever style the GNOME HIG specifies for
        tooltips.  In English, this is currently sentence case.  Use one
        complete sentence, and always omit the the trailing period.

        """
        return u""

    def __unicode__(self):
        return self.get_name()

    ## Associated action

    def get_action(self):
        """Returns any app action associated with the mode."""
        if self.doc and hasattr(self.doc, "app"):
            if self.ACTION_NAME:
                return self.doc.app.find_action(self.ACTION_NAME)

    ## Mode icon

    def get_icon_name(self):
        """Returns the icon to use when representing the mode.

        If there's an associated action, this method returns the icon
        associated with the action.
        """
        icon_name = None
        action = self.get_action()
        if action:
            icon_name = action.get_icon_name()
        if not icon_name:
            return 'missing-icon'
        return icon_name

    ## Mode stacking interface

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
        no longer the top mode on its ModeStack. It should commit any
        uncommitted work to the undo stack, just as `checkpoint()` does.

        """
        self.doc = None
        assert not hasattr(super(InteractionMode, self), "leave")

    def checkpoint(self):
        """Commits any of the mode's uncommitted work

        This is called at points in time when any uncommitted work needs to be
        made undoable right away. The mode continues to be active.  It isn't
        used when changing modes: leave() should manage that transition.

        If the current mode writes incrementally to the current command on the
        undo stack, this method should commit that work and construct a new
        command for its future state. If the command state is being constructed
        elsewhere, that state should be finalized to a new command.
        """
        assert not hasattr(super(InteractionMode, self), "checkpoint")

    ## Event handler defaults (no-ops)

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

    ## Drag sub-API (FIXME: this is in the wrong place)
    # Defined here to allow mixins to provide behaviour for both both drags and
    # regular events without having to derive from DragMode. Really these
    # buck-stops-here definitions belong in DragMode, so consider moving them
    # somewhere more sensible.

    def drag_start_cb(self, tdw, event):
        assert not hasattr(super(InteractionMode, self), "drag_start_cb")

    def drag_update_cb(self, tdw, event, dx, dy):
        assert not hasattr(super(InteractionMode, self), "drag_update_cb")

    def drag_stop_cb(self, tdw):
        assert not hasattr(super(InteractionMode, self), "drag_stop_cb")

    ## Internal utility functions

    def current_modifiers(self):
        """Returns the current set of modifier keys as a Gdk bitmask.

        For use in handlers for keypress events when the key in question is
        itself a modifier, handlers of multiple types of event, and when the
        triggering event isn't available. Pointer button event handling should
        use ``event.state & gtk.accelerator_get_default_mod_mask()``.
        """
        display = gdk.Display.get_default()
        screen, x, y, modifiers = display.get_pointer()
        modifiers &= gtk.accelerator_get_default_mod_mask()
        return modifiers

    def current_position(self):
        """Returns the current client pointer position on the main TDW

        For use in enter() methods: since the mode may be being entered by the
        user pressing a key, no position is available at this point. Normal
        event handlers should use their argument GdkEvents to determing position.
        """
        disp = self.doc.tdw.get_display()
        mgr = disp.get_device_manager()
        dev = mgr.get_client_pointer()
        win = self.doc.tdw.get_window()
        underwin, x, y, mods = win.get_device_position(dev)
        return x, y


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


class PaintingModeOptionsWidgetBase (gtk.Grid):
    """Base class for the options widget of a generic painting mode"""

    _COMMON_SETTINGS = [
        #TRANSLATORS: "Brush radius" for the options panel. Short.
        ('radius_logarithmic', _("Size:")),
        #TRANSLATORS: "Brush opacity" for the options panel. Short.
        ('opaque', _("Opaque:")),
        #TRANSLATORS: "Brush hardness/sharpness" for the options panel. Short.
        ('hardness', _("Sharp:")),
        #TRANSLATORS: "Additional pressure gain" for the options panel. Short.
        ('pressure_gain_log', _("Gain:")),
    ]

    def __init__(self):
        gtk.Grid.__init__(self)
        self.set_row_spacing(6)
        self.set_column_spacing(6)
        from application import get_app
        self.app = get_app()
        self.adjustable_settings = set()  #: What the reset button resets
        row = self.init_common_widgets(0)
        row = self.init_specialized_widgets(row)
        row = self.init_reset_widgets(row)

    def init_common_widgets(self, row):
        for cname, text in self._COMMON_SETTINGS:
            label = gtk.Label()
            label.set_text(text)
            label.set_alignment(1.0, 0.5)
            label.set_hexpand(False)
            self.adjustable_settings.add(cname)
            adj = self.app.brush_adjustment[cname]
            scale = gtk.HScale(adj)
            scale.set_draw_value(False)
            scale.set_hexpand(True)
            self.attach(label, 0, row, 1, 1)
            self.attach(scale, 1, row, 1, 1)
            row += 1
        return row

    def init_specialized_widgets(self, row):
        return row

    def init_reset_widgets(self, row):
        align = gtk.Alignment(0.5, 1.0, 1.0, 0.0)
        align.set_vexpand(True)
        self.attach(align, 0, row, 2, 1)
        button = gtk.Button(_("Reset"))
        button.connect("clicked", self.reset_button_clicked_cb)
        align.add(button)
        row += 1
        return row

    def reset_button_clicked_cb(self, button):
        app = self.app
        bm = app.brushmanager
        parent_brush = bm.get_parent_brush(brushinfo=app.brush)
        parent_binf = parent_brush.get_brushinfo()
        for cname in self.adjustable_settings:
            parent_value = parent_binf.get_base_value(cname)
            adj = self.app.brush_adjustment[cname]
            adj.set_value(parent_value)
        app.brushmodifier.normal_mode.activate()


class BrushworkModeMixin (InteractionMode):
    """Mixin for modes using brushes

    This mixin adds the ability to paint undoably to the current layer
    with proper atomicity and handling of checkpoints, and time-based
    automatic commits.

    Classes using this mixin should use `stroke_to()` to paint, and then
    may use the `brushwork_commit()` method to commit completed segments
    atomically to the command stack.  If a subclass needs greater
    control over new segments, `brushwork_begin()` can be used to start
    them recording.

    The `leave()` and `checkpoint()` methods defined here cooperatively
    commit all outstanding brushwork.
    """

    def __init__(self, abrupt_start=False, **kwds):
        """Cooperative init (this mixin initializes some private fields)

        :param bool abrupt_start: Make the 1st brushwork_begin() abrupt
        :param bool \*\*kwds: Passed through to other __init__s.

        Starting the first segment of brushwork abruptly makes the first
        segment cleaner in a (very limited) number of cases.
        See https://github.com/mypaint/mypaint/issues/11.
        """
        super(BrushworkModeMixin, self).__init__(**kwds)
        self.__abrupt_start = abrupt_start
        self.__active_brushwork = {}  # {model: Brushwork}

    def brushwork_begin(self, model, description=None, abrupt=False):
        """Begins a new segment of active brushwork for a model

        :param lib.document.Document model: The model to begin work on
        :param unicode description: Optional description of the work
        :param bool abrupt: Fake a zero-pressure "stroke_to()" at start

        Passing ``None`` for the description is suitable for freehand
        drawing modes.  This method will be called automatically with
        the default options by `stroke_to()` if needed, so not all
        subclasses need to use it.
        """
        # Commit any previous work for this model
        cmd = self.__active_brushwork.get(model)
        if cmd is not None:
            self.brushwork_commit(model, abrupt=abrupt)
        # New segment of brushwork
        layer_path = model.layer_stack.current_path
        cmd = lib.command.Brushwork(
            model, layer_path,
            description=description,
            abrupt_start=(abrupt or self.__abrupt_start),
        )
        self.__abrupt_start = False
        cmd.__last_pos = None
        self.__active_brushwork[model] = cmd

    def brushwork_commit(self, model, abrupt=False):
        """Commits active brushwork for a model to the command stack

        :param lib.document.Document model: The model to commit work to
        :param bool abrupt: End with a faked zero pressure "stroke_to()"
        """
        cmd = self.__active_brushwork.pop(model, None)
        if cmd is None:
            return
        if abrupt and cmd.__last_pos is not None:
            x, y, xtilt, ytilt = cmd.__last_pos
            pressure = 0.0
            dtime = 0.0
            cmd.stroke_to(dtime, x, y, pressure, xtilt, ytilt)
        changed = cmd.stop_recording()
        if changed:
            model.do(cmd)

    def __commit_all(self, abrupt=False):
        """Commits all active brushwork"""
        for model in list(self.__active_brushwork.keys()):
            self.brushwork_commit(model, abrupt=abrupt)

    def stroke_to(self, model, dtime, x, y, pressure, xtilt, ytilt,
                  auto_split=True):
        """Feeds an updated stroke position to the brush engine

        :param lib.document.Document model: model on which to paint
        :param float dtime: Seconds since the last call to this method
        :param float x: Document X position update
        :param float y: Document Y position update
        :param float pressure: Pressure, ranging from 0.0 to 1.0
        :param float xtilt: X-axis tilt, ranging from -1.0 to 1.0
        :param float ytilt: Y-axis tilt, ranging from -1.0 to 1.0
        :param bool auto_split: Split ongoing brushwork if due

        During normal operation, succesive calls to `stroke_to()` record
        an ongoing sequence of `lib.command.Brushwork` commands on the
        undo stack, stopping and committing the currently recording
        command when it becomes due.
        """
        cmd = self.__active_brushwork.get(model, None)
        desc0 = None
        if auto_split and cmd and cmd.split_due:
            desc0 = cmd.description  # retain for the next cmd
            self.brushwork_commit(model, abrupt=False)
            assert model not in self.__active_brushwork
            cmd = None
        if not cmd:
            self.brushwork_begin(model, description=desc0, abrupt=False)
            cmd = self.__active_brushwork[model]
        cmd.stroke_to(dtime, x, y, pressure, xtilt, ytilt)
        cmd.__last_pos = (x, y, xtilt, ytilt)

    def leave(self, **kwds):
        """Leave mode, committing any outstanding brushwork

        The leave action defined here is careful to tail off strokes
        cleanly: certain subclasses are geared towards fast capture of
        data and queued delivery of stroke information, so we have to
        reset the brush engine's idea of pressure fast. If we don't, an
        interrupted queued stroke can result in a *huge* sequence of
        dabs from the last processed position to wherever the cursor is
        right now.
        """
        logger.debug("BrushworkModeMixin: leave()")
        self.__commit_all(abrupt=True)
        super(BrushworkModeMixin, self).leave(**kwds)

    def checkpoint(self, **kwds):
        """Committing any outstanding brushwork

        Like `leave()`, this commits the currently recording Brushwork
        command for each known model; however we do not attempt to tail
        off brushstrokes cleanly.
        """
        logger.debug("BrushworkModeMixin: checkpoint()")
        super(BrushworkModeMixin, self).checkpoint(**kwds)
        self.__commit_all(abrupt=False)


class SingleClickMode (InteractionMode):
    """Base class for non-drag (single click) modes"""

    #: The cursor to use when entering the mode
    cursor = gdk.Cursor(gdk.BOGOSITY)

    def __init__(self, ignore_modifiers=False, **kwds):
        super(SingleClickMode, self).__init__(**kwds)
        self._button_pressed = None

    def enter(self, **kwds):
        super(SingleClickMode, self).enter(**kwds)
        assert self.doc is not None
        self.doc.tdw.set_override_cursor(self.cursor)

    def leave(self, **kwds):
        if self.doc is not None:
            self.doc.tdw.set_override_cursor(None)
        super(SingleClickMode, self).leave(**kwds)

    def button_press_cb(self, tdw, event):
        if event.button == 1 and event.type == gdk.BUTTON_PRESS:
            self._button_pressed = 1
            return False
        else:
            return super(SingleClickMode, self).button_press_cb(tdw, event)

    def button_release_cb(self, tdw, event):
        if event.button == self._button_pressed:
            self._button_pressed = None
            self.clicked_cb(tdw, event)
            return False
        else:
            return super(SingleClickMode, self).button_press_cb(tdw, event)

    def clicked_cb(self, tdw, event):
        assert not hasattr(super(SingleClickMode, self), "clicked_cb")


class DragMode (InteractionMode):
    """Base class for drag activities.

    Dragm modes can be entered when the button is pressed, or not yet
    pressed.  If the button is pressed when the mode is entered. the
    initial position will be determined from the first motion event.

    Drag modes are normally "spring-loaded", meaning that when a drag
    mode is first entered, it remembers which modifier keys were held
    down at that time. When these keys are released, the mode will exit.

    """

    inactive_cursor = gdk.Cursor(gdk.BOGOSITY)
    active_cursor = None

    #: If true, exit mode when initial modifiers are released
    SPRING_LOADED = True

    def __init__(self, ignore_modifiers=False, **kwds):
        """Construct, possibly ignoring initial modifiers.

        :param ignore_modifiers: If True, ignore initial modifier keys.

        Drag modes can be instructed to ignore the initial set of
        modifiers when they're entered. This is appropriate when the
        mode is being entered in response to a keyboard shortcut.
        Modifiers don't mean the same thing for keyboard shortcuts.
        Conversely, toolbar buttons and mode-switching via pointer
        buttons should use the default behaviour.

        In practice, it's not quite so clear cut. Instead we have
        keyboard-friendly "Flip*" actions which allow the mode to be
        toggled off with a second press.  These actions use the
        `ignore_modifiers` behaviour, and coexist with a secondary layer
        of radioactions which don't do this, but which reflect the state
        prettily.

        """
        super(DragMode, self).__init__(**kwds)
        self._tdw_grab_broken_conninfo = None
        self._reset_drag_state()
        self.initial_modifiers = None
        #: Ignore the initial modifiers (FIXME: bad name, maybe not public?)
        self.ignore_modifiers = ignore_modifiers

    def _reset_drag_state(self):
        self.last_x = None
        self.last_y = None
        self.start_x = None
        self.start_y = None
        self._start_keyval = None
        self._start_button = None
        self._grab_widget = None
        if self._tdw_grab_broken_conninfo is not None:
            tdw, connid = self._tdw_grab_broken_conninfo
            tdw.disconnect(connid)
            self._tdw_grab_broken_conninfo = None

    def _stop_drag(self, t=gdk.CURRENT_TIME):
        # Stops any active drag, calls drag_stop_cb(), and cleans up.
        if not self.in_drag:
            return
        tdw = self._grab_widget
        tdw.grab_remove()
        gdk.keyboard_ungrab(t)
        gdk.pointer_ungrab(t)
        self._grab_widget = None
        self.drag_stop_cb(tdw)
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
        event_mask = (gdk.BUTTON_PRESS_MASK |
                      gdk.BUTTON_RELEASE_MASK |
                      gdk.POINTER_MOTION_MASK)
        cursor = self.active_cursor
        if cursor is None:
            cursor = self.inactive_cursor

        # Grab the pointer
        grab_status = gdk.pointer_grab(tdw_window, False, event_mask, None,
                                       cursor, event.time)
        if grab_status != gdk.GRAB_SUCCESS:
            logger.warning("pointer grab failed: %r", grab_status)
            logger.debug("gdk_pointer_is_grabbed(): %r",
                         gdk.pointer_is_grabbed())
            # There seems to be a race condition between this grab under
            # PyGTK/GTK2 and some other grab - possibly just the implicit grabs
            # on color selectors: https://gna.org/bugs/?20068 Only pointer
            # events are affected, and PyGI+GTK3 is unaffected.
            #
            # It's probably safest to exit the mode and not start the drag.
            # This condition should be rare enough for this to be a valid
            # approach: the irritation of having to click again to do something
            # should be far less than that of getting "stuck" in a drag.
            self._bailout()

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
        connid = tdw.connect("grab-broken-event", self._tdw_grab_broken_cb)
        self._tdw_grab_broken_conninfo = (tdw, connid)

        # Grab the keyboard too, to be certain of getting the key release event
        # for a spacebar drag.
        grab_status = gdk.keyboard_grab(tdw_window, False, event.time)
        if grab_status != gdk.GRAB_SUCCESS:
            logger.warning("Keyboard grab failed: %r", grab_status)
            self._bailout()
            gdk.pointer_ungrab(event.time)
            return

        # GTK too...
        tdw.grab_add()
        self._grab_widget = tdw

        # Drag has started, perform whatever action the mode needs.
        self.drag_start_cb(tdw, event)

    def _bailout(self):
        """Attempt to exit this mode safely, via an idle routine

        The actual task is handled by an idle callback to make this
        method safe to call during a mode's enter() or leave() methods.
        Modes on top of the one requesting bailout will also be ejected.

        """
        from application import get_app
        app = get_app()
        if self not in app.doc.modes:
            logger.debug(
                "bailout: cannot bail out of %r: "
                "mode is not in the mode stack",
                self,
            )
            return
        logger.debug("bailout: starting idler to safely bail out of %r", self)
        gobject.idle_add(self._bailout_idle_cb, app.doc.modes)

    def _bailout_idle_cb(self, modestack):
        """Bail out of this mode if it's anywhere in the mode stack"""
        while self in modestack:
            logger.debug("bailout idler: leaving %r", modestack.top)
            modestack.pop()
        logger.debug("bailout idler: done")
        return False

    def _tdw_grab_broken_cb(self, tdw, event):
        # Cede control as cleanly as possible if something else grabs either
        # the keyboard or the pointer while a grab is active.
        # One possible cause for https://gna.org/bugs/?20333
        logger.debug("grab-broken-event on %r", tdw)
        logger.debug(" send_event  : %r", event.send_event)
        logger.debug(" keyboard    : %r", event.keyboard)
        logger.debug(" implicit    : %r", event.implicit)
        logger.debug(" grab_window : %r", event.grab_window)
        self._bailout()
        return True

    @property
    def in_drag(self):
        return self._grab_widget is not None

    def enter(self, **kwds):
        """Enter the mode, recording the held modifier keys the 1st time

        The attribute `self.initial_modifiers` is set the first time the
        mode is entered.

        """
        super(DragMode, self).enter(**kwds)
        assert self.doc is not None
        self.doc.tdw.set_override_cursor(self.inactive_cursor)

        if self.SPRING_LOADED:
            if self.ignore_modifiers:
                self.initial_modifiers = 0
                return
            old_modifiers = getattr(self, "initial_modifiers", None)
            if old_modifiers is not None:
                # Re-entering due to an overlying mode being popped
                if old_modifiers != 0:
                    # This mode started with modifiers held
                    modifiers = self.current_modifiers()
                    if (modifiers & old_modifiers) == 0:
                        # But nonee of them are held any more,
                        # so queue a further pop.
                        gobject.idle_add(self.__pop_modestack_idle_cb)
            else:
                # This mode is being entered for the first time;
                # record modifiers
                modifiers = self.current_modifiers()
                self.initial_modifiers = self.current_modifiers()

    def __pop_modestack_idle_cb(self):
        # Pop the mode stack when this mode is re-entered but has to leave
        # straight away because its modifiers are no longer held. Doing it in
        # an idle function avoids confusing the derived class's enter() method:
        # a leave() during an enter() would be strange.
        if self.initial_modifiers is not None:
            if self is self.doc.modes.top:
                self.doc.modes.pop()
        return False

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

        if self.SPRING_LOADED:
            if event.is_modifier and self.in_drag:
                return False
            if self.initial_modifiers:
                modifiers = self.current_modifiers()
                if modifiers & self.initial_modifiers == 0:
                    if self is self.doc.modes.top:
                        self.doc.modes.pop()
                    return True

        # Fall through to other behavioral mixins
        return super(DragMode, self).key_release_cb(win, tdw, event)


class OneshotDragMode (DragMode):
    """Drag modes that can exit immediately when the drag stops

    These are utility modes which allow the user to do quick, simple
    tasks with the canvas like pick a color from it or pan the view.
    """

    #: If true, and spring-loaded, stay active if no modifiers were
    #: held initially.
    unmodified_persist = False

    def stackable_on(self, mode):
        """Oneshot modes return to the mode the user came from on exit"""
        return not isinstance(mode, OneshotDragMode)

    def get_options_widget(self):
        """Don't replace stuff in the options panel by default"""
        return None

    def drag_stop_cb(self, tdw):
        if not hasattr(self, "initial_modifiers"):
            # Always exit at the end of a drag if not spring-loaded.
            if self is self.doc.modes.top:
                self.doc.modes.pop()
        elif self.initial_modifiers != 0:
            # If started with modifiers, keeping the modifiers held keeps
            # spring-loaded modes active. If not, exit the mode.
            if (self.initial_modifiers & self.current_modifiers()) == 0:
                if self is self.doc.modes.top:
                    self.doc.modes.pop()
        else:
            # No modifiers were held when this mode was entered.
            if not self.unmodified_persist:
                if self is self.doc.modes.top:
                    self.doc.modes.pop()
        return super(OneshotDragMode, self).drag_stop_cb(tdw)


## Mode stack


class _NullMode (InteractionMode):
    """A mode that does nothing (placeholder only)"""


class ModeStack (object):
    """A stack of InteractionModes. The top mode is the active one.

    Mode stacks can never be empty. If the final element is popped, it
    will be replaced with a new instance of its ``default_mode_class``,
    instantiated with ``**default_mode_kwargs``.

    """

    def __init__(self, doc):
        """Initialize for a particular controller

        :param doc: Controller instance
        :type doc: gui.document.CanvasController

        The main MyPaint app uses an instance of `gui.document.Document`
        as `doc`. Simpler drawing canvases can use a basic
        CanvasController and a simpler `default_mode_class`.
        """
        object.__init__(self)
        self._stack = []
        self._doc = doc
        self._flushing_model_updates = False
        if hasattr(doc, "model"):
            doc.model.flush_updates += self._flush_model_updates_cb
        #: Class to instantiate if stack is empty: callable with 0 args.
        default_mode_class = _NullMode
        #: Keyword parameters for default_mode_class.
        default_mode_kwargs = {}

    def _flush_model_updates_cb(self, model):
        """Flushes pending model updates from the current mode

        This issues a `checkpoint()` on the current InteractionMode.
        """
        if self._flushing_model_updates:
            return
        self._flushing_model_updates = True
        self.top.checkpoint()
        self._flushing_model_updates = False

    @event
    def changed(self, old, new):
        """Event: emitted when the active mode changes

        :param old: The previous active mode
        :param new: The new `top` (current) mode

        This event is emitted after the ``enter()`` method of the new
        mode has been called, and therefore after the ``leave()`` of the
        old mode too. On occasion, the old mode may be null.

        Context-aware pushes call this only once, with the old active
        and newly active mode only regardless of how many modes were
        skipped.

        """

    @property
    def top(self):
        """The top node on the stack"""
        # Perhaps rename to "active()"?
        new_mode = self._check()
        if new_mode is not None:
            new_mode.enter(doc=self._doc)
            self.changed(None, new_mode)
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
        old_mode = None
        if len(self._stack) > 0:
            old_mode = self._stack[-1]
        while len(self._stack) > 0:
            if mode.stackable_on(self._stack[-1]):
                break
            self._stack.pop(-1).leave()
            if len(self._stack) > 0:
                self._stack[-1].enter(doc=self._doc)
        # Stack on top of any remaining compatible mode
        if len(self._stack) > 0:
            self._stack[-1].leave()
        self._stack.append(mode)
        mode.enter(doc=self._doc)
        self.changed(old=old_mode, new=mode)

    def pop(self):
        """Pops a mode, leaving the old top mode and entering the exposed top.
        """
        old_mode = None
        if len(self._stack) > 0:
            old_mode = self._stack.pop(-1)
            old_mode.leave()
        top_mode = self._check()
        if top_mode is None:
            top_mode = self._stack[-1]
        # No need to checkpoint user activity here: leave() was already called
        top_mode.enter(doc=self._doc)
        self.changed(old=old_mode, new=top_mode)

    def push(self, mode):
        """Pushes a mode, and enters it.

        :param mode: Mode to be stacked and made active
        :type mode: InteractionMode
        """
        old_mode = None
        if len(self._stack) > 0:
            old_mode = self._stack[-1]
            old_mode.leave()
        # No need to checkpoint user activity here: leave() was already called
        self._stack.append(mode)
        mode.enter(doc=self._doc)
        self.changed(old=old_mode, new=mode)

    def reset(self, replacement=None):
        """Clears the stack, popping the final element and replacing it.

        :param replacement: Optional mode to go on top of the cleared stack.
        :type replacement: `InteractionMode`.

        """
        old_top_mode = None
        if len(self._stack) > 0:
            old_top_mode = self._stack[-1]
        while len(self._stack) > 0:
            old_mode = self._stack.pop(-1)
            old_mode.leave()
            if len(self._stack) > 0:
                self._stack[-1].enter(doc=self._doc)
        top_mode = self._check(replacement)
        assert top_mode is not None
        self.changed(old=old_top_mode, new=top_mode)

    def _check(self, replacement=None):
        """Ensures that the stack is non-empty

        :param replacement: Optional replacement mode instance.
        :type replacement: `InteractionMode`.

        Returns the new top mode if one was pushed.
        """
        if len(self._stack) > 0:
            return None
        if replacement is not None:
            mode = replacement
        else:
            mode = self.default_mode_class(**self.default_mode_kwargs)
        self._stack.append(mode)
        mode.enter(doc=self._doc)
        return mode

    def __repr__(self):
        """Plain-text representation."""
        s = '<ModeStack ['
        s += ", ".join([m.__class__.__name__ for m in self._stack])
        s += ']>'
        return s

    def __len__(self):
        """Returns the number of modes on the stack."""
        return len(self._stack)

    def __nonzero__(self):
        """Mode stacks never test false, regardless of length."""
        return True

    def __iter__(self):
        for mode in self._stack:
            yield mode
