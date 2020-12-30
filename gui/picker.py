# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""UI behaviour for picking things from the canvas.

The grab and button behaviour objects work like MVP presenters
with a rather wide scope.

"""

## Imports
from __future__ import division, print_function

from gui.tileddrawwidget import TiledDrawWidget
from gui.document import Document
from lib.gettext import C_
import gui.cursor

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib

import abc
import logging
logger = logging.getLogger(__name__)


_RELEASE_EVENTS = {Gdk.EventType.BUTTON_RELEASE, Gdk.EventType.TOUCH_END}
_PRESS_EVENTS = {Gdk.EventType.BUTTON_PRESS, Gdk.EventType.TOUCH_BEGIN}


## Class definitions

class PickingGrabPresenter (object):
    """Picking something via a grab (abstract base, MVP presenter)

    This presenter mediates between passive GTK view widgets
    accessed via the central app,
    and a model consisting of some drawing state within the application.
    When activated, it establishes a pointer grab and a keyboard grab,
    updates the thing being grabbed zero or more times,
    then exits making sure that the grab is cleaned up correctly.

    """

    ## Class configuration

    __metaclass__ = abc.ABCMeta

    _GRAB_MASK = (Gdk.EventMask.BUTTON_RELEASE_MASK
                  | Gdk.EventMask.BUTTON_PRESS_MASK
                  | Gdk.EventMask.BUTTON_MOTION_MASK)

    ## Initialization

    def __init__(self):
        """Basic initialization."""
        super(PickingGrabPresenter, self).__init__()
        self._app = None
        self._statusbar_info_cache = None
        self._grab_button_num = None
        self._grabbed_pointer_dev = None
        self._grabbed_keyboard_dev = None
        self._grab_event_handler_ids = None
        self._delayed_picking_update_id = None

    @property
    def app(self):
        """The coordinating app object."""
        # FIXME: The view (statusbar, grab owner widget) is accessed
        # FIXME: through this, which may be a problem in the long term.
        # FIXME: There's a need to set up event masks before starting
        # FIXME: the grab, and this may make _start_grab() more fragile.
        # Ref: https://github.com/mypaint/mypaint/issues/324
        return self._app

    @app.setter
    def app(self, app):
        self._app = app
        self._statusbar_info_cache = None

    ## Internals

    @property
    def _grab_owner(self):
        """The view widget owning the grab."""
        return self.app.drawWindow

    @property
    def _statusbar_info(self):
        """The view widget and context for displaying status msgs."""
        if not self._statusbar_info_cache:
            statusbar = self.app.statusbar
            cid = statusbar.get_context_id("picker-button")
            self._statusbar_info_cache = (statusbar, cid)
        return self._statusbar_info_cache

    def _hide_status_message(self):
        """Remove all statusbar messages coming from this class"""
        statusbar, cid = self._statusbar_info
        statusbar.remove_all(cid)

    def _show_status_message(self):
        """Display a status message via the view."""
        statusbar, cid = self._statusbar_info
        statusbar.push(cid, self.picking_status_text)

    ## Activation

    def activate_from_button_event(self, event):
        """Activate during handling of a GdkEventButton (press/release)

        If the event is a button press, then the grab will start
        immediately, begin updating immediately, and will terminate by
        the release of the initiating button.

        If the event is a button release, then the grab start will be
        deferred to start in an idle handler. When the grab starts, it
        won't begin updating until the user clicks button 1 (and only
        button 1), and it will only be terminated with a button1
        release. This covers the case of events delivered to "clicked"
        signal handlers

        """
        if event.type in _PRESS_EVENTS:
            logger.debug("Starting picking grab")
            has_button_info, button_num = event.get_button()
            if not has_button_info:
                return
            self._start_grab(event.device, event.time, button_num)
        elif event.type in _RELEASE_EVENTS:
            logger.debug("Queueing picking grab")
            GLib.idle_add(
                self._start_grab,
                event.device,
                event.time,
                None,
            )

    ## Required interface for subclasses

    @abc.abstractproperty
    def picking_cursor(self):
        """The cursor to use while picking.

        :returns: The cursor to use during the picking grab.
        :rtype: Gdk.Cursor

        This abstract property must be overridden with an implementation
        giving an appropriate cursor to display during the picking grab.

        """

    @abc.abstractproperty
    def picking_status_text(self):
        """The statusbar text to use during the grab."""

    @abc.abstractmethod
    def picking_update(self, device, x_root, y_root):
        """Update whatever's being picked during & after picking.

        :param Gdk.Device device: Pointer device currently grabbed
        :param int x_root: Absolute screen X coordinate
        :param int y_root: Absolute screen Y coordinate

        This abstract method must be overridden with an implementation
        which updates the model object being picked.
        It is always called at the end of the picking grab
        when button1 is released,
        and may be called several times during the grab
        while button1 is held.

        See gui.tileddrawwidget.TiledDrawWidget.get_tdw_under_device()
        for details of how to get canvas widgets
        and their related document models and controllers.

        """

    ## Internals

    def _start_grab(self, device, time, inibutton):
        """Start the pointer grab, and enter the picking state.

        :param Gdk.Device device: Initiating pointer device.
        :param int time: The grab start timestamp.
        :param int inibutton: Initiating pointer button.

        The associated keyboard device is grabbed too.
        This method assumes that inibutton is currently held. The grab
        terminates when inibutton is released.

        """
        logger.debug("Starting picking grab...")

        # The device to grab must be a virtual device,
        # because we need to grab its associated virtual keyboard too.
        # We don't grab physical devices directly.
        if device.get_device_type() == Gdk.DeviceType.SLAVE:
            device = device.get_associated_device()
        elif device.get_device_type() == Gdk.DeviceType.FLOATING:
            logger.warning(
                "Cannot start grab on floating device %r",
                device.get_name(),
            )
            return
        assert device.get_device_type() == Gdk.DeviceType.MASTER

        # Find the keyboard paired to this pointer.
        assert device.get_source() != Gdk.InputSource.KEYBOARD
        keyboard_device = device.get_associated_device()  # again! top API!
        assert keyboard_device.get_device_type() == Gdk.DeviceType.MASTER
        assert keyboard_device.get_source() == Gdk.InputSource.KEYBOARD

        # Internal state checks
        assert not self._grabbed_pointer_dev
        assert not self._grab_button_num
        assert self._grab_event_handler_ids is None

        # Validate the widget we're expected to grab.
        owner = self._grab_owner
        assert owner.get_has_window()
        window = owner.get_window()
        assert window is not None

        # Ensure that it'll receive termination events.
        owner.add_events(self._GRAB_MASK)
        assert (int(owner.get_events() & self._GRAB_MASK) == int(self._GRAB_MASK)), \
            "Grab owner's events must match %r" % (self._GRAB_MASK,)

        # There should be no message in the statusbar from this Grab,
        # but clear it out anyway.
        self._hide_status_message()

        # Grab item, pointer first
        result = device.grab(
            window = window,
            grab_ownership = Gdk.GrabOwnership.APPLICATION,
            owner_events = False,
            event_mask = self._GRAB_MASK,
            cursor = self.picking_cursor,
            time_ = time,
        )
        if result != Gdk.GrabStatus.SUCCESS:
            logger.error(
                "Failed to create pointer grab on %r. "
                "Result: %r.",
                device.get_name(),
                result,
            )
            device.ungrab(time)
            return False  # don't requeue

        # Need to grab the keyboard too, since Mypaint uses hotkeys.
        keyboard_mask = Gdk.EventMask.KEY_PRESS_MASK \
            | Gdk.EventMask.KEY_RELEASE_MASK
        result = keyboard_device.grab(
            window = window,
            grab_ownership = Gdk.GrabOwnership.APPLICATION,
            owner_events = False,
            event_mask = keyboard_mask,
            cursor = self.picking_cursor,
            time_ = time,
        )
        if result != Gdk.GrabStatus.SUCCESS:
            logger.error(
                "Failed to create grab on keyboard associated with %r. "
                "Result: %r",
                device.get_name(),
                result,
            )
            device.ungrab(time)
            keyboard_device.ungrab(time)
            return False  # don't requeue

        # Grab is established
        self._grabbed_pointer_dev = device
        self._grabbed_keyboard_dev = keyboard_device
        logger.debug(
            "Grabs established on pointer %r and keyboard %r",
            device.get_name(),
            keyboard_device.get_name(),
        )

        # Tell the user how to work the thing.
        self._show_status_message()

        # Establish temporary event handlers during the grab.
        # These are responsible for ending the grab state.
        handlers = {
            "button-release-event": self._in_grab_button_release_cb,
            "motion-notify-event": self._in_grab_motion_cb,
            "grab-broken-event": self._in_grab_grab_broken_cb,
        }
        if not inibutton:
            handlers["button-press-event"] = self._in_grab_button_press_cb
        else:
            self._grab_button_num = inibutton
        handler_ids = []
        for signame, handler_cb in handlers.items():
            hid = owner.connect(signame, handler_cb)
            handler_ids.append(hid)
            logger.debug("Added handler for %r: hid=%d", signame, hid)
        self._grab_event_handler_ids = handler_ids

        return False   # don't requeue

    def _in_grab_button_press_cb(self, widget, event):
        assert self._grab_button_num is None
        if event.type not in _PRESS_EVENTS:
            return False
        if not self._check_event_devices_still_grabbed(event):
            return
        has_button_info, button_num = event.get_button()
        if not has_button_info:
            return False
        if event.device is not self._grabbed_pointer_dev:
            return False
        self._grab_button_num = button_num
        return True

    def _in_grab_button_release_cb(self, widget, event):
        assert self._grab_button_num is not None
        if event.type not in _RELEASE_EVENTS:
            return False
        if not self._check_event_devices_still_grabbed(event):
            return
        has_button_info, button_num = event.get_button()
        if not has_button_info:
            return False
        if button_num != self._grab_button_num:
            return False
        if event.device is not self._grabbed_pointer_dev:
            return False
        self._end_grab(event)
        assert self._grab_button_num is None
        return True

    def _in_grab_motion_cb(self, widget, event):
        assert self._grabbed_pointer_dev is not None
        if not self._check_event_devices_still_grabbed(event):
            return True
        if event.device is not self._grabbed_pointer_dev:
            return False
        if not self._grab_button_num:
            return False
        # Due to a performance issue, picking can take more time
        # than we have between two motion events (about 8ms).
        if self._delayed_picking_update_id:
            GLib.source_remove(self._delayed_picking_update_id)
        self._delayed_picking_update_id = GLib.idle_add(
            self._delayed_picking_update_cb,
            event.device,
            event.x_root,
            event.y_root,
        )
        return True

    def _in_grab_grab_broken_cb(self, widget, event):
        logger.debug("Grab broken, cleaning up.")
        self._ungrab_grabbed_devices()
        return False

    def _end_grab(self, event):
        """Finishes the picking grab normally."""
        if not self._check_event_devices_still_grabbed(event):
            return
        device = event.device
        try:
            self.picking_update(device, event.x_root, event.y_root)
        finally:
            self._ungrab_grabbed_devices(time=event.time)

    def _check_event_devices_still_grabbed(self, event):
        """Abandon picking if devices aren't still grabbed.

        This can happen if the escape key is pressed during the grab -
        the gui.keyboard handler is still invoked in the normal way,
        and Escape just does an ungrab.

        """
        cleanup_needed = False
        for dev in (self._grabbed_pointer_dev, self._grabbed_keyboard_dev):
            if not dev:
                cleanup_needed = True
                continue
            display = dev.get_display()
            if not display.device_is_grabbed(dev):
                logger.debug(
                    "Device %r is no longer grabbed: will clean up",
                    dev.get_name(),
                )
                cleanup_needed = True
        if cleanup_needed:
            self._ungrab_grabbed_devices(time=event.time)
        return not cleanup_needed

    def _ungrab_grabbed_devices(self, time=Gdk.CURRENT_TIME):
        """Ungrabs devices thought to be grabbed, and cleans up."""
        for dev in (self._grabbed_pointer_dev, self._grabbed_keyboard_dev):
            if not dev:
                continue
            logger.debug("Ungrabbing device %r", dev.get_name())
            dev.ungrab(time)
        # Unhook potential grab leave handlers
        # but only if the pick succeeded.
        if self._grab_event_handler_ids:
            for hid in self._grab_event_handler_ids:
                owner = self._grab_owner
                owner.disconnect(hid)
        self._grab_event_handler_ids = None
        # Update state (prevents the idler updating a 2nd time)
        self._grabbed_pointer_dev = None
        self._grabbed_keyboard_dev = None
        self._grab_button_num = None
        self._hide_status_message()

    def _delayed_picking_update_cb(self, ptrdev, x_root, y_root):
        """Delayed picking updates during grab.

        Some picking operations can be CPU-intensive, so this is called
        by an idle handler. If the user clicks and releases immediately,
        this never gets called, so a final call to picking_update() is
        made separately after the grab finishes.

        See: picking_update().

        """
        try:
            if ptrdev is self._grabbed_pointer_dev:
                self.picking_update(ptrdev, x_root, y_root)
        except:
            logger.exception("Exception in picking idler")
            # HMM: if it's not logged here, it won't be recorded...
        finally:
            self._delayed_picking_update_id = None
            return False


class ContextPickingGrabPresenter (PickingGrabPresenter):
    """Context picking behaviour (concrete MVP presenter)"""

    @property
    def picking_cursor(self):
        """The cursor to use while picking"""
        return self.app.cursors.get_icon_cursor(
            icon_name = "mypaint-brush-tip-symbolic",
            cursor_name = gui.cursor.Name.CROSSHAIR_OPEN_PRECISE,
        )

    @property
    def picking_status_text(self):
        """The statusbar text to use during the grab."""
        return C_(
            "context picker: statusbar text during grab",
            u"Pick brushstroke settings, stroke color, and layer…",
        )

    def picking_update(self, device, x_root, y_root):
        """Update brush and layer during & after picking."""
        # Can only pick from TDWs
        tdw, x, y = TiledDrawWidget.get_tdw_under_device(device)
        if tdw is None:
            return
        # Determine which document controller owns that tdw
        doc = None
        for d in Document.get_instances():
            if tdw is d.tdw:
                doc = d
                break
        if doc is None:
            return
        # Get that controller to do the pick.
        # Arguably this should be direct to the model.
        x, y = tdw.display_to_model(x, y)
        doc.pick_context(x, y)


class ColorPickingGrabPresenter (PickingGrabPresenter):
    """Color picking behaviour (concrete MVP presenter)"""

    @property
    def picking_cursor(self):
        """The cursor to use while picking"""
        return self.app.cursors.get_icon_cursor(
            icon_name = "mypaint-colors-symbolic",
            cursor_name = gui.cursor.Name.PICKER,
        )

    @property
    def picking_status_text(self):
        """The statusbar text to use during the grab."""
        return C_(
            "color picker: statusbar text during grab",
            u"Pick color…",
        )

    def picking_update(self, device, x_root, y_root):
        """Update brush and layer during & after picking."""
        tdw, x, y = TiledDrawWidget.get_tdw_under_device(device)
        if tdw is None:
            return
        color = tdw.pick_color(x, y)
        cm = self.app.brush_color_manager
        cm.set_color(color)


class ButtonPresenter (object):
    """Picking behaviour for a button (MVP presenter)

    This presenter mediates between a passive view consisting of a
    button, and a peer PickingGrabPresenter instance which does the
    actual work after the button is clicked.

    """

    ## Initialization

    def __init__(self):
        """Initialize."""
        super(ButtonPresenter, self).__init__()
        self._evbox = None
        self._button = None
        self._grab = None

    def set_picking_grab(self, grab):
        self._grab = grab

    def set_button(self, button):
        """Connect view button.

        :param Gtk.Button button: the initiator button

        """
        button.connect("clicked", self._clicked_cb)
        self._button = button

    ## Event handling

    def _clicked_cb(self, button):
        """Handle click events on the initiator button."""
        event = Gtk.get_current_event()
        assert event is not None
        assert event.type in _RELEASE_EVENTS, (
            "The docs lie! Current event's type is %r." % (event.type,),
        )
        self._grab.activate_from_button_event(event)
