# This file is part of MyPaint.
# Copyright (C) 2010-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Base classes for window types and window-related helper functions"""


## Imports

from __future__ import division, print_function
import logging

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib

from lib.helpers import clamp, Rect
from lib.pycompat import xrange

logger = logging.getLogger(__name__)


## Base class definitions

class Dialog (Gtk.Dialog):
    """Base dialog accepting all keyboard input.

    Dialogs hide when closed. By default, they accept all keyboard input and
    are not modal. They can (and should) be kept around as references, and can
    be freely hidden and shown after construction.

    """
    def __init__(self, app, *args, **kwargs):
        Gtk.Dialog.__init__(self, *args, **kwargs)
        self.app = app
        if app and app.drawWindow:
            self.set_transient_for(app.drawWindow)
        self.connect('delete-event', lambda w, e: self.hide_on_delete())


class SubWindow (Gtk.Window):
    """A subwindow in the GUI.

    SubWindows don't accept keyboard input by default, but if your subclass
    requires it, pass key_input to the constructor.

    """

    def __init__(self, app, key_input=False):
        """Initialize, as an application subwindow.

        :param app: The main application instance. May be None for testing.
        :param key_input: set to True to accept keyboard input.
        :param hide_on_delete: set False to turn off hide-on-delete behavior.

        """
        Gtk.Window.__init__(self)
        self.app = app
        if app and not key_input:
            self.app.kbm.add_window(self)
            # TODO: do we need a separate class for keyboard-input-friendly
            # windows? Do they share anything in common with dialogs (could
            # they all be implemented as dialogs?)
        self.set_accept_focus(key_input)
        self.pre_hide_pos = None
        # Only hide when the close button is pressed if running as a subwindow
        if app:
            self.connect('delete-event', lambda w, e: self.hide_on_delete())
        # Win32 and some Linux DEs are responsive to the following: keeps the
        # window above the main window in fullscreen.
        if app:
            self.set_transient_for(app.drawWindow)

    def show_all(self):
        pos = self.pre_hide_pos
        Gtk.Window.show_all(self)
        if pos:
            self.move(*pos)
        # To keep Compiz happy, move() must be called in the very same
        # handler as show_all(), immediately after. Wiring it in to
        # a map-event or show event handler won't do. Workaround for
        # https://bugs.launchpad.net/ubuntu/+source/compiz/+bug/155101

    def hide(self):
        self.pre_hide_pos = self.get_position()
        Gtk.Window.hide(self)


class PopupWindow (Gtk.Window):
    """
    A popup window, with no decoration. Popups always appear centred under the
    mouse, and don't accept keyboard input.
    """
    def __init__(self, app):
        Gtk.Window.__init__(self, type=Gtk.WindowType.POPUP)
        self.set_gravity(Gdk.Gravity.CENTER)
        self.set_position(Gtk.WindowPosition.MOUSE)
        self.app = app
        self.app.kbm.add_window(self)


class ChooserPopup (Gtk.Window):
    """A resizable popup window used for making fast choices

    Chooser popups can be used for fast selection of items
    from a list of alternatives.
    They normally appear under the mouse pointer,
    but can be popped up next to another on-screen widget
    to provide a menu-like response.

    The popup can be resized using its edges.
    To cancel and hide the popup without making a choice,
    move the pointer outside the window beyond a certain distance or
    click outside the window.

    Code using this class should also hide() the popup
    when the user has made a definite, complete choice
    from what's on offer.

    Popup choosers theoretically permit keyboard input as far as the WM
    is concerned, but eat most keypresses except those whose actions
    have been nominated to be dispatched via ``app.kbm``. As such,
    they're not suited for keyboard data entry, but are fine for
    clicking on brushes, colours etc.

    """

    ## Class constants

    MIN_WIDTH = 256
    MIN_HEIGHT = 256
    MAX_HEIGHT = 512
    MAX_WIDTH = 512

    LEAVE_SLACK = 64
    EDGE_SIZE = 12
    EDGE_CURSORS = {
        None: None,
        Gdk.WindowEdge.NORTH_EAST: Gdk.CursorType.TOP_RIGHT_CORNER,
        Gdk.WindowEdge.NORTH_WEST: Gdk.CursorType.TOP_LEFT_CORNER,
        Gdk.WindowEdge.SOUTH_EAST: Gdk.CursorType.BOTTOM_RIGHT_CORNER,
        Gdk.WindowEdge.SOUTH_WEST: Gdk.CursorType.BOTTOM_LEFT_CORNER,
        Gdk.WindowEdge.WEST: Gdk.CursorType.LEFT_SIDE,
        Gdk.WindowEdge.EAST: Gdk.CursorType.RIGHT_SIDE,
        Gdk.WindowEdge.SOUTH: Gdk.CursorType.BOTTOM_SIDE,
        Gdk.WindowEdge.NORTH: Gdk.CursorType.TOP_SIDE,
    }

    ## Method defs

    def __init__(self, app, actions, config_name):
        """Initialize.

        :param app: the main Application object.
        :param iterable actions: keyboard action names to pass through.
        :param str config_name: config prefix for saving window size.

        Use a simple "lowercase_with_underscores" name for the
        configuration key prefix.

        See also: `gui.keyboard.KeyboardManager.add_window()`.
        """
        # Superclass
        Gtk.Window.__init__(self, type=Gtk.WindowType.POPUP)
        self.set_modal(True)

        # Internal state
        self.app = app
        self._size = None  # last recorded size from any show()
        self._motion_handler_id = None
        self._prefs_size_key = "%s.window_size" % (config_name,)
        self._resize_info = None   # state during an edge resize
        self._outside_grab_active = False
        self._outside_cursor = Gdk.Cursor(Gdk.CursorType.LEFT_PTR)
        self._popup_info = None

        # Initial positioning
        self._initial_move_pos = None  # used when forcing a specific position
        self._corrected_pos = None  # used when keeping the widget on-screen

        # Resize cursors
        self._edge_cursors = {}
        for edge, cursor in self.EDGE_CURSORS.items():
            if cursor is not None:
                cursor = Gdk.Cursor(cursor)
            self._edge_cursors[edge] = cursor

        # Default size
        self.set_gravity(Gdk.Gravity.NORTH_WEST)
        default_size = (self.MIN_WIDTH, self.MIN_HEIGHT)
        w, h = app.preferences.get(self._prefs_size_key, default_size)
        w = clamp(int(w), self.MIN_WIDTH, self.MAX_WIDTH)
        h = clamp(int(h), self.MIN_HEIGHT, self.MAX_HEIGHT)
        default_size = (w, h)
        self.set_transient_for(app.drawWindow)
        self.set_default_size(*default_size)
        self.set_position(Gtk.WindowPosition.MOUSE)

        # Register with the keyboard manager, but only let certain actions be
        # driven from the keyboard.
        app.kbm.add_window(self, actions)

        # Event handlers
        self.connect("realize", self._realize_cb)
        self.connect("configure-event", self._configure_cb)
        self.connect("enter-notify-event", self._crossing_cb)
        self.connect("leave-notify-event", self._crossing_cb)
        self.connect("show", self._show_cb)
        self.connect("hide", self._hide_cb)
        self.connect("button-press-event", self._button_press_cb)
        self.connect("button-release-event", self._button_release_cb)
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.BUTTON_RELEASE_MASK)

        # Appearance
        self._frame = Gtk.Frame()
        self._frame.set_shadow_type(Gtk.ShadowType.OUT)
        self._align = Gtk.Alignment.new(0.5, 0.5, 1.0, 1.0)
        self._align.set_padding(self.EDGE_SIZE, self.EDGE_SIZE,
                                self.EDGE_SIZE, self.EDGE_SIZE)
        self._frame.add(self._align)
        Gtk.Window.add(self, self._frame)

    def _crossing_cb(self, widget, event):
        if self._resize_info:
            return
        if event.mode != Gdk.CrossingMode.NORMAL:
            return
        if event.detail != Gdk.NotifyType.NONLINEAR:
            return
        if event.get_window() is not self.get_window():
            return
        x, y, w, h = self._get_size()
        inside = (x <= event.x_root < x+w) and (y <= event.y_root < y+h)
        logger.debug("crossing: inside=%r", inside)
        # Grab the pointer if crossing from inside the popup to the
        # outside. Ungrab if doing the reverse.
        if inside:
            self._ungrab_pointer_outside(
                device = event.get_device(),
                time = event.time,
            )
        else:
            self._grab_pointer_outside(
                device = event.get_device(),
                time = event.time,
            )

    def _grab_pointer_outside(self, device, time):
        if self._outside_grab_active:
            logger.warning("grab: outside-popup grab already active: "
                           "regrabbing")
            self._ungrab_pointer_outside(device, time)
        event_mask = (
            Gdk.EventMask.POINTER_MOTION_MASK
            | Gdk.EventMask.ENTER_NOTIFY_MASK
            | Gdk.EventMask.LEAVE_NOTIFY_MASK
            | Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
        )
        cursor = self._outside_cursor
        grab_status = device.grab(
            window = self.get_window(),
            grab_ownership = Gdk.GrabOwnership.APPLICATION,
            owner_events = False,
            event_mask = Gdk.EventMask(event_mask),
            cursor = cursor,
            time_ = time,
        )
        if grab_status == Gdk.GrabStatus.SUCCESS:
            logger.debug("grab: acquired grab on %r successfully", device)
            self._outside_grab_active = True
        else:
            logger.warning("grab: failed to acquire grab on %r (status=%s)",
                           device, grab_status.value_nick)

    def _ungrab_pointer_outside(self, device, time):
        if not self._outside_grab_active:
            logger.debug("ungrab: outside-popup grab not active")
        device.ungrab(time_=time)
        logger.debug("ungrab: released grab on %r", device)
        self._outside_grab_active = False

    def _configure_cb(self, widget, event):
        """Internal: Update size and prefs when window is adjusted"""
        # Constrain window to fit on its current monitor, if possible.
        screen = event.get_screen()
        mon = screen.get_monitor_at_point(event.x, event.y)
        mon_geom = screen.get_monitor_geometry(mon)
        # Constrain width and height
        w = clamp(int(event.width), self.MIN_WIDTH, self.MAX_WIDTH)
        h = clamp(int(event.height), self.MIN_HEIGHT, self.MAX_HEIGHT)
        # Constrain position
        x, y = event.x, event.y
        if y+h > mon_geom.y + mon_geom.height:
            y = mon_geom.y + mon_geom.height - h
        if x+w > mon_geom.x + mon_geom.width:
            x = mon_geom.x + mon_geom.width - w
        if x < mon_geom.x:
            x = mon_geom.x
        if y < mon_geom.y:
            y = mon_geom.y
        event_size = (event.x, event.y, event.width, event.height)
        ex, ey, ew, eh = [int(c) for c in event_size]
        x, y, w, h = [int(c) for c in (x, y, w, h)]
        if not self._corrected_pos:
            if (x, y) != (ex, ey):
                GLib.idle_add(self.move, x, y)
            if (w, h) != (ew, eh):
                GLib.idle_add(self.resize, w, h)
            self._corrected_pos = True
        # Record size
        self._size = (x, y, w, h)
        self.app.preferences[self._prefs_size_key] = (w, h)

    def _get_size(self):
        if not self._size:
            # From time to time, popups presented in fullscreen don't
            # receive configure events. Why?
            win = self.get_window()
            x, y = win.get_position()
            w = win.get_width()
            h = win.get_height()
            x, y = self.get_position()
            self._size = (x, y, w, h)
        return self._size

    def _realize_cb(self, widget):
        gdk_window = self.get_window()
        gdk_window.set_type_hint(Gdk.WindowTypeHint.POPUP_MENU)

    def popup(self, widget=None, above=False, textwards=True, event=None):
        """Display, with an optional position relative to a widget

        :param widget: The widget defining the pop-up position
        :param above: If true, pop up above from `widget`
        :param textwards: If true, pop up in the text direction from `widget`
        :param event: the originating event

        """
        if not widget:
            self.set_position(Gtk.WindowPosition.MOUSE)
            self.set_gravity(Gdk.Gravity.NORTH_WEST)
        else:
            win = widget.get_window()
            x, y = win.get_origin()[1:]
            alloc = widget.get_allocation()
            style = widget.get_style_context()
            rtl = (style.get_direction() == Gtk.TextDirection.RTL)
            grav_table = {
                # (Above, rtl, textwards): Gravity
                (True, True, True): Gdk.Gravity.SOUTH_EAST,
                (True, False, False): Gdk.Gravity.SOUTH_EAST,
                (True, True, False): Gdk.Gravity.SOUTH_WEST,
                (True, False, True): Gdk.Gravity.SOUTH_WEST,
                (False, True, True): Gdk.Gravity.NORTH_EAST,
                (False, False, False): Gdk.Gravity.NORTH_EAST,
                (False, True, False): Gdk.Gravity.NORTH_WEST,
                (False, False, True): Gdk.Gravity.NORTH_WEST,
            }
            grav = grav_table.get((above, rtl, textwards),
                                  Gdk.Gravity.NORTH_WEST)
            if not above:
                y += alloc.height
            if (rtl and textwards) or ((not rtl) and (not textwards)):
                x += alloc.width
            self.set_position(Gtk.WindowPosition.NONE)
            self.set_gravity(grav)
            x = int(x)
            y = int(y)
            self._initial_move_pos = (x, y)
            if self._size:
                self._do_initial_move()
        popup_info = None
        if event:
            popup_info = (event.get_device(), event.time)
        self._popup_info = popup_info
        self.present()

    def _do_initial_move(self):
        x, y, w, h = self._get_size()
        x, y = self._initial_move_pos
        grav = self.get_gravity()
        if grav in [Gdk.Gravity.SOUTH_EAST, Gdk.Gravity.SOUTH_WEST]:
            y -= h
        if grav in [Gdk.Gravity.NORTH_EAST, Gdk.Gravity.SOUTH_EAST]:
            x -= w
        self.move(x, y)

    def _show_cb(self, widget):
        """Internal: show child widgets, grab, start the motion handler"""
        self._frame.show_all()
        if not self._motion_handler_id:
            h_id = self.connect("motion-notify-event", self._motion_cb)
            self._motion_handler_id = h_id
        if self._initial_move_pos:
            self._do_initial_move()
            self._initial_move_pos = None
            self.set_gravity(Gdk.Gravity.NORTH_WEST)
        # Grab the device used for the click
        # if popped up next to a widget - the cursor counts as outside
        # initially.
        if self._popup_info:
            device, time = self._popup_info
            self._grab_pointer_outside(
                device = device,
                time = time,
            )

    def _hide_cb(self, widget):
        """Internal: reset during-show state when the window is hidden"""
        if self._motion_handler_id is not None:
            self.disconnect(self._motion_handler_id)
        self._motion_handler_id = None
        self._corrected_pos = None
        self._initial_move_pos = None
        self._popup_info = None
        self._resize_info = None
        self._outside_grab_active = False

    def add(self, child):
        """Override: add() adds the child widget to an internal alignment"""
        return self._align.add(child)

    def remove(self, child):
        """Override: remove() removes the child from an internal alignment"""
        return self._align.remove(child)

    def _get_edge(self, px, py):
        """Internal: returns which window edge the pointer is pointing at"""
        size = self._get_size()
        if size is None:
            return None
        x, y, w, h = size
        s = self.EDGE_SIZE
        inside = (x <= px < x+w) and (y <= py < y+h)
        if not inside:
            return None
        north = east = south = west = False
        if px >= x and px <= x+s:
            west = True
        elif px <= x+w and px >= x+w-s:
            east = True
        if py >= y and py <= y+s:
            north = True
        elif py <= y+h and py >= y+h-s:
            south = True
        if north:
            if east:
                return Gdk.WindowEdge.NORTH_EAST
            elif west:
                return Gdk.WindowEdge.NORTH_WEST
            else:
                return Gdk.WindowEdge.NORTH
        elif south:
            if east:
                return Gdk.WindowEdge.SOUTH_EAST
            elif west:
                return Gdk.WindowEdge.SOUTH_WEST
            else:
                return Gdk.WindowEdge.SOUTH
        elif east:
            return Gdk.WindowEdge.EAST
        elif west:
            return Gdk.WindowEdge.WEST
        else:
            return None

    def _get_cursor(self, px, py):
        x, y, w, h = self._get_size()
        edge = self._get_edge(px, py)
        outside_window = px < x or py < y or px > x+w or py > y+h
        if outside_window:
            return self._outside_cursor
        else:
            return self._edge_cursors.get(edge, None)

    def _button_press_cb(self, widget, event):
        """Internal: starts resizing if the pointer is at the edge"""
        win = self.get_window()
        if not win:
            return False
        if event.button != 1:
            return False
        if not self.get_visible():
            return False

        rx, ry = event.x_root, event.y_root
        edge = self._get_edge(rx, ry)
        size = self._get_size()
        if edge is not None:
            self._resize_info = (rx, ry, size, edge)
            return True
        else:
            x, y, w, h = size
            inside = (x <= rx < x+w) and (y <= ry < y+h)
            if not inside:
                logger.debug("click outside detected: hiding popup")
                self.hide()
                return True
        return False

    def _button_release_cb(self, widget, event):
        """Internal: stops any active resize"""
        if event.button != 1:
            return False
        if not self.get_visible():
            return False
        if self._resize_info:
            self._resize_info = None
        # IDEA: re-grab the pointer here if the event is inside?
        return True

    def _motion_cb(self, widget, event):
        """Internal: handle motions: resizing, or leave checks"""

        # Ensure that at some point the user started inside a visible window
        if not self.get_visible():
            return
        win = self.get_window()
        if not win:
            return
        size = self._get_size()
        assert size is not None

        # Resizing
        px, py = event.x_root, event.y_root
        if self._resize_info:
            px0, py0, size0, edge0 = self._resize_info
            x0, y0, w0, h0 = size0
            dx, dy = (px-px0, py-py0)
            x, y = x0, y0
            w, h = w0, h0
            if edge0 in (Gdk.WindowEdge.NORTH_WEST, Gdk.WindowEdge.WEST,
                         Gdk.WindowEdge.SOUTH_WEST):
                x += dx
                w -= dx
            elif edge0 in (Gdk.WindowEdge.NORTH_EAST, Gdk.WindowEdge.EAST,
                           Gdk.WindowEdge.SOUTH_EAST):
                w += dx
            if edge0 in (Gdk.WindowEdge.NORTH_WEST, Gdk.WindowEdge.NORTH,
                         Gdk.WindowEdge.NORTH_EAST):
                y += dy
                h -= dy
            elif edge0 in (Gdk.WindowEdge.SOUTH_EAST, Gdk.WindowEdge.SOUTH,
                           Gdk.WindowEdge.SOUTH_WEST):
                h += dy

            # Apply constraints
            # One caveat: if the minimum size of the widgets themselves
            # is larger than the hardcoded MIN_* sizes here, moving the
            # top or left edges can look weird until the calculated w or h
            # butts up against the minimum: it effectively does a move.
            if w < self.MIN_WIDTH:
                if x != x0:
                    x -= self.MIN_WIDTH - w
                w = self.MIN_WIDTH
            if h < self.MIN_HEIGHT:
                if y != y0:
                    y -= self.MIN_HEIGHT - h
                h = self.MIN_HEIGHT

            # Move and/or resize
            # Can't use gdk_window_move_resize because that screws up
            # child widgets with GtkScrolledWindows.
            if (x, y) != (x0, y0):
                self.move(x, y)
            if (w, h) != (w0, h0):
                self.resize(w, h)

            return True

        x, y, w, h = size
        inside = (x <= px < x+w) and (y < py < y+h)

        # One oddity of the outside grab handling code is that it
        # doesn't always detect re-entry (in fact, it's cyclic; every
        # *other* crossing to the inside gets dropped). As a workaround,
        # if the grab is active and the cursor is *moving* inside,
        # release the grab so that the user can interact with the
        # window's contents.
        if inside and self._outside_grab_active:
            self._ungrab_pointer_outside(event.get_device(), event.time)

        cursor = self._get_cursor(px, py)
        win.set_cursor(cursor)

        if inside:
            return False

        if self._popup_info:
            return False

        any_button_mask = (
            Gdk.ModifierType.BUTTON1_MASK
            | Gdk.ModifierType.BUTTON2_MASK
            | Gdk.ModifierType.BUTTON3_MASK
            | Gdk.ModifierType.BUTTON4_MASK
            | Gdk.ModifierType.BUTTON5_MASK
        )

        if event.state & any_button_mask:
            return False

        # Moving outside the window more than a certain amount causes it
        # to close.
        s = self.LEAVE_SLACK
        outside_tolerance = px < x-s or py < y-s or px > x+w+s or py > y+h+s
        if outside_tolerance:
            self.hide()
            return True


# General window-related helper functions

def clear_focus(window):
    """Clear focus and any selection in widget with focus

    Immediately after calling this, there should be no widget with focus in the
    window, and if the previously focused widget had an active selection (such
    as a selection in a textbox, or an editable spinbutton), the selection will
    be cleared (the selection is removed, the content is not).
    """
    focus = window.get_focus()
    # If the current focused widget is an Entry, deselect any active selection.
    # This is mostly for aesthetics, but also to avoid widgets looking like
    # they have focus when they don't, due to selections being marked.
    if focus and isinstance(focus, Gtk.Entry):
        focus.select_region(0, 0)
    window.set_focus(None)


# Window positioning helper functions

def _final_rectangle(
        x, y, w, h, screen_w, screen_h, targ_geom, min_usable_size):
    """ Tries to create sensible (x, y, w, h) window pos/dim
    """
    # Generate a sensible, positive x and y position
    final_x, final_y, final_w, final_h = None, None, None, None
    if x is not None and y is not None:
        if x >= 0:
            final_x = x
        else:
            assert w is not None
            assert w > 0
            final_x = targ_geom.x + (targ_geom.w - w - abs(x))
        if y >= 0:
            final_y = y
        else:
            assert h is not None
            assert h > 0
            final_y = targ_geom.y + (targ_geom.h - h - abs(y))
        if final_x < 0 or final_x > screen_w - min_usable_size:
            final_x = None
        if final_y < 0 or final_y > screen_h - min_usable_size:
            final_y = None
    # and a sensible, positive width and height
    if w is not None and h is not None:
        final_w = w
        final_h = h
        if w < 0 or h < 0:
            if w < 0:
                if x is not None:
                    final_w = max(0, targ_geom.w - abs(x) - abs(w))
                else:
                    final_w = max(0, targ_geom.w - 2*abs(w))
            if h < 0:
                if x is not None:
                    final_h = max(0, targ_geom.h - abs(y) - abs(h))
                else:
                    final_h = max(0, targ_geom.h - 2*abs(h))
        if final_w > screen_w or final_w < min_usable_size:
            final_w = None
        if final_h > screen_h or final_h < min_usable_size:
            final_h = None

    return final_x, final_y, final_w, final_h


def set_initial_window_position(win, pos):
    """Set the position of a Gtk.Window, used during initial positioning.

    This is used both for restoring a saved window position, and for the
    application-wide defaults. The ``pos`` argument is a dict containing the
    following optional keys

        "w": <int>
        "h": <int>
            If positive, the size of the window.
            If negative, size is calculated based on the size of the
            monitor with the pointer on it, and x (or y) if given, e.g.

                width = mouse_mon_w -  abs(x) + abs(w)   # or (if no x)
                width = mouse_mon_w - (2 * abs(w))

            The same is true of calculated heights.

        "x": <int>
        "y": <int>
            If positive, the left/top of the window.
            If negative, the bottom/right of the window on the monitor
            with the pointer on it: you MUST provide a positive w and h
            if you do this.

    If the window's calculated top-left would place it offscreen, it will be
    placed in its default, window manager provided position. If its calculated
    size is larger than the screen, the window will be given its natural size
    instead.

    Returns the final, chosen (x, y) pair for forcing the window position on
    first map, or None if defaults are being used.

    """

    min_usable_size = 100

    # Where the mouse is right now - identifies the current monitor.
    ptr_x, ptr_y = 0, 0
    screen = win.get_screen()
    display = win.get_display()
    devmgr = display and display.get_device_manager() or None
    ptrdev = devmgr and devmgr.get_client_pointer() or None
    if ptrdev:
        ptr_screen, ptr_x, ptr_y = ptrdev.get_position()
        assert ptr_screen is screen, (
            "Screen containing core pointer != screen containing "
            "the window for positioning (%r != %r)" % (ptr_screen, screen)
        )
        logger.debug("Core pointer position from display: %r", (ptr_x, ptr_y))
    else:
        logger.warning(
            "Could not determine core pointer position from display. "
            "Using %r instead.",
            (ptr_x, ptr_y),
        )
    screen_w = screen.get_width()
    screen_h = screen.get_height()
    assert screen_w > min_usable_size
    assert screen_h > min_usable_size

    # The target area is ideally the current monitor.
    targ_mon_num = screen.get_monitor_at_point(ptr_x, ptr_y)
    targ_geom = _get_target_area_geometry(screen, targ_mon_num)

    # Positioning arguments
    x, y, w, h = (pos.get(k, None) for k in ("x", "y", "w", "h"))
    final_x, final_y, final_w, final_h = _final_rectangle(
        x, y, w, h, screen_w, screen_h, targ_geom, min_usable_size)

    # If the window is positioned, make sure it's on a monitor which still
    # exists. Users change display layouts...
    if None not in (final_x, final_y):
        onscreen = False
        for mon_num in xrange(screen.get_n_monitors()):
            targ_geom = _get_target_area_geometry(screen, mon_num)
            in_targ_geom = (
                final_x < (targ_geom.x + targ_geom.w)
                and final_y < (targ_geom.x + targ_geom.h)
                and final_x >= targ_geom.x
                and final_y >= targ_geom.y
            )
            if in_targ_geom:
                onscreen = True
                break
        if not onscreen:
            logger.warning("Calculated window position is offscreen; "
                           "ignoring %r" % ((final_x, final_y), ))
            final_x = None
            final_y = None

    # Attempt to set up with a geometry string first. Repeats the block below
    # really, but this helps smaller windows receive the right position in
    # xfwm (at least), possibly because the right window hints will be set.
    if None not in (final_w, final_h, final_x, final_y):
        geom_str = "%dx%d+%d+%d" % (final_w, final_h, final_x, final_y)
        win.connect("realize", lambda *a: win.parse_geometry(geom_str))

    # Set what we can now.
    if None not in (final_w, final_h):
        win.set_default_size(final_w, final_h)
    if None not in (final_x, final_y):
        win.move(final_x, final_y)
        return final_x, final_y

    return None


def _get_target_area_geometry(screen, mon_num):
    """Get a rect for putting windows in: normally based on monitor.

    :param Gdk.Screen screen: Target screen.
    :param int mon_num: Monitor number, e.g. that of the pointer.
    :returns: A hopefully usable target area.
    :rtype: Rect

    This function operates like gdk_screen_get_monitor_geometry(), but
    falls back to the screen geometry for cases when that returns NULL.
    It also returns a type which has (around GTK 3.18.x) fewer weird
    typelib issues with construction or use.

    Ref: https://github.com/mypaint/mypaint/issues/424
    Ref: https://github.com/mypaint/mypaint/issues/437

    """
    geom = None
    if mon_num >= 0:
        geom = screen.get_monitor_geometry(mon_num)
    if geom is not None:
        geom = Rect.new_from_gdk_rectangle(geom)
    else:
        logger.warning(
            "gdk_screen_get_monitor_geometry() returned NULL: "
            "using screen size instead as a fallback."
        )
        geom = Rect(0, 0, screen.get_width(), screen.get_height())
    return geom
