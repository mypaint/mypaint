# This file is part of MyPaint.
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Workspaces with a central canvas, sidebars and saved layouts.
"""

## Imports

import os
from warnings import warn
import logging
logger = logging.getLogger(__name__)

from gettext import gettext as _
import cairo
from gi.repository import GObject
from gi.repository import Gtk
from gi.repository import Gdk


## Exceptions

class ToolWidgetConstructError (Exception):
    """Errors raised during tool-widget construction.
    """
    pass


## Class defs

class Workspace (Gtk.VBox, Gtk.Buildable):
    """Widget housing a central canvas flanked by two sidebar toolstacks.

    Workspaces also manage zero or more floating ToolStacks, and can set the
    initial size and position of their own toplevel window.

    Instances of tool widget classes can be added and removed from the
    workspace programatically via their GType name.  They must support zero
    argument construction for this to work, and should support the following
    Python properties.  Defaults will be used if these properties aren't
    defined, but the defaults are unlikely to be useful.

    * ``tool_widget_icon_name``: the name of the icon to use.
    * ``tool_widget_title``: the title to display in the tooltip, and in
      floating window titles.
    * ``tool_widget_description``: the description string to show in the
      tooltip.

    The entire layout of a Workspace, including the toplevel window of the
    Workspace itself, can be dumped to and built from structures containing
    only simple Python types.  Widgets can be removed from the workspace by the
    user by clicking on a tab group close button within a ToolStack widget,
    moved around between stacks, or snapped out of stacks into new floating
    windows.

    Workspaces observe their toplevel window, and automatically hide their
    sidebars and floating windows in fullscreen. Auto-hidden elements are
    revealed temporarily when the pointer moves near them. Autohide can be
    toggled off or on, and the setting is retained in the layout definition.
    
    Workspaces can also manage the visibility of a header and a footer bar
    widget in the same manner: these bars are assumed to be packed above or
    below the Workspace respectively.

    """

    ## Class vars

    #: How near the pointer needs to be to a window edge or a hidden window to
    #: automatically reveal it when autohide is enabled in fullscreen.
    AUTOHIDE_REVEAL_BORDER = 12

    #: Time in milliseconds to wait before hiding UI elements when autohide is
    #: enabled in fullscreen.
    AUTOHIDE_TIMEOUT = 800

    #: Mask for all buttons.
    #: Used to prevent autohide reveals when a pointer button is pressed
    #: down. Prevents a possible source of distraction when the user is
    #: drawing.
    _ALL_BUTTONS_MASK = (
        Gdk.ModifierType.BUTTON1_MASK | Gdk.ModifierType.BUTTON2_MASK |
        Gdk.ModifierType.BUTTON3_MASK | Gdk.ModifierType.BUTTON4_MASK |
        Gdk.ModifierType.BUTTON5_MASK  )

    # Edges the pointer can bump: used for autohide reveals

    _EDGE_NONE = 0x00
    _EDGE_LEFT = 0x01
    _EDGE_RIGHT = 0x02
    _EDGE_TOP = 0x04
    _EDGE_BOTTOM = 0x08

    # GObject integration (type name, signals, properties)

    __gtype_name__ = 'MyPaintWorkspace'
    __gsignals__ = {
        "tool-widget-added": (GObject.SIGNAL_RUN_FIRST, None,
                              (Gtk.Widget, str)),
        "tool-widget-removed": (GObject.SIGNAL_RUN_FIRST, None,
                                (Gtk.Widget, str)),
        "floating-window-created": (GObject.SIGNAL_RUN_FIRST, None,
                                    (Gtk.Window,)),
        "floating-window-destroy": (GObject.SIGNAL_RUN_FIRST, None,
                                    (Gtk.Window,)),
        }
    """
    Signals
    -------

    * ``tool-widget-added``, called after a tool widget is added.
      Signature: ``callback(widget, gtype_name)``.
    * ``tool-widget-removed``, called after a tool widget is removed.
      Signature: ``callback(widget, gtype_name)``.
    * ``floating-window-created``, called after a new floating window is
      created either by the user snapping out a tool tab, or at startup.
      Signature: ``callback(widget)``.
    * ``floating-window-destroy``, called before a floating window is
      destroyed. Signature: ``callback(widget)``.

    """

    #: Title suffix property for floating windows.
    floating_window_title_suffix = GObject.property(
            type=str, flags=GObject.PARAM_READWRITE,
            nick='Floating window title suffix',
            blurb='The suffix to append to floating windows: typically a '
                  'hyphen followed by the application name.',
            default=None)

    #: Title separator property for floating windows.
    floating_window_title_separator = GObject.property(
            type=str, flags=GObject.PARAM_READWRITE,
            nick='Floating window title separator',
            blurb='String used to separate the names of tools in a '
                  'floating window. By default, a comma is used.',
            default=", ")

    #: Header bar widget, to be hidden when entering fullscreen mode. This
    #: widget should be packed externally to the workspace, and to its top.
    header_bar = GObject.property(
            type=Gtk.Widget, flags=GObject.PARAM_READWRITE,
            nick='Header bar widget',
            blurb="External Menubar/toolbar widget to be hidden when "
                  "entering fullscreen mode, and re-shown when leaving "
                  "it. The pointer position is also used for reveals and "
                  "hides in fullscreen.",
            default=None)

    #: Footer bar widget, to be hidden when entering fullscreen mode. This
    #: widget should be packed externally to the workspace, and to its bottom.
    footer_bar = GObject.property(
            type=Gtk.Widget, flags=GObject.PARAM_READWRITE,
            nick='Footer bar widget',
            blurb="External footer bar widget to be hidden when entering "
                  "fullscreen mode, and re-shown when leaving it. The "
                  "pointer position is also used for reveals and hides "
                  "in fullscreen.",
            default=None)



    def __init__(self):
        """Initializes, with a placeholder canvas widget and no tool widgets.
        """
        Gtk.VBox.__init__(self)
        # Sidebar stacks
        self._lstack = lstack = ToolStack()
        self._rstack = rstack = ToolStack()
        lscrolls = Gtk.ScrolledWindow()
        rscrolls = Gtk.ScrolledWindow()
        self._lscrolls = lscrolls
        self._rscrolls = rscrolls
        lscrolls.add_with_viewport(lstack)
        rscrolls.add_with_viewport(rstack)
        lscrolls.set_shadow_type(Gtk.ShadowType.NONE)
        rscrolls.set_shadow_type(Gtk.ShadowType.NONE)
        lscrolls.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        rscrolls.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._lpaned = lpaned = Gtk.HPaned()
        self._rpaned = rpaned = Gtk.HPaned()
        for stack, paned in [(lstack, lpaned), (rstack, rpaned)]:
            stack.workspace = self
            stack.connect("hide", self._sidebar_stack_hide_cb, paned)
        # Sidebar packing
        lpaned.pack1(lscrolls, resize=False, shrink=False)
        lpaned.pack2(rpaned, resize=True, shrink=False)
        rpaned.pack2(rscrolls, resize=False, shrink=False)
        self.pack_start(lpaned, True, True, 0)
        # Autohide
        self._autohide_enabled = True
        self._autohide_timeout = None
        # Window tracking
        self._floating = set()
        self._toplevel_pos = dict()
        self._save_toplevel_pos_timeout = None
        self._is_fullscreen = False
        self._is_maximized = False
        self._fs_event_handlers = []
        # Initial layout happens in several phases
        self._initial_layout = None
        self._complete_initial_layout_cb_id = None
        self.connect("realize", self._realize_cb)

    ## GtkBuildable implementation (pre-realize)


    def do_add_child(self, builder, child, type_):
        """Adds a child as the canvas: gtk_buildable_add_child() implementation
        """
        self.set_canvas(child)


    ## Setup from layout descriptions (pre-realize)


    def build_from_layout(self, layout):
        """Builds the workspace from a definition dict.

        :param layout: a layout definition
        :type layout: dict

        In order to have any effect, this must be called before the workspace
        widget is realized, but after it has been packed into its toplevel
        window. Keys and values in the dict are as follows:

        * position: an initial window position dict for the toplevel
          window. See `set_initial_window_position()`.
        * left_sidebar, right_sidebar: `ToolStack` definition lists.
          See `Toolstack.build_from_layout()`.
        * floating: a list of floating window definitions. Each element
          is a dict with the following keys:
          - contents: a `ToolStack` definition dict: see above.
          - position: an initial window position dict: see above.
        * autohide: whether autohide is enabled when fullscreening.
        * fullsceen: whether to start in fullscreen mode.
        * maximized: whether to start maximized.

        See also `get_layout()`.

        """
        toplevel_win = self.get_toplevel()
        assert toplevel_win is not None
        assert toplevel_win is not self
        assert not toplevel_win.get_visible()
        # Set initial position and fullscreen state
        toplevel_pos = layout.get("position", None)
        if toplevel_pos:
            set_initial_window_position(toplevel_win, toplevel_pos)
        if layout.get("fullscreen", False):
            GObject.idle_add(lambda *a: toplevel_win.fullscreen())
        if layout.get("maximized", False):
            GObject.idle_add(lambda *a: toplevel_win.maximize())
        toplevel_win.connect("window-state-event",
                             self._toplevel_window_state_event_cb)
        self.autohide_enabled = layout.get("autohide", True)
        self._initial_layout = layout


    def get_layout(self):
        """Returns a layout definition dict for the workspace.

        This should be called before the toplevel window is fully destroyed,
        or the dicts representing the tool stacks will be empty.

        """
        llayout = self._lstack.get_layout()
        rlayout = self._rstack.get_layout()
        float_layouts = [w.get_layout() for w in self._floating]
        return dict(left_sidebar=llayout, right_sidebar=rlayout,
                    floating=float_layouts, position=self._toplevel_pos,
                    autohide=self._autohide_enabled,
                    fullscreen=self._is_fullscreen,
                    maximized=self._is_maximized)


    ## Initial layout (pre/post-realize)


    def _realize_cb(self, widget):
        """Kick off the deferred layout code when the widget is realized.
        """
        self._start_initial_layout()


    def _start_initial_layout(self):
        """Layout: all that can be done before the toplevel is positioned.
        """

        # Set up monitoring of the toplevel's size changes.
        toplevel = self.get_toplevel()
        toplevel.connect("configure-event", self._toplevel_configure_cb)

        # Do the initial layout 
        layout = self._initial_layout
        if layout is None:
            return
        if layout.get("fullscreen", False):
            complete_state = Gdk.WindowState.FULLSCREEN
        elif layout.get("mazimize", False):
            complete_state = Gdk.WindowState.MAXIMIZE
        else:
            complete_state = None
        llayout = layout.get("left_sidebar", {})
        rlayout = layout.get("right_sidebar", {})
        self._lstack.build_from_layout(llayout)
        self._rstack.build_from_layout(rlayout)
        # Floating windows
        for flayout in layout.get("floating", []):
            win = ToolStackWindow()
            self.emit("floating-window-created", win)
            win.stack.workspace = self
            win.build_from_layout(flayout)
            self._floating.add(win)
        # Reveal floating windows only after floating-window-created handlers
        # have had a chance to run.
        for win in self._floating:
            GObject.idle_add(win.show_all)
        # Arrange for part 2 to be run
        toplevel = self.get_toplevel()
        if not complete_state:
            # Nothing too fancy; we're hopefully going to be mapped with
            # the right initial size for the sidebars etc.
            assert not self.get_mapped()
            cb = self._complete_initial_layout_map_cb
            cb_id = self.connect("map", cb)
        else:
            # If we're about to fullscreen or maximize, wait for that state.
            # Otherwise the paned positions won't be right for the window.
            cb = self._complete_layout_toplevel_window_state_cb
            cb_id = toplevel.connect("window-state-event", cb, complete_state)
            # But time out just case the state is never reached. Window
            # managers can be fickle.
            timeout = self._complete_layout_toplevel_window_state_timeout_cb
            GObject.timeout_add_seconds(3, timeout)
        self._complete_initial_layout_cb_id = cb_id
        # Give the toolstacks a chance to do something here too.
        for stack in self._get_tool_stacks():
            stack._start_initial_layout()


    def _complete_initial_layout_map_cb(self, widget):
        logger.debug("Completing layout (mapped)")
        GObject.idle_add(self._complete_initial_layout)
        widget.disconnect(self._complete_initial_layout_cb_id)
        self._complete_initial_layout_cb_id = None


    def _complete_layout_toplevel_window_state_cb(self, toplevel, event,
                                                  expected_state):
        # Wait for the window to transition to the right initial state
        if event.changed_mask & expected_state:
            if event.new_window_state & expected_state:
                logger.debug("Completing layout (toplevel state-transition)")
                GObject.idle_add(self._complete_initial_layout)
                toplevel.disconnect(self._complete_initial_layout_cb_id)
                self._complete_initial_layout_cb_id = None


    def _complete_layout_toplevel_window_state_timeout_cb(self):
        # Too long waiting for the expected state transition...
        if self._complete_initial_layout_cb_id is None:
            return False
        toplevel = self.get_toplevel()
        toplevel.disconnect(self._complete_initial_layout_cb_id)
        self._complete_initial_layout_cb_id = None
        logger.debug("Completing layout (expected toplevel state-transition "
                     "didn't happen within the timeout)")
        GObject.idle_add(self._complete_initial_layout)
        return False


    def _complete_initial_layout(self):
        """Finish initial layout; called after toplevel win is positioned.
        """
        assert self.get_realized()
        # Restore saved widths for the sidebar
        layout = self._initial_layout
        if layout is not None:
            left_width = layout.get("left_sidebar", {}).get("w", None)
            if left_width is not None:
                self.set_left_sidebar_width(left_width)
            right_width = layout.get("right_sidebar", {}).get("w", None)
            if right_width is not None:
                self.set_right_sidebar_width(right_width)
        # Sidebar stacks are initially shown, but their contents are not
        # because building from layout was deferred. Hide empties and issue
        # show_all()s.
        if self._lstack.is_empty():
            self._lstack.hide()
        else:
            self._lstack.show_all()
        if self._rstack.is_empty():
            self._rstack.hide()
        else:
            self._rstack.show_all()
        # Toolstacks are responsible for their groups' positions
        for stack in self._get_tool_stacks():
            stack._complete_initial_layout()


    ## Canvas widget


    def set_canvas(self, widget):
        """Canvas widget (setter)
        """
        assert self.get_canvas() is None
        self._rpaned.pack1(widget, resize=True, shrink=False)


    def get_canvas(self):
        """Canvas widget (getter)
        """
        return self._rpaned.get_child1()


    ## Tool widgets


    def add_tool_widget(self, gtype_name):
        """Adds a new tool widget by GType name

        The widget will be instantiated with no args, and added to the first
        stack available. Existing floating windows will be favoured over the
        sidebars; if there are no stacks visible, a sidebar will be made
        visible to receive the new widget.

        """
        logger.debug("Adding a %r to the most favorable stack", gtype_name)
        stacks = list(self._get_tool_stacks())
        assert stacks
        stack = None
        widget = None
        maxpages = 1
        while not widget:
            for stack in stacks:
                try:
                    widget = stack.add_tool_widget(gtype_name, maxnotebooks=3,
                                                   maxpages=maxpages)
                except ToolWidgetConstructError as ex:
                    warn("add_tool_widget: %s" % (ex.message,),
                         RuntimeWarning)
                    return None
                if widget:
                    break
            maxpages += 1
            if maxpages > 99:
                logger.warning("All stacks appear to have 100+ pages. "
                               "Probably untrue, but giving up anyway.")
                break
        if not widget:
            logger.error("Cant find space for a %r in any stack", gtype_name)
            return None
        assert stack.has_tool_widget(gtype_name)
        stack_toplevel = stack.get_toplevel()
        if stack_toplevel not in self._floating:
            scrolls = stack.get_parent().get_parent()
            scrolls.show_all()
        else:
            if self._is_fullscreen and self.autohide_enabled:
                for floating in self._floating:
                    floating.show_all()
        assert self.has_tool_widget(gtype_name)
        logger.debug("Added %r successfully", gtype_name)
        return widget


    def tool_widget_added(self, page, gtype_name):
        """Emits the "tool-widget-added" signal
        """
        # The signal is emitted in an idle callback to give the initial setup a
        # chance to complete.
        GObject.idle_add(self.emit, "tool-widget-added", page, gtype_name)


    def remove_tool_widget(self, gtype_name):
        """Removes a new tool widget by GType name

        If there is more than one instance of the named widget in the
        hierarchy, only the first one will be removed.

        """
        for stack in self._get_tool_stacks():
            removed = stack.remove_tool_widget(gtype_name)
            if removed:
                return removed
        return None


    def tool_widget_removed(self, page, gtype_name):
        """Emits the "tool-widget-removed" signal.
        """
        self.emit("tool-widget-removed", page, gtype_name)


    def has_tool_widget(self, gtype_name):
        """Tests whether a tool widget is in the workspace.
        """
        widget = self.get_tool_widget(gtype_name)
        if widget:
            return True
        return False


    def get_tool_widget(self, gtype_name):
        """Finds the first matching tool widget.
        """
        for stack in self._get_tool_stacks():
            widget = stack.get_tool_widget(gtype_name)
            if widget:
                return widget
        return None


    ## Sidebar toolstack width


    def set_right_sidebar_width(self, width):
        """Sets the width of the right sidebar toolstack
        """
        if self._rstack.is_empty():
            return
        width = max(width, 100)
        handle_size = GObject.Value()
        handle_size.init(int)
        self._rpaned.style_get_property("handle-size", handle_size)
        position = self._rpaned.get_allocated_width()
        position -= width
        position -= handle_size.get_int()
        self._rpaned.set_position(position)


    def set_left_sidebar_width(self, width):
        """Sets the width of the left sidebar toolstack
        """
        if self._lstack.is_empty():
            return
        width = max(width, 100)
        self._lpaned.set_position(width)


    ## Position saving (toplevel window)


    def _toplevel_configure_cb(self, toplevel, event):
        """Record the toplevel window's position ("configure-event" callback)
        """
        # Avoid saving fullscreen positions. The timeout is a bit of hack, but
        # it's necessary because the state change event and the configure event
        # when fullscreening don't have a sensible order.
        w, h = event.width, event.height
        srcid = self._save_toplevel_pos_timeout
        if srcid:
            GObject.source_remove(srcid)
        srcid = GObject.timeout_add(250, self._save_toplevel_pos_timeout_cb,
                                    w, h)
        self._save_toplevel_pos_timeout = srcid


    def _save_toplevel_pos_timeout_cb(self, w, h):
        """Toplevel window position recording (post-"configure-event" oneshot)
        
        Saves the (x,y) from the frame and the (w,h) from the configure event:
        the combination can be used as an initial size and position next time.

        """
        self._save_toplevel_pos_timeout = None
        if self._is_fullscreen or self._is_maximized:
            return False
        toplevel = self.get_toplevel()
        gdk_win = toplevel.get_window()
        extents = gdk_win.get_frame_extents()
        x = max(0, extents.x)
        y = max(0, extents.y)
        pos = dict(x=x, y=y, w=w, h=h)
        self._toplevel_pos = pos
        return False


    ## Toolstack order for searching, tool insertion etc.


    def _get_tool_stacks(self):
        """Yields all known ToolStacks, in floating-first order.
        """
        for win in self._floating:
            yield win.stack
        yield self._rstack
        yield self._lstack


    ## Tool widget tab dragging (event callbacks)


    def _tool_tab_drag_begin_cb(self):
        """Shows all possible drag targets at the start of a tool tab drag

        Ensures that all the known toolstacks are visible to receive the drag,
        even those which are empty or hidden due to fullscreening. Called by
        stack notebooks in this workspace when tab drags start.

        """
        # First cancel any pending hides.
        if self._is_fullscreen:
            self._cancel_autohide_timeout()
        # Ensure the left and right stacks are visible at the beginning of
        # a tab drag even if empty so that the user can drop a tab there.
        for stack in (self._lstack, self._rstack):
            scrolls = stack.get_parent().get_parent()
            empty = stack.is_empty()
            visible = stack.get_visible() and scrolls.get_visible()
            if (empty and not visible) or (not empty and not visible):
                scrolls = stack.get_parent().get_parent()
                scrolls.show_all()


    def _tool_tab_drag_end_cb(self):
        """Hides empty toolstacks at the end of a tool tab drag

        Called by stack notebooks in this workspace when tab drags finish.

        """
        for stack in (self._lstack, self._rstack):
            scrolls = stack.get_parent().get_parent()
            empty = stack.is_empty()
            visible = stack.get_visible() and scrolls.get_visible()
            if empty and visible:
                stack.hide()


    def _sidebar_stack_hide_cb(self, stack, paned):
        """Resets sidebar sizes when they're emptied (sidebar "hide" callback)

        If the hide is due to the sidebar stack having been emptied out,
        resetting the size means that it'll show at its placeholder's size when
        it's next shown by the drag-start handler: this makes a narrower but
        less intrusive target for the drag.

        """
        if stack.is_empty():
            paned.set_position(-1)
        # Hide the parent GtkScrolledWindow too, allowing the paned to not
        # show the sidebar pane at all - GtkScrolledWindows have an allocation
        # even if they're empty, assuming they have scrollbars.
        scrolls = stack.get_parent().get_parent()
        scrolls.hide()


    ## Fullscreen (event callbacks)


    def _toplevel_window_state_event_cb(self, widget, event):
        # Handle transitions between fullscreen and windowed.
        if event.changed_mask & Gdk.WindowState.FULLSCREEN:
            fullscreen = event.new_window_state & Gdk.WindowState.FULLSCREEN
            if fullscreen:
                if self.autohide_enabled:
                    self._connect_autohide_events()
                    self._start_autohide_timeout()
                # Showing the floating windows makes an initial fullscreen
                # look a little nicer.
                for floating in self._floating:
                    floating.show_all()
            else:
                self._disconnect_autohide_events()
                self._show_autohide_widgets()
            self._is_fullscreen = bool(fullscreen)
        if event.changed_mask & Gdk.WindowState.MAXIMIZED:
            maximized = event.new_window_state & Gdk.WindowState.MAXIMIZED
            self._is_maximized = bool(maximized)


    ## Autohide flag


    def get_autohide_enabled(self):
        """Auto-hide is enabled in fullscreen (getter)
        """
        return self._autohide_enabled


    def set_autohide_enabled(self, autohide_enabled):
        """Auto-hide is enabled in fullscreen (setter)
        """
        if self._is_fullscreen:
            if autohide_enabled:
                self._connect_autohide_events()
                self._hide_autohide_widgets()
            else:
                self._disconnect_autohide_events()
                self._show_autohide_widgets()
        self._autohide_enabled = bool(autohide_enabled)


    autohide_enabled = property(get_autohide_enabled, set_autohide_enabled)


    def _hide_autohide_widgets(self):
        """Hides all auto-hiding widgets immediately.
        """
        if not self._is_fullscreen:
            return
        self._cancel_autohide_timeout()
        for widget in self._get_autohide_widgets():
            if widget.get_visible():
                widget.hide()


    def _show_autohide_widgets(self):
        """Shows all auto-hiding widgets immediately.
        """
        self._cancel_autohide_timeout()
        for w in self._get_autohide_widgets():
            w.show_all()


    def _get_autohide_widgets(self):
        """List of autohide widgets

        Returns a list of the widgets which should be revealed by edge bumping,
        rollovers, or hidden by the timeout.

        """
        widgets = []
        for stack in [self._lstack, self._rstack]:
            if not stack.is_empty():
                scrolls = stack.get_parent().get_parent()
                widgets.append(scrolls)
        widgets.extend(list(self._floating))
        for bar in [self.header_bar, self.footer_bar]:
            if bar:
                widgets.append(bar)
        return widgets


    ## Autohide mode: auto-hide timer


    def _start_autohide_timeout(self):
        # Hide the UI after a brief period of inactivity.
        self._cancel_autohide_timeout()
        srcid = GObject.timeout_add(self.AUTOHIDE_TIMEOUT,
                                    self._autohide_timeout_cb)
        self._autohide_timeout = srcid


    def _cancel_autohide_timeout(self):
        # Cancel any pending hide.
        if not self._autohide_timeout:
            return
        GObject.source_remove(self._autohide_timeout)
        self._autohide_timeout = None



    def _autohide_timeout_cb(self):
        self._hide_autohide_widgets()
        return False


    ## Autohide mode: event handling on the canvas widget


    def _connect_autohide_events(self):
        # Start listening for autohide events.
        if self._fs_event_handlers:
            return
        evwidget = self.get_canvas()
        if not evwidget:
            return
        mask = (Gdk.EventMask.POINTER_MOTION_HINT_MASK |
                Gdk.EventMask.POINTER_MOTION_MASK |
                Gdk.EventMask.LEAVE_NOTIFY_MASK |
                Gdk.EventMask.ENTER_NOTIFY_MASK )
        evwidget.add_events(mask)
        handlers = [("motion-notify-event", self._fs_motion_cb),
                    ("leave-notify-event", self._fs_leave_cb),
                    ("enter-notify-event", self._fs_enter_cb)]
        for event_name, handler_callback in handlers:
            handler_id = evwidget.connect(event_name, handler_callback)
            self._fs_event_handlers.append((evwidget, handler_id))


    def _disconnect_autohide_events(self):
        # Stop listening for autohide events.
        for evwidget, handler_id in self._fs_event_handlers:
            evwidget.disconnect(handler_id)
        self._fs_event_handlers = []


    def _fs_leave_cb(self, widget, event):
        # Handles leaving the canvas in fullscreen.
        # Perhaps the user is using a sidebar. Leave it open so they can.
        self._cancel_autohide_timeout()
        return False


    def _fs_enter_cb(self, widget, event):
        # Handles entering the canvas in fullscreen.
        # If we're safely in the middle, the autohide timer can begin now.
        edges = self._get_bumped_edges(widget, event)
        if not edges:
            self._start_autohide_timeout()
        return False


    def _fs_motion_cb(self, widget, event):
        # Handles edge bumping and other rollovers in fullscreen mode.
        assert self._is_fullscreen
        # Firstly, if the user appears to be drawing, be as stable as we can.
        if event.state & self._ALL_BUTTONS_MASK:
            self._cancel_autohide_timeout()
            return False
        # Floating window rollovers
        show_floating = False
        for win in self._floating:
            if win.get_visible():
                continue
            x, y = event.x_root, event.y_root
            b = self.AUTOHIDE_REVEAL_BORDER
            if win.contains_point(x, y, b=b):
                show_floating = True
                break
        if show_floating:
            for win in self._floating:
                win.show_all()
        # Edge bumping
        # Bump the mouse into the edge of the screen to get back the stuff
        # that was hidden there, similar to media players etc.
        edges = self._get_bumped_edges(widget, event)
        if not edges:
            self._start_autohide_timeout()
            return False
        if edges & self._EDGE_TOP and self.header_bar:
            self.header_bar.show_all()
        if edges & self._EDGE_BOTTOM and self.footer_bar:
            self.footer_bar.show_all()
        if edges & self._EDGE_LEFT and not self._lstack.is_empty():
            self._lscrolls.show_all()
        if edges & self._EDGE_RIGHT and not self._rstack.is_empty():
            self._rscrolls.show_all()


    @classmethod
    def _get_bumped_edges(cls, widget, event):
        # Returns a bitmask of the edges bumped by the pointer.
        alloc = widget.get_allocation()
        w, h = alloc.width, alloc.height
        x, y = event.x, event.y
        b = cls.AUTOHIDE_REVEAL_BORDER
        if not (x < b or x > w-b or y < b or y > h-b):
            return cls._EDGE_NONE
        edges = cls._EDGE_NONE
        if y < b or (y < 5*b and (x < b or x > w-b)):
            edges |= cls._EDGE_TOP
        if y > h - b:
            edges |= cls._EDGE_BOTTOM
        if x < b:
            edges |= cls._EDGE_LEFT
        if x > w - b:
            edges |= cls._EDGE_RIGHT
        return edges



class ToolStack (Gtk.EventBox):
    """Vertical stack of tool widget groups.
    
    The layout has movable dividers between groups of tool widgets, and an
    empty group on the end which accepts tabs dragged to it. The groups are
    implmented as `Gtk.Notebook`s, but that interface is not exposed.
    ToolStacks are built up from layout definitions represented by simple
    types: see `Workspace` and `build_from_layout()` for details.

    """

    ## Behavioural constants

    RESIZE_STICKINESS = 20


    ## GObject integration (type name, properties)


    __gtype_name__ = 'MyPaintToolStack'


    workspace = GObject.Property(
          type=Workspace, flags=GObject.PARAM_READWRITE,
          nick='Workspace',
          blurb='The central Workspace object, used to coordinate drags',
          default=None)


    ## Internal classes: Paned/Notebook tree elements

    class _Paned (Gtk.VPaned):
        """GtkVPaned specialization acting as an intermediate note in a tree.
        """

        ## Construction

        def __init__(self, toolstack, placeholder):
            """Initialize, replacing and splitting an existing placeholder.

            :param toolstack: ancestor tool stack object
            :type toolstack: ToolStack
            :param placeholder: empty placeholder notebook, must be packed as
              the child2 of another ToolStack._Paned, or as the first child of
              a ToolStack.
            :type placeholder: GtkNotebook

            The old placeholder will be removed from its parent, and re-packed as
            the child1 of the new paned. A new placeholder is created as the new
            paned's child2. The new paned is then packed to replace the old
            placeholder in its former parent.

            """
            super(ToolStack._Paned, self).__init__()
            self._toolstack = toolstack
            parent = placeholder.get_parent()
            assert parent is not None, "'placeholder' must have a parent"
            self.set_border_width(0)
            if isinstance(parent, Gtk.Paned):
                assert placeholder is not parent.get_child1()
                assert placeholder     is parent.get_child2()
                parent.remove(placeholder)
                parent.pack2_subpaned(self)
            else:
                assert isinstance(parent, ToolStack)
                assert parent is toolstack
                parent.remove(placeholder)
                parent.add(self)
            new_placeholder = ToolStack._Notebook(self._toolstack)
            self.pack1_tool_widget_notebook(placeholder)
            self.pack2_placeholder_notebook(new_placeholder)
            if parent.get_visible():
                self.show_all()
                self.queue_resize()


        ## Custom widget packing


        def pack1_tool_widget_notebook(self, notebook):
            """Pack a notebook indended for tool widgets as child1.
            """
            assert isinstance(notebook, ToolStack._Notebook)
            self.pack1(notebook, False, False)


        def pack2_placeholder_notebook(self, notebook):
            """Pack a notebook intended as a placeholder into child2.
            """
            assert isinstance(notebook, ToolStack._Notebook)
            self.pack2(notebook, True, False)


        def pack2_subpaned(self, paned):
            """Pack a subpaned into child2.
            """
            assert isinstance(paned, ToolStack._Paned)
            self.pack2(paned, True, False)

    ## Notebook

    class _Notebook (Gtk.Notebook):
        """Tabbed notebook containng a tool widget group.
        """

        ## Behavioural constants

        NOTEBOOK_GROUP_NAME = 'mypaint-workspace-layout-group'
        PLACEHOLDER_HEIGHT = 8
        PLACEHOLDER_WIDTH = 16
        CLOSE_BUTTON_ICON_SIZE = Gtk.IconSize.MENU
        TAB_ICON_SIZE = Gtk.IconSize.LARGE_TOOLBAR
        TAB_TOOLTIP_ICON_SIZE = Gtk.IconSize.DIALOG


        ## Construction

        def __init__(self, toolstack):
            """Initialise, with an ancestor ToolStack.
            """
            super(ToolStack._Notebook, self).__init__()
            self._toolstack = toolstack
            assert self._toolstack is not None
            self.set_group_name(self.NOTEBOOK_GROUP_NAME)
            self.connect("create-window", self._create_window_cb)
            self.connect("page-added", self._page_added_cb)
            self.connect("page-removed", self._page_removed_cb)

            self.connect_after("drag-begin", self._drag_begin_cb)
            self.connect_after("drag-end", self._drag_end_cb)
            self.set_scrollable(True)

            btn = Gtk.Button()
            self._close_button = btn
            btn.set_tooltip_text(_("Close current tab"))
            img = Gtk.Image()
            img.set_from_stock(Gtk.STOCK_CLOSE, self.CLOSE_BUTTON_ICON_SIZE)
            btn.add(img)
            btn.set_relief(Gtk.ReliefStyle.NONE)
            self.set_action_widget(btn, Gtk.PackType.END)
            self.connect("show", lambda *a: self._close_button.show_all())
            btn.connect("clicked", self._close_button_clicked_cb)


        ## Tool widget pages


        def append_tool_widget_page(self, tool_widget):
            """Appends a tool widget as a new page/tab.
            """
            page = Gtk.Frame()
            page.set_shadow_type(Gtk.ShadowType.NONE)
            page.add(tool_widget)
            assert isinstance(page, Gtk.Frame)
            label = self._make_tab_label(tool_widget)
            self.append_page(page, label)
            self.set_tab_reorderable(page, True)
            self.set_tab_detachable(page, True)
            if self.get_visible():
                page.show_all()
            self.set_current_page(-1)
            return page


        ## ToolStack structure: event callbacks


        def _page_added_cb(self, notebook, child, page_num):
            GObject.idle_add(self._toolstack._update_structure)


        def _page_removed_cb(self, notebook, child, page_num):
            GObject.idle_add(self._toolstack._update_structure)


        ## ToolStack structure: utility methods


        def split_former_placeholder(self):
            """Splits the space used by a placeholder after a tab drag into it.

            After the placeholder has a tab dragged into it, it can no longer fill
            the placeholder's role. This method creates a new empty placeholder
            after it in the stack, and updates the hierarchy appropriately. It
            also tries to retain the dragged-in tab's page's size as much as
            possible by setting paned divider positions appropriately.

            """
            # Bail if not a former placeholder
            assert self.get_n_pages() > 0
            toolstack = self._toolstack
            toolstack_was_empty = self.get_parent() is toolstack
            assert toolstack_was_empty or self is self.get_parent().get_child2()
            # Reparenting dance
            parent_paned = ToolStack._Paned(toolstack, self)
            assert self is parent_paned.get_child1()
            new_placeholder = parent_paned.get_child2()
            # Set the vpaneds surrounding the widget to the dragged page's size
            page = self.get_nth_page(0)
            try:
                w, h = page.__prev_size
            except AttributeError:
                return new_placeholder
            parent_paned.set_position(h)
            # Sidebars adopt the size of the widget dragged there if this
            # placeholder used to be the only thing occupying them.
            if toolstack_was_empty:
                workspace = toolstack.workspace
                if toolstack is workspace._lstack:
                    workspace.set_left_sidebar_width(w)
                elif toolstack is workspace._rstack:
                    workspace.set_right_sidebar_width(w)
            return new_placeholder


        ## Close-current-tab button


        def _close_button_clicked_cb(self, button):
            """Remove the current page (close button "clicked" event callback)
            """
            page_num = self.get_current_page()
            page = self.get_nth_page(page_num)
            tool_widget = page.get_child()
            gtype_name = tool_widget.__gtype_name__
            self.remove_page(page_num)
            self._toolstack.workspace.tool_widget_removed(page, gtype_name)


        ## Dragging tabs


        def _drag_begin_cb(self, nb, *a):
            # Record the notebook's size in the page; this will be recreated
            # if a valid drop happens into a fresh ToolStackWindow or into a
            # placeholder notebook.
            alloc = self.get_allocation()
            page_num = self.get_current_page()
            page = self.get_nth_page(page_num)
            page.__prev_size = (alloc.width, alloc.height)
            # Notify the workspace: causes empty sidebars to show.
            self._toolstack.workspace._tool_tab_drag_begin_cb()


        def _drag_end_cb(self, nb, *a):
            # Notify the workspace that dragging has finished. Causes empty
            # sidebars to hide again.
            self._toolstack.workspace._tool_tab_drag_end_cb()


        def _create_window_cb(self, notebook, page, x, y):
            # Dragging into empty space creates a new stack in a new window,
            # and stashes the page there.
            win = ToolStackWindow()
            self._toolstack.workspace.emit('floating-window-created', win)
            win.stack.workspace = self._toolstack.workspace
            self.remove(page)
            w, h = page.__prev_size
            new_nb = win.stack._get_first_notebook()
            tool_widget = page.get_child()
            page.remove(tool_widget)
            new_nb.append_tool_widget_page(tool_widget)
            new_placeholder = win.stack._append_new_placeholder(new_nb)
            new_paned = new_placeholder.get_parent()
            new_paned.set_position(h)
            # Initial position. Hopefully this will work.
            win.move(x, y)
            win.set_default_size(w, h)
            win.show_all()


        ## Tab labels


        @classmethod
        def _make_tab_label(cls, tool_widget):
            """Creates and returns a tab label for a tool widget.
            """
            try:
                icon_name = tool_widget.tool_widget_icon_name
            except AttributeError:
                icon_name = 'missing-image'

            title = _tool_widget_get_title(tool_widget)

            try:
                desc = tool_widget.tool_widget_description
            except AttributeError:
                desc = '(no description)'

            label = Gtk.Image()
            label.set_from_icon_name(icon_name, cls.TAB_ICON_SIZE)
            label.connect("query-tooltip", cls._tab_label_tooltip_query_cb,
                          title, desc, icon_name)
            label.set_property("has-tooltip", True)
            return label


        @classmethod
        def _tab_label_tooltip_query_cb(cls, widget, x, y, kbd, tooltip,
                                        title, desc, icon_name):
            """The query-tooltip routine for tool widgets.
            """
            tooltip.set_icon_from_icon_name(icon_name, cls.TAB_TOOLTIP_ICON_SIZE)
            markup = "<b>%s</b>\n%s" % (title, desc)
            tooltip.set_markup(markup)
            return True

        ## XXX Dead code? XXX


        def _first_alloc_cb(self, widget, alloc):
            # Normally, if child widgets declare a real minimum size then in a
            # structure like this they'll be allocated their minimum size even when
            # there's enough space to give them their natural size. As a
            # workaround, set the bar position on the very first size-allocate
            # event to the best compromise we can calculate.
            self.disconnect(self._first_alloc_id)
            self._first_alloc_id = None

            # Child natural and minimum heights.
            c1 = self.get_child1()
            c2 = self.get_child2()
            if not (c1 and c2):
                return
            c1min, c1nat = c1.get_preferred_height_for_width(alloc.width)
            c2min, c2nat = c2.get_preferred_height_for_width(alloc.width)

            # We don't have a real bar height yet, so fudge it.
            # Not too bad really: in effect this becomes slack in the system.
            bar_height = 25

            # Strategy here is to try and give one child widget its natural size
            # first, slightly favouring the first (top) child.
            # We could be more egalitarian by inspecting the deep structure.
            if c1nat + c2min <= alloc.height - bar_height:
                self.set_position(c1nat)
            elif c1min + c2nat <= alloc.height - bar_height:
                self.set_position(alloc.height - c2nat - bar_height)
            elif c1min + c2min <= alloc.height - bar_height:
                self.set_position(alloc.height - c2min - bar_height)
            else:
                self.set_position(-1)

    ## Construction


    def __init__(self):
        """Constructs a new stack with a single placeholder group.
        """
        Gtk.EventBox.__init__(self)
        self.add(self._make_notebook())
        self.connect("size-allocate", self._size_alloc_cb)
        self.__initial_paned_positions = []


    ## Setup from layout descriptions (pre-realize)


    def build_from_layout(self, desc, init_sizes_state=None):
        """Loads groups and pages from a layout description.

        :param desc: stack definition
        :type desc: dict
        :param init_sizes_state: toplevel window state transition on
            which to set the group dividers' initial positions. If left
            unset, set the sizes immediately.
        :type init_sizes_state: Gdk.WindowState
        
        The `desc` parameter has the following keys and values:

        * w: integer width (ignored here)
        * h: integer height (ignored here)
        * groups: list of group defintions - see below

        Width and height may be of relevance to the parent widget, but are not
        consumed by this method. `get_layout()` writes them, however.  Each
        group definition is a dict with the following keys and values.

        * tools: a list of tool defintions - see below
        * h: integer height: used here to set the height of the group
        * w: integer width (ignored here)

        Each tool definition is a tuple of the form (GTYPENAME,*CONSTRUCTARGS).
        GTYPENAME is a string containing a GType name which is used for finding
        and constructing the tool instance. CONSTRUCTARGS is currently ignored.
 
        """
        next_nb = self._get_first_notebook()
        self._initial_paned_positions = []
        for group_desc in desc.get("groups", []):
            next_nb.get_n_pages() == 0
            tool_descs = group_desc.get("tools", [])
            if not tool_descs:
                continue
            nb = next_nb
            next_nb = self._append_new_placeholder(nb)
            for tool_desc in tool_descs:
                gtype_name = tool_desc[0]
                try:
                    tool_widget = self._tool_widget_new(gtype_name)
                except ToolWidgetConstructError as ex:
                    warn("build_from_layout: %s" % (ex.message),
                         RuntimeWarning)
                    continue
                assert tool_widget is not None
                nb.append_tool_widget_page(tool_widget)
                if self.workspace:
                    self.workspace.tool_widget_added(tool_widget, gtype_name)
            group_h = group_desc.get("h", None)
            if group_h is not None:
                group_h = max(100, int(group_h))
                nb_parent = nb.get_parent()
                assert isinstance(nb_parent, ToolStack._Paned)
                nb_parent.set_position(group_h)
                self._initial_paned_positions.append((nb_parent, group_h))



    def get_layout(self):
        """Returns a description of the current layout using simple types.

        :rtype: dict

        See `build_from_layout()` for details of the dict which is returned.

        """
        group_descs = []
        for nb in self._get_notebooks():
            tool_descs = []
            for page in nb:
                tool_widget = page.get_child()
                gtype_name = tool_widget.__gtype_name__
                tool_desc = (gtype_name, )
                tool_descs.append(tool_desc)
            group_desc = {"tools": tool_descs}
            if tool_descs:
                width = nb.get_allocated_width()
                height = nb.get_allocated_height()
                if width is not None and height is not None:
                    group_desc["w"] = max(width, 1)
                    group_desc["h"] = max(height, 1)
            group_descs.append(group_desc)
        stack_desc = {"groups": group_descs}
        if group_descs:
            width = self.get_allocated_width()
            height = self.get_allocated_height()
            if width is not None and height is not None:
                stack_desc["w"] = max(width, 1)
                stack_desc["h"] = max(height, 1)
        return stack_desc


    ## Initial layout (pre/post-realize)


    def _start_initial_layout(self):
        """Layout: all that can be done before the toplevel is positioned.
        """
        pass


    def _complete_initial_layout(self):
        """Finish initial layout; called after toplevel win is positioned.
        """
        # Init tool group sizes by setting vpaned positions
        for paned, pos in self._initial_paned_positions:
            paned.set_position(pos)
        self._initial_paned_positions = None


    ## Tool widgets


    def add_tool_widget(self, gtype_name, maxnotebooks=None, maxpages=3):
        """Tries to find space for, and create, a tool widget via GType name.

        :param gtype_name: the GType name of the class to load.
        :type gtype_name: str
        :param maxnotebooks: never make more than this many groups
        :type maxnotebooks: int
        :param maxpages: never make more than this many pages in a group
        :type maxpages: int
        :rtype: Gtk.Widget or None

        """
        target_notebook = None
        notebooks = self._get_notebooks()
        for nb in notebooks:
            if nb.get_n_pages() < maxpages:
                target_notebook = nb
                break
        # Last one should always be the placeholder...
        assert target_notebook is not None
        # ... but don't always just use it.
        if target_notebook.get_n_pages() == 0:
            num_populated = len(notebooks) - 1
            if maxnotebooks is not None and num_populated >= maxnotebooks:
                return None
        tool_widget = self._tool_widget_new(gtype_name)
        if not tool_widget:
            raise ToolWidgetConstructError, \
                  "Cannot construct a '%s': unknown reason" % (gtype_name,)
        target_notebook.append_tool_widget_page(tool_widget)
        if self.workspace:
            self.workspace.tool_widget_added(tool_widget, gtype_name)
        return tool_widget


    def remove_tool_widget(self, gtype_name):
        """Removes a tool widget by its GType name from the stack.

        If the stack contains multiple instances of the named class, then only
        the first such tool widget will be removed.

        :param gtype_name: the GType name of the tab to be removed.
        :type gtype_name: str.
        :rtype: the removed tool widget, or None.

        """
        for notebook in self._get_notebooks():
            for index in xrange(notebook.get_n_pages()):
                page = notebook.get_nth_page(index)
                tool_widget = page.get_child()
                if tool_widget.__gtype_name__ != gtype_name:
                    continue
                notebook.remove_page(index)
                if self.workspace:
                    self.workspace.tool_widget_removed(page, gtype_name)
                return tool_widget
        return None


    def has_tool_widget(self, gtype_name):
        """Tests if a tool widget instance exists in this stack.
        """
        if self.get_tool_widget(gtype_name):
            return True
        return False


    def get_tool_widget(self, gtype_name):
        """Returns the first matching tool widget.
        """
        for notebook in self._get_notebooks():
            for index in xrange(notebook.get_n_pages()):
                page = notebook.get_nth_page(index)
                tool_widget = page.get_child()
                if tool_widget.__gtype_name__ == gtype_name:
                    return tool_widget
        return None


    def is_empty(self):
        """Returns true if this stack contains only a tab drop placeholder.
        """
        widget = self.get_child()
        if isinstance(widget, Gtk.Paned):
            return False
        assert isinstance(widget, Gtk.Notebook)
        return widget.get_n_pages() == 0


    ## Internal structure helpers


    def _get_first_notebook(self):
        widget = self.get_child()
        if isinstance(widget, Gtk.Paned):
            widget = widget.get_child1()
        assert isinstance(widget, Gtk.Notebook)
        return widget


    def _get_notebooks(self):
        child = self.get_child()
        if child is None:
            return []
        queue = [child]
        notebooks = []
        while len(queue) > 0:
            widget = queue.pop(0)
            if isinstance(widget, Gtk.Paned):
                queue.append(widget.get_child1())
                queue.append(widget.get_child2())
            elif isinstance(widget, Gtk.Notebook):
                notebooks.append(widget)
            else:
                warn("Unknown member type: %s" % str(widget), RuntimeWarning)
        assert len(notebooks) > 0
        return notebooks


    def _get_final_paned(self):
        child = self.get_child()
        if child is None:
            return None
        queue = [child]
        result = None
        while len(queue) > 0:
            widget = queue.pop(0)
            if isinstance(widget, Gtk.Paned):
                result = widget
                queue.append(widget.get_child1())
                queue.append(widget.get_child2())
        return result


    ## Group size management (somewhat dubious)


    def _size_alloc_cb(self, widget, alloc):
        # When the size changes, manage the divider position of the final
        # paned, shrinking or growing the final set of tabs.
        paned = self._get_final_paned()
        if paned is None:
            return
        # Did the bottom set of tabs fill the available space the last time
        # this was called?
        try:
            paned_was_filled = paned.__filled
        except AttributeError:
            paned_was_filled = False
        pos = paned.get_position()
        max_pos = paned.get_property("max-position")
        min_pos = paned.get_property("min-position")
        stickiness = self.RESIZE_STICKINESS
        # Reset the flag if it's now near the top of its range. This allows
        # the user to reset the stickiness by moving the divider to the
        # top.
        if paned_was_filled and pos - min_pos < stickiness:
            paned_was_filled = False
        # However, keep the flag set and move the bar if it's now near the
        # bottom of its range. Lets the user set stickiness by moving the
        # divider to the bottom.
        if paned_was_filled or max_pos - pos < stickiness:
            paned.set_position(max_pos)
            paned.__filled = True
        else:
            paned.__filled = False


    ## Paned/Notebook tree structure: node and leaf creation


    def _make_notebook(self):
        """Creates a new notebook widget in the right way.
        """
        return ToolStack._Notebook(self)


    def _append_new_placeholder(self, old_placeholder):
        """Appends a new placeholder after a current or former placeholder.
        """
        paned = ToolStack._Paned(self, old_placeholder)
        return paned.get_child2()


    def _tool_widget_new(self, gtype_name):
        """Constructs a new tool widget based on its GType name.
        """
        try:
            gtype = GObject.type_from_name(gtype_name)
        except RuntimeError:
            raise ToolWidgetConstructError, \
                  "Cannot construct a '%s': not loaded?" % (gtype_name,)
        if not gtype.is_a(Gtk.Widget):
            raise ToolWidgetConstructError, \
                  "%s is not a Gtk.Widget subclass" % (gtype_name,)
        tool_widget_class = gtype.pytype
        tool_widget = tool_widget_class()
        return tool_widget



    ## Paned/Notebook tree structure: maintenance


    def _update_structure(self):
        """Maintains structure after "page-added" & "page-deleted" events.

        If a page is added to the placeholder notebook on the end by the user
        dragging a tab there, a new placeholder must be created and the tree
        structure repacked. Similarly emptying out a notebook by dragging
        tabs around must result in the empty notebook being removed.

        This callback is queued as an idle function in response to the above
        events because moving from one paned to another invokes both remove
        and add. If the structure doesn't need changing, calling it multiple
        times is harmless.

        """

        # Ensure that the final notebook is always an empty placeholder.
        notebooks = self._get_notebooks()
        if len(notebooks) == 0:
            return
        placeholder_nb = notebooks.pop(-1)
        nb_parent = placeholder_nb.get_parent()
        if placeholder_nb.get_n_pages() > 0:
            old_placeholder = placeholder_nb
            placeholder_nb = old_placeholder.split_former_placeholder()
            notebooks.append(old_placeholder)

        # Detect emptied middle notebooks and remove them. There should be no
        # notebooks in the stack whose parent is not a Paned at this point.
        while len(notebooks) > 0:
            nb = notebooks.pop(0)
            nb_parent = nb.get_parent()
            assert isinstance(nb_parent, Gtk.Paned)
            if nb.get_n_pages() > 0:
                continue
            nb_grandparent = nb_parent.get_parent()
            assert nb     is nb_parent.get_child1()
            assert nb is not nb_parent.get_child2()
            sib = nb_parent.get_child2()
            nb_parent.remove(nb)
            nb_parent.remove(sib)
            if isinstance(nb_grandparent, Gtk.Paned):
                assert nb_parent is not nb_grandparent.get_child1()
                assert nb_parent     is nb_grandparent.get_child2()
                nb_grandparent.remove(nb_parent)
                if sib is placeholder_nb:
                    nb_grandparent.pack2_placeholder_notebook(sib)
                else:
                    assert isinstance(sib, Gtk.Paned)
                    nb_grandparent.pack2_subpaned(sib)
            else:
                assert nb_grandparent is self
                nb_grandparent.remove(nb_parent)
                nb_grandparent.add(sib)

        # Detect empty stacks
        n_tabs_total = 0
        for nb in self._get_notebooks():
            n_tabs_total += nb.get_n_pages()
        parent = self.get_parent()
        if n_tabs_total == 0:
            if isinstance(parent, ToolStackWindow):
                self.workspace.emit("floating-window-destroy", parent)
                parent.destroy()
            else:
                self.hide()
            return

        # Update title of parent ToolStackWindows
        if isinstance(parent, ToolStackWindow):
            page_titles = []
            for nb in self._get_notebooks():
                for page in nb:
                    tool_widget = page.get_child()
                    title = _tool_widget_get_title(tool_widget)
                    page_titles.append(title)
            parent.update_title(page_titles)


class ToolStackWindow (Gtk.Window):
    """A floating utility window containing a single `ToolStack`.
    """

    ## GObject integration (type name)

    __gtype_name__ = "MyPaintToolStackWindow"


    ## Construction


    def __init__(self):
        Gtk.Window.__init__(self)
        self.set_type_hint(Gdk.WindowTypeHint.UTILITY)
        self.set_accept_focus(False)
        self.connect("destroy", self._destroy_cb)
        self.stack = ToolStack() #: The ToolStack child of the window
        self.add(self.stack)
        self.connect("realize", self._realize_cb)
        self.connect("map", self._map_cb)
        self.connect("configure-event", self._configure_cb)
        self.update_title([])
        self._pos = None
        self._initial_xy = None
        self._frame_size = None
        self._mapped_once = False


    ## Setup from layout definitions (pre-realize)


    def build_from_layout(self, layout):
        """Build the window's contents from a layout description.
        """
        logger.debug("build_from_layout %r", self)
        self.stack.build_from_layout(layout.get("contents", {}))
        pos = layout.get("position", None)
        if pos:
            self._pos = pos.copy()


    def get_layout(self):
        """Get the window's position and contents in simple dict form.
        """
        return { "position": self._pos,
                 "contents": self.stack.get_layout(), }


    ## Window lifecycle events (initial state, position tracking)


    def _realize_cb(self, widget):
        logger.debug("Realize %r", self)
        if not self._pos:
            return
        xy = set_initial_window_position(self, self._pos)
        self._initial_xy = xy


    def _map_cb(self, widget):
        logger.debug("map %r", self)
        workspace = self.stack.workspace
        if workspace:
            # Prevent subwindows from taking keyboard focus from the main
            # window (in Metacity) by presenting it again.
            # https://gna.org/bugs/?17899
            toplevel = workspace.get_toplevel()
            GObject.idle_add(lambda *a: toplevel.present())
        if self._mapped_once:
            return
        # First map stuff
        if workspace:
            self.set_transient_for(toplevel)
            workspace._floating.add(self)
        win = widget.get_window()
        decor = Gdk.WMDecoration.BORDER|Gdk.WMDecoration.RESIZEH
        win.set_decorations(decor)
        wmfuncs = Gdk.WMFunction.RESIZE|Gdk.WMFunction.MOVE
        win.set_functions(wmfuncs)
        self._mapped_once = True
        if self._initial_xy:
            # Hack to force the initial x,y position
            GObject.idle_add(lambda *a: self.move(*self._initial_xy))


    def _configure_cb(self, widget, event):
        """Track the window size and position when it changes.
        """
        frame = self.get_window().get_frame_extents()
        x = max(0, frame.x)
        y = max(0, frame.y)
        self._pos = dict(x=x, y=y, w=event.width, h=event.height)
        self._frame_size = frame.width, frame.height


    def _destroy_cb(self, widget):
        workspace = self.stack.workspace
        if workspace is not None:
            if self in workspace._floating:
                workspace._floating.remove(self)


    ## Autohide in fullscreen


    def contains_point(self, x, y, b=0):
        """True if a screen point is over this window's last known position.

        :param x: Root window X-coordinate 
        :param y: Root window Y-coordinate 
        :param b: Additional sensitive border around the window, in pixels
        :rtype: bool

        Used for revealing floating windows after they have been auto-hidden.

        """
        if not (self._pos and self._frame_size):
            return False
        fx, fy = self._pos["x"], self._pos["y"]
        fw, fh = self._frame_size
        return x >= fx-b and x <= fx+fw+b and y >= fy-b and y <= fy+fh+b


    ## Window title


    def update_title(self, tool_widget_titles):
        titles = [unicode(s) for s in tool_widget_titles]
        workspace = self.stack.workspace
        if workspace is not None:
            title_sep = unicode(workspace.floating_window_title_separator)
            title = title_sep.join(titles)
            title_suffix = unicode(workspace.floating_window_title_suffix)
            if title_suffix:
                title += unicode(title_suffix)
            self.set_title(title)



## Utility functions


def _tool_widget_get_title(widget):
    """Returns the title to use for a tool-widget.

    :param widget: a tool widget
    :type widget: Gtk.Widget
    :rtype: unicode

    """
    for attr in ("tool_widget_title", "__gtype_name__"):
        try:
            return unicode(getattr(widget, attr))
        except AttributeError:
            pass
    return unicode(widget.__class__.__name__)


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

    MIN_USABLE_SIZE = 100

    # Final calculated positions
    final_x, final_y = None, None
    final_w, final_h = None, None

    # Positioning arguments
    x = pos.get("x", None)
    y = pos.get("y", None)
    w = pos.get("w", None)
    h = pos.get("h", None)

    # Where the mouse is right now
    display = win.get_display()
    screen, ptr_x, ptr_y, _modmask = display.get_pointer()
    if screen is None:
        raise RuntimeError, "No cursor on the default screen. Eek."
    screen_w = screen.get_width()
    screen_h = screen.get_height()
    assert screen_w > MIN_USABLE_SIZE
    assert screen_h > MIN_USABLE_SIZE

    mon_num = screen.get_monitor_at_point(ptr_x, ptr_y)
    mon_geom = screen.get_monitor_geometry(mon_num)

    # Generate a sensible, positive x and y position
    if x is not None and y is not None:
        if x >= 0:
            final_x = x
        else:
            assert w is not None
            assert w > 0
            final_x = mon_geom.x + (mon_geom.width - w - abs(x))
        if y >= 0:
            final_y = y
        else:
            assert h is not None
            assert h > 0
            final_y = mon_geom.y + (mon_geom.height - h - abs(y))
        if final_x < 0 or final_x > screen_w - MIN_USABLE_SIZE:
            final_x = None
        if final_y < 0 or final_y > screen_h - MIN_USABLE_SIZE:
            final_y = None

    # And a sensible, positive width and height
    if w is not None and h is not None:
        final_w = w
        final_h = h
        if w < 0 or h < 0:
            mon_num = screen.get_monitor_at_point(ptr_x, ptr_y)
            mon_geom = screen.get_monitor_geometry(mon_num)
            if w < 0:
                if x is not None:
                    final_w = max(0, mon_geom.width - abs(x) - abs(w))
                else:
                    final_w = max(0, mon_geom.width - 2*abs(w))
            if h < 0:
                if x is not None:
                    final_h = max(0, mon_geom.height - abs(y) - abs(h))
                else:
                    final_h = max(0, mon_geom.height - 2*abs(h))
        if final_w > screen_w or final_w < MIN_USABLE_SIZE:
            final_w = None
        if final_h > screen_h or final_h < MIN_USABLE_SIZE:
            final_h = None

    # If the window is positioned, make sure it's on a monitor which still
    # exists. Users change display layouts...
    if None not in (final_x, final_y):
        on_existing_mon = False
        for mon_num in xrange(screen.get_n_monitors()):
            mon_geom = screen.get_monitor_geometry(mon_num)
            on_this_mon = (final_x < (mon_geom.x + mon_geom.width) and
                           final_y < (mon_geom.x + mon_geom.height) and
                           final_x >= mon_geom.x and
                           final_y >= mon_geom.y)
            if on_this_mon:
                on_existing_mon = True
                break
        if not on_existing_mon:
            logger.warning("Calculated window position is offscreen; "
                           "ignoring %r" % ((final_x, final_y), ))
            final_x = None
            final_y = None

    # Attempt to set up with a geometry string first. Repeats the block below
    # really, but this helps smaller windows receive the right position in
    # xfwm (at least), possibly because the right window hints will be set.
    if None not in (final_w, final_h, final_x, final_y):
        geom_str = "%dx%d+%d+%d" % (final_w, final_h, final_x, final_y)
        realize_cb = lambda *a: win.parse_geometry(geom_str)
        win.connect("realize", realize_cb)

    # Set what we can now.
    if None not in (final_w, final_h):
        win.set_default_size(final_w, final_h)
    if None not in (final_x, final_y):
        win.move(final_x, final_y)
        return final_x, final_y

    return None


## Module testing (interactive, but fairly minimal)


def _test():
    logging.basicConfig(level=logging.DEBUG)
    import os, sys
    class _TestLabel (Gtk.Label):
        __gtype_name__ = 'TestLabel'
        tool_widget_icon_name = 'gtk-ok'
        tool_widget_description = "Just a test widget"
        def __init__(self):
            Gtk.Label.__init__(self, "Hello, World")
            self.set_size_request(200, 150)
    class _TestSpinner (Gtk.Spinner):
        __gtype_name__ = "TestSpinner"
        tool_widget_icon_name = 'gtk-cancel'
        tool_widget_description = "Spinner test"
        def __init__(self):
            Gtk.Spinner.__init__(self)
            self.set_size_request(150, 150)
            self.set_property("active", True)
    def _tool_added_cb(*a):
        logger.debug("TOOL-ADDED %r", a)
    def _tool_removed_cb(*a):
        logger.debug("TOOL-REMOVED %r", a)
    workspace = Workspace()
    workspace.floating_window_title_suffix = u" - Test"
    canvas = Gtk.Label("<Placeholder>")
    frame = Gtk.Frame()
    frame.add(canvas)
    frame.set_shadow_type(Gtk.ShadowType.IN)
    workspace.set_canvas(frame)
    window = Gtk.Window()
    window.add(workspace)
    window.set_title(os.path.basename(sys.argv[0]))
    workspace.set_size_request(600, 400)
    workspace.connect("tool-widget-added", _tool_added_cb)
    workspace.connect("tool-widget-removed", _tool_removed_cb)
    workspace.build_from_layout({
        'position': {'x': 100, 'y': 75, 'h': -100, 'w': -100},
        'floating': [{
            'position': {'y': -100, 'h': 189, 'w': 152, 'x': -200},
            'contents': {
                'groups': [{'tools': [('TestSpinner',), ('TestLabel',)]}],
            }}],
        'right_sidebar': {
            'w': 400,
            'groups': [{'tools': [('TestSpinner',)]}],
        },
        'left_sidebar': {
            'w': 250,
            'groups': [{'tools': [('TestSpinner',), ('TestLabel',)]}],
        },
        'maximized': False,
    })
    window.show_all()
    def _quit_cb(*a):
        logger.info("Demo quit, workspace dump follows")
        print workspace.get_layout()
        Gtk.main_quit()
    window.connect("destroy", _quit_cb)
    Gtk.main()


if __name__ == '__main__':
    _test()

