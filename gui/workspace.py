# This file is part of MyPaint.
# Copyright (C) 2014-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Workspaces with a central canvas, sidebars and saved layouts"""

## Imports

from __future__ import division, print_function
from warnings import warn
import sys
import math
import logging

from gi.repository import GObject
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib

from lib.observable import event
import lib.xml
import lib.helpers
from . import objfactory
from .widgets import borderless_button
from lib.gettext import C_
from lib.pycompat import xrange
from lib.pycompat import unicode

logger = logging.getLogger(__name__)

## Tool widget size constants

# Tool widgets should use GTK3-style sizing, and the lollowing layout
# constants.

#: Minimum width for a "sidebar-dockable" tool widget.
TOOL_WIDGET_MIN_WIDTH = 220

#: Minimum height for a "sidebar-dockable" tool widget.
TOOL_WIDGET_MIN_HEIGHT = 25

# Tool widgets should declare natural heights that result in nice ratios: not
# too short and not too tall. The GNOME HIG recommends that the longer
# dimension of a window not be more than 50% longer than the shorter dimension.
# The layout code will respect widgets' natural sizes vertically. For the look
# of the UI as a whole, it's best to use one of the sizing constants below for
# the natural height in most cases.

#: Natural height for shorter tool widgets
TOOL_WIDGET_NATURAL_HEIGHT_SHORT = TOOL_WIDGET_MIN_WIDTH

#: Natural height for taller tool widget
TOOL_WIDGET_NATURAL_HEIGHT_TALL = 1.25 * TOOL_WIDGET_MIN_WIDTH

## Class defs


class Workspace (Gtk.VBox, Gtk.Buildable):
    """Widget housing a central canvas flanked by two sidebar toolstacks

    Workspaces also manage zero or more floating ToolStacks, and can set the
    initial size and position of their own toplevel window.

    Instances of tool widget classes can be constructed, then shown and hdden
    by the workspace programmatically using their GType name and an optional
    sequence of construction parameters as a key.  They should support the
    following Python properties:

    * ``tool_widget_icon_name``: the name of the icon to use.
    * ``tool_widget_title``: the title to display in the tooltip, and in
      floating window titles.
    * ``tool_widget_description``: the description string to show in the
      tooltip.

    and the following methods:

    * ``tool_widget_properties()``: show the properties dialog.
    * ``tool_widget_get_icon_pixbuf(size)``: returns a pixbuf icon for a
      particular pixel size. This is used in preference to the icon name.

    Defaults will be used if these properties and methods aren't defined, but
    the defaults are unlikely to be useful.

    The entire layout of a Workspace, including the toplevel window of the
    Workspace itself, can be dumped to and built from structures containing
    only simple Python types.  Widgets can be hidden by the user by clicking on
    a tab group close button within a ToolStack widget, moved around between
    stacks, or snapped out of stacks into new floating windows.

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
    AUTOHIDE_REVEAL_BORDER = 50

    #: Time in milliseconds to wait before revealing UI elements when pointer
    #: is near a window edge.  Does not affect floating hidden windows.
    AUTOHIDE_REVEAL_TIMEOUT = 500

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
        Gdk.ModifierType.BUTTON5_MASK
    )

    # Edges the pointer can bump: used for autohide reveals

    _EDGE_NONE = 0x00
    _EDGE_LEFT = 0x01
    _EDGE_RIGHT = 0x02
    _EDGE_TOP = 0x04
    _EDGE_BOTTOM = 0x08

    ## GObject integration (type name, properties)

    __gtype_name__ = 'MyPaintWorkspace'

    #: Title suffix property for floating windows.
    floating_window_title_suffix = GObject.property(
        type=str,
        flags=GObject.PARAM_READWRITE,
        nick='Floating window title suffix',
        blurb='The suffix to append to floating windows: typically a '
              'hyphen followed by the application name.',
        default=None
    )

    #: Title separator property for floating windows.
    floating_window_title_separator = GObject.property(
        type=str,
        flags=GObject.PARAM_READWRITE,
        nick='Floating window title separator',
        blurb='String used to separate the names of tools in a '
              'floating window. By default, a comma is used.',
        default=", "
    )

    #: Header bar widget, to be hidden when entering fullscreen mode. This
    #: widget should be packed externally to the workspace, and to its top.
    header_bar = GObject.property(
        type=Gtk.Widget,
        flags=GObject.PARAM_READWRITE,
        nick='Header bar widget',
        blurb="External Menubar/toolbar widget to be hidden when "
              "entering fullscreen mode, and re-shown when leaving "
              "it. The pointer position is also used for reveals and "
              "hides in fullscreen.",
        default=None
    )

    #: Footer bar widget, to be hidden when entering fullscreen mode. This
    #: widget should be packed externally to the workspace, and to its bottom.
    footer_bar = GObject.property(
        type=Gtk.Widget,
        flags=GObject.PARAM_READWRITE,
        nick='Footer bar widget',
        blurb="External footer bar widget to be hidden when entering "
              "fullscreen mode, and re-shown when leaving it. The "
              "pointer position is also used for reveals and hides "
              "in fullscreen.",
        default=None
    )

    def __init__(self):
        """Initializes, with a placeholder canvas widget and no tool widgets"""
        Gtk.VBox.__init__(self)
        # Sidebar stacks
        self._lstack = lstack = ToolStack()
        self._rstack = rstack = ToolStack()
        lscrolls = Gtk.ScrolledWindow()
        rscrolls = Gtk.ScrolledWindow()
        self._lscrolls = lscrolls
        self._rscrolls = rscrolls
        lscrolls.add(lstack)
        rscrolls.add(rstack)
        for scrolls in [lscrolls, rscrolls]:
            try:
                # Fix 3.16+'s mystery meat: don't use fading scrollbars.
                # Corrects a terrible upstream UX decision which makes
                # the scrollbar cover our widgets.
                scrolls.set_overlay_scrolling(False)
            except AttributeError:
                pass
            scrolls.set_shadow_type(Gtk.ShadowType.IN)
            scrolls.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._lpaned = lpaned = Gtk.HPaned()
        self._rpaned = rpaned = Gtk.HPaned()
        for stack, paned in [(lstack, lpaned), (rstack, rpaned)]:
            stack.workspace = self
            stack.connect("hide", self._sidebar_stack_hide_cb, paned)
            try:
                # Fix 3.16+'s mystery meat: don't hide the divider.
                # Corrects a terrible upstream UX decision which makes
                # things hugely worse for touchscreen and tablet users.
                paned.set_wide_handle(True)
            except AttributeError:
                pass
        # Canvas scrolls. The canvas isn't scrollable yet, but use the same
        # class as the right and left sidebars so that the shadows match
        # in all themes.
        cscrolls = Gtk.ScrolledWindow()
        cscrolls.set_shadow_type(Gtk.ShadowType.IN)
        cscrolls.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.NEVER)
        self._canvas_scrolls = cscrolls
        # Sidebar packing
        lpaned.pack1(lscrolls, resize=False, shrink=False)
        lpaned.pack2(rpaned, resize=True, shrink=False)
        rpaned.pack2(rscrolls, resize=False, shrink=False)
        rpaned.pack1(cscrolls, resize=True, shrink=False)
        self.pack_start(lpaned, True, True, 0)
        # Autohide
        self._autohide_enabled = True
        self._autohide_timeout = None
        self._autoreveal_timeout = []
        # Window tracking
        self._floating = set()
        self._toplevel_pos = dict()
        self._save_toplevel_pos_timeout = None
        self._is_fullscreen = False
        self._is_maximized = False
        self._fs_event_handlers = []
        # Initial layout happens in several phases
        self._initial_layout = None
        self.connect("realize", self._realize_cb)
        self.connect("map", self._map_cb)
        # Tool widget cache and factory
        self._tool_widgets = objfactory.ObjFactory(gtype=Gtk.Widget)
        self._tool_widgets.object_rebadged += self._tool_widget_rebadged

    ## GtkBuildable implementation (pre-realize)

    def do_add_child(self, builder, child, type_):
        """Adds a child as the canvas: gtk_buildable_add_child() impl."""
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
            toplevel_win.fullscreen()
            GLib.idle_add(lambda *a: toplevel_win.fullscreen())
        elif layout.get("maximized", False):
            toplevel_win.maximize()
            GLib.idle_add(lambda *a: toplevel_win.maximize())
        toplevel_win.connect("window-state-event",
                             self._toplevel_window_state_event_cb)
        self.autohide_enabled = layout.get("autohide", True)
        self._initial_layout = layout

    def get_layout(self):
        """Returns a layout definition dict for the workspace

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
        """Kick off the deferred layout code when the widget is realized"""

        # Set up monitoring of the toplevel's size changes.
        toplevel = self.get_toplevel()
        toplevel.connect("configure-event", self._toplevel_configure_cb)

        # Do the initial layout
        layout = self._initial_layout
        if layout is None:
            return
        llayout = layout.get("left_sidebar", {})
        if llayout:
            logger.debug("Left sidebar: building from saved layout...")
            num_added_l = self._lstack.build_from_layout(llayout)
            logger.debug("Left sidebar: added %d group(s)", num_added_l)
        rlayout = layout.get("right_sidebar", {})
        if rlayout:
            logger.debug("Right sidebar: building from saved layout...")
            num_added_r = self._rstack.build_from_layout(rlayout)
            logger.debug("Right sidebar: added %d group(s)", num_added_r)
        # Floating windows
        for fi, flayout in enumerate(layout.get("floating", [])):
            logger.debug(
                "Floating window %d: building from saved layout...",
                fi,
            )
            win = ToolStackWindow()
            self.floating_window_created(win)
            win.stack.workspace = self
            num_added_f = win.build_from_layout(flayout)
            logger.debug(
                "Floating window %d: added %d group(s)",
                fi,
                num_added_f,
            )
            # The populated ones are only revealed after their
            # floating_window_created have had a chance to run.
            if num_added_f > 0:
                self._floating.add(win)
                GLib.idle_add(win.show_all)
            else:
                logger.warning(
                    "Floating window %d is initially unpopulated. "
                    "Destroying it.",
                    fi,
                )
                win.stack.workspace = None
                GLib.idle_add(win.destroy)

    def _map_cb(self, widget):
        assert self.get_realized()
        logger.debug("Completing layout (mapped)")
        GLib.idle_add(self._complete_initial_layout)

    def _complete_initial_layout(self):
        """Finish initial layout; called after toplevel win is positioned"""
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
        """Canvas widget (setter)"""
        assert self.get_canvas() is None
        widget = self._canvas_scrolls.add(widget)

    def get_canvas(self):
        """Canvas widget (getter)"""
        widget = self._canvas_scrolls.get_child()
        return widget

    def _update_canvas_scrolledwindow(self):
        """Update the canvas ScrolledWindow's border."""
        parent = self._canvas_scrolls
        if not self._is_fullscreen:
            parent.set_shadow_type(Gtk.ShadowType.NONE)
        else:
            parent.set_shadow_type(Gtk.ShadowType.IN)
        # TODO: this should really be done with CSS now.

    ## Tool widgets

    def reveal_tool_widget(self, tool_gtypename, tool_params):
        """Show and present a widget.

        This is a keyboard-friendly alternative to the add/remove toggle
        actions. If the widget is not currently added, it is added.
        Otherwise it is revealed, meaning that its enclosing stack is
        presented to the user and the widget's tab is set as the current
        one.

        """
        if not self.get_tool_widget_added(tool_gtypename, tool_params):
            logger.debug(
                "Reveal: %r %r not in UI, attempting to add it",
                tool_gtypename, tool_params,
            )
            self.add_tool_widget(tool_gtypename, tool_params)
            return
        logger.debug(
            "Reveal: %r %r already in UI, finding and revealing its stack",
            tool_gtypename, tool_params,
        )
        assert self._tool_widgets.cache_has(tool_gtypename, *tool_params)
        widget = self._tool_widgets.get(tool_gtypename, *tool_params)
        assert widget is not None
        stack = self._get_tool_widget_stack(widget)
        assert stack and isinstance(stack, ToolStack)
        stack.reveal_tool_widget(widget)

    def add_tool_widget(self, tool_gtypename, tool_params):
        """Shows a tool widget identified by GType name and construct params

        :param tool_gtypename: GType system name for the new widget's class
        :param tool_params: parameters for the class's Python constructor

        The widget will be created if it doesn't already exist. It will be
        added to the first stack available. Existing floating windows will be
        favoured over the sidebars; if there are no stacks visible, a sidebar
        will be made visible to receive the new widget.

        """
        logger.debug("Adding %r %r", tool_gtypename, tool_params)
        # Attempt to get the widget, potentially creating it here
        try:
            widget = self._tool_widgets.get(tool_gtypename, *tool_params)
        except objfactory.ConstructError as ex:
            logger.error("add_tool_widget: %s", ex.message)
            return
        # Inject it into a suitable ToolStack
        stack = None
        if widget.get_parent() is not None:
            logger.debug("Existing %r is already added", widget)
            stack = self._get_tool_widget_stack(widget)
        else:
            logger.debug("Showing %r, which is currently hidden", widget)
            maxpages = 1
            added = False
            stack = None
            while maxpages < 100 and not added:
                for stack in self._get_tool_stacks():
                    if stack.add_tool_widget(widget, maxnotebooks=3,
                                             maxpages=maxpages):
                        added = True
                        break
                maxpages += 1
            if not added:
                logger.error("Can't find space for %r in any stack", widget)
                return
        # Reveal the widget's ToolStack
        assert stack and isinstance(stack, ToolStack)
        stack.reveal_tool_widget(widget)

    def remove_tool_widget(self, tool_gtypename, tool_params):
        """Removes a tool widget by typename+params

        This hides the widget and orphans it from the widget hierarchy, but a
        reference to it is kept in the Workspace's internal cache. Further
        calls to add_tool_widget() will use the cached object.

        :param tool_gtypename: GType system name for the widget's class
        :param tool_params: construct params further identifying the widget
        :returns: whether the widget was found and hidden
        :rtype: bool

        """
        # First, does it even exist?
        if not self._tool_widgets.cache_has(tool_gtypename, *tool_params):
            return False
        # Can't remove anything that's already removed
        widget = self._tool_widgets.get(tool_gtypename, *tool_params)
        if widget.get_parent() is None:
            return False
        # The widget should exist in a known stack; find and remove
        for stack in self._get_tool_stacks():
            if stack.remove_tool_widget(widget):
                return True
        # Should never happen...
        warn("Asked to remove a visible widget, but it wasn't in any stack",
             RuntimeWarning)
        return False

    def get_tool_widget_added(self, gtype_name, params):
        """Returns whether a tool widget is currently in the widget tree"""
        # Nonexistent objects are not parented or showing
        if not self._tool_widgets.cache_has(gtype_name, *params):
            return False
        # Otherwise, just test whether it's in a widget tree
        widget = self._tool_widgets.get(gtype_name, *params)
        return widget.get_parent() is not None

    def _get_tool_widget_stack(self, widget):
        """Gets the ToolStack a widget is currently parented underneath."""
        stack = None
        if widget.get_parent() is not None:
            stack = widget.get_parent()
            while stack and not isinstance(stack, ToolStack):
                stack = stack.get_parent()
        return stack

    def update_tool_widget_params(self, tool_gtypename,
                                  old_params, new_params):
        """Update the construction params of a tool widget

        :param tool_gtypename: GType system name for the widget's class
        :param old_params: old parameters for the class's Python constructor
        :param new_params: new parameters for the class's Python constructor

        If an object has changed so that its construction parameters must be
        update, this method should be called to keep track of its identity
        within the workspace.  This method will not show or hide it: its
        current state remains the same.

        See also `update_tool_widget_ui()`.

        """
        # If it doesn't exist yet, updating what is effectively
        # a cache key used for accessing it makes no sense.
        if not self._tool_widgets.cache_has(tool_gtypename, *old_params):
            return
        # Update the params of an existing object.
        widget = self._tool_widgets.get(tool_gtypename, *old_params)
        if old_params == new_params:
            logger.devug("No construct params update needed for %r", widget)
        else:
            logger.debug("Updating construct params for %r", widget)
            self._tool_widgets.rebadge(widget, new_params)

    def update_tool_widget_ui(self, gtype_name, params):
        """Updates tooltips and tab labels for a specified tool widget

        Calling this method causes the workspace to re-read the tool widget's
        tab label, window title, and tooltip properties, and update the
        display.  Use it when things like the icon pixbuf have changed.  It's
        not necessary to call this after `update_tool_widget_params()` has been
        used: that's handled with an event.

        """
        if not self.get_tool_widget_added(gtype_name, params):
            return
        widget = self._tool_widgets.get(gtype_name, *params)
        logger.debug("Updating workspace UI widgets for %r", widget)
        self._update_tool_widget_ui(widget)

    ## Tool widget events

    @event
    def tool_widget_added(self, widget):
        """Event: tool widget added"""

    @event
    def tool_widget_removed(self, widget):
        """Event: tool widget removed either by the user or programmatically"""

    @event
    def floating_window_created(self, toplevel):
        """Event: a floating window was created to house a toolstack."""

    @event
    def floating_window_destroyed(self, toplevel):
        """Event: a floating window was just `destroy()`ed."""

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
            GLib.source_remove(srcid)
        srcid = GLib.timeout_add(
            250,
            self._save_toplevel_pos_timeout_cb,
            w, h,
        )
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

    def _toplevel_window_state_event_cb(self, toplevel, event):
        """Handle transitions between fullscreen and windowed."""
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
            self._update_canvas_scrolledwindow()
        if event.changed_mask & Gdk.WindowState.MAXIMIZED:
            maximized = event.new_window_state & Gdk.WindowState.MAXIMIZED
            self._is_maximized = bool(maximized)

    ## Autohide flag

    def get_autohide_enabled(self):
        """Auto-hide is enabled in fullscreen (getter)"""
        return self._autohide_enabled

    def set_autohide_enabled(self, autohide_enabled):
        """Auto-hide is enabled in fullscreen (setter)"""
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
        """Hides all auto-hiding widgets immediately"""
        if not self._is_fullscreen:
            return
        self._cancel_autohide_timeout()
        display = self.get_display()
        if display.pointer_is_grabbed():
            logger.warning("Pointer grabbed: not auto-hiding")
            return
        ah_widgets = self._get_autohide_widgets()
        logger.debug("Hiding %d autohide widget(s)", len(ah_widgets))
        for widget in ah_widgets:
            if widget.get_visible():
                widget.hide()

    def _show_autohide_widgets(self):
        """Shows all auto-hiding widgets immediately"""
        self._cancel_autohide_timeout()
        ah_widgets = self._get_autohide_widgets()
        logger.debug("Hiding %d autohide widget(s)", len(ah_widgets))
        for widget in ah_widgets:
            widget.show_all()

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
        """Start a timer to hide the UI after a brief period of inactivity"""
        if not self._autohide_timeout:
            logger.debug("Starting autohide timeout (%d milliseconds)",
                         self.AUTOHIDE_TIMEOUT)
        else:
            self._cancel_autohide_timeout()
        srcid = GLib.timeout_add(
            self.AUTOHIDE_TIMEOUT,
            self._autohide_timeout_cb,
        )
        self._autohide_timeout = srcid

    def _cancel_autohide_timeout(self):
        """Cancels any pending auto-hide"""
        if not self._autohide_timeout:
            return
        GLib.source_remove(self._autohide_timeout)
        self._autohide_timeout = None

    def _autohide_timeout_cb(self):
        """Hide auto-hide widgets when the auto-hide timer finishes"""
        self._hide_autohide_widgets()
        return False

    # Autohide mode: auto-reveal timer

    def _start_autoreveal_timeout(self, widget):
        """Start a timer to reveal the widget after a brief period
        of edge contact
        """
        if not self._autoreveal_timeout:
            logger.debug("Starting autoreveal timeout (%d milliseconds)",
                         self.AUTOHIDE_REVEAL_TIMEOUT)
        else:
            self._cancel_autoreveal_timeout()
        srcid = GLib.timeout_add(
            self.AUTOHIDE_REVEAL_TIMEOUT,
            self._autoreveal_timeout_cb,
            widget,
        )
        self._autoreveal_timeout.append(srcid)

    def _cancel_autoreveal_timeout(self):
        """Cancels any pending auto-reveal"""
        if not self._autoreveal_timeout:
            return
        for timer in self._autoreveal_timeout:
            GLib.source_remove(timer)
        self._autoreveal_timeout = []

    def _autoreveal_timeout_cb(self, widget):
        """Show widgets when the auto-reveal timer finishes"""
        widget.show_all()
        self._autoreveal_timeout = []
        return False

    ## Autohide mode: event handling on the canvas widget

    def _connect_autohide_events(self):
        """Start listening for autohide events"""
        if self._fs_event_handlers:
            return
        evwidget = self.get_canvas()
        if not evwidget:
            return
        mask = (
            Gdk.EventMask.POINTER_MOTION_HINT_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK |
            Gdk.EventMask.LEAVE_NOTIFY_MASK |
            Gdk.EventMask.ENTER_NOTIFY_MASK
        )
        evwidget.add_events(mask)
        handlers = [("motion-notify-event", self._fs_motion_cb),
                    ("leave-notify-event", self._fs_leave_cb),
                    ("enter-notify-event", self._fs_enter_cb)]
        for event_name, handler_callback in handlers:
            handler_id = evwidget.connect(event_name, handler_callback)
            self._fs_event_handlers.append((evwidget, handler_id))

    def _disconnect_autohide_events(self):
        """Stop listening for autohide events"""
        for evwidget, handler_id in self._fs_event_handlers:
            evwidget.disconnect(handler_id)
        self._fs_event_handlers = []

    def _fs_leave_cb(self, widget, event):
        """Handles leaving the canvas in fullscreen"""
        assert self._is_fullscreen
        # if event.state & self._ALL_BUTTONS_MASK:
        #    # "Starting painting", except not quite.
        #    # Can't use this: there's no way of distinguishing it from
        #    # resizing a floating window. Hiding the window being resized
        #    # breaks window management badly! (Xfce 4.10)
        #    self._cancel_autohide_timeout()
        #    self._hide_autohide_widgets()
        if event.mode == Gdk.CrossingMode.UNGRAB:
            # Finished painting. To appear more consistent with a mouse,
            # restart the hide timer now rather than waiting for a motion
            # event.
            self._start_autohide_timeout()
        elif event.mode == Gdk.CrossingMode.NORMAL:
            # User may be using a sidebar. Leave it open.
            self._cancel_autohide_timeout()
        return False

    def _fs_enter_cb(self, widget, event):
        """Handles entering the canvas in fullscreen"""
        assert self._is_fullscreen
        # If we're safely in the middle, the autohide timer can begin now.
        if not self._get_bumped_edges(widget, event):
            self._start_autohide_timeout()
        return False

    def _fs_motion_cb(self, widget, event):
        """Handles edge bumping and other rollovers in fullscreen mode"""
        assert self._is_fullscreen
        # Firstly, if the user appears to be drawing, be as stable as we can.
        if event.state & self._ALL_BUTTONS_MASK:
            self._cancel_autohide_timeout()
            self._cancel_autoreveal_timeout()
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
        if show_floating:
            for win in self._floating:
                win.show_all()
            self._cancel_autohide_timeout()
            return False
        # Edge bumping
        # Bump the mouse into the edge of the screen to get back the stuff
        # that was hidden there, similar to media players etc.
        edges = self._get_bumped_edges(widget, event)
        if not edges:
            self._start_autohide_timeout()
            self._cancel_autoreveal_timeout()
            return False
        if edges & self._EDGE_TOP and self.header_bar:
            self._start_autoreveal_timeout(self.header_bar)
        if edges & self._EDGE_BOTTOM and self.footer_bar:
            self._start_autoreveal_timeout(self.footer_bar)
        if edges & self._EDGE_LEFT and not self._lstack.is_empty():
            self._start_autoreveal_timeout(self._lscrolls)
        if edges & self._EDGE_RIGHT and not self._rstack.is_empty():
            self._start_autoreveal_timeout(self._rscrolls)

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

    ## Tool widget tab & title updates

    def _tool_widget_rebadged(self, factory, product, old_params, new_params):
        """Internal: update UI elements when the ID of a tool widget changes

        For parameterized ones like the brush group tool widget, the tooltip
        and titlebar are dependent on the identity strings and must be update
        when the tab is renamed.

        """
        self._update_tool_widget_ui(product)

    def _update_tool_widget_ui(self, widget):
        """Internal: update UI elements for a known descendent tool widget"""
        page = widget.get_parent()
        notebook = page.get_parent()
        notebook.update_tool_widget_ui(widget)


class ToolStack (Gtk.EventBox):
    """Vertical stack of tool widget groups

    The layout has movable dividers between groups of tool widgets, and an
    empty group on the end which accepts tabs dragged to it. The groups are
    implemented as `Gtk.Notebook`s, but that interface is not exposed.
    ToolStacks are built up from layout definitions represented by simple
    types: see `Workspace` and `build_from_layout()` for details.

    """

    ## Behavioural constants

    RESIZE_STICKINESS = 20

    ## GObject integration (type name, properties)

    __gtype_name__ = 'MyPaintToolStack'

    workspace = GObject.Property(
        type=Workspace,
        flags=GObject.PARAM_READWRITE,
        nick='Workspace',
        blurb='The central Workspace object, used to coordinate drags',
        default=None
    )

    ## Internal classes: Paned/Notebook tree elements

    class _Paned (Gtk.VPaned):
        """GtkVPaned specialization acting as an intermediate node in a tree"""

        ## Construction

        def __init__(self, toolstack, placeholder):
            """Initialize, replacing and splitting an existing placeholder.

            :param toolstack: ancestor tool stack object
            :type toolstack: ToolStack
            :param placeholder: empty placeholder notebook, must be packed as
              the child2 of another ToolStack._Paned, or as the first child of
              a ToolStack.
            :type placeholder: GtkNotebook

            The old placeholder will be removed from its parent,
            and re-packed as the child1 of the new paned.
            A new placeholder is created as the new paned's child2.
            The new paned is then packed to replace the old placeholder
            in its former parent.

            """
            super(ToolStack._Paned, self).__init__()
            self._toolstack = toolstack
            parent = placeholder.get_parent()
            assert parent is not None, "'placeholder' must have a parent"
            self.set_border_width(0)
            if isinstance(parent, Gtk.Paned):
                assert placeholder is not parent.get_child1()
                assert placeholder is parent.get_child2()
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

            # Initial sizing and allocation
            self._initial_divider_position = None
            self._first_alloc_id = self.connect("size-allocate",
                                                self._first_alloc_cb)

            # Don't hide stuff in 3.16+
            try:
                self.set_wide_handle(True)
            except AttributeError:
                pass

        ## Custom widget packing

        def pack1_tool_widget_notebook(self, notebook):
            """Pack a notebook intended for tool widgets as child1.
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

        def _first_alloc_cb(self, widget, alloc):
            """Try to allocate child widgets their natural size when alloced.
            """
            # Normally, if child widgets declare a real minimum size then in a
            # structure like this they'll be allocated their minimum size even
            # when there's enough space to give them their natural size. As a
            # workaround, set the bar position on the very first size-allocate
            # event to the best compromise we can calculate.

            # Child natural and minimum heights.
            c1 = self.get_child1()
            c2 = self.get_child2()
            if not (c1 and c2):
                return
            c1min, c1nat = c1.get_preferred_height_for_width(alloc.width)
            c2min, c2nat = c2.get_preferred_height_for_width(alloc.width)

            # Disconnect the handler; only run the 1st time.
            self.disconnect(self._first_alloc_id)
            self._first_alloc_id = None

            # If ToolStack.build_from_layout set an initial position,
            # then code elsewhere will be allocating a size.
            if self._initial_divider_position is not None:
                return

            # Get handle size
            handle_size = GObject.Value()
            handle_size.init(int)
            handle_size.set_int(12)  # conservative initial guess
            self.style_get_property("handle-size", handle_size)
            bar_height = handle_size.get_int()
            # Strategy here is to try and give one child widget its natural
            # size first, slightly favouring the first (top) child.  We
            # could be more egalitarian by inspecting the deep structure.
            pos = -1
            if c1nat + c2min <= alloc.height - bar_height:
                pos = c1nat
            elif c1min + c2nat <= alloc.height - bar_height:
                pos = alloc.height - c2nat - bar_height
            elif c1min + c2min <= alloc.height - bar_height:
                pos = alloc.height - c2min - bar_height

            # The position setting must be done outside this handler
            # or it'll look weird.
            GLib.idle_add(self.set_position, pos)

    ## Notebook

    class _Notebook (Gtk.Notebook):
        """Tabbed notebook containng a tool widget group"""

        ## Behavioural constants

        NOTEBOOK_GROUP_NAME = 'mypaint-workspace-layout-group'
        PLACEHOLDER_HEIGHT = 8
        PLACEHOLDER_WIDTH = 16
        TAB_ICON_SIZE = Gtk.IconSize.MENU  # FIXME: use a central setting
        ACTION_BUTTON_ICON_SIZE = TAB_ICON_SIZE
        TAB_TOOLTIP_ICON_SIZE = Gtk.IconSize.DIALOG

        ## Construction

        def __init__(self, toolstack):
            """Initialise, with an ancestor ToolStack.

            :param toolstack: the ancestor ToolStack

            """
            super(ToolStack._Notebook, self).__init__()
            self._toolstack = toolstack
            assert self._toolstack is not None
            self.set_group_name(self.NOTEBOOK_GROUP_NAME)
            self.connect("create-window", self._create_window_cb)
            self.connect("page-added", self._page_added_cb)
            self.connect("page-removed", self._page_removed_cb)
            self.connect("switch-page", self._switch_page_cb)
            self.connect_after("drag-begin", self._drag_begin_cb)
            self.connect_after("drag-end", self._drag_end_cb)
            self.set_scrollable(True)
            # Minimum sizing, for the placeholder case
            self.set_size_request(8, -1)
            # Action buttons
            action_hbox = Gtk.HBox()
            action_hbox.set_homogeneous(True)
            action_hbox.set_spacing(0)
            self.set_action_widget(action_hbox, Gtk.PackType.END)
            self.connect("show", lambda *a: action_hbox.show_all())
            # Properties button
            btn = borderless_button(icon_name="mypaint-tab-options-symbolic",
                                    size=self.ACTION_BUTTON_ICON_SIZE)
            btn.connect("clicked", self._properties_button_clicked_cb)
            action_hbox.pack_start(btn, False, False, 0)
            self._properties_button = btn
            btn.set_sensitive(False)

            # Sidebar swapper button
            btn = borderless_button(
                icon_name="mypaint-tab-sidebar-swap-symbolic",
                size=self.ACTION_BUTTON_ICON_SIZE,
            )
            btn.connect("clicked", self._sidebar_swap_button_clicked_cb)
            action_hbox.pack_start(btn, False, False, 0)
            self._sidebar_swap_button = btn

            # Close tab button
            btn = borderless_button(icon_name="mypaint-close-symbolic",
                                    size=self.ACTION_BUTTON_ICON_SIZE)
            btn.connect("clicked", self._close_button_clicked_cb)
            action_hbox.pack_start(btn, False, False, 0)
            self._close_button = btn

        ## Tool widget pages

        def append_tool_widget_page(self, tool_widget):
            """Appends a tool widget as a new page/tab."""
            page = _ToolWidgetNotebookPage()
            page.add(tool_widget)
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
            stack = self._toolstack
            GLib.idle_add(stack._update_structure)
            # Reinstate the previous size on the divider
            # if this is the result of dragging a tab out
            # into a new window.
            try:
                size = child.__prev_size
            except AttributeError:
                return
            if self is not stack._get_first_notebook():
                return
            if self.get_n_pages() != 1:
                return
            # The size-setting must be done outside the event handler
            # for it to take effect.
            w, h = size
            GLib.idle_add(stack._set_first_paned_position, h)

        def _page_removed_cb(self, notebook, child, page_num):
            GLib.idle_add(self._toolstack._update_structure)

        ## ToolStack structure: utility methods

        def split_former_placeholder(self):
            """Splits the space used by a placeholder after a tab drag into it.

            After the placeholder has a tab dragged into it, it can no longer
            fill the placeholder's role. This method creates a new empty
            placeholder after it in the stack, and updates the hierarchy
            appropriately. It also tries to retain the dragged-in tab's page's
            size as much as possible by setting paned divider positions
            appropriately.

            """
            # Bail if not a former placeholder
            assert self.get_n_pages() > 0
            toolstack = self._toolstack
            toolstack_was_empty = self.get_parent() is toolstack
            assert toolstack_was_empty \
                or self is self.get_parent().get_child2()
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

        ## Action buttons

        def _switch_page_cb(self, notebook, page, page_num):
            tool_widget = page.get_child()
            has_properties = hasattr(tool_widget, "tool_widget_properties")
            self._properties_button.set_sensitive(has_properties)
            title = _tool_widget_get_title(tool_widget)
            close_tooltip = C_(
                "workspace: sidebar tabs: button tooltips",
                u"{tab_title}: close tab",
            ).format(tab_title=title)
            props_tooltip = C_(
                "workspace: sidebar tabs: button tooltips",
                u"{tab_title}: tab options and properties",
            ).format(tab_title=title)
            swap_tooltip = C_(
                "workspace: sidebar tabs: button tooltips",
                u"{tab_title}: move tab to other sidebar",
            ).format(tab_title=title)
            if not has_properties:
                props_tooltip = u""
            self._properties_button.set_tooltip_text(props_tooltip)
            self._close_button.set_tooltip_text(close_tooltip)
            self._sidebar_swap_button.set_tooltip_text(swap_tooltip)

        def _close_button_clicked_cb(self, button):
            """Remove the current page (close button "clicked" event callback)

            Ultimately fires the ``tool_widget_removed()`` @event of the owning
            workspace.

            """
            page_num = self.get_current_page()
            page = self.get_nth_page(page_num)
            if page is not None:
                GLib.idle_add(self._deferred_remove_tool_widget, page)
            # As of 3.14.3, removing the tool widget must be deferred
            # until after internal handling of button-release-event
            # by the notebook itself. gtk_notebook_button_release()
            # needs the structure to be unchanging or it'll segfault.

        def _deferred_remove_tool_widget(self, page):
            assert page is not None
            tool_widget = page.get_child()
            self._toolstack.remove_tool_widget(tool_widget)
            return False

        def _properties_button_clicked_cb(self, button):
            """Invoke the current page's properties callback."""
            page_num = self.get_current_page()
            page = self.get_nth_page(page_num)
            tool_widget = page.get_child()
            if hasattr(tool_widget, "tool_widget_properties"):
                tool_widget.tool_widget_properties()

        def _sidebar_swap_button_clicked_cb(self, button):
            """Switch the current page's sidebar ("clicked" event handler)

            Ultimately fires the tool_widget_removed() and
            tool_widget_added() @events of the owning workspace.

            """
            page_num = self.get_current_page()
            page = self.get_nth_page(page_num)
            if page is not None:
                GLib.idle_add(self._deferred_swap_tool_widget_sidebar, page)

        def _deferred_swap_tool_widget_sidebar(self, page):
            try:
                workspace = self._toolstack.workspace
            except AttributeError:
                logger.warning("swap: notebook is not in a workspace")
                return False

            assert page is not None
            tool_widget = page.get_child()

            src_stack = self._toolstack
            if src_stack is workspace._rstack:
                targ_stack = workspace._lstack
            else:
                targ_stack = workspace._rstack

            src_stack.remove_tool_widget(tool_widget)
            targ_stack.add_tool_widget(tool_widget)
            targ_stack.reveal_tool_widget(tool_widget)

            return False

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
            # Create and show the parent window for a dragged-out tab.
            win = ToolStackWindow()
            self._toolstack.workspace.floating_window_created(win)
            win.stack.workspace = self._toolstack.workspace
            w, h = page.__prev_size
            new_nb = win.stack._get_first_notebook()
            win.stack._append_new_placeholder(new_nb)
            # Initial position. Hopefully this will work.
            win.move(x, y)
            win.set_default_size(w, h)
            win.show_all()
            # Tell GTK which Notebook to move the tab to.
            return new_nb

        ## Tab labels

        @classmethod
        def _make_tab_label(cls, tool_widget):
            """Creates and returns a tab label widget for a tool widget"""
            label = Gtk.Image()
            lsize = cls.TAB_ICON_SIZE
            icon_pixbuf, icon_name = _tool_widget_get_icon(tool_widget, lsize)
            if icon_pixbuf:
                label.set_from_pixbuf(icon_pixbuf)
            else:
                label.set_from_icon_name(icon_name, lsize)
            title = _tool_widget_get_title(tool_widget)
            desc = getattr(tool_widget, "tool_widget_description", None)
            ttsize = cls.TAB_TOOLTIP_ICON_SIZE
            tooltip_icon_pixbuf, tooltip_icon_name = _tool_widget_get_icon(
                tool_widget,
                ttsize
            )
            label.connect("query-tooltip", cls._tab_label_tooltip_query_cb,
                          title, desc, tooltip_icon_pixbuf, tooltip_icon_name)
            label.set_property("has-tooltip", True)
            return label

        @classmethod
        def _tab_label_tooltip_query_cb(cls, widget, x, y, kbd, tooltip,
                                        title, desc, icon_pixbuf, icon_name):
            """The query-tooltip routine for tool widgets"""
            if icon_pixbuf is not None:
                tooltip.set_icon(icon_pixbuf)
            else:
                ttsize = cls.TAB_TOOLTIP_ICON_SIZE
                tooltip.set_icon_from_icon_name(icon_name, ttsize)
            if desc is not None:
                markup_tmpl = "<b>{title}</b>\n{desc}"
            else:
                markup_tmpl = "<b>{title}</b>"
            tooltip.set_markup(markup_tmpl.format(
                title = lib.xml.escape(title),
                desc = lib.xml.escape(desc),
            ))
            return True

        ## Updates

        def update_tool_widget_ui(self, tool_widget):
            # Update the tab label
            logger.debug("notebook: updating UI parts for %r", tool_widget)
            label = self._make_tab_label(tool_widget)
            page = tool_widget.get_parent()
            self.set_tab_label(page, label)
            label.show_all()
            # Window title too, if that's appropriate
            self._toolstack._update_window_title()

    ## Construction

    def __init__(self):
        """Constructs a new stack with a single placeholder group"""
        Gtk.EventBox.__init__(self)
        self.add(ToolStack._Notebook(self))
        self.__initial_paned_positions = []

    ## Setup from layout descriptions (pre-realize)

    def build_from_layout(self, desc, init_sizes_state=None):
        """Loads groups and pages from a layout description

        :param desc: stack definition
        :type desc: dict
        :param init_sizes_state: toplevel window state transition on
            which to set the group dividers' initial positions. If left
            unset, set the sizes immediately.
        :type init_sizes_state: Gdk.WindowState
        :rtype: int
        :returns: the number of groups added

        The `desc` parameter has the following keys and values:

        * w: integer width (ignored here)
        * h: integer height (ignored here)
        * groups: list of group definitions - see below

        Width and height may be of relevance to the parent widget, but are not
        consumed by this method. `get_layout()` writes them, however.  Each
        group definition is a dict with the following keys and values.

        * tools: a list of tool definitions - see below
        * h: integer height: used here to set the height of the group
        * w: integer width (ignored here)

        Each tool definition is a tuple of the form (GTYPENAME,*CONSTRUCTARGS).
        GTYPENAME is a string containing a GType name which is used for finding
        and constructing the tool instance. CONSTRUCTARGS is currently ignored.

        """
        next_nb = self._get_first_notebook()
        factory = self.workspace._tool_widgets
        num_groups_added = 0
        for group_desc in desc.get("groups", []):
            assert next_nb.get_n_pages() == 0
            # Only add unique tool widgets. Assume this is being called on
            # startup, with an initially empty factory cache.
            tool_widgets = []
            for tool_desc in group_desc.get("tools", []):
                if factory.cache_has(*tool_desc):
                    logger.warning("Duplicate entry %r ignored", tool_desc)
                    continue
                logger.debug("build_from_layout: building tool %r",
                             tool_desc)
                try:
                    tool_widget = factory.get(*tool_desc)
                    tool_widgets.append(tool_widget)
                except objfactory.ConstructError as ex:
                    logger.error("build_from_layout: %s", ex.message)
            # Group might be empty if construction fails or if everything's a
            # duplicate.
            if not tool_widgets:
                logger.debug("Empty tab group in workspace, not added")
                continue
            # We have something to add, so create a new Notebook with the
            # pages, and move the insert ref
            nb = next_nb
            next_nb = self._append_new_placeholder(nb)
            for tool_widget in tool_widgets:
                nb.append_tool_widget_page(tool_widget)
                if self.workspace:
                    GLib.idle_add(
                        self.workspace.tool_widget_added,
                        tool_widget,
                    )
            active_page = group_desc.get("active_page", -1)
            nb.set_current_page(active_page)
            num_groups_added += 1
            # Position the divider between the new notebook and the next.
            group_min_h = 1
            group_h = int(group_desc.get("h", group_min_h))
            group_h = max(group_min_h, group_h)
            nb_parent = nb.get_parent()
            assert isinstance(nb_parent, ToolStack._Paned)
            nb_parent._initial_divider_position = group_h
        return num_groups_added

    def get_layout(self):
        """Returns a description of the current layout using simple types

        :rtype: dict

        See `build_from_layout()` for details of the dict which is returned.

        """
        group_descs = []
        factory = self.workspace._tool_widgets
        for nb in self._get_notebooks():
            tool_descs = []
            for page in nb:
                tool_widget = page.get_child()
                tool_desc = factory.identify(tool_widget)
                if tool_desc:
                    tool_descs.append(tool_desc)
            active_page = nb.get_current_page()
            group_desc = {"tools": tool_descs, "active_page": active_page}
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

    ## Initial layout (post-realize)

    def _complete_initial_layout(self):
        """Finish initial layout; called after toplevel win is positioned"""
        # Init tool group sizes by setting vpaned positions
        for paned in self._get_paneds():
            if paned._initial_divider_position:
                pos = paned._initial_divider_position
                GLib.idle_add(paned.set_position, pos)

    ## Tool widgets

    def add_tool_widget(self, widget, maxnotebooks=None, maxpages=3):
        """Tries to find space for, then add and show a tool widget

        Finding space is based on constraints, adjustable via the parameters.
        The process is driven by `Workspace.add_tool_widget()`.

        :param widget: the widget that needs adoption.
        :type widget: Gtk.Widget created by the Workspace's factory.
        :param maxnotebooks: never make more than this many groups in the stack
        :type maxnotebooks: int
        :param maxpages: never make more than this many pages in a group
        :type maxpages: int
        :return: whether space was found for the widget
        :rtype: bool

        The idea is to try repeatedly with gradually relaxing constraint
        parameters across all stacks in the system until space is found
        somewhere.

        """
        # Try to find a notebook with few enough pages.
        # It might be the zero-pages placeholder on the end.
        target_notebook = None
        notebooks = self._get_notebooks()
        assert len(notebooks) > 0, (
            "There should always be at least one Notebook widget "
            "in any ToolStack."
        )
        for nb in notebooks:
            if nb.get_n_pages() < maxpages:
                target_notebook = nb
                break
        # The placeholder tends to be recreated in idle routines,
        # so it may not be present on the end of the stack just yet.
        if target_notebook is None:
            assert nb.get_n_pages() > 0
            new_placeholder = nb.split_former_placeholder()
            target_notebook = new_placeholder
        # Adding a page to the placeholder would result in a split
        # in the idle routine later. Check constraint now.
        if maxnotebooks and (target_notebook.get_n_pages() == 0):
            n_populated_notebooks = len([
                n for n in notebooks
                if n.get_n_pages() > 0
            ])
            if n_populated_notebooks >= maxnotebooks:
                return False
        # We're good to go.
        target_notebook.append_tool_widget_page(widget)
        if self.workspace:
            GLib.idle_add(self.workspace.tool_widget_added, widget)
        return True

    def remove_tool_widget(self, widget):
        """Removes a tool widget from the stack, hiding it

        :param widget: the GType name of the tab to be removed.
        :type widget: Gtk.Widget created by the Workspace's factory
        :rtype: bool
        :returns: whether the widget was removed

        """
        target_notebook = None
        target_index = None
        target_page = None
        for notebook in self._get_notebooks():
            for index in xrange(notebook.get_n_pages()):
                page = notebook.get_nth_page(index)
                if widget is page.get_child():
                    target_index = index
                    target_notebook = notebook
                    target_page = page
                    break
            if target_notebook:
                break
        if target_notebook:
            assert target_page is not None
            assert target_index is not None
            logger.debug("Removing tool widget i=%d, p=%r, n=%r",
                         target_index, target_page, target_notebook)
            target_page.hide()
            widget.hide()
            target_page.remove(widget)
            target_notebook.remove_page(target_index)
            target_page.destroy()
            if self.workspace:
                self.workspace.tool_widget_removed(widget)
            return True
        return False

    def is_empty(self):
        """Returns true if this stack contains only a tab drop placeholder"""
        widget = self.get_child()
        if isinstance(widget, Gtk.Paned):
            return False
        assert isinstance(widget, Gtk.Notebook)
        return widget.get_n_pages() == 0

    def reveal_tool_widget(self, widget):
        """Reveals a widget in this tool stack"""
        toplevel = self.get_toplevel()
        if widget is None or (widget.get_toplevel() is not toplevel):
            logger.warning("Can't reveal %r: not in this toolstack", widget)
            return
        # Show the stack's toplevel, or unfold sidebars
        if toplevel is self.workspace.get_toplevel():
            logger.debug("Showing %r (ancestor of freshly shown tool %r)",
                         self, widget)
            scrolls = self.get_parent().get_parent()
            scrolls.show_all()
        else:
            toplevel.present()
        # Switch to the widget's tab
        page = widget.get_parent()
        nb = page.get_parent()
        page_num = nb.page_num(page)
        nb.set_current_page(page_num)

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

    def _set_first_paned_position(self, size):
        widget = self.get_child()
        if isinstance(widget, Gtk.Paned):
            widget.set_position(size)
        return GLib.SOURCE_REMOVE

    ## Group size management (somewhat dubious)

    def _get_paneds(self):
        child = self.get_child()
        if child is None:
            return []
        queue = [child]
        result = []
        while len(queue) > 0:
            widget = queue.pop(0)
            if isinstance(widget, Gtk.Paned):
                result.append(widget)
                queue.append(widget.get_child1())
                queue.append(widget.get_child2())
        return result

    def do_size_allocate(self, alloc):
        # When the size changes, manage the divider position of the final
        # paned, shrinking or growing the final set of tabs.
        paneds = self._get_paneds()
        if paneds:
            final_paned = paneds[-1]

            pos = final_paned.get_position()
            max_pos = final_paned.get_property("max-position")

            if max_pos - pos <= self.RESIZE_STICKINESS:
                final_paned.set_position(alloc.height)

        Gtk.EventBox.do_size_allocate(self, alloc)

    ## Paned/Notebook tree structure

    def _append_new_placeholder(self, old_placeholder):
        """Appends a new placeholder after a current or former placeholder.
        """
        paned = ToolStack._Paned(self, old_placeholder)
        return paned.get_child2()

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
            assert nb is nb_parent.get_child1()
            assert nb is not nb_parent.get_child2()
            sib = nb_parent.get_child2()
            nb_parent.remove(nb)
            nb_parent.remove(sib)
            if isinstance(nb_grandparent, Gtk.Paned):
                assert nb_parent is not nb_grandparent.get_child1()
                assert nb_parent is nb_grandparent.get_child2()
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
                parent.destroy()
                self.workspace.floating_window_destroyed(parent)
            else:
                self.hide()
            return

        # Floating window title too, if appropriate
        self._update_window_title()

    def _update_window_title(self):
        """Updates the title of parent ToolStackWindows"""
        toplevel = self.get_toplevel()
        if not isinstance(toplevel, ToolStackWindow):
            return
        logger.debug("toolstack: updating title of %r", toplevel)
        page_titles = []
        for nb in self._get_notebooks():
            for page in nb:
                tool_widget = page.get_child()
                title = _tool_widget_get_title(tool_widget)
                page_titles.append(title)
        toplevel.update_title(page_titles)


class _ToolWidgetNotebookPage (Gtk.Frame):
    """Page widget container within a notebook.

    Intercepts drag-related events which have propagated up from the
    tool widget itself to prevent them propagating out to the containing
    notebook. This prevents accidental drags of the tab itself starting
    since 3.20: https://github.com/mypaint/mypaint/issues/643

    """

    def __init__(self):
        super(_ToolWidgetNotebookPage, self).__init__()
        self.set_shadow_type(Gtk.ShadowType.NONE)
        self.connect("button-press-event", self._event_sink)
        self.connect("button-release-event", self._event_sink)
        self.connect("motion-notify-event", self._event_sink)

    def _event_sink(self, *args, **kwargs):
        return True


class ToolStackWindow (Gtk.Window):
    """A floating utility window containing a single `ToolStack`"""

    ## Class constants

    __gtype_name__ = "MyPaintToolStackWindow"
    _AGGRESSIVE_POSITIONING_HACK = False

    ## Construction

    def __init__(self):
        Gtk.Window.__init__(self)
        self.set_type_hint(Gdk.WindowTypeHint.UTILITY)
        self.set_deletable(False)
        self.connect("realize", self._realize_cb)
        self.connect("destroy", self._destroy_cb)
        self.stack = ToolStack()  #: The ToolStack child of the window
        self.add(self.stack)
        self.update_title([])
        # Position tracking
        self._layout_position = None  # last cfg'd position and content size
        self._frame_size = None  # used for rollover accuracy, not saved
        self.connect("configure-event", self._configure_cb)
        # On-map hacks
        self._onmap_position = None  # position to be forced on window map
        self._mapped_once = False
        self.connect("map", self._map_cb)
        self.connect("hide", self._hide_cb)

    ## Setup from layout definitions (pre-realize)

    def build_from_layout(self, layout):
        """Build the window's contents from a layout description.

        :param dict layout: A layout defnition
        :returns: the number of groups added (can be zero)
        :rtype: int

        """
        logger.debug("build_from_layout %r", self)
        n_added = self.stack.build_from_layout(layout.get("contents", {}))
        pos = layout.get("position", None)
        if pos:
            self._layout_position = pos.copy()
        return n_added

    def get_layout(self):
        """Get the window's position and contents in simple dict form.
        """
        return {
            "position": self._layout_position,
            "contents": self.stack.get_layout(),
        }

    ## Window lifecycle events (initial state, position tracking)

    def _realize_cb(self, widget):
        """Set the initial position (with lots of sanity checks)"""
        lpos = self._layout_position
        if lpos is None:
            return
        self._onmap_position = set_initial_window_position(self, lpos)

    def _map_cb(self, widget):
        """Window map event actions"""
        toplevel = None
        workspace = self.stack.workspace
        if workspace:
            toplevel = workspace.get_toplevel()
        # Things we only need to do on the first window map
        if not self._mapped_once:
            self._mapped_once = True
            if toplevel:
                self.set_transient_for(toplevel)
            if workspace:
                workspace._floating.add(self)
            win = widget.get_window()
            decor = (
                Gdk.WMDecoration.TITLE
                | Gdk.WMDecoration.BORDER
                | Gdk.WMDecoration.RESIZEH
            )
            win.set_decorations(decor)
            wmfuncs = Gdk.WMFunction.RESIZE | Gdk.WMFunction.MOVE
            win.set_functions(wmfuncs)
        # Hack to force an initial x,y position to be what was saved, used
        # as a workaround for WM bugs and misfeatures.
        # Forcing the position up front rather than in an idle handler
        # avoids flickering in Xfce 4.8, when this is necessary.
        # Xfce 4.8 requires position forcing for second and subsequent
        # map events too, if a window has been resized due its content growing.
        # Hopefully we never have to do this twice. Once is too much really.
        if self._onmap_position is not None:
            if self._AGGRESSIVE_POSITIONING_HACK:
                self._set_onmap_position(False)
                GLib.idle_add(self._set_onmap_position, True)
            else:
                self._set_onmap_position(True)
        # Prevent subwindows from taking keyboard focus from the main window
        # in Metacity by presenting it again. https://gna.org/bugs/?17899
        # Still affects GNOME 3.14.
        # https://github.com/mypaint/mypaint/issues/247
        if toplevel:
            GLib.idle_add(lambda *a: toplevel.present())

    def _set_onmap_position(self, reset):
        """Hack to set the requested position, as much as one can

        Window managers don't always get it right when the window is initially
        positioned, and some don't keep window positions always when a window
        is hidden and later re-shown. Doing a move() in a map handler improves
        the user experience vastly in these WMs.

        """
        if self._onmap_position:
            self.move(*self._onmap_position)
            if reset:
                self._onmap_position = None
        return False

    def _configure_cb(self, widget, event):
        """Track the window size and position when it changes"""
        frame = self.get_window().get_frame_extents()
        x = max(0, frame.x)
        y = max(0, frame.y)
        # The content size, and upper-left frame position; will be saved
        self._layout_position = dict(x=x, y=y, w=event.width, h=event.height)
        # Frame extents, used internally for rollover accuracy; not saved
        self._frame_size = frame.width, frame.height

    def _hide_cb(self, widget):
        """Ensure a correct position after the next window map"""
        if self._layout_position is None:
            return
        pos = (self._layout_position.get("x", None),
               self._layout_position.get("y", None))
        if None not in pos:
            self._onmap_position = pos

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
        if not (self._layout_position and self._frame_size):
            return False
        fx = self._layout_position.get("x", None)
        fy = self._layout_position.get("y", None)
        fw, fh = self._frame_size
        if None in (fx, fy, fw, fh):
            return False
        return x >= fx-b and x <= fx+fw+b and y >= fy-b and y <= fy+fh+b

    ## Window title

    def update_title(self, tool_widget_titles):
        """Update the title from a list of strings"""
        titles = [unicode(s) for s in tool_widget_titles]
        workspace = self.stack.workspace
        if workspace is not None:
            title_sep = unicode(workspace.floating_window_title_separator)
            title = title_sep.join(titles)
            title_suffix = unicode(workspace.floating_window_title_suffix)
            if title_suffix:
                title += unicode(title_suffix)
            logger.debug(u"Renamed floating window title to \"%s\"", title)
            self.set_title(title)


## Convenience base classes for implementing tool widgets

class SizedVBoxToolWidget (Gtk.VBox):
    """Base class for VBox tool widgets, with convenient natural height setting.

    This mixin can be used for tool widgets implemented as `GtkVBox`es to give
    them a default natural height which might be greater than the sum of their
    consituent widgets' minimum heights.

    """

    #: Suggested natural height for the widget.
    SIZED_VBOX_NATURAL_HEIGHT = TOOL_WIDGET_NATURAL_HEIGHT_TALL

    def do_get_request_mode(self):
        return Gtk.SizeRequestMode.HEIGHT_FOR_WIDTH

    def do_get_preferred_width(self):
        minw, natw = Gtk.VBox.do_get_preferred_width(self)
        minw = max(minw, TOOL_WIDGET_MIN_WIDTH)
        natw = max(natw, TOOL_WIDGET_MIN_WIDTH)
        return minw, max(minw, natw)

    def do_get_preferred_height_for_width(self, width):
        minh, nath = Gtk.VBox.do_get_preferred_height_for_width(self, width)
        nath = max(nath, self.SIZED_VBOX_NATURAL_HEIGHT)
        minh = max(minh, TOOL_WIDGET_MIN_HEIGHT)
        return minh, max(minh, nath)

## Utility functions


def _tool_widget_get_title(widget):
    """Returns the title to use for a tool-widget.

    :param widget: a tool widget
    :type widget: Gtk.Widget
    :rtype: unicode

    """
    for attr in ("tool_widget_title", "__gtype_name__"):
        title = getattr(widget, attr, None)
        if title is not None:
            return unicode(title)
    return unicode(widget.__class__.__name__)


def _tool_widget_get_icon(widget, icon_size):
    """Returns the pixbuf or icon name to use for a tool widget

    :param widget: a tool widget
    :param icon_size: a registered Gtk.IconSize
    :returns: a pixbuf or an icon name: as a pair, one of which is None
    :rtype: (GdkPixbuf.Pixbuf, str)

    Use whichever of the return values is not None. To get the pixbuf or icon
    name, one or both of

    * ``widget.tool_widget_get_icon_pixbuf(pixel_size)``
    * ``widget.tool_widget_icon_name``

    are tried, in that order. The former should create and return a new pixbuf,
    the latter should be an icon name string.
    """
    # Try the pixbuf method first.
    # Only brush group tool widgets will define this, typically.
    size_valid, width_px, height_px = Gtk.icon_size_lookup(icon_size)
    if not size_valid:
        return None
    size_px = min(width_px, height_px)
    if hasattr(widget, "tool_widget_get_icon_pixbuf"):
        pixbuf = widget.tool_widget_get_icon_pixbuf(size_px)
        if pixbuf:
            return (pixbuf, None)
    # Try the icon name property. Fallback is a name we know will work.
    icon_name = getattr(widget, "tool_widget_icon_name", 'missing-image')
    return (None, icon_name)


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

    # Final calculated positions
    final_x, final_y = None, None
    final_w, final_h = None, None

    # Positioning arguments
    x = pos.get("x", None)
    y = pos.get("y", None)
    w = pos.get("w", None)
    h = pos.get("h", None)

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

    # Generate a sensible, positive x and y position
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

    # And a sensible, positive width and height
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
    :rtype: lib.helpers.Rect

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
        geom = lib.helpers.Rect.new_from_gdk_rectangle(geom)
    else:
        logger.warning(
            "gdk_screen_get_monitor_geometry() returned NULL: "
            "using screen size instead as a fallback."
        )
        geom = lib.helpers.Rect(0, 0, screen.get_width(), screen.get_height())
    return geom


## Module testing (interactive, but fairly minimal)


def _test():
    logging.basicConfig(level=logging.DEBUG)
    import os
    import sys

    class _TestLabel (Gtk.Label):
        __gtype_name__ = 'TestLabel'
        tool_widget_icon_name = 'gtk-ok'
        tool_widget_description = "Just a test widget"

        def __init__(self, text):
            Gtk.Label.__init__(self, text)
            self.set_size_request(200, 150)

    class _TestSpinner (Gtk.Spinner):
        __gtype_name__ = "TestSpinner"
        tool_widget_icon_name = 'gtk-cancel'
        tool_widget_description = "Spinner test"

        def __init__(self):
            Gtk.Spinner.__init__(self)
            self.set_size_request(150, 150)
            self.set_property("active", True)

    def _tool_shown_cb(*a):
        logger.debug("TOOL-SHOWN %r", a)

    def _tool_hidden_cb(*a):
        logger.debug("TOOL-HIDDEN %r", a)

    def _floating_window_created(*a):
        logger.debug("FLOATING-WINDOW-CREATED %r", a)
    workspace = Workspace()
    workspace.floating_window_title_suffix = u" - Test"
    button = Gtk.Button("Click to close this demo")
    frame = Gtk.Frame()
    frame.add(button)
    frame.set_shadow_type(Gtk.ShadowType.IN)
    workspace.set_canvas(frame)
    window = Gtk.Window()
    window.add(workspace)
    window.set_title(os.path.basename(sys.argv[0]))
    workspace.set_size_request(600, 400)
    workspace.tool_widget_added += _tool_shown_cb
    workspace.tool_widget_removed += _tool_hidden_cb
    workspace.floating_window_created += _floating_window_created
    workspace.build_from_layout({
        'position': {'x': 100, 'y': 75, 'h': -100, 'w': -100},
        'floating': [{
            'position': {'y': -100, 'h': 189, 'w': 152, 'x': -200},
            'contents': {
                'groups': [{
                    'tools': [('TestLabel', "1"), ('TestLabel', "2")],
                }],
            }}],
        'right_sidebar': {
            'w': 400,
            'groups': [{
                'tools': [('TestSpinner',), ("TestLabel", "3")],
            }],
        },
        'left_sidebar': {
            'w': 250,
            'groups': [{
                'tools': [('TestLabel', "4"), ('TestLabel', "5")],
            }],
        },
        'maximized': False,
        'fullscreen': True,
    })
    window.show_all()

    def _quit_cb(*a):
        logger.info("Demo quit, workspace dump follows")
        print(workspace.get_layout())
        Gtk.main_quit()
    window.connect("destroy", _quit_cb)
    button.connect("clicked", _quit_cb)
    Gtk.main()


if __name__ == '__main__':
    _test()
