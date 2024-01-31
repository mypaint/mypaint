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
import logging

from lib.gibindings import GObject
from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib

from gui.toolstack import ToolStack, ToolStackWindow
from gui.windowing import set_initial_window_position
from lib.observable import event
from . import objfactory

logger = logging.getLogger(__name__)


# Class defs


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
    floating_window_title_suffix = GObject.Property(
        type=str,
        flags=GObject.ParamFlags.READWRITE,
        nick='Floating window title suffix',
        blurb='The suffix to append to floating windows: typically a '
              'hyphen followed by the application name.',
        default=None
    )

    #: Title separator property for floating windows.
    floating_window_title_separator = GObject.Property(
        type=str,
        flags=GObject.ParamFlags.READWRITE,
        nick='Floating window title separator',
        blurb='String used to separate the names of tools in a '
              'floating window. By default, a comma is used.',
        default=", "
    )

    #: Header bar widget, to be hidden when entering fullscreen mode. This
    #: widget should be packed externally to the workspace, and to its top.
    header_bar = GObject.Property(
        type=Gtk.Widget,
        flags=GObject.ParamFlags.READWRITE,
        nick='Header bar widget',
        blurb="External Menubar/toolbar widget to be hidden when "
              "entering fullscreen mode, and re-shown when leaving "
              "it. The pointer position is also used for reveals and "
              "hides in fullscreen.",
        default=None
    )

    #: Footer bar widget, to be hidden when entering fullscreen mode. This
    #: widget should be packed externally to the workspace, and to its bottom.
    footer_bar = GObject.Property(
        type=Gtk.Widget,
        flags=GObject.ParamFlags.READWRITE,
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
        self._lstack = lstack = ToolStack(self)
        self._rstack = rstack = ToolStack(self)
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
            win = ToolStackWindow(self)
            self.floating_window_created(win)
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
    button = Gtk.Button(label="Click to close this demo")
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
