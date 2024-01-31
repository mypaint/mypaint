# This file is part of MyPaint.
# Copyright (C) 2014-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Toolstacks with panels that can be closed/detached/reordered"""

from __future__ import division, print_function
from warnings import warn
import logging
import weakref

from lib.gibindings import GObject
from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib

from gui.windowing import set_initial_window_position

import lib.xml
import lib.helpers
from . import objfactory
from .widgets import borderless_button
from .windowing import clear_focus
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
            win = ToolStackWindow(self._toolstack.workspace)
            self._toolstack.workspace.floating_window_created(win)
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

    def __init__(self, workspace):
        """Constructs a new stack with a single placeholder group"""
        Gtk.EventBox.__init__(self)
        self._workspace_ref = weakref.ref(workspace)
        self.add(ToolStack._Notebook(self))
        self.__initial_paned_positions = []

    @property
    def workspace(self):
        """ Returns reference to the parent workspace, or None
        """
        return self._workspace_ref()

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

    def remove_all_tool_widgets(self):
        """Remove all tool widgets in the stack, without cleanup

        This method is only intended to be used right before the
        parent of a ToolStack is destroyed. It does not remove
        any placeholders, only the individual tool widgets.
        """
        for notebook in self._get_notebooks():
            for index in xrange(notebook.get_n_pages()):
                page = notebook.get_nth_page(index)
                widget = page.get_child()
                widget.hide()
                page.remove(widget)
                if self.workspace:
                    self.workspace.tool_widget_removed(widget)

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

    def __init__(self, workspace):
        Gtk.Window.__init__(self)
        self.set_type_hint(Gdk.WindowTypeHint.UTILITY)
        self.set_deletable(False)
        self.connect("realize", self._realize_cb)
        self.connect("destroy", self._destroy_cb)
        self.connect("delete-event", self._delete_cb)
        self.connect("button-press-event", self._clear_focus)
        self.stack = ToolStack(workspace)  #: The ToolStack child of the window
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

    def _delete_cb(self, widget, event):
        self.stack.remove_all_tool_widgets()

    def _destroy_cb(self, widget):
        workspace = self.stack.workspace
        if workspace is not None:
            if self in workspace._floating:
                workspace._floating.remove(self)

    def _clear_focus(self, *args):
        clear_focus(self)

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


class SizedVBoxToolWidget (Gtk.VBox):
    """Base class for VBox tool widgets, with convenient natural height setting.

    This mixin can be used for tool widgets implemented as `GtkVBox`es to give
    them a default natural height which might be greater than the sum of their
    constituent widgets' minimum heights.

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
