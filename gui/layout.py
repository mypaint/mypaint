# This file is part of MyPaint.
# Copyright (C) 2007-2011 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Manage a main window with a sidebar, plus a number of subwindows.
Tools can be snapped in and out of the sidebar.
Window sizes and sidebar positions are stored to user preferences.
"""

import gtk
import gobject
from gtk import gdk
from math import sqrt
from warnings import warn
import pango
from gettext import gettext as _


class LayoutManager:
    """Keeps track of tool positions, and main window state.
    """

    def __init__(self, factory, factory_opts=[], prefs=None):
        """Constructor.

            "prefs"
                should be a ref to a dict-like object. Persistent prefs
                will be read from and written back to it.
            
            "factory"
                is a callable which has the signature

                    factory(role, layout_manager, *factory_opts)

                and which will be called for each needed tool. It should
                return a value of the form:
                
                   1. (None,)
                   2. (window,)
                   3. (widget,)
                   4. (tool_widget, title_str)

                Form 1 indicates that no matching widget could be found.
                Form 2 should be used for floating dialog windows, popup
                windowrs and similar, or for "main-window". Form 3 is
                expected when the "role" parameter is "main-widget"; the
                returned widgets are packed into the main window.
        """

        if prefs is None:
            prefs = {}
        self.prefs = prefs
        self.window_group = gtk.WindowGroup()
        self.factory = factory
        self.factory_opts = factory_opts
        self.widgets = {}  # {role: <gtk.Widget>}   as returned by factory()
        self.tools = {}   # {role: Tool}
        self.subwindows = {}   # {role: <gtk.Window>}
        self.main_window = None
        self.saved_user_tools = []

    def set_main_window_title(self, title):
        """Set the title for the main window.
        """
        self.main_window.set_title(title)

    def get_widget_by_role(self, role):
        """Returns the raw widget for a particular role name.
        
        Internally cached; will invoke the factory method once for each
        unique role name the first time it's seen.
        """
        if role in self.widgets:
            return self.widgets[role]
        else:
            result = self.factory(role, self, *self.factory_opts)
            if role == 'main-window':
                self.main_window = result[0]
                assert isinstance(self.main_window, MainWindow)
                self.drag_state = ToolDragState(self)
                    # ^^^ Yuck. Would be nicer to init in constructor
                self.widgets[role] = self.main_window
                return self.main_window
            elif role == 'main-widget':
                widget = result[0]
                self.widgets[role] = widget
                return widget
            if result is None:
                return None
            else:
                widget = result[0]
                if isinstance(widget, gtk.Window):
                    self.subwindows[role] = widget
                    self.widgets[role] = widget
                    widget.set_role(role)
                    widget.set_transient_for(self.main_window)
                    return widget
                elif isinstance(widget, gtk.Widget):
                    title = result[1]
                    tool = Tool(widget, role, title, title, self)
                    self.tools[role] = tool
                    self.widgets[role] = tool
                    return widget
                else:
                    self.widgets[role] = None
                    warn("Unknown role \"%s\"" % role, RuntimeWarning,
                         stacklevel=2)
                    return None

    def get_subwindow_by_role(self, role):
        """Returns the managed subwindow for a given role, or None
        """
        if self.get_widget_by_role(role):
            return self.subwindows.get(role, None)
        return None

    def get_tool_by_role(self, role):
        """Returns the tool wrapper for a given role.
        
        Only valid for roles for which a corresponding packable widget was
        created by the factory method.
        """
        newly_loaded = role not in self.widgets
        _junk = self.get_widget_by_role(role)
        tool = self.tools.get(role, None)
        if tool is not None and newly_loaded:
            hidden = self.prefs.get(tool.role, {}).get("hidden", True)
            floating = self.prefs.get(tool.role, {}).get("floating", False)
            tool.set_floating(floating)
            tool.set_hidden(hidden)
            # XXX move the above to get_widget_by_role()?
        return tool

    def get_tools_in_sbindex_order(self):
        """Lists all loaded tools in order of ther sbindex setting.
        
        The returned list contains all tools known to the LayoutManager
        even if they aren't currently docked in a sidebar.
        """
        tools = []
        sbindexes = [s.get("sbindex", 0) for r, s in self.prefs.iteritems()]
        sbindexes.append(0)
        index_hwm = len(self.tools) + max(sbindexes)
        for role, tool in self.tools.iteritems():
            index = self.prefs.get(role, {}).get("sbindex", index_hwm)
            index_hwm += 1
            tools.append((index, tool.role, tool))
        tools.sort()
        return [t for i,r,t in tools]

    def toggle_user_tools(self, on=None):
        """Temporarily toggle the user's chosen tools on or off.
        """
        if on is None:
            on = False
            off = False
        else:
            off = not on
        if on or self.saved_user_tools:
            for tool in self.saved_user_tools:
                tool.set_hidden(False, temporary=True)
                tool.set_floating(tool.floating)
            self.saved_user_tools = []
        elif off or not self.saved_user_tools:
            for tool in self.get_tools_in_sbindex_order():
                if tool.hidden:
                    continue
                tool.set_hidden(True, temporary=True)
                self.saved_user_tools.append(tool)
        # Prevent the tool windows from taking keyboard focus from the
        # main window (in Metacity) by presenting it again.
        # https://gna.org/bugs/?17899
        gobject.idle_add(self.main_window.present)

    def show_all(self):
        """Displays all initially visible tools.
        """
        # Ensure that everything not hidden in prefs is loaded up front
        for role, win_pos in self.prefs.iteritems():
            hidden = win_pos.get("hidden", False)
            if not hidden:
                self.get_widget_by_role(role)

        # Insert those that are tools into sidebar, or float them
        # free as appropriate
        floating_tools = []
        for tool in self.get_tools_in_sbindex_order():
            floating = self.prefs.get(tool.role, {}).get("floating", False)
            if floating:
                floating_tools.append(tool)
            else:
                self.main_window.sidebar.add_tool(tool)
            tool.set_floating(floating)

        self.main_window.show_all()

        # Floating tools
        for tool in floating_tools:
            win = tool.floating_window
            tool.set_floating(True)
            win.show_all()

        # Present the main window for consistency with the toggle action.
        gobject.idle_add(self.main_window.present)


class ElasticContainer:
    """Mixin for containers which mirror certain size changes of descendents

    Descendents which wish to report internally-generated size changes should
    add the ElasticContent mixin, and should be packed into a container that
    derives from ElasticContainer. More than one ElasticContent widget can be
    packed under an ElasticContainer, and each can report different types of
    resizes - however the sub-hierarchies cannot overlap. You -can- nest
    ElasticContainers though: it's the outer one that receives the resize
    request.
    """

    def __init__(self):
        """Mixin constructor (construct as a gtk.Widget before calling)
        """
        self.__last_size = None
        self.__saved_size_request = None

    def mirror_elastic_content_resize(self, dx, dy):
        """Resize by a given amount.

        This is called by some ElasticContent widget below here in the
        hierarchy when it notices a request-size change on itself.
        """
        p = self.parent
        while p is not None:
            if isinstance(p, ElasticContainer):
                # propagate up and don't handle here
                p.mirror_elastic_content_resize(dx, dy)
                return
            p = p.parent
        self.__saved_size_request = self.get_size_request()
        alloc = self.get_allocation()
        w = alloc.width+dx
        h = alloc.height+dy
        if isinstance(self, gtk.Window):
            self.resize(w, h)
        self.set_size_request(w, h)
        self.queue_resize()


class ElasticContent:
    """Mixin for GTK widgets which want some parent to change size to match.
    """

    def __init__(self, mirror_vertical=True, mirror_horizontal=True):
        """Mixin constructor (construct as a gtk.Widget before calling)

        The options control which size changes are reported to the
        ElasticContainer ancestor and how:

            `mirror_vertical`:
                mirror vertical size changes

            `mirror_horizontal`:
                mirror horizontal size changes

        """
        self.__vertical = mirror_vertical
        self.__horizontal = mirror_horizontal
        if not mirror_horizontal and not mirror_vertical:
            return
        self.__last_req = None
        self.__expose_connid = self.connect_after("expose-event",
            self.__after_expose_event)
        self.__notify_parent = False
        self.connect_after("size-request", self.__after_size_request)

    def __after_expose_event(self, widget, event):
        # Begin notifying changes to the ancestor after the first expose event.
        # It doesn't matter for widgets that know their size before hand.
        # Assume widgets which initially don't know their size *do* know their
        # proper size after their first draw, even if they drew themselves
        # wrongly and now have to resize and do another size-request...
        connid = self.__expose_connid
        if not connid:
            return
        self.__expose_connid = None
        self.__notify_parent = True
        self.disconnect(connid)

    def __after_size_request(self, widget, req):
        # Catch the final value of each size-request, calculate the
        # difference needed and throw it back up the widget hierarchy
        # to interested parties.
        if not self.__last_req:
            dx, dy = 0, 0
        else:
            w0, h0 = self.__last_req
            dx, dy = req.width - w0, req.height - h0
        self.__last_req = (req.width, req.height)
        if not self.__notify_parent:
            return
        p = self.parent
        while p is not None:
            if isinstance(p, ElasticContainer):
                if not self.__vertical:
                    dy = 0
                if not self.__horizontal:
                    dx = 0
                if dy != 0 or dx != 0:
                    p.mirror_elastic_content_resize(dx, dy)
                break
            if isinstance(p, ElasticContent):
                break
            p = p.parent


class ElasticVBox (gtk.VBox, ElasticContainer):
    def __init__(self, *args, **kwargs):
        gtk.VBox.__init__(self, *args, **kwargs)
        ElasticContainer.__init__(self)


class ElasticExpander (gtk.Expander, ElasticContent):
    def __init__(self, *args, **kwargs):
        gtk.Expander.__init__(self, *args, **kwargs)
        ElasticContent.__init__(self, mirror_horizontal=False,
                                mirror_vertical=True)


class WindowWithSavedPosition:
    """Mixin for gtk.Windows which load/save their position via a LayoutManager.

    Classes using this interface must provide an attribute named layout_manager
    which exposes the LayoutManager whose prefs attr this window will read its
    initial position from and update. As a consequence of how this mixin
    interacts with the LayoutManager, it must have a meaningful window role at
    realize time.
    """

    def __init__(self):
        """Mixin constructor. Construction order does not matter.
        """
        self.__last_conf_pos = None
        self.__pos_save_timer = None
        self.__is_fullscreen = False
        self.__is_maximized = False
        self.connect("realize", self.__on_realize)
        self.connect("configure-event", self.__on_configure_event)
        self.connect("window-state-event", self.__on_window_state_event)

    @property
    def __pos(self):
        lm = self.layout_manager
        role = self.get_role()
        if not role in lm.prefs:
            lm.prefs[role] = dict()
        return lm.prefs[role]

    def __on_realize(self, widget):
        set_initial_window_position(self, self.__pos)

    def __on_window_state_event(self, widget, event):
        # Respond to changes of the fullscreen or maximized state only
        interesting = gdk.WINDOW_STATE_MAXIMIZED | gdk.WINDOW_STATE_FULLSCREEN
        if not event.changed_mask & interesting:
            return
        state = event.new_window_state
        self.__is_maximized = state & gdk.WINDOW_STATE_MAXIMIZED
        self.__is_fullscreen = state & gdk.WINDOW_STATE_FULLSCREEN

    def __on_configure_event(self, widget, event):
        """Store the current size of the window for future launches.
        """
        # Save the new position in the prefs...
        f_ex = self.window.get_frame_extents()
        x = max(0, f_ex.x)
        y = max(0, f_ex.y)
        conf_pos = dict(x=x, y=y, w=event.width, h=event.height)
        self.__last_conf_pos = conf_pos
        if self.get_role() == 'main-window':
            # ... however, wait for a bit so window-state-event has a chance to
            # fire first if the window can be meaningfully fullscreened. Compiz
            # in particular enjoys firing up to three configure-events (at
            # various sizes) before the window-state-event describing the
            # fullscreen.
            if not self.__pos_save_timer:
                self.__pos_save_timer \
                 = gobject.timeout_add_seconds(2, self.__save_position_cb)
        else:
            # Save the position now for non-main windows
            self.__save_position_cb()

    def __save_position_cb(self):
        if not (self.__is_maximized or self.__is_fullscreen):
            self.__pos.update(self.__last_conf_pos)
        self.__pos_save_timer = None
        return False



class MainWindow (WindowWithSavedPosition):
    """Mixin for main gtk.Windows in a layout.

    Contains slots and initialisation stuff for various widget slots, which
    can be provided by the layout manager's factory callable. These are:

        main-menubar
        main-toolbar
        main-widget
        main-statusbar

    """

    def __init__(self, layout_manager):
        """Mixin constructor: initialise as a gtk.Window vefore calling.

        This builds the sidebar and packs the main UI with pieces provided
        either by the factory or by overriding the various init_*() methods.
        This also sets up the window with an initial position and configures
        it to save its position and sidebar width when reconfigured to the
        LayoutManager's prefs.
        """
        assert isinstance(self, gtk.Window)
        WindowWithSavedPosition.__init__(self)
        self.layout_manager = layout_manager
        self.set_role("main-window")
        self.menubar = None; self.init_menubar()
        self.toolbar = None; self.init_toolbar()
        self.statusbar = None; self.init_statusbar()
        self.main_widget = None; self.init_main_widget()
        self.sidebar = Sidebar(layout_manager)
        self.hpaned = gtk.HPaned()
        self.layout_vbox = gtk.VBox()
        self.hpaned_position_loaded = False
        self.hpaned.pack1(self.main_widget, True, False)
        self.hpaned.pack2(self.sidebar, False, False)
        if self.menubar is not None:
            self.layout_vbox.pack_start(self.menubar, False, False)
        if self.toolbar is not None:
            self.layout_vbox.pack_start(self.toolbar, False, False)
        self.layout_vbox.pack_start(self.hpaned, True, True)
        if self.statusbar is not None:
            self.layout_vbox.pack_start(self.statusbar, False, False)
        self.add(self.layout_vbox)
        self.last_conf_size = None
        self.connect("configure-event", self.__on_configure_event)
        self.sidebar.connect("size-allocate", self.__on_sidebar_size_allocate)
        self.connect("map-event", self.__on_map_event)

    def __on_map_event(self, widget, event):
        if self.sidebar.is_empty():
            self.sidebar.hide()

    def init_menubar(self):
        self.menubar = self.layout_manager.get_widget_by_role("main-menubar")

    def init_toolbar(self):
        self.toolbar = self.layout_manager.get_widget_by_role("main-toolbar")

    def init_statusbar(self):
        self.statusbar = self.layout_manager.get_widget_by_role("main-statusbar")

    def init_main_widget(self):
        self.main_widget = self.layout_manager.get_widget_by_role("main-widget")

    def __on_configure_event(self, widget, event):
        self.last_conf_size = (event.width, event.height)

    def __on_sidebar_size_allocate(self, sidebar, allocation):
        lm = self.layout_manager
        role = self.get_role()
        if not role in lm.prefs:
            lm.prefs[role] = {}
        if self.hpaned_position_loaded:
            # Save the sidebar's width each time space is allocated to it.
            # Responds to the user adjusting the HPaned's divider position.
            sbwidth = allocation.width
            self.layout_manager.prefs["main-window"]["sbwidth"] = sbwidth
            self.layout_manager.main_window.sidebar.set_tool_widths(sbwidth)
        else:
            # Except, ugh, the first time. If the position isn't loaded yet,
            # load the main-window's sbwidth from the settings.
            if self.last_conf_size:
                width, height = self.last_conf_size
                sbwidth = lm.prefs[role].get("sbwidth", None)
                if sbwidth is not None:
                    handle_size = self.hpaned.style_get_property("handle-size")
                    pos = width - handle_size - sbwidth
                    self.hpaned.set_position(pos)
                    self.hpaned.queue_resize()
                self.hpaned_position_loaded = True


gtk.rc_parse_string ("""
    style "small-stock-image-button-style" {
        GtkWidget::focus-padding = 0
        GtkWidget::focus-line-width = 0
        xthickness = 0
        ythickness = 0
    }
    widget "*.small-stock-image-button"
    style "small-stock-image-button-style"
    """)


class SmallImageButton (gtk.Button):
    """A small button containing an image.

    Instances are used for the close button and snap in/snap out buttons in a
    ToolDragHandle.
    """

    ICON_SIZE = gtk.ICON_SIZE_MENU

    def __init__(self, stock_id=None, tooltip=None):
        gtk.Button.__init__(self)
        self.image = None
        if stock_id is not None:
            self.set_image_from_stock(stock_id)
        else:
            self.set_image_from_stock(gtk.STOCK_MISSING_IMAGE)
        self.set_name("small-stock-image-button")
        self.set_relief(gtk.RELIEF_NONE)
        self.set_property("can-focus", False)
        if tooltip is not None:
            self.set_tooltip_text(tooltip)
        self.connect("style-set", self.on_style_set)

    def set_image_from_stock(self, stock_id):
        if self.image:
            self.remove(self.image)
        self.image = gtk.image_new_from_stock(stock_id, self.ICON_SIZE)
        self.add(self.image)
        self.show_all()

    # TODO: support image.set_from_icon_name maybe

    def on_style_set(self, widget, prev_style):
        settings = self.get_settings()
        w, h = gtk.icon_size_lookup_for_settings(settings, gtk.ICON_SIZE_MENU)
        self.set_size_request(w+4, h+4)


class ToolResizeGrip (gtk.DrawingArea): 
    """A draggable bar for resizing a Tool vertically."""

    handle_size = gtk.HPaned().style_get_property("handle-size") + 2

    def __init__(self, tool):
        gtk.DrawingArea.__init__(self)
        self.tool = tool
        self.set_size_request(self.handle_size, self.handle_size)
        self.connect("configure-event", self.on_configure_event)
        self.connect("expose-event", self.on_expose_event)
        self.connect("button-press-event", self.on_button_press_event)
        self.connect("button-release-event", self.on_button_release_event)
        self.connect("motion-notify-event", self.on_motion_notify_event)
        self.connect("enter-notify-event", self.on_enter_notify_event)
        self.connect("leave-notify-event", self.on_leave_notify_event)
        self.width = self.height = self.handle_size
        mask = gdk.BUTTON_PRESS_MASK | gdk.BUTTON_RELEASE_MASK \
            | gdk.BUTTON1_MOTION_MASK | gdk.ENTER_NOTIFY_MASK \
            | gdk.LEAVE_NOTIFY_MASK
        self.set_events(mask)
        self.resize = None
    
    def on_configure_event(self, widget, event):
        self.width = event.width
        self.height = event.height

    def on_expose_event(self, widget, event):
        self.window.clear()
        self.window.begin_paint_rect(event.area)
        x = (self.width - self.handle_size) / 2
        y = (self.height - self.handle_size) / 2

        self.style.paint_handle(self.window, gtk.STATE_NORMAL, 
            gtk.SHADOW_NONE, event.area, self, 'paned',
            0, 0, self.width, self.height,
            gtk.ORIENTATION_HORIZONTAL)
        self.window.end_paint()
    
    def on_button_press_event(self, widget, event):
        if event.button != 1:
            return
        self.resize = (event.x, event.y)
        self.grab_add()
    
    def on_button_release_event(self, widget, event):
        self.in_resize_drag = False
        self.resize = None
        self.grab_remove()
    
    def on_motion_notify_event(self, widget, event):
        if not self.resize:
            return
        (x0, y0) = self.resize
        tool_alloc = self.tool.allocation
        w, h = tool_alloc.width, tool_alloc.height

        lm = self.tool.layout_manager
        max_w, max_h = lm.main_window.sidebar.max_tool_size()

        min_w, min_h = max_w, 0
        for child in self.tool.handle, self.tool.resize_grip, self.tool.widget:
            child_alloc = child.size_request()
            child_min_w, child_min_h = child_alloc
            min_w = max(min_w, child_min_w)
            min_h += child_min_h

        dx = 0 #event.x - x0
        dy = event.y - y0
        w += dx
        h += dy
        w = int(min(max(min_w, w), max_w))
        h = int(min(max(min_h, h), max_h))
        self.tool.set_size_request(w, h)
        self.tool.queue_resize()
    
    def on_leave_notify_event(self, widget, event):
        self.window.set_cursor(None)
        
    def on_enter_notify_event(self, widget, event):
        self.window.set_cursor(gdk.Cursor(gdk.SB_V_DOUBLE_ARROW))


class GripDecoration (gtk.Label):
    
    def __init__(self):
        gtk.Label.__init__(self)
        settings = gtk.settings_get_default()
        size = gtk.icon_size_lookup_for_settings(settings, gtk.ICON_SIZE_MENU)
        self.set_size_request(*size)
        self.connect("expose-event", self.on_expose_event)

    def on_expose_event(self, widget, event):
        widget.ensure_style()
        alloc = widget.get_allocation()
        state = widget.get_state()
        self.style.paint_handle(self.window, state,
            gtk.SHADOW_NONE, event.area, self,
            'handlebox', alloc.x, alloc.y, alloc.width, alloc.height,
            gtk.ORIENTATION_VERTICAL)


class FoldOutArrow (gtk.Button):
    
    TEXT_EXPANDED = _("Collapse")
    TEXT_COLLAPSED = _("Expand")

    def __init__(self, tool):
        gtk.Button.__init__(self)
        self.tool = tool
        self.set_name("small-stock-image-button")
        self.set_relief(gtk.RELIEF_NONE)
        self.set_property("can-focus", False)
        self.arrow = gtk.Arrow(gtk.ARROW_DOWN, gtk.SHADOW_NONE)
        self.set_tooltip_text(self.TEXT_EXPANDED)
        self.add(self.arrow)
        self.connect("clicked", self.on_click)
    
    def on_click(self, *a):
        self.tool.set_rolled_up(not self.tool.rolled_up)

    def set_arrow(self, rolled_up):
        if rolled_up:
            self.arrow.set(gtk.ARROW_RIGHT, gtk.SHADOW_NONE)
            self.set_tooltip_text(self.TEXT_COLLAPSED)
        else:
            self.arrow.set(gtk.ARROW_DOWN, gtk.SHADOW_NONE)
            self.set_tooltip_text(self.TEXT_EXPANDED)


class ToolDragHandle (gtk.EventBox):
    """A draggable handle for repositioning a Tool.
    """

    min_drag_distance = 10

    def __init__(self, tool, label_text):
        gtk.EventBox.__init__(self)
        self.tool = tool
        self.frame = frame = gtk.Frame()
        frame.set_shadow_type(gtk.SHADOW_NONE)
        self.hbox = hbox = gtk.HBox()
        self.gripdeco = GripDecoration()
        hbox.pack_start(self.gripdeco, False, False)
        self.roll_up_button = FoldOutArrow(self.tool)
        hbox.pack_start(self.roll_up_button, False, False)
        self.label = label = gtk.Label(label_text)
        #self.label.set_alignment(0.0, 0.5)
        self.label.set_ellipsize(pango.ELLIPSIZE_END)
        hbox.pack_start(label, True, True)
        self.snap_button = SmallImageButton(tooltip="Snap out")  # XX update this when pressed
        self.snap_button.connect("clicked", tool.on_snap_button_pressed)
        self.close_button = SmallImageButton(gtk.STOCK_CLOSE, "Close")
        self.close_button.connect("clicked", tool.on_close_button_pressed)
        hbox.pack_start(self.snap_button, False, False)
        hbox.pack_start(self.close_button, False, False)
        frame.add(hbox)
        self.add(frame)
        self.connect("button-press-event", self.on_button_press_event)
        self.connect("button-release-event", self.on_button_release_event)
        self.connect("motion-notify-event", self.on_motion_notify_event)
        self.connect("enter-notify-event", self.on_enter_notify_event)
        self.connect("leave-notify-event", self.on_leave_notify_event)
        # Drag initiation
        self.set_events(gdk.BUTTON_PRESS_MASK | gdk.BUTTON_RELEASE_MASK)
        self.button_press_xy = None
        self.in_reposition_drag = False
        # Floating status
        self.set_floating(False)
        self.frame.connect("expose-event", self.on_frame_expose_event)

    def on_frame_expose_event(self, widget, event):
        state = self.get_state()
        alloc = self.get_allocation()
        w = alloc.width
        h = alloc.height
        self.style.paint_box(widget.window, state,
            gtk.SHADOW_OUT, event.area, widget,
            'button', 0, 0, w, h)

    def set_floating(self, floating):
        if floating:
            stock_id = gtk.STOCK_GOTO_LAST
        else:
            stock_id = gtk.STOCK_GOTO_FIRST
        self.snap_button.set_image_from_stock(stock_id)

    def set_rolled_up(self, rolled_up):
        self.roll_up_button.set_arrow(rolled_up)

    def on_button_press_event(self, widget, event):
        if event.button != 1:
            return
        if event.type == gdk._2BUTTON_PRESS:
            if not self.tool.floating:
                self.tool.set_rolled_up(not self.tool.rolled_up)
                self._reset()
        elif event.type == gdk.BUTTON_PRESS:
            self.set_state(gtk.STATE_ACTIVE)
            self.window.set_cursor(gdk.Cursor(gdk.FLEUR))
            self.button_press_xy = event.x, event.y

    def on_button_release_event(self, widget, event):
        self._reset()

    def _reset(self):
        self.button_press_xy = None
        self.set_state(gtk.STATE_NORMAL)
        if self.window:
            self.window.set_cursor(None)

    def on_motion_notify_event(self, widget, event):
        if not self.button_press_xy:
            return False
        if not event.state & gdk.BUTTON1_MASK:
            return
        lm = self.tool.layout_manager
        ix, iy = self.button_press_xy
        dx, dy = event.x - ix, event.y - iy
        dd = sqrt(dx**2 + dy**2)
        if dd > self.min_drag_distance:
            self.start_reposition_drag()

    def start_reposition_drag(self):  # XXX: move to Tool?
        """Begin repositioning this tool."""
        lm = self.tool.layout_manager
        ix, iy = self.button_press_xy
        self.button_press_xy = None
        self.window.set_cursor(gdk.Cursor(gdk.FLEUR))
        self.set_state(gtk.STATE_ACTIVE)
        self.tool.layout_manager.drag_state.begin(self.tool, ix, iy)

    def on_reposition_drag_finished(self):   # XXX: move to Tool?
        """Called when repositioning has finished."""
        self._reset()

    def on_leave_notify_event(self, widget, event):
        self.window.set_cursor(None)
        self.set_state(gtk.STATE_NORMAL)
        
    def on_enter_notify_event(self, widget, event):
        self.window.set_cursor(gdk.Cursor(gdk.HAND2))
        #if not self.in_reposition_drag:
        #    self.set_state(gtk.STATE_PRELIGHT)


class ToolWindow (gtk.Window, ElasticContainer, WindowWithSavedPosition):
    """Window containing a Tool in the floating state.
    """

    def __init__(self, title, role, layout_manager):
        gtk.Window.__init__(self)
        ElasticContainer.__init__(self)
        WindowWithSavedPosition.__init__(self)
        self.layout_manager = layout_manager
        self.set_type_hint(gdk.WINDOW_TYPE_HINT_UTILITY)
        self.role = role
        self.set_role(role)
        self.set_title(title)
        self.set_transient_for(layout_manager.main_window)
        self.tool = None
        self.connect("configure-event", self.on_configure_event)
        self.pre_hide_pos = None

    def add(self, tool):
        self.tool = tool
        gtk.Window.add(self, tool)

    def remove(self, tool):
        self.tool = None
        gtk.Window.remove(self, tool)

    def on_configure_event(self, widget, event):
        if self.pre_hide_pos:
            return
        role = self.role
        lm = self.layout_manager
        if not lm.prefs.get(role, False):
            lm.prefs[role] = {}
        lm.prefs[role]['floating'] = True

    def show(self):
        gtk.Window.show(self)
        if self.pre_hide_pos:
            self.move(*self.pre_hide_pos)
        self.pre_hide_pos = None

    def show_all(self):
        gtk.Window.show_all(self)
        if self.pre_hide_pos:
            self.move(*self.pre_hide_pos)
        self.pre_hide_pos = None

    def hide(self):
        self.pre_hide_pos = self.get_position()
        gtk.Window.hide(self)


class Tool (gtk.VBox, ElasticContainer):
    """Container for a dockable tool widget.
    
    The widget may be packed into a Sidebar, or snapped out into its own
    floating ToolWindow.
    """

    def __init__(self, widget, role, title, gloss, layout_manager):
        gtk.VBox.__init__(self)
        ElasticContainer.__init__(self)
        self.role = role
        self.handle = ToolDragHandle(self, gloss)
        self.pack_start(self.handle, False, False)
        self.layout_manager = layout_manager
        self.widget_frame = frame = gtk.Frame()
        frame.set_shadow_type(gtk.SHADOW_IN)
        frame.add(widget)
        self.widget = widget
        self.pack_start(frame, True, True)
        self.resize_grip = ToolResizeGrip(self)
        self.pack_start(self.resize_grip, False, False)
        self.floating_window = ToolWindow(title, role, layout_manager)
        self.floating_window.connect("delete-event", self.on_floating_window_delete_event)
        self.layout_manager.window_group.add_window(self.floating_window)
        self.floating = False
        self.hidden = False
        self.rolled_up = False
        self.rolled_up_prev_size = None
        self.connect("size-allocate", self.on_size_allocate)
    
    def on_size_allocate(self, widget, allocation):
        if self.rolled_up:
            return
        lm = self.layout_manager
        if self not in lm.main_window.sidebar.tools_vbox:
            return
        if not lm.prefs.get(self.role, False):
            lm.prefs[self.role] = {}
        lm.prefs[self.role]["sbheight"] = allocation.height

    def on_floating_window_delete_event(self, window, event):
        self.set_hidden(True)
        return True   # Suppress ordinary deletion. We'll be wanting it again.

    def set_show_resize_grip(self, show):
        if not show:
            if self.resize_grip in self:
                self.resize_grip.hide()
                self.remove(self.resize_grip)
        else:
            if self.resize_grip not in self:
                self.pack_start(self.resize_grip, False, False)
                self.resize_grip.show()

    def set_show_widget_frame(self, show):
        if not show:
            if self.widget_frame in self:
                self.widget_frame.hide()
                self.remove(self.widget_frame)
        else:
            if self.widget_frame not in self:
                self.pack_start(self.widget_frame, True, True)
                self.widget_frame.show()

    def restore_sbheight(self):
        """Restore the height of the tool when docked."""
        lm = self.layout_manager
        role = self.role
        sbheight = lm.prefs.get(role, {}).get("sbheight", None)
        if sbheight is not None:
            self.set_size_request(-1, sbheight)

    def set_floating(self, floating):
        """Flips the widget beween floating and non-floating, reparenting it.
        """
        self.handle.set_floating(floating)
        self.set_rolled_up(False)
        
        # Clear any explicit size requests so that the frame is able to adopt a
        # natural size again.
        for wid in (self.handle, self):
            if wid.get_visible():
                wid.set_size_request(-1, -1)
                wid.queue_resize()

        # When changing states, the size is changed and the tool is
        # repacked. Forget the size mirroring history because it doesn't
        # make sense any more.

        if self.parent:
            self.parent.remove(self)
        lm = self.layout_manager
        if lm.prefs.get(self.role, None) is None:
            lm.prefs[self.role] = {}
        if not floating:
            if lm.main_window.sidebar.is_empty():
                lm.main_window.sidebar.show_all()
            sbindex = lm.prefs[self.role].get("sbindex", None)
            lm.main_window.sidebar.add_tool(self, index=sbindex)
            self.floating_window.hide()
            self.floating = lm.prefs[self.role]["floating"] = False
            self.restore_sbheight()
            lm.main_window.sidebar.reassign_indices()
            self.set_show_resize_grip(True)
        else:
            self.set_show_resize_grip(False)
            self.floating_window.add(self)
            # Defer the show_all(), seems to be needed when toggling on a
            # hidden, floating window which hasn't yet been loaded.
            gobject.idle_add(self.floating_window.show_all)
            self.floating = lm.prefs[self.role]["floating"] = True
            lm.main_window.sidebar.reassign_indices()
            if lm.main_window.sidebar.is_empty():
                lm.main_window.sidebar.hide()

    def set_hidden(self, hidden, temporary=False):
        """Sets a tool as hidden, hiding or showing it as appropriate.
        
        Note that this does not affect whether the window is floating or not
        when un-hidden it will restore to the same place in the UI it was
        hidden.

        If the `temporary` argument is true, the new state will not be saved in
        the preferences.
        """
        self.set_rolled_up(False)
        role = self.role
        lm = self.layout_manager
        if not lm.prefs.get(role, False):
            lm.prefs[role] = {}
        if hidden:
            self.hide()
            if self.parent:
                self.parent.remove(self)
            if self.floating:
                self.floating_window.hide()
        else:
            self.set_floating(self.floating)
            # Which will restore it to the correct state
        self.hidden = hidden
        if not temporary:
            lm.prefs[role]["hidden"] = hidden
        if lm.main_window.sidebar.is_empty():
            lm.main_window.sidebar.hide()
        else:
            lm.main_window.sidebar.show_all()

    def set_rolled_up(self, rolled_up):
        resize_needed = False
        if rolled_up:
            if not self.rolled_up:
                self.rolled_up = True
                alloc = self.get_allocation()
                self.rolled_up_prev_size = (alloc.width, alloc.height)
                self.set_show_widget_frame(False)
                self.set_show_resize_grip(False)
                newalloc = self.handle.get_allocation()
                self.set_size_request(newalloc.width, newalloc.height)
                resize_needed = True
        else:
            if self.rolled_up:
                self.rolled_up = False
                self.set_show_widget_frame(True)
                self.set_show_resize_grip(True)
                self.set_size_request(*self.rolled_up_prev_size)
                resize_needed = True
                self.rolled_up_prev_size = None
        # Since other handlers call this to unroll a window before moving it
        # or hiding it, it's best to perform any necessary resizes now before
        # returning. Otherwise sizes can become screwy when the operation 
        # is reversed.
        if resize_needed:
            self.queue_resize()
            while gtk.events_pending():
                gtk.main_iteration(False)
        # Notify the indicator arrow
        self.handle.set_rolled_up(rolled_up)


    def on_close_button_pressed(self, window):
        self.set_hidden(True)
    
    def on_snap_button_pressed(self, window):
        def handle_snap():
            sidebar = self.layout_manager.main_window.sidebar
            if not self.floating:
                self.set_floating(True)
                self.handle.snap_button.set_state(gtk.STATE_NORMAL)
                if sidebar.is_empty():
                    sidebar.hide()
            else:
                if sidebar.is_empty():
                   sidebar.show()
                self.set_floating(False)
                self.handle.snap_button.set_state(gtk.STATE_NORMAL)
            return False
        gobject.idle_add(handle_snap)

    def get_preview_size(self):
        alloc = self.get_allocation()
        return alloc.width, alloc.height


class ToolDragPreviewWindow (gtk.Window):
    """A shaped outline window showing where the current drag will move a Tool.
    """

    def __init__(self, layout_manager):
        gtk.Window.__init__(self, gtk.WINDOW_POPUP)
        self.owner = layout_manager.main_window
        #self.set_opacity(0.5)
        self.set_decorated(False)
        self.set_position(gtk.WIN_POS_MOUSE)
        self.connect("map-event", self.on_map_event)
        self.connect("expose-event", self.on_expose_event)
        self.connect("configure-event", self.on_configure_event)
        self.bg = None
        self.set_default_size(1,1)

    def show_all(self):
        gtk.Window.show_all(self)
        self.window.move_resize(0, 0, 1, 1)
    
    def on_map_event(self, window, event):
        owner_win = self.owner.get_toplevel()
        self.set_transient_for(owner_win)
        # The first time we're mapped, set a background pixmap.
        # Background pixmap, a checkerboard
        if self.bg is None:
            self.bg = gdk.Pixmap(drawable=self.window, width=2, height=2)
            cmap = gdk.colormap_get_system()
            black = cmap.alloc_color(gdk.Color(0.0, 0.0, 0.0))
            white = cmap.alloc_color(gdk.Color(1.0, 1.0, 1.0))
            self.bg.set_colormap(cmap)
            gc = self.bg.new_gc(black, white)
            self.bg.draw_rectangle(gc, True, 0, 0, 2, 2)
            gc = self.bg.new_gc(white, black)
            self.bg.draw_points(gc, [(0,0), (1,1)])
            self.window.set_back_pixmap(self.bg, False)
            self.window.clear()
    
    def on_configure_event(self, window, event):
        # Shape the window
        w = event.width
        h = event.height
        r = gdk.Region()
        s = 4
        r.union_with_rect(gdk.Rectangle(0, 0, w, s))
        r.union_with_rect(gdk.Rectangle(0, 0, s, h))
        r.union_with_rect(gdk.Rectangle(0, h-s, w, s))
        r.union_with_rect(gdk.Rectangle(w-s, 0, s, h))
        self.window.shape_combine_region(r, 0, 0)

    def on_expose_event(self, window, event):
        # Clear to the backing pixmap established earlier
        if self.bg is not None:
            self.window.begin_paint_rect(event.area)
            self.window.set_back_pixmap(self.bg, False)
            self.window.clear()
            self.window.end_paint()


class ToolDragState:
    
    """Manages visual state during tool repositioning.

    The resize grip can largely take care of itself, and deploy its own grabs.
    However when tools are repositioned, they get reparented between floating
    windows and the sidebar and a grab on either tends to become confused. The
    DragState object establishes whatever grabs are necessary for tools to be
    repositioned; all they have to do is detect when the process starts and
    call enter().
    """

    def __init__(self, layout_manager):
        self.layout_manager = layout_manager
        self.insert_pos = None
        self.preview = ToolDragPreviewWindow(layout_manager)
        self.preview_size = None
        self.tool = None        # the box being dragged around & previewed
        self.handle_pos = None
        self.pointer_inside_sidebar = None
        self.conn_ids = []

    def connect_reposition_handlers(self):
        """Connect the event handlers used while repositioning tools.
        """
        win = self.layout_manager.main_window
        conn_id = win.connect("motion-notify-event",
                              self.on_reposition_motion_notify)
        self.conn_ids.append(conn_id)
        conn_id = win.connect("button-release-event",
                              self.on_reposition_button_release)
        self.conn_ids.append(conn_id)

    def disconnect_reposition_handlers(self):
        """Disconnect the handlers set up by connect_reposition_handlers().
        """
        win = self.layout_manager.main_window
        while self.conn_ids:
            conn_id = self.conn_ids.pop()
            win.disconnect(conn_id)

    def begin(self, tool, handle_x, handle_y):
        """Begin repositioning a tool using its drag handle.
        
        It's assumed the pointer button is held down at this point. The rest of
        the procedure happens behind a pointer grab on the main window, and
        ends when the button is released.
        """
        lm = self.layout_manager

        # Show the preview window
        width, height = tool.get_preview_size()
        self.preview.show_all()
        self.preview_size = (width, height)
        self.handle_pos = (handle_x, handle_y)

        # Establish a pointer grab on the main window (which stays mapped
        # throughout the drag).
        main_win = lm.main_window
        main_win_gdk = main_win.window
        events = gdk.BUTTON_PRESS_MASK | gdk.BUTTON_RELEASE_MASK \
            | gdk.BUTTON1_MOTION_MASK
        grab_status = gdk.pointer_grab(main_win_gdk, False, events,
            None, gdk.Cursor(gdk.FLEUR), 0L)
        if grab_status != gtk.gdk.GRAB_SUCCESS:
            warn("Grab failed, aborting", RuntimeWarning, 2)
            return
        try:
            self.tool = tool
            self.connect_reposition_handlers()
        except:
            gdk.pointer_ungrab()
            self.disconnect_reposition_handlers()
            raise

    def on_reposition_button_release(self, widget, event):
        """Ends the current tool reposition drag.
        """
        gdk.pointer_ungrab()
        self.disconnect_reposition_handlers()
        self.end()

    def on_reposition_motion_notify(self, widget, event):
        """Bulk of window shuffling within a tool reposition drag.
        """
        x_root = event.x_root
        y_root = event.y_root
        w, h = self.preview_size
        # Show the sidebar if it's not currently visible
        sidebar = self.layout_manager.main_window.sidebar
        if sidebar.is_empty():
            sidebar.show_all()
        # Moving into or out of the box defined by the sidebar
        # changes the mode of the current drag.
        sb_left, sb_top = sidebar.window.get_origin()
        sb_alloc = sidebar.allocation
        sb_right = sb_left + sb_alloc.width
        sb_bottom = sb_top + sb_alloc.height
        if (x_root < sb_left or x_root > sb_right
            or y_root < sb_top or y_root > sb_bottom):
            self.pointer_inside_sidebar = False
        else:
            self.pointer_inside_sidebar = True
        # Move the preview window
        if self.pointer_inside_sidebar:
            tool = self.tool
            ins = sidebar.insertion_point_at_pointer()
            if ins is not None and ins != self.insert_pos:
                self.insert_pos = ins
                if not tool in sidebar.tools_vbox:
                    tool.set_floating(False)
                    # Update preview size: the widget will be a different
                    # size after being shoved in the sidebar.
                    while gtk.events_pending():
                        gtk.main_iteration(False)
                    w, h = self.preview_size = tool.get_preview_size()
                sidebar.reorder_item(tool, ins)
                sidebar.reassign_indices()
            x, y = tool.handle.window.get_origin()
        else:
            ix, iy = self.handle_pos
            x = int(x_root - ix)
            y = int(y_root - iy)
        self.handle_pos_root = (x, y)
        self.preview.window.move_resize(x, y, w, h)

    def end(self):
        """Invoked at the end of repositioning. Snapping out happens here.
        """
        sidebar = self.layout_manager.main_window.sidebar
        if self.pointer_inside_sidebar:
            pass  # Widget has already been reordered,
                  # or snapped in and then reordered.
        else:
            # Snap out
            if self.tool in sidebar.tools_vbox:
                self.tool.set_floating(True)
            # Set window position to that of the floating window
            x, y = self.handle_pos_root
            w, h = self.preview_size
            def set_pos(tool):
                tool.floating_window.resize(w, h)
                tool.floating_window.move(x, y)
                return False  #oneshot
            gobject.idle_add(set_pos, self.tool)
        self.preview.hide()
        self.tool.handle.on_reposition_drag_finished()
        self.tool = None
        self.handle_pos = None
        self.insert_pos = None
        self.pointer_inside_sidebar = None
        self.preview_pos_root = (0, 0)
        if sidebar.is_empty():
           sidebar.hide()


class Sidebar (gtk.EventBox):
    """Vertical sidebar containing reorderable tools which can be snapped out.
    """

    MIN_SIZE = 150

    def __init__(self, layout_manager):
        gtk.EventBox.__init__(self)
        self.layout_manager = layout_manager
        self.scrolledwin = gtk.ScrolledWindow()
        self.scrolledwin.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_ALWAYS)
        self.add(self.scrolledwin)
        self.tools_vbox = gtk.VBox()
        self.scrolledwin.add_with_viewport(self.tools_vbox)
        self.slack = gtk.EventBox()   # needs its own window
        self.tools_vbox.pack_start(self.slack, True, True)
        self.set_size_request(self.MIN_SIZE, self.MIN_SIZE)

    def add_tool(self, widget, index=None):
        assert isinstance(widget, Tool)
        if self.is_empty():
            self.show()
        self.tools_vbox.pack_start(widget, expand=False, fill=False)
        if index is not None:
            self.tools_vbox.reorder_child(widget, index)
        self.reassign_indices()
        self.tools_vbox.reorder_child(self.slack, -1)

    def remove_tool(self, widget):
        assert isinstance(widget, Tool)
        self.reassign_indices()
        self.tools_vbox.remove(widget)
        if self.is_empty():
            self.hide()
        else:
            self.reassign_indices()

    def reorder_item(self, item, pos):
        assert isinstance(item, Tool)
        self.tools_vbox.reorder_child(item, pos)
        self.tools_vbox.reorder_child(self.slack, -1)
        self.reassign_indices()

    def is_empty(self):
        """True if there are no tools in the sidebar."""
        num_children = len(self.tools_vbox.get_children())
        return num_children == 1

    def insertion_point_at_pointer(self):
        """Returns where in the sidebar a tool would be inserted.

        Returns an integer position for passing to reorder_item(), or None.
        Currently only tool drag handles are valid insertion points.
        """
        window_info = gdk.window_at_pointer()
        if window_info is None:
            return None
        pointer_window = window_info[0]
        current_tool = self.layout_manager.drag_state.tool
        i = 0
        for widget in self.tools_vbox:
            if widget is self.slack:
                if widget.window is pointer_window:
                    return i
            if isinstance(widget, Tool):
                if widget is not current_tool:
                    dragger = widget.handle
                    if dragger.window is pointer_window:
                        return i
            i += 1
        return None

    def max_tool_size(self):
        """Returns the largest a packed tool is allowed to be.
        """
        scrwin = self.scrolledwin
        viewpt = scrwin.get_child()
        sb_pad = 2 * scrwin.style_get_property("scrollbar-spacing")
        vp_alloc = viewpt.allocation
        max_size = (vp_alloc.width-sb_pad, vp_alloc.height-sb_pad)
        return max_size

    def reassign_indices(self):
        """Calculates and reassigns the "sbindex" prefs value.
        """
        lm = self.layout_manager
        i = 0
        for tool in [t for t in self.tools_vbox if t is not self.slack]:
            role = tool.role
            if not lm.prefs.get(role, False):
                lm.prefs[role] = {}
            lm.prefs[role]["sbindex"] = i
            i += 1

    def set_tool_widths(self, width):
        """Constrain all packed tools' widths to a certain size.
        """
        lm = self.layout_manager
        max_w, max_h = self.max_tool_size()
        for tool in [t for t in self.tools_vbox if t is not self.slack]:
            natural_w, natural_h = tool.size_request()
            req_w, req_h = tool.get_size_request()
            if req_w == -1:
                # Only constrain if the natual width is larger than permitted
                if natural_w > max_w:
                    tool.set_size_request(max_w, req_h)
            else:
                if req_w > max_w:
                    if natural_w <= max_w:
                        tool.set_size_request(-1, req_h)
                        # Dubious. Could this lead to an infinite loop?
                    else:
                        tool.set_size_request(max_w, req_h)


def set_initial_window_position(win, pos):
    """Set the position of a gtk.Window, used during initial positioning.

    The pos argument is a dict containing the following optional keys

        "w": <int>
        "h": <int>
            If positive, the size of the window.
            If negative, size is calculated based on the size of the
            monitor with the pointer on it, and x (or y) if given, e.g.

                width = mouse_mon_w -  abs(x) + abs(w)   # or (if no x)
                width = mouse_mon_w - (2 * abs(w))

            The same is true of calulated heights.

        "x": <int>
        "y": <int>
            If positive, the left/top of the window.
            If negative, the bottom/right of the window: you MUST provide
            a positive w,h if you do this!

    If the window's calculated top-left would place it offscreen, it will be
    placed in its default, window manager provided position. If its calculated
    size is larger than the screen, the window will be given its natural size
    instead.

    TODO: should gdk.HINT_USER_SIZE and gdk.HINT_USER_POS be set on the
    window as appropriate?
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
    display = gdk.display_get_default()
    screen, ptr_x, ptr_y, _modmask = display.get_pointer()
    if screen is None:
        raise RuntimeError, "No cursor on the default screen. Eek."
    screen_w = screen.get_width()
    screen_h = screen.get_height()
    assert screen_w > MIN_USABLE_SIZE
    assert screen_h > MIN_USABLE_SIZE

    if x is not None and y is not None:
        if x >= 0:
            final_x = x
        else:
            assert w is not None
            assert w > 0
            final_x = screen_w - w - abs(x)
        if y >= 0:
            final_y = y
        else:
            assert h is not None
            assert h > 0
            final_y = screen_h - h - abs(y)
        if final_x < 0 or final_x > screen_w - MIN_USABLE_SIZE: final_x = None
        if final_y < 0 or final_y > screen_h - MIN_USABLE_SIZE: final_h = None

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
        if final_w > screen_w or final_w < MIN_USABLE_SIZE: final_w = None
        if final_h > screen_h or final_h < MIN_USABLE_SIZE: final_h = None

    if None not in (final_w, final_h, final_x, final_y):
        geom_str = "%dx%d+%d+%d" % (final_w, final_h, final_x, final_y)
        win.parse_geometry(geom_str)
    else:
        if final_w and final_h:
            win.set_default_size(final_w, final_h)
        if final_x and final_y:
            win.move(final_x, final_y)

