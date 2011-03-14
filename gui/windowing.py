# Base classes for window types

import sys
import os.path

import gtk
import gtk.gdk as gdk
import gobject

from layout import WindowWithSavedPosition

def hide_window_cb(window, event):
    # used by some of the windows
    window.hide()
    return True

def on_tool_widget_map(widget, app, role, connids):
    # Ensure that tool widgets' floating windows are added to the kbm when
    # mapped for the first time.
    while connids:
        connid = connids.pop()
        widget.disconnect(connid)
    tool = app.layout_manager.get_tool_by_role(role)
    window = tool.floating_window
    app.kbm.add_window(window)

def window_factory(role, layout_manager, app):
    """Window factory method, for use by app.layout_manager.

    Modules are located by lowercasing `role` and importing that. Modules may
    provide either `ToolWidget` or `Window` classes, but not both. The first
    listed will be initialised and used for the returned widget. Both
    ToolWidgets and Windows are instantiated as Constructor(app).

    ToolWidget classes must provide a tool_widget_title variable accessible via
    the instance, which contains the title used in the titlebar. """

    if role in ["main-toolbar",'main-widget','main-statusbar','main-menubar']:
        # These slots are either unused, or are populated internally right now.
        return None
    # Layout will build of these at startup; we load it through the same
    # mechanismas below, but with a specific rather than a generic name.
    if role == 'main-window':
        role = "drawWindow"
    # Load module, and initialize tool widget or subwindow from it
    try:
        module = __import__(role.lower(), globals(), locals(), [])
    except ImportError:
        return None
    if hasattr(module, "ToolWidget"):
        widget = module.ToolWidget(app)
        title = widget.tool_widget_title
        connids = []
        connid = widget.connect("map", on_tool_widget_map, app, role, connids)
        connids.append(connid)
        return (widget, title)
    else:
        window = module.Window(app)
        return (window, )


class Dialog (gtk.Dialog):
    """
    A dialog. Dialogs are a bit of a rough edge at the moment; currently only
    the preferences window is one. Dialogs accept keyboard input, and hide and
    show with Tab (though it's arguable that they shouldn't).
    """
    def __init__(self, app, *args, **kwargs):
        gtk.Dialog.__init__(self, *args, **kwargs)
        self.app = app
        self.connect('delete-event', hide_window_cb)


class MainWindow (gtk.Window):
    """The main window in the GUI.

    Not much code to see here yet, go look at drawwindow.DrawWindow.
    """
    def __init__(self, app):
        gtk.Window.__init__(self, type=gtk.WINDOW_TOPLEVEL)
        self.app = app
        self.app.kbm.add_window(self)


class AppWindowWithSavedPosition (WindowWithSavedPosition):
    """Mixin for windows with LayoutManager-stored positions and an app attr.
    """

    @property
    def layout_manager(self):
        return self.app.layout_manager


class SubWindow (gtk.Window, AppWindowWithSavedPosition):
    """
    A subwindow in the GUI. All are utility windows that are transients for
    the main window. Subwindows remember their position when hidden or shown,
    in a way that keeps as many window managers as possible happy.

    Most SubWindows don't accept keyboard input, but if your subclass requires
    it, pass key_input to the constructor.
    """
    
    def __init__(self, app, key_input=False):
        gtk.Window.__init__(self, type=gtk.WINDOW_TOPLEVEL)
        AppWindowWithSavedPosition.__init__(self)
        self.app = app
        if not key_input:
            self.app.kbm.add_window(self)
            # TODO: do we need a separate class for keyboard-input-friendly
            # windows? Do they share anything in common with dialogs (could
            # they all be implmented as dialogs?)
        self.pre_hide_pos = None
        self.connect("realize", self.on_realize)
        self.connect('delete-event', hide_window_cb)

    def on_realize(self, widget):
        # Mark subwindows as utility windows: many X11 WMs handle this sanely
        # OSX with x11.app does not handle this well; https://gna.org/bugs/?15838
        if not sys.platform == 'darwin':
            widget.window.set_type_hint(gdk.WINDOW_TYPE_HINT_UTILITY)
        # Win32 is responsive to the following: keeps the utility
        # window above the main window in fullscreen.
        widget.set_transient_for(self.app.drawWindow)
    
    def show_all(self):
        pos = self.pre_hide_pos
        gtk.Window.show_all(self)
        if pos:
            self.move(*pos)
        # To keep Compiz happy, move() must be called in the very same
        # handler as show_all(), immediately after. Wiring it in to
        # a map-event or show event handler won't do. Workaround for
        # https://bugs.launchpad.net/ubuntu/+source/compiz/+bug/155101
    
    def hide(self):
        self.pre_hide_pos = self.get_position()
        gtk.Window.hide(self)


class PopupWindow (gtk.Window):
    """
    A popup window, with no decoration. Popups always appear centred under the
    mouse, and don't accept keyboard input.
    """
    def __init__(self, app):
        gtk.Window.__init__(self, type=gtk.WINDOW_POPUP)
        self.set_gravity(gdk.GRAVITY_CENTER)
        self.set_position(gtk.WIN_POS_MOUSE)
        self.app = app
        self.app.kbm.add_window(self)


