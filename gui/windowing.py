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

    if role in ['main-toolbar','main-widget','main-statusbar','main-menubar']:
        # These slots are either unused, or are populated internally right now.
        return None
    # Layout will build of these at startup; we load it through the same
    # mechanismas below, but with a specific rather than a generic name.
    if role == 'main-window':
        role = "drawWindow"
    else:
        if role not in app.window_names:
            raise ValueError, 'Window %r is missing in DEFAULT_CONFIG (application.py)' % role
    # Load module, and initialize tool widget or subwindow from it
    module = __import__(role.lower(), globals(), locals(), [])
    if hasattr(module, "ToolWidget"):
        widget = module.ToolWidget(app)
        connids = []
        connid = widget.connect("map", on_tool_widget_map, app, role, connids)
        connids.append(connid)
        return (widget, widget.stock_id, widget.tool_widget_title)
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


class ChooserDialog (gtk.Dialog):
    """Partly modal dialog for making a single, fast choice.

    Chooser dialogs are modal, and permit input, but dispatch a subset of
    events via ``app.kbm``. As such, they're not suited for keyboard data
    entry, but are fine for clicking on brushes, colours etc. and may have
    cancel buttons.

    Chooser dialogs save their size to the app preferences, and appear under
    the mouse. They operate within a grab as well as being modal and issue a
    gtk.RESPONSE_REJECT response when the pointer leaves the window. There is
    some slack in the leave behaviour to allow the window to be resized under
    fancy modern window managers.

    They can (and should) be kept around as references, and can be freely
    hidden and shown after construction.

    """

    #: Minimum dialog width
    MIN_WIDTH = 256

    #: Minimum dialog height
    MIN_HEIGHT = 256

    #: How far outside the window the pointer has to go before the response is
    #: treated as a reject (see class docs).
    LEAVE_SLACK = 48


    def __init__(self, app, title, actions, config_name,
                 buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_REJECT)):
        """Initialize.

        :param app: the main Application object.
        :param title: title for the dialog
        :param actions: iterable of action names to respect; others are
           rejected. See `gui.keyboard.KeyboardManager.add_window()`.
        :param config_name: config string base-name for saving window size;
           use a simple "lowercase_with_underscores" name.
        :param buttons: Button list for `gtk.Dialog` construction.

        """

        # Public member vars
        self.app = app

        # Internal state
        self._size = None
        self._entered = False
        self._motion_handler_id = None
        self._prefs_size_key = "%s.window_size" % (config_name,)

        # Superclass construction; default size
        default_size = (self.MIN_WIDTH, self.MIN_HEIGHT)
        w, h = app.preferences.get(self._prefs_size_key, default_size)
        flags = gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT
        parent = app.drawWindow
        gtk.Dialog.__init__(self, title, parent, flags, buttons)
        self.set_default_size(w, h)
        self.set_position(gtk.WIN_POS_MOUSE)

        # Cosmetic/behavioural hints
        if not sys.platform == 'darwin':
            self.set_type_hint(gdk.WINDOW_TYPE_HINT_UTILITY)

        # Register with the keyboard manager, but only let certain actions be
        # driven from the keyboard.
        app.kbm.add_window(self, actions)

        # Event handlers
        self.connect("configure-event", self._configure_cb)
        self.connect("enter-notify-event", self._enter_cb)
        self.connect("show", self._show_cb)
        self.connect("hide", self._hide_cb)

        # Keep around if the user closes the window.
        self.connect("delete-event", self._hide_on_delete)


    def _hide_on_delete(self, widget, event):
        # Can't use gtk_widget_hide_on_delete via bound method in GTK3 due to
        # args mismatch. Oh well, just reinvent the wheel: hide the dialog and
        # eat the event.
        self.hide()
        return True


    def _configure_cb(self, widget, event):
        # Update _size and prefs when window is adjusted
        x, y = self.get_position()
        self._size = (x, y, event.width, event.height)
        w = max(self.MIN_WIDTH, int(event.width))
        h = max(self.MIN_HEIGHT, int(event.height))
        self.app.preferences[self._prefs_size_key] = (w, h)


    def _show_cb(self, widget):
        for w in self.get_content_area():
            w.show_all()
        if not self._motion_handler_id:
            h_id = self.connect("motion-notify-event", self._motion_cb)
            self._motion_handler_id = h_id
        self.grab_add()


    def _hide_cb(self, widget):
        self.grab_remove()
        if self._motion_handler_id is not None:
            self.disconnect(self._motion_handler_id)
        self._motion_handler_id = None
        self._size = None
        self._entered = None


    def _motion_cb(self, widget, event):
        # Ensure that at some point the user started inside a visible window
        if not self.get_visible():
            return
        if not self._entered:
            return
        if self._size is None:
            return
        # Moving outside the window is equivalent to rejecting the choices
        # on offer. Leave some slack so that more recent WMs/themes with
        # invisible grabs outside the window can resize the window.
        x, y, w, h = self._size
        px, py = event.x_root, event.y_root
        s = self.LEAVE_SLACK
        moved_outside = px < x-s or py < y-s or px > x+w+s or py > y+h+s
        if moved_outside:
            self.response(gtk.RESPONSE_REJECT)


    def _enter_cb(self, widget, event):
        if event.mode != gdk.CROSSING_NORMAL:
            return
        self._entered = True


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
            gdk_window = widget.get_window()
            gdk_window.set_type_hint(gdk.WINDOW_TYPE_HINT_UTILITY)
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


