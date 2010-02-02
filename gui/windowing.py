# Base classes for window types
# Management of the user's chosen subwindows as a group.

import gtk
import gtk.gdk as gdk

class UserSubWindows:
    """
    The set of windows that the user has chosen to have open.
    """

    def __init__(self, app):
        "Constructor."
        self.app = app
        self.windows = []
        self.hidden = False
        self._in_command = False

    def register(self, window):
        "Register a window for monitoring: called during window construction."
        window.connect('map-event', self.on_registered_window_map_event)
        window.connect('hide', self.on_registered_window_hide)
        if window.flags() & gtk.VISIBLE:
            self.windows.append(widget)

    def toggle(self):
        "Toggles the user's subwindows between shown and not-shown."
        if self.hidden:
            self.show()
        else:
            self.hide()
    
    def show(self):
        "Shows all of the user's subwindows reversibly."
        self._in_command = True
        for w in self.windows:
            w.show_all()
        self.hidden = False
        self._in_command = False

    def hide(self):
        "Hides all of the user's subwindows reversibly."
        self._in_command = True
        for w in self.windows:
            w.hide()
        self.hidden = True
        self._in_command = False
   
    def on_registered_window_map_event(self, window, event):
        "Track external shows, update internal state accordingly"
        # Add subwindows that are shown by external causes to the user subset
        if not self._in_command:
            if window not in self.windows:
                self.windows.append(window)

    def on_registered_window_hide(self, window):
        "Track external hides, update internal state accordingly"
        if not self._in_command:
            if window in self.windows:
                self.windows.remove(window)


class Dialog (gtk.Dialog):
    """
    A dialog. Dialogs are a bit of a rough edge at the moment; currently only
    the preferences window is one. Dialogs accept keyboard input, and hide and
    show with Tab (though it's arguable that they shouldn't).
    """
    def __init__(self, app, *args, **kwargs):
        gtk.Dialog.__init__(self, *args, **kwargs)
        self.app = app
        self.app.user_subwindows.register(self)
    # TODO: dialogs should be freely positioned by the window manager



class MainWindow (gtk.Window):
    """
    The main window in the GUI. No code to see here yet, go look at
    drawwindow.DrawWindow.
    """
    def __init__(self, app):
        gtk.Window.__init__(self, type=gtk.WINDOW_TOPLEVEL)
        self.app = app
        self.app.kbm.add_window(self)


class SubWindow (gtk.Window):
    """
    A subwindow in the GUI. All are utility windows that are transients for
    the main window. Subwindows remember their position when hidden or shown,
    in a way that keeps as many window managers as possible happy.

    Most SubWindows don't accept keyboard input, but if your subclass requires
    it, pass key_input to the constructor.
    """
    
    def __init__(self, app, key_input=False):
        gtk.Window.__init__(self, type=gtk.WINDOW_TOPLEVEL)
        self.app = app
        if not key_input:
            self.app.kbm.add_window(self)
            # TODO: do we need a separate class for keyboard-input-friendly
            # windows? Do they share anything in common with dialogs (could
            # they all be implmented as dialogs?)
        self.pre_hide_pos = None
        self.app.user_subwindows.register(self)
        self.connect("realize", self.on_realize)

    def on_realize(self, widget):
        # Mark subwindows as utility windows: many X11 WMs handle this sanely
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
