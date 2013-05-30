# Base classes for window types

import sys
import os.path

import gtk
import gtk.gdk as gdk
import gobject


class Dialog (gtk.Dialog):
    """Base dialog accepting all keyboard input.

    Dialogs hide when closed. By default, they accept all keyboard input and
    are not modal. They can (and should) be kept around as references, and can
    be freely hidden and shown after construction.

    """
    def __init__(self, app, *args, **kwargs):
        gtk.Dialog.__init__(self, *args, **kwargs)
        self.app = app
        self.connect('delete-event', lambda w,e: self.hide_on_delete())


class ChooserDialog (Dialog):
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
        Dialog.__init__(self, app, title=title, parent=parent,
                        flags=flags, buttons=buttons)
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


class SubWindow (gtk.Window):
    """A subwindow in the GUI.

    SubWindows don't accept keyboard input by default, but if your subclass
    requires it, pass key_input to the constructor.

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
        self.connect('delete-event', lambda w,e: self.hide_on_delete())
        # Mark subwindows as utility windows: many X11 WMs handle this sanely
        # This has caused issues with OSX and X11.app under GTK2/PyGTK in the
        # past. OSX builds no longer use X11.app, so this should no longer
        # need special-casing. Testers: use if not sys.platform == 'darwin':
        # if needed, and please submit a patch. https://gna.org/bugs/?15838
        self.set_type_hint(gdk.WINDOW_TYPE_HINT_UTILITY)
        # Win32 and some Linux DEs are responsive to the following: keeps the
        # window above the main window in fullscreen.
        self.set_transient_for(app.drawWindow)

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


