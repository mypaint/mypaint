# Base classes for window types
# Management of the user's chosen subwindows as a group.

import sys
import os.path

import gtk
import gtk.gdk as gdk
import gobject

# The window geometry that will be assumed at save time.
SAVE_POS_GRAVITY = gdk.GRAVITY_NORTH_WEST

# Window names and their default positions and visibilities. They interrelate,
# so define them together.
WINDOW_DEFAULTS = {
    'drawWindow':           (None, True), # initial geom overridden, but set vis
    # Colour choosers go in the top-right quadrant by default
    'colorSamplerWindow':   ("220x220-50+100",  False),
    'colorSelectionWindow': ("-250+150",        True), # positions strangely
    # Brush-related dialogs go in the bottom right by default
    'brushSelectionWindow': ("240x350-50-50",   True ),
    'brushSettingsWindow':  ("400x400-325-50",  False),
    # Preferences go "inside" the main window (TODO: just center-on-parent?)
    'preferencesWindow':    ("+200+100",        False),
    # Layer details in the bottom-left quadrant.
    'layersWindow':         ("200x300+50-50",   False),
    'backgroundWindow':     ("500x400-275+75", False),
    # Debug menu
    'inputTestWindow':      (None, False),
}

# The main drawWindow gets centre stage, sized around the default set of
# visible utility windows. Sizing constraints for this:
CENTRE_STAGE_CONSTRAINTS = (\
    166, # left: leave some space for icons (but: OSX?)
    50,  # top: avoid panel
    75,  # bottom: avoid panel
    10,  # right (small screens): ensure close button not covered at least
    220, # right (big screens): don't overlap brushes or colour (much)
    # "big screens" are anything 3 times wider than the big-right margin:
    ## > Size. The Center Stage content should be at least twice as wide as
    ## > whatever is in its side margins, and twice as tall as its top and
    ## > bottom margins. (The user may change its size, but this is how it
    ## > should be when the user first sees it.)
    ## >     -- http://designinginterfaces.com/Center_Stage
)


def hide_window_cb(window, event):
    # used by some of the windows
    window.hide()
    return True

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
            self.windows.append(window)

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
            w.present()
        self.hidden = False

        # For metacity. See https://gna.org/bugs/?15990
        def refocus_drawwindow(*junk):
            self.app.drawWindow.window.focus()
        gobject.idle_add(refocus_drawwindow)

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

        self.connect('delete-event', hide_window_cb)

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


def centre_stage(window, l, t, b, rsmall, rbig):
    """
    Applies a "Centre Stage" geometry to a given window, which is moved
    and resized to fit within its current monitor using the given constraints.
    There are two right margins: one for small screens and one for big
    screens; a big screen is one that's 3 times greater than the "big"
    right margin.
    """
    screen = window.get_screen()
    gdk_window = window.window
    if screen is None or gdk_window is None:
        return
    monitor_num = screen.get_monitor_at_window(gdk_window)
    monitor_geom = screen.get_monitor_geometry(monitor_num)
    # Determine frame geometry if we can...
    w, h = window.get_size()
    frame_size = gdk_window.get_frame_extents()
    frame_w = frame_size.width - w
    frame_h = frame_size.height - h
    
    mon_w, mon_h = monitor_geom.width, monitor_geom.height
    mon_x, mon_y = monitor_geom.x, monitor_geom.y
    window.move(mon_x + l, mon_y + t)
    if monitor_geom.width <= rbig * 3:
        window.resize(mon_w-(rsmall+l+frame_w), mon_h-(t+b+frame_h))
    else:
        window.resize(mon_w-(rbig+l+frame_w), mon_h-(t+b+frame_h))
        

def move_to_monitor_of(win, targ_win):
    """
    Moves win to the same monitor as targ_win, preserving its gravity and
    screen position.
    """
    s = win.get_screen()
    targ_s = targ_win.get_screen()
    if s is None or targ_s is None or s.get_number() != targ_s.get_number():
        return
    w = win.window
    targ_w = targ_win.window
    if w is None or targ_w is None:
        return
    mon = s.get_monitor_at_window(w)
    targ_mon = s.get_monitor_at_window(targ_w)
    if mon == targ_mon:
        return
    mg = s.get_monitor_geometry(mon)
    targ_mg = s.get_monitor_geometry(targ_mon)
    grav = win.get_gravity()
    pos = win.get_position()
    targ_pos = pos[0]-mg.x+targ_mg.x, pos[1]-mg.y+targ_mg.y
    print "gravitate: correcting position from %s to %s" % (pos, targ_pos)
    win.move(*targ_pos)

class WindowManager(object):

    def __init__(self, app):
        self.app = app
        self.config_file_path = os.path.join(self.app.confpath, 'windowpos.conf')

        self.windows = {} # {'windowname': window_instance }
        self.window_names = [n for n in WINDOW_DEFAULTS.keys()]

        self.main_window = self.get_window('drawWindow')
        self.user_subwindows = UserSubWindows(self.app)

        self.app.drawWindow = self.main_window

        # Can't initialize on-demand because of save_window_positions implementation
        for name in self.window_names:
            self.get_window(name)

    def get_window(self, window_name):
        """Return the window instance of window_name. Initializes the window on-demand.
        Imports the python module with the same name as the window (all lower-case),
        and instantiate an object of the class Window in that module."""
        if window_name in self.windows:
            return self.windows[window_name]

        # Load and initialize window
        module = __import__(window_name.lower(), globals(), locals(), [])
        window = self.windows[window_name] = module.Window(self.app)
        using_default = self.load_window_position(window_name, window)

        if using_default:
            if window_name == 'drawWindow':
                def on_map_event(win, ev):
                    centre_stage(win, *CENTRE_STAGE_CONSTRAINTS)
                window.connect('map-event', on_map_event)
            else:
                def on_map_event(win, ev, main_window):
                    move_to_monitor_of(win, main_window)
                window.connect('map-event', on_map_event, self.main_window)

        if window_name != 'drawWindow':
            self.user_subwindows.register(window)

        return window

    def load_window_position(self, name, window):
        geometry, visible = WINDOW_DEFAULTS.get(name, (None, False))
        using_default = True
        try:
            f = open(self.config_file_path)
            for line in f:
                if line.startswith(name):
                    parts = line.split()
                    visible = parts[1] == 'True'
                    x, y, w, h = [int(i) for i in parts[2:2+4]]
                    geometry = '%dx%d+%d+%d' % (w, h, x, y)
                    using_default = False
                    break
            f.close()
        except IOError:
            pass
        # Initial gravities can be all over the place. Fix aberrant ones up
        # when the windows are safely on-screen so their position can be
        # saved sanely. Doing this only when the window's mapped means that the
        # window position stays where we specified in the defaults, and the
        # window manager (should) compensate for us without the position
        # changing.
        if geometry is not None:
            window.parse_geometry(geometry)
            if using_default:
                initial_gravity = window.get_gravity()
                if initial_gravity != SAVE_POS_GRAVITY:
                    def fix_gravity(w, event):
                        if w.get_gravity() != SAVE_POS_GRAVITY:
                            w.set_gravity(SAVE_POS_GRAVITY)
                    window.connect("map-event", fix_gravity)
        if visible:
            window.show_all()
        return using_default

    # FIXME: assumes all windows have been initialized
    def save_window_positions(self):
        f = open(self.config_file_path, 'w')
        f.write('# name visible x y width height\n')
        for name in self.window_names:
            window = self.get_window(name)
            x, y = window.get_position()
            w, h = window.get_size()
            gravity = window.get_gravity()
            if gravity != SAVE_POS_GRAVITY:
                # Then it was never mapped, and must still be using the
                # defaults. Don't save position (which'd be wrong anyway).
                continue
            if hasattr(window, 'geometry_before_fullscreen'):
                x, y, w, h = window.geometry_before_fullscreen
            visible = window in self.user_subwindows.windows \
                or window.get_property('visible')
            f.write('%s %s %d %d %d %d\n' % (name, visible, x, y, w, h))
        f.close()

