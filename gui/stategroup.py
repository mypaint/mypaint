import gtk, gobject
gdk = gtk.gdk
import time

class StateGroup():
    """
    Supervisor instance for GUI states.

    A GUI state is a mode which the GUI is in, for example an active
    popup window or a special (usually short-lived) view on the
    document. The application defines functions to be called when the
    state is entered or left.

    This class mainly deals with the various ways how the user can
    leave such a mode, eg. if the mode is entered by holding down a
    key long enough, it will be left when the key is released.
    """

    def __init__(self):
        self.states = []
        self.keys_pressed = {}

    def add_source_widget(self, widget):
        """
        This should be called for every widget from which a state can
        be activated through a keypress.
        """
        widget.connect("key-press-event", self.key_press_cb)
        widget.connect("key-release-event", self.key_release_cb)

    def get_active_states(self):
        return [s for s in self.states if s.active]
    active_states = property(get_active_states)

    def key_press_cb(self, widget, event):
        self.keys_pressed[event.keyval] = True
        for s in self.active_states:
            s.key_press_cb(widget, event)

    def key_release_cb(self, widget, event):
        self.keys_pressed[event.keyval] = False
        for s in self.active_states:
            s.key_release_cb(widget, event)

    def create_state(self, enter, leave, popup=None):
        s = State(self, popup)
        s.popup = None
        s.on_enter = enter
        s.on_leave = leave
        self.states.append(s)
        return s

    def create_popup_state(self, popup):
        return self.create_state(popup.enter, popup.leave, popup)

class State:
    "how long a key can be held down to go through as single hit (and not press-and-hold)"
    max_key_hit_duration = 0.200
    "the state is automatically left after this time (ignored during press-and-hold)"
    autoleave_timeout = 0.800
    #"popups only: how long the cursor is allowed outside before closing (ignored during press-and-hold)"
    #outside_popup_timeout = 0.050
    "state to activate when this state is activated while already active (None = just leave this state)"
    next_state = None

    def __init__(self, stategroup, popup):
        self.sg = stategroup
        self.active = False
        self.action = None
        self.popup = popup
        self.autoleave_timer = None
        self.outside_popup_timer = None
        if popup:
            popup.connect("enter-notify-event", self.popup_enter_notify_cb)
            popup.connect("leave-notify-event", self.popup_leave_notify_cb)
            popup.popup_state = self # FIXME: hacky?
            self.outside_popup_timeout = popup.outside_popup_timeout

    def enter(self):
        #print 'entering state, calling', self.on_enter.__name__
        assert not self.active
        self.active = True
        self.enter_time = time.time()
        if self.autoleave_timeout:
            self.autoleave_timer = gobject.timeout_add(int(1000*self.autoleave_timeout), self.autoleave_timeout_cb)
        self.on_enter()

    def leave(self):
        #print 'leaving state, calling', self.on_leave.__name__
        assert self.active
        self.active = False
        self.action = None
        if self.autoleave_timer:
            gobject.source_remove(self.autoleave_timer)
            self.autoleave_timer = None
        if self.outside_popup_timer:
            gobject.source_remove(self.outside_popup_timer)
            self.outside_popup_timer = None
        self.on_leave()

    def activate(self, action=None):
        """
        Called from the GUI code, eg. when a gtk.Action is
        activated. The action is used to figure out the key.
        """
        if self.active:
            if not self.keydown:
                # pressing the key again
                # TODO: allow different actions (eg. bring up another dialog on double-hit)
                # FIXME: should special-case automatic double-activations vs user double-keyhit
                self.leave()
                if self.next_state:
                    self.next_state.activate()
            return
        self.action = action
        self.keydown = False
        if action:
            keyval, modifiers = gtk.accel_map_lookup_entry(action.get_accel_path())
            if self.sg.keys_pressed.get(keyval):
                # The user has activated the action by hitting the
                # accelerator key. If we do not see any key released
                # event, we know that the user is holding the key down
                # deliberately long.
                self.keydown = True
            self.keyval = keyval
        else:
            self.keyval = None

        self.enter()

    def key_press_cb(self, widget, event):
        if event.keyval == self.keyval:
            pass # probably keyboard autorepeat (self.activate will be called also)
        else:
            # any keypress leaves the action
            self.leave()

    def key_release_cb(self, widget, event):
        if event.keyval == self.keyval:
            if self.keydown:
                self.keydown = False
                if time.time() - self.enter_time < self.max_key_hit_duration:
                    pass # accept as one-time hit
                else:
                    self.leave()

    def autoleave_timeout_cb(self):
        if not self.keydown:
            self.leave()
    def outside_popup_timeout_cb(self):
        if not self.keydown:
            self.leave()

    def popup_enter_notify_cb(self, widget, event):
        if not self.active:
            return
        if self.outside_popup_timer:
            gobject.source_remove(self.outside_popup_timer)
            self.outside_popup_timer = None

    def popup_leave_notify_cb(self, widget, event):
        if not self.active:
            return
        # allow to leave the window for a short time
        if self.outside_popup_timer:
            gobject.source_remove(self.outside_popup_timer)
        self.outside_popup_timer = gobject.timeout_add(int(1000*self.outside_popup_timeout), self.outside_popup_timeout_cb)

