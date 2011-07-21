import gtk
gdk = gtk.gdk
from gettext import gettext as _
import gobject
import pango

import tileddrawwidget, document

import dialogs
import os

from hashlib import md5

def stock_button_generic(stock_id, b):
    img = gtk.Image()
    img.set_from_stock(stock_id, gtk.ICON_SIZE_MENU)
    b.add(img)
    return b

def stock_button(stock_id):
    b = gtk.Button()
    return stock_button_generic(stock_id, b)

class ToolWidget (gtk.VBox):

    tool_widget_title = _("Scratchpad")

    def __init__(self, app):
        gtk.VBox.__init__(self)
        self.app = app
        #self.set_size_request(200, 250)

        self.is_updating = False
        
        # Common controls
        load_button = self.load_button = stock_button(gtk.STOCK_OPEN)
        load_button.set_tooltip_text(_("Load Scratchpad"))
        save_as_button = self.save_as_button = stock_button(gtk.STOCK_SAVE_AS)
        save_as_button.set_tooltip_text(_("Save Scratchpad as..."))
        revert_button = self.revert_button = stock_button(gtk.STOCK_UNDO)
        revert_button.set_tooltip_text(_("Revert Scratchpad"))
        new_button = self.delete_button = stock_button(gtk.STOCK_NEW)
        new_button.set_tooltip_text(_("Clear Scratchpad"))

        load_button.connect('clicked', self.load_cb)
        save_as_button.connect('clicked', self.save_as_cb)
        revert_button.connect('clicked', self.revert_cb)
        new_button.connect('clicked', self.new_cb)

        buttons_hbox = gtk.HBox()
        buttons_hbox.pack_start(new_button)
        buttons_hbox.pack_start(load_button)
        buttons_hbox.pack_start(save_as_button)
        buttons_hbox.pack_start(revert_button)

        scratchpad_view = app.filehandler.scratchpad_doc.tdw

        self.connect("button-press-event", self.button_press_cb)
        self.connect("button-release-event",self.button_release_cb)
        self.connect("destroy-event", self.save_cb)
        self.connect("delete-event", self.save_cb)

        scratchpad_box = gtk.EventBox()
        scratchpad_box.add(scratchpad_view)

        self.pack_start(scratchpad_box)
        self.pack_start(buttons_hbox, expand=False)

        

        # Updates
        doc = app.filehandler.scratchpad_doc.model
        doc.doc_observers.append(self.update)

        # FIXME pull the scratchpad filename from preferences instead of this
        # self.app.filehandler.scratchpad_filename = self.scratchpad_filename = os.path.join(self.app.filehandler.get_scratchpad_prefix(), "scratchpad_default.ora")

        self.update(app.filehandler.scratchpad_doc)

    def zoom_in_cb(self, action):
        self.app.filehandler.scratchpad_doc.zoom("ZoomIn")
    
    def zoom_out_cb(self, action):
        self.app.filehandler.scratchpad_doc.zoom("ZoomOut")

    def new_cb(self, action):
        self.app.filehandler.scratchpad_doc.model.clear()

    def revert_cb(self, action):
        # Load last scratchpad
        if os.path.isfile(self.app.filehandler.scratchpad_filename):
            self.app.filehandler.open_scratchpad(self.app.filehandler.scratchpad_filename)
            print "Reverted to %s" % self.app.filehandler.scratchpad_filename
        else:
            print "No file to revert to yet."

    def load_cb(self, action):
        if self.app.filehandler.scratchpad_filename:
            self.save_cb(action)
        self.app.filehandler.open_scratchpad_dialog()

    def update(self, doc):
        if self.is_updating:
            return
        self.is_updating = True

    def save_as_cb(self, action):
        self.app.filehandler.save_scratchpad_as_dialog()

    def save_cb(self, action):
        print "Saving the scratchpad"
        self.app.filehandler.save_scratchpad(self.scratchpad_filename)

    def button_press_cb(self, win, event):
        #print event.device, event.button

        ## Ignore accidentals
        # Single button-presses only, not 2ble/3ple
        if event.type != gdk.BUTTON_PRESS:
            # ignore the extra double-click event
            return False

        if event.button != 1:
            # check whether we are painting (accidental)
            if event.state & gdk.BUTTON1_MASK:
                # Do not allow dragging in the middle of
                # painting. This often happens by accident with wacom
                # tablet's stylus button.
                #
                # However we allow dragging if the user's pressure is
                # still below the click threshold.  This is because
                # some tablet PCs are not able to produce a
                # middle-mouse click without reporting pressure.
                # https://gna.org/bugs/index.php?15907
                return False

        # Pick a suitable config option
        ctrl = event.state & gdk.CONTROL_MASK
        alt  = event.state & gdk.MOD1_MASK
        shift = event.state & gdk.SHIFT_MASK
        if shift:
            modifier_str = "_shift"
        elif alt or ctrl:
            modifier_str = "_ctrl"
        else:
            modifier_str = ""
        prefs_name = "input.button%d%s_action" % (event.button, modifier_str)
        action_name = self.app.preferences.get(prefs_name, "no_action")

        # No-ops
        if action_name == 'no_action':
            return True  # We handled it by doing nothing

        """
        # Straight line
        # Really belongs in the tdw, but this is the only object with access
        # to the application preferences.
        if action_name == 'straight_line':
            self.app.doc.tdw.straight_line_from_last_pos(is_sequence=False)
            return True
        if action_name == 'straight_line_sequence':
            self.app.doc.tdw.straight_line_from_last_pos(is_sequence=True)
            return True
        """

        # View control
        if action_name.endswith("_canvas"):
            dragfunc = None
            if action_name == "pan_canvas":
                dragfunc = self.app.filehandler.scratchpad_doc.dragfunc_translate
            elif action_name == "zoom_canvas":
                dragfunc = self.app.filehandler.scratchpad_doc.dragfunc_zoom
            elif action_name == "rotate_canvas":
                dragfunc = self.app.filehandler.scratchpad_doc.dragfunc_rotate
            if dragfunc is not None:
                self.app.filehandler.scratchpad_doc.tdw.start_drag(dragfunc)
                return True
            return False
        """
        # Application menu
        if action_name == 'popup_menu':
            self.show_popupmenu(event=event)
            return True

        if action_name in self.popup_states:
            state = self.popup_states[action_name]
            state.activate(event)
            return True
        """
        # Dispatch regular GTK events.
        for ag in [self.app.doc.action_group]:
            action = ag.get_action(action_name)
            if action is not None:
                action.activate()
                return True

    def button_release_cb(self, win, event):
        #print event.device, event.button
        doc = self.app.filehandler.scratchpad_doc
        tdw = doc.tdw
        if tdw.dragfunc is not None:
            tdw.stop_drag(doc.dragfunc_translate)
            tdw.stop_drag(doc.dragfunc_rotate)
            tdw.stop_drag(doc.dragfunc_zoom)
        return False

