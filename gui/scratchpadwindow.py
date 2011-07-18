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

def stock_checkbutton(stock_id):
    b = gtk.CheckButton()
    return stock_button_generic(stock_id, b)

class ToolWidget (gtk.VBox):

    tool_widget_title = _("Scratchpad")

    def __init__(self, app):
        gtk.VBox.__init__(self)
        self.app = app
        #self.set_size_request(200, 250)

        self.is_updating = False
        
        # Common controls
        add_button = self.add_button = stock_button(gtk.STOCK_ADD)
        save_button = self.save_button = stock_button(gtk.STOCK_SAVE)
        next_button = self.next_button = stock_button(gtk.STOCK_GO_FORWARD)
        previous_button = self.previous_button = stock_button(gtk.STOCK_GO_BACK)
        delete_button = self.delete_button = stock_button(gtk.STOCK_DELETE)

        zoom_in_button = self.zoom_in = stock_button(gtk.STOCK_ZOOM_IN)
        zoom_out_button = self.zoom_out = stock_button(gtk.STOCK_ZOOM_OUT)
        load_special_button = self.load_special = stock_button(gtk.STOCK_HOME)

        add_button.connect('clicked', self.add_cb)
        save_button.connect('clicked', self.save_cb)
        next_button.connect('clicked', self.next_scratchpad_cb)
        previous_button.connect('clicked', self.previous_scratchpad_cb)
        delete_button.connect('clicked', self.delete_cb)

        zoom_in_button.connect('clicked', self.zoom_in_cb)
        zoom_out_button.connect('clicked', self.zoom_out_cb)
        load_special_button.connect('clicked', self.load_special_cb)

        buttons_hbox = gtk.HBox()
        buttons_hbox.pack_start(add_button)
        buttons_hbox.pack_start(save_button)
        buttons_hbox.pack_start(previous_button)
        buttons_hbox.pack_start(next_button)
        buttons_hbox.pack_start(delete_button)

        zoom_box = gtk.VBox()
        zoom_box.pack_start(load_special_button)
        zoom_box.pack_start(zoom_in_button)
        zoom_box.pack_start(zoom_out_button)

        scratchpad_view = app.filehandler.scratchpad_doc.tdw

        self.connect("button-press-event", self.button_press_cb)
        self.connect("button-release-event",self.button_release_cb)

        scratchpad_box = gtk.HBox()
        scratchpad_box.pack_start(scratchpad_view)
        scratchpad_box.pack_start(zoom_box, expand=False)

        self.pack_start(scratchpad_box)
        self.pack_start(buttons_hbox, expand=False)

        # Updates
        doc = app.filehandler.scratchpad_doc.model
        doc.doc_observers.append(self.update)

        # Load last? scratchpad
        self.scratchpads = app.filehandler.list_scratchpads_grouped()

        # scratchpad cursor+flags
        self.started_scratchpads = False
        self.cursor = 0

        self.update(app.filehandler.scratchpad_doc)

    def zoom_in_cb(self, action):
        self.app.filehandler.scratchpad_doc.zoom("ZoomIn")
    
    def zoom_out_cb(self, action):
        self.app.filehandler.scratchpad_doc.zoom("ZoomOut")

    def delete_cb(self, action):
        # Remove all the scratchpads in this group
        if self.app.filehandler.scratchpad_filename and self.cursor >= 0 and self.cursor < len(self.scratchpads) and len(self.scratchpads):
            g = self.scratchpads[self.cursor]
            if g:
                self.app.filehandler.delete_scratchpads(g)
                self.scratchpads = self.app.filehandler.list_scratchpads_grouped()
                self.add_cb(action)

    def update(self, doc):
        if self.is_updating:
            return
        self.is_updating = True

    def save_cb(self, action):
        self.app.filehandler.save_scratchpad_cb(action)
        self.scratchpads = self.app.filehandler.list_scratchpads_grouped()
        for idx in xrange(len(self.scratchpads)):
            if self.app.filehandler.scratchpad_filename in self.scratchpads[idx]:
                self.cursor = idx

    def add_cb(self, action):
        self.app.filehandler.scratchpad_filename = None
        self.app.filehandler.scratchpad_doc.model.clear()

    def load_special_cb(self, action):
        if self.app.filehandler.scratchpad_filename:
            scratchpad_prefix, scratchpad_file = os.path.split(self.app.filehandler.scratchpad_filename)
        else:
            scratchpad_prefix, scratchpad_file = self.app.filehandler.get_scratchpad_prefix(), None
        filename = self.app.filehandler.filename

        md5_filename = None

        if filename:
            md5_filename = "_md5" + md5(filename).hexdigest() + ".ora"
            if not os.path.isdir(os.path.join(scratchpad_prefix, "special")):
                os.mkdir(os.path.join(scratchpad_prefix, "special"))

        scratchpads = [t for h,t in map(os.path.split, self.app.filehandler.list_scratchpads())]
        
        if not filename:
            # File hasn't been saved yet
            # TODO pop up an alert to warn about this
            print "File hasn't been saved yet and so doesn't have a filename"
            self.app.message_dialog("You cannot link a scratchpad to the canvas until you save the main canvas.", type=gtk.MESSAGE_ERROR)
        elif md5_filename == scratchpad_file:
            # Scratchpad is already linked to current working file
            # Act as if the 'save' button has been pressed
            print "Scratchpad is already linked to current working file"
            # Save
            self.app.filehandler.scratchpad_filename = os.path.join(scratchpad_prefix, "special", md5_filename)
            self.save_cb(action)
            # # Reload file
            # self.app.filehandler.open_scratchpad(self.app.filehandler.scratchpad_filename)
        elif md5_filename in scratchpads:
            if scratchpad_file:
                print "There is a filename, but it's not the same as the current filename's md5"
                # There is a filename, and the scratchpad has been saved but under a different name
                # -> if a scratchpad exists with the same filename, load that after saving current
                # -> if no scratchpad exists, rename the current one to match
                if self.app.filehandler.scratchpad_filename:
                    self.app.filehandler.save_scratchpad(self.app.filehandler.scratchpad_filename)
            self.app.filehandler.scratchpad_filename = os.path.join(scratchpad_prefix, "special", md5_filename)
            self.app.filehandler.open_scratchpad(self.app.filehandler.scratchpad_filename)
        else:
            # No special scratchpad file exists yet
            print "No linked scratchpad found for this file - creating it."
            self.app.message_dialog("Saving the current scratchpad to this canvas.", type=gtk.MESSAGE_INFO)
            self.app.filehandler.scratchpad_filename = os.path.join(scratchpad_prefix, "special", md5_filename)
            self.save_cb(action)
            if self.app.filehandler.lastsavefailed:
                self.app.filehandler.scratchpad_filename = None

    def next_scratchpad_cb(self, action):
        if self.app.filehandler.scratchpad_filename:
            self.save_cb(action)
        if self.started_scratchpads:
            self.cursor += 1
            if self.cursor == len(self.scratchpads):
                self.cursor = 0
        self.started_scratchpads = True
        if len(self.scratchpads) > 0:
            self.app.filehandler.scratchpad_filename = self.scratchpads[self.cursor][-1]
            self.app.filehandler.open_scratchpad(self.app.filehandler.scratchpad_filename)

    def previous_scratchpad_cb(self, action):
        if self.app.filehandler.scratchpad_filename:
            self.save_cb(action)
        if self.started_scratchpads:
            self.cursor -= 1
            if self.cursor < 0:
                self.cursor = len(self.scratchpads) - 1
        self.started_scratchpads = True
        if len(self.scratchpads) > 0:
            self.app.filehandler.scratchpad_filename = self.scratchpads[self.cursor][-1]
            self.app.filehandler.open_scratchpad(self.app.filehandler.scratchpad_filename)

    
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
        for ag in self.action_group, self.app.doc.action_group:
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

