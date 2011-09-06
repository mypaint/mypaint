# This file is part of MyPaint.
# Copyright (C) 2011 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
from gtk import gdk
import gobject


class DropdownPanelButton (gtk.ToggleButton):
    """Button which drops down a panel with arbitrary widgets.

    The panel iself is a specialised `gtk.Window` tightly bound to its button.
    A widget can be added to it using `set_panel_widget()`.
    """

    def __init__(self, label_widget):
        """Construct with a button label widget.
        """
        gtk.ToggleButton.__init__(self)
        arrow = gtk.Arrow(gtk.ARROW_DOWN, gtk.SHADOW_IN)
        hbox = gtk.HBox()
        hbox.pack_start(label_widget, True, True)
        hbox.pack_start(arrow, False, False)
        self.add(hbox)
        self._panel = DropdownPanel(self)
        self.connect("button-press-event", self._button_press_cb)
        self._panel.connect("hide", lambda *a: self.set_active(False))
        self._panel.connect("show", lambda *a: self.set_active(True))


    def set_panel_widget(self, widget):
        """Sets the dropdown panel's content widget.
        """
        old = self._panel.get_child()
        if old:
            self._panel.remove(old)
        self._panel.add(widget)


    def panel_hide(self):
        """Hides the panel.
        
        Call this after the user makes a final selection.
        """
        self._panel.popdown()


    def _button_press_cb(self, widget, event):
        if not self.get_active():
            # feels nicer to do this in an idle handler
            gobject.idle_add(self._panel.popup, event.time)
        return True


class DropdownPanel (gtk.Window):

    def __init__(self, panel_button):
        gtk.Window.__init__(self, type=gtk.WINDOW_POPUP)
        self.set_modal(True)
        self.set_focus_on_map(False)
        self.set_can_focus(False)
        self._panel_button = panel_button
        self._mapped_once = False
        self._configured = False
        self._grabbed = False
        self.connect("map-event", self._map_event_cb)
        self.connect("button-press-event", self._button_press_event_cb)
        self.connect("hide", self._hide_cb)
        self.connect("configure-event", self._configure_cb)
        self.connect("grab-broken-event", self._grab_broken_event_cb)
        self.connect("leave-notify-event", self._leave_notify_cb)


    def _leave_notify_cb(self, widget, event):
        # Widgets packed inside the the panel may grab the pointer, breaking
        # our grab. That's mostly OK provided we arrange to pop the panel down
        # when the pointer leaves it after the packed widget has done its
        # thing. This limits the choice of widgets though, since those that pop
        # up their own stuff can't be reliably used: the pointer may be
        # naturally outside our panel afterwards as a result of selection.
        #
        # Moral: pack simple widgets and not fancy ones with popup menus. Grabs
        # for things like the colour triangle break ours, but that now might
        # actually be beneficial.

        if self._grabbed:
            return
        if event.mode != gdk.CROSSING_NORMAL:
            # Not simply leaving/entering a window
            return

        ## We *could* regrab assertively here in respone to motion, leave and
        ## enter, but it's fairly unreliable.
        #if False:
        #    print "regrabbing"
        #    self.establish_grab(event.time)
        #    return

        ex, ey = event.x_root, event.y_root
        x, y = self.window.get_origin()
        w, h = self.allocation.width, self.allocation.height
        is_inside = ex < x or ey < y or ex > x+w or ey > y+h

        # Test is needed because widgets that break grabs vary in what gets
        # reported after they do their own grab release (GtkComboBox can report
        # strange is_inside CROSSING_NORMAL leaves when its little menu pops
        # down, for example). Don't use that (see above), but let's try to be
        # as sane as we can be even if we see one.

        if is_inside:
            self.popdown()
            return True


    def establish_grab(self, t=0):
        # Grab, permitting normal interaction with the app (i.e. with the widgets
        # in the panel).
        mask = gdk.BUTTON_PRESS_MASK
        grab_result = gdk.pointer_grab(self.window, True, mask, None, None, t)
        if grab_result != gdk.GRAB_SUCCESS:
            print "grab failed:", grab_result
            return False
        # But limit events to just the panel for neatness and a hint of modality.
        self.grab_add()
        self._grabbed = True
        return True

    def _grab_broken_event_cb(self, widget, event):
        rival = event.grab_window
        if rival is not None:
            print "grab broken by", rival
        self._grabbed = False

    def popup(self, t=0):
        parent_window = self._panel_button.get_toplevel()
        self.set_transient_for(parent_window)
        self.show_all()
        if not self.establish_grab(t):
            self.hide()
            return

    def popdown(self):
        self.hide()

    def _hide_cb(self, widget):
        if self._grabbed:
            gdk.pointer_ungrab()
            self._grabbed = False
        self.grab_remove()

    def _realize_cb(self, widget):
        self.window.set_type_hint(gdk.WINDOW_TYPE_HINT_DROPDOWN_MENU)

    def _map_event_cb(self, widget, event):
        # Set initial geometry as best we can.
        x, y = self._panel_button.window.get_origin()
        x += self._panel_button.allocation.x
        y += self._panel_button.allocation.y
        y += self._panel_button.allocation.height
        x, y = int(x), int(y)
        if self._mapped_once:
            self.move(x, y)
        else:
            self.parse_geometry("+%d+%d" % (x, y))
        self._mapped_once = True
        self._configured = False

    def _configure_cb(self, widget, event):
        # Constrain window to fit on the current monitor, if possible.
        if self._configured:
            return
        self._configured = True
        screen = event.get_screen()
        mon = screen.get_monitor_at_window(self.window)
        mon_geom = screen.get_monitor_geometry(mon)
        x, y, w, h = (event.x, event.y, event.width, event.height)
        if y+h > mon_geom.y + mon_geom.height:
            y = mon_geom.y + mon_geom.height - h
        if x+w > mon_geom.x + mon_geom.width:
            x = mon_geom.x + mon_geom.width - w
        if x < mon_geom.x:
            x = mon_geom.x
        if y < mon_geom.y:
            y = mon_geom.y
        if (x, y) != (event.x, event.y):
            self.move(x, y)

    def _button_press_event_cb(self, widget, event):
        # Dismiss if the user clicks outside the panel.
        if not self._grabbed:
            return
        if not self.window:
            return
        x, y = self.window.get_origin()
        w, h = self.allocation.width, self.allocation.height
        if event.x_root < x or event.y_root < y \
                or event.x_root > x+w or event.y_root > y+h:
            self.popdown()
            return True

if __name__ == '__main__':
    import os, sys
    script = os.path.basename(sys.argv[0])
    win = gtk.Window()
    win.set_title(script)
    win.connect("destroy", lambda *a: gtk.main_quit())
    i = gtk.Image()
    i.set_from_stock(gtk.STOCK_MISSING_IMAGE, gtk.ICON_SIZE_LARGE_TOOLBAR)
    dd = DropdownPanelButton(i)
    button1 = gtk.Button("Hello World!")
    hsv = gtk.HSV()
    button2 = gtk.Button("Test test!")
    button1.connect("clicked", lambda *a: dd.panel_hide())
    button2.connect("clicked", lambda *a: dd.panel_hide())
    def _on_hsv_changed(hsv):
        if hsv.is_adjusting():
            return
        print "HSV:", hsv.get_color()
        #dd.panel_hide()
    hsv.connect("changed", _on_hsv_changed)
    vbox = gtk.VBox()
    vbox.pack_start(button1, False, False)
    vbox.pack_start(hsv, True, True)
    vbox.pack_start(button2, False, False)
    dd.set_panel_widget(vbox)
    win.add(dd)
    win.show_all()
    gtk.main()

