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
import widgets


class DropdownPanelButton (gtk.ToggleButton):
    """Button which drops down a panel with (almost) arbitrary widgets.

    The panel iself is a specialised `gtk.Window` bound to its button.
    """

    __gtype_name__ = "DropdownPanelButton"
    __gproperties__ = {
        'panel-widget': (gtk.Widget,
                         "dropdown panel's widget",
                         'The widget to display in the dropdown panel',
                         gobject.PARAM_READWRITE),
        }

    def __init__(self, label_widget=None):
        """Construct with a button label widget.
        """
        gtk.ToggleButton.__init__(self)
        if label_widget is not None:
            self.add_label_widget_with_arrow(label_widget)
        self._panel = DropdownPanel(self)
        self.connect("button-release-event", self._button_release_cb)
        self._panel.connect("hide", lambda *a: self.set_active(False))
        self._panel.connect("show", lambda *a: self.set_active(True))
        # Common L&F for buttons. Should probably leave to Glade.
        self.set_name(widgets.BORDERLESS_BUTTON_NAME)
        self.set_relief(gtk.RELIEF_NONE)
        self.set_can_default(False)
        self.set_can_focus(False)


    def do_set_property(self, prop, value):
        if prop.name == 'panel-widget':
            self._panel.content_widget = value
        else:
            raise AttributeError, 'unknown property %s' % prop.name


    def do_get_property(self, prop):
        if prop.name == 'panel-widget':
            return self._panel.content_widget
        else:
            raise AttributeError, 'unknown property %s' % prop.name


    def add_label_widget_with_arrow(self, label_widget):
        arrow = gtk.Arrow(gtk.ARROW_DOWN, gtk.SHADOW_IN)
        old_child = self.get_child()
        hbox = gtk.HBox()
        label_alignment = gtk.Alignment(0.0, 0.0, 1.0, 1.0)
        label_alignment.set_padding(0, 0, widgets.SPACING_TIGHT, 0)
        label_alignment.add(label_widget)
        hbox.pack_start(label_alignment, True, True)
        hbox.pack_start(arrow, False, False)
        self.add(hbox)


    def panel_hide(self, immediate=True, release=True, leave=True):
        """Hides the panel.
        
        Call this after the user makes a final selection.
        """
        if immediate:
            self._panel.popdown()
        else:
            self._panel.hide_on_leave = leave
            self._panel.hide_on_release = release


    def _button_release_cb(self, widget, event):
        if not self.get_active():
            # feels nicer to do this in an idle handler
            gobject.idle_add(self._panel.popup, event.time)
        return True


class DropdownPanel (gtk.Window):

    def __init__(self, panel_button):
        gtk.Window.__init__(self, type=gtk.WINDOW_POPUP)
        self.content_widget = None
        self.set_modal(True)
        self.set_focus_on_map(False)
        self.set_can_focus(False)
        self._panel_button = panel_button
        self._grabbed = False
        self._corrected_pos = False
        self.connect("realize", self._realize_cb)
        self.connect("map-event", self._map_event_cb)
        self.connect("button-release-event", self._button_release_event_cb)
        self.connect("hide", self._hide_cb)
        self.connect("configure-event", self._configure_cb)
        self.connect("grab-broken-event", self._grab_broken_event_cb)
        self.connect("leave-notify-event", self._leave_notify_cb)
        self.connect("key-press-event", self._key_press_cb)
        self.hide_on_leave = False
        self.hide_on_release = False


    def _is_outside(self, x, y):
        gdk_window = self.get_window()
        alloc = self.get_allocation()
        wx, wy = gdk_window.get_origin()
        ww, wh = alloc.width, alloc.height
        return x < wx or y < wy or x > wx+ww or y > wy+wh


    def _leave_notify_cb(self, widget, event):
        if not self.hide_on_leave:
            return
        if event.mode != gdk.CROSSING_NORMAL:
            return
        if self._is_outside(event.x_root, event.y_root):
            self.popdown()
            return


    def _button_release_event_cb(self, widget, event):
        # Dismiss if we've been asked to by an appropriate call to
        # DropdownPanelButton.panel_hide(). Doing this in the release handler
        # helps avoid accidental dabs with certain widgets.
        if self.hide_on_release:
            self.popdown()
            return True
        # Dismiss too if the user clicks outside the panel (during a grab).
        if not self.get_window():
            return
        if self._is_outside(event.x_root, event.y_root):
            self.popdown()
            return True


    def _key_press_cb(self, widget, event):
        if not self._grabbed:
            return
        if gdk.keyval_name(event.keyval).upper() == "ESCAPE":
            self.popdown()
            return True


    def establish_grab(self, t=0):
        # Grab, permitting normal interaction with the app (i.e. with the widgets
        # in the panel).
        mask = gdk.BUTTON_PRESS_MASK
        gdk_window = self.get_window()
        grab_result = gdk.pointer_grab(gdk_window, True, mask, None, None, t)
        if grab_result != gdk.GRAB_SUCCESS:
            print "pointer grab failed:", grab_result
            return False

        # Keyboard grab too, to prevent workspace switching.
        grab_result = gdk.keyboard_grab(gdk_window, False, t)
        if grab_result != gdk.GRAB_SUCCESS:
            print "keyboard grab failed:", grab_result
            gdk.pointer_ungrab(gdk.CURRENT_TIME)
            return False

        # But limit events to just the panel for neatness and a hint of modality.
        self.grab_add()
        self._grabbed = True
        return True


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

    def _grab_broken_event_cb(self, widget, event):
        rival = event.grab_window
        if rival is not None:
            print "grab broken by", rival
        self.hide_on_leave = True
        self._grabbed = False


    def popup(self, t=0):
        parent_window = self._panel_button.get_toplevel()
        self.set_transient_for(parent_window)
        self.hide_on_leave = False
        self.hide_on_release = False
        child = self.get_child()
        if child is None or child is not self.content_widget:
            if child is not None:
                self.remove(child)
            p = self._panel_button
            child = self.content_widget
            while p is not None:
                if "Glade" in p.get_name():
                    label = "In Glade: cannot show this widget."
                    if self.content_widget is None:
                        label += "\n\nPanel Widget is not set.\n"
                        label += "Please set it to a top-level widget."
                    child = gtk.Label(label)
                    break
                p = p.get_parent()
            self.add(child)
        self.show_all()
        def deferred_grab():
            if not self.establish_grab(t):
                self.popdown()
        gobject.idle_add(deferred_grab)


    def popdown(self):
        gobject.idle_add(self.hide)


    def _hide_cb(self, widget):
        if self._grabbed:
            gdk.keyboard_ungrab(gdk.CURRENT_TIME)
            gdk.pointer_ungrab(gdk.CURRENT_TIME)
            self._grabbed = False
        self.grab_remove()


    # Positioning and initial geometry

    def _realize_cb(self, widget):
        gdk_window = self.get_window()
        gdk_window.set_type_hint(gdk.WINDOW_TYPE_HINT_DROPDOWN_MENU)
        x, y = self._get_panel_pos()
        self.parse_geometry("+%d+%d" % (x, y))

    def _map_event_cb(self, widget, event):
        x, y = self._get_panel_pos()
        gobject.idle_add(self.move, x, y)
        self._corrected_pos = False

    def _get_panel_pos(self):
        button = self._panel_button
        button_win = button.get_window()
        x, y = button_win.get_origin()
        button_alloc = button.get_allocation()
        x += button_alloc.x
        y += button_alloc.y
        y += button_alloc.height
        return int(x), int(y)

    def _configure_cb(self, widget, event):
        # Constrain window to fit on its current monitor, if possible.
        screen = event.get_screen()
        mon = screen.get_monitor_at_point(event.x, event.y)
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
            if not self._corrected_pos:
                gobject.idle_add(self.move, x, y)
                self._corrected_pos = True



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
    dd.set_property('panel-widget', vbox)
    win.add(dd)
    win.show_all()
    gtk.main()
