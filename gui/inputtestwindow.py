# This file is part of MyPaint.
# Copyright (C) 2010 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gettext import gettext as _
import gtk
from gtk import gdk
import gobject
import pango

import gtk2compat
import windowing


class Window(windowing.SubWindow):
    def __init__(self, app):
        windowing.SubWindow.__init__(self, app)
        self.last_selected_brush = None

        self.set_title(_('Input Device Test'))
        self.set_role('Test')
        self.connect('map-event', self.map_cb)

        self.initialized = False
        self.motion_reports = []
        self.motion_event_counter = 0
        self.motion_dtime_sample = []
        self.last_device = None
        self.last_motion_time = 0

        #main container
        vbox = gtk.VBox()
        self.add(vbox)

        table = gtk.Table(2, 4)
        vbox.pack_start(table, expand=False, fill=True)

        def add(row, name, value_widget):
            l1 = gtk.Label(name)
            l1.set_justify(gtk.JUSTIFY_LEFT)
            l1.set_alignment(0.0, 0.5)
            l2 = value_widget
            l2.set_alignment(0.0, 0.5)
            table.attach(l1, 0, 1, row, row+1, gtk.FILL, 0, 5, 0)
            table.attach(l2, 1, 2, row, row+1, gtk.FILL, 0, 5, 0)

        l = self.pressure_label = gtk.Label(_('(no pressure)'))
        add(0, _('Pressure:'), l)

        l = self.tilt_label = gtk.Label(_('(no tilt)'))
        add(1, _('Tilt:'), l)

        l = self.motion_event_counter_label = gtk.Label()
        add(2, 'Motion:', l)

        l = self.device_label = gtk.Label(_('(no device)'))
        add(3, _('Device:'), l)

        vbox.pack_start(gtk.HSeparator(), expand=False, fill=False)

        tv = self.tv = gtk.TextView()
        tv.set_editable(False)
        tv.modify_font(pango.FontDescription("Monospace"))
        tv.set_cursor_visible(False)
        vbox.pack_start(tv, expand=True, fill=True)
        self.log = []

    def map_cb(self, *junk):
        if self.initialized:
            return
        print 'Event statistics enabled.'
        self.initialized = True
        self.app.doc.tdw.connect("event", self.event_cb)
        self.app.drawWindow.connect("event", self.event_cb)
        gobject.timeout_add(1000, self.second_timer_cb, priority=gobject.PRIORITY_HIGH)

    def second_timer_cb(self):
        s = str(self.motion_event_counter)
        s += ' events, timestamp spacing: '
        if self.motion_dtime_sample:
            for dtime in self.motion_dtime_sample:
                s += '%d, ' % dtime
            s += '...'
        else:
            s += '-'
        self.motion_event_counter_label.set_text(s)
        self.motion_event_counter = 0
        self.motion_dtime_sample = []
        return True

    def event2str(self, widget, event):
        t = str(getattr(event, 'time', '-'))
        msg = '% 6s % 15s' % (t[-6:], event.type.value_name.replace('GDK_', ''))

        if hasattr(event, 'x') and hasattr(event, 'y'):
            msg += ' x=% 7.2f y=% 7.2f' % (event.x, event.y)

        if gtk2compat.USE_GTK3:
            axis_found, pressure = event.get_axis(gdk.AXIS_PRESSURE)
            if not axis_found:
                pressure = None
        else: # PyGTK
            pressure = event.get_axis(gdk.AXIS_PRESSURE)
        if pressure is not None:
            self.pressure_label.set_text('%4.4f' % pressure)
            msg += ' pressure=% 4.4f' % pressure

        if hasattr(event, 'state'):
            msg += ' state=0x%04x' % event.state

        if hasattr(event, 'button'):
            if gtk2compat.USE_GTK3:
                has_button, button = event.get_button()
                if not has_button:
                    button = None
            else:
                button = event.button
                has_button = True
            if has_button:
                msg += ' button=%d' % button

        if hasattr(event, 'keyval'):
            msg += ' keyval=%s' % event.keyval

        if hasattr(event, 'hardware_keycode'):
            msg += ' hw_keycode=%s' % event.hardware_keycode

        device = getattr(event, 'device', None)
        if device:
            if gtk2compat.USE_GTK3:
                device = event.get_source_device()
                device = device.get_name()
            else: # PyGTK
                device = device.name
            if self.last_device != device:
                self.last_device = device
                self.device_label.set_text(device)

        if gtk2compat.USE_GTK3:
            has_xtilt, xtilt = event.get_axis(gdk.AXIS_XTILT)
            has_ytilt, ytilt = event.get_axis(gdk.AXIS_YTILT)
            have_tilts = has_xtilt and has_ytilt
        else: #PyGTK
            xtilt = event.get_axis(gdk.AXIS_XTILT)
            ytilt = event.get_axis(gdk.AXIS_YTILT)
            have_tilts = xtilt is not None and ytilt is not None
        if have_tilts:
            self.tilt_label.set_text('%+4.4f / %+4.4f' % (xtilt, ytilt))

        if widget is not self.app.doc.tdw:
            if widget is self.app.drawWindow:
                msg += ' [drawWindow]'
            else:
                msg += ' [%r]' % widget
        return msg

    def report(self, msg):
        print msg
        self.log.append(msg)
        self.log = self.log[-28:]
        buf = self.tv.get_buffer()
        buf.set_text('\n'.join(self.log))

    def event_cb(self, widget, event):
        if event.type == gdk.EXPOSE:
            return False
        msg = self.event2str(widget, event)
        if event.type == gdk.MOTION_NOTIFY:
            # statistics
            self.motion_event_counter += 1
            self.motion_dtime_sample.append(event.time - self.last_motion_time)
            self.motion_dtime_sample = self.motion_dtime_sample[-10:]
            self.last_motion_time = event.time
            # report suppression
            if not self.motion_reports:
                self.report(msg) # report first motion event immediately
            self.motion_reports.append(msg)
        else:
            if self.motion_reports:
                self.motion_reports.pop(0) # already reported the first motion event
                if self.motion_reports:
                    last_report = self.motion_reports.pop()
                    if self.motion_reports:
                        self.report('...      MOTION_NOTIFY %d events suppressed' % len(self.motion_reports))
                    self.report(last_report)
                self.motion_reports = []
            self.report(msg)
        return False

