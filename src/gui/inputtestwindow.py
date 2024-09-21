# This file is part of MyPaint.
# Copyright (C) 2010-2018 by the MyPaint Development Team.
# Copyright (C) 2010-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function
import logging
from gettext import gettext as _

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib
from lib.gibindings import Pango

from . import windowing

logger = logging.getLogger(__name__)


class InputTestWindow (windowing.SubWindow):

    def __init__(self):
        from gui.application import get_app
        app = get_app()
        super(InputTestWindow, self).__init__(app)
        self.last_selected_brush = None

        self.set_title(_('Input Device Test'))
        self.set_role('Test')
        self.connect('map', self.map_cb)
        self.connect('unmap', self.unmap_cb)

        self._timer_id = 0

        self.motion_reports = []
        self.motion_event_counter = 0
        self.motion_dtime_sample = []
        self.last_device = None
        self.last_motion_time = 0

        # main container
        vbox = Gtk.VBox()
        self.add(vbox)

        table = Gtk.Table(2, 4)
        vbox.pack_start(table, False, True, 0)

        def add(row, name, value_widget):
            l1 = Gtk.Label(label=name)
            l1.set_justify(Gtk.Justification.LEFT)
            l1.set_alignment(0.0, 0.5)
            l2 = value_widget
            l2.set_alignment(0.0, 0.5)
            table.attach(l1, 0, 1, row, row+1, Gtk.AttachOptions.FILL, 0, 5, 0)
            table.attach(l2, 1, 2, row, row+1, Gtk.AttachOptions.FILL, 0, 5, 0)

        label = self.pressure_label = Gtk.Label(label=_('(no pressure)'))
        add(0, _('Pressure:'), label)

        label = self.tilt_label = Gtk.Label(label=_('(no tilt)'))
        add(1, _('Tilt:'), label)

        label = self.motion_event_counter_label = Gtk.Label()
        add(2, 'Motion:', label)

        label = self.device_label = Gtk.Label(label=_('(no device)'))
        add(3, _('Device:'), label)

        label = Gtk.Label(label=_('(No Barrel Rotation)'))
        self.barrel_rotation_label = label
        add(4, _('Barrel Rotation:'), label)

        vbox.pack_start(Gtk.HSeparator(), False, False, 0)

        tv = self.tv = Gtk.TextView()
        tv.set_editable(False)
        tv.modify_font(Pango.FontDescription("Monospace"))
        tv.set_cursor_visible(False)
        vbox.pack_start(tv, True, True, 0)
        self.log = []

    def map_cb(self, *junk):
        logger.info('Event statistics enabled.')

        self.app.doc.tdw.connect("event", self.event_cb)
        self.app.drawWindow.connect("event", self.event_cb)

        self._timer_id = GLib.timeout_add(
            1000, self.second_timer_cb, priority=GLib.PRIORITY_HIGH)

    def unmap_cb(self, *junk):
        GLib.source_remove(self._timer_id)

        self.app.doc.tdw.disconnect_by_func(self.event_cb)
        self.app.drawWindow.disconnect_by_func(self.event_cb)

        logger.info('Event statistics disabled.')

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
        msg = '% 6s % 15s' % (
            t[-6:],
            event.type.value_name.replace('GDK_', ''),
        )

        if hasattr(event, 'x') and hasattr(event, 'y'):
            msg += ' x=% 7.2f y=% 7.2f' % (event.x, event.y)

        has_pressure, pressure = event.get_axis(Gdk.AxisUse.PRESSURE)
        if has_pressure:
            self.pressure_label.set_text('%4.4f' % pressure)
            msg += ' pressure=% 4.4f' % pressure

        if hasattr(event, 'state'):
            msg += ' state=0x%04x' % event.state

        if hasattr(event, 'button'):
            has_button, button = event.get_button()
            if has_button:
                msg += ' button=%d' % button

        if hasattr(event, 'keyval'):
            msg += ' keyval=%s' % event.keyval

        if hasattr(event, 'hardware_keycode'):
            msg += ' hw_keycode=%s' % event.hardware_keycode

        device = getattr(event, 'device', None)
        if device:
            device = event.get_source_device().get_name()
            if self.last_device != device:
                self.last_device = device
                self.device_label.set_text(device)

        has_xtilt, xtilt = event.get_axis(Gdk.AxisUse.XTILT)
        has_ytilt, ytilt = event.get_axis(Gdk.AxisUse.YTILT)
        if has_xtilt and has_ytilt:
            self.tilt_label.set_text('%+4.4f / %+4.4f' % (xtilt, ytilt))

        has_barrel_rotation, barrel_rotation = event.get_axis(Gdk.AxisUse.WHEEL)
        if has_barrel_rotation:
            self.barrel_rotation_label.set_text('%+4.4f' % (barrel_rotation))

        if widget is not self.app.doc.tdw:
            if widget is self.app.drawWindow:
                msg += ' [drawWindow]'
            else:
                msg += ' [%r]' % widget
        return msg

    def report(self, msg):
        logger.info(msg)
        self.log.append(msg)
        self.log = self.log[-28:]
        GLib.idle_add(
            lambda: self.tv.get_buffer().set_text('\n'.join(self.log))
        )

    def event_cb(self, widget, event):
        if event.type == Gdk.EventType.EXPOSE:
            return False
        msg = self.event2str(widget, event)
        motion_reports_limit = 5
        if event.type == Gdk.EventType.MOTION_NOTIFY:
            if widget is self.app.doc.tdw:
                # statistics
                dt = event.time - self.last_motion_time
                self.motion_event_counter += 1
                self.motion_dtime_sample.append(dt)
                self.motion_dtime_sample = self.motion_dtime_sample[-10:]
                self.last_motion_time = event.time
            # report suppression
            if len(self.motion_reports) < motion_reports_limit:
                self.report(msg)  # report first few motion event immediately
            self.motion_reports.append(msg)
        else:
            unreported = self.motion_reports[motion_reports_limit:]
            if unreported:
                last_report = unreported.pop()
                if unreported:
                    self.report('...      MOTION_NOTIFY %d events suppressed'
                                % len(unreported))
                self.report(last_report)
            self.motion_reports = []
            self.report(msg)
        return False
