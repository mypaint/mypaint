# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Device specific stuff"""


## Imports

import logging
logger = logging.getLogger(__name__)
import re

from gettext import gettext as _
from gi.repository import Gtk
from gi.repository import Gdk

from lib.observable import event


## Package consts

_PREFS_ROOT = "input.devices"

_INPUT_MODE_PREFS_SUBKEY = "mode"
_INPUT_MODE_BY_PREFS_VALUE = {
    "screen": Gdk.InputMode.SCREEN,
    "window": Gdk.InputMode.WINDOW,
}

# The default input mode string.
_INPUT_MODE_DEFAULT = None

#: UI string definitions for the device modes saved into prefs.
#: Simple strings only, for that reason.
INPUT_MODE_STRINGS = (
    ("screen", _("Screen (recommended)")),
    ("window", _("Window")),
    )


class Monitor (object):
    """Monitors device use & plugging, and configures devices as needed

    An instance resides in the main application,
    and is responsible for monitoring which device is current.
    It also configures devices when they are plugged in, at startup,
    or when they are first used.

    Per-device settings are stored in the main application preferences.
    """

    def __init__(self, app):
        """Initializes, assigning initial input device modes

        :param app: the owning Application singleton.
        """
        super(Monitor, self).__init__()
        self._app = app

        if not _PREFS_ROOT in app.preferences:
            app.preferences[_PREFS_ROOT] = {}
        self._prefs = app.preferences[_PREFS_ROOT]

        self._devices = {}  # {device => set(features...)}
        self._last_event_device = None
        self._last_pen_device = None

        disp = Gdk.Display.get_default()
        mgr = disp.get_device_manager()
        mgr.connect("device-added", self._device_added_cb)
        mgr.connect("device-removed", self._device_removed_cb)
        self._device_manager = mgr

        logger.info("Looking for GDK devices with pressure...")
        for physical_device in mgr.list_devices(Gdk.DeviceType.SLAVE):
            self._init_device(physical_device)
        logger.info("Initial scan done.")


    ## Devices list

    def get_input_mode(self, device):
        """Returns the explicit mode setting for a device, via prefs"""
        dev_prefs = self._get_device_prefs(device)
        modestr = dev_prefs.get(_INPUT_MODE_PREFS_SUBKEY)
        return _INPUT_MODE_BY_PREFS_VALUE.get(modestr, _INPUT_MODE_DEFAULT)

    def _get_device_prefs(self, device):
        """Gets the prefs hash for a device"""
        root = self._prefs.get(_PREFS_ROOT, None)
        if root is None:
            root = {}
            self._prefs[_PREFS_ROOT] = root
        device_subkey = device_prefs_key(device)
        device_prefs = root.get(device_subkey, None)
        if device_prefs is None:
            device_prefs = {}
            self._prefs[device_subkey] = device_prefs
        return device_prefs

    def _init_device(self, device):
        """Initializes a device for use in MyPaint"""
        # Already noted?
        if device in self._devices:
            return
        # Relevant for mode setting and use?
        is_physical = (device.get_property("type") == Gdk.DeviceType.SLAVE)
        if not is_physical:
            return
        source = device.get_source()
        if source == Gdk.InputSource.KEYBOARD:
            return
        # Record and configure
        name = device.get_name()
        n_axes = device.get_n_axes()
        has_pressure_axis = False
        for i in xrange(n_axes):
            use = device.get_axis_use(i)
            if use == Gdk.AxisUse.PRESSURE:
                has_pressure_axis = True
                break
        logger.info("Adding device %r (source=%r, axes=%r, pressure=%r)",
                    name, source.value_name, n_axes, has_pressure_axis)
        if has_pressure_axis:
            mode = self.get_input_mode(device)
            if mode is not None:
                if device.get_mode() != mode:
                    logger.info('Setting %s for %r', mode.value_name,
                                device.get_name())
                    device.set_mode(mode)
        # Internal record of info about the device, not saved to prefs
        self._devices[device] = {
            "pressure": has_pressure_axis,
            }

    def _device_added_cb(self, mgr, device):
        """Informs that a device has been plugged in"""
        self._init_device(device)

    def _device_removed_cb(self, mgr, device):
        """Informs that a device has been plugged in"""
        if device not in self._devices:
            return
        name = device.get_name()
        source = device.get_source()
        n_axes = device.get_n_axes()
        has_pressure_axis = False
        logger.info("Removing device %r (source=%r, axes=%r, pressure=%r)",
                    name, source.value_name, n_axes, has_pressure_axis)
        self._devices.pop(device)


    ## Current device

    @event
    def current_device_changed(self, old_device, new_device):
        """Event: the current device has changed

        :param Gdk.Device old_device: Previous device used
        :param Gdk.Device new_device: New device used
        """

    def device_used(self, device):
        """Informs about a device being used, for use by controllers

        :param Gdk.Device device: the device being used
        :returns: whether the device changed

        If the device has changed, this method then notifies interested
        parties via the device_changed observable @event.

        This method returns True if the device was the same as the previous
        device, and False if it has changed.
        """
        if device not in self._devices:
            self._init_device(device)
        if device == self._last_event_device:
            return True
        self.current_device_changed(self._last_event_device, device)
        old_device = self._last_event_device
        new_device = device
        self._last_event_device = device

        # small problem with this code: it doesn't work well with brushes that
        # have (eraser not in [1.0, 0.0])

        new_device.name = new_device.props.name
        new_device.source = new_device.props.input_source

        logger.debug('Device change: name=%r source=%s',
                     new_device.name, new_device.source.value_name)

        # When editing brush settings, it is often more convenient to use the
        # mouse. Because of this, we don't restore brushsettings when switching
        # to/from the mouse. We act as if the mouse was identical to the last
        # active pen device.

        if ( new_device.source == Gdk.InputSource.MOUSE and
             self._last_pen_device ):
            new_device = self._last_pen_device
        if new_device.source == Gdk.InputSource.PEN:
            self._last_pen_device = new_device
        if ( old_device and old_device.source == Gdk.InputSource.MOUSE and
             self._last_pen_device ):
            old_device = self._last_pen_device

        bm = self._app.brushmanager
        if old_device:
            # Clone for saving
            old_brush = bm.clone_selected_brush(name=None)
            bm.store_brush_for_device(old_device.name, old_brush)

        if new_device.source == Gdk.InputSource.MOUSE:
            # Avoid fouling up unrelated devbrushes at stroke end
            self._app.preferences.pop('devbrush.last_used', None)
        else:
            # Select the brush and update the UI.
            # Use a sane default if there's nothing associated
            # with the device yet.
            brush = bm.fetch_brush_for_device(new_device.name)
            if brush is None:
                if device_is_eraser(new_device):
                    brush = bm.get_default_eraser()
                else:
                    brush = bm.get_default_brush()
            self._app.preferences['devbrush.last_used'] = new_device.name
            bm.select_brush(brush)


def device_prefs_key(device):
    """Returns the subkey to use in the app prefs for a device"""
    source = device.get_source()
    name = device.get_name()
    n_axes = device.get_n_axes()
    return u"%s:%s:%d" % (name, source.value_name, n_axes)


def device_is_eraser(device):
    """Tests whether a device appears to be an eraser"""
    if device is None:
        return False
    if device.get_source() == Gdk.InputSource.ERASER:
        return True
    if re.search(r'\<eraser\>', device.get_name(), re.I):
        return True
    return False

