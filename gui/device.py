# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Device specific settings and configuration"""


## Imports

import logging
logger = logging.getLogger(__name__)
import collections
import re

from gettext import gettext as _
from gettext import ngettext
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import Pango

from lib.observable import event
from lib.helpers import escape
import gui.mode


## Device prefs

# The per-device settings are stored in the prefs in a sub-dict whose
# string keys are formed from the device name and enough extra
# information to (hopefully) identify the device uniquely. Names are not
# unique, and IDs vary according to the order in which you plug devices
# in. So for now, our unique strings use a combination of the device's
# name, its source as presented by GDK, and the number of axes.

_PREFS_ROOT = "input.devices"
_PREFS_DEVICE_SUBKEY_FMT = "{name}:{source}:{num_axes}"

## Device usage options

_DEFAULT_USAGE = gui.mode.Behavior.ALL
_USAGE_CONFIG_SUBKEY = "usage"

# This key stores a string representing a permissions mask, to allow us
# to change flag bit order later on.

_USAGE_CONFVAL_INFO = [
    #(Config value, compatible mode behaviour mask, UI string)
    ("all_tasks", gui.mode.Behavior.ALL,
        _("Any task")),
    ("non_paint_tasks", gui.mode.Behavior.NON_PAINTING,
        _("Non-painting tasks")),
    ("none", gui.mode.Behavior.NONE,
        _("Ignore")),
    ]
_USAGE_STRING_BY_CONFVAL = {t[0]: t[2] for t in _USAGE_CONFVAL_INFO}
_USAGE_CONFVAL_BY_FLAGS = {t[1]: t[0] for t in _USAGE_CONFVAL_INFO}
_USAGE_FLAGS_BY_CONFVAL = {t[0]: t[1] for t in _USAGE_CONFVAL_INFO}

#: All valid usage configuration values.
USAGE_CONFIG_VALUES = set([t[0] for t in _USAGE_CONFVAL_INFO])
USAGE_CONFIG_DEFAULT_VALUE = _USAGE_CONFVAL_BY_FLAGS[_DEFAULT_USAGE]

## Class defs


class Monitor (object):
    """Monitors device use & plugging, and manages their configuration

    An instance resides in the main application. It is responsible for
    monitoring known devices, determining their characteristics, and
    storing their settings. Per-device settings are stored in the main
    application preferences.

    """

    def __init__(self, app):
        """Initializes, assigning initial input device uses

        :param app: the owning Application instance.
        :type app: gui.application.Application
        """
        super(Monitor, self).__init__()
        self._app = app
        if app is not None:
            self._prefs = app.preferences
        else:
            self._prefs = {}
        if _PREFS_ROOT not in self._prefs:
            self._prefs[_PREFS_ROOT] = {}

        # Transient device information
        self._device_info = collections.OrderedDict()  # {dev: info_dict}
        self._last_event_device = None
        self._last_pen_device = None

        disp = Gdk.Display.get_default()
        mgr = disp.get_device_manager()
        mgr.connect("device-added", self._device_added_cb)
        mgr.connect("device-removed", self._device_removed_cb)
        self._device_manager = mgr

        for physical_device in mgr.list_devices(Gdk.DeviceType.SLAVE):
            self._update_device_info(physical_device)

    ## Devices list

    def get_device_usage_flags(self, device):
        """Returns the cached usage flags for a given device

        :param Gdk.Device device: physical device
        :returns: Flags declaring allowed mode behaviours
        :rtype: gui.mode.Behavior

        The usage flags are read-only within the application but can be
        changed or updated using `set_device_usage_option()`.

        """
        info = self._device_info.get(device, {})
        return info.get("usage", _DEFAULT_USAGE)

    def get_device_usage_config(self, device):
        """Returns the usage configuration string for a given device

        :param Gdk.Device device: physical device
        :returns: Option string as read from the device's stored prefs
        :rtype: str

        This returns a useful default option string in cases where the
        preference is undefined or unknown.
        """
        device_prefs = self._get_device_prefs(device)
        usage_key = _USAGE_CONFIG_SUBKEY
        return device_prefs.get(usage_key, USAGE_CONFIG_DEFAULT_VALUE)

    def set_device_usage_config(self, device, config_value):
        """Sets the usage for a given device by configuration string"""
        device_prefs = self._get_device_prefs(device)
        if config_value not in _USAGE_FLAGS_BY_CONFVAL:
            config_value = USAGE_CONFIG_DEFAULT_VALUE
        device_prefs[_USAGE_CONFIG_SUBKEY] = config_value
        self._update_device_info(device)

    def _get_device_prefs(self, device):
        """Gets the prefs hash for a device"""
        root = self._prefs.get(_PREFS_ROOT, None)
        if root is None:
            root = {}
            self._prefs[_PREFS_ROOT] = root
        device_key = device_prefs_key(device)
        device_prefs = root.get(device_key, None)
        if device_prefs is None:
            device_prefs = {}
            root[device_key] = device_prefs
        return device_prefs

    def _update_device_info(self, device):
        """Initialize/update a device's runtime info dict"""
        source = device.get_source()
        self._device_info.pop(device, None)
        if source == Gdk.InputSource.KEYBOARD:
            return
        num_axes = device.get_n_axes()
        if num_axes < 2:
            return
        info = {
            "usage": _USAGE_FLAGS_BY_CONFVAL.get(
                self.get_device_usage_config(device),
                _DEFAULT_USAGE,
                ),
            }
        self._device_info[device] = info

    def _device_added_cb(self, mgr, device):
        """Informs that a device has been plugged in"""
        logger.info("Added %r", device.get_name())
        self._update_device_info(device)
        self.devices_updated()

    def _device_removed_cb(self, mgr, device):
        """Informs that a device has been plugged in"""
        logger.info("Removed %r", device.get_name())
        if device not in self._device_info:
            return
        self._device_info.pop(device, None)
        self.devices_updated()

    @event
    def devices_updated(self):
        """Event: the devices list was changed"""

    def get_devices(self):
        """Yields devices and their usage config string, for UI stuff"""
        for device, usage_flags in self._device_info.items():
            usage_config = self.get_device_usage_config(device)
            yield (device, usage_config)

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
        if device not in self._device_info:
            self._update_device_info(device)
        if device not in self._device_info:
            return False
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

        logger.debug(
            "Device change: name=%r source=%s",
            new_device.name, new_device.source.value_name,
            )

        # When editing brush settings, it is often more convenient to use the
        # mouse. Because of this, we don't restore brushsettings when switching
        # to/from the mouse. We act as if the mouse was identical to the last
        # active pen device.

        if (new_device.source == Gdk.InputSource.MOUSE and
                self._last_pen_device):
            new_device = self._last_pen_device
        if new_device.source == Gdk.InputSource.PEN:
            self._last_pen_device = new_device
        if (old_device and old_device.source == Gdk.InputSource.MOUSE and
                self._last_pen_device):
            old_device = self._last_pen_device

        bm = self._app.brushmanager
        if old_device:
            # Clone for saving
            old_brush = bm.clone_selected_brush(name=None)
            bm.store_brush_for_device(old_device.name, old_brush)

        if new_device.source == Gdk.InputSource.MOUSE:
            # Avoid fouling up unrelated devbrushes at stroke end
            self._prefs.pop('devbrush.last_used', None)
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
            self._prefs['devbrush.last_used'] = new_device.name
            bm.select_brush(brush)


class SettingsEditor (Gtk.Grid):
    """Per-device settings editor"""

    ## Class consts

    _USAGE_CONFIG_COL = 0
    _USAGE_STRING_COL = 1

    __gtype_name__ = "MyPaintDeviceSettingsEditor"

    ## Initialization

    def __init__(self, monitor=None):
        """Initialize

        :param Monitor monitor: monitor instance (for testing)

        By default, the central app's `device_monitor` is used to permit
        parameterless construction.
        """
        super(SettingsEditor, self).__init__()
        if monitor is None:
            from gui.application import get_app
            app = get_app()
            monitor = app.device_monitor
        self._monitor = monitor

        self._devices_store = Gtk.ListStore(object)
        self._devices_view = Gtk.TreeView(self._devices_store)

        # Device info column
        col = Gtk.TreeViewColumn(_("Device"))
        col.set_min_width(250)
        col.set_expand(True)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        self._devices_view.append_column(col)
        cell = Gtk.CellRendererText()
        cell.set_property("ellipsize", Pango.EllipsizeMode.MIDDLE)
        col.pack_start(cell, expand=True)
        col.set_cell_data_func(cell, self._device_name_datafunc)

        col = Gtk.TreeViewColumn(_("Axes"))
        col.set_min_width(30)
        col.set_expand(False)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        self._devices_view.append_column(col)
        cell = Gtk.CellRendererText()
        col.pack_start(cell, expand=True)
        col.set_cell_data_func(cell, self._device_axes_datafunc)

        col = Gtk.TreeViewColumn(_("Type"))
        col.set_min_width(30)
        col.set_expand(False)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        self._devices_view.append_column(col)
        cell = Gtk.CellRendererText()
        col.pack_start(cell, expand=True)
        col.set_cell_data_func(cell, self._device_type_datafunc)

        # Usage config value => string store (dropdowns)
        store = Gtk.ListStore(str, str)
        for config, flag, string in _USAGE_CONFVAL_INFO:
            store.append([config, string])
        self._usage_store = store

        # Mode setting & current value column
        col = Gtk.TreeViewColumn(_("Allow..."))
        col.set_min_width(150)
        col.set_resizable(False)
        col.set_expand(True)
        self._devices_view.append_column(col)

        cell = Gtk.CellRendererCombo()
        cell.set_property("model", self._usage_store)
        cell.set_property("text-column", self._USAGE_STRING_COL)
        cell.set_property("mode", Gtk.CellRendererMode.EDITABLE)
        cell.set_property("editable", True)
        cell.set_property("has-entry", False)
        cell.connect("changed", self._usage_config_cell_changed_cb)
        col.pack_start(cell, expand=True)
        col.set_cell_data_func(cell, self._usage_config_string_datafunc)

        # Pretty borders
        view_scroll = Gtk.ScrolledWindow()
        view_scroll.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
        pol = Gtk.PolicyType.AUTOMATIC
        view_scroll.set_policy(pol, pol)
        view_scroll.add(self._devices_view)
        view_scroll.set_hexpand(True)
        view_scroll.set_vexpand(True)
        self.attach(view_scroll, 0, 0, 1, 1)

        self._update_devices_store()
        self._monitor.devices_updated += self._update_devices_store

    ## Display and sort funcs

    def _device_name_datafunc(self, column, cell, model, iter_, *data):
        device = model.get_value(iter_, 0)
        cell.set_property("text", device.get_name())

    def _device_axes_datafunc(self, column, cell, model, iter_, *data):
        device = model.get_value(iter_, 0)
        n_axes = device.get_n_axes()
        cell.set_property("text", "%d" % (n_axes,))

    def _device_type_datafunc(self, column, cell, model, iter_, *data):
        device = model.get_value(iter_, 0)
        source = device.get_source()
        text = {
            Gdk.InputSource.CURSOR: _("Cursor/puck"),
            Gdk.InputSource.ERASER: _("Eraser"),
            Gdk.InputSource.KEYBOARD: _("Keyboard"),
            Gdk.InputSource.MOUSE: _("Mouse"),
            Gdk.InputSource.PEN: _("Pen"),
            Gdk.InputSource.TOUCHPAD: _("Touchpad"),
            Gdk.InputSource.TOUCHSCREEN: _("Touchscreen"),
            }.get(source, source.value_nick)
        cell.set_property("text", text)

    def _usage_config_string_datafunc(self, column, cell, model, iter_, *data):
        device = model.get_value(iter_, 0)
        device_usage_config = self._monitor.get_device_usage_config(device)
        device_usage_string = _USAGE_STRING_BY_CONFVAL.get(device_usage_config)
        cell.set_property("text", device_usage_string)

    ## Updates

    def _usage_config_cell_changed_cb(self, combo, device_path_str, usage_iter, *etc):
        config = self._usage_store.get_value(usage_iter, self._USAGE_CONFIG_COL)
        device_iter = self._devices_store.get_iter(device_path_str)
        device = self._devices_store.get_value(device_iter, 0)
        self._monitor.set_device_usage_config(device, config)
        self._devices_view.columns_autosize()

    def _update_devices_store(self, *_ignored):
        """Repopulates the displayed list"""
        updated_list = list(self._monitor.get_devices())
        updated_list_map = dict(updated_list)
        paths_for_removal = []
        devices_retained = set()
        for row in self._devices_store:
            device, = row
            if device not in updated_list_map:
                paths_for_removal.append(row.path)
                continue
            devices_retained.add(device)
        for device, config in updated_list:
            if device in devices_retained:
                continue
            self._devices_store.append([device])
        for unwanted_row_path in reversed(paths_for_removal):
            unwanted_row_iter = self._devices_store.get_iter(unwanted_row_path)
            self._devices_store.remove(unwanted_row_iter)
        self._devices_view.queue_draw()


## Helper funcs


def device_prefs_key(device):
    """Returns the subkey to use in the app prefs for a device"""
    source = device.get_source()
    name = device.get_name()
    n_axes = device.get_n_axes()
    return u"%s:%s:%d" % (name, source.value_nick, n_axes)


def device_is_eraser(device):
    """Tests whether a device appears to be an eraser"""
    if device is None:
        return False
    if device.get_source() == Gdk.InputSource.ERASER:
        return True
    if re.search(r'\<eraser\>', device.get_name(), re.I):
        return True
    return False


## Testing

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    win = Gtk.Window()
    win.set_title("gui.device.SettingsEditor")
    win.set_default_size(500, 400)
    win.connect("destroy", Gtk.main_quit)
    monitor = Monitor(app=None)
    editor = SettingsEditor(monitor)
    win.add(editor)
    win.show_all()
    Gtk.main()
    print monitor._prefs
