# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2014-2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Device specific settings and configuration"""


## Imports

from __future__ import division, print_function
import logging
import collections
import re

from lib.gettext import C_
from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import Pango

from lib.observable import event
import gui.application
import gui.mode

logger = logging.getLogger(__name__)

## Device prefs

# The per-device settings are stored in the prefs in a sub-dict whose
# string keys are formed from the device name and enough extra
# information to (hopefully) identify the device uniquely. Names are not
# unique, and IDs vary according to the order in which you plug devices
# in. So for now, our unique strings use a combination of the device's
# name, its source as presented by GDK, and the number of axes.

_PREFS_ROOT = "input.devices"
_PREFS_DEVICE_SUBKEY_FMT = "{name}:{source}:{num_axes}"


## Device type strings

_DEVICE_TYPE_STRING = {
    Gdk.InputSource.CURSOR: C_(
        "prefs: device's type label",
        "Cursor/puck",
    ),
    Gdk.InputSource.ERASER: C_(
        "prefs: device's type label",
        "Eraser",
    ),
    Gdk.InputSource.KEYBOARD: C_(
        "prefs: device's type label",
        "Keyboard",
    ),
    Gdk.InputSource.MOUSE: C_(
        "prefs: device's type label",
        "Mouse",
    ),
    Gdk.InputSource.PEN: C_(
        "prefs: device's type label",
        "Pen",
    ),
    Gdk.InputSource.TOUCHPAD: C_(
        "prefs: device's type label",
        "Touchpad",
    ),
    Gdk.InputSource.TOUCHSCREEN: C_(
        "prefs: device's type label",
        "Touchscreen",
    ),
}


## Settings consts and classes


class AllowedUsage:
    """Consts describing how a device may interact with the canvas"""

    ANY = "any"  #: Device can be used for any tasks.
    NOPAINT = "nopaint"  #: No direct painting, but can manipulate objects.
    NAVONLY = "navonly"  #: Device can only be used for navigation.
    IGNORED = "ignored"  #: Device cannot interact with the canvas at all.

    VALUES = (ANY, IGNORED, NOPAINT, NAVONLY)
    DISPLAY_STRING = {
        IGNORED: C_(
            "device settings: allowed usage",
            u"Ignore",
        ),
        ANY: C_(
            "device settings: allowed usage",
            u"Any Task",
        ),
        NOPAINT: C_(
            "device settings: allowed usage",
            u"Non-painting tasks",
        ),
        NAVONLY: C_(
            "device settings: allowed usage",
            u"Navigation only",
        ),
    }
    BEHAVIOR_MASK = {
        ANY: gui.mode.Behavior.ALL,
        IGNORED: gui.mode.Behavior.NONE,
        NOPAINT: gui.mode.Behavior.NON_PAINTING,
        NAVONLY: gui.mode.Behavior.CHANGE_VIEW,
    }


class ScrollAction:
    """Consts describing how a device's scroll events should be used.

    The user can assign one of these values to a device to configure
    whether they'd prefer panning or scrolling for unmodified scroll
    events. This setting can be queried via the device monitor.

    """

    ZOOM = "zoom"  #: Alter the canvas scaling
    PAN = "pan"   #: Pan across the canvas

    VALUES = (ZOOM, PAN)
    DISPLAY_STRING = {
        ZOOM: C_("device settings: unmodified scroll action", u"Zoom"),
        PAN: C_("device settings: unmodified scroll action", u"Pan"),
    }


class Settings (object):
    """A device's settings"""

    DEFAULT_USAGE = AllowedUsage.VALUES[0]
    DEFAULT_SCROLL = ScrollAction.VALUES[0]

    def __init__(self, prefs, usage=DEFAULT_USAGE, scroll=DEFAULT_SCROLL):
        super(Settings, self).__init__()
        self._usage = self.DEFAULT_USAGE
        self._update_usage_mask()
        self._scroll = self.DEFAULT_SCROLL
        self._prefs = prefs
        self._load_from_prefs()

    @property
    def usage(self):
        return self._usage

    @usage.setter
    def usage(self, value):
        if value not in AllowedUsage.VALUES:
            raise ValueError("Unrecognized usage value")
        self._usage = value
        self._update_usage_mask()
        self._save_to_prefs()

    @property
    def usage_mask(self):
        return self._usage_mask

    @property
    def scroll(self):
        return self._scroll

    @scroll.setter
    def scroll(self, value):
        if value not in ScrollAction.VALUES:
            raise ValueError("Unrecognized scroll value")
        self._scroll = value
        self._save_to_prefs()

    def _load_from_prefs(self):
        usage = self._prefs.get("usage", self.DEFAULT_USAGE)
        if usage not in AllowedUsage.VALUES:
            usage = self.DEFAULT_USAGE
        self._usage = usage
        scroll = self._prefs.get("scroll", self.DEFAULT_SCROLL)
        if scroll not in ScrollAction.VALUES:
            scroll = self.DEFAULT_SCROLL
        self._scroll = scroll
        self._update_usage_mask()

    def _save_to_prefs(self):
        self._prefs.update({
            "usage": self._usage,
            "scroll": self._scroll,
        })

    def _update_usage_mask(self):
        self._usage_mask = AllowedUsage.BEHAVIOR_MASK[self._usage]


## Main class defs


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
        self._device_settings = collections.OrderedDict()  # {dev: settings}
        self._last_event_device = None
        self._last_pen_device = None

        disp = Gdk.Display.get_default()
        mgr = disp.get_device_manager()
        mgr.connect("device-added", self._device_added_cb)
        mgr.connect("device-removed", self._device_removed_cb)
        self._device_manager = mgr

        for physical_device in mgr.list_devices(Gdk.DeviceType.SLAVE):
            self._init_device_settings(physical_device)

    ## Devices list

    def get_device_settings(self, device):
        """Gets the settings for a device

        :param Gdk.Device device: a physical ("slave") device
        :returns: A settings object which can be manipulated, or None
        :rtype: Settings

        Changes to the returned object made via its API are saved to the
        user preferences immediately.

        If the device is a keyboard, or is otherwise unsuitable as a
        pointing device, None is returned instead. The caller needs to
        check this case.

        """
        return (self._device_settings.get(device)
                or self._init_device_settings(device))

    def _init_device_settings(self, device):
        """Ensures that the device settings are loaded for a device"""
        source = device.get_source()
        if source == Gdk.InputSource.KEYBOARD:
            return
        num_axes = device.get_n_axes()
        if num_axes < 2:
            return
        settings = self._device_settings.get(device)
        if not settings:
            try:
                vendor_id = device.get_vendor_id()
                product_id = device.get_product_id()
            except AttributeError:
                # New in GDK 3.16
                vendor_id = "?"
                product_id = "?"
            logger.info(
                "New device %r"
                " (%s, axes:%d, class=%s, vendor=%r, product=%r)",
                device.get_name(),
                source.value_name,
                num_axes,
                device.__class__.__name__,
                vendor_id,
                product_id,
            )
            dev_prefs_key = _device_prefs_key(device)
            dev_prefs = self._prefs[_PREFS_ROOT].setdefault(dev_prefs_key, {})
            settings = Settings(dev_prefs)
            self._device_settings[device] = settings
            self.devices_updated()
        assert settings is not None
        return settings

    def _device_added_cb(self, mgr, device):
        """Informs that a device has been plugged in"""
        logger.debug("device-added %r", device.get_name())
        self._init_device_settings(device)

    def _device_removed_cb(self, mgr, device):
        """Informs that a device has been unplugged"""
        logger.debug("device-removed %r", device.get_name())
        self._device_settings.pop(device, None)
        self.devices_updated()

    @event
    def devices_updated(self):
        """Event: the devices list was changed"""

    def get_devices(self):
        """Yields devices and their settings, for UI stuff

        :rtype: iterator
        :returns: ultimately a sequence of (Gdk.Device, Settings) pairs

        """
        for device, settings in self._device_settings.items():
            yield (device, settings)

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
        if not self.get_device_settings(device):
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
    _SCROLL_CONFIG_COL = 0
    _SCROLL_STRING_COL = 1

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
            app = gui.application.get_app()
            monitor = app.device_monitor
        self._monitor = monitor

        self._devices_store = Gtk.ListStore(object)
        self._devices_view = Gtk.TreeView(model=self._devices_store)

        col = Gtk.TreeViewColumn(C_(
            "prefs: devices table: column header",
            # TRANSLATORS: Column's data is the device's name
            "Device",
        ))
        col.set_min_width(200)
        col.set_expand(True)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        self._devices_view.append_column(col)
        cell = Gtk.CellRendererText()
        cell.set_property("ellipsize", Pango.EllipsizeMode.MIDDLE)
        col.pack_start(cell, True)
        col.set_cell_data_func(cell, self._device_name_datafunc)

        col = Gtk.TreeViewColumn(C_(
            "prefs: devices table: column header",
            # TRANSLATORS: Column's data is the number of axes (an integer)
            "Axes",
        ))
        col.set_min_width(30)
        col.set_resizable(True)
        col.set_expand(False)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        self._devices_view.append_column(col)
        cell = Gtk.CellRendererText()
        col.pack_start(cell, True)
        col.set_cell_data_func(cell, self._device_axes_datafunc)

        col = Gtk.TreeViewColumn(C_(
            "prefs: devices table: column header",
            # TRANSLATORS: Column shows type labels ("Touchscreen", "Pen" etc.)
            "Type",
        ))
        col.set_min_width(120)
        col.set_resizable(True)
        col.set_expand(False)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        self._devices_view.append_column(col)
        cell = Gtk.CellRendererText()
        cell.set_property("ellipsize", Pango.EllipsizeMode.END)
        col.pack_start(cell, True)
        col.set_cell_data_func(cell, self._device_type_datafunc)

        # Usage config value => string store (dropdowns)
        store = Gtk.ListStore(str, str)
        for conf_val in AllowedUsage.VALUES:
            string = AllowedUsage.DISPLAY_STRING[conf_val]
            store.append([conf_val, string])
        self._usage_store = store

        col = Gtk.TreeViewColumn(C_(
            "prefs: devices table: column header",
            # TRANSLATORS: Column's data is a dropdown allowing the allowed
            # TRANSLATORS: tasks for the row's device to be configured.
            u"Use for…",
        ))
        col.set_min_width(100)
        col.set_resizable(True)
        col.set_expand(False)
        self._devices_view.append_column(col)

        cell = Gtk.CellRendererCombo()
        cell.set_property("model", self._usage_store)
        cell.set_property("text-column", self._USAGE_STRING_COL)
        cell.set_property("mode", Gtk.CellRendererMode.EDITABLE)
        cell.set_property("editable", True)
        cell.set_property("has-entry", False)
        cell.set_property("ellipsize", Pango.EllipsizeMode.END)
        cell.connect("changed", self._usage_cell_changed_cb)
        col.pack_start(cell, True)
        col.set_cell_data_func(cell, self._device_usage_datafunc)

        # Scroll action config value => string store (dropdowns)
        store = Gtk.ListStore(str, str)
        for conf_val in ScrollAction.VALUES:
            string = ScrollAction.DISPLAY_STRING[conf_val]
            store.append([conf_val, string])
        self._scroll_store = store

        col = Gtk.TreeViewColumn(C_(
            "prefs: devices table: column header",
            # TRANSLATORS: Column's data is a dropdown for how the device's
            # TRANSLATORS: scroll wheel or scroll-gesture events are to be
            # TRANSLATORS: interpreted normally.
            u"Scroll…",
        ))
        col.set_min_width(100)
        col.set_resizable(True)
        col.set_expand(False)
        self._devices_view.append_column(col)

        cell = Gtk.CellRendererCombo()
        cell.set_property("model", self._scroll_store)
        cell.set_property("text-column", self._USAGE_STRING_COL)
        cell.set_property("mode", Gtk.CellRendererMode.EDITABLE)
        cell.set_property("editable", True)
        cell.set_property("has-entry", False)
        cell.set_property("ellipsize", Pango.EllipsizeMode.END)
        cell.connect("changed", self._scroll_cell_changed_cb)
        col.pack_start(cell, True)
        col.set_cell_data_func(cell, self._device_scroll_datafunc)

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
        text = _DEVICE_TYPE_STRING.get(source, source.value_nick)
        cell.set_property("text", text)

    def _device_usage_datafunc(self, column, cell, model, iter_, *data):
        device = model.get_value(iter_, 0)
        settings = self._monitor.get_device_settings(device)
        if not settings:
            return
        text = AllowedUsage.DISPLAY_STRING[settings.usage]
        cell.set_property("text", text)

    def _device_scroll_datafunc(self, column, cell, model, iter_, *data):
        device = model.get_value(iter_, 0)
        settings = self._monitor.get_device_settings(device)
        if not settings:
            return
        text = ScrollAction.DISPLAY_STRING[settings.scroll]
        cell.set_property("text", text)

    ## Updates

    def _usage_cell_changed_cb(self, combo, device_path_str,
                               usage_iter, *etc):
        config = self._usage_store.get_value(
            usage_iter,
            self._USAGE_CONFIG_COL,
        )
        device_iter = self._devices_store.get_iter(device_path_str)
        device = self._devices_store.get_value(device_iter, 0)
        settings = self._monitor.get_device_settings(device)
        if not settings:
            return
        settings.usage = config
        self._devices_view.columns_autosize()

    def _scroll_cell_changed_cb(self, conf_combo, device_path_str,
                                conf_iter, *etc):
        conf_store = self._scroll_store
        conf_col = self._SCROLL_CONFIG_COL
        conf_value = conf_store.get_value(conf_iter, conf_col)
        device_store = self._devices_store
        device_iter = device_store.get_iter(device_path_str)
        device = device_store.get_value(device_iter, 0)
        settings = self._monitor.get_device_settings(device)
        if not settings:
            return
        settings.scroll = conf_value
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


def _device_prefs_key(device):
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


def _test():
    """Interactive UI testing for SettingsEditor and Monitor"""
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
    print(monitor._prefs)


if __name__ == '__main__':
    _test()
