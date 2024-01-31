# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Manager+adjuster bases for tweaking a single color via many widgets.
"""

from __future__ import division, print_function

import math
from copy import deepcopy, copy
from warnings import warn
import weakref
import os
import logging
from gettext import gettext as _

from lib.gibindings import GObject
from lib.gibindings import Gdk
from lib.gibindings import Gtk
from lib.gibindings import GdkPixbuf
import cairo

from .util import clamp
from .util import add_distance_fade_stops
from .util import draw_marker_circle
from lib.color import RGBColor, HCYColor
from .bases import CachedBgDrawingArea
from .bases import IconRenderable
from . import uimisc
from lib.palette import Palette
from lib.observable import event
import gui.dialogs
import gui.uicolor
from lib.pycompat import xrange

logger = logging.getLogger(__name__)


## Module constants

PREFS_KEY_CURRENT_COLOR = 'colors.current'
PREFS_KEY_COLOR_HISTORY = 'colors.history'
PREFS_KEY_WHEEL_TYPE = 'colors.wheels.type'
PREFS_PALETTE_DICT_KEY = "colors.palette"
DATAPATH_PALETTES_SUBDIR = 'palettes'
DEFAULT_PALETTE_FILE = 'MyPaint_Default.gpl'

## API deprecation support


class DeprecatedAPIWarning (UserWarning):
    pass


## Class definitions


class ColorManager (GObject.GObject):
    """Manages the data common to several attached `ColorAdjuster`s.

    This data is basically everything that one or more adjusters might want to
    display. It includes a working color (used by the main app as the active
    brush color), a short history of previously-painted-with colors, tables of
    angular distortions for color wheels etc.

    """
    ## GObject integration (type name)

    __gtype_name__ = "ColorManager"  #: GObject integration

    ## Behavioural constants

    _DEFAULT_HIST = ['#ee3333', '#336699', '#44aa66', '#aa6633', '#003153']
    _HIST_LEN = 5
    _HUE_DISTORTION_TABLES = {
        # {"PREFS_KEY_WHEEL_TYPE-name": table-of-ranges}
        "rgb": None,
        "ryb": [
            ((0.0, 1 / 6.0), (0.0, 1 / 3.0)),  # red -> yellow
            ((1 / 6.0, 1 / 3.0), (1 / 3.0, 1 / 2.0)),  # yellow -> green
            ((1 / 3.0, 2 / 3.0), (1 / 2.0, 2 / 3.0)),  # green -> blue
        ],
        "rygb": [
            ((0.0, 1 / 6.0), (0.0, 0.25)),   # red -> yellow
            ((1 / 6.0, 1 / 3.0), (0.25, 0.5)),  # yellow -> green
            ((1 / 3.0, 2 / 3.0), (0.5, 0.75)),  # green -> blue
            ((2 / 3.0, 1.0), (0.75, 1.0)),   # blue -> red
        ],
    }
    _DEFAULT_WHEEL_TYPE = "rgb"

    ## Construction

    def __init__(self, prefs, datapath):
        """Initialises with default colors and an empty adjuster list.

        :param prefs: Prefs dict for saving settings.
        :param datapath: Base path for saving palettes and masks.

        """
        super(ColorManager, self).__init__()

        # Defaults
        self._color = None  #: Currently edited color, a UIColor object
        self._hist = []  #: List of previous colors, most recent last
        self._palette = None  #: Current working palette
        self._adjusters = weakref.WeakSet()  #: The set of registered adjusters
        #: Cursor for pickers
        # FIXME: Use Gdk.Cursor.new_for_display()
        self._picker_cursor = Gdk.Cursor.new(Gdk.CursorType.CROSSHAIR)
        self._datapath = datapath  #: Base path for saving palettes and masks
        self._hue_distorts = None  #: Hue-remapping table for color wheels
        self._prefs = prefs  #: Shared preferences dictionary

        # Build the history. Last item is most recent.
        hist_hex = list(prefs.get(PREFS_KEY_COLOR_HISTORY, []))
        hist_hex = self._DEFAULT_HIST + hist_hex
        self._hist = [RGBColor.new_from_hex_str(s) for s in hist_hex]
        self._trim_hist()

        # Restore current color, or use the most recent color.
        col_hex = prefs.get(PREFS_KEY_CURRENT_COLOR, None)
        if col_hex is None:
            col_hex = hist_hex[-1]
        self._color = RGBColor.new_from_hex_str(col_hex)

        # Initialize angle distort table
        wheel_type = prefs.get(PREFS_KEY_WHEEL_TYPE, self._DEFAULT_WHEEL_TYPE)
        distorts_table = self._HUE_DISTORTION_TABLES[wheel_type]
        self._hue_distorts = distorts_table

        # Initialize working palette
        palette_dict = prefs.get(PREFS_PALETTE_DICT_KEY, None)
        if palette_dict is not None:
            palette = Palette.new_from_simple_dict(palette_dict)
        else:
            datapath = self.get_data_path()
            palettes_dir = os.path.join(datapath, DATAPATH_PALETTES_SUBDIR)
            default = os.path.join(palettes_dir, DEFAULT_PALETTE_FILE)
            palette = Palette(filename=default)
        self._palette = palette

        # Capture updates to the working palette
        palette.info_changed += self._palette_changed_cb
        palette.match_changed += self._palette_changed_cb
        palette.sequence_changed += self._palette_changed_cb
        palette.color_changed += self._palette_changed_cb

    ## Picker cursor

    def set_picker_cursor(self, cursor):
        """Sets the color picker cursor.
        """
        self._picker_cursor = cursor

    def get_picker_cursor(self):
        """Return the color picker cursor.

        This shared cursor is for use by adjusters connected to this manager
        which have a screen color picker. The default is a crosshair.

        """
        return self._picker_cursor

    # TODO: if the color picker function needs to be made partly app-aware,
    # move it here and let BrushColorManager override/extend it.

    ## Template/read-only data path for palettes, masks etc.

    def set_data_path(self, datapath):
        """Sets the template/read-only data path for palettes, masks etc.
        """
        self._datapath = datapath

    def get_data_path(self):
        """Returns the template/read-only data path for palettes, masks etc.

        This is for use by adjusters connected to this manager which need to
        load template resources, e.g. palette selectors.

        """
        return self._datapath

    ## Attached ColorAdjusters

    def add_adjuster(self, adjuster):
        """Adds an adjuster to the internal set of adjusters."""
        self._adjusters.add(adjuster)

    def remove_adjuster(self, adjuster):
        """Removes an adjuster."""
        self._adjusters.remove(adjuster)

    def get_adjusters(self):
        """Returns an iterator over the set of registered adjusters."""
        return iter(self._adjusters)

    ## Main shared UIColor object

    def set_color(self, color):
        """Sets the shared `UIColor`, and notifies all registered adjusters.

        Calling this invokes the `color_updated()` method on each
        registered color adjuster after the color has been updated. The
        change is also announced via the color_updated method of the
        ColorManager itself.

        """
        if color == self._color:
            return
        self._color = copy(color)
        self._prefs[PREFS_KEY_CURRENT_COLOR] = color.to_hex_str()
        for adj in self._adjusters:
            adj.color_updated()
        self._palette.match_color(color)
        self.color_updated()

    def get_color(self):
        """Gets a copy of the shared `UIColor`.
        """
        return copy(self._color)

    color = property(get_color, set_color)

    @event
    def color_updated(self):
        """Event: announces changes to the managed color.

        See lib.observable.event for details of how to subscribe.
        """

    ## History of colors used for painting

    def _trim_hist(self):
        self._hist = self._hist[-self._HIST_LEN:]

    def push_history(self, color):
        """Pushes a color to the user history list.

        Calling this invokes the `color_history_updated()` method on
        each registered color adjuster after the history has been
        updated. The change is also announced via the
        color_history_updated method of the ColorManager itself.

        """
        self._hist[:] = [
            c for c in self._hist
            if not (c == color or color == c)
            # Direction of the comparison can matter now that color
            # classes have overridden __eq__ methods.
        ]
        self._hist.append(color)
        self._trim_hist()
        key = PREFS_KEY_COLOR_HISTORY
        val = []
        for c in self._hist:
            s = c.to_hex_str()
            val.append(s)
        self._prefs[key] = val
        for adj in self._adjusters:
            adj.color_history_updated()
        self.color_history_updated()

    @event
    def color_history_updated(self):
        """Event: announces changes to the color history.

        See lib.observable.event for details of how to subscribe.
        """

    def get_history(self):
        """Returns a copy of the color history.
        """
        return deepcopy(self._hist)

    def get_previous_color(self):
        """Returns the most recently used color from the user history list.
        """
        return deepcopy(self._hist[-1])

    ## Prefs access

    def get_prefs(self):
        """Returns the current preferences hash."""
        return self._prefs

    ## Color wheel distortion table (support for RYGB/RGB/RYB-wheels)

    def set_wheel_type(self, typename):
        """Sets the type of attached color wheels by name.

        :param typename: Wheel type name: "rgb", "ryb", or "rygb".
        :type typename: str

        This corresponds to a hue-angle remapping, which will be adopted by
        all wheel-style adjusters attached to this ColorManager.

        """
        old_typename = self.get_wheel_type()
        if typename not in self._HUE_DISTORTION_TABLES:
            typename = self._DEFAULT_WHEEL_TYPE
        if typename == old_typename:
            return
        self._hue_distorts = self._HUE_DISTORTION_TABLES[typename]
        self._prefs[PREFS_KEY_WHEEL_TYPE] = typename
        for adj in self._adjusters:
            if isinstance(adj, HueSaturationWheelAdjuster):
                adj.clear_background()

    def get_wheel_type(self):
        """Returns the current color wheel type name.
        """
        default = self._DEFAULT_WHEEL_TYPE
        return self._prefs.get(PREFS_KEY_WHEEL_TYPE, default)

    def distort_hue(self, h):
        """Distorts a hue from RGB-wheel angles to the current wheel type's.
        """
        if self._hue_distorts is None:
            return h
        h %= 1.0
        for rgb_wheel_range, distorted_wheel_range in self._hue_distorts:
            in0, in1 = rgb_wheel_range
            out0, out1 = distorted_wheel_range
            if h > in0 and h <= in1:
                h -= in0
                h *= (out1 - out0) / (in1 - in0)
                h += out0
                break
        return h

    def undistort_hue(self, h):
        """Reverses the mapping imposed by ``distort_hue()``.
        """
        if self._hue_distorts is None:
            return h
        h %= 1.0
        for rgb_wheel_range, distorted_wheel_range in self._hue_distorts:
            out0, out1 = rgb_wheel_range
            in0, in1 = distorted_wheel_range
            if h > in0 and h <= in1:
                h -= in0
                h *= (out1 - out0) / (in1 - in0)
                h += out0
                break
        return h

    ## Palette access

    @property
    def palette(self):
        """Gets the shared Palette instance."""
        return self._palette

    def _palette_changed_cb(self, palette, *args, **kwargs):
        """Stores changes made to the palette to the prefs"""
        prefs = self.get_prefs()
        prefs[PREFS_PALETTE_DICT_KEY] = palette.to_simple_dict()


class ColorAdjuster(object):
    """Base class for any object which can manipulate a shared `UIColor`.

    Color adjusters are used for changing one or more elements of a color.
    Several are bound to a central `ColorManager`, and broadcast
    changes to it.

    """

    ## Constants

    _DEFAULT_COLOR = RGBColor(0.0, 0.19, 0.33)

    ## Central ColorManager instance (accessors)

    def set_color_manager(self, manager):
        """Sets the shared color adjustment manager this adjuster points to.
        """
        if manager is not None:
            if self in manager.get_adjusters():
                return
        existing = self.get_color_manager()
        if existing is not None:
            existing.remove_adjuster(self)
        self.__manager = manager
        if self.__manager is not None:
            self.__manager.add_adjuster(self)

    def get_color_manager(self):
        """Gets the shared color adjustment manager.
        """
        try:
            return self.__manager
        except AttributeError:
            self.__manager = None
            return None

    color_manager = property(get_color_manager, set_color_manager)

    ## Access to the central managed UIColor (convenience methods)

    def get_managed_color(self):
        """Gets the managed color. Convenience method for use by subclasses.
        """
        if self.color_manager is None:
            return RGBColor(color=self._DEFAULT_COLOR)
        return self.color_manager.get_color()

    def set_managed_color(self, color):
        """Sets the managed color. Convenience method for use by subclasses.
        """
        if self.color_manager is None:
            return
        if color is not None:
            self.color_manager.set_color(color)

    managed_color = property(get_managed_color, set_managed_color)

    ## Central shared prefs access (convenience methods)

    def get_prefs(self):
        if self.color_manager is not None:
            return self.color_manager.get_prefs()
        return {}

    ## Update notification

    def color_updated(self):
        """Called by the manager when the shared `UIColor` changes.
        """
        pass

    def color_history_updated(self):
        """Called by the manager when the color usage history changes.
        """
        pass


class ColorAdjusterWidget (CachedBgDrawingArea, ColorAdjuster):
    """Base class for sliders, wheels, picker areas etc.

    Provides access to the central color manager via the gobject property
    ``color-manager``, and click/drag event handlers for picking colors.
    Derived classes should draw a colorful background by overriding
    `CachedBgWidgetMixin.render_background_cb()`, and keep handlers registered
    here happy by implementing `get_color_at_position()`.

    Color adjusters can operate as sources for dragging colors: subclasses
    should set `IS_DRAG_SOURCE` to `True` before the object is realized to
    enable this.

    """

    ## Behavioural and stylistic class constants

    SCROLL_DELTA = 0.015  #: Delta for a scroll event
    IS_DRAG_SOURCE = False  #: Set to True to make press+move do a select+drag
    IS_IMMEDIATE = True  #: Set False to make bg change only on btn1-release
    _DRAG_COLOR_ID = 1
    _DRAG_TARGETS = [("application/x-color", 0, _DRAG_COLOR_ID)]
    HAS_DETAILS_DIALOG = False  #: Set true for a double-click details dialog
    BORDER_WIDTH = 2  #: Size of the border around the widget.
    STATIC_TOOLTIP_TEXT = None  #: Static tooltip, used during constructor
    OUTLINE_WIDTH = 3  #: Dark outline around shapes: size
    OUTLINE_RGBA = (0, 0, 0, 0.4)  #: Dark shape outline: color
    EDGE_HIGHLIGHT_WIDTH = 1.0  #: Light Tango-ish border for shapes: size
    EDGE_HIGHLIGHT_RGBA = (1, 1, 1, 0.25)  #: Light Tango-ish border: xolor
    ALLOW_HCY_TWEAKING = True  #: Tweak H, C, or Y on non-btn1 drags

    ## Deprecated property names

    @property
    def border(self):
        warn("Use BORDER_WIDTH instead", DeprecatedAPIWarning, 2)
        return self.BORDER_WIDTH

    @property
    def outline_width(self):
        warn("Use OUTLINE_WIDTH instead", DeprecatedAPIWarning, 2)
        return self.OUTLINE_WIDTH

    @property
    def outline_rgba(self):
        warn("Use OUTLINE_RGBA instead", DeprecatedAPIWarning, 2)
        return self.OUTLINE_RGBA

    @property
    def edge_highlight_rgba(self):
        warn("Use EDGE_HIGHLIGHT_RGBA instead", DeprecatedAPIWarning, 2)
        return self.EDGE_HIGHLIGHT_RGBA

    @property
    def edge_highlight_width(self):
        warn("Use EDGE_HIGHLIGHT_WIDTH instead", DeprecatedAPIWarning, 2)
        return self.EDGE_HIGHLIGHT_WIDTH

    @property
    def tooltip_text(self):
        warn("Use STATIC_TOOLTIP_TEXT instead", DeprecatedAPIWarning, 2)
        return self.STATIC_TOOLTIP_TEXT

    ## GObject integration (type name, properties)

    __gtype_name__ = "ColorAdjusterWidget"
    __gproperties__ = {
        'color-manager': (ColorManager,
                          "Color manager",
                          "The ColorManager owning the color to be adjusted",
                          GObject.ParamFlags.READWRITE),
    }

    ## Construction (TODO: rename internals at some point)

    def __init__(self):
        """Initializes, and registers click and drag handlers.
        """
        CachedBgDrawingArea.__init__(self)
        self.__button_down = None
        self.__drag_start_pos = None
        self.__drag_start_color = None
        self.__initial_bg_validity = None
        self.connect("button-press-event", self.__button_press_cb)
        self.connect("motion-notify-event", self.__motion_notify_cb)
        self.connect("button-release-event", self.__button_release_cb)
        self.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.BUTTON_MOTION_MASK
        )
        self._init_color_drag()
        if self.STATIC_TOOLTIP_TEXT is not None:
            self.set_tooltip_text(self.STATIC_TOOLTIP_TEXT)

        # Use twice the standard threshold because tablets always send
        # more tiny motions than mice.
        settings = Gtk.Settings.get_default()
        thres = min(4, int(settings.get_property("gtk-dnd-drag-threshold")))
        self._drag_threshold = 2 * thres

    ## Color drag and drop

    def _init_color_drag(self, *_junk):
        # Drag init
        self._drag_dest_set()
        self.connect("drag-motion", self.drag_motion_cb)
        self.connect('drag-leave', self.drag_leave_cb)
        self.connect('drag-begin', self.drag_begin_cb)
        self.connect('drag-end', self.drag_end_cb)
        if self.IS_DRAG_SOURCE:
            self.connect("drag-data-get", self.drag_data_get_cb)
        self.connect("drag-data-received", self.drag_data_received_cb)

    def _drag_dest_set(self):
        targets = [Gtk.TargetEntry.new(*e) for e in self._DRAG_TARGETS]
        flags = Gtk.DestDefaults.MOTION | Gtk.DestDefaults.DROP
        actions = Gdk.DragAction.MOVE | Gdk.DragAction.COPY
        self.drag_dest_set(flags, targets, actions)

    def drag_motion_cb(self, widget, context, x, y, t):
        pass

    def drag_leave_cb(self, widget, context, time):
        pass

    def drag_begin_cb(self, widget, context):
        color = self.get_managed_color()
        preview = GdkPixbuf.Pixbuf.new(
            GdkPixbuf.Colorspace.RGB, False, 8, 32, 32)
        pixel = color.to_fill_pixel()
        preview.fill(pixel)
        Gtk.drag_set_icon_pixbuf(context, preview, 0, 0)

    def drag_end_cb(self, widget, context):
        pass

    def drag_data_get_cb(self, widget, context, selection, target_type,
                         time):
        """Gets the current color when a drop happens somewhere"""
        if "application/x-color" not in map(str, context.list_targets()):
            return False
        color = self.get_managed_color()
        data = gui.uicolor.to_drag_data(color)
        selection.set(Gdk.atom_intern("application/x-color", False),
                      16, data)
        logger.debug(
            "drag-data-get: sending type=%r",
            selection.get_data_type(),
        )
        logger.debug("drag-data-get: sending fmt=%r", selection.get_format())
        logger.debug("drag-data-get: sending data=%r len=%r",
                     selection.get_data(), len(selection.get_data()))
        return True

    def drag_data_received_cb(self, widget, context, x, y, selection,
                              info, time):
        """Gets the current color when a drop happens on the widget"""
        if "application/x-color" not in map(str, context.list_targets()):
            return False
        data = selection.get_data()
        data_type = selection.get_data_type()
        fmt = selection.get_format()
        logger.debug("drag-data-received: got type=%r", data_type)
        logger.debug("drag-data-received: got fmt=%r", fmt)
        logger.debug("drag-data-received: got data=%r len=%r", data, len(data))
        color = gui.uicolor.from_drag_data(data)
        context.finish(True, True, time)
        self.set_managed_color(color)
        return True

    ## GObject properties (TODO: use decorator syntax instead)

    def do_set_property(self, prop, value):
        if prop.name == 'color-manager':
            self.set_color_manager(value)
        else:
            raise AttributeError('unknown property %s' % prop.name)

    def do_get_property(self, prop):
        if prop.name == 'color-manager':
            return self.get_color_manager()
        else:
            raise AttributeError('unknown property %s' % prop.name)

    ## Color-at-position interface (for subclasses, primarily)

    def get_color_at_position(self, x, y):
        """Get the color a position represents. Subclasses must override.

        Can be legitimately used by drawing routines, but at this level
        it's used only by the private handlers for button presses etc.

        """
        raise NotImplementedError

    def set_color_at_position(self, x, y, color):
        """Handles colors set by the double-click color selection dialog.

        Certain subclasses which are sensitive to the `x` and `y` position of
        the double click that launches the dialog override this. At this level
        these parameters are ignored.

        """
        self.set_managed_color(color)

    ## CachedBgDrawingArea implementation: bg validity determined by color

    def get_background_validity(self):
        """Returns a validity token for the displayed background.

        This implementation of `CachedBgWidgetMixin.get_background_validity()`
        uses the full string representation of the managed color, but can be
        overridden to return a smaller subset of its channels or quantize it
        for fewer redraws.

        This implementation also respects the IS_IMMEDIATE flag:
        if background rerendering is non-immediate,
        it ensures that the token doesn't change
        while the pointer button is held.

        """
        if (not self.IS_IMMEDIATE) and self.__button_down == 1:
            return self.__initial_bg_validity
        return repr(self.get_managed_color())

    ## Pointer event handling

    def __button_press_cb(self, widget, event):
        """Button press handler."""
        self.__button_down = event.button
        color = self.get_color_at_position(event.x, event.y)
        self.__initial_bg_validity = repr(self.get_managed_color())
        pos = event.x, event.y

        self.__drag_start_pos = pos

        # A single click on button1 sets the current colour,
        # and may start DnD drags.
        if event.button == 1 and event.type == Gdk.EventType.BUTTON_PRESS:
            self.set_managed_color(color)
            if self.IS_DRAG_SOURCE:
                self.__drag_start_color = color
                if color is not None:
                    return True  # drag may be about to start

        # Double-click shows the details adjuster
        elif (event.button == 1
                and event.type == Gdk.EventType._2BUTTON_PRESS
                and self.HAS_DETAILS_DIALOG):
            self.__button_down = None
            self.__drag_start_color = None
            prev_color = self.get_color_manager().get_previous_color()
            color = gui.dialogs.ask_for_color(
                title=_("Color details"),
                color=color,
                previous_color=prev_color,
                parent=self.get_toplevel()
            )
            if color is not None:
                self.set_color_at_position(event.x, event.y, color)
            return True  # dialog was used for colour selection

        # Button2 or Button3 drag begins tweaking/"bending" the colour
        elif event.button != 1 and self.ALLOW_HCY_TWEAKING:
            self.__drag_start_color = color

        # Let other registered handlers run, normally.
        return False

    def __motion_notify_cb(self, widget, event):
        """Button1 motion handler."""
        pos = event.x, event.y
        if (self.__button_down is not None) and self.__button_down == 1:
            if self.IS_DRAG_SOURCE:
                if not self.__drag_start_color:
                    return False
                start_pos = self.__drag_start_pos
                dx = pos[0] - start_pos[0]
                dy = pos[1] - start_pos[1]
                dist = math.hypot(dx, dy)
                if (dist > self._drag_threshold) and self.__drag_start_color:
                    logger.debug(
                        "Start drag (dist=%0.3f) with colour %r",
                        dist, self.__drag_start_color,
                    )
                    self.__drag_start_color = None
                    self.drag_begin_with_coordinates(
                        targets = Gtk.TargetList.new([
                            Gtk.TargetEntry.new(*e)
                            for e in self._DRAG_TARGETS
                        ]),
                        actions = Gdk.DragAction.MOVE | Gdk.DragAction.COPY,
                        button = 1,
                        event = event,
                        x = event.x,
                        y = event.y,
                    )
                    return True  # a drag was just started
            else:
                # Non-drag-source widgets update the color continuously while
                # the mouse button is held down and the pointer moved.
                color = self.get_color_at_position(event.x, event.y)
                self.set_managed_color(color)

        # Relative chroma/luma/hue bending with other buttons.
        elif ((self.__button_down is not None) and self.ALLOW_HCY_TWEAKING
              and self.__button_down > 1):
            if self.__drag_start_color is None:
                return False
            col = HCYColor(color=self.__drag_start_color)
            alloc = self.get_allocation()
            w, h = alloc.width, alloc.height
            size = max(w, h)
            ex, ey = event.x, event.y
            sx, sy = self.__drag_start_pos
            dx, dy = sx - ex, sy - ey

            # Pick a dimension to tweak
            if event.state & Gdk.ModifierType.SHIFT_MASK:
                bend = "chroma"
                dy = -dy
            elif event.state & Gdk.ModifierType.CONTROL_MASK:
                bend = "hue"
            else:
                bend = "luma"
                dy = -dy

            # Interpretation of dx depends on text direction
            if widget.get_direction() == Gtk.TextDirection.RTL:
                dx = -dx

            # Use the delta with the largest absolute value
            # FIXME: this has some jarring discontinities
            dd = dx if abs(dx) > abs(dy) else dy

            if bend == "chroma":
                c0 = clamp(col.c, 0., 1.)
                p = (c0 * size) - dd
                col.c = clamp(p / size, 0., 1.)
            elif bend == "hue":
                h0 = clamp(col.h, 0., 1.)
                p = (h0 * size) - dd
                h = p / size
                while h < 0:
                    h += 1.0
                col.h = h % 1.0
            else:   # luma
                y0 = clamp(col.y, 0., 1.)
                p = (y0 * size) - dd
                col.y = clamp(p / size, 0., 1.)
            self.set_managed_color(col)

        # Let other registered handlers run, normally.
        return False

    def __button_release_cb(self, widget, event):
        """Button release handler."""
        if event.button == self.__button_down:
            xa, ya = self.__drag_start_pos
            xb, yb = event.x, event.y
            self.__initial_bg_validity = None
            self.__button_down = None
            self.__drag_start_pos = None
            self.__drag_start_color = None
            # Redraw background if we've been suppressing bg redraws
            if (not self.IS_IMMEDIATE) and event.button == 1:
                self.queue_draw()

            if math.hypot(xa - xb, ya - yb) < self._drag_threshold:
                self.clicked(event, (xa, ya))
        return False

    @event
    def clicked(self, event, pos):
        """The user has clicked. Implies no drag was started.

        :param Gdk.Event event: The final button release GdkEvent.
        :param tuple pos: The initial click position.

        Only do things in response to this observable.event if you need
        to do something in addition to setting the colour.  This event
        fires on button-up events, i.e. after initial click.  The
        pointer is guraraneed not to have travelled further than the
        drag threshold.

        """

    ## Update notification

    def color_updated(self):
        """Called in response to the managed color changing: queues a redraw.
        """
        self.queue_draw()


class IconRenderableColorAdjusterWidget (ColorAdjusterWidget, IconRenderable):
    """Base class for ajuster widgets whose background can be used for icons.

    Typically the background of something like a wheel adjuster is the most
    useful part for the purposes of icon making.

    """

    ## Rendering

    def render_as_icon(self, cr, size):
        """Renders the background into an icon.

        This implementation requires a `render_background_cb()` method which
        supports an extra argument named ``icon_border``, the pixel size of a
        suggested small outer border.

        """
        b = max(2, int(size // 16))
        self.render_background_cb(cr, wd=size, ht=size, icon_border=b)


class PreviousCurrentColorAdjuster (ColorAdjusterWidget):
    """Shows the current and previous color side by side for comparison.
    """

    ## Constants (behavioural specialization)

    # Class specialisation
    IS_DRAG_SOURCE = True
    HAS_DETAILS_DIALOG = True
    STATIC_TOOLTIP_TEXT = _("Current brush color, and the color "
                            "most recently used for painting")

    ## Construction

    def __init__(self):
        ColorAdjusterWidget.__init__(self)
        s = (self.BORDER_WIDTH * 2) + 4
        self.set_size_request(s, s)

    ## Rendering

    def render_background_cb(self, cr, wd, ht):
        mgr = self.get_color_manager()
        if mgr is None:
            return
        curr = mgr.get_color()
        prev = mgr.get_previous_color()
        b = self.BORDER_WIDTH

        eff_wd = wd - b - b
        eff_ht = ht - b - b

        cr.rectangle(b + 0.5, b + 0.5, eff_wd - 1, eff_ht - 1)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.set_source_rgba(*self.OUTLINE_RGBA)
        cr.set_line_width(self.OUTLINE_WIDTH)
        cr.stroke()

        cr.rectangle(b, b, int(eff_wd // 2), eff_ht)
        cr.set_source_rgb(*curr.get_rgb())
        cr.fill()
        cr.rectangle(wd // 2, b, eff_wd - int(eff_wd // 2), eff_ht)
        cr.set_source_rgb(*prev.get_rgb())
        cr.fill()

        cr.rectangle(b + 0.5, b + 0.5, eff_wd - 1, eff_ht - 1)
        cr.set_source_rgba(*self.EDGE_HIGHLIGHT_RGBA)
        cr.set_line_width(self.EDGE_HIGHLIGHT_WIDTH)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.stroke()

    def get_background_validity(self):
        mgr = self.get_color_manager()
        if mgr is None:
            return
        curr = mgr.get_color()
        prev = mgr.get_previous_color()
        return (curr.get_rgb(), prev.get_rgb())

    def paint_foreground_cb(self, cr, wd, ht):
        pass

    ## Color-at-position

    def get_color_at_position(self, x, y):
        alloc = self.get_allocation()
        mgr = self.get_color_manager()
        if x < alloc.width // 2:
            color = mgr.get_color()
        else:
            color = mgr.get_previous_color()
        return deepcopy(color)

    ## Update notifications

    def color_history_updated(self):
        self.queue_draw()


class SliderColorAdjuster (ColorAdjusterWidget):
    """Base class for slider controls with a colored background.

    Supports both simple and complex gradients. A simple gradient is a
    continuous linear interpolation between the two endpoints; complex
    gradients are sampled many times along their length and then interpolated
    between linearly.

    """

    # GObject integration
    __gtype_name__ = "SliderColorAdjuster"

    vertical = False  #: Bar orientation.
    samples = 0       #: How many extra samples to use along the bar length.

    def __init__(self):
        """Initialise; state variables can be set here.

        The state variables, `vertical`, `border`, and `samples`
        can be set here, but not after the widget has been realized.

        """
        ColorAdjusterWidget.__init__(self)
        self.connect("realize", self.__realize_cb)
        self.connect("scroll-event", self.__scroll_cb)
        self.add_events(Gdk.EventMask.SCROLL_MASK)

    def __realize_cb(self, widget):
        """Realize handler; establishes sizes based on `vertical` etc.
        """
        bw = uimisc.SLIDER_MIN_WIDTH
        bl = uimisc.SLIDER_MIN_LENGTH
        if self.vertical:
            self.set_size_request(bw, bl)
        else:
            self.set_size_request(bl, bw)

    def render_background_cb(self, cr, wd, ht):
        b = self.BORDER_WIDTH
        bar_length = (self.vertical and ht or wd) - b - b
        b_x = b + 0.5
        b_y = b + 0.5
        b_w = wd - b - b - 1
        b_h = ht - b - b - 1

        # Build the gradient
        if self.vertical:
            bar_gradient = cairo.LinearGradient(0, b, 0, b + bar_length)
        else:
            bar_gradient = cairo.LinearGradient(b, 0, b + bar_length, 0)
        samples = self.samples + 2
        for s in xrange(samples + 1):
            p = s / samples
            col = self.get_color_for_bar_amount(p)
            r, g, b = col.get_rgb()
            if self.vertical:
                p = 1 - p
            bar_gradient.add_color_stop_rgb(p, r, g, b)

        # Paint bar with Tango-like edges
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.set_source_rgba(*self.OUTLINE_RGBA)
        cr.set_line_width(self.OUTLINE_WIDTH)
        cr.rectangle(b_x, b_y, b_w, b_h)
        cr.stroke()

        ## Paint bar
        cr.set_source(bar_gradient)
        cr.rectangle(b_x - 0.5, b_y - 0.5, b_w + 1, b_h + 1)
        cr.fill()

        ## Highlighted edge
        if b_w > 5 and b_h > 5:
            cr.set_line_width(self.EDGE_HIGHLIGHT_WIDTH)
            cr.set_source_rgba(*self.EDGE_HIGHLIGHT_RGBA)
            cr.rectangle(b_x, b_y, b_w, b_h)
            cr.stroke()

    def get_bar_amount_for_color(self, color):
        """Bar amount for a given `UIColor`; subclasses must implement.
        """
        raise NotImplementedError

    def get_color_for_bar_amount(self, amt):
        """The `UIColor` for a given bar amount; subclasses must implement.
        """
        raise NotImplementedError

    def get_color_at_position(self, x, y):
        """Color for a particular position using ``bar_amount`` methods.
        """
        amt = self.point_to_amount(x, y)
        return self.get_color_for_bar_amount(amt)

    def paint_foreground_cb(self, cr, wd, ht):
        b = int(self.BORDER_WIDTH)
        col = self.get_managed_color()
        amt = self.get_bar_amount_for_color(col)
        amt = float(clamp(amt, 0, 1))
        bar_size = int((self.vertical and ht or wd) - 1 - 2 * b)
        if self.vertical:
            amt = 1.0 - amt
            x1 = b + 0.5
            x2 = wd - x1
            y1 = y2 = int(amt * bar_size) + b + 0.5
        else:
            x1 = x2 = int(amt * bar_size) + b + 0.5
            y1 = b + 0.5
            y2 = ht - y1

        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_width(5)
        cr.move_to(x1, y1)
        cr.line_to(x2, y2)
        cr.set_source_rgb(0, 0, 0)
        cr.stroke_preserve()

        cr.set_source_rgb(1, 1, 1)
        cr.set_line_width(3.5)
        cr.stroke_preserve()

        cr.set_source_rgb(*col.get_rgb())
        cr.set_line_width(0.25)
        cr.stroke()

    def point_to_amount(self, x, y):
        alloc = self.get_allocation()
        if self.vertical:
            len = alloc.height - 2 * self.BORDER_WIDTH
            p = y
        else:
            len = alloc.width - 2 * self.BORDER_WIDTH
            p = x
        p = clamp(p - self.BORDER_WIDTH, 0, len)
        amt = float(p) / len
        if self.vertical:
            amt = 1 - amt
        return amt

    def __scroll_cb(self, widget, event):
        d = self.SCROLL_DELTA
        if not self.vertical:
            d *= -1
        if event.direction in (
                Gdk.ScrollDirection.DOWN, Gdk.ScrollDirection.LEFT):
            d *= -1
        col = self.get_managed_color()
        amt = self.get_bar_amount_for_color(col)
        amt = clamp(amt + d, 0.0, 1.0)
        col = self.get_color_for_bar_amount(amt)
        self.set_managed_color(col)
        return True


class HueSaturationWheelMixin(object):
    """Mixin for wheel-style hue/saturation adjusters, indep. of color space

    Implementing most of the wheel-drawing machinery as a mixin allows the
    methods to be reused independently of the usual base classes for
    Adjusters, which might be inconvenient if sub-widgets are required.

    This base class is independent of the color space, but assumes a
    cylindrical shape with the central axis representing lightness and angle
    representing hue.

    Desaturated colors reside at the centre of the wheel. This makes them
    somewhat harder to pick ordinarily, but desaturated colors are handy for
    artists. Therefore, we apply a subtle gamma curve when drawing, and when
    interpreting clicked values at this level. The internal API presented here
    for use by subclasses already has this compensation applied.

    """

    # Class configuration vars, for overriding

    #: How many slices to render
    HUE_SLICES = 64

    #: How many divisions of grey to use for gamma interp.
    SAT_SLICES = 5

    #: Greyscale gamma
    SAT_GAMMA = 1.50

    def get_radius(self, wd=None, ht=None, border=None, alloc=None):
        """Returns the radius, suitable for a pixel-edge-aligned centre.
        """
        if wd is None or ht is None:
            if alloc is None:
                alloc = self.get_allocation()
            wd = alloc.width
            ht = alloc.height
        if border is None:
            border = self.BORDER_WIDTH
        return int((min(wd, ht) / 2.0)) - int(border) + 0.5

    def get_center(self, wd=None, ht=None, alloc=None):
        """Returns the wheel centre, suitable for an N+0.5 radius.
        """
        if wd is None or ht is None:
            if alloc is None:
                alloc = self.get_allocation()
            wd = alloc.width
            ht = alloc.height
        cx = int(wd // 2)
        cy = int(ht // 2)
        return cx, cy

    def get_background_validity(self):
        """Gets the bg validity token, for `CachedBgWidgetMixin` impls.
        """
        # The wheel's background is valid if the central grey hasn't changed.
        grey = self.color_at_normalized_polar_pos(0, 0)
        rgb = grey.get_rgb()
        k = max(rgb)
        assert k == min(rgb)
        # Quantize a bit to reduce redraws due to conversion noise.
        return int(k * 1000)

    def get_color_at_position(self, x, y):
        """Gets the color at a position, for `ColorAdjusterWidget` impls.
        """
        alloc = self.get_allocation()
        cx, cy = self.get_center(alloc=alloc)
        # Normalized radius
        r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        radius = self.get_radius(alloc=alloc)
        if r > radius:
            r = radius
        r /= radius
        r **= self.SAT_GAMMA
        # Normalized polar angle
        theta = 1.25 - (math.atan2(x - cx, y - cy) / (2 * math.pi))
        while theta <= 0:
            theta += 1.0
        theta %= 1.0
        mgr = self.get_color_manager()
        if mgr:
            theta = mgr.undistort_hue(theta)
        return self.color_at_normalized_polar_pos(r, theta)

    def render_background_cb(self, cr, wd, ht, icon_border=None):
        """Renders the offscreen bg, for `ColorAdjusterWidget` impls.
        """
        cr.save()

        ref_grey = self.color_at_normalized_polar_pos(0, 0)

        border = icon_border
        if border is None:
            border = self.BORDER_WIDTH
        radius = self.get_radius(wd, ht, border)

        steps = self.HUE_SLICES
        sat_slices = self.SAT_SLICES
        sat_gamma = self.SAT_GAMMA

        # Move to the centre
        cx, cy = self.get_center(wd, ht)
        cr.translate(cx, cy)

        # Clip, for a slight speedup
        cr.arc(0, 0, radius + border, 0, 2 * math.pi)
        cr.clip()

        # Tangoesque outer border
        cr.set_line_width(self.OUTLINE_WIDTH)
        cr.arc(0, 0, radius, 0, 2 * math.pi)
        cr.set_source_rgba(*self.OUTLINE_RGBA)
        cr.stroke()

        # Each slice in turn
        cr.save()
        cr.set_line_width(1.0)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        step_angle = 2.0 * math.pi / steps
        mgr = self.get_color_manager()
        for ih in xrange(steps + 1):  # overshoot by 1, no solid bit for final
            h = ih / steps
            if mgr:
                h = mgr.undistort_hue(h)
            edge_col = self.color_at_normalized_polar_pos(1.0, h)
            rgb = edge_col.get_rgb()
            if ih > 0:
                # Backwards gradient
                cr.arc_negative(0, 0, radius, 0, -step_angle)
                x, y = cr.get_current_point()
                cr.line_to(0, 0)
                cr.close_path()
                lg = cairo.LinearGradient(radius, 0, (x + radius) / 2, y)
                lg.add_color_stop_rgba(0, rgb[0], rgb[1], rgb[2], 1.0)
                lg.add_color_stop_rgba(1, rgb[0], rgb[1], rgb[2], 0.0)
                cr.set_source(lg)
                cr.fill()
            if ih < steps:
                # Forward solid
                cr.arc(0, 0, radius, 0, step_angle)
                x, y = cr.get_current_point()
                cr.line_to(0, 0)
                cr.close_path()
                cr.set_source_rgb(*rgb)
                cr.stroke_preserve()
                cr.fill()
            cr.rotate(step_angle)
        cr.restore()

        # Cheeky approximation of the right desaturation gradients
        rg = cairo.RadialGradient(0, 0, 0, 0, 0, radius)
        add_distance_fade_stops(rg, ref_grey.get_rgb(),
                                nstops=sat_slices,
                                gamma=1.0 / sat_gamma)
        cr.set_source(rg)
        cr.arc(0, 0, radius, 0, 2 * math.pi)
        cr.fill()

        # Tangoesque inner border
        cr.set_source_rgba(*self.EDGE_HIGHLIGHT_RGBA)
        cr.set_line_width(self.EDGE_HIGHLIGHT_WIDTH)
        cr.arc(0, 0, radius, 0, 2 * math.pi)
        cr.stroke()

        # Some small notches on the disc edge for pure colors
        if wd > 75 or ht > 75:
            cr.save()
            cr.arc(0, 0, radius + self.EDGE_HIGHLIGHT_WIDTH, 0, 2 * math.pi)
            cr.clip()
            pure_cols = [
                RGBColor(1, 0, 0), RGBColor(1, 1, 0), RGBColor(0, 1, 0),
                RGBColor(0, 1, 1), RGBColor(0, 0, 1), RGBColor(1, 0, 1),
            ]
            for col in pure_cols:
                x, y = self.get_pos_for_color(col)
                x = int(x) - cx
                y = int(y) - cy
                cr.set_source_rgba(*self.EDGE_HIGHLIGHT_RGBA)
                cr.arc(
                    x + 0.5, y + 0.5,
                    1.0 + self.EDGE_HIGHLIGHT_WIDTH,
                    0, 2 * math.pi,
                )
                cr.fill()
                cr.set_source_rgba(*self.OUTLINE_RGBA)
                cr.arc(
                    x + 0.5, y + 0.5,
                    self.EDGE_HIGHLIGHT_WIDTH,
                    0, 2 * math.pi,
                )
                cr.fill()
            cr.restore()

        cr.restore()

    def color_at_normalized_polar_pos(self, r, theta):
        """Get the color represented by a polar position.

        The terms `r` and `theta` are normalised to the range 0...1 and refer
        to the undistorted color space.

        """
        raise NotImplementedError

    def get_normalized_polar_pos_for_color(self, col):
        """Inverse of `color_at_normalized_polar_pos`.
        """
        # FIXME: make the names consistent
        raise NotImplementedError

    def get_pos_for_color(self, col):
        nr, ntheta = self.get_normalized_polar_pos_for_color(col)
        mgr = self.get_color_manager()
        if mgr:
            ntheta = mgr.distort_hue(ntheta)
        nr **= 1.0 / self.SAT_GAMMA
        alloc = self.get_allocation()
        wd, ht = alloc.width, alloc.height
        radius = self.get_radius(wd, ht, self.BORDER_WIDTH)
        cx, cy = self.get_center(wd, ht)
        r = radius * clamp(nr, 0, 1)
        t = clamp(ntheta, 0, 1) * 2 * math.pi
        x = int(cx + r * math.cos(t)) + 0.5
        y = int(cy + r * math.sin(t)) + 0.5
        return x, y

    def paint_foreground_cb(self, cr, wd, ht):
        """Fg marker painting, for `ColorAdjusterWidget` impls.
        """
        col = self.get_managed_color()
        radius = self.get_radius(wd, ht, self.BORDER_WIDTH)
        cx = int(wd // 2)
        cy = int(ht // 2)
        cr.arc(cx, cy, radius + 0.5, 0, 2 * math.pi)
        cr.clip()
        x, y = self.get_pos_for_color(col)
        draw_marker_circle(cr, x, y, size=2)


class HueSaturationWheelAdjuster (HueSaturationWheelMixin,
                                  IconRenderableColorAdjusterWidget):
    """Concrete base class for hue/saturation wheels, indep. of color space.
    """

    def __init__(self):
        IconRenderableColorAdjusterWidget.__init__(self)
        w = uimisc.PRIMARY_ADJUSTERS_MIN_WIDTH
        h = uimisc.PRIMARY_ADJUSTERS_MIN_HEIGHT
        self.set_size_request(w, h)
