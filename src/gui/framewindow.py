# This file is part of MyPaint.
# Copyright (C) 2010-2020 by the MyPaint Development Team
# Copyright (C) 2012-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Frame manipulation mode and subwindow"""

## Imports

from __future__ import division, print_function
import math
import functools

from lib.gibindings import Gtk
from lib.gibindings import Gdk
from lib.gibindings import GLib
from gettext import gettext as _
import cairo

from gui.tileddrawwidget import TiledDrawWidget  # noqa
import gui.mode
from lib.color import RGBColor
from lib.alg import pairwise, intersection_of_vector_and_poly, LineType
from . import uicolor
from .overlays import Overlay
import lib.helpers
from lib.helpers import Rect
from lib.document import DEFAULT_RESOLUTION
import gui.cursor
import gui.style


class _EditZone:
    INSIDE = 0x00
    LEFT = 0x01
    RIGHT = 0x02
    TOP = 0x04
    BOTTOM = 0x08
    OUTSIDE = 0x10
    REMOVE_FRAME = 0x20
    CREATE_FRAME = 0x30


_SIDES = (_EditZone.LEFT, _EditZone.TOP, _EditZone.RIGHT, _EditZone.BOTTOM)


class FrameEditMode (gui.mode.ScrollableModeMixin,
                     gui.mode.DragMode):
    """Stackable interaction mode for editing the document frame.

    The frame editing mode has an associated settings panel.
    """

    # Class-level configuration
    ACTION_NAME = 'FrameEditMode'

    pointer_behavior = gui.mode.Behavior.EDIT_OBJECTS
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    # These will be overridden on enter()
    inactive_cursor = None
    active_cursor = None

    unmodified_persist = True
    permitted_switch_actions = {
        'ShowPopupMenu',
        'RotateViewMode',
        'ZoomViewMode',
        'PanViewMode',
    }

    EDGE_SENSITIVITY = 10  # pixels

    _MIN_FRAME_SIZE = 1

    # Dragging interpretation by hit zone
    DRAG_EFFECTS = {
        # Values: (dx, dy, dw, dh)
        # These are multipliers of the mouse movement's effect.
        # dy and dx are always either 0 or 1.
        # dh and dw sometimes compensate for the effect of dy and dx.
        _EditZone.LEFT + _EditZone.TOP: (1, 1, -1, -1),
        _EditZone.LEFT: (1, 0, -1, 0),
        _EditZone.LEFT + _EditZone.BOTTOM: (1, 0, -1, +1),
        _EditZone.TOP: (0, 1, 0, -1),
        _EditZone.INSIDE: (1, 1, 0, 0),
        _EditZone.BOTTOM: (0, 0, 0, +1),
        _EditZone.RIGHT + _EditZone.TOP: (0, 1, +1, -1),
        _EditZone.RIGHT: (0, 0, +1, 0),
        _EditZone.RIGHT + _EditZone.BOTTOM: (0, 0, +1, +1),
        _EditZone.OUTSIDE: (0, 0, 0, 0),
    }

    # Options widget singleton
    _OPTIONS_WIDGET = None

    @classmethod
    def get_name(cls):
        return _(u"Edit Frame")

    def get_usage(cls):
        return _(u"Adjust the document frame")

    def __init__(self, **kwds):
        """Initialize."""
        super(FrameEditMode, self).__init__(**kwds)
        self._zone = None
        self._orig_frame = None
        self._start_model_pos = None
        self.remove_button_pos = None   # updated by overlay's paint()
        self._click_info = None
        self._entered_before = False
        self._change_timeout_id = None
        self._queued_frame = None

    def enter(self, doc, **kwds):
        """Enter the mode"""
        super(FrameEditMode, self).enter(doc, **kwds)
        # Assign cursors
        mkcursor = functools.partial(
            self.doc.app.cursors.get_action_cursor,
            self.ACTION_NAME,
        )
        cn = gui.cursor.Name
        self.cursor_move_w_e = mkcursor(cn.MOVE_WEST_OR_EAST)
        self.cursor_move_n_s = mkcursor(cn.MOVE_NORTH_OR_SOUTH)
        self.cursor_move_nw_se = mkcursor(cn.MOVE_NORTHWEST_OR_SOUTHEAST)
        self.cursor_move_ne_sw = mkcursor(cn.MOVE_NORTHEAST_OR_SOUTHWEST)
        self.cursor_hand_closed = mkcursor(cn.HAND_CLOSED)
        self.cursor_hand_open = mkcursor(cn.HAND_OPEN)
        self.cursor_forbidden = mkcursor(cn.ARROW_FORBIDDEN)
        self.cursor_remove = mkcursor(cn.ARROW)
        self.cursor_create = mkcursor(cn.ADD)
        # If the frame isn't visible, show it. If it doesn't yet have a size,
        # then assign a sensible one which makes the frame visible on screen.
        model = self.doc.model
        if not self._entered_before:
            self._entered_before = True
            if not model.get_frame_enabled():
                x, y, w, h = model.get_frame()
                if w > 0 and h > 0:
                    model.set_frame_enabled(True, user_initiated=True)
                else:
                    self._place_new_frame(self.doc.tdw, pos=None)
        # Overlay needs to be drawn
        self.doc.tdw.queue_draw()

    def _place_new_frame(self, tdw, pos=None):
        """Place a new frame on the screen so that it's visible

        :param TiledDrawWidget tdw: canvas widget
        :param tuple pos: position of the frame centre, display (x, y)

        The existing frame position is discarded, and a new position is
        chosen for the frame so that as many of its edges are as visible
        as possible.

        """
        corners = tdw.get_corners_model_coords()
        if pos:
            cx, cy = tdw.display_to_model(*pos)
        else:
            (_x1, _y1), (_x2, _y2) = corners[::2]
            cx, cy = (_x2 + _x1) / 2.0, (_y1 + _y2) / 2.0
        # Set the frame size to the smallest diagonal
        shortest = min(math.hypot(_x - cx, _y - cy) for _x, _y in corners[:2])
        frame_size = int(round(max(100, shortest)))
        x = int(round(cx - frame_size/2.0))
        y = int(round(cy - frame_size/2.0))
        tdw.doc.set_frame([x, y, frame_size, frame_size], user_initiated=True)

    def leave(self, **kwds):
        if not self._change_timeout_id and self._queued_frame is not None:
            self._set_frame()
        if self.doc:
            self.doc.tdw.queue_draw()
        super(FrameEditMode, self).leave(**kwds)

    def _get_zone(self, tdw, xd, yd):
        """Get an edit zone for a cursor position"""
        # Test button hits
        if self.remove_button_pos:
            xbd, ybd = self.remove_button_pos
            dist = math.hypot(xbd-xd, ybd-yd)
            if dist <= gui.style.FLOATING_BUTTON_RADIUS:
                return _EditZone.REMOVE_FRAME
        # Click anywhere when the frame is off to create a new one
        model = self.doc.model
        if not model.get_frame_enabled():
            return _EditZone.CREATE_FRAME
        # Rest of the code works mostly in model coords
        x, y = tdw.display_to_model(xd, yd)
        # Frame top-left and bottom-right, in model coords
        fx1, fy1, fw, fh = model.get_frame()
        fx2, fy2 = fx1+fw, fy1+fh
        # Calculate the maximum permissible distance from the edges
        dx1, dy1 = tdw.display_to_model(0, 0)
        dx2, dy2 = tdw.display_to_model(0, self.EDGE_SENSITIVITY)
        max_d = math.hypot(dx1-dx2, dy1-dy2)
        cursor_is_outside = (
            x < fx1 - max_d or
            x > fx2 + max_d or
            y < fy1 - max_d or
            y > fy2 + max_d
        )
        if cursor_is_outside:
            return _EditZone.OUTSIDE
        # It's on an edge, or inside
        zone = 0
        if abs(fx2 - x) <= max_d:
            zone |= _EditZone.RIGHT
        elif abs(fx1 - x) <= max_d:
            zone |= _EditZone.LEFT
        if abs(fy2 - y) <= max_d:
            zone |= _EditZone.BOTTOM
        elif abs(fy1 - y) <= max_d:
            zone |= _EditZone.TOP
        if zone == 0:
            zone = _EditZone.INSIDE
        return zone

    def _update_cursors(self, tdw):
        """Update the cursors based on the current zone

        Only need to call this when the edit zone changes.
        Still need the TDW the event that may required the change
        originated on for coordinate translations.

        """

        # Simpler interpretations
        zone = self._zone
        if zone == _EditZone.OUTSIDE:
            self.active_cursor = self.cursor_forbidden
            self.inactive_cursor = self.cursor_forbidden
            tdw.set_override_cursor(self.inactive_cursor)
            return
        elif zone == _EditZone.INSIDE:
            self.active_cursor = self.cursor_hand_closed
            self.inactive_cursor = self.cursor_hand_open
            tdw.set_override_cursor(self.inactive_cursor)
            return
        elif zone == _EditZone.REMOVE_FRAME:
            self.active_cursor = self.cursor_remove
            self.inactive_cursor = self.cursor_remove
            tdw.set_override_cursor(self.inactive_cursor)
            return
        elif zone == _EditZone.CREATE_FRAME:
            self.active_cursor = self.cursor_create
            self.inactive_cursor = self.cursor_create
            tdw.set_override_cursor(self.inactive_cursor)
            return

        # Otherwise the current zone is over one of the edges or a
        # corner blob. Pick a move cursor with an appropriate direction,
        # noting that we're limited to just compass directions.

        # Centre of frame, in display coordinates
        model = self.doc.model
        fx, fy, fw, fh = model.get_frame()
        cx, cy = fx+(fw/2.0), fy+(fh/2.0)
        cxd, cyd = tdw.model_to_display(cx, cy)

        # A reference point, reflecting the side or edge where the pointer is
        rx, ry = cx, cy
        if zone & _EditZone.RIGHT:
            rx = fx+fw
        elif zone & _EditZone.LEFT:
            rx = fx
        if zone & _EditZone.BOTTOM:
            ry = fy+fh
        elif zone & _EditZone.TOP:
            ry = fy
        rxd, ryd = tdw.model_to_display(rx, ry)

        # Angle of the line from (cx, cy) to (rx, ry), in display space
        # First constrain to {0..2pi}
        theta = math.atan2((ryd-cyd), (rxd-cxd))
        while theta < 2*math.pi:
            theta += 2*math.pi
        theta %= 2*math.pi
        assert theta >= 0
        assert theta < 2*math.pi

        # The cursor chosen reflects how the chosen edge can be moved.
        cursors = [
            (1, self.cursor_move_w_e),     # right side
            (3, self.cursor_move_nw_se),   # bottom right corner
            (5, self.cursor_move_n_s),     # bottom side
            (7, self.cursor_move_ne_sw),   # bottom left corner
            (9, self.cursor_move_w_e),     # left side
            (11, self.cursor_move_nw_se),  # top left corner
            (13, self.cursor_move_n_s),    # top side
            (15, self.cursor_move_ne_sw),  # top right corner
            (17, self.cursor_move_w_e),    # right side
        ]
        for i, cursor in cursors:
            if theta < i*(2.0/16)*math.pi:
                self.inactive_cursor = cursor
                self.active_cursor = cursor
                tdw.set_override_cursor(self.inactive_cursor)
                return

        # This should never happen.
        self.cursor = Gdk.Cursor.new(Gdk.BOGOSITY)
        tdw.set_override_cursor(self.cursor)

    def _update_zone_and_cursors(self, tdw, x, y):
        zone = self._get_zone(tdw, x, y)
        if zone == self._zone:
            return
        self._zone = zone
        self._update_cursors(tdw)
        tdw.queue_draw()

    def button_press_cb(self, tdw, event):
        self._update_zone_and_cursors(tdw, event.x, event.y)
        if self._zone in (_EditZone.CREATE_FRAME, _EditZone.REMOVE_FRAME):
            button = event.button
            if button == 1 and event.type == Gdk.EventType.BUTTON_PRESS:
                self._click_info = (button, self._zone)
                return False
        return super(FrameEditMode, self).button_press_cb(tdw, event)

    def button_release_cb(self, tdw, event):
        if self._click_info:
            button0, zone0 = self._click_info
            if event.button == button0:
                if self._zone == zone0:
                    model = tdw.doc
                    if zone0 == _EditZone.REMOVE_FRAME:
                        model.set_frame_enabled(False, user_initiated=True)
                    elif zone0 == _EditZone.CREATE_FRAME:
                        self._place_new_frame(tdw, pos=(event.x, event.y))
                self._click_info = None
                self._update_zone_and_cursors(tdw, event.x, event.y)
                return False
        return super(FrameEditMode, self).button_release_cb(tdw, event)

    def motion_notify_cb(self, tdw, event):
        if self._click_info:
            return False
        if not self.in_drag:
            self._update_zone_and_cursors(tdw, event.x, event.y)
        return super(FrameEditMode, self).motion_notify_cb(tdw, event)

    def drag_start_cb(self, tdw, event):
        tdw.renderer.defer_hq_rendering(20)
        model = self.doc.model
        self._orig_frame = tuple(model.get_frame())  # independent copy
        x0, y0 = self.start_x, self.start_y
        if self._zone is None:
            # This can happen if started from another mode with a key-down
            self._zone = self._get_zone(tdw, x0, y0)
            self._update_cursors(tdw)
        self._start_model_pos = tdw.display_to_model(x0, y0)
        return super(FrameEditMode, self).drag_start_cb(tdw, event)

    def drag_stop_cb(self, tdw):
        tdw.renderer.defer_hq_rendering(0)
        return super(FrameEditMode, self).drag_stop_cb(tdw)

    def drag_update_cb(self, tdw, event, ev_x, ev_y, dx, dy):
        model = self.doc.model
        if model.frame_enabled:
            drag_effect = self.DRAG_EFFECTS.get(self._zone)
            if drag_effect:
                mx0, my0 = self._start_model_pos
                mx, my = tdw.display_to_model(ev_x, ev_y)
                fdx = int(round(mx - mx0))
                fdy = int(round(my - my0))
                mdx, mdy, mdw, mdh = drag_effect
                x, y, w, h = self._orig_frame
                x0, y0 = x, y
                if mdx:
                    x += mdx * fdx
                    if mdw == -1:  # compensating: user is dragging left edge
                        x = min(x, x0+w-self._MIN_FRAME_SIZE)
                if mdy:
                    y += mdy * fdy
                    if mdh == -1:  # compensating: user is dragging top edge
                        y = min(y, y0+h-self._MIN_FRAME_SIZE)
                if mdw:
                    w += mdw * fdx
                    w = max(w, self._MIN_FRAME_SIZE)
                if mdh:
                    h += mdh * fdy
                    h = max(h, self._MIN_FRAME_SIZE)
                self._queue_frame_change(model, (x, y, w, h))
        return super(FrameEditMode, self).drag_update_cb(
            tdw, event, ev_x, ev_y, dx, dy)

    def _queue_frame_change(self, model, new_frame):
        """Queue a frame change (that may trigger a redraw)"""
        self._queued_frame = (model, new_frame)
        if not self._change_timeout_id:
            self._change_timeout_id = GLib.timeout_add(
                interval=33.33,  # 30 fps cap
                function=self._set_queued_frame,
            )

    def _set_queued_frame(self):
        model, new_frame = self._queued_frame
        self._queued_frame = None
        if new_frame != model.get_frame():
            model.set_frame(new_frame, user_initiated=True)
        self._change_timeout_id = None

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = FrameEditOptionsWidget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET


class FrameEditOptionsWidget (Gtk.Grid):
    """An options widget for directly editing frame values"""

    def __init__(self):
        super(FrameEditOptionsWidget, self).__init__()

        from gui.application import get_app
        self.app = get_app()

        self.callbacks_active = False

        docmodel = self.app.doc.model
        x, y, w, h = docmodel.get_frame()

        dpi = docmodel.get_resolution()

        self.width_adj = UnitAdjustment(
            w, upper=32000, lower=1,
            step_increment=1, page_increment=128,
            dpi=dpi
        )
        self.height_adj = UnitAdjustment(
            h, upper=32000, lower=1,
            step_increment=1, page_increment=128,
            dpi=dpi
        )
        self.dpi_adj = Gtk.Adjustment(
            value=dpi, upper=9600, lower=1,
            step_increment=76,  # hack: 3 clicks 72->300
            page_increment=dpi)

        frame_overlays = list(filter(
            lambda o: isinstance(o, FrameOverlay),
            self.app.doc.tdw.display_overlays))
        assert len(frame_overlays) == 1, "Should be exactly 1 frame overlay!"
        self._overlay = frame_overlays[0]

        docmodel.frame_updated += self._frame_updated_cb

        self._init_ui()
        self.width_adj.connect('value-changed',
                               self.on_size_adjustment_changed)
        self.height_adj.connect('value-changed',
                                self.on_size_adjustment_changed)
        self.dpi_adj.connect('value-changed',
                             self.on_dpi_adjustment_changed)

    def _init_ui(self):

        height_label = self._new_key_label(_('Height:'))
        width_label = self._new_key_label(_('Width:'))
        dpi_label1 = self._new_key_label(_('Resolution:'))

        dpi_label2 = self._new_key_label(_('DPI'))
        dpi_label2.set_tooltip_text(
            _("Dots Per Inch (really Pixels Per Inch)")
        )

        color_label = self._new_key_label(_('Color:'))

        height_entry = Gtk.SpinButton(
            adjustment=self.height_adj,
            climb_rate=0.25,
            digits=0
        )
        height_entry.set_vexpand(False)
        height_entry.set_hexpand(True)
        self.height_adj.set_spin_button(height_entry)

        width_entry = Gtk.SpinButton(
            adjustment=self.width_adj,
            climb_rate=0.25,
            digits=0
        )
        width_entry.set_vexpand(False)
        width_entry.set_hexpand(True)
        self.width_adj.set_spin_button(width_entry)

        dpi_entry = Gtk.SpinButton(
            adjustment=self.dpi_adj,
            climb_rate=0.0,
            digits=0
        )
        dpi_entry.set_vexpand(False)
        dpi_entry.set_hexpand(True)

        color_button = Gtk.ColorButton()
        color_rgba = self.app.preferences.get("frame.color_rgba")
        color_rgba = [min(max(c, 0), 1) for c in color_rgba]
        color_gdk = uicolor.to_gdk_color(RGBColor(*color_rgba[0:3]))
        color_alpha = int(65535 * color_rgba[3])
        color_button.set_color(color_gdk)
        color_button.set_use_alpha(True)
        color_button.set_alpha(color_alpha)
        color_button.set_title(_("Frame Color"))
        color_button.connect("color-set", self._color_set_cb)

        unit_combobox = Gtk.ComboBoxText()
        for unit in sorted(UnitAdjustment.CONVERT_UNITS.keys()):
            unit_combobox.append_text(_Unit.STRINGS[unit])
        unit_combobox.set_active(_Unit.PX)
        unit_combobox.connect('changed', self.on_unit_changed)
        unit_combobox.set_hexpand(False)
        unit_combobox.set_vexpand(False)
        self._unit_combobox = unit_combobox

        # Options panel UI
        self.set_border_width(3)
        self.set_row_spacing(6)
        self.set_column_spacing(6)

        row = 0

        self.enable_button = Gtk.CheckButton()
        frame_toggle_action = self.app.find_action("FrameToggle")
        self.enable_button.set_related_action(frame_toggle_action)
        self.enable_button.set_label(_('Enabled'))
        self.attach(self.enable_button, 0, row, 3, 1)

        row += 1
        label = self._new_header_label(_("<b>Frame dimensions</b>"))
        self.attach(label, 0, row, 3, 1)

        row += 1
        self.attach(width_entry, 1, row, 1, 1)
        self.attach(unit_combobox, 2, row, 1, 1)
        self.attach(width_label, 0, row, 1, 1)

        row += 1
        self.attach(height_label, 0, row, 1, 1)
        self.attach(height_entry, 1, row, 1, 1)

        row += 1
        self.attach(dpi_label1, 0, row, 1, 1)
        self.attach(dpi_entry, 1, row, 1, 1)
        self.attach(dpi_label2, 2, row, 1, 1)

        row += 1
        self.attach(color_label, 0, row, 1, 1)
        self.attach(color_button, 1, row, 3, 1)

        crop_layer_button = Gtk.Button(label=_('Set Frame to Layer'))
        crop_layer_button.set_tooltip_text(_("Set frame to the extents of "
                                             "the current layer"))
        crop_document_button = Gtk.Button(label=_('Set Frame to Document'))
        crop_document_button.set_tooltip_text(_("Set frame to the combination "
                                                "of all layers"))
        crop_layer_button.connect('clicked', self.crop_frame_cb,
                                  'CropFrameToLayer')
        crop_document_button.connect('clicked', self.crop_frame_cb,
                                     'CropFrameToDocument')

        trim_button = Gtk.Button()
        trim_action = self.app.find_action("TrimLayer")
        trim_button.set_related_action(trim_action)
        trim_button.set_label(_('Trim Layer to Frame'))
        trim_button.set_tooltip_text(_("Trim parts of the current layer "
                                       "which lie outside the frame"))

        row += 1
        self.attach(crop_layer_button, 0, row, 3, 1)

        row += 1
        self.attach(crop_document_button, 0, row, 3, 1)

        row += 1
        self.attach(trim_button, 0, row, 3, 1)

    @classmethod
    def _new_header_label(cls, markup):
        label = Gtk.Label()
        label.set_markup(markup)
        label.set_alignment(0.0, 0.5)
        label.set_hexpand(True)
        label.set_vexpand(False)
        label.set_margin_top(18)
        label.set_margin_bottom(6)
        return label

    @classmethod
    def _new_key_label(cls, text):
        label = Gtk.Label(label=text)
        label.set_alignment(0.0, 0.5)
        label.set_hexpand(False)
        label.set_vexpand(False)
        label.set_margin_start(6)
        label.set_margin_end(6)
        return label

    def crop_frame_cb(self, button, command):
        model = self.app.doc.model
        if command == 'CropFrameToLayer':
            model.set_frame_to_current_layer(user_initiated=True)
        elif command == 'CropFrameToDocument':
            model.set_frame_to_document(user_initiated=True)

    def _color_set_cb(self, colorbutton):
        color_gdk = colorbutton.get_color()
        r, g, b = uicolor.from_gdk_color(color_gdk).get_rgb()
        a = colorbutton.get_alpha() / 65535
        self.app.preferences["frame.color_rgba"] = (r, g, b, a)
        self._overlay.redraw(color_change=True)

    def on_unit_changed(self, unit_combobox):
        active_unit = unit_combobox.get_active()
        self.width_adj.set_unit(active_unit)
        self.height_adj.set_unit(active_unit)

    def on_size_adjustment_changed(self, adjustment):
        """Update the frame size in the model."""
        if self.callbacks_active:
            return
        self.width_adj.update_px_value()
        self.height_adj.update_px_value()
        width = int(self.width_adj.get_px_value())
        height = int(self.height_adj.get_px_value())
        self.app.doc.model.update_frame(width=width, height=height,
                                        user_initiated=True)

    def on_dpi_adjustment_changed(self, adjustment):
        """Update the resolution used to calculate framesize in px."""
        if self.callbacks_active:
            return
        dpi = self.dpi_adj.get_value()
        self.width_adj.set_dpi(dpi)
        self.height_adj.set_dpi(dpi)
        self.app.doc.model.set_resolution(dpi)
        self.on_size_adjustment_changed(self.width_adj)
        self.on_size_adjustment_changed(self.height_adj)

    def _frame_updated_cb(self, model, old_frame, new_frame):
        """Update the UI to reflect the model."""
        self.callbacks_active = True  # Prevent callback loops
        self.dpi_adj.set_value(model.get_resolution())
        x, y, w, h = new_frame
        self.width_adj.set_px_value(w)
        self.height_adj.set_px_value(h)
        self.callbacks_active = False

class FrameOverlay (Overlay):
    """Overlay showing the frame, and edit boxes if in FrameEditMode

    This is a display-space overlay, since the edit boxes need to be drawn with
    pixel precision at a consistent weight regardless of zoom.

    Only the main TDW is supported."""

    OUTLINE_WIDTH = 1

    # Which edges belong to which corner, CW from top left
    _ZONE_EDGES = [sum(p) for p in pairwise(_SIDES)]

    def __init__(self, doc):
        """Initialize overlay"""
        Overlay.__init__(self)
        self.doc = doc
        self.app = doc.app
        self._trash_icon_pixbuf = None
        # Cached data used for painting and to minimize redraw areas
        self._canvas_rect = None
        self._canvas_rect_offset = None
        self._display_corners = []
        self._prev_display_corners = []
        self._trash_btn_pos = None
        # Stores per-edge invalidation rectangles,
        # indexed canonically: top, right, bottom, left
        self._prev_rectangles = [(), (), (), ()]
        self._new_rectangles = [(), (), (), ()]
        self._prev_disable_button_rectangle = None
        self._disable_button_rectangle = None

        # Calculate initial data - recalculate on frame changes
        # and view changes.
        self._recalculate_coordinates(True)
        self.app.doc.model.frame_updated += self._frame_updated_cb
        self.doc.tdw.transformation_updated += self._transformation_updated_cb

    def _frame_updated_cb(self, *args):
        self._recalculate_coordinates(True, *args)

    def _transformation_updated_cb(self, *args):
        # The redraw is already triggered at this point
        self._recalculate_coordinates(False, *args)

    def _recalculate_coordinates(self, redraw, *args):
        """Calculates geometric data that does not need updating every time"""
        # Skip calculations when the frame is not enabled (this is important
        # because otherwise all of this would be recalculated on moving,
        # scaling and rotating the canvas.
        if not (self.doc.model.frame_enabled or redraw):
            return
        tdw = self.doc.tdw
        # Canvas rectangle - regular and offset
        self._canvas_rect = Rect.new_from_gdk_rectangle(tdw.get_allocation())
        self._canvas_rect_offset = self._canvas_rect.expanded(
            self.OUTLINE_WIDTH * 4)
        # Frame corners in model coordinates
        x, y, w, h = tuple(self.doc.model.get_frame())
        corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        # Pixel-aligned frame corners in display space
        d_corners = [tdw.model_to_display(mx, my) for mx, my in corners]
        pxoffs = 0.5 if (self.OUTLINE_WIDTH % 2) else 0.0
        self._prev_display_corners = self._display_corners
        self._display_corners = tuple(
            (int(x) + pxoffs, int(y) + pxoffs)
            for x, y in d_corners)
        # Position of the button for disabling/deleting the frame
        # Placed near the center of the frame, clamped to the viewport,
        # with an offset so it does not cover visually small frames
        # (when the frame _is_ small, or when zoomed out).
        xs, ys = zip(*d_corners)
        r = gui.style.FLOATING_BUTTON_RADIUS
        tx, ty = self._canvas_rect.expanded(-2 * r).clamped_point(
            sum(xs) / 4.0, sum(ys) / 4.0)
        self._trash_btn_pos = tx, ty
        r += 6  # margin for drop shadows
        self._prev_disable_button_rectangle = self._disable_button_rectangle
        self._disable_button_rectangle = (tx - r, ty - r, r * 2, r * 2)
        # Corners
        self._zone_corners = []
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        canvas_limit = self._canvas_rect.expanded(radius)
        for i, (cx, cy) in enumerate(d_corners):
            if canvas_limit.contains_pixel(cx, cy):
                self._zone_corners.append((cx, cy, self._ZONE_EDGES[i]))
        # Intersecting frame lines & calculation of rectangles
        l_type = LineType.SEGMENT
        cx, cy, cw, ch = self._canvas_rect
        canvas_corners = (
            (cx, cy), (cx + cw, cy), (cx + cw, cy + ch), (cx, cy + ch))
        intersections = [
            intersection_of_vector_and_poly(canvas_corners, p1, p2, l_type)
            for p1, p2 in pairwise(d_corners)]

        self._prev_rectangles = self._new_rectangles
        self._new_rectangles = [(), (), (), ()]
        if intersections != [None, None, None, None]:
            self._new_rectangles = []
            m = radius + 6  # margin for handle drop shadows
            for intersection in intersections:
                if not intersection:
                    self._new_rectangles.append(())
                    continue
                (x0, y0), (x1, y1) = intersection
                w = abs(x1 - x0) + 2 * m
                h = abs(y1 - y0) + 2 * m
                x = min(x0, x1) - m
                y = min(y0, y1) - m
                self._new_rectangles.append(Rect(x, y, w, h))
        if redraw:
            self.redraw()

    def redraw(self, color_change=False):
        tdw = self.doc.tdw
        if color_change:
            tdw.queue_draw()
        else:
            prev_trash = self._prev_disable_button_rectangle
            new_trash = self._disable_button_rectangle
            if prev_trash:
                tdw.queue_draw_area(*prev_trash)
            if new_trash:
                tdw.queue_draw_area(*new_trash)
            old_corners = tuple(pairwise(self._prev_display_corners))
            new_corners = tuple(pairwise(self._display_corners))
            prev = self._prev_rectangles
            new = self._new_rectangles
            for i, (r1, r2) in enumerate(zip(prev, new)):
                if r1 == r2:
                    continue  # Skip if unchanged
                if r1 and r2:
                    r1.expand_to_include_rect(r2)
                    tdw.queue_draw_area(*r1)
                elif r1 or r2:
                    r = r1 or r2
                    corners = new_corners if r == r1 else old_corners
                    if corners:
                        (cx0, cy0), (cx1, cy1) = corners[i]
                        r.expand_to_include_point(cx0, cy0)
                        r.expand_to_include_point(cx1, cy1)
                        tdw.queue_draw_area(*r)
                    else:
                        tdw.queue_draw()
                        return

    def paint(self, cr):
        """Paints the frame, and the edit boxes if appropriate"""

        if not self.doc.model.frame_enabled:
            return

        # Frame mask: outer closed rectangle just outside the viewport
        cr.rectangle(*self._canvas_rect_offset)

        # Frame mask: inner closed rectangle
        p1, p2, p3, p4 = self._display_corners
        cr.move_to(*p1)
        cr.line_to(*p2)
        cr.line_to(*p3)
        cr.line_to(*p4)
        cr.close_path()

        # Fill the frame mask. We may need the shape again.
        frame_rgba = self.app.preferences["frame.color_rgba"]
        frame_rgba = [lib.helpers.clamp(c, 0, 1) for c in frame_rgba]
        cr.set_source_rgba(*frame_rgba)
        cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        cr.fill_preserve()

        # If the doc controller is not in a frame-editing mode, no edit
        # controls will be drawn. The frame mask drawn above normally
        # has some alpha, and may be unclear as a result. To make it
        # clearer, double-strike the edges.

        editmode = None
        for m in self.doc.modes:
            if isinstance(m, FrameEditMode):
                editmode = m
                break
        if not editmode:
            cr.set_line_width(self.OUTLINE_WIDTH)
            cr.stroke()
            return

        # Editable frame: shadows for the frame edge lines
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        zonelines = [
            (_EditZone.TOP, p1, p2),
            (_EditZone.RIGHT, p2, p3),
            (_EditZone.BOTTOM, p3, p4),
            (_EditZone.LEFT, p4, p1),
        ]
        cr.set_line_width(gui.style.DRAGGABLE_EDGE_WIDTH)
        for zone, p, q in zonelines:
            cr.move_to(*p)
            cr.line_to(*q)
        gui.drawutils.render_drop_shadow(cr, z=1)
        cr.new_path()
        for zone, p, q in zonelines:
            cr.move_to(*p)
            cr.line_to(*q)
            if editmode._zone and (editmode._zone == zone):
                rgb = gui.style.ACTIVE_ITEM_COLOR.get_rgb()
            else:
                rgb = gui.style.EDITABLE_ITEM_COLOR.get_rgb()
            cr.set_source_rgb(*rgb)
            cr.stroke()

        # Editable corners: drag handles (with hover)
        for x, y, zonemask in self._zone_corners:
            if editmode._zone and (editmode._zone == zonemask):
                col = gui.style.ACTIVE_ITEM_COLOR
            else:
                col = gui.style.EDITABLE_ITEM_COLOR
            gui.drawutils.render_round_floating_color_chip(
                cr=cr,
                x=x, y=y,
                color=col,
                radius=gui.style.DRAGGABLE_POINT_HANDLE_SIZE,
            )

        # Frame remove button position, frame center, constrained to viewport
        if editmode._zone == _EditZone.REMOVE_FRAME:
            button_color = gui.style.ACTIVE_ITEM_COLOR
        else:
            button_color = gui.style.EDITABLE_ITEM_COLOR
        bx, by = self._trash_btn_pos
        gui.drawutils.render_round_floating_button(
            cr=cr,
            x=bx, y=by,
            color=button_color,
            radius=gui.style.FLOATING_BUTTON_RADIUS,
            pixbuf=self._trash_icon(),
        )
        editmode.remove_button_pos = (bx, by)

    def _trash_icon(self):
        """Return trash icon, using cached instance if it exists"""
        if not self._trash_icon_pixbuf:
            self._trash_icon_pixbuf = gui.drawutils.load_symbolic_icon(
                icon_name="mypaint-trash-symbolic",
                size=gui.style.FLOATING_BUTTON_ICON_SIZE,
                fg=(0, 0, 0, 1),
            )
        return self._trash_icon_pixbuf


class _Unit:
    PX = 0
    IN = 1
    CM = 2
    MM = 3

    STRINGS = {
        PX: _('px'),
        IN: _('in'),
        CM: _('cm'),
        MM: _('mm'),
    }


class UnitAdjustment(Gtk.Adjustment):

    CONVERT_UNITS = {
        # {unit: (conv_factor, upper, lower, step_incr, page_incr, digits)}
        _Unit.PX: (0.0, 32000, 1, 1, 128, 0),
        _Unit.IN: (1.0, 200, 0.01, 0.01, 1, 2),
        _Unit.CM: (2.54, 500, 0.1, 0.1, 1, 1),
        _Unit.MM: (25.4, 5000, 1, 1, 10, 0),
    }

    def __init__(self, value=0, lower=0, upper=0, step_increment=0,
                 page_increment=0, page_size=0, dpi=DEFAULT_RESOLUTION):
        Gtk.Adjustment.__init__(
            self, value=value, lower=lower,
            upper=upper,
            step_increment=step_increment,
            page_increment=page_increment,
            page_size=page_size
        )
        self.px_value = value
        self.unit_value = value
        self.active_unit = _Unit.PX
        self.old_unit = _Unit.PX
        self.dpi = dpi

    def set_spin_button(self, button):
        self.spin_button = button

    def set_dpi(self, dpi):
        self.dpi = dpi

    def set_unit(self, unit):
        self.old_unit = self.active_unit
        self.active_unit = unit

        self.set_upper(UnitAdjustment.CONVERT_UNITS[unit][1])
        self.set_lower(UnitAdjustment.CONVERT_UNITS[unit][2])
        self.set_step_increment(UnitAdjustment.CONVERT_UNITS[unit][3])
        self.set_page_increment(UnitAdjustment.CONVERT_UNITS[unit][4])
        self.spin_button.set_digits(UnitAdjustment.CONVERT_UNITS[unit][5])

        self.px_value = self.convert_to_px(self.get_value(), self.old_unit)
        self.unit_value = self.convert_to_unit(self.px_value, self.active_unit)
        self.set_value(self.unit_value)

    def update_px_value(self):
        self.px_value = self.convert_to_px(self.get_value(), self.active_unit)

    def get_unit(self):
        return self.unit

    def get_unit_value(self):
        return self.unit_value

    def get_unit_value_display(self):
        """Get the per-unut value, rounded appropriately for display"""
        unit = self.active_unit
        digits = self.CONVERT_UNITS[unit][5]
        return round(self.unit_value, digits)

    def set_px_value(self, value):
        self.px_value = value
        self.unit_value = self.convert_to_unit(value, self.active_unit)
        self.set_value(self.unit_value)

    def get_px_value(self):
        self.px_value = self.convert_to_px(self.get_value(), self.active_unit)
        return self.px_value

    def convert(self, value, unit_from, unit_to):
        px = self.convert_to_px(value, unit_from)
        uvalue = self.convert_to_unit(px, unit_to)
        return uvalue

    def convert_to_px(self, value, unit):
        if unit == _Unit.PX:
            return value
        return value / UnitAdjustment.CONVERT_UNITS[unit][0] * self.dpi

    def convert_to_unit(self, px, unit):
        if unit == _Unit.PX:
            return px
        return px * UnitAdjustment.CONVERT_UNITS[unit][0] / self.dpi
