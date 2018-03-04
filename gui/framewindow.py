# This file is part of MyPaint.
# Copyright (C) 2010-2018 by the MyPaint Development Team
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

from gi.repository import Gtk
from gi.repository import Gdk
from gettext import gettext as _
import cairo

from . import windowing
import gui.mode
from lib.color import RGBColor
from . import uicolor
from .overlays import Overlay
import lib.helpers
from lib.document import DEFAULT_RESOLUTION
import gui.cursor
import gui.style


## Class defs


class _EditZone:
    INSIDE = 0x00
    LEFT = 0x01
    RIGHT = 0x02
    TOP = 0x04
    BOTTOM = 0x08
    OUTSIDE = 0x10
    REMOVE_FRAME = 0x20
    CREATE_FRAME = 0x30


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
    permitted_switch_actions = set([
        'ShowPopupMenu',
        'RotateViewMode',
        'ZoomViewMode',
        'PanViewMode',
    ])

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

        :param gui.tileddrawwidget.TiledDrawWidget tdw: canvas widget
        :param tuple pos: position of the frame centre, display (x, y)

        The existing frame position is discarded, and a new position is
        chosen for the frame so that as many of its edges are as visible
        as possible.

        """
        model = tdw.doc
        alloc = tdw.get_allocation()
        if not pos:
            pos = (
                alloc.width * 0.5 + alloc.x,
                alloc.height * 0.5 + alloc.y,
            )
        frame_center = tdw.display_to_model(*pos)
        frame_size = None
        corners_disp = (
            (alloc.x, alloc.y),
            (alloc.x + alloc.width, alloc.y),
            (alloc.x + alloc.width, alloc.y + alloc.height),
            (alloc.x, alloc.y + alloc.height),
        )
        for corner_disp in corners_disp:
            corner = tdw.display_to_model(*corner_disp)
            dist = math.hypot(corner[0] - frame_center[0],
                              corner[1] - frame_center[1])
            if frame_size is None or dist < frame_size:
                frame_size = dist
        # frame_size /= 2.0
        frame_size = int(round(max(frame_size, 100)))
        w = frame_size
        h = frame_size
        x = int(round(frame_center[0] - frame_size/2.0))
        y = int(round(frame_center[1] - frame_size/2.0))
        model.set_frame([x, y, w, h], user_initiated=True)

    def leave(self, **kwds):
        """Exit the mode, hiding any dialogs"""
        dialog = self.get_options_widget()._size_dialog
        if self.doc:
            self.doc.tdw.queue_draw()
            if self not in self.doc.modes:
                dialog.hide()
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
        model = self.doc.model
        self._orig_frame = tuple(model.get_frame())  # independent copy
        x0, y0 = self.start_x, self.start_y
        if self._zone is None:
            # This can happen if started from another mode with a key-down
            self._zone = self._get_zone(tdw, x0, y0)
            self._update_cursors(tdw)
        self._start_model_pos = tdw.display_to_model(x0, y0)
        return super(FrameEditMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, dx, dy):
        model = self.doc.model
        if model.frame_enabled:
            mx0, my0 = self._start_model_pos
            mx, my = tdw.display_to_model(event.x, event.y)
            fdx = int(round(mx - mx0))
            fdy = int(round(my - my0))
            drag_effect = self.DRAG_EFFECTS.get(self._zone)
            if drag_effect:
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
                new_frame = (x, y, w, h)
                if new_frame != model.get_frame():
                    model.set_frame(new_frame, user_initiated=True)
        return super(FrameEditMode, self).drag_update_cb(tdw, event, dx, dy)

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = FrameEditOptionsWidget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET


class FrameEditOptionsWidget (Gtk.Alignment):
    """An options widget for directly editing frame values"""

    def __init__(self):
        super(FrameEditOptionsWidget, self).__init__(
            xalign=0.5,
            yalign=0.5,
            xscale=1.0,
            yscale=1.0,
        )

        from gui.application import get_app
        self.app = get_app()

        self.callbacks_active = False

        docmodel = self.app.doc.model
        x, y, w, h = docmodel.get_frame()

        dpi = docmodel.get_resolution()

        self.width_adj = UnitAdjustment(
            w, upper=32000, lower=1,
            step_incr=1, page_incr=128,
            dpi=dpi
        )
        self.height_adj = UnitAdjustment(
            h, upper=32000, lower=1,
            step_incr=1, page_incr=128,
            dpi=dpi
        )
        self.dpi_adj = Gtk.Adjustment(dpi, upper=9600, lower=1,
                                      step_incr=76,  # hack: 3 clicks 72->300
                                      page_incr=dpi)

        docmodel.frame_updated += self._frame_updated_cb

        self._init_ui()
        self.width_adj.connect('value-changed',
                               self.on_size_adjustment_changed)
        self.height_adj.connect('value-changed',
                                self.on_size_adjustment_changed)
        self.dpi_adj.connect('value-changed',
                             self.on_dpi_adjustment_changed)

        self._update_size_button()

    def _init_ui(self):
        # Dialog for editing dimensions (width, height, DPI)
        app = self.app
        buttons = (Gtk.STOCK_OK, Gtk.ResponseType.ACCEPT)
        self._size_dialog = windowing.Dialog(
            app, _("Frame Size"), app.drawWindow,
            buttons=buttons
        )
        unit = _('px')

        height_label = self._new_key_label(_('Height:'))
        width_label = self._new_key_label(_('Width:'))
        dpi_label1 = self._new_key_label(_('Resolution:'))

        dpi_label2 = Gtk.Label(label=_('DPI'))
        dpi_label2.set_alignment(0.0, 0.5)
        dpi_label2.set_hexpand(False)
        dpi_label2.set_vexpand(False)
        dpi_label2.set_tooltip_text(
            _("Dots Per Inch (really Pixels Per Inch)")
        )

        color_label = Gtk.Label(label=_('Color:'))
        color_label.set_alignment(0.0, 0.5)

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
        color_align = Gtk.Alignment.new(0, 0.5, 0, 0)
        color_align.add(color_button)

        size_grid = Gtk.Grid()
        size_grid.set_border_width(12)

        size_grid.set_row_spacing(6)
        size_grid.set_column_spacing(6)

        unit_combobox = Gtk.ComboBoxText()
        for unit in UnitAdjustment.CONVERT_UNITS.keys():
            unit_combobox.append_text(unit)
        for i, key in enumerate(UnitAdjustment.CONVERT_UNITS):
            if key == _('px'):
                unit_combobox.set_active(i)
        unit_combobox.connect('changed', self.on_unit_changed)
        unit_combobox.set_hexpand(False)
        unit_combobox.set_vexpand(False)
        self._unit_combobox = unit_combobox

        row = 0
        label = self._new_header_label(_("<b>Frame dimensions</b>"))
        label.set_margin_top(0)
        size_grid.attach(label, 0, row, 3, 1)

        row += 1
        size_grid.attach(width_label, 0, row, 1, 1)
        size_grid.attach(width_entry, 1, row, 1, 1)
        size_grid.attach(unit_combobox, 2, row, 1, 1)

        row += 1
        size_grid.attach(height_label, 0, row, 1, 1)
        size_grid.attach(height_entry, 1, row, 1, 1)

        row += 1
        label = self._new_header_label(_("<b>Pixel density</b>"))
        size_grid.attach(label, 0, row, 3, 1)

        row += 1
        size_grid.attach(dpi_label1, 0, row, 1, 1)
        size_grid.attach(dpi_entry, 1, row, 1, 1)
        size_grid.attach(dpi_label2, 2, row, 1, 1)

        # Options panel UI
        opts_table = Gtk.Table(3, 3)
        opts_table.set_border_width(3)
        xopts = Gtk.AttachOptions.FILL | Gtk.AttachOptions.EXPAND
        yopts = Gtk.AttachOptions.FILL
        xpad = ypad = 3

        row = 0
        size_button = Gtk.Button("<size-summary>")
        self._size_button = size_button
        size_button.connect("clicked", self._size_button_clicked_cb)
        opts_table.attach(size_button, 0, 2, row, row+1,
                          xopts, yopts, xpad, ypad)

        row += 1
        opts_table.attach(color_label, 0, 1, row, row+1,
                          xopts, yopts, xpad, ypad)
        opts_table.attach(color_align, 1, 2, row, row+1,
                          xopts, yopts, xpad, ypad)

        crop_layer_button = Gtk.Button(_('Set Frame to Layer'))
        crop_layer_button.set_tooltip_text(_("Set frame to the extents of "
                                             "the current layer"))
        crop_document_button = Gtk.Button(_('Set Frame to Document'))
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

        self.enable_button = Gtk.CheckButton()
        frame_toggle_action = self.app.find_action("FrameToggle")
        self.enable_button.set_related_action(frame_toggle_action)
        self.enable_button.set_label(_('Enabled'))

        row += 1
        opts_table.attach(self.enable_button, 1, 2, row, row+1,
                          xopts, yopts, xpad, ypad)

        row += 1
        opts_table.attach(crop_layer_button, 0, 2, row, row+1,
                          xopts, yopts, xpad, ypad)

        row += 1
        opts_table.attach(crop_document_button, 0, 2, row, row+1,
                          xopts, yopts, xpad, ypad)

        row += 1
        opts_table.attach(trim_button, 0, 2, row, row+1,
                          xopts, yopts, xpad, ypad)

        content_area = self._size_dialog.get_content_area()
        content_area.pack_start(size_grid, True, True, 0)

        self._size_dialog.connect('response', self._size_dialog_response_cb)

        self.add(opts_table)

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
        label.set_margin_left(6)   # FIXME: Use margin-start etc. when 3.10
        label.set_margin_right(6)  # FIXME: support can be dropped.
        return label

    def _size_dialog_response_cb(self, dialog, response_id):
        if response_id == Gtk.ResponseType.ACCEPT:
            dialog.hide()

    def get_unit_text(self):
        combobox = self._unit_combobox
        model = combobox.get_model()
        active = combobox.get_active()
        if active < 0:
            return None
        return model[active][0]

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
        self.app.doc.tdw.queue_draw()

    def on_unit_changed(self, unit_combobox):
        active_unit = self.get_unit_text()
        self.width_adj.set_unit(active_unit)
        self.height_adj.set_unit(active_unit)
        self._update_size_button()

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
        dpi = model.get_resolution()
        self.dpi_adj.set_value(dpi)
        x, y, w, h = new_frame
        self.width_adj.set_px_value(w)
        self.height_adj.set_px_value(h)
        self._update_size_button()
        self.callbacks_active = False

    def _update_size_button(self):
        text = _(u"{width:g}\u00D7{height:g} {units}").format(
            width=self.width_adj.get_unit_value_display(),
            height=self.height_adj.get_unit_value_display(),
            units=self.get_unit_text(),
        )
        self._size_button.set_label(text)

    def _size_button_clicked_cb(self, button):
        self._size_dialog.show_all()


class FrameOverlay (Overlay):
    """Overlay showing the frame, and edit boxes if in FrameEditMode

    This is a display-space overlay, since the edit boxes need to be drawn with
    pixel precision at a consistent weight regardless of zoom.

    Only the main TDW is supported."""

    OUTLINE_WIDTH = 1

    def __init__(self, doc):
        """Initialize overlay"""
        Overlay.__init__(self)
        self.doc = doc
        self.app = doc.app
        self._trash_icon_pixbuf = None

    def _frame_corners(self):
        """Calculates the frame's corners, in display space"""
        tdw = self.doc.tdw
        x, y, w, h = tuple(self.doc.model.get_frame())
        points_model = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        points_disp = [tdw.model_to_display(*p) for p in points_model]
        return points_disp

    def paint(self, cr):
        """Paints the frame, and the edit boxes if appropriate"""

        if not self.doc.model.frame_enabled:
            return

        # Frame mask: outer closed rectangle just outside the viewport
        tdw = self.doc.tdw
        alloc = tdw.get_allocation()
        w, h = alloc.width, alloc.height
        canvas_bbox = (
            -5 * self.OUTLINE_WIDTH,
            -5 * self.OUTLINE_WIDTH,
            w + 5 * self.OUTLINE_WIDTH,
            h + 5 * self.OUTLINE_WIDTH,
        )
        cr.rectangle(*canvas_bbox)
        view_x0, view_y0 = alloc.x, alloc.y
        view_x1, view_y1 = view_x0+alloc.width, view_y0+alloc.height

        # Frame mask: inner closed rectangle
        corners = self._frame_corners()
        pxoff = 0.5 if (self.OUTLINE_WIDTH % 2) else 0.0
        p1, p2, p3, p4 = [(int(x)+pxoff, int(y)+pxoff) for x, y in corners]
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
        for m in reversed(list(self.doc.modes)):
            if isinstance(m, FrameEditMode):
                editmode = m
                break
        if not editmode:
            cr.set_line_width(self.OUTLINE_WIDTH)
            cr.stroke()
            return

        # Editable frame: shadows for the frame edge lines
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        edge_width = gui.style.DRAGGABLE_EDGE_WIDTH
        pxoff = 0.5 if (edge_width % 2) else 0.0
        p1, p2, p3, p4 = [(int(x)+pxoff, int(y)+pxoff) for x, y in corners]
        zonelines = [
            (_EditZone.TOP, p1, p2),
            (_EditZone.RIGHT, p2, p3),
            (_EditZone.BOTTOM, p3, p4),
            (_EditZone.LEFT, p4, p1),
        ]
        cr.set_line_width(edge_width)
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
        zonecorners = [
            (p1, _EditZone.TOP + _EditZone.LEFT),
            (p2, _EditZone.TOP + _EditZone.RIGHT),
            (p3, _EditZone.BOTTOM + _EditZone.RIGHT),
            (p4, _EditZone.BOTTOM + _EditZone.LEFT)
        ]
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        for p, zonemask in zonecorners:
            x, y = p
            if editmode._zone and (editmode._zone == zonemask):
                col = gui.style.ACTIVE_ITEM_COLOR
            else:
                col = gui.style.EDITABLE_ITEM_COLOR
            gui.drawutils.render_round_floating_color_chip(
                cr=cr,
                x=x, y=y,
                color=col,
                radius=radius,
            )

        # Frame remove button
        p_center = [(p1[i]+p2[i]+p3[i]+p4[i])/4.0 for i in (0, 1)]
        y_lowest = p_center[1]
        for p, zone in zonecorners:
            y = p[1]
            if y > y_lowest:
                y_lowest = y
        y_lowest += 2 * gui.style.FLOATING_BUTTON_RADIUS
        button_pos = (p_center[0], y_lowest)

        # Constrain the position so that it appears within the viewport
        margin = 2 * gui.style.FLOATING_BUTTON_RADIUS
        button_pos = [
            lib.helpers.clamp(
                button_pos[0],
                view_x0 + margin,
                view_x1 - margin,
            ),
            lib.helpers.clamp(
                button_pos[1],
                view_y0 + margin,
                view_y1 - margin,
            ),
        ]

        if not self._trash_icon_pixbuf:
            self._trash_icon_pixbuf = gui.drawutils.load_symbolic_icon(
                icon_name="mypaint-trash-symbolic",
                size=gui.style.FLOATING_BUTTON_ICON_SIZE,
                fg=(0, 0, 0, 1),
            )
        icon_pixbuf = self._trash_icon_pixbuf

        if editmode._zone == _EditZone.REMOVE_FRAME:
            button_color = gui.style.ACTIVE_ITEM_COLOR
        else:
            button_color = gui.style.EDITABLE_ITEM_COLOR

        gui.drawutils.render_round_floating_button(
            cr=cr,
            x=button_pos[0],
            y=button_pos[1],
            color=button_color,
            radius=gui.style.FLOATING_BUTTON_RADIUS,
            pixbuf=icon_pixbuf,
        )
        editmode.remove_button_pos = button_pos


class UnitAdjustment(Gtk.Adjustment):

    CONVERT_UNITS = {
        # {unit: (conv_factor, upper, lower, step_incr, page_incr, digits)}
        _('px'): (0.0, 32000, 1, 1, 128, 0),
        _('inch'): (1.0, 200, 0.01, 0.01, 1, 2),
        _('cm'): (2.54, 500, 0.1, 0.1, 1, 1),
        _('mm'): (25.4, 5000, 1, 1, 10, 0),
    }

    def __init__(self, value=0, lower=0, upper=0, step_incr=0,
                 page_incr=0, page_size=0, dpi=DEFAULT_RESOLUTION):
        Gtk.Adjustment.__init__(self, value, lower, upper, step_incr,
                                page_incr, page_size)
        self.px_value = value
        self.unit_value = value
        self.active_unit = _('px')
        self.old_unit = _('px')
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
        if unit == _('px'):
            return value
        return value / UnitAdjustment.CONVERT_UNITS[unit][0] * self.dpi

    def convert_to_unit(self, px, unit):
        if unit == _('px'):
            return px
        return px * UnitAdjustment.CONVERT_UNITS[unit][0] / self.dpi
