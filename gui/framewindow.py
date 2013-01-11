# This file is part of MyPaint.
# Copyright (C) 2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Frame manipulation mode and subwindow
"""

import math

import pygtkcompat
import gtk
from gtk import gdk
from gettext import gettext as _

import canvasevent
import linemode
import windowing
from colors.uicolor import RGBColor


class FrameEditMode (canvasevent.SwitchableModeMixin,
                     canvasevent.SpringLoadedDragMode,
                     canvasevent.ScrollableModeMixin,
                     canvasevent.OneshotDragModeMixin):
    """Stackable interaction mode for editing the document frame.

    The frame editing mode has an associated details dialog which is open for
    the length of time the mode is active.

    """

    # Class-level configuration
    __action_name__ = 'FrameEditMode'

    # These will be overridden on enter()
    inactive_cursor = None
    active_cursor = None

    unmodified_persist = True
    permitted_switch_actions = set([
            'ShowPopupMenu', 'RotateViewMode', 'ZoomViewMode', 'PanViewMode',
        ])

    # Hit zones
    INSIDE  = 0x00
    LEFT    = 0x01
    RIGHT   = 0x02
    TOP     = 0x04
    BOTTOM  = 0x08
    OUTSIDE = 0x10

    EDGE_WIDTH = 20  # pixels

    # Dragging interpretation by hit zone
    DRAG_EFFECTS = {
        #               (dx, dy, dw, dh)
        LEFT + TOP:     (+1, +1, -1, -1),
        LEFT:           (+1,  0, -1,  0),
        LEFT + BOTTOM:  (+1,  0, -1, +1),
        TOP:            ( 0, +1,  0, -1),
        INSIDE:         (+1, +1,  0,  0),
        BOTTOM:         ( 0,  0,  0, +1),
        RIGHT + TOP:    ( 0, +1, +1, -1),
        RIGHT:          ( 0,  0, +1,  0),
        RIGHT + BOTTOM: ( 0,  0, +1, +1),
        OUTSIDE:        ( 0,  0,  0,  0),
        }


    def stackable_on(self, mode):
        # Any drawing mode
        return isinstance(mode, linemode.LineModeBase) \
            or isinstance(mode, canvasevent.SwitchableFreehandMode)


    def __init__(self, **kwds):
        """Initialize."""

        super(FrameEditMode, self).__init__(**kwds)
        self._zone = None
        self._orig_frame = None


    def enter(self, **kwds):
        """Enter the mode, showing the dialog unless modifiers are held.

        If modifiers are held down when the mode is entered, its oneshot
        behaviour kicks in. Oneshot frame adjustment modes do not show the
        details dialog.

        """
        super(FrameEditMode, self).enter(**kwds)
        if not self.initial_modifiers:
            lm = self.doc.app.layout_manager
            dialog = lm.get_subwindow_by_role("frameWindow")
            dialog.show_all()
        self.cursor_move_w_e = self.doc.app.cursors.get_action_cursor(
            self.__action_name__, "cursor_move_w_e")
        self.cursor_move_n_s = self.doc.app.cursors.get_action_cursor(
            self.__action_name__, "cursor_move_n_s")
        self.cursor_move_nw_se = self.doc.app.cursors.get_action_cursor(
            self.__action_name__, "cursor_move_nw_se")
        self.cursor_move_ne_sw = self.doc.app.cursors.get_action_cursor(
            self.__action_name__, "cursor_move_ne_sw")
        self.cursor_hand_closed = self.doc.app.cursors.get_action_cursor(
            self.__action_name__, "cursor_hand_closed")
        self.cursor_hand_open = self.doc.app.cursors.get_action_cursor(
            self.__action_name__, "cursor_hand_open")
        self.cursor_forbidden = self.doc.app.cursors.get_action_cursor(
            self.__action_name__, "cursor_arrow_forbidden")

    def leave(self, **kwds):
        """Exit the mode, hiding any dialogs.
        """
        if not self.initial_modifiers:
            lm = self.doc.app.layout_manager
            dialog = lm.get_subwindow_by_role("frameWindow")
            dialog.hide()
        super(FrameEditMode, self).leave(**kwds)


    def _get_zone(self, tdw, xd, yd):
        model = self.doc.model
        x, y = tdw.display_to_model(xd, yd)
        fx1, fy1, fw, fh = model.get_frame()
        fx2, fy2 = fx1+fw, fy1+fh
        fx1d, fy1d = tdw.model_to_display(fx1, fy1)
        fx2d, fy2d = tdw.model_to_display(fx2, fy2)

        # Calculate the maximum permissible distance
        dx1, dy1 = tdw.display_to_model(0, 0)
        dx2, dy2 = tdw.display_to_model(0, self.EDGE_WIDTH)
        max_d = math.sqrt((dx1-dx2)**2 + (dy1-dy2)**2)

        zone = self.INSIDE  # zero
        if x <= fx1:
            if abs(fx1 - x) <= max_d: zone |= self.LEFT
            else: return self.OUTSIDE
        elif x >= fx2:
            if abs(fx2 - x) <= max_d: zone |= self.RIGHT
            else: return self.OUTSIDE
        if y <= fy1:
            if abs(fy1 - y) <= max_d: zone |= self.TOP
            else: return self.OUTSIDE
        elif y >= fy2:
            if abs(fy2 - y) <= max_d: zone |= self.BOTTOM
            else: return self.OUTSIDE
        return zone


    def _update_cursors(self, tdw, xd, yd):
        model = self.doc.model
        if not model.frame_enabled:
            self.active_cursor = self.cursor_forbidden
            self.inactive_cursor = self.cursor_forbidden
            return

        # Simpler interpretations
        zone = self._get_zone(tdw, xd, yd)
        if zone == self.OUTSIDE:
            self.active_cursor = self.cursor_forbidden
            self.inactive_cursor = self.cursor_forbidden
            return
        elif zone == self.INSIDE:
            self.active_cursor = self.cursor_hand_closed
            self.inactive_cursor = self.cursor_hand_open
            return

        # Centre of frame, in display coordinates
        fx, fy, fw, fh = model.get_frame()
        cx, cy = fx+(fw/2.0), fy+(fh/2.0)
        cxd, cyd = tdw.model_to_display(cx, cy)

        # A reference point, reflecting the side or edge where the pointer is
        rx, ry = cx, cy
        if zone & self.LEFT: rx = fx
        elif zone & self.RIGHT: rx = fx+fw
        if zone & self.TOP: ry = fy
        elif zone & self.BOTTOM: ry = fy+fh
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
        cursors = [ (1, self.cursor_move_w_e),     # right side
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
                return

        # This should never happen.
        self.cursor = gdk.Cursor(gdk.BOGOSITY)


    def motion_notify_cb(self, tdw, event):
        if not self.in_drag:
            self._update_cursors(tdw, event.x, event.y)
            tdw.set_override_cursor(self.inactive_cursor)
            self._zone = self._get_zone(tdw, event.x, event.y)
        return super(FrameEditMode, self).motion_notify_cb(tdw, event)


    def drag_start_cb(self, tdw, event):
        model = self.doc.model
        self._orig_frame = model.get_frame()
        if self._zone is None:
            # This can happen if started from another mode with a key-down
            self._zone = self._get_zone(tdw, self.start_x, self.start_y)
        return super(FrameEditMode, self).drag_start_cb(tdw, event)


    def drag_update_cb(self, tdw, event, dx, dy):
        model = self.doc.model
        if model.frame_enabled:
            x, y = float(self.last_x), float(self.last_y)
            x0, y0 = tdw.display_to_model(x, y)
            x1, y1 = tdw.display_to_model(x+dx, y+dy)
            fdx = int(x1 - x0)
            fdy = int(y1 - y0)

            mdx, mdy, mdw, mdh = self.DRAG_EFFECTS[self._zone]
            x, y, w, h = frame = self._orig_frame
            x += mdx*fdx
            y += mdy*fdy
            w += mdw*fdx
            h += mdh*fdy
            new_frame = (x, y, w, h)
            if new_frame != frame:
                if w > 0 and h > 0:
                    model.set_frame(*new_frame)
        return super(FrameEditMode, self).drag_update_cb(tdw, event, dx, dy)


class Window (windowing.Dialog):
    """A dialog window for directly editing frame values.
    """

    def __init__(self, app):
        buttons = (gtk.STOCK_OK, gtk.RESPONSE_ACCEPT)
        windowing.Dialog.__init__(self, app, _("Frame"), app.drawWindow, buttons=buttons)

        self.callbacks_active = False

        x, y, w, h = self.app.doc.model.get_frame()

        self.width_adj  = UnitAdjustment(w, upper=32000, lower=1, step_incr=1, page_incr=128)
        self.height_adj = UnitAdjustment(h, upper=32000, lower=1, step_incr=1, page_incr=128)
        self.dpi_adj = gtk.Adjustment(300, upper=9600, lower = 1, step_incr = 20, page_incr=300)
        self.unit_label = gtk.Label(_('px'))
        self.unit_label.set_alignment(0, 0.5)

        self.width_adj.connect('value-changed', self.on_size_adjustment_changed)
        self.height_adj.connect('value-changed', self.on_size_adjustment_changed)
        self.dpi_adj.connect('value-changed', self.on_dpi_adjustment_changed)

        self.app.doc.model.frame_observers.append(self.on_frame_changed)

        self._init_ui()

    def _init_ui(self):
        unit = _('px')
        height_label = gtk.Label(_('Height:'))
        height_label.set_alignment(0.0, 0.5)
        width_label = gtk.Label(_('Width:'))
        width_label.set_alignment(0.0, 0.5)
        dpi_label = gtk.Label(_('Resolution:'))
        dpi_label.set_alignment(0.0, 0.5)
        color_label = gtk.Label(_('Color:'))
        color_label.set_alignment(0.0, 0.5)

        height_entry = gtk.SpinButton()
        height_entry.set_adjustment(self.height_adj)
        self.height_adj.set_spin_button(height_entry)
        width_entry = gtk.SpinButton()
        width_entry.set_adjustment(self.width_adj)
        self.width_adj.set_spin_button(width_entry)
        dpi_entry = gtk.SpinButton()
        dpi_entry.set_adjustment(self.dpi_adj)

        color_button = gtk.ColorButton()
        color_rgba = self.app.preferences.get("frame.color_rgba")
        color_rgba = [min(max(c, 0), 1) for c in color_rgba]
        color_gdk = RGBColor(*color_rgba[0:3]).to_gdk_color()
        color_alpha = int(65535 * color_rgba[3])
        color_button.set_color(color_gdk)
        color_button.set_use_alpha(True)
        color_button.set_alpha(color_alpha)
        color_button.set_title(_("Frame Color"))
        color_button.connect("color-set", self._color_set_cb)
        color_align = gtk.Alignment(0, 0.5, 0, 0)
        color_align.add(color_button)

        size_table = gtk.Table(6, 3)
        size_table.set_border_width(9)
        xopts = yopts = gtk.FILL|gtk.EXPAND
        xpad = ypad = 3

        if pygtkcompat.USE_GTK3:
            unit_combobox = gtk.ComboBoxText()
        else:
            unit_combobox = gtk.combo_box_new_text()
        for unit in UnitAdjustment.CONVERT_UNITS.keys():
            unit_combobox.append_text(unit)

        for i, key in enumerate(UnitAdjustment.CONVERT_UNITS):
            if key == _('px'):
                unit_combobox.set_active(i)

        unit_combobox.connect('changed', self.on_unit_changed)
        row = 0
        size_table.attach(width_label, 0, 1, row, row+1,
                          xopts, yopts, xpad, ypad)
        size_table.attach(width_entry, 1, 2, row, row+1,
                          xopts, yopts, xpad, ypad)
        size_table.attach(self.unit_label, 2, 3, row, row+1,
                          xopts, yopts, xpad + 4, ypad)

        row += 1
        size_table.attach(height_label, 0, 1, row, row+1,
                          xopts, yopts, xpad, ypad)
        size_table.attach(height_entry, 1, 2, row, row+1,
                          xopts, yopts, xpad, ypad)
        size_table.attach(unit_combobox, 2, 3, row, row+1,
                          xopts, yopts, xpad, ypad)

        row += 1
        size_table.attach(dpi_label, 0, 1, row, row+1,
                          xopts, yopts, xpad, ypad)
        size_table.attach(dpi_entry, 1, 2, row, row+1,
                          xopts, yopts, xpad, ypad)


        row += 1
        size_table.attach(color_label, 0, 1, row, row+1,
                          xopts, yopts, xpad, ypad)
        size_table.attach(color_align, 1, 2, row, row+1,
                          xopts, yopts, xpad, ypad)


        crop_layer_button = gtk.Button(_('Crop to Current Layer'))
        crop_layer_button.set_tooltip_text(_("Crop frame to the currently "
                                             "active layer"))
        crop_document_button = gtk.Button(_('Crop to Document'))
        crop_document_button.set_tooltip_text(_("Crop frame to the combination "
                                                "of all layers"))
        crop_layer_button.connect('clicked', self.crop_frame_cb,
                                  'CropFrameToLayer')
        crop_document_button.connect('clicked', self.crop_frame_cb,
                                     'CropFrameToDocument')

        hint_label = gtk.Label(_('While the frame is enabled, it \n'
                                 'can be adjusted on the canvas'))
        hint_label.set_padding(0, 6)

        self.enable_button = gtk.CheckButton()
        frame_toggle_action = self.app.find_action("FrameToggle")
        self.enable_button.set_related_action(frame_toggle_action)
        self.enable_button.set_label(_('Enabled'))

        row += 1
        size_table.attach(self.enable_button, 2, 3, row, row+1,
                          xopts, yopts, xpad, ypad+6)

        row += 1
        size_table.attach(hint_label, 0, 3, row, row+1,
                          xopts, yopts, xpad, ypad)

        row += 1
        size_table.attach(crop_layer_button, 0, 3, row, row+1,
                          xopts, yopts, xpad, ypad)

        row += 1
        size_table.attach(crop_document_button, 0, 3, row, row+1,
                          xopts, yopts, xpad, ypad)

        content_area = self.get_content_area()
        content_area.pack_start(size_table, True, True)
        self.connect('response', self.on_response)

    def _leave_mode(self, *a):
        if isinstance(self.app.doc.modes.top, FrameEditMode):
            self.app.doc.modes.pop()

    def on_response(self, dialog, response_id):
        self._leave_mode()
        if response_id == gtk.RESPONSE_ACCEPT:
            self.hide()

    def get_active_text(self, combobox):
        model = combobox.get_model()
        active = combobox.get_active()
        if active < 0:
            return None
        return model[active][0]

    # FRAME
    def crop_frame_cb(self, button, command):
        if command == 'CropFrameToLayer':
            bbox = self.app.doc.model.get_current_layer().get_bbox()
        elif command == 'CropFrameToDocument':
            bbox = self.app.doc.model.get_bbox()
        else: assert 0
        self.app.doc.model.set_frame_enabled(True)
        self.app.doc.model.set_frame(*bbox)
        self.width_adj.set_px_value(bbox.w)
        self.height_adj.set_px_value(bbox.h)

    def _color_set_cb(self, colorbutton):
        color_gdk = colorbutton.get_color()
        r,g,b = RGBColor.new_from_gdk_color(color_gdk).get_rgb()
        a = float(colorbutton.get_alpha()) / 65535
        self.app.preferences["frame.color_rgba"] = (r, g, b, a)
        self.app.doc.tdw.queue_draw()

    def on_unit_changed(self, unit_combobox):
        active_unit = self.get_active_text(unit_combobox)
        self.width_adj.set_unit(active_unit)
        self.height_adj.set_unit(active_unit)
        self.unit_label.set_text(active_unit)

    def on_size_adjustment_changed(self, adjustment):
        """Update the frame size in the model."""
        if self.callbacks_active:
            return
        self.width_adj.update_px_value()
        self.height_adj.update_px_value()
        width = int(self.width_adj.get_px_value())
        height = int(self.height_adj.get_px_value())
        self.app.doc.model.set_frame(width=width, height=height)

    def on_dpi_adjustment_changed(self, adjustment):
        """Update the resolution used to calculate framesize in px."""
        if self.callbacks_active:
            return
        dpi = self.dpi_adj.get_value()
        self.width_adj.set_dpi(dpi)
        self.height_adj.set_dpi(dpi)
        self.on_size_adjustment_changed(self.width_adj)
        self.on_size_adjustment_changed(self.height_adj)

    def on_frame_changed(self):
        """Update the UI to reflect the model."""
        self.callbacks_active = True # Prevent callback loops
        x, y, w, h = self.app.doc.model.get_frame()
        self.width_adj.set_px_value(w)
        self.height_adj.set_px_value(h)
        self.callbacks_active = False

class UnitAdjustment(gtk.Adjustment):

    CONVERT_UNITS = { # unit :  (conversion_factor, upper, lower, step_incr, page_incr, digits)
                      _('px') :   (0.0,     32000,     1,       1, 128, 0),
                      _('inch') : (1.0,     200,    0.01,    0.01,   1, 2),
                      _('cm') :   (2.54,    500,     0.1,     0.1,   1, 1),
                      _('mm') :   (25.4,    5000,      1,       1,  10, 0),
                    }


    def __init__(self, value=0, lower=0, upper=0, step_incr=0, page_incr=0, page_size=0, dpi=300):
        gtk.Adjustment.__init__(self, value, lower, upper, step_incr, page_incr, page_size)
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

    def set_px_value(self, value):
        self.px_value = value
        self.set_value(self.convert_to_unit(value, self.active_unit))

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
