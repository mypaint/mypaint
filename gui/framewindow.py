# This file is part of MyPaint.
# Copyright (C) 2012-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Frame manipulation mode and subwindow"""

## Imports

import math

import gtk2compat
import glib
import gtk
from gtk import gdk
from gettext import gettext as _
import cairo

import windowing
import gui.mode
from colors.uicolor import RGBColor
from overlays import Overlay
from lib import helpers
from lib.document import DEFAULT_RESOLUTION
import gui.cursor
import gui.style


## Class defs


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

    # Whether or not it's the first time the user has activated this tool.
    # This is needed as otherwise the frame will be enabled once the user
    # stops panning/rotating/etc, regardless of whether it was before.
    first_time = True

    unmodified_persist = True
    permitted_switch_actions = set([
        'ShowPopupMenu',
        'RotateViewMode',
        'ZoomViewMode',
        'PanViewMode',
    ])

    # Hit zones
    INSIDE = 0x00
    LEFT = 0x01
    RIGHT = 0x02
    TOP = 0x04
    BOTTOM = 0x08
    OUTSIDE = 0x10

    EDGE_SENSITIVITY = 10  # pixels

    _MIN_FRAME_SIZE = 1

    # Dragging interpretation by hit zone
    DRAG_EFFECTS = {
        #               (dx, dy, dw, dh)
        LEFT + TOP:     (+1, +1, -1, -1),
        LEFT:           (+1,  0, -1,  0),
        LEFT + BOTTOM:  (+1,  0, -1, +1),
        TOP:             (0, +1,  0, -1),
        INSIDE:         (+1, +1,  0,  0),
        BOTTOM:          (0,  0,  0, +1),
        RIGHT + TOP:     (0, +1, +1, -1),
        RIGHT:           (0,  0, +1,  0),
        RIGHT + BOTTOM:  (0,  0, +1, +1),
        OUTSIDE:         (0,  0,  0,  0),
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

    def enter(self, **kwds):
        """Enter the mode"""
        super(FrameEditMode, self).enter(**kwds)
        # Assign cursors
        mkcursor = lambda name: self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            name,
        )
        cn = gui.cursor.Name
        self.cursor_move_w_e = mkcursor(cn.MOVE_WEST_OR_EAST)
        self.cursor_move_n_s = mkcursor(cn.MOVE_NORTH_OR_SOUTH)
        self.cursor_move_nw_se = mkcursor(cn.MOVE_NORTHWEST_OR_SOUTHEAST)
        self.cursor_move_ne_sw = mkcursor(cn.MOVE_NORTHEAST_OR_SOUTHWEST)
        self.cursor_hand_closed = mkcursor(cn.HAND_CLOSED)
        self.cursor_hand_open = mkcursor(cn.HAND_OPEN)
        self.cursor_forbidden = mkcursor(cn.ARROW_FORBIDDEN)
        # If the frame isn't visible, show it. If it doesn't yet have a size,
        # then assign a sensible one which makes the frame visible on screen.
        model = self.doc.model
        if self.first_time and not model.get_frame_enabled():
            self.first_time = False
            x, y, w, h = model.get_frame()
            if w > 0 and h > 0:
                model.set_frame_enabled(True, user_initiated=True)
            else:
                x, y, w, h = model.get_bbox()
                if not (w > 0 and h > 0):
                    tdw = self.doc.tdw
                    alloc = tdw.get_allocation()
                    x1, y1 = tdw.display_to_model(0, 0)
                    x2, y2 = tdw.display_to_model(alloc.width * 0.5,
                                                  alloc.height * 0.5)
                    s = int(math.sqrt((x2-x1)**2 + (y2-y1)**2) * 0.666)
                    s = max(s, 64)
                    w = s
                    h = s
                    x, y = int(x2-s/2), int(y2-s/2)
                model.set_frame([x, y, w, h], user_initiated=True)
        # Overlay needs to be drawn
        self.doc.tdw.queue_draw()

    def leave(self, **kwds):
        """Exit the mode, hiding any dialogs"""
        dialog = self.get_options_widget()._size_dialog
        if self.doc:
            self.doc.tdw.queue_draw()
            if self not in self.doc.modes:
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
        dx2, dy2 = tdw.display_to_model(0, self.EDGE_SENSITIVITY)
        max_d = math.sqrt((dx1-dx2)**2 + (dy1-dy2)**2)

        if (x < fx1 - max_d or x > fx2 + max_d or
                y < fy1 - max_d or y > fy2 + max_d):
            return self.OUTSIDE
        zone = self.INSIDE  # zero

        if abs(fx2 - x) <= max_d:
            zone |= self.RIGHT
        elif abs(fx1 - x) <= max_d:
            zone |= self.LEFT
        if abs(fy2 - y) <= max_d:
            zone |= self.BOTTOM
        elif abs(fy1 - y) <= max_d:
            zone |= self.TOP

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
        if zone & self.RIGHT:
            rx = fx+fw
        elif zone & self.LEFT:
            rx = fx
        if zone & self.BOTTOM:
            ry = fy+fh
        elif zone & self.TOP:
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
                return

        # This should never happen.
        self.cursor = gdk.Cursor(gdk.BOGOSITY)

    def motion_notify_cb(self, tdw, event):
        if not self.in_drag:
            self._update_cursors(tdw, event.x, event.y)
            tdw.set_override_cursor(self.inactive_cursor)
            zone = self._get_zone(tdw, event.x, event.y)
            if zone != self._zone:
                self._zone = zone
                tdw.queue_draw()
        return super(FrameEditMode, self).motion_notify_cb(tdw, event)

    def drag_start_cb(self, tdw, event):
        model = self.doc.model
        self._orig_frame = tuple(model.get_frame())  # independent copy
        x0, y0 = self.start_x, self.start_y
        if self._zone is None:
            # This can happen if started from another mode with a key-down
            self._zone = self._get_zone(tdw, x0, y0)
        self._start_model_pos = tdw.display_to_model(x0, y0)
        return super(FrameEditMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, dx, dy):
        model = self.doc.model
        if model.frame_enabled:
            mx0, my0 = self._start_model_pos
            mx, my = tdw.display_to_model(event.x, event.y)
            fdx = int(round(mx - mx0))
            fdy = int(round(my - my0))

            mdx, mdy, mdw, mdh = self.DRAG_EFFECTS[self._zone]
            x, y, w, h = self._orig_frame

            x += min(w-self._MIN_FRAME_SIZE, mdx*fdx)
            y += min(h-self._MIN_FRAME_SIZE, mdy*fdy)

            w = max(self._MIN_FRAME_SIZE, w + mdw*fdx)
            h = max(self._MIN_FRAME_SIZE, h + mdh*fdy)

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


class FrameEditOptionsWidget (gtk.Alignment):
    """An options widget for directly editing frame values"""

    def __init__(self):
        gtk.Alignment.__init__(self, 0.5, 0.5, 1.0, 1.0)

        from application import get_app
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
        self.dpi_adj = gtk.Adjustment(dpi, upper=9600, lower=1,
                                      step_incr=76,  # hack: 3 clicks 72->300
                                      page_incr=dpi)
        self.unit_label = gtk.Label(_('px'))
        self.unit_label.set_alignment(0, 0.5)

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
        buttons = (gtk.STOCK_OK, gtk.RESPONSE_ACCEPT)
        self._size_dialog = windowing.Dialog(
            app, _("Frame Size"), app.drawWindow,
            buttons=buttons
        )
        unit = _('px')

        height_label = gtk.Label(_('Height:'))
        height_label.set_alignment(0.0, 0.5)
        width_label = gtk.Label(_('Width:'))
        width_label.set_alignment(0.0, 0.5)
        dpi_label = gtk.Label(_('Resolution:'))
        dpi_label.set_alignment(0.0, 0.5)
        color_label = gtk.Label(_('Color:'))
        color_label.set_alignment(0.0, 0.5)

        height_entry = gtk.SpinButton(
            adjustment=self.height_adj,
            climb_rate=0.25,
            digits=0
        )
        self.height_adj.set_spin_button(height_entry)

        width_entry = gtk.SpinButton(
            adjustment=self.width_adj,
            climb_rate=0.25,
            digits=0
        )
        self.width_adj.set_spin_button(width_entry)
        dpi_entry = gtk.SpinButton(
            adjustment=self.dpi_adj,
            climb_rate=0.0,
            digits=0
        )

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
        size_table.set_border_width(3)
        xopts = gtk.FILL | gtk.EXPAND
        yopts = gtk.FILL
        xpad = ypad = 3

        unit_combobox = gtk.ComboBoxText()
        for unit in UnitAdjustment.CONVERT_UNITS.keys():
            unit_combobox.append_text(unit)
        for i, key in enumerate(UnitAdjustment.CONVERT_UNITS):
            if key == _('px'):
                unit_combobox.set_active(i)
        unit_combobox.connect('changed', self.on_unit_changed)
        self._unit_combobox = unit_combobox

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

        # Options panel UI
        opts_table = gtk.Table(3, 3)
        opts_table.set_border_width(3)

        row = 0
        size_button = gtk.Button("<size-summary>")
        self._size_button = size_button
        size_button.connect("clicked", self._size_button_clicked_cb)
        opts_table.attach(size_button, 0, 2, row, row+1,
                          xopts, yopts, xpad, ypad)

        row += 1
        opts_table.attach(color_label, 0, 1, row, row+1,
                          xopts, yopts, xpad, ypad)
        opts_table.attach(color_align, 1, 2, row, row+1,
                          xopts, yopts, xpad, ypad)

        crop_layer_button = gtk.Button(_('Set Frame to Layer'))
        crop_layer_button.set_tooltip_text(_("Set frame to the extents of "
                                             "the current layer"))
        crop_document_button = gtk.Button(_('Set Frame to Document'))
        crop_document_button.set_tooltip_text(_("Set frame to the combination "
                                                "of all layers"))
        crop_layer_button.connect('clicked', self.crop_frame_cb,
                                  'CropFrameToLayer')
        crop_document_button.connect('clicked', self.crop_frame_cb,
                                     'CropFrameToDocument')

        trim_button = gtk.Button()
        trim_action = self.app.find_action("TrimLayer")
        trim_button.set_related_action(trim_action)
        trim_button.set_label(_('Trim Layer to Frame'))
        trim_button.set_tooltip_text(_("Trim parts of the current layer "
                                       "which lie outside the frame"))

        self.enable_button = gtk.CheckButton()
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
        content_area.pack_start(size_table, True, True)

        self._size_dialog.connect('response', self._size_dialog_response_cb)

        self.add(opts_table)

    def _size_dialog_response_cb(self, dialog, response_id):
        if response_id == gtk.RESPONSE_ACCEPT:
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
        r, g, b = RGBColor.new_from_gdk_color(color_gdk).get_rgb()
        a = float(colorbutton.get_alpha()) / 65535
        self.app.preferences["frame.color_rgba"] = (r, g, b, a)
        self.app.doc.tdw.queue_draw()

    def on_unit_changed(self, unit_combobox):
        active_unit = self.get_unit_text()
        self.width_adj.set_unit(active_unit)
        self.height_adj.set_unit(active_unit)
        self.unit_label.set_text(active_unit)
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
        text = _(u"{width}\u00D7{height} {units}").format(
            width=self.width_adj.get_unit_value(),
            height=self.height_adj.get_unit_value(),
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

    def _frame_corners(self):
        """Calculates the frame's corners, in display space"""
        tdw = self.doc.tdw
        x, y, w, h = tuple(self.doc.model.get_frame())
        points_model = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        points_disp = [tdw.model_to_display(*p) for p in points_model]
        return points_disp

    def paint(self, cr):
        """Paints the frame, and edit boxes if the app is in FrameEditMode"""
        if not self.doc.model.frame_enabled:
            return

        tdw = self.doc.tdw
        allocation = tdw.get_allocation()
        w, h = allocation.width, allocation.height
        canvas_bbox = (-1, -1, w+2, h+2)

        cr.rectangle(*canvas_bbox)

        corners = self._frame_corners()
        p1, p2, p3, p4 = [(int(x)+0.5, int(y)+0.5) for x, y in corners]
        cr.move_to(*p1)
        cr.line_to(*p2)
        cr.line_to(*p3)
        cr.line_to(*p4)
        cr.close_path()
        cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)

        frame_rgba = self.app.preferences["frame.color_rgba"]
        frame_rgba = [helpers.clamp(c, 0, 1) for c in frame_rgba]
        cr.set_source_rgba(*frame_rgba)
        editmode = None
        for m in self.doc.modes:
            if isinstance(m, FrameEditMode):
                editmode = m
                break
        if not editmode:
            # Frame mask
            cr.fill_preserve()
            # Doublestrike the edge for emphasis, assuming an alpha frame
            cr.set_line_width(self.OUTLINE_WIDTH)
            cr.stroke()
        else:
            # Frame mask
            cr.fill()
            # Editable/active frame outline
            cr.set_line_cap(cairo.LINE_CAP_ROUND)
            zonelines = [(FrameEditMode.TOP,    p1, p2),
                         (FrameEditMode.RIGHT,  p2, p3),
                         (FrameEditMode.BOTTOM, p3, p4),
                         (FrameEditMode.LEFT,   p4, p1)]
            edge_width = gui.style.DRAGGABLE_EDGE_WIDTH
            for zone, p, q in zonelines:
                gui.drawutils.draw_draggable_edge_drop_shadow(
                    cr=cr,
                    p0=p,
                    p1=q,
                    width=edge_width,
                )
            cr.set_line_width(edge_width)
            for zone, p, q in zonelines:
                if editmode._zone and (editmode._zone == zone):
                    rgb = gui.style.ACTIVE_ITEM_COLOR.get_rgb()
                else:
                    rgb = gui.style.EDITABLE_ITEM_COLOR.get_rgb()
                cr.set_source_rgb(*rgb)
                cr.move_to(*p)
                cr.line_to(*q)
                cr.stroke()
            # Dots corresponding to editable corners
            zonecorners = [
                (p1, FrameEditMode.TOP | FrameEditMode.LEFT),
                (p2, FrameEditMode.TOP | FrameEditMode.RIGHT),
                (p3, FrameEditMode.BOTTOM | FrameEditMode.RIGHT),
                (p4, FrameEditMode.BOTTOM | FrameEditMode.LEFT)
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


class UnitAdjustment(gtk.Adjustment):

    CONVERT_UNITS = {
        # unit :  (conversion_factor, upper, lower, step_incr, page_incr, digits)
        _('px'):   (0.0,     32000,     1,       1, 128, 0),
        _('inch'): (1.0,     200,    0.01,    0.01,   1, 2),
        _('cm'):   (2.54,    500,     0.1,     0.1,   1, 1),
        _('mm'):   (25.4,    5000,      1,       1,  10, 0),
    }

    def __init__(self, value=0, lower=0, upper=0, step_incr=0,
                 page_incr=0, page_size=0, dpi=DEFAULT_RESOLUTION):
        gtk.Adjustment.__init__(self, value, lower, upper, step_incr,
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
