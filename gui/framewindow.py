
"""Frame manipulation mode and subwindow
"""

import math

import gtk
from gtk import gdk
from gettext import gettext as _

import canvasevent
import windowing
from colors.uicolor import RGBColor


class FrameEditMode (canvasevent.DragMode):
    """Stackable interaction mode for editing the document frame.

    The frame editing mode has an associated details dialog which is open for
    the length of time the mode is active.

    """

    __action_name__ = 'FrameEditMode'
    cursor = gdk.Cursor(gdk.ICON)

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


    def __init__(self, oneshot=False):
        """Initialize.

        :param oneshot: Auto-exit when the drag stops.
        :type oneshot: bool

        Oneshot frame adjustment modes do not show the details dialog: they're
        intended for use only while the key or pointer button which initiated
        the drag is held down.

        """
        canvasevent.DragMode.__init__(self)
        self._zone = None
        self._orig_frame = None
        self._oneshot = oneshot


    def enter(self, doc):
        canvasevent.DragMode.enter(self, doc)
        if not self._oneshot:
            lm = self.doc.app.layout_manager
            dialog = lm.get_subwindow_by_role("frameWindow")
            dialog.show_all()


    def leave(self):
        if not self._oneshot:
            lm = self.doc.app.layout_manager
            dialog = lm.get_subwindow_by_role("frameWindow")
            dialog.hide()
        canvasevent.DragMode.leave(self)


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


    def _update_cursor(self, tdw, xd, yd):
        model = self.doc.model
        if not model.frame_enabled:
            self.cursor = gdk.Cursor(gdk.ICON)
            return

        # Simpler interpretations
        zone = self._get_zone(tdw, xd, yd)
        if zone == self.OUTSIDE:
            self.cursor = gdk.Cursor(gdk.ICON)
            return
        elif zone == self.INSIDE:
            self.cursor = gdk.Cursor(gdk.FLEUR)
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

        # The cursor chosen reflects the frame edge to be moved in display
        # space. Looks a little funny if the canvas is rotated at a 45 degree
        # angle, but it's not bad.
        cursors = [ (1, gdk.RIGHT_SIDE),
                    (3, gdk.BOTTOM_RIGHT_CORNER),
                    (5, gdk.BOTTOM_SIDE),
                    (7, gdk.BOTTOM_LEFT_CORNER),
                    (9, gdk.LEFT_SIDE),
                    (11, gdk.TOP_LEFT_CORNER),
                    (13, gdk.TOP_SIDE),
                    (15, gdk.TOP_RIGHT_CORNER),
                    (17, gdk.RIGHT_SIDE),         ]
        for i, cursor in cursors:
            if theta < i*(2.0/16)*math.pi:
                self.cursor = gdk.Cursor(cursor)
                return

        # This should never happen.
        self.cursor = gdk.Cursor(gdk.BOGOSITY)


    def motion_notify_cb(self, tdw, event):
        if not self.in_drag:
            self._update_cursor(tdw, event.x, event.y)
            tdw.set_override_cursor(self.cursor)
            self._zone = self._get_zone(tdw, event.x, event.y)
        return canvasevent.DragMode.motion_notify_cb(self, tdw, event)


    def drag_start_cb(self, tdw, event):
        model = self.doc.model
        self._orig_frame = model.get_frame()
        if self._zone is None:
            # This can happen if started from another mode with a key-down
            self._zone = self._get_zone(tdw, self.start_x, self.start_y)
        self._update_cursor(tdw, self.start_x, self.start_y)
        tdw.set_override_cursor(self.cursor)


    def drag_stop_cb(self):
        if self._oneshot:
            self.doc.modes.pop()


    def drag_update_cb(self, tdw, event, dx, dy):
        model = self.doc.model
        if not model.frame_enabled:
            return

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


class Window (windowing.Dialog):
    """A dialog window for directly editing frame values.
    """

    def __init__(self, app):
        buttons = (gtk.STOCK_OK, gtk.RESPONSE_ACCEPT)
        windowing.Dialog.__init__(self, app, ("Frame"), app.drawWindow, buttons=buttons)

        self.callbacks_active = False

        x, y, w, h = self.app.doc.model.get_frame()

        self.width_adj  = gtk.Adjustment(w, upper=32000, lower=1, step_incr=1, page_incr=128)
        self.height_adj = gtk.Adjustment(h, upper=32000, lower=1, step_incr=1, page_incr=128)

        self.width_adj.connect('value-changed', self.on_size_adjustment_changed)
        self.height_adj.connect('value-changed', self.on_size_adjustment_changed)

        self.app.doc.model.frame_observers.append(self.on_frame_changed)

        self._init_ui()

    def _init_ui(self):
        height_label = gtk.Label(_('Height:'))
        height_label.set_alignment(0.0, 0.5)
        width_label = gtk.Label(_('Width:'))
        width_label.set_alignment(0.0, 0.5)
        color_label = gtk.Label(_('Color:'))
        color_label.set_alignment(0.0, 0.5)

        height_entry = gtk.SpinButton()
        height_entry.set_adjustment(self.height_adj)
        width_entry = gtk.SpinButton()
        width_entry.set_adjustment(self.width_adj)
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

        size_table = gtk.Table(6, 2)
        size_table.set_border_width(9)
        xopts = yopts = gtk.FILL|gtk.EXPAND
        xpad = ypad = 3

        row = 0
        size_table.attach(width_label, 0, 1, row, row+1,
                          xopts, yopts, xpad, ypad)
        size_table.attach(width_entry, 1, 2, row, row+1,
                          xopts, yopts, xpad, ypad)

        row += 1
        size_table.attach(height_label, 0, 1, row, row+1,
                          xopts, yopts, xpad, ypad)
        size_table.attach(height_entry, 1, 2, row, row+1,
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

        self.enable_button = gtk.CheckButton(_('Enabled'))
        self.enable_button.connect('toggled', self.on_frame_toggled)
        enabled = self.app.doc.model.frame_enabled
        self.enable_button.set_active(enabled)

        row += 1
        size_table.attach(self.enable_button, 1, 2, row, row+1,
                          xopts, yopts, xpad, ypad+6)

        row += 1
        size_table.attach(hint_label, 0, 2, row, row+1,
                          xopts, yopts, xpad, ypad)

        row += 1
        size_table.attach(crop_layer_button, 0, 2, row, row+1,
                          xopts, yopts, xpad, ypad)

        row += 1
        size_table.attach(crop_document_button, 0, 2, row, row+1,
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

    # FRAME
    def crop_frame_cb(self, button, command):
        if command == 'CropFrameToLayer':
            bbox = self.app.doc.model.get_current_layer().get_bbox()
        elif command == 'CropFrameToDocument':
            bbox = self.app.doc.model.get_bbox()
        else: assert 0
        self.app.doc.model.set_frame_enabled(True)
        self.app.doc.model.set_frame(*bbox)

    def _color_set_cb(self, colorbutton):
        color_gdk = colorbutton.get_color()
        r,g,b = RGBColor.new_from_gdk_color(color_gdk).get_rgb()
        a = float(colorbutton.get_alpha()) / 65535
        self.app.preferences["frame.color_rgba"] = (r, g, b, a)
        self.app.doc.tdw.queue_draw()

    def on_frame_toggled(self, button):
        """Update the frame state in the model."""
        if self.callbacks_active:
            return

        self.app.doc.model.set_frame_enabled(button.get_active())

    def on_size_adjustment_changed(self, adjustment):
        """Update the frame size in the model."""
        if self.callbacks_active:
            return

        width = int(self.width_adj.get_value())
        height = int(self.height_adj.get_value())

        self.app.doc.model.set_frame(width=width, height=height)

    def on_frame_changed(self):
        """Update the UI to reflect the model."""
        self.callbacks_active = True # Prevent callback loops

        x, y, w, h = self.app.doc.model.get_frame()
        self.width_adj.set_value(w)
        self.height_adj.set_value(h)
        enabled = self.app.doc.model.frame_enabled
        self.enable_button.set_active(enabled)

        self.callbacks_active = False
