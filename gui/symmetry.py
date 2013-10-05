# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


import cairo

import overlays


class SymmetryOverlay (overlays.Overlay):
    """Symmetry overlay, operating in display coordinates.
    """

    DASH_OUTLINE_COLOR = (0.8, 0.4, 0.0)
    DASH_LINE_COLOR    = (1.0, 0.666, 0.333)
    DASH_LINE_PATTERN  = [5.0, 5.0]


    def __init__(self, doc):
        overlays.Overlay.__init__(self)
        self.doc = doc
        self.tdw = self.doc.tdw
        self.axis = doc.model.get_symmetry_axis()
        self.doc.model.symmetry_observers.append(self.symmetry_changed_cb)


    def symmetry_changed_cb(self):
        new_axis = self.doc.model.get_symmetry_axis()
        if new_axis != self.axis:
            self.axis = new_axis
            self.tdw.queue_draw()


    def paint(self, cr):
        """Paint the overlay, in display coordinates.
        """

        # The symmetry axis is a line (x==self.axis) in model coordinates
        axis_x_m = self.axis
        if axis_x_m is None:
            return

        # allocation, in display coords
        alloc = self.tdw.get_allocation()
        view_x0, view_y0 = 0, 0
        view_x1, view_y1 = view_x0+alloc.width, view_y0+alloc.height

        # Viewing rectangle extents, in model coords
        corners = [ (view_x0, view_y0), (view_x0, view_y1),
                    (view_x1, view_y1), (view_x1, view_y0), ]
        corners_m = [self.tdw.display_to_model(*c) for c in corners]
        min_corner_y_m = min([c_m[1] for c_m in corners_m])
        max_corner_y_m = max([c_m[1] for c_m in corners_m])

        # Back to display coords, with rounding and pixel centring
        ax_x0, ax_y0 = [int(c)+0.5 for c in
            self.tdw.model_to_display(axis_x_m, min_corner_y_m) ]
        ax_x1, ax_y1 = [int(c)+0.5 for c in
            self.tdw.model_to_display(axis_x_m, max_corner_y_m) ]

        # Paint axis
        cr.save()
        offs = 1 + int(0.5 * self.DASH_LINE_PATTERN[0])
        cr.set_line_width(3.0)
        cr.set_source_rgb(*self.DASH_OUTLINE_COLOR)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_dash(self.DASH_LINE_PATTERN, offs)
        cr.move_to(ax_x0, ax_y0)
        cr.line_to(ax_x1, ax_y1)
        cr.stroke_preserve()
        cr.set_line_width(1.0)
        cr.set_source_rgb(*self.DASH_LINE_COLOR)
        cr.set_line_cap(cairo.LINE_CAP_SQUARE)
        cr.set_dash(self.DASH_LINE_PATTERN, offs)
        cr.stroke()
        cr.restore()
