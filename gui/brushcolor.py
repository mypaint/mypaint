# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Brush color changer."""

from __future__ import division, print_function

from . import colors
import lib.color
import colour


class BrushColorManager (colors.ColorManager):
    """Color manager mediating between brush settings and the color adjusters.
    """

    __brush = None
    __in_callback = False

    def __init__(self, app):
        """Initialize, binding to certain events.
        """
        colors.ColorManager.__init__(self, app.preferences, app.datapath)
        self.__brush = app.brush
        app.brush.observers.append(self.__settings_changed_cb)
        app.doc.input_stroke_ended += self.__input_stroke_ended_cb
        app.doc.model.sync_pending_changes += self.__sync_pending_changes_cb
        self._app = app

    def set_color(self, color):
        """Propagate user-set colors to the brush too (extension).
        """

        colors.ColorManager.set_color(self, color)
        if not self.__in_callback:

            if not isinstance(color, lib.color.CAM16Color):
#                prefs = self.get_prefs()
#                illuminant = prefs['color.dimension_illuminant']

#                if illuminant == "custom_XYZ":
#                    illuminant = prefs['color.dimension_illuminant_XYZ']
#                else:
#                    illuminant = colour.xy_to_XYZ(
#                        colour.ILLUMINANTS['cie_2_1931'][illuminant]) * 100.0
#                # standard sRGB view environment except adjustable illuminant
#                cieaxes = prefs['color.dimension_value'] + \
#                    prefs['color.dimension_purity'] + "h"
                color = lib.color.CAM16Color(color=color,
                    illuminant=self.__brush.CAM16Color.illuminant)
            self._app.doc.last_color_target = None
            self._app.doc.last_palette_color = None
            self._app.doc.current_palette_color_target = None
            self.__brush.set_cam16_color(color)
            self.__brush.set_color_hsv(color.get_hsv())

    def __settings_changed_cb(self, settings):
        # When the color changes by external means, update the adjusters.
        if not settings.intersection(('color_h', 'color_s', 'color_v')):
            return
        brush_color = self.__brush.CAM16Color
        if brush_color == self.get_color():
            return
        self.__in_callback = True
        self.set_color(brush_color)
        self.__in_callback = False



    def __input_stroke_ended_cb(self, doc, event):
        # Update the color usage history immediately after the user
        # makes an explicit (pen down/stroke/up) brushstroke,
        # for responsiveness.
        brush = self.__brush
        if not brush.is_eraser():
            col = brush.CAM16Color
            self.push_history(col)

    def __sync_pending_changes_cb(self, model, flush=True, **kwargs):
        # Update the color usage history after any flushing sync.
        #
        # Freehand mode sends a flushing sync request after painting
        # continuously for some seconds if pixel changes were made.
        # Pixel changes can happen even without pressure so we need
        # this hook as well as __input_stroke_ended_cb for correctness.
        if not flush:
            return
        brush = self.__brush
        if not brush.is_eraser():
            col = brush.CAM16Color
            self.push_history(col)
