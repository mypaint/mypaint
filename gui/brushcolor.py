# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Brush colour changer.
"""

import gtk
from gtk import gdk
from gettext import gettext as _

import colors


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
        app.doc.input_stroke_ended_observers.append(self.__input_stroke_ended_cb)
        app.doc.model.stroke_observers.append(self.__stroke_observers_cb)

    def set_color(self, color):
        """Propagate user-set colours to the brush too (extension).
        """
        colors.ColorManager.set_color(self, color)
        if not self.__in_callback:
            self.__brush.set_color_hsv(color.get_hsv())

    def __settings_changed_cb(self, settings):
        # When the colour changes by external means, update the adjusters.
        if not settings.intersection(('color_h', 'color_s', 'color_v')):
            return
        brush_color = colors.HSVColor(*self.__brush.get_color_hsv())
        if brush_color == self.get_color():
            return
        self.__in_callback = True
        self.set_color(brush_color)
        self.__in_callback = False

    def __input_stroke_ended_cb(self, event):
        # Update the colour usage history immediately after the user paints
        # with a new colour, for responsiveness.
        brush = self.__brush
        if not brush.is_eraser():
            col = colors.HSVColor(*brush.get_color_hsv())
            self.push_history(col)

    def __stroke_observers_cb(self, stroke, brush):
        # Update the colour usage history whenever the stroke is split, for
        # correctness with splatter brushes which don't depend on pressure.
        brush = self.__brush
        if not brush.is_eraser():
            col = colors.HSVColor(*brush.get_color_hsv())
            self.push_history(col)
