# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2007-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2009-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function

from gi.repository import Gtk


class BrushWidget(Gtk.DrawingArea):
    """A widget that holds a reference to a brush too

    """

    __gtype_name__ = "MyPaintBrushWidget"

    def __init__(self):
        self.brush = None
