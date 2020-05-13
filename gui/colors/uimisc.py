# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""UI miscellanea.
"""

from __future__ import division, print_function

from lib.gibindings import Gtk


## Layout constants ##

# Most pages in the old combined adjuster widget were split into a primary area
# containing two of the channels, and a secondary area to its left containing
# the third.
PRIMARY_ADJUSTERS_MIN_WIDTH = 100
PRIMARY_ADJUSTERS_MIN_HEIGHT = 100

# Slider minimum dimensions
SLIDER_MIN_WIDTH = 18
SLIDER_MIN_LENGTH = 50


## Helper functions ##

def borderless_button(stock_id=None,
                      icon_name=None,
                      size=Gtk.IconSize.SMALL_TOOLBAR,
                      tooltip=None):
    button = Gtk.Button()
    if stock_id is not None:
        image = Gtk.Image()
        image.set_from_stock(stock_id, size)
        button.add(image)
    elif icon_name is not None:
        image = Gtk.Image()
        image.set_from_icon_name(icon_name, size)
        button.add(image)
    button.set_name("borderless-button")
    button.set_relief(Gtk.ReliefStyle.NONE)
    button.set_can_default(False)
    button.set_can_focus(False)
    has_tooltip = tooltip is not None
    if has_tooltip:
        button.set_tooltip_text(tooltip)
    button.set_has_tooltip(has_tooltip)
    return button
