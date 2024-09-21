# This file is part of MyPaint.
# Copyright (C) 2012-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Deprecated interface for tabbed color selectors.

Newly written color adjusters should present the attributes described in
gui.workspace instead of implementing this interface.

"""

from __future__ import division, print_function

from .adjbases import ColorAdjuster


class CombinedAdjusterPage (ColorAdjuster):
    """Interface for multitab page content.

    Page instances are expected to distribute `set_color_manager()` to each of
    their component controls, and also to implement the methods defined in this
    interface.

    """

    @classmethod
    def get_page_icon_name(class_):
        """Returns the page's icon name.
        """
        raise NotImplementedError

    @classmethod
    def get_page_title(class_):
        """Returns the title for the page.

        Word as "this page/tab contains a [...]", in titlecase.
        """
        raise NotImplementedError

    @classmethod
    def get_page_description(class_):
        """Returns the descriptive text for the page.

        Word as "this page/tab lets you [...]", in titlecase.
        """
        raise NotImplementedError

    @classmethod
    def get_properties_description(class_):
        """Override & return a string if `show_properties()` is implemented.

        The returned string should explain what the properties button does. The
        default implemented here returns None, which also indicates that no
        properties dialog is implemented.
        """
        return None

    def show_properties(self):
        """Override to show the page's properties dialog.
        """
        pass

    def get_page_widget(self):
        """Returns the `Gtk.Table` instance for the page body.
        """
        raise NotImplementedError
