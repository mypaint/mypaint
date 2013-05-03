# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""A tabbed, multi-page colour adjuster.
"""

import gui.gtk2compat

import gtk
from gtk import gdk
import gobject
from gettext import gettext as _

from adjbases import ColorAdjuster
from adjbases import PreviousCurrentColorAdjuster
from uicolor import RGBColor
from uimisc import borderless_button
from picker import ColorPickerButton


PREFS_KEY_CURRENT_TAB = 'colors.tab'


class CombinedAdjusterPage (ColorAdjuster):
    """Interface for page content in a `CombinedColorAdjuster`.

    Page instances are expect to distribute `set_color_manager()` to each of
    their component controls, and also to implement the methods defined in
    this interface.
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
        """Returns the `gtk.Table` instance for the page body.
        """
        raise NotImplementedError


class CombinedColorAdjuster (gtk.VBox, ColorAdjuster):
    """Composite colour adjuster consisting of several tabbed pages.
    """

    __adjusters = None
    __palette_page = None


    def __init__(self):
        import hcywheel
        import hsvtriangle
        import hsvcube
        import hsvwheel
        import sliders
        import paletteview
        palette_class = paletteview.PalettePage
        palette_page = None
        page_classes = (hcywheel.HCYAdjusterPage,
                        hsvwheel.HSVAdjusterPage,
                        palette_class,
                        hsvtriangle.HSVTrianglePage,
                        hsvcube.HSVCubePage,
                        sliders.ComponentSlidersAdjusterPage,
                        )

        gtk.VBox.__init__(self)
        self.__adjusters = []
        nb = self.__notebook = gtk.Notebook()
        nb.set_property("scrollable", True)
        page_index = 0
        for page_class in page_classes:
            icon_name = page_class.get_page_icon_name()
            icon_img = gtk.Image()
            icon_img.set_from_icon_name(icon_name, gtk.ICON_SIZE_SMALL_TOOLBAR)
            icon_img.connect("query-tooltip", self.__tab_tooltip_query_cb,
                             page_class)
            icon_img.set_property("has-tooltip", True)
            page_title = page_class.get_page_title()
            page = page_class()
            page_table = page.get_page_widget()

            picker = ColorPickerButton()
            comparator = PreviousCurrentColorAdjuster()
            bookmark_btn = borderless_button(
                        icon_name="bookmark-new",
                        tooltip=_("Add color to Palette"))
            bookmark_btn.connect("clicked", self.__bookmark_button_clicked_cb)
            properties_desc = page_class.get_properties_description()
            if properties_desc is not None:
                properties_btn = borderless_button(
                        stock_id=gtk.STOCK_PROPERTIES,
                        tooltip=properties_desc)
                properties_btn.connect("clicked",
                        self.__properties_button_clicked_cb,
                        page)
            else:
                properties_btn = borderless_button(
                        stock_id=gtk.STOCK_PROPERTIES)
                properties_btn.set_sensitive(False)

            # Common footer
            hbox = gtk.HBox()
            hbox.set_spacing(3)
            hbox.pack_start(picker, False, False)
            hbox.pack_start(comparator, True, True)
            hbox.pack_start(bookmark_btn, False, False)
            hbox.pack_start(properties_btn, False, False)

            # Full page layout
            vbox = gtk.VBox()
            vbox.set_spacing(3)
            vbox.set_border_width(3)
            vbox.pack_start(page_table, True, True)
            vbox.pack_start(hbox, False, False)
            vbox.__page = page

            nb.append_page(vbox, icon_img)

            self.__adjusters.append(page)
            self.__adjusters.append(comparator)
            self.__adjusters.append(picker)

            # Bookmark button writes colours here
            if page_class is palette_class:
                self.__palette_page = page
                self.__palette_page_index = page_index
            page_index += 1

        self.__shown = False
        self.connect("show", self.__first_show_cb)
        self.pack_start(nb, True, True)


    def get_palette_view(self):
        """Returns the palette view adjuster.
        """
        return self.__palette_page.get_page_widget()


    def show_palette_view(self):
        """Switches to the palette view tab.
        """
        self.__notebook.set_current_page(self.__palette_page_index)


    def __first_show_cb(self, widget):
        if self.__shown:
            return
        self.__shown = True
        nb = self.__notebook
        prefs = self.get_color_manager().get_prefs()
        if PREFS_KEY_CURRENT_TAB in prefs:
            prev_tab_icon_name = prefs[PREFS_KEY_CURRENT_TAB]
            for page_num, page_vbox in enumerate(nb):
                page = page_vbox.__page
                icon_name = page.get_page_icon_name()
                if icon_name == prev_tab_icon_name:
                    nb.set_current_page(page_num)
                    break
        nb.connect("switch-page", self.__notebook_switch_page_cb)


    def __bookmark_button_clicked_cb(self, widget):
        col = self.get_managed_color()
        self.__palette_page.add_color_to_palette(col)
        self.show_palette_view()


    def __properties_button_clicked_cb(self, widget, page):
        page.show_properties()


    def __tab_tooltip_query_cb(self, widget, x, y, kbd, tooltip, page_class):
        icon_name = page_class.get_page_icon_name()
        icon_title = page_class.get_page_title()
        icon_desc = page_class.get_page_description()
        tooltip.set_icon_from_icon_name(icon_name, gtk.ICON_SIZE_DIALOG)
        markup = "<b>%s</b>\n%s" % (icon_title, icon_desc)
        tooltip.set_markup(markup)
        return True


    def set_color_manager(self, manager):
        ColorAdjuster.set_color_manager(self, manager)
        for adj in self.__adjusters:
            adj.set_color_manager(manager)


    def __notebook_switch_page_cb(self, notebook,
                                  page_vbox_gpointer,
                                  page_num):
        page_vbox = notebook.get_nth_page(page_num)
        #  GPointers are not usable in pygtk
        page = page_vbox.__page
        icon_name = page.get_page_icon_name()
        prefs = self.get_color_manager().get_prefs()
        prefs[PREFS_KEY_CURRENT_TAB] = icon_name



if __name__ == '__main__':
    from adjbases import ColorManager
    import os
    import sys

    icon_theme = gtk.icon_theme_get_default()
    icon_theme.append_search_path("desktop/icons")

    combi = CombinedColorAdjuster()
    mgr = ColorManager()
    combi.set_color_manager(mgr)
    combi.set_managed_color(RGBColor(0.2, 0.5, 0.4))
    window = gtk.Window()
    window.add(combi)
    window.set_title(os.path.basename(sys.argv[0]))
    window.connect("destroy", lambda *a: gtk.main_quit())
    window.show_all()
    gtk.main()
