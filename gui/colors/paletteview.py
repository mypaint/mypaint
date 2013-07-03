# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Viewer and editor widgets for palettes."""

# Editor ideas:
#   - Interpolate between two colours, into empty slots
#   - "Insert lighter/darker copy of row".
#   - repack palette (remove duplicates and blanks)
#   - sort palette by approx. hue+chroma binning, then luma variations


## Imports

import math
from copy import deepcopy
import os
import re
import logging
logger = logging.getLogger(__name__)

import gi
from gi.repository import Gdk
from gi.repository import Gtk
import cairo
from gettext import gettext as _

from lib.observable import event
from util import clamp

from palette import Palette
from uicolor import RGBColor
from uicolor import HCYColor


## Imports still requiring gtk2compat

if __name__ == '__main__':
    import gui.gtk2compat
from uimisc import borderless_button
from adjbases import ColorAdjuster
from adjbases import ColorAdjusterWidget
from adjbases import ColorManager
from adjbases import DATAPATH_PALETTES_SUBDIR
from combined import CombinedAdjusterPage


## Class defs


class PalettePage (CombinedAdjusterPage):
    """User-editable palette, as a `CombinedAdjuster` element.
    """


    def __init__(self):
        view = PaletteView()
        view.grid.show_matched_color = True
        view.can_select_empty = False
        self._adj = view
        self._edit_dialog = None


    @classmethod
    def get_properties_description(class_):
        return _("Palette properties")


    @classmethod
    def get_page_icon_name(self):
        return "mypaint-tool-color-palette"


    @classmethod
    def get_page_title(self):
        return _("Palette")


    @classmethod
    def get_page_description(self):
        return _("Set the color from a loadable, editable palette.")


    def get_page_widget(self):
        """Page widget: returns the PaletteView adjuster widget itself."""
        # FIXME: The PaletteNext and PalettePrev actions of the main
        #        app require access to the PaletteView itself.
        return self._adj


    def set_color_manager(self, manager):
        CombinedAdjusterPage.set_color_manager(self, manager)
        self._adj.set_color_manager(manager)




    def show_properties(self):
        if self._edit_dialog is None:
            toplevel = self._adj.get_toplevel()
            dialog = PaletteEditorDialog(toplevel, self.get_color_manager())
            self._edit_dialog = dialog
        self._edit_dialog.run()


class PaletteEditorDialog (Gtk.Dialog):
    """Dialog for editing, loading and saving the current palette.
    """

    def __init__(self, parent, target_color_manager):
        flags = Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT
        Gtk.Dialog.__init__(self, _("Palette Editor"), parent, flags,
                            (Gtk.STOCK_CANCEL, Gtk.ResponseType.REJECT,
                             Gtk.STOCK_OK, Gtk.ResponseType.ACCEPT))
        self.set_position(Gtk.WindowPosition.MOUSE)

        assert isinstance(target_color_manager, ColorManager)
        #: ColorManager containing the palette to be edited.
        self._target_color_manager = target_color_manager

        view = PaletteView()
        view.set_size_request(400, 300)
        view.grid.show_matched_color = True
        view.grid.can_select_empty = True
        self._view = view

        #: The working ColorManager, holding a working copy of the palette
        #: to be edited.
        self._mgr = ColorManager(prefs=None,
                        datapath=target_color_manager.get_data_path())
        self._mgr.set_color(RGBColor(1,1,1))
        view.set_color_manager(self._mgr)

        # Action buttons, positiopned down the right hand side
        action_bbox = Gtk.VButtonBox()
        load_btn = self._load_button = Gtk.Button(stock=Gtk.STOCK_OPEN)
        save_btn = self._save_button = Gtk.Button(stock=Gtk.STOCK_SAVE)
        add_btn = self._add_button = Gtk.Button(stock=Gtk.STOCK_ADD)
        remove_btn = self._remove_button = Gtk.Button(stock=Gtk.STOCK_REMOVE)
        clear_btn = self._clear_button = Gtk.Button(stock=Gtk.STOCK_CLEAR)
        action_bbox.pack_start(load_btn)
        action_bbox.pack_start(save_btn)
        action_bbox.pack_start(add_btn)
        action_bbox.pack_start(remove_btn)
        action_bbox.pack_start(clear_btn)
        action_bbox.set_layout(Gtk.ButtonBoxStyle.START)
        load_btn.connect("clicked", self._load_btn_clicked)
        save_btn.connect("clicked", self._save_btn_clicked)
        remove_btn.connect("clicked", self._remove_btn_clicked)
        add_btn.connect("clicked", self._add_btn_clicked)
        clear_btn.connect("clicked", self._clear_btn_clicked)
        load_btn.set_tooltip_text(_("Load from a GIMP palette file"))
        save_btn.set_tooltip_text(_("Save to a GIMP palette file"))
        add_btn.set_tooltip_text(_("Add a new empty swatch"))
        remove_btn.set_tooltip_text(_("Remove the current swatch"))
        clear_btn.set_tooltip_text(_("Remove all swatches"))

        # Button initial state and subsequent updates
        remove_btn.set_sensitive(False)
        self._mgr.palette.match_changed += self._palette_match_changed_cb
        self._mgr.palette.info_changed += self._palette_changed_cb
        self._mgr.palette.sequence_changed += self._palette_changed_cb
        self._mgr.palette.color_changed += self._palette_changed_cb

        # Palette name and number of entries
        palette_details_hbox = Gtk.HBox()
        palette_name_label = Gtk.Label(_("Name:"))
        palette_name_label.set_tooltip_text(
          _("Name or description for this palette"))
        palette_name_entry = Gtk.Entry()
        palette_name_entry.connect("changed", self._palette_name_changed_cb)
        self._palette_name_entry = palette_name_entry
        self._columns_adj = Gtk.Adjustment(
          value=0, lower=0, upper=99,
          step_incr=1, page_incr=1, page_size=0 )
        self._columns_adj.connect("value-changed", self._columns_changed_cb)
        columns_label = Gtk.Label(_("Columns:"))
        columns_label.set_tooltip_text(_("Number of columns"))
        columns_label.set_tooltip_text(_("Number of columns"))
        columns_spinbutton = Gtk.SpinButton(
          adjustment=self._columns_adj,
          climb_rate=1.5,
          digits=0 )
        palette_details_hbox.set_spacing(0)
        palette_details_hbox.set_border_width(0)
        palette_details_hbox.pack_start(palette_name_label, False, False, 0)
        palette_details_hbox.pack_start(palette_name_entry, True, True, 6)
        palette_details_hbox.pack_start(columns_label, False, False, 6)
        palette_details_hbox.pack_start(columns_spinbutton, False, False, 0)

        color_name_hbox = Gtk.HBox()
        color_name_label = Gtk.Label(_("Color name:"))
        color_name_label.set_tooltip_text(_("Current colour's name"))
        color_name_entry = Gtk.Entry()
        color_name_entry.connect("changed", self._color_name_changed_cb)
        color_name_entry.set_sensitive(False)
        self._color_name_entry = color_name_entry
        color_name_hbox.set_spacing(6)
        color_name_hbox.pack_start(color_name_label, False, False, 0)
        color_name_hbox.pack_start(color_name_entry, True, True, 0)

        palette_vbox = Gtk.VBox()
        palette_vbox.set_spacing(12)
        palette_vbox.pack_start(palette_details_hbox, False, False)
        palette_vbox.pack_start(view, True, True)
        palette_vbox.pack_start(color_name_hbox, False, False)

        # Dialog contents
        # Main edit area to the left, buttons to the right
        hbox = Gtk.HBox()
        hbox.set_spacing(12)
        hbox.pack_start(palette_vbox, True, True)
        hbox.pack_start(action_bbox, False, False)
        hbox.set_border_width(12)
        self.vbox.pack_start(hbox, True, True)

        # Dialog vbox contents must be shown separately
        for w in self.vbox:
            w.show_all()

        self.connect("response", self._response_cb)
        self.connect("show", self._show_cb)


    def _show_cb(self, widget, *a):
        # Each time the dialog is shown, update with the target
        # palette, for editing.
        self.vbox.show_all()
        palette = self._target_color_manager.palette
        name = palette.get_name()
        if name is None:
            name = ""
        self._palette_name_entry.set_text(name)
        self._columns_adj.set_value(palette.get_columns())
        self._mgr.palette.update(palette)


    def _palette_name_changed_cb(self, editable):
        name = editable.get_chars(0, -1)
        if name == "":
            name = None
        pal = self._mgr.palette
        pal.name = unicode(name)


    def _columns_changed_cb(self, adj):
        ncolumns = int(adj.get_value())
        pal = self._mgr.palette
        pal.set_columns(ncolumns)


    def _color_name_changed_cb(self, editable):
        name = editable.get_chars(0, -1)
        grid = self._view.grid
        palette = self._mgr.palette
        i = palette.match_position
        if i is None:
            return
        old_name = palette.get_color_name(i)
        if name == "":
            name = None
        if name != old_name:
            palette.set_color_name(i, name)


    def _response_cb(self, widget, response_id):
        if response_id == Gtk.ResponseType.ACCEPT:
            palette = self._mgr.palette
            target_palette = self._target_color_manager.palette
            target_palette.update(palette)
        self.hide()
        return True


    def _palette_match_changed_cb(self, palette):
        col_name_entry = self._color_name_entry
        remove_btn = self._remove_button
        i = palette.match_position
        if i is not None:
            col = palette[i]
            if col is not None:
                name = palette.get_color_name(i)
                if name is None:
                    name = ""
                col_name_entry.set_sensitive(True)
                col_name_entry.set_text(name)
            else:
                col_name_entry.set_sensitive(False)
                col_name_entry.set_text(_("Empty palette slot"))
        else:
            col_name_entry.set_sensitive(False)
            col_name_entry.set_text("")
        self._update_buttons()


    def _update_buttons(self):
        palette = self._mgr.palette
        emptyish = len(palette) == 0
        if len(palette) == 1:
            if palette[0] is None:
                emptyish = True
        can_save = not emptyish
        can_clear = not emptyish
        can_remove = True
        if emptyish or self._mgr.palette.match_position is None:
            can_remove = False
        self._save_button.set_sensitive(can_save)
        self._remove_button.set_sensitive(can_remove)
        self._clear_button.set_sensitive(can_clear)


    def _palette_changed_cb(self, palette, *args, **kwargs):
        new_name = palette.get_name()
        if new_name is None:
            new_name = ""
        old_name = self._palette_name_entry.get_chars(0, -1)
        if old_name != new_name:
            self._palette_name_entry.set_text(new_name)
        self._columns_adj.set_value(palette.get_columns())
        self._update_buttons()


    def _add_btn_clicked(self, button):
        grid = self._view.grid
        palette = self._mgr.palette
        i = palette.match_position
        if i is None:
            i = len(palette)
            palette.append(None)
            palette.match_position = i
        else:
            palette.insert(i, None)


    def _remove_btn_clicked(self, button):
        grid = self._view.grid
        palette = self._mgr.palette
        i = palette.match_position
        if i >= 0 and i < len(palette):
            palette.pop(i)
            if len(palette) == 0:
                palette.append(None)


    def _load_btn_clicked(self, button):
        preview = _PalettePreview()
        manager = self._target_color_manager
        datapath = manager.get_data_path()
        palettes_dir = os.path.join(datapath, DATAPATH_PALETTES_SUBDIR)
        palette = palette_load_via_dialog(title=_("Load palette"),
                                          parent=self,
                                          preview=preview,
                                          shortcuts=[palettes_dir])
        if palette is not None:
            self._mgr.palette.update(palette)


    def _save_btn_clicked(self, button):
        preview = _PalettePreview()
        palette_save_via_dialog(self._mgr.palette, title=_("Save palette"),
                                parent=self, preview=preview)


    def _clear_btn_clicked(self, button):
        pal = self._mgr.palette
        pal.clear()



class PaletteView (ColorAdjuster, Gtk.ScrolledWindow):
    """Scrollable view of a palette.

    Palette entries can be clicked to select the colour, and all instances of
    the current shared colour in the palette are highlighted.

    """

    ## Sizing contraint constants
    _MIN_HEIGHT = 32
    _MIN_WIDTH = 150
    _MAX_NATURAL_HEIGHT = 300
    _MAX_NATURAL_WIDTH = 300


    def __init__(self):
        Gtk.ScrolledWindow.__init__(self)
        self.grid = _PaletteGridLayout()
        self.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.add_with_viewport(self.grid)

    def set_color_manager(self, mgr):
        self.grid.set_color_manager(mgr)
        ColorAdjuster.set_color_manager(self, mgr)


    ## Sizing boilerplate
    # Reflect what the embedded grid widget tells us, but limit its natural
    # size to something sensible. Huge palettes make huge grids...

    def do_get_request_mode(self):
        return self.grid.get_request_mode()

    def do_get_preferred_width(self):
        gminw, gnatw = self.grid.get_preferred_width()
        minw = self._MIN_WIDTH
        natw = min(gnatw, self._MAX_NATURAL_WIDTH)
        return minw, max(minw, natw)

    def do_get_preferred_height(self):
        gminh, gnath = self.grid.get_preferred_height()
        minh = self._MIN_HEIGHT
        nath = min(gnath, self._MAX_NATURAL_HEIGHT)
        return minh, max(minh, nath)

    def do_get_preferred_width_for_height(self, height):
        gminw, gnatw = self.grid.get_preferred_width_for_height(height)
        minw = self._MIN_WIDTH
        natw = min(gnatw, self._MAX_NATURAL_WIDTH)
        return minw, max(minw, natw)

    def do_get_preferred_height_for_width(self, width):
        gminh, gnath = self.grid.get_preferred_height_for_width(width)
        minh = self._MIN_HEIGHT
        nath = min(gnath, self._MAX_NATURAL_HEIGHT)
        return minh, max(minh, nath)


class _PalettePreview (Gtk.DrawingArea):
    """Preview-only palette view."""

    _palette = None

    def __init__(self):
        Gtk.DrawingArea.__init__(self)
        self.connect("draw", self._draw_cb)
        self.set_size_request(128, 256)

    def _draw_cb(self, widget, cr):
        if self._palette is None:
            return
        alloc = widget.get_allocation()
        w, h = alloc.width, alloc.height
        s_max = 16  # min(w, h)
        s_min = 4
        ncolumns = self._palette.get_columns()
        ncolors = len(self._palette)
        if ncolors == 0:
            return
        if not ncolumns == 0:
            s = w / ncolumns
            s = clamp(s, s_min, s_max)
            s = int(s)
            if s*ncolumns > w:
                ncolumns = 0
        if ncolumns == 0:
            s = math.sqrt(float(w*h) / ncolors)
            s = clamp(s, s_min, s_max)
            s = int(s)
            ncolumns = max(1, int(w / s))
        nrows = int(ncolors // ncolumns)
        if ncolors % ncolumns != 0:
            nrows += 1
        nrows = max(1, nrows)
        dx, dy = 0, 0
        if nrows*s < h:
            dy = int(h - nrows*s) / 2
        if ncolumns*s < w:
            dx = int(w - ncolumns*s) / 2

        state = self.get_state_flags()
        style = self.get_style_context()
        bg_rgba = style.get_background_color(state)
        bg_color = RGBColor.new_from_gdk_rgba(bg_rgba)

        _palette_render(self._palette, cr, rows=nrows, columns=ncolumns,
                        swatch_size=s, bg_color=bg_color,
                        offset_x=dx, offset_y=dy,
                        rtl=False)

    def set_palette(self, palette):
        self._palette = palette
        self.queue_draw()


class _PaletteGridLayout (ColorAdjusterWidget):
    """The palette layout embedded in a scrolling PaletteView.
    """

    ## Class settings
    IS_DRAG_SOURCE = True
    HAS_DETAILS_DIALOG = True
    STATIC_TOOLTIP_TEXT = _("Color swatch palette.\nDrop colors here,\n"
                            "drag them to organize.")

    ## Layout constants
    _SWATCH_SIZE_MIN = 8
    _SWATCH_SIZE_MAX = 50
    _SWATCH_SIZE_NOMINAL = 20
    _PREFERRED_COLUMNS = 5 #: Preferred width in cells for free-flow mode.


    def __init__(self):
        ColorAdjusterWidget.__init__(self)
        # Sizing
        s = self._SWATCH_SIZE_NOMINAL
        self.set_size_request(s, s)
        self.connect("size-allocate", self._size_alloc_cb)
        #: Highlight the currently matched color
        self.show_matched_color = False
        #: User can click on empty slots
        self.can_select_empty = False
        # Current index
        self.connect("button-press-event", self._button_press_cb)
        self.connect_after("button-release-event", self._button_release_cb)
        # Dragging
        self._drag_insertion_index = None
        self.connect("motion-notify-event", self._motion_notify_cb)
        self.add_events(Gdk.EventMask.POINTER_MOTION_MASK)
        # Tooltips
        self._tooltip_index = None
        self.set_has_tooltip(True)
        # Cached layout details
        self._rows = None
        self._columns = None
        self._swatch_size = self._SWATCH_SIZE_NOMINAL


    def _size_alloc_cb(self, widget, alloc):
        """Caches layout details after size negotiation.
        """
        width = alloc.width
        height = alloc.height
        ncolors, nrows, ncolumns = self._get_palette_dimensions()
        if nrows and ncolumns:
            # Fitted to the major dimension
            size = int( min(width/ncolumns, height/nrows) )
            size = self._constrain_swatch_size(size)
        else:
            # Free-flowing
            if ncolors > 0:
                size = int(math.sqrt(float(width*height) / ncolors))
                size = self._constrain_swatch_size(size)
                ncolumns = max(1, min(ncolors, width / size))
                nrows = max(1, int(ncolors / ncolumns))
                if int(ncolors % ncolumns) > 0:
                    nrows += 1
                if nrows * size > height or ncolumns * size > width:
                    size = max(1, min(int(height / nrows),
                                      int(width / ncolumns)))
                    size = self._constrain_swatch_size(size)
                    ncolumns = max(1, min(ncolors, width / size))
                    nrows = max(1, int(ncolors / ncolumns))
                    if int(ncolors % ncolumns) > 0:
                        nrows += 1
            else:
                nrows = 0
                ncolumns = 0
                size = self._SWATCH_SIZE_NOMINAL
        self._rows = nrows
        self._columns = ncolumns
        self._swatch_size = size



    ## Palette monitoring

    def set_color_manager(self, mgr):
        ColorAdjusterWidget.set_color_manager(self, mgr)
        # Could be smarter about these: probably no need to redraw on
        # every little change.
        mgr.palette.info_changed += self._palette_changed_cb
        mgr.palette.match_changed += self._palette_changed_cb
        mgr.palette.sequence_changed += self._palette_changed_cb
        mgr.palette.color_changed += self._palette_changed_cb


    def _palette_changed_cb(self, palette, *args, **kwargs):
        """Called after each change made to the palette."""
        # Determine if the layout has changed since the last time the palette
        # was draw.
        layout_changed = False
        if None in (self._rows, self._columns):
            logger.debug("layout changed: null preexisting layout info")
            layout_changed = True
        if not layout_changed and palette.columns is not None:
            layout_changed = palette.columns != self._columns
            if layout_changed:
                logger.debug("layout changed: different number of columns")
        if not layout_changed:
            ncells = self._rows * self._columns
            ncolors = len(palette)
            if ncolors > ncells or ncolors <= ncells - self._columns:
                logger.debug("layout changed: cannot fit palette into "
                             "currently calculated space")
                layout_changed = True
        # Queue a resize (and an implicit redraw) if the layout has changed,
        # or just a redraw.
        if layout_changed:
            self._rows = None
            self._columns = None
            self.queue_resize()
            self._drag_insertion_index = None
            self._tooltip_index = None
        else:
            logger.debug("layout unchanged, redraw")
            self.queue_draw()


    ## Pointer event handling


    def _motion_notify_cb(self, widget, event):
        x, y = event.x, event.y
        i = self.get_index_at_pos(x, y)
        # Set the tooltip.
        # Passing the tooltip through a value of None is necessary for its
        # position on the screen to be updated to where the pointer is. Setting
        # it to None, and then to the desired value must happen in two separate
        # events for the tooltip window position update to be honoured.
        if i is None:
            # Not over a colour, so use the static default
            if self._tooltip_index not in (-1, -2):
                # First such event: reset the tooltip.
                self._tooltip_index = -1
                self.set_has_tooltip(False)
                self.set_tooltip_text("")
            elif self._tooltip_index != -2:
                # Second event over a non-colour: set the tooltip text.
                self._tooltip_index = -2
                self.set_has_tooltip(True)
                self.set_tooltip_text(self.STATIC_TOOLTIP_TEXT)
        elif self._tooltip_index != i:
            # Mouse pointer has moved to a different colour, or away
            # from the two states above.
            if self._tooltip_index is not None:
                # First event for this i: reset the tooltip.
                self._tooltip_index = None
                self.set_has_tooltip(False)
                self.set_tooltip_text("")
            else:
                # Second event for this i: set the desired tooltip text.
                self._tooltip_index = i
                mgr = self.get_color_manager()
                tip = mgr.palette.get_color_name(i)
                color = mgr.palette.get_color(i)
                if color is None:
                    tip = _("Empty palette slot (drag a color here)")
                elif tip is None or tip.strip() == "":
                    tip = ""  # Anonymous colors don't get tooltips
                self.set_has_tooltip(True)
                self.set_tooltip_text(tip)


    def _button_press_cb(self, widget, event):
        """Select color on a single click."""
        if event.type == Gdk.EventType.BUTTON_PRESS:
            if event.button == 1:
                x, y = event.x, event.y
                i = self.get_index_at_pos(x, y)
                mgr = self.get_color_manager()
                if not self.can_select_empty:
                    if mgr.palette.get_color(i) is None:
                        return False
                mgr.palette.set_match_position(i)
                mgr.palette.set_match_is_approx(False)


    def _button_release_cb(self, widget, event):
        pass


    ## Dimensions and sizing

    @classmethod
    def _constrain_swatch_size(cls, size):
        size = min(cls._SWATCH_SIZE_MAX, max(cls._SWATCH_SIZE_MIN, size))
        # Restrict to multiples of 2 for patterns, plus one for the border
        if size % 2 == 0:
            size -= 1
        return size


    def _get_palette_dimensions(self):
        """Normalized palette dimensions: (ncolors, nrows, ncolumns).

        Row and columns figures are None if the layout is to be free-flowing.

        """
        mgr = self.get_color_manager()
        ncolumns = mgr.palette.get_columns()
        ncolors = len(mgr.palette)
        if ncolumns is None or ncolumns < 1:
            nrows = None
            ncolumns = None
        else:
            ncolumns = int(ncolumns)
            if ncolors > 0:
                ncolumns = min(ncolumns, ncolors)
                nrows = max(1, int(ncolors / ncolumns))
                if int(ncolors % ncolumns) > 0:
                    nrows += 1
            else:
                ncolumns = 1
                nrows = 1
        return (ncolors, nrows, ncolumns)


    def do_get_request_mode(self):
        """GtkWidget size negotiation implementation
        """
        ncolors, nrows, ncolumns = self._get_palette_dimensions()
        mode = Gtk.SizeRequestMode.HEIGHT_FOR_WIDTH
        if nrows and ncolumns:
            if nrows > ncolumns:
                mode = Gtk.SizeRequestMode.WIDTH_FOR_HEIGHT
        return mode


    def do_get_preferred_width(self):
        """GtkWidget size negotiation implementation.
        """
        ncolors, nrows, ncolumns = self._get_palette_dimensions()
        if ncolumns and ncolumns:
            # Horizontal fit, assume rows <= columns
            min_w = self._SWATCH_SIZE_MIN * ncolumns
            nat_w = self._SWATCH_SIZE_NOMINAL * ncolumns
        else:
            # Free-flowing, across and then down
            ncolumns = max(1, min(self._PREFERRED_COLUMNS, ncolors))
            min_w = self._SWATCH_SIZE_MIN
            nat_w = self._SWATCH_SIZE_NOMINAL * ncolumns
        return min_w, max(min_w, nat_w)


    def do_get_preferred_height_for_width(self, width):
        """GtkWidget size negotiation implementation.
        """
        ncolors, nrows, ncolumns = self._get_palette_dimensions()
        if nrows and ncolumns:
            # Horizontal fit
            swatch_size = self._constrain_swatch_size(int(width / ncolumns))
            min_h = self._SWATCH_SIZE_MIN * nrows
            nat_h = swatch_size * nrows
        else:
            # Free-flowing, across and then down
            # Since s = sqrt((w*h)/n),
            min_h = int((((self._SWATCH_SIZE_MIN)**2)*ncolors) / width)
            nat_h = int((((self._SWATCH_SIZE_NOMINAL)**2)*ncolors) / width)
        return min_h, max(min_h, nat_h)


    def do_get_preferred_height(self):
        """GtkWidget size negotiation implementation.
        """
        ncolors, nrows, ncolumns = self._get_palette_dimensions()
        if nrows and ncolumns:
            # Vertical fit, assume rows > columns
            min_h = self._SWATCH_SIZE_MIN * nrows
            nat_h = self._SWATCH_SIZE_NOMINAL * nrows
        else:
            # Height required for our own minimum width (note do_())
            min_w, nat_w = self.do_get_preferred_width()
            min_h, nat_h = self.do_get_preferred_height_for_width(min_w)
        return min_h, max(min_h, nat_h)


    def do_get_preferred_width_for_height(self, height):
        """GtkWidget size negotiation implementation.
        """
        ncolors, nrows, ncolumns = self._get_palette_dimensions()
        if nrows and ncolumns:
            # Vertical fit
            swatch_size = self._constrain_swatch_size(int(height / nrows))
            min_w = self._SWATCH_SIZE_MIN * ncolumns
            nat_w = swatch_size * ncolumns
        else:
            # Just the minimum and natural width (note do_())
            min_w, nat_w = self.do_get_preferred_width()
        return min_w, max(min_w, nat_w)


    def _get_background_size(self):
        # HACK. it's quicker for this widget to render in the foreground
        return 1, 1


    def get_background_validity(self):
        return 1


    def render_background_cb(self, cr, wd, ht):
        return


    def _paint_palette_layout(self, cr):
        mgr = self.get_color_manager()
        if mgr.palette is None:
            return
        state = self.get_state_flags()
        style = self.get_style_context()
        bg_rgba = style.get_background_color(state)
        bg_col = RGBColor.new_from_gdk_rgba(bg_rgba)
        dx, dy = self.get_painting_offset()
        _palette_render(mgr.palette, cr,
                        rows=self._rows, columns=self._columns,
                        swatch_size=self._swatch_size,
                        bg_color=bg_col,
                        offset_x=dx, offset_y=dy,
                        rtl=False)


    def _paint_marker(self, cr, x, y, insert=False,
                      bg_rgb=(0,0,0), fg_rgb=(1,1,1),
                      bg_dash=[1,2], fg_dash=[1,2],
                      bg_width=2, fg_width=1):
        cr.save()
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        size = self._swatch_size
        w = h = size
        # Background (shadow)
        cr.set_source_rgb(*bg_rgb)
        cr.set_line_width(bg_width)
        if insert:
            cr.move_to(x, y-1)
            cr.line_to(x, y+h)
            sw = int(w/4)
            cr.move_to(x-sw, y-1)
            cr.line_to(x+sw, y-1)
            cr.move_to(x-sw, y+h)
            cr.line_to(x+sw, y+h)
        else:
            cr.rectangle(x, y, w-1, h-1)
        cr.set_dash(bg_dash)
        cr.stroke_preserve()
        # Foreground
        cr.set_line_width(fg_width)
        cr.set_dash(fg_dash)
        cr.set_source_rgb(*fg_rgb)
        cr.stroke()
        cr.restore()


    def paint_foreground_cb(self, cr, wd, ht):
        mgr = self.get_color_manager()
        if len(mgr.palette) < 1:
            return

        # Palette cells
        self._paint_palette_layout(cr)

        # Highlights
        cr.set_line_cap(cairo.LINE_CAP_SQUARE)

        # Current drag/drop target
        if self._drag_insertion_index is not None:
            i = self._drag_insertion_index
            x, y = self.get_position_for_index(i)
            insert = mgr.palette.get_color(i) is not None
            self._paint_marker(cr, x, y, insert=insert)
        # Position of the previous click
        if self.show_matched_color:
            i = mgr.palette.match_position
            if i is not None:
                x, y = self.get_position_for_index(i)
                marker_args = [cr, x, y]
                marker_kw = dict(bg_width=3, fg_width=1,
                                 bg_dash=[2,3], fg_dash=[2,3])
                if not mgr.palette.match_is_approx:
                    marker_kw.update(dict(bg_width=4, fg_width=1))
                self._paint_marker(*marker_args, **marker_kw)


    def get_position_for_index(self, i):
        """Gets the X and Y positions for a color cell at the given index"""
        if None in (self._rows, self._columns):
            return 0, 0
        dx, dy = self.get_painting_offset()
        s_w = s_h = self._swatch_size
        c = i % self._columns
        r = int(i / self._columns)
        x = 0.5 + c*s_w
        y = 0.5 + r*s_h
        return x+dx, y+dy


    def get_painting_offset(self):
        if None in (self._rows, self._columns):
            return 0, 0
        sw = sh = self._swatch_size
        l_wd = sw * self._columns
        l_ht = sh * self._rows
        alloc = self.get_allocation()
        wd, ht = alloc.width, alloc.height
        dx, dy = 0, 0
        if l_wd < wd:
            dx = (wd - l_wd)/2.0
        if l_ht < ht:
            dy = (ht - l_ht)/2.0
        return 1+int(dx), 1+int(dy)


    def get_color_at_position(self, x, y):
        i = self.get_index_at_pos(x, y)
        if i is not None:
            mgr = self.get_color_manager()
            col = mgr.palette.get_color(i)
            if col is None:
                return None
            return col


    def set_color_at_position(self, x, y, color):
        i = self.get_index_at_pos(x, y)
        mgr = self.get_color_manager()
        if i is None:
            mgr.palette.append(color)
        else:
            mgr.palette[i] = color
        ColorAdjusterWidget.set_color_at_position(self, x, y, color)


    def get_index_at_pos(self, x, y):
        mgr = self.get_color_manager()
        if mgr.palette is None:
            return None
        if None in (self._rows, self._columns):
            return None
        dx, dy = self.get_painting_offset()
        x -= dx
        y -= dy
        s_wd = s_ht = self._swatch_size
        r = int(y // s_ht)
        c = int(x // s_wd)
        if r < 0 or r >= self._rows:
            return None
        if c < 0 or c >= self._columns:
            return None
        i = r*self._columns + c
        if i >= len(mgr.palette):
            return None
        return i


    ## Drag handling overrides


    def drag_motion_cb(self, widget, context, x, y, t):
        if "application/x-color" not in map(str, context.list_targets()):
            return False

        # Default action: copy means insert or overwrite
        action = Gdk.DragAction.COPY

        # Update the insertion marker
        i = self.get_index_at_pos(x, y)
        if i != self._drag_insertion_index:
            self.queue_draw()
        self._drag_insertion_index = i

        # Dragging around inside the widget implies moving, by default
        source_widget = Gtk.drag_get_source_widget(context)
        if source_widget is self:
            action = Gdk.DragAction.MOVE
            if i is None:
                action = Gdk.DragAction.DEFAULT  # it'll be ignored
            else:
                mgr = self.get_color_manager()
                if mgr.palette.get_color(i) is None:
                    # Empty swatch, convert moves to copies
                    action = Gdk.DragAction.COPY

        # Cursor and status update
        Gdk.drag_status(context, action, t)


    def drag_data_received_cb(self, widget, context, x, y,
                               selection, info, t):
        if "application/x-color" not in map(str, context.list_targets()):
            return False
        data = selection.get_data()
        data_type = selection.get_data_type()
        fmt = selection.get_format()
        logger.debug("drag-data-received: got type=%r", data_type)
        logger.debug("drag-data-received: got fmt=%r", fmt)
        logger.debug("drag-data-received: got data=%r len=%r", data, len(data))
        color = RGBColor.new_from_drag_data(data)
        target_index = self.get_index_at_pos(x, y)

        mgr = self.get_color_manager()
        if Gtk.drag_get_source_widget(context) is self:
            # Move/copy
            current_index = mgr.palette.match_position
            logger.debug("Move/copy %r -> %r", current_index, target_index)
            assert current_index is not None
            mgr.palette.reposition(current_index, target_index)
        else:
            if target_index is None:
                # Append if the drop wasn't over a swatch
                target_index = len(mgr.palette)
            else:
                # Insert before populated swatches, or overwrite empties
                if mgr.palette.get_color(target_index) is None:
                    mgr.palette.pop(target_index)
            mgr.palette.insert(target_index, color)
        self.queue_draw()
        self._drag_insertion_index = None
        context.finish(True, True, t)
        self.set_managed_color(color)
        mgr.palette.set_match_position(target_index)


    def drag_end_cb(self, widget, context):
        self._drag_insertion_index = None
        self.queue_draw()


    def drag_leave_cb(self, widget, context, time):
        self._drag_insertion_index = None
        self.queue_draw()


## Loading and saving of palettes via a dialog


def palette_load_via_dialog(title, parent=None, preview=None,
                            shortcuts=None):
    """Runs a file chooser dialog, returning a palette or `None`.

    The dialog is both modal and blocking. A new `Palette` object is returned
    if the load was successful. The value `None` is returned otherwise.

    :param parent: specifies the parent window
    :param title: dialog title
    :param preview: any preview widget with a ``set_palette()`` method
    :param shortcuts: optional list of shortcut folders

    """
    dialog = Gtk.FileChooserDialog(
      title=title,
      parent=parent,
      action=Gtk.FileChooserAction.OPEN,
      buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.REJECT,
               Gtk.STOCK_OPEN, Gtk.ResponseType.ACCEPT),
      )
    if preview is not None:
        dialog.set_preview_widget(preview)
        dialog.connect("update-preview",
                       _palette_loadsave_dialog_update_preview_cb,
                       preview)
    if shortcuts is not None:
        for shortcut in shortcuts:
            dialog.add_shortcut_folder(shortcut)
    dialog.set_do_overwrite_confirmation(True)
    filter = Gtk.FileFilter()
    filter.add_pattern("*.gpl")
    filter.set_name(_("GIMP palette file (*.gpl)"))
    dialog.add_filter(filter)
    filter = Gtk.FileFilter()
    filter.add_pattern("*")
    filter.set_name(_("All files (*)"))
    dialog.add_filter(filter)
    response_id = dialog.run()
    palette = None
    if response_id == Gtk.ResponseType.ACCEPT:
        filename = dialog.get_filename()
        logger.info("Loading palette from %r", filename)
        palette = Palette(filename=filename)
    dialog.destroy()
    return palette



def palette_save_via_dialog(palette, title, parent=None, preview=None):
    """Runs a file chooser dialog for saving.

    The dialog is both modal and blocking. Returns True if the file was saved
    successfully.

    :paraqm palette: the palette to save
    :param parent: specifies the parent window
    :param title: dialog title
    :param preview: any preview widget with a ``set_palette()`` method

    """
    dialog = Gtk.FileChooserDialog(
      title=title,
      parent=parent,
      action=Gtk.FileChooserAction.SAVE,
      buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.REJECT,
               Gtk.STOCK_SAVE, Gtk.ResponseType.ACCEPT),
      )
    if preview is not None:
        dialog.set_preview_widget(preview)
        dialog.connect("update-preview",
                       _palette_loadsave_dialog_update_preview_cb,
                       preview)
    dialog.set_do_overwrite_confirmation(True)
    filter = Gtk.FileFilter()
    filter.add_pattern("*.gpl")
    filter.set_name(_("GIMP palette file (*.gpl)"))
    dialog.add_filter(filter)
    filter = Gtk.FileFilter()
    filter.add_pattern("*")
    filter.set_name(_("All files (*)"))
    dialog.add_filter(filter)
    response_id = dialog.run()
    result = False
    if response_id == Gtk.ResponseType.ACCEPT:
        filename = dialog.get_filename()
        filename = re.sub(r'[.]?(?:[Gg][Pp][Ll])?$', "", filename)
        palette_name = os.path.basename(filename)
        filename += ".gpl"
        logger.info("Saving palette to %r", filename)
        # FIXME: this can overwrite files without prompting the user, if
        # the name hacking above changed the filename.  Should do the name
        # tweak within the dialog somehow and get that to confirm.
        fp = open(filename, 'w')
        palette.save(fp)
        fp.flush()
        fp.close()
        result = True
    dialog.destroy()
    return result


def _palette_loadsave_dialog_update_preview_cb(dialog, preview):
    """Updates the preview widget when loading/saving palettes via dialog"""
    filename = dialog.get_preview_filename()
    palette = None
    if filename is not None and os.path.isfile(filename):
        try:
            palette = Palette(filename=filename)
        except Exception, ex:
            logger.warning("Couldn't update preview widget: %s", str(ex))
            return
    if palette is not None and len(palette) > 0:
        dialog.set_preview_widget_active(True)
        preview.set_palette(palette)
        preview.queue_draw()
    else:
        dialog.set_preview_widget_active(False)


## Palette rendering using Cairo


def _palette_render(palette, cr, rows, columns, swatch_size,
                    bg_color, offset_x=0, offset_y=0,
                    rtl=False):
    """Renders a Palette according to a precalculated grid.

    :param cr: a Cairo context
    :param rows: number of rows in the layout
    :param columns: number of columns in the layout
    :param swatch_size: size of each swatch, in pixels
    :param bg_color: a `uicolor.UIColor` used when rendering the patterned
                      placeholder for an empty palette slot.
    :param rtl: layout direction: set to True to render right to left,
                 instead of left to right. Currently ignored.
    """

    HIGHLIGHT_DLUMA = 0.05

    if len(palette) == 0:
        return
    if rows is None or columns is None:
        return

    cr.save()
    cr.translate(offset_x, offset_y)

    # Sizes and colours
    swatch_w = swatch_h = swatch_size
    light_col = HCYColor(color=bg_color)
    dark_col = HCYColor(color=bg_color)
    light_col.y = clamp(light_col.y + HIGHLIGHT_DLUMA, 0, 1)
    dark_col.y = clamp(dark_col.y - HIGHLIGHT_DLUMA, 0, 1)

    # Upper left outline (bottom right is covered below by the
    # individual chips' shadows)
    ul_col = HCYColor(color=bg_color)
    ul_col.y *= 0.75
    ul_col.c *= 0.5
    cr.set_line_join(cairo.LINE_JOIN_ROUND)
    cr.set_line_cap(cairo.LINE_CAP_ROUND)
    cr.set_source_rgb(*ul_col.get_rgb())
    cr.move_to(0.5, rows*swatch_h - 1)
    cr.line_to(0.5, 0.5)
    row1cells = min(columns, len(palette)) # needed?
    cr.line_to(row1cells*swatch_w - 1, 0.5)
    cr.set_line_width(2)
    cr.stroke()

    # Draw into the predefined grid
    r = c = 0
    cr.set_line_width(1.0)
    cr.set_line_cap(cairo.LINE_CAP_SQUARE)
    for col in palette.iter_colors():
        s_x = c*swatch_w
        s_y = r*swatch_h
        s_w = swatch_w
        s_h = swatch_h

        # Select fill bg and pattern fg colours, Tango-style edge highlight
        # and lower-right shadow.
        if col is None:
            # Empty slot, fill with a pattern
            hi_rgb = light_col.get_rgb()
            fill_bg_rgb = dark_col.get_rgb()
            fill_fg_rgb = light_col.get_rgb()
            sh_col = HCYColor(color=bg_color)
            sh_col.y *= 0.75
            sh_col.c *= 0.5
            sh_rgb = sh_col.get_rgb()
        else:
            # Colour swatch
            hi_col = HCYColor(color=col)
            hi_col.y = min(hi_col.y * 1.1, 1)
            hi_col.c = min(hi_col.c * 1.1, 1)
            sh_col = HCYColor(color=col)
            sh_col.y *= 0.666
            sh_col.c *= 0.5
            hi_rgb = hi_col.get_rgb()
            fill_bg_rgb = col.get_rgb()
            fill_fg_rgb = None
            sh_rgb = sh_col.get_rgb()

        # Draw the swatch / colour chip
        cr.set_source_rgb(*sh_rgb)
        cr.rectangle(s_x, s_y, s_w, s_h)
        cr.fill()
        cr.set_source_rgb(*fill_bg_rgb)
        cr.rectangle(s_x, s_y, s_w-1, s_h-1)
        cr.fill()
        if fill_fg_rgb is not None:
            s_w2 = int((s_w-1) / 2)
            s_h2 = int((s_h-1) / 2)
            cr.set_source_rgb(*fill_fg_rgb)
            cr.rectangle(s_x, s_y, s_w2, s_h2)
            cr.fill()
            cr.rectangle(s_x+s_w2, s_y+s_h2, s_w2, s_h2)
            cr.fill()
        cr.set_source_rgb(*hi_rgb)
        cr.rectangle(s_x+0.5, s_y+0.5, s_w-2, s_h-2)
        cr.stroke()

        c += 1
        if c >= columns:
            c = 0
            r += 1

    cr.restore()


## Module testing

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import sys
    win = Gtk.Window()
    win.set_title("palette view")
    win.connect("destroy", lambda *a: Gtk.main_quit())
    mgr = ColorManager()
    spv = PaletteView()
    spv.grid.show_matched_color = True
    spv.grid.can_select_empty = True
    spv.set_color_manager(mgr)
    spv.set_size_request(150, 150)
    if len(sys.argv[1:]) > 0:
        palette_file = sys.argv[1] # GIMP palette file (*.gpl)
        palette = Palette(filename=palette_file)
        mgr.palette.update(palette)
    win.add(spv)
    win.show_all()
    Gtk.main()

