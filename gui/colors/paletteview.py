# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Viewer and editor widgets for palettes.
"""

import gui.pygtkcompat as pygtkcompat

import gtk
from gtk import gdk
import cairo
from gettext import gettext as _
import math
from copy import deepcopy
import os

from palette import Palette
from util import clamp
from uicolor import RGBColor
from uicolor import HCYColor
from uicolor import YCbCrColor
from adjbases import ColorAdjuster
from adjbases import ColorAdjusterWidget
from adjbases import ColorManager
from combined import CombinedAdjusterPage
from uimisc import borderless_button


PREFS_PALETTE_DICT_KEY = "colors.palette"
DATAPATH_PALETTES_SUBDIR = 'palettes'
DEFAULT_PALETTE_FILE = 'MyPaint_Default.gpl'


# Editor ideas:
#   - "Insert lighter/darker copy of row".
#   - repack palette (remove duplicates and blanks)
#   - sort palette by approx. hue+chroma binning, then luma variations


class PalettePage (CombinedAdjusterPage):
    """User-editable palette, as a `CombinedAdjuster` element.
    """


    def __init__(self):
        view = PaletteView()
        view.grid.show_current_index = True
        view.grid.manage_current_index = True
        view.can_select_empty = False
        self._adj = view
        self._edit_dialog = None


    @classmethod
    def get_properties_description(class_):
        return _("Palette properties")


    @classmethod
    def get_page_icon_name(self):
        return "gtk-select-color"


    @classmethod
    def get_page_title(self):
        return _("Palette")


    @classmethod
    def get_page_description(self):
        return _("Set the color from a loadable, editable palette.")


    def get_page_widget(self):
        """Page widget: returns the PaletteView adjuster widget itself.
        """
        # The PaletteNext and PalettePrev actions of the main
        # app require access to the PaletteView itself.
        return self._adj


    def set_color_manager(self, manager):
        CombinedAdjusterPage.set_color_manager(self, manager)
        self._adj.set_color_manager(manager)
        prefs = self._get_prefs()
        palette_dict = prefs.get(PREFS_PALETTE_DICT_KEY, None)
        if palette_dict is not None:
            palette = Palette.new_from_simple_dict(palette_dict)
        else:
            datapath = manager.get_data_path()
            palettes_dir = os.path.join(datapath, DATAPATH_PALETTES_SUBDIR)
            default = os.path.join(palettes_dir, DEFAULT_PALETTE_FILE)
            palette = Palette(filename=default)
        self._adj.set_palette(palette)


    def add_color_to_palette(self, color):
        # Used for "bookmarking" the current colour.
        self._adj.append_color(color)


    def show_properties(self):
        if self._edit_dialog is None:
            toplevel = self._adj.get_toplevel()
            dialog = PaletteEditorDialog(toplevel, self._adj)
            self._edit_dialog = dialog
        self._edit_dialog.run()


class PaletteEditorDialog (gtk.Dialog):
    """Dialog for editing, loading and saving the current palette.
    """

    def __init__(self, parent, target):
        gtk.Dialog.__init__(self, _("Palette Editor"), parent,
                            gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
                            (gtk.STOCK_CANCEL, gtk.RESPONSE_REJECT,
                             gtk.STOCK_OK, gtk.RESPONSE_ACCEPT))
        self.set_position(gtk.WIN_POS_MOUSE)
        self._target = target #: The target PaletteView, where changes are sent
        view = PaletteView()
        view.set_size_request(400, 300)
        view.grid.show_current_index = True
        view.grid.manage_current_index = True
        view.grid.can_select_empty = True
        self._view = view

        self._mgr = ColorManager()
        self._mgr.set_color(RGBColor(1,1,1))
        view.set_color_manager(self._mgr)

        # Action buttons, positiopned down the right hand side
        action_bbox = gtk.VButtonBox()
        load_btn = self._load_button = gtk.Button(stock=gtk.STOCK_OPEN)
        save_btn = self._save_button = gtk.Button(stock=gtk.STOCK_SAVE)
        add_btn = self._add_button = gtk.Button(stock=gtk.STOCK_ADD)
        remove_btn = self._remove_button = gtk.Button(stock=gtk.STOCK_REMOVE)
        clear_btn = self._clear_button = gtk.Button(stock=gtk.STOCK_CLEAR)
        action_bbox.pack_start(load_btn)
        action_bbox.pack_start(save_btn)
        action_bbox.pack_start(add_btn)
        action_bbox.pack_start(remove_btn)
        action_bbox.pack_start(clear_btn)
        action_bbox.set_layout(gtk.BUTTONBOX_START)
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
        view.grid.current_index_observers.append(
          self._current_index_changed_cb)
        view.grid.palette_observers.append(self._palette_changed_cb)

        # Palette name and number of entries
        palette_details_hbox = gtk.HBox()
        palette_name_label = gtk.Label(_("Name:"))
        palette_name_label.set_tooltip_text(
          _("Name or description for this palette"))
        palette_name_entry = gtk.Entry()
        palette_name_entry.connect("changed", self._palette_name_changed_cb)
        self._palette_name_entry = palette_name_entry
        self._columns_adj = gtk.Adjustment(
          value=0, lower=0, upper=99,
          step_incr=1, page_incr=1, page_size=0 )
        self._columns_adj.connect("value-changed", self._columns_changed_cb)
        columns_label = gtk.Label(_("Columns:"))
        columns_label.set_tooltip_text(_("Number of columns"))
        columns_label.set_tooltip_text(_("Number of columns"))
        columns_spinbutton = gtk.SpinButton(
          adjustment=self._columns_adj,
          climb_rate=1.5,
          digits=0 )
        palette_details_hbox.set_spacing(0)
        palette_details_hbox.set_border_width(0)
        palette_details_hbox.pack_start(palette_name_label, False, False, 0)
        palette_details_hbox.pack_start(palette_name_entry, True, True, 6)
        palette_details_hbox.pack_start(columns_label, False, False, 6)
        palette_details_hbox.pack_start(columns_spinbutton, False, False, 0)

        color_name_hbox = gtk.HBox()
        color_name_label = gtk.Label(_("Color name:"))
        color_name_label.set_tooltip_text(_("Current colour's name"))
        color_name_entry = gtk.Entry()
        color_name_entry.connect("changed", self._color_name_changed_cb)
        color_name_entry.set_sensitive(False)
        self._color_name_entry = color_name_entry
        color_name_hbox.set_spacing(6)
        color_name_hbox.pack_start(color_name_label, False, False, 0)
        color_name_hbox.pack_start(color_name_entry, True, True, 0)

        palette_vbox = gtk.VBox()
        palette_vbox.set_spacing(12)
        palette_vbox.pack_start(palette_details_hbox, False, False)
        palette_vbox.pack_start(view, True, True)
        palette_vbox.pack_start(color_name_hbox, False, False)

        # Dialog contents
        # Main edit area to the left, buttons to the right
        hbox = gtk.HBox()
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
        # Each time the dialog is shown, clone the target's palette for
        # editing.
        self.vbox.show_all()
        palette = deepcopy(self._target.get_palette())
        name = palette.get_name()
        if name is None:
            name = ""
        self._palette_name_entry.set_text(name)
        self._columns_adj.set_value(palette.get_columns())
        self._view.set_palette(palette)


    def _palette_name_changed_cb(self, editable):
        name = editable.get_chars(0, -1)
        if name == "":
            name = None
        pal = self._view.get_palette()
        pal.set_name(name)
        self._view.set_palette(pal)


    def _columns_changed_cb(self, adj):
        ncolumns = int(adj.get_value())
        pal = self._view.get_palette()
        pal.set_columns(ncolumns)
        self._view.set_palette(pal)


    def _color_name_changed_cb(self, editable):
        name = editable.get_chars(0, -1)
        grid = self._view.grid
        palette = grid._palette
        i = grid._current_index
        if i is None:
            return
        old_name = palette.get_color_name(i)
        if name == "":
            name = None
        if name != old_name:
            palette.set_color_name(i, name)


    def _response_cb(self, widget, response_id):
        if response_id == gtk.RESPONSE_ACCEPT:
            palette = self._view.get_palette()
            self._target.set_palette(palette)
        self.hide()
        return True


    def _current_index_changed_cb(self, i):
        col_name_entry = self._color_name_entry
        remove_btn = self._remove_button
        palette = self._view.get_palette()
        if i is not None:
            col = palette.get_color(i)
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
        palette = self._view.grid._palette
        emptyish = len(palette) == 0
        if len(palette) == 1:
            if palette[0] is None:
                emptyish = True
        can_save = not emptyish
        can_clear = not emptyish
        can_remove = True
        if emptyish or self._view.grid._current_index is None:
            can_remove = False
        self._save_button.set_sensitive(can_save)
        self._remove_button.set_sensitive(can_remove)
        self._clear_button.set_sensitive(can_clear)


    def _palette_changed_cb(self, palette):
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
        palette = self._view.get_palette()
        i = grid._current_index
        if i is None:
            i = len(palette)
            palette.append(None)
        else:
            palette.insert(i, None)
            i += 1
        self._view.set_palette(palette)
        grid.set_current_index(i)


    def _remove_btn_clicked(self, button):
        grid = self._view.grid
        palette = self._view.get_palette()
        i = grid._current_index
        if i >= 0 and i < len(palette):
            palette.pop(i)
            if len(palette) == 0:
                palette.append(None)
            self._view.set_palette(palette)
            if i > 0:
                i -= 1
            grid.set_current_index(i)


    def _load_btn_clicked(self, button):
        preview = _PalettePreview()
        manager = self._target.get_color_manager()
        datapath = manager.get_data_path()
        palettes_dir = os.path.join(datapath, DATAPATH_PALETTES_SUBDIR)
        palette = Palette.load_via_dialog(title=_("Load palette"),
                                          parent=self,
                                          preview=preview,
                                          shortcuts=[palettes_dir])
        if palette is not None:
            self._view.set_palette(palette)


    def _save_btn_clicked(self, button):
        preview = _PalettePreview()
        pal = self._view.get_palette()
        pal.save_via_dialog(title=_("Save palette"), parent=self,
                            preview=preview)


    def _clear_btn_clicked(self, button):
        pal = Palette()
        pal.append(None)
        self._view.set_palette(pal)



class PaletteView (ColorAdjuster, gtk.ScrolledWindow):
    """Scrollable view of a palette.

    Palette entries can be clicked to select the colour, and all instances of
    the current shared colour in the palette are highlighted.

    """

    def __init__(self):
        gtk.ScrolledWindow.__init__(self)
        self.grid = _PaletteGridLayout()
        self.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        self.add_with_viewport(self.grid)
        self.set_palette(Palette())
        self.grid.current_index_observers.append(self._current_index_changed_cb)

    def set_palette(self, palette):
        self.grid.set_palette(palette)

    def get_palette(self):
        return self.grid.get_palette()

    def append_color(self, color):
        self.grid.append_color(color)

    def set_color_manager(self, mgr):
        self.grid.set_color_manager(mgr)
        ColorAdjuster.set_color_manager(self, mgr)

    def _current_index_changed_cb(self, i):
        pass
        # TODO: scroll the vertical adjuster to show i if necessary:
        # use self.grid.get_allocation() & calculate using the
        # adjuster's properties.


def outwards_from(n, i):
    """Search order for a sequence of len() n, outwards from index i.
    """
    assert i < n and i >= 0
    yield i
    for j in xrange(n):
        exhausted = True
        if i - j >= 0:
            yield i - j
            exhausted = False
        if i + j < n:
            yield i + j
            exhausted = False
        if exhausted:
            break


def color_distance(c1, c2):
    """Distance metric.

    Use a geometric YCbCr distance, as recommended by Graphics Programming
    with Perl, chapter 1, Martien Verbruggen.

    """
    c1 = YCbCrColor(color=c1)
    c2 = YCbCrColor(color=c2)
    d_Cb = c1.Cb - c2.Cb
    d_Cr = c1.Cr - c2.Cr
    d_Y = c1.Y - c2.Y
    return ((d_Cb**2) + (d_Cr**2) + (d_Y)**2) ** (1.0/3)



class _PalettePreview (gtk.DrawingArea):

    _palette = None

    def __init__(self):
        gtk.DrawingArea.__init__(self)
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

        self._palette.render(cr, rows=nrows, columns=ncolumns,
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
    is_drag_source = True
    has_details_dialog = True
    tooltip_text = _("Color swatch palette. Drop colors here and "
                     "drag them around to organize.")

    ## Layout constants
    _SWATCH_SIZE_MIN = 8
    _SWATCH_SIZE_MAX = 50
    _SWATCH_SIZE_NOMINAL = 20
    _PREFERRED_COLUMNS = 5 #: Preferred width in cells for free-flow mode.

    ## Member variables & defaults
    show_current_index = False #: Highlight the current index (last click)
    manage_current_index = False  #: Index follows the managed colour
    can_select_empty = False #: User can click on empty slots


    def __init__(self):
        ColorAdjusterWidget.__init__(self)
        # Sizing
        s = self._SWATCH_SIZE_NOMINAL
        self.set_size_request(s, s)
        self.connect("size-allocate", self._size_alloc_cb)
        # Current index
        self._current_index = None
        self._current_index_approx = False
        self._button_down = None
        self.connect("button-press-event", self._button_press_cb)
        self.connect_after("button-release-event", self._button_release_cb)
        self.current_index_observers = []
        self.current_index_observers.append(self._current_index_changed_cb)
        # Dragging
        self._drag_insertion_index = None
        self.connect("motion-notify-event", self._motion_notify_cb)
        self.add_events(gdk.POINTER_MOTION_MASK)
        # Palette to display
        self._palette = None
        self.palette_observers = []
        self.palette_observers.append(self._palette_changed_cb)
        self.set_palette(Palette())
        # Tooltips
        self_tooltip_index = None
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


    def set_current_index(self, i, approx=False):
        """Sets the current swatch to the given index.

        The current swatch is drawn with a dashed border. It can be marked as
        an approximate match, meaning it's drawn with a lighter outline.

        """
        if i is not None:
            if i < 0 or i >= len(self._palette):
                i = None
        updated = False
        if i != self._current_index:
            self._current_index = i
            updated = True
        if approx != self._current_index_approx:
            self._current_index_approx = approx
            updated = True
        if updated:
            for cb in self.current_index_observers:
                cb(i)


    def _current_index_changed_cb(self, i):
        self.queue_draw()


    def select_next(self):
        self._match_move_and_select(1)


    def select_previous(self):
        self._match_move_and_select(-1)


    def _match_move_and_select(self, delta_i):
        i = self._current_index
        new_index = i
        select_color = False
        if i is None:
            if not self.match_managed_color():
                new_index = None
        elif self._current_index_approx:
            new_index = i
            select_color = True
        else:
            while i < len(self._palette) and i >= 0:
                i += delta_i
                if self._palette[i] is not None:
                    new_index = i
                    select_color = True
                    break
            if i >= len(self._palette) or i < 0:
                new_index = self._current_index
                select_color = False
        if select_color:
            col = self._palette[new_index]
            if col is not None:
                self.set_managed_color(col)
        self.set_current_index(new_index)


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
                self.set_tooltip_text(self.tooltip_text)
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
                tip = self._palette.get_color_name(i)
                color = self._palette.get_color(i)
                if color is None:
                    tip = _("Empty palette slot (drag a color here)")
                elif tip is None or tip.strip() == "":
                    tip = self.tooltip_text   # would None be nicer?
                self.set_has_tooltip(True)
                self.set_tooltip_text(tip)
        return False


    def _button_press_cb(self, widget, event):
        self._button_down = event.button
        if self._button_down == 1:
            if event.type == gdk.BUTTON_PRESS:
                x, y = event.x, event.y
                i = self.get_index_at_pos(x, y)
                if not self.can_select_empty:
                    if self._palette.get_color(i) is None:
                        return
                self.set_current_index(i)


    def _button_release_cb(self, widget, event):
        self._button_down = None


    @classmethod
    def _constrain_swatch_size(cls, size):
        size = min(cls._SWATCH_SIZE_MAX, max(cls._SWATCH_SIZE_MIN, size))
        # Restrict to multiples of 2 for patterns, plus one for the border
        if size % 2 == 0:
            size -= 1
        return size


    def _get_palette_dimensions(self):
        """Normalized palette dimensions: (ncolors, nrows, ncolumns).

        Row and columns figures are None if the layout it to be free-flowing.

        """
        ncolumns = self._palette.get_columns()
        ncolors = len(self._palette)
        if ncolumns is None or ncolumns < 1:
            nrows = None
            ncolumns = None
        else:
            ncolumns = min(int(ncolumns), ncolors)
            nrows = max(1, int(ncolors / ncolumns))
            if int(ncolors % ncolumns) > 0:
                nrows += 1
        return (ncolors, nrows, ncolumns)


    def do_get_request_mode(self):
        """GtkWidget size negotiation implementation
        """
        ncolors, nrows, ncolumns = self._get_palette_dimensions()
        mode = gtk.SizeRequestMode.HEIGHT_FOR_WIDTH
        if nrows and ncolumns:
            if nrows > ncolumns:
                mode = gtk.SizeRequestMode.WIDTH_FOR_HEIGHT
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
        return min_w, nat_w


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
        return min_h, nat_h


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
        return min_h, nat_h


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
        return min_w, nat_w


    def update_cb(self):
        """Callback: managed color updated.
        """
        self._drag_insertion_index = None
        ColorAdjusterWidget.update_cb(self)
        if self.manage_current_index and not self._button_down:
            self.match_managed_color()


    def match_managed_color(self):
        """Moves current index to the most similar colour to the managed one.

        The matching algorithm favours exact or near-exact matches which are
        close in index number to the current index. If the current index is
        unset, this search starts at 0. If there are no exact or near-exact
        matches, a looser approximate match will be used, again favouring
        matches with nearby index numbers. Returns true if the match succeeded.

        """
        col_m = self.get_managed_color()
        if self._current_index is not None:
            search_order = outwards_from(len(self._palette),
                                         self._current_index)
        else:
            search_order = xrange(len(self._palette))
        bestmatch_i = None
        bestmatch_d = None
        is_approx = True
        for i in search_order:
            col = self._palette[i]
            if col is None:
                continue
            # Closest exact or near-exact match by index distance (according to
            # the search_order). Considering near-exact matches as equivalent
            # to exact matches improves the feel of PaletteNext and
            # PalettePrev.
            d = color_distance(col_m, col)
            if col == col_m or d < 0.06:
                # Measuring over a blend into solid equiluminant 0-chroma
                # grey for the orange #DA5D2E with an opaque but feathered
                # brush made huge, and picking just inside the point where the
                # palette widget begins to call it approximate:
                #
                # 0.05 is a difference only discernible (to me) by tilting LCD
                # 0.066 to 0.075 appears slightly greyer for large areas
                # 0.1 and above is very clearly distinct
                bestmatch_i = i
                is_approx = False
                break
            if bestmatch_d is None or d < bestmatch_d:
                bestmatch_i = i
                bestmatch_d = d
        # If there are no exact or near-exact matches, choose the most similar
        # colour anywhere in the palette.
        if bestmatch_i is not None:
            self.set_current_index(bestmatch_i, is_approx)
            return True
        return False


    def set_palette(self, palette):
        self._palette = palette
        self._call_palette_observers()


    def _call_palette_observers(self):
        for cb in self.palette_observers:
            cb(self._palette)


    def _palette_changed_cb(self, palette):
        """Called after each change made to the palette.
        """
        # Assume insertion and selection indices are no longer valid
        self._drag_insertion_index = None
        self._tooltip_index = None
        self.set_current_index(None)
        # Clear drawing and layout state, and arrange for a redraw
        self._swatch_size = None
        self._rows = None
        self._columns = None
        self.queue_resize()  # maybe?
        self.queue_draw()
        # Store the changed palette to the prefs
        prefs = self._get_prefs()
        prefs[PREFS_PALETTE_DICT_KEY] = palette.to_simple_dict()


    def get_palette(self):
        return deepcopy(self._palette)


    def append_color(self, color):
        # Select the final occurence of the colour if it's already present
        for i, col in enumerate(self._palette.iter_colors()):
            if col == color:
                self.set_current_index(i)
                return

        # Append new colour and select that
        i = len(self._palette)
        self._palette.append(deepcopy(color))
        self.set_current_index(i)
        self._call_palette_observers()


    def _get_background_size(self):
        # HACK. it's quicker for this widget to render in the foreground
        return 1, 1


    def get_background_validity(self):
        return 1


    def render_background_cb(self, cr, wd, ht):
        return


    def _paint_palette_layout(self, cr):
        if self._palette is None:
            return
        state = self.get_state_flags()
        style = self.get_style_context()
        bg_rgba = style.get_background_color(state)
        bg_col = RGBColor.new_from_gdk_rgba(bg_rgba)
        dx, dy = self.get_painting_offset()
        self._palette.render(cr, rows=self._rows, columns=self._columns,
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
        cr.set_line_width(bg_width)
        cr.set_dash(bg_dash)
        cr.set_source_rgb(*bg_rgb)
        cr.rectangle(x, y, w-1, h-1)
        cr.stroke_preserve()
        cr.set_line_width(fg_width)
        cr.set_dash(fg_dash)
        cr.set_source_rgb(*fg_rgb)
        cr.stroke()
        cr.restore()


    def paint_foreground_cb(self, cr, wd, ht):
        # Palette cells
        self._paint_palette_layout(cr)

        # Highlights
        cr.set_line_cap(cairo.LINE_CAP_SQUARE)

        # Current drag/drop target
        if self._drag_insertion_index is not None:
            i = self._drag_insertion_index
            x, y = self.get_position_for_index(i)
            self._paint_marker(cr, x, y)
        # Position of the previous click
        if self.show_current_index:
            if self._current_index is not None:
                i = self._current_index
                x, y = self.get_position_for_index(i)
                marker_args = [cr, x, y]
                marker_kw = dict(bg_width=3, fg_width=1,
                                 bg_dash=[2,3], fg_dash=[2,3])
                if not self._current_index_approx:
                    marker_kw.update(dict(bg_width=4, fg_width=1))
                self._paint_marker(*marker_args, **marker_kw)


    def get_indices_for_color(self, col):
        if self._palette is None:
            return
        index = 0
        s_w = s_h = self._swatch_size
        rgb0 = col.get_rgb()
        for c in self._palette.iter_colors():
            if c is not None:
                rgb1 = c.get_rgb()
                d = sum([abs(rgb1[i]-rgb0[i]) for i in 0,1,2])
                if d < 0.001:
                    yield index
            index += 1


    def get_position_for_index(self, i):
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
            col = self._palette.get_color(i)
            if col is None:
                return None
            return col


    def set_color_at_position(self, x, y, color):
        i = self.get_index_at_pos(x, y)
        if i is None:
            self._palette.append(color)
        else:
            self._palette[i] = color
        self._call_palette_observers()
        ColorAdjusterWidget.set_color_at_position(self, x, y, color)


    def get_index_at_pos(self, x, y):
        if self._palette is None:
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
        if i >= len(self._palette):
            return None
        return i


    ## Drag handling overrides


    def _drag_motion_cb(self, widget, context, x, y, t):
        if "application/x-color" not in context.targets:
            return False
        action = None
        source_widget = context.get_source_widget()

        # Assume the target is not an empty swatch for now.
        if source_widget is self:
            action = gdk.ACTION_MOVE
        else:
            action = gdk.ACTION_COPY

        # Update the insertion marker
        i = self.get_index_at_pos(x, y)
        if i != self._drag_insertion_index:
            self.queue_draw()
        self._drag_insertion_index = i

        # Dragging around inside the widget allows more feedback
        if self is source_widget:
            if i is None:
                action = gdk.ACTION_DEFAULT  # it'll be ignored
            else:
                if self._palette.get_color(i) is None:
                    # Empty swatch, convert moves to copies
                    action = gdk.ACTION_COPY

        # Cursor and status update
        context.drag_status(action, t)


    def _drag_data_received_cb(self, widget, context, x, y,
                               selection, info, t):
        if "application/x-color" not in context.targets:
            return False
        color = RGBColor.new_from_drag_data(selection.data)
        target_index = self.get_index_at_pos(x, y)
        source_widget = context.get_source_widget()

        if self is source_widget:
            # Move/copy
            assert self._current_index is not None
            self._palette.move(self._current_index, target_index)
        else:
            if target_index is None:
                # Append if the drop wasn't over a swatch
                target_index = len(self._palette)
            else:
                # Insert before populated swatches, or overwrite empties
                if self._palette.get_color(target_index) is None:
                    self._palette.pop(target_index)
            self._palette.insert(target_index, color)
        self._call_palette_observers()
        self.queue_draw()
        self._drag_insertion_index = None
        context.finish(True, True, t)
        self.set_managed_color(color)
        self.set_current_index(target_index)


    def _drag_end_cb(self, widget, context):
        self._drag_insertion_index = None
        self.queue_draw()


    def _drag_leave_cb(self, widget, context, time):
        self._drag_insertion_index = None
        self.queue_draw()


if __name__ == '__main__':
    from adjbases import ColorManager
    import sys
    win = gtk.Window()
    win.set_title("palette view")
    win.connect("destroy", lambda *a: gtk.main_quit())
    mgr = ColorManager()
    spv = PaletteView()
    spv.grid.show_current_index = True
    spv.grid.can_select_empty = True
    spv.set_color_manager(mgr)
    spv.set_size_request(150, 150)
    if len(sys.argv[1:]) > 0:
        palette_file = sys.argv[1] # GIMP palette file (*.gpl)
        palette = Palette(filename=palette_file)
        spv.set_palette(palette)
    win.add(spv)
    win.show_all()
    gtk.main()

