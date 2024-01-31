# This file is part of MyPaint.
# Copyright (C) 2012-2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Viewer and editor widgets for palettes."""

# Editor ideas
#   - "Insert lighter/darker copy of row".
#   - repack palette (remove duplicates and blanks)
#   - sort palette by approx. hue+chroma binning, then luma variations


## Imports
from __future__ import division, print_function

import math
import os
import re
import logging
from io import open

from lib.gibindings import Gdk
from lib.gibindings import Gtk
from lib.gibindings import GLib
import cairo
from lib.gettext import C_

from .util import clamp
from lib.palette import Palette
from lib.color import RGBColor
from lib.color import HCYColor
from lib.color import HSVColor
import gui.uicolor
from .adjbases import ColorAdjuster
from .adjbases import ColorAdjusterWidget
from .adjbases import ColorManager
from .adjbases import DATAPATH_PALETTES_SUBDIR
from .combined import CombinedAdjusterPage

from lib.pycompat import unicode


logger = logging.getLogger(__name__)


## Class defs

class PalettePage (CombinedAdjusterPage):
    """User-editable palette, as a `CombinedAdjuster` element."""

    def __init__(self):
        view = PaletteView()
        view.grid.show_matched_color = True
        view.can_select_empty = False
        self._adj = view
        self._edit_dialog = None

    @classmethod
    def get_properties_description(cls):
        return C_(
            "palette panel: properties button tooltip",
            "Palette properties",
        )

    @classmethod
    def get_page_icon_name(cls):
        return "mypaint-tool-color-palette"

    @classmethod
    def get_page_title(cls):
        return C_("palette panel tab tooltip title", "Palette")

    @classmethod
    def get_page_description(cls):
        return C_(
            "palette panel tab tooltip description",
            "Set the color from a loadable, editable palette.",
        )

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
    """Dialog for editing, loading and saving the current palette."""

    _UNTITLED_PALETTE_NAME = C_(
        "palette editor dialog: palette name entry",
        "Untitled Palette",
    )

    def __init__(self, parent, target_color_manager):
        Gtk.Dialog.__init__(
            self,
            title=C_("palette editor dialog: title", "Palette Editor"),
            transient_for=parent,
            modal=True,
            destroy_with_parent=True,
        )
        self.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.REJECT)
        self.add_button(Gtk.STOCK_OK, Gtk.ResponseType.ACCEPT)
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
        self._mgr = ColorManager(prefs={},
                                 datapath=target_color_manager.get_data_path())
        self._mgr.set_color(RGBColor(1, 1, 1))
        view.set_color_manager(self._mgr)

        # Action buttons, positiopned down the right hand side
        action_bbox = Gtk.VButtonBox()
        load_btn = self._load_button = Gtk.Button(stock=Gtk.STOCK_OPEN)
        save_btn = self._save_button = Gtk.Button(stock=Gtk.STOCK_SAVE)
        add_btn = self._add_button = Gtk.Button(stock=Gtk.STOCK_ADD)
        remove_btn = self._remove_button = Gtk.Button(stock=Gtk.STOCK_REMOVE)
        clear_btn = self._clear_button = Gtk.Button(stock=Gtk.STOCK_CLEAR)
        action_bbox.pack_start(load_btn, True, True, 0)
        action_bbox.pack_start(save_btn, True, True, 0)
        action_bbox.pack_start(add_btn, True, True, 0)
        action_bbox.pack_start(remove_btn, True, True, 0)
        action_bbox.pack_start(clear_btn, True, True, 0)
        action_bbox.set_layout(Gtk.ButtonBoxStyle.START)
        load_btn.connect("clicked", self._load_btn_clicked)
        save_btn.connect("clicked", self._save_btn_clicked)
        remove_btn.connect("clicked", self._remove_btn_clicked)
        add_btn.connect("clicked", self._add_btn_clicked)
        clear_btn.connect("clicked", self._clear_btn_clicked)
        load_btn.set_tooltip_text(C_(
            "palette editor dialog: action buttons: tooltips",
            "Load from a GIMP palette file",
        ))
        save_btn.set_tooltip_text(C_(
            "palette editor dialog: action buttons: tooltips",
            "Save to a GIMP palette file",
        ))
        add_btn.set_tooltip_text(C_(
            "palette editor dialog: action buttons: tooltips",
            "Add a new empty swatch",
        ))
        remove_btn.set_tooltip_text(C_(
            "palette editor dialog: action buttons: tooltips",
            "Remove the current swatch",
        ))
        clear_btn.set_tooltip_text(C_(
            "palette editor dialog: action buttons: tooltips",
            "Remove all swatches",
        ))

        # Button initial state and subsequent updates
        remove_btn.set_sensitive(False)
        self._mgr.palette.match_changed += self._palette_match_changed_cb
        self._mgr.palette.info_changed += self._palette_changed_cb
        self._mgr.palette.sequence_changed += self._palette_changed_cb
        self._mgr.palette.color_changed += self._palette_changed_cb

        # Palette name and number of entries
        palette_details_hbox = Gtk.HBox()
        palette_name_label = Gtk.Label(label=C_(
            "palette editor dialog: palette name/title entry: label",
            "Title:",
        ))
        palette_name_label.set_tooltip_text(C_(
            "palette editor dialog: palette name/title entry: tooltip",
            "Name or description for this palette",
        ))
        palette_name_entry = Gtk.Entry()
        palette_name_entry.connect("changed", self._palette_name_changed_cb)
        self._palette_name_entry = palette_name_entry
        self._columns_adj = Gtk.Adjustment(
            value=0, lower=0, upper=99,
            step_increment=1, page_increment=1, page_size=0
        )
        self._columns_adj.connect("value-changed", self._columns_changed_cb)
        columns_label = Gtk.Label(label=C_(
            "palette editor dialog: number-of-columns spinbutton: title",
            "Columns:"
        ))
        columns_label.set_tooltip_text(C_(
            "palette editor dialog: number-of-columns spinbutton: tooltip",
            "Number of columns",
        ))
        columns_spinbutton = Gtk.SpinButton(
            adjustment=self._columns_adj,
            climb_rate=1.5,
            digits=0
        )
        palette_details_hbox.set_spacing(0)
        palette_details_hbox.set_border_width(0)
        palette_details_hbox.pack_start(palette_name_label, False, False, 0)
        palette_details_hbox.pack_start(palette_name_entry, True, True, 6)
        palette_details_hbox.pack_start(columns_label, False, False, 6)
        palette_details_hbox.pack_start(columns_spinbutton, False, False, 0)

        color_name_hbox = Gtk.HBox()
        color_name_label = Gtk.Label(label=C_(
            "palette editor dialog: color name entry: label",
            "Color name:",
        ))
        color_name_label.set_tooltip_text(C_(
            "palette editor dialog: color name entry: tooltip",
            "Current color's name",
        ))
        color_name_entry = Gtk.Entry()
        color_name_entry.connect("changed", self._color_name_changed_cb)
        color_name_entry.set_sensitive(False)
        self._color_name_entry = color_name_entry
        color_name_hbox.set_spacing(6)
        color_name_hbox.pack_start(color_name_label, False, False, 0)
        color_name_hbox.pack_start(color_name_entry, True, True, 0)

        palette_vbox = Gtk.VBox()
        palette_vbox.set_spacing(12)
        palette_vbox.pack_start(palette_details_hbox, False, False, 0)
        palette_vbox.pack_start(view, True, True, 0)
        palette_vbox.pack_start(color_name_hbox, False, False, 0)

        # Dialog contents
        # Main edit area to the left, buttons to the right
        hbox = Gtk.HBox()
        hbox.set_spacing(12)
        hbox.pack_start(palette_vbox, True, True, 0)
        hbox.pack_start(action_bbox, False, False, 0)
        hbox.set_border_width(12)
        self.vbox.pack_start(hbox, True, True, 0)

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
        if not name:
            name = self._ensure_valid_palette_name()
        self._palette_name_entry.set_text(name)
        self._columns_adj.set_value(palette.get_columns())
        self._mgr.palette.update(palette)

    def _palette_name_changed_cb(self, editable):
        name = editable.get_chars(0, -1)
        if not name:
            name = ""  # note: not None (it'll be stringified)
        pal = self._mgr.palette
        pal.name = unicode(name)

    def _columns_changed_cb(self, adj):
        ncolumns = int(adj.get_value())
        pal = self._mgr.palette
        pal.set_columns(ncolumns)

    def _color_name_changed_cb(self, editable):
        name = editable.get_chars(0, -1)
        palette = self._mgr.palette
        i = palette.match_position
        if i is None:
            return
        old_name = palette.get_color_name(i)
        if not name:
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
                col_name_entry.set_text(C_(
                    "palette editor dialog: color name entry",
                    "Empty palette slot",
                ))
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
            new_name = self._ensure_valid_palette_name()
        old_name = self._palette_name_entry.get_chars(0, -1)
        if old_name != new_name:
            self._palette_name_entry.set_text(new_name)
        self._columns_adj.set_value(palette.get_columns())
        self._update_buttons()

    def _add_btn_clicked(self, button):
        palette = self._mgr.palette
        i = palette.match_position
        if i is None:
            i = len(palette)
            palette.append(None)
            palette.match_position = i
        else:
            palette.insert(i, None)

    def _remove_btn_clicked(self, button):
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
        palette = palette_load_via_dialog(
            title=C_("palette load dialog: title", "Load palette"),
            parent=self,
            preview=preview,
            shortcuts=[palettes_dir],
        )
        if palette is not None:
            self._mgr.palette.update(palette)
            self._ensure_valid_palette_name()

    def _save_btn_clicked(self, button):
        preview = _PalettePreview()
        palette_save_via_dialog(
            self._mgr.palette,
            title=C_("palette save dialog: title", "Save palette"),
            parent=self,
            preview=preview,
        )

    def _clear_btn_clicked(self, button):
        pal = self._mgr.palette
        pal.clear()
        self._ensure_valid_palette_name()

    def _ensure_valid_palette_name(self):
        pal = self._mgr.palette
        if not pal.name:
            pal.name = self._UNTITLED_PALETTE_NAME
        return pal.name


class PaletteView (ColorAdjuster, Gtk.ScrolledWindow):
    """Scrollable view of a palette.

    Palette entries can be clicked to select the color, and all instances of
    the current shared color in the palette are highlighted.

    """

    ## Sizing constraint constants
    _MIN_HEIGHT = 32
    _MIN_WIDTH = 150
    _MAX_NATURAL_HEIGHT = 300
    _MAX_NATURAL_WIDTH = 300

    def __init__(self):
        Gtk.ScrolledWindow.__init__(self)
        self.grid = _PaletteGridLayout()
        self.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.add(self.grid)

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
            if (s * ncolumns) > w:
                ncolumns = 0
        if ncolumns == 0:
            s = math.sqrt((w * h) / ncolors)
            s = clamp(s, s_min, s_max)
            s = int(s)
            ncolumns = max(1, int(w // s))
        nrows = int(ncolors // ncolumns)
        if ncolors % ncolumns != 0:
            nrows += 1
        nrows = max(1, nrows)
        dx, dy = 0, 0
        if (nrows * s) < h:
            dy = int(h - (nrows * s)) // 2
        if (ncolumns * s) < w:
            dx = int(w - (ncolumns * s)) // 2
        bg_color = _widget_get_bg_color(self)
        _palette_render(self._palette, cr, rows=nrows, columns=ncolumns,
                        swatch_size=s, bg_color=bg_color,
                        offset_x=dx, offset_y=dy,
                        rtl=False)

    def set_palette(self, palette):
        self._palette = palette
        self.queue_draw()


def _widget_get_bg_color(widget):
    """Valid background color from the first ancestor widget having one

    Workaround for some widget arrangements in Adwaita for 3.14 having
    null background colors. Fallback is a medium grey, which should be
    acceptable with most styles.

    """
    while widget is not None:
        state = widget.get_state_flags()
        style_context = widget.get_style_context()
        bg_rgba = style_context.get_background_color(state)
        if bg_rgba.alpha != 0:
            return gui.uicolor.from_gdk_rgba(bg_rgba)
        widget = widget.get_parent()
    return RGBColor(0.5, 0.5, 0.5)


class _PaletteGridLayout (ColorAdjusterWidget):
    """The palette layout embedded in a scrolling PaletteView.
    """

    ## Class settings
    IS_DRAG_SOURCE = True
    HAS_DETAILS_DIALOG = True
    STATIC_TOOLTIP_TEXT = C_(
        "palette view",
        "Color swatch palette.\nDrop colors here,\ndrag them to organize.",
    )
    ALLOW_HCY_TWEAKING = False   # Interacts badly with menus

    ## Layout constants
    _SWATCH_SIZE_MIN = 8
    _SWATCH_SIZE_MAX = 50
    _SWATCH_SIZE_NOMINAL = 20
    _PREFERRED_COLUMNS = 5  #: Preferred width in cells for free-flow mode.

    def __init__(self):
        ColorAdjusterWidget.__init__(self)

        evbox = Gtk.EventBox()
        self.add(evbox)

        # Sizing
        s = self._SWATCH_SIZE_NOMINAL
        self.set_size_request(s, s)
        self.connect("size-allocate", self._size_alloc_cb)
        #: Highlight the currently matched color
        self.show_matched_color = False
        #: User can click on empty slots
        self.can_select_empty = False
        # Current index
        evbox.connect("button-press-event", self._button_press_cb)
        evbox.connect_after("button-release-event", self._button_release_cb)
        # Dragging
        evbox.connect("motion-notify-event", self._motion_notify_cb)
        evbox.add_events(Gdk.EventMask.POINTER_MOTION_MASK)
        # Tooltips
        self._tooltip_index = None
        self.set_has_tooltip(True)
        # Target markers while dragging or invoking the context menu
        self._insert_target_index = None
        # Cached layout details
        self._rows = None
        self._columns = None
        self._last_palette_columns = None
        self._swatch_size = self._SWATCH_SIZE_NOMINAL

    def _size_alloc_cb(self, widget, alloc):
        """Caches layout details after size negotiation.
        """
        width = alloc.width
        height = alloc.height
        ncolors, nrows, ncolumns = self._get_palette_dimensions()
        if nrows and ncolumns:
            # Fitted to the major dimension
            size = int(min(width / ncolumns, height / nrows))
            size = self._constrain_swatch_size(size)
        else:
            # Free-flowing
            if ncolors > 0:
                size = int(math.sqrt((width * height) / ncolors))
                size = self._constrain_swatch_size(size)
                ncolumns = max(1, min(ncolors, width // size))
                nrows = max(1, int(ncolors // ncolumns))
                if int(ncolors % ncolumns) > 0:
                    nrows += 1
                if nrows * size > height or ncolumns * size > width:
                    size = max(1, min(int(height // nrows),
                                      int(width // ncolumns)))
                    size = self._constrain_swatch_size(size)
                    ncolumns = max(1, min(ncolors, width // size))
                    nrows = max(1, int(ncolors // ncolumns))
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
        else:
            if palette.columns != self._last_palette_columns:
                layout_changed = True
                logger.debug("layout changed: different number of columns")
            else:
                ncells = self._rows * self._columns
                ncolors = len(palette)
                if ncolors > ncells or ncolors <= ncells - self._columns:
                    logger.debug("layout changed: cannot fit palette into "
                                 "currently calculated space")
                    layout_changed = True
        # Queue a resize (and an implicit redraw) if the layout has changed,
        # or just a redraw.
        if layout_changed:
            logger.debug("queuing full resize")
            self._rows = None
            self._columns = None
            self.queue_resize()
            self._insert_target_index = None
            self._tooltip_index = None
        else:
            logger.debug("layout unchanged, queuing redraw")
            self.queue_draw()
        self._last_palette_columns = palette.columns

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
            # Not over a color, so use the static default
            if self._tooltip_index not in (-1, -2):
                # First such event: reset the tooltip.
                self._tooltip_index = -1
                self.set_has_tooltip(False)
                self.set_tooltip_text("")
            elif self._tooltip_index != -2:
                # Second event over a non-color: set the tooltip text.
                self._tooltip_index = -2
                self.set_has_tooltip(True)
                self.set_tooltip_text(self.STATIC_TOOLTIP_TEXT)
        elif self._tooltip_index != i:
            # Mouse pointer has moved to a different color, or away
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
                    tip = C_(
                        "palette view",
                        "Empty palette slot (drag a color here)",
                    )
                elif tip is None or tip.strip() == "":
                    tip = ""  # Anonymous colors don't get tooltips
                self.set_has_tooltip(True)
                self.set_tooltip_text(tip)

    def _button_press_cb(self, widget, event):
        """Handle button presses."""
        # The base class has a separate handler which
        # changes the managed colour, so don't need to do that here.
        if event.type != Gdk.EventType.BUTTON_PRESS:
            return False
        # Move the highlight
        x, y = event.x, event.y
        i = self.get_index_at_pos(x, y, nearest=False)
        mgr = self.get_color_manager()
        is_empty = mgr.palette.get_color(i) is None
        if event.button == 1:
            if not (is_empty and not self.can_select_empty):
                mgr.palette.set_match_position(i)
                mgr.palette.set_match_is_approx(False)
            return False
        # Button 3 shows a menu
        if event.button != 3:
            return False
        self._popup_context_menu(event)

    def _popup_context_menu(self, event):
        x, y = event.x, event.y
        i = self.get_index_at_pos(x, y, nearest=True, insert=True)
        mx, my = self.get_position_for_index(i)
        mx = event.x_root - x + mx + self._swatch_size
        my = event.y_root - y + my + self._swatch_size
        menu = self._get_context_menu(i)
        menu.show_all()
        menu.popup(
            parent_menu_shell = None,
            parent_menu_item = None,
            func = lambda *a: (mx, my, True),
            data = None,
            button = event.button,
            activate_time = event.time,
        )
        self._insert_target_index = i
        self.queue_draw()
        return False

    def _get_empty_range(self, index):
        """Returns the populated start and end of a range of empty slots

        Returns the indices of two populated swatches around the target
        swatch, or None if there's no run of one or more empty slots
        between them.

        """
        start_index = None
        end_index = None
        palette = self.get_color_manager().palette
        if palette[index] is not None:
            return None
        i = index
        while i >= 0:
            i -= 1
            if palette[i] is not None:
                start_index = i
                break
        i = index
        while i < len(palette):
            i += 1
            if palette[i] is not None:
                end_index = i
                break
        if None not in (start_index, end_index):
            assert start_index < end_index
            if start_index < end_index - 1:
                return (start_index, end_index)
        return None

    def _get_context_menu(self, i):
        menu = Gtk.Menu()
        menu.connect_after("deactivate", self._context_menu_deactivate_cb)
        palette = self.get_color_manager().palette
        empty_range = self._get_empty_range(i)
        item_defs = [
            (
                # TRANSLATORS: inserting gaps (empty color swatches)
                C_("palette view: context menu", "Add Empty Slot"),
                self._insert_empty_slot_cb,
                True,
                [i],
            ),
            (
                # TRANSLATORS: inserting gaps (empty color swatches)
                C_("palette view: context menu", "Insert Row"),
                self._insert_empty_row_cb,
                True,
                [i],
            ),
            (
                # TRANSLATORS: inserting gaps (empty color swatches)
                C_("palette view: context menu", "Insert Column"),
                self._insert_empty_column_cb,
                bool(palette.get_columns()),
                [i],
            ),
            None,
            (
                # TRANSLATORS: Color interpolations
                C_("palette view: context menu", "Fill Gap (RGB)"),
                self._interpolate_empty_range_cb,
                bool(empty_range),
                [RGBColor, empty_range],
            ),
            (
                # TRANSLATORS: Color interpolations
                C_("palette view: context menu", "Fill Gap (HCY)"),
                self._interpolate_empty_range_cb,
                bool(empty_range),
                [HCYColor, empty_range],
            ),
            (
                # TRANSLATORS: Color interpolations
                C_("palette view: context menu", "Fill Gap (HSV)"),
                self._interpolate_empty_range_cb,
                bool(empty_range),
                [HSVColor, empty_range],
            ),
        ]
        for item_def in item_defs:
            if not item_def:
                item = Gtk.SeparatorMenuItem()
            else:
                label_str, activate_cb, sensitive, args = item_def
                item = Gtk.MenuItem()
                item.set_label(label_str)
                if activate_cb:
                    item.connect("activate", activate_cb, *args)
                item.set_sensitive(sensitive)
            menu.append(item)
        menu.attach_to_widget(self)
        return menu

    def _button_release_cb(self, widget, event):
        pass

    ## Context menu handlers

    def _context_menu_deactivate_cb(self, menu):
        self._insert_target_index = None
        self.queue_draw()
        GLib.idle_add(menu.destroy)

    def _insert_empty_slot_cb(self, menuitem, target_i):
        palette = self.get_color_manager().palette
        palette.insert(target_i, None)

    def _insert_empty_row_cb(self, menuitem, target_i):
        row_start_i = (target_i // self._columns) * self._columns
        palette = self.get_color_manager().palette
        for i in range(palette.get_columns()):
            palette.insert(row_start_i, None)

    def _insert_empty_column_cb(self, menuitem, target_i):
        palette = self.get_color_manager().palette
        assert palette.get_columns(), \
            "Can't insert columns into a free-flowing palette"
        row_di = target_i % self._columns
        columns_new = palette.get_columns() + 1
        r = 0
        while r * columns_new < len(palette):
            i = r * columns_new + row_di
            if i >= len(palette):
                break
            palette.insert(i, None)
            r += 1
        palette.set_columns(columns_new)

    def _interpolate_empty_range_cb(self, menuitem, color_class, range):
        i0, ix = range
        palette = self.get_color_manager().palette
        c0 = color_class(color=palette[i0])
        cx = color_class(color=palette[ix])
        nsteps = ix - i0 + 1
        if nsteps < 3:
            return
        interpolated = list(c0.interpolate(cx, nsteps))
        assert len(interpolated) == nsteps
        interpolated.pop(0)
        interpolated.pop(-1)
        for i, c in enumerate(interpolated):
            palette[i0 + 1 + i] = c

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
                nrows = max(1, int(ncolors // ncolumns))
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
            swatch_size = self._constrain_swatch_size(int(width // ncolumns))
            min_h = self._SWATCH_SIZE_MIN * nrows
            nat_h = swatch_size * nrows
        else:
            # Free-flowing, across and then down
            # Since s = sqrt((w*h)/n),
            min_h = int(((self._SWATCH_SIZE_MIN ** 2) * ncolors) // width)
            nat_h = int(((self._SWATCH_SIZE_NOMINAL ** 2) * ncolors) // width)
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
            swatch_size = self._constrain_swatch_size(int(height // nrows))
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
        bg_col = _widget_get_bg_color(self)
        dx, dy = self.get_painting_offset()
        _palette_render(mgr.palette, cr,
                        rows=self._rows, columns=self._columns,
                        swatch_size=self._swatch_size,
                        bg_color=bg_col,
                        offset_x=dx, offset_y=dy,
                        rtl=False)

    def _paint_marker(self, cr, x, y, insert=False,
                      bg_rgb=(0, 0, 0), fg_rgb=(1, 1, 1),
                      bg_dash=[1, 2], fg_dash=[1, 2],
                      bg_width=2, fg_width=1):
        cr.save()
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        size = self._swatch_size
        w = h = size
        # Background (shadow)
        cr.set_source_rgb(*bg_rgb)
        cr.set_line_width(bg_width)
        cr.rectangle(x, y, w - 1, h - 1)

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

        # Target marker
        if self._insert_target_index is not None:
            i = self._insert_target_index
            x, y = self.get_position_for_index(i)
            insert = mgr.palette.get_color(i) is not None
            self._paint_marker(cr, x, y, insert=insert)

        # Position of the previous click
        elif self.show_matched_color:
            i = mgr.palette.match_position
            if i is not None:
                x, y = self.get_position_for_index(i)
                marker_args = [cr, x, y]
                marker_kw = dict(bg_width=3, fg_width=1,
                                 bg_dash=[2, 3], fg_dash=[2, 3])
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
        r = int(i // self._columns)
        x = 0.5 + (c * s_w)
        y = 0.5 + (r * s_h)
        return (x + dx, y + dy)

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
            dx = (wd - l_wd) / 2.0
        if l_ht < ht:
            dy = (ht - l_ht) / 2.0
        return 1 + int(dx), 1 + int(dy)

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

    def get_index_at_pos(self, x, y, nearest=False, insert=False):
        """Convert a position to a palette index

        :param int x: X coord, in widget pixels
        :param int y: Y coord, in widget pixels
        :param bool nearest: Pick nearest index if (x, y) lies outside
        :param bool insert: Get an insertion index (requires nearest)
        :rtype: int
        :returns: An index, or None

        The returned index value may be None. Insertion indices are not
        guaranteed to identify existing entries.

        """
        mgr = self.get_color_manager()
        if mgr.palette is None:
            return None
        if None in (self._rows, self._columns):
            return None
        dx, dy = self.get_painting_offset()
        s_wd = s_ht = self._swatch_size
        # Calculate a raw row and column
        r = int((y - dy) // s_ht)
        c = int((x - dx) // s_wd)
        # Check position is within range, or constrain for nearest
        if r < 0:
            if not nearest:
                return None
            r = 0
        elif r >= self._rows:
            if not nearest:
                return None
            r = self._rows - 1
        if c < 0:
            if not nearest:
                return None
            c = 0
        elif c >= self._columns:
            if not nearest:
                return None
            c = self._columns - 1
        # Index range check too: the last row may not be fully populated
        i = (r * self._columns) + c
        if i >= len(mgr.palette):
            if not nearest:
                return None
            i = len(mgr.palette)
            if not insert:
                i -= 1
        return i

    ## Drag handling overrides

    def drag_motion_cb(self, widget, context, x, y, t):
        if "application/x-color" not in map(str, context.list_targets()):
            return False

        # Default action: copy means insert or overwrite
        action = Gdk.DragAction.COPY

        # Update the insertion marker
        i = self.get_index_at_pos(x, y)
        if i != self._insert_target_index:
            self.queue_draw()
        self._insert_target_index = i

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
                    # TODO: record this as a target range for redraws,
                    # and reset _insert_target_index

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
        color = gui.uicolor.from_drag_data(data)
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
        self._insert_target_index = None
        self.queue_draw()
        context.finish(True, True, t)
        self.set_managed_color(color)
        mgr.palette.set_match_position(target_index)

    def drag_end_cb(self, widget, context):
        self._insert_target_index = None
        self.queue_draw()

    def drag_leave_cb(self, widget, context, time):
        self._insert_target_index = None
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
        transient_for=parent,
        action=Gtk.FileChooserAction.OPEN,
    )
    dialog.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.REJECT)
    dialog.add_button(Gtk.STOCK_OPEN, Gtk.ResponseType.ACCEPT)
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
    filter.set_name(C_(
        "palette load dialog: filters",
        "GIMP palette file (*.gpl)",
    ))
    dialog.add_filter(filter)
    filter = Gtk.FileFilter()
    filter.add_pattern("*")
    filter.set_name(C_(
        "palette load dialog: filters",
        "All files (*)",
    ))
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
        transient_for=parent,
        action=Gtk.FileChooserAction.SAVE,
    )
    dialog.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.REJECT)
    dialog.add_button(Gtk.STOCK_SAVE, Gtk.ResponseType.ACCEPT)
    if preview is not None:
        dialog.set_preview_widget(preview)
        dialog.connect("update-preview",
                       _palette_loadsave_dialog_update_preview_cb,
                       preview)
    dialog.set_do_overwrite_confirmation(True)
    filter = Gtk.FileFilter()
    filter.add_pattern("*.gpl")
    filter.set_name(C_(
        "palette save dialog: filters",
        "GIMP palette file (*.gpl)",
    ))
    dialog.add_filter(filter)
    filter = Gtk.FileFilter()
    filter.add_pattern("*")
    filter.set_name(C_(
        "palette save dialog: filters",
        "All files (*)",
    ))
    dialog.add_filter(filter)
    response_id = dialog.run()
    result = False
    if response_id == Gtk.ResponseType.ACCEPT:
        filename = dialog.get_filename()
        filename = re.sub(r'[.]?(?:[Gg][Pp][Ll])?$', "", filename)
        filename += ".gpl"
        logger.info("Saving palette to %r", filename)
        # FIXME: this can overwrite files without prompting the user, if
        # the name hacking above changed the filename.  Should do the name
        # tweak within the dialog somehow and get that to confirm.
        with open(filename, 'w', encoding="utf-8", errors="replace") as fp:
            palette.save(fp)
            fp.flush()
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
        except Exception as ex:
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
    :param int rows: number of rows in the layout
    :param int columns: number of columns in the layout
    :param int swatch_size: size of each swatch, in pixels
    :param lib.color.UIColor bg_color: color used when rendering the patterned
                      placeholder for an empty palette slot.
    :param bool rtl: layout direction: set to True to render right to left,
                 instead of left to right. Currently ignored.
    """

    highlight_dluma = 0.05

    if len(palette) == 0:
        return
    if rows is None or columns is None:
        return

    cr.save()
    cr.translate(offset_x, offset_y)

    # Sizes and colors
    swatch_w = swatch_h = swatch_size
    light_col = HCYColor(color=bg_color)
    light_col.y += highlight_dluma
    dark_col = HCYColor(color=bg_color)
    dark_col.y -= highlight_dluma
    if light_col.y >= 1:
        light_col.y = 1.0
        dark_col.y = 1.0 - (2 * highlight_dluma)
    if dark_col.y <= 0:
        dark_col.y = 0.0
        light_col.y = 0.0 + (2 * highlight_dluma)

    # Upper left outline (bottom right is covered below by the
    # individual chips' shadows)
    ul_col = HCYColor(color=bg_color)
    ul_col.y *= 0.75
    ul_col.c *= 0.5
    cr.set_line_join(cairo.LINE_JOIN_ROUND)
    cr.set_line_cap(cairo.LINE_CAP_ROUND)
    cr.set_source_rgb(*ul_col.get_rgb())
    cr.move_to(0.5, (rows * swatch_h) - 1)
    cr.line_to(0.5, 0.5)
    row1cells = min(columns, len(palette))  # needed?
    cr.line_to((row1cells * swatch_w) - 1, 0.5)
    cr.set_line_width(2)
    cr.stroke()

    # Draw into the predefined grid
    r = c = 0
    cr.set_line_width(1.0)
    cr.set_line_cap(cairo.LINE_CAP_SQUARE)
    for col in palette.iter_colors():
        s_x = c * swatch_w
        s_y = r * swatch_h
        s_w = swatch_w
        s_h = swatch_h

        # Select fill bg and pattern fg colors, Tango-style edge highlight
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
            # Color swatch
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

        # Draw the swatch / color chip
        cr.set_source_rgb(*sh_rgb)
        cr.rectangle(s_x, s_y, s_w, s_h)
        cr.fill()
        cr.set_source_rgb(*fill_bg_rgb)
        cr.rectangle(s_x, s_y, s_w - 1, s_h - 1)
        cr.fill()
        if fill_fg_rgb is not None:
            s_w2 = int((s_w - 1) // 2)
            s_h2 = int((s_h - 1) // 2)
            cr.set_source_rgb(*fill_fg_rgb)
            cr.rectangle(s_x, s_y, s_w2, s_h2)
            cr.fill()
            cr.rectangle(s_x + s_w2, s_y + s_h2, s_w2, s_h2)
            cr.fill()
        cr.set_source_rgb(*hi_rgb)
        cr.rectangle(s_x + 0.5, s_y + 0.5, s_w - 2, s_h - 2)
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
    mgr = ColorManager(prefs={}, datapath=".")
    spv = PaletteView()
    spv.grid.show_matched_color = True
    spv.grid.can_select_empty = True
    spv.set_color_manager(mgr)
    spv.set_size_request(150, 150)
    if len(sys.argv[1:]) > 0:
        palette_file = sys.argv[1]  # GIMP palette file (*.gpl)
        palette = Palette(filename=palette_file)
        mgr.palette.update(palette)
    win.add(spv)
    win.show_all()
    Gtk.main()
