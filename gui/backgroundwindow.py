# This file is part of MyPaint.
# Copyright (C) 2009-2018 by the MyPaint Development Team.
# Copyright (C) 2008-2014 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Background tile chooser dialog"""

## Imports

from __future__ import division, print_function
import os
import sys
import logging

from gettext import gettext as _
from lib.gibindings import Gtk
from lib.gibindings import GdkPixbuf

from . import pixbuflist
from . import windowing
from lib import tiledsurface
from lib import helpers
import lib.pixbuf
from lib.pycompat import unicode
from lib.pycompat import xrange

logger = logging.getLogger(__name__)

## Settings and consts

N = tiledsurface.N
DEFAULT_BACKGROUND = 'default.png'
FALLBACK_BACKGROUND = 'mrmamurk/mamurk_e_1.png'
BACKGROUNDS_SUBDIR = 'backgrounds'
RESPONSE_SAVE_AS_DEFAULT = 1
BLOAT_MAX_SIZE = 1024


## Class defs

class BackgroundWindow (windowing.Dialog):

    def __init__(self):
        from gui import application
        app = application.get_app()
        assert app is not None

        windowing.Dialog.__init__(
            self,
            app=app,
            title=_('Background'),
            modal=True
        )
        self.add_button(_('Save as Default'), RESPONSE_SAVE_AS_DEFAULT)
        self.add_button(Gtk.STOCK_OK, Gtk.ResponseType.ACCEPT)

        self._current_background_pixbuf = None  # set when changed

        # Set up window.
        self.connect('response', self._response_cb)

        notebook = self.nb = Gtk.Notebook()
        self.vbox.pack_start(notebook, True, True, 0)

        # Set up patterns tab.
        patterns_scroll = Gtk.ScrolledWindow()
        patterns_scroll.set_policy(
            Gtk.PolicyType.NEVER,
            Gtk.PolicyType.AUTOMATIC,
        )
        notebook.append_page(patterns_scroll, Gtk.Label(label=_('Pattern')))

        self.bgl = BackgroundList(self)
        patterns_scroll.add(self.bgl)

        self.connect("realize", self._realize_cb)
        self.connect("show", self._show_cb)
        self.connect("hide", self._hide_cb)

        # Set up colors tab.
        color_vbox = Gtk.VBox()
        notebook.append_page(color_vbox, Gtk.Label(label=_('Color')))

        self.cs = Gtk.ColorSelection()
        self.cs.connect('color-changed', self._color_changed_cb)
        color_vbox.pack_start(self.cs, True, True, 0)

        b = Gtk.Button(label=_('Add color to Patterns'))
        b.connect('clicked', self._add_color_to_patterns_cb)
        color_vbox.pack_start(b, False, True, 0)

    def _realize_cb(self, dialog):
        if not self.bgl.initialized:
            self.bgl.initialize()

    def _show_cb(self, dialog):
        self._current_background_pixbuf = None
        self.set_response_sensitive(RESPONSE_SAVE_AS_DEFAULT, False)

    def _hide_cb(self, dialog):
        self._current_background_pixbuf = None

    def _response_cb(self, dialog, response, *args):
        if response == RESPONSE_SAVE_AS_DEFAULT:
            self._save_as_default_cb()
        elif response == Gtk.ResponseType.ACCEPT:
            self.hide()

    def _color_changed_cb(self, widget):
        pixbuf = self._get_selected_color_pixbuf()
        self.set_background(pixbuf)

    def _get_selected_color_pixbuf(self):
        rgb = self.cs.get_current_color()
        rgb = (rgb.red, rgb.green, rgb.blue)
        rgb = (c / 0xffff for c in rgb)
        pixbuf = new_blank_pixbuf(rgb, N, N)
        return pixbuf

    def _save_as_default_cb(self):
        pixbuf = self._current_background_pixbuf
        assert pixbuf is not None, "BG pixbuf was not changed."
        path = os.path.join(
            self.app.user_datapath,
            BACKGROUNDS_SUBDIR,
            DEFAULT_BACKGROUND,
        )
        lib.pixbuf.save(pixbuf, path, 'png')
        self.hide()

    def set_background(self, pixbuf):
        doc = self.app.doc.model
        doc.layer_stack.set_background(pixbuf, make_default=True)
        self._current_background_pixbuf = pixbuf
        self.set_response_sensitive(RESPONSE_SAVE_AS_DEFAULT, True)

    def _add_color_to_patterns_cb(self, widget):
        pixbuf = self._get_selected_color_pixbuf()
        i = 1
        while True:
            filename = os.path.join(self.app.user_datapath,
                                    BACKGROUNDS_SUBDIR,
                                    'color%02d.png' % i)
            if not os.path.exists(filename):
                break
            i += 1
        lib.pixbuf.save(pixbuf, filename, 'png')
        self.bgl.backgrounds.append(pixbuf)
        self.bgl.update()
        self.bgl.set_selected(pixbuf)
        self.nb.set_current_page(0)


class BackgroundList (pixbuflist.PixbufList):

    _SUFFIXES = ('.jpg', '.jpeg', '.png')

    def __init__(self, win):
        pixbuflist.PixbufList.__init__(
            self,
            None,
            N, N,
            namefunc=self._get_tooltip,
            pixbuffunc=self._get_preview_pixbuf,
        )
        self.app = win.app
        self.win = win

        stock_path = os.path.join(self.app.datapath, BACKGROUNDS_SUBDIR)
        user_path = os.path.join(self.app.user_datapath, BACKGROUNDS_SUBDIR)
        if not os.path.isdir(user_path):
            os.mkdir(user_path)

        self._background_files = self._list_dir(stock_path)
        self._background_files.sort()
        self._background_files += self._list_dir(user_path)

        # Exclude DEFAULT_BACKGROUND from the list shown to the user
        for filename in reversed(self._background_files):
            file_basename = os.path.basename(filename)
            if file_basename.lower() == DEFAULT_BACKGROUND:
                self._background_files.remove(filename)

        self._pixbuf_tooltip = {}
        self._pixbufs_scaled = {}  # lazily loaded by self.initialize()
        self.backgrounds = []

        self.item_selected += self._item_selected_cb

    @classmethod
    def _list_dir(cls, path):
        """Recursively find images by suffix"""
        contents = []
        for dir_path, dir_subdirs, dir_files in os.walk(path):
            for file_name in dir_files:
                is_matched = False
                file_name_lowercase = file_name.lower()
                for suffix in cls._SUFFIXES:
                    if not file_name_lowercase.endswith(suffix):
                        continue
                    is_matched = True
                    break
                if is_matched:
                    file_path = os.path.join(dir_path, file_name)
                    contents.append(file_path)
        contents.sort(key=os.path.getmtime)
        return contents

    @property
    def initialized(self):
        return len(self.backgrounds) != 0

    def initialize(self):
        self.backgrounds = self._load_pixbufs(self._background_files)
        self.set_itemlist(self.backgrounds)

    def _load_pixbufs(self, files, exclude_default=False):
        pixbufs = []
        load_errors = []
        for filename in files:
            is_matched = False
            for suffix in self._SUFFIXES:
                if not filename.lower().endswith(suffix):
                    continue
                is_matched = True
                break
            if not is_matched:
                logger.warning(
                    "Excluding %r: not in %r",
                    filename,
                    self._SUFFIXES,
                )
                continue
            pixbuf, errors = load_background(filename)
            if errors:
                for err in errors:
                    logger.error("Error loading %r: %r", filename, err)
                    load_errors.append(err)
                continue
            if os.path.basename(filename).lower() == DEFAULT_BACKGROUND:
                if exclude_default:
                    logger.warning("Excluding %r: is default background (%r)",
                                   filename, DEFAULT_BACKGROUND)
                    continue
            pixbufs.append(pixbuf)
            tooltip = _filename_to_display(filename)
            self._pixbuf_tooltip[pixbuf] = tooltip

        if load_errors:
            msg = "\n\n".join(load_errors)
            self.app.message_dialog(
                text=_("One or more backgrounds could not be loaded"),
                title=_("Error loading backgrounds"),
                secondary_text=_("Please remove the unloadable files, or "
                                 "check your libgdkpixbuf installation."),
                long_text=msg,
                message_type=Gtk.MessageType.WARNING,
                modal=True,
            )

        logger.info("Loaded %d of %d background(s), with %d error(s)",
                    len(pixbufs), len(files), len(errors))
        return pixbufs

    def _get_preview_pixbuf(self, pixbuf):
        if pixbuf in self._pixbufs_scaled:
            return self._pixbufs_scaled[pixbuf]
        w, h = pixbuf.get_width(), pixbuf.get_height()
        if w == N and h == N:
            return pixbuf
        assert w >= N
        assert h >= N
        scale = max(0.25, N / min(w, h))
        scaled = new_blank_pixbuf((0, 0, 0), N, N)
        pixbuf.composite(
            dest=scaled,
            dest_x=0, dest_y=0,
            dest_width=N, dest_height=N,
            offset_x=0, offset_y=0,
            scale_x=scale, scale_y=scale,
            interp_type=GdkPixbuf.InterpType.BILINEAR,
            overall_alpha=255,
        )
        self.app.pixmaps.plus.composite(
            dest=scaled,
            dest_x=0, dest_y=0,
            dest_width=N, dest_height=N,
            offset_x=0, offset_y=0,
            scale_x=1.0, scale_y=1.0,
            interp_type=GdkPixbuf.InterpType.BILINEAR,
            overall_alpha=255,
        )
        self._pixbufs_scaled[pixbuf] = scaled
        return scaled

    def _get_tooltip(self, pixbuf):
        return self._pixbuf_tooltip.get(pixbuf, None)

    def _item_selected_cb(self, self_, pixbuf):
        self.win.set_background(pixbuf)


## Helpers


def _filename_to_display(s):
    """Convert a str filename to Unicode without obsessing too much."""
    # That said, try to be be correct about Windows/POSIX weirdness.
    if not isinstance(s, unicode):
        if sys.platform == "win32":
            enc = "UTF-8"  # always, and sys.getfilesystemencoding() breaks
        else:
            enc = sys.getfilesystemencoding()
        s = s.decode(enc, "replace")
    return s


def new_blank_pixbuf(rgb, w, h):
    """Create a blank pixbuf with all pixels set to a color

    :param tuple rgb: Color to blank the pixbuf to (``R,G,B``, floats)
    :param int w: Width for the new pixbuf
    :param int h: Width for the new pixbuf

    The returned pixbuf has no alpha channel.

    """
    pixbuf = GdkPixbuf.Pixbuf.new(
        GdkPixbuf.Colorspace.RGB, False, 8,
        w, h,
    )
    r, g, b = (helpers.clamp(int(round(0xff * x)), 0, 0xff) for x in rgb)
    rgba_pixel = (r << 24) + (g << 16) + (b << 8) + 0xff
    pixbuf.fill(rgba_pixel)
    return pixbuf


def load_background(filename, bloatmax=BLOAT_MAX_SIZE):
    """Load a pixbuf, testing it for suitability as a background

    :param str filename: Full path to the filename to load.
    :param int bloatmax: Repeat up to this size
    :rtype: tuple

    The returned tuple is a pair ``(PIXBUF, ERRORS)``,
    where ``ERRORS`` is a list of localized strings
    describing the errors encountered,
    and ``PIXBUF`` contains the loaded background pixbuf.
    If there were errors, ``PIXBUF`` is None.

    The MyPaint rendering engine can only manage
    background layers which fit into its tile structure.
    Formerly, only background images with dimensions
    which were exact multiples of the tile size were permitted.
    We have a couple of workarounds now:

    * "Bloating" the background by repetition (pixel-perfect)
    * Scaling the image down to fit (distorts the image)

    """
    filename_display = _filename_to_display(filename)
    load_errors = []
    try:
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
    except Exception as ex:
        logger.error("Failed to load background %r: %s", filename, ex)
        msg = unicode(_(
            'Gdk-Pixbuf couldn\'t load "{filename}", and reported "{error}"'
        ))
        load_errors.append(msg.format(
            filename=filename_display,
            error=repr(ex),
        ))
        return (None, load_errors)
    # Validity check
    w, h = pixbuf.get_width(), pixbuf.get_height()
    if w == 0 or h == 0:
        msg = unicode(_("{filename} has zero size (w={w}, h={h})"))
        load_errors.append(msg.format(
            filename=filename_display,
            w=w, h=h,
        ))
        return (None, load_errors)
    # Flatten
    if pixbuf.get_has_alpha():
        logger.warning(
            "%r has an alpha channel, which should be removed manually",
            filename,
        )
        new_pixbuf = new_blank_pixbuf((0, 0, 0), w, h)
        pixbuf.composite(
            dest=new_pixbuf,
            dest_x=0, dest_y=0,
            dest_width=w, dest_height=h,
            offset_x=0, offset_y=0,
            scale_x=1.0, scale_y=1.0,
            interp_type=GdkPixbuf.InterpType.NEAREST,
            overall_alpha=255,
        )
        pixbuf = new_pixbuf
        logger.debug(
            "Flattened %s by compositing it onto a black backdrop",
            filename,
        )
    # Attempt to fit the image into our grid.
    exact_fit = ((w % N, h % N) == (0, 0))
    if not exact_fit:
        logger.warning(
            "%r (%dx%d) does not fit the %dx%d tile grid exactly",
            filename,
            w, h,
            N, N,
        )
        repeats_x = _best_nrepeats_for_scaling(w, bloatmax)
        repeats_y = _best_nrepeats_for_scaling(h, bloatmax)
        if repeats_x > 1 or repeats_y > 1:
            logger.info(
                "Tiling %r to %dx%d (was: %dx%d, repeats: %d vert, %d horiz)",
                filename,
                w * repeats_x, h * repeats_y,
                w, h,
                repeats_x, repeats_y,
            )
            pixbuf = _tile_pixbuf(pixbuf, repeats_x, repeats_y)
        w, h = pixbuf.get_width(), pixbuf.get_height()
        if (w % N != 0) or (h % N != 0):
            orig_w, orig_h = w, h
            w = max(1, w // N) * N
            h = max(1, h // N) * N
            logger.info(
                "Scaling %r to %dx%d (was: %dx%d)",
                filename,
                w, h,
                orig_w, orig_h,
            )
            pixbuf = pixbuf.scale_simple(
                dest_width=w, dest_height=h,
                interp_type=GdkPixbuf.InterpType.BILINEAR,
            )
        assert (w % N == 0) and (h % N == 0)
    if load_errors:
        pixbuf = None
    return pixbuf, load_errors


def _tile_pixbuf(pixbuf, repeats_x, repeats_y):
    """Make a repeated tiled image of a pixbuf"""
    w, h = pixbuf.get_width(), pixbuf.get_height()
    result = new_blank_pixbuf((0, 0, 0), repeats_x * w, repeats_y * h)
    for xi in xrange(repeats_x):
        for yi in xrange(repeats_y):
            pixbuf.copy_area(0, 0, w, h, result, w * xi, h * yi)
    return result


def _best_nrepeats_for_scaling(src_size, max_dest_size):
    min_remainder = N
    min_remainder_nrepeats = 1
    nrepeats = 0
    dest_size = 0
    while dest_size <= max_dest_size:
        nrepeats += 1
        dest_size += src_size
        remainder = dest_size % N
        if remainder < min_remainder:
            min_remainder_nrepeats = nrepeats
            min_remainder = remainder
        if remainder == 0:
            break
    return min_remainder_nrepeats
