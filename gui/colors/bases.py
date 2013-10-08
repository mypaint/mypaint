# This file is part of MyPaint.
# Copyright (C) 2012 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Base widgets for the colour selector module.
"""

import logging
logger = logging.getLogger(__name__)

import gtk
from gtk import gdk
import cairo


class CachedBgWidgetMixin(object):
    """Provides widgets with a cached background and bg/fg drawing callbacks.

    The background is invalidated whenever the size changes, or when the
    overridable background-validity method returns something new.
    """

    __bg = None   #: The cached background: a `cairo.ImageSurface`.
    __bg_validity = None #: Validity token for `__bg`.

    def __init__(self):
        """Instantiate, binding events; call *after* `gtk.Widget.__init__()`.
        """
        try:
            self.connect("draw", self.__draw_cb)
        except TypeError:
            self.connect("expose-event", self.__expose_cb)
        self.connect("size-allocate", self.__size_allocate_cb)
        self.__bg = None
        self.__bg_validity = None

    def _get_cairo_context(self, event):
        """Return a clipped, translated Cairo context for an expose event.
        """
        raise NotImplementedError


    def __expose_cb(self, widget, event):
        self.__draw_cb(widget, self._get_cairo_context(event))


    def __draw_cb(self, widget, cr):
        bg_valid = self.__bg is not None
        if bg_valid:
            validity = self.get_background_validity()
            if validity != self.__bg_validity:
                bg_valid = False
        if not bg_valid:
            self.__rerender_background()
            assert self.__bg is not None
        alloc = self.get_allocation()
        cr.set_source_surface(self.__bg, 0, 0)
        cr.paint()
        self.paint_foreground_cb(cr, alloc.width, alloc.height)


    def __size_allocate_cb(self, widget, alloc):
        self.clear_background()
        self.__rerender_background()


    def _get_background_size(self):
        alloc = self.get_allocation()
        return alloc.width, alloc.height


    def __rerender_background(self):
        w, h = self._get_background_size()
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        cr = cairo.Context(surf)
        self.render_background_cb(cr, w, h)
        self.__bg = surf
        self.__bg_validity = self.get_background_validity()


    def get_background_validity(self):
        """Return a value which, if changed, means the bg must be redrawn.

        These validity tokens may be anything comparable by ``==``, and should
        be simple to compute (or at least, simpler than re-rendering the
        background every time).
        """
        raise NotImplementedError


    def render_background_cb(self, cr, w, h, *kw):
        """Render the background when needed.
        """
        raise NotImplementedError


    def paint_foreground_cb(self, cr, w, h):
        """Painting the foreground over the background.
        """
        raise NotImplementedError


    def clear_background(self):
        """Clears the cached background, making it invalid.
        """
        self.__bg = None
        self.__bg_validity = None
        self.queue_draw()



class CachedBgDrawingArea (CachedBgWidgetMixin, gtk.DrawingArea):

    def __init__(self):
        gtk.DrawingArea.__init__(self)
        CachedBgWidgetMixin.__init__(self)

    def _get_cairo_context(self, event):
        """Transformed Cairo context, suitable for windowless impls.
        """
        gdk_win = self.get_window()
        cr = gdk_win.cairo_create()
        # clip to the expose area, which is always relative to gdk_win
        cr.rectangle(event.area)
        cr.clip()
        if not self.get_has_window():
            # gdk_win belongs to some parent widget
            # need to translate cr for correct alignment.
            x, y, w, h = self.get_allocation()
            cr.translate(x, y)
        return cr



class IconRenderable(object):
    """Mixin for anything which can render itself as an XDG icon via Cairo.

    Typically a cached icon file from disk will be quicker and more convenient,
    and this interface can be used for generating iconic forms for saving to
    disk.
    """


    def render_as_icon(self, cr, size):
        """Renders as an icon into a Cairo context (unimplemented).

        The icon pixel size, `size`, is one of 48, 32, 24, 22 or 16.
        """
        raise NotImplementedError


    def save_icon_tree(self, dir_name, icon_name):
        """Renders a full set of XDG icons under a given root directory.
        """
        import os
        dpi = 90.0
        for size in (48, 32, 24, 22, 16, 'scalable'):
            if size == 'scalable':
                path = "%s/hicolor/scalable/actions" % (dir_name,)
                if not os.path.exists(path):
                    os.makedirs(path)
                filename = "%s/%s.svg" % (path, icon_name)
                s = 48.0
                scale = 72/dpi
                pts = s*scale
                surf = cairo.SVGSurface(filename, pts, pts)
                cr = cairo.Context(surf)
                cr.scale(scale, scale)
            else:
                path = "%s/hicolor/%dx%d/actions" % (dir_name, size, size)
                if not os.path.exists(path):
                    os.makedirs(path)
                filename = "%s/%s.png" % (path, icon_name)
                surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
                s = size
                cr = cairo.Context(surf)
            self.render_as_icon(cr, size=int(s))
            surf.flush()
            if size != 'scalable':
                surf.write_to_png(filename)
            logger.info("rendered %r (size=%s)...", filename, size)


