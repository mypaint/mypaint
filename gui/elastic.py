# This file is part of MyPaint.
# Copyright (C) 2011 by Andrew Chadwick <andrewc-git@piffle.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gtk
from gtk import gdk


class ElasticContainer:
    """Mixin for containers which mirror certain size changes of descendents

    Descendents which wish to report internally-generated size changes should
    add the ElasticContent mixin, and should be packed into a container that
    derives from ElasticContainer. More than one ElasticContent widget can be
    packed under an ElasticContainer, and each can report different types of
    resizes - however the sub-hierarchies cannot overlap. You -can- nest
    ElasticContainers though: it's the outer one that receives the resize
    request.
    """

    def __init__(self):
        """Mixin constructor (construct as a gtk.Widget before calling)
        """
        self.__last_size = None
        self.__saved_size_request = None

    def mirror_elastic_content_resize(self, dx, dy):
        """Resize by a given amount.

        This is called by some ElasticContent widget below here in the
        hierarchy when it notices a request-size change on itself.
        """
        p = self.parent
        while p is not None:
            if isinstance(p, ElasticContainer):
                # propagate up and don't handle here
                p.mirror_elastic_content_resize(dx, dy)
                return
            p = p.parent
        self.__saved_size_request = self.get_size_request()
        alloc = self.get_allocation()
        w = alloc.width+dx
        h = alloc.height+dy
        if isinstance(self, gtk.Window):
            self.resize(w, h)
        self.set_size_request(w, h)
        self.queue_resize()


class ElasticContent:
    """Mixin for GTK widgets which want some parent to change size to match.
    """

    def __init__(self, mirror_vertical=True, mirror_horizontal=True):
        """Mixin constructor (construct as a gtk.Widget before calling)

        The options control which size changes are reported to the
        ElasticContainer ancestor and how:

            `mirror_vertical`:
                mirror vertical size changes

            `mirror_horizontal`:
                mirror horizontal size changes

        """
        self.__vertical = mirror_vertical
        self.__horizontal = mirror_horizontal
        if not mirror_horizontal and not mirror_vertical:
            return
        self.__last_req = None
        self.__expose_connid = self.connect_after("expose-event",
            self.__after_expose_event)
        self.__notify_parent = False
        self.connect_after("size-request", self.__after_size_request)

    def __after_expose_event(self, widget, event):
        # Begin notifying changes to the ancestor after the first expose event.
        # It doesn't matter for widgets that know their size before hand.
        # Assume widgets which initially don't know their size *do* know their
        # proper size after their first draw, even if they drew themselves
        # wrongly and now have to resize and do another size-request...
        connid = self.__expose_connid
        if not connid:
            return
        self.__expose_connid = None
        self.__notify_parent = True
        self.disconnect(connid)

    def __after_size_request(self, widget, req):
        # Catch the final value of each size-request, calculate the
        # difference needed and throw it back up the widget hierarchy
        # to interested parties.
        if not self.__last_req:
            dx, dy = 0, 0
        else:
            w0, h0 = self.__last_req
            dx, dy = req.width - w0, req.height - h0
        self.__last_req = (req.width, req.height)
        if not self.__notify_parent:
            return
        p = self.parent
        while p is not None:
            if isinstance(p, ElasticContainer):
                if not self.__vertical:
                    dy = 0
                if not self.__horizontal:
                    dx = 0
                if dy != 0 or dx != 0:
                    p.mirror_elastic_content_resize(dx, dy)
                break
            if isinstance(p, ElasticContent):
                break
            p = p.parent


# XXX unused, may remove
class ElasticVBox (gtk.VBox, ElasticContainer):
    __gtype_name__ = "ElasticVBox"

    def __init__(self, *args, **kwargs):
        gtk.VBox.__init__(self, *args, **kwargs)
        ElasticContainer.__init__(self)


class ElasticExpander (gtk.Expander, ElasticContent):
    """Buildable elastic-content version of a regular GtkExpander.
    """
    __gtype_name__ = "ElasticExpander"

    def __init__(self, *args, **kwargs):
        gtk.Expander.__init__(self, *args, **kwargs)
        ElasticContent.__init__(self, mirror_horizontal=False,
                                mirror_vertical=True)


class ElasticWindow (gtk.Window, ElasticContainer):
    """Buildable elastic-container version of a regular gtk.Window.
    """
    __gtype_name__ = "ElasticWindow"

    def __init__(self, *args, **kwargs):
        gtk.Window.__init__(self, *args, **kwargs)
        ElasticContainer.__init__(self)


if __name__ == '__main__':
    win1 = ElasticWindow()
    vbox = gtk.VBox()
    ee = ElasticExpander()
    ee.set_label("expand (expands window)")

    l1 = gtk.Label("Main content\ngoes here.")
    l2 = gtk.Label("Extra content\n\nBlah blah")
    vbox.pack_start(l1, True, True)
    vbox.pack_start(ee, False, True)
    ee.add(l2)
    win1.add(vbox)
    win1.set_title("elastic test")
    win1.connect("destroy", lambda *a: gtk.main_quit())
    win1.show_all()

    gtk.main()


