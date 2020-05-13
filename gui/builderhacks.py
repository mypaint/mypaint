# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2013-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Hacks for loading stuff from GtkBuilder files."""

## Imports
from __future__ import division, print_function

from lib.gibindings import Gtk

import lib.xml


## Public functions

def add_objects_from_template_string(builder, buffer_, object_ids, params):
    """Templatizes, parses, merges, and returns objects from a Builder UI-def

    This function wraps `Gtk.Builder.add_objects_from_string()`, with the
    addition that the `buffer_` parameter, and each element of `object_ids` is
    formatted using `str.format()` using `params` before use. This templatizing
    is required to produce a different result for the string buffer of XML
    data, and for each object ID.

    :param builder: a Gtk.Buider
    :param buffer_: the string to templatize then parse
    :param object_ids: list of object names to build (after templatizing)
    :param params: dict of template params
    :returns: a list of constructed objects

    The constructed objects are returned in a Python list if this wrapped
    method call is successful.

    When templatizing the XML fragment, parameter values will be escaped using
    `lib.xml.escape()`. Therefore `params` is limited to fairly simple
    dicts.

    """
    object_ids2 = []
    for oid in object_ids:
        oid2 = oid.format(**params)
        if oid == oid2:
            raise ValueError("object_id %s unchanged after .format()ing"
                             % oid)
        object_ids2.append(oid2)
    params_esc = {}
    for p, v in params.items():
        params_esc[p] = lib.xml.escape(v)
    buffer_2 = buffer_.format(**params_esc)
    if buffer_2 == buffer_:
        raise ValueError("buffer_ unchanged after .format()ing")
    result = []
    if builder.add_objects_from_string(buffer_2, object_ids2):
        for oid2 in object_ids2:
            obj2 = builder.get_object(oid2)
            assert obj2 is not None
            result.append(obj2)
    return result


## Module testing


_TEST_TEMPLATE = """
<interface>
    <object class="GtkLabel" id="never_instantiated">
        <property name="label">This should never be instantiated</property>
    </object>
    <object class="GtkButton" id="button_{id}">
        <property name="label">{label}</property>
        <signal name="clicked" handler="button_{id}_clicked"/>
    </object>
</interface>
"""


def _test():
    """Interactive module test function"""
    import os
    import sys
    vbox = Gtk.VBox()
    builder = Gtk.Builder()

    # Handlers can find out about their template values by parsing their
    # name (using the GtkBuildable interface). Alternatively, you can set
    # up private attributes in the instantiation loop.

    def _test_button_clicked_cb(widget):
        id_ = Gtk.Buildable.get_name(widget)
        if isinstance(id_, bytes):
            id_ = id_.decode("utf-8")
        print("Clicked: id=%r" % (id_, ))
        print("          i=%r" % (widget._i, ))

    # Unicode is supported in IDs and template values.
    # The XML template may be plain ASCII since escape() is used when
    # filling it.
    object_ids = [u"button_{id}"]
    words = [u"à", u"chacun", u"son", u"goût"]
    for i in words:
        params = {"id": i, "label": i.upper()}
        objs = add_objects_from_template_string(builder, _TEST_TEMPLATE,
                                                object_ids, params)
        for w in objs:
            w.connect("clicked", _test_button_clicked_cb)
            vbox.pack_start(w, True, True, 0)
            w._i = i
    # The label should never be instantiated by this code. In fact, only
    # the four buttons should.
    for obj in builder.get_objects():
        assert isinstance(obj, Gtk.Button)
    # Remainder of the demo code
    window = Gtk.Window()
    window.add(vbox)
    window.set_title(os.path.basename(sys.argv[0]))
    window.connect("destroy", lambda *a: Gtk.main_quit())
    window.set_size_request(250, 200)
    window.show_all()
    Gtk.main()


if __name__ == '__main__':
    _test()
