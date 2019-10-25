# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Helpers, and constants for the XML dialects MyPaint uses."""

## Imports

from __future__ import absolute_import, division, print_function

from lib.pycompat import PY3

import xml.etree.ElementTree as _ET

## Consts for XML dialects
# Namespaces are registered by importing this module.

OPENRASTER_MEDIA_TYPE = "image/openraster"
OPENRASTER_VERSION = u"0.0.5"
OPENRASTER_MYPAINT_NS = "http://mypaint.org/ns/openraster"

_OPENRASTER_NAMESPACES = {
    "mypaint": OPENRASTER_MYPAINT_NS,
}
for prefix, uri in _OPENRASTER_NAMESPACES.items():
    _ET.register_namespace(prefix, uri)


## Helper functions

def indent_etree(elem, level=0):
    """Indent an XML etree.

    This does not seem to come with python?
    Source: http://effbot.org/zone/element-lib.htm#prettyprint

    """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_etree(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def xsd2bool(arg):
    """Converts an XSD boolean datatype string from XML to a Python bool

      >>> xsd2bool("true")
      True
      >>> xsd2bool("false")
      False
      >>> xsd2bool(1)
      True
      >>> xsd2bool(0)
      False

    Ref: http://www.w3.org/TR/xmlschema-2/#boolean

    """
    return str(arg).lower() in ("true", "1")


def escape(u, quot=False, apos=False):
    """Escapes a Unicode string for use in XML/HTML.

      >>> u = u'<foo> & "bar"'
      >>> escape(u)
      '&lt;foo&gt; &amp; "bar"'
      >>> escape(u, quot=True)
      '&lt;foo&gt; &amp; &quot;bar&quot;'
      >>> escape(None) is None
      True

    Works like ``cgi.escape()``, but adds character ref encoding for
    characters which lie outside the ASCII range.
    The returned str is ASCII.

    """
    if u is None:
        return None
    u = u.replace("&", "&amp;")
    u = u.replace("<", "&lt;")
    u = u.replace(">", "&gt;")
    if apos:
        u = u.replace("'", "&apos;")
    if quot:
        u = u.replace('"', "&quot;")
    s = u.encode("ascii", "xmlcharrefreplace")
    if PY3:
        s = s.decode("ascii")
    return s


## Module testing


def _test(self):
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()
