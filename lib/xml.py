# -*- encoding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Helpers, and constants for the XML dialects MyPaint uses."""


## Consts for XML dialects

OPENRASTER_MEDIA_TYPE = "image/openraster"
OPENRASTER_VERSION = u"0.0.5"


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

    Works like ``cgi.escape()``, but adds character ref encoding for
    characters which lie outside the ASCII range.
    The returned string is ASCII.

    """
    u = u.replace("&", "&amp;")
    u = u.replace("<", "&lt;")
    u = u.replace(">", "&gt;")
    if apos:
        u = u.replace("'", "&apos;")
    if quot:
        u = u.replace('"', "&quot;")
    return u.encode("ascii", "xmlcharrefreplace")


## Module testing


def _test(self):
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()
