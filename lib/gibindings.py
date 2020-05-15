# Copyright (C) 2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Specifies gi module requirements & imports the binding modules

The GObject Introspection (gi) bindings should always be imported via this
module, and never directly from ``gi.repository``. This is to make sure that
the correct versions are always specified before import, even when individual
modules are being loaded in isolation (for purposes of testing).
"""

from __future__ import division, print_function

import gi

gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
gi.require_version("Gio", "2.0")
gi.require_version("GLib", "2.0")
gi.require_version("GObject", "2.0")
gi.require_version("Gtk", "3.0")
gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")

from gi.repository import Gdk  # noqa
from gi.repository import GdkPixbuf  # noqa
from gi.repository import Gio  # noqa
from gi.repository import GLib  # noqa
from gi.repository import GObject  # noqa
from gi.repository import Pango  # noqa
from gi.repository import PangoCairo  # noqa


# The import of the actual Gtk bindings needs to be deferred until the locale
# has been configured, in order for locale-specific layouts to be respected
# (right-to-left layouts).
# A GtkWrapper instance is used as a go-between that only triggers the real
# import when an attribute of the module is requested.

class GtkWrapper(object):

    def __getattribute__(self, attr):
        from gi.repository import Gtk as RealGtk
        return getattr(RealGtk, attr)

    def __getattr__(self, attr):
        return self.__getattribute__(attr)


Gtk = GtkWrapper()
