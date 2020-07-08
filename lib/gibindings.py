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

# This may look pointless, but is required to set up types
# prior to their use, in dynamic property creation. See
# gui/sliderwidget.py for an instance of this.
for i in dir(Gdk):
    getattr(Gdk, i)


# The import of the actual Gtk bindings needs to be deferred until the locale
# has been configured, in order for locale-specific layouts to be respected
# (right-to-left layouts).
# A GtkWrapper instance is used as a go-between that only triggers the real
# import when an attribute of the module is requested.

class GtkWrapper(object):

    _initialized = False

    def __getattr__(self, attr):
        # Deferred import
        from gi.repository import Gtk as RealGtk
        # Create attributes on this instance reflecting
        # everything in the real proxy module - this also allows the use of the
        # derived python versions of classes and enums in custom properties,
        # prior to any real use of those classes. To see the consequences,
        # remove the following 4 lines and check the resulting error messages.
        if not self._initialized:
            for att in (a for a in dir(RealGtk) if not a.startswith("_")):
                setattr(self, att, getattr(RealGtk, att))
            self._initialized = True
        return getattr(RealGtk, attr)


Gtk = GtkWrapper()
