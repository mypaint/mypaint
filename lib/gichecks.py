# Copyright (C) 2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Specifies gi module requirements for MyPaint at runtime.

Import this before any "from gi import <module>" lines to suppress those
annoying "<module> was imported without specifying a version first"
warning messages.

"""

from __future__ import division, print_function

import gi
gi.require_version('GdkPixbuf', '2.0')
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
gi.require_version('PangoCairo', '1.0')
