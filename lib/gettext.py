# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import absolute_import

"""Function equivalents of (GLib) gettext's C macros.

Recommended usage:

  >>> from lib.gettext import C_
  >>> from lib.gettext import ngettext

Also supported, but mildly deprecated (consider C_ instead!):

  >>> from lib.gettext import gettext as _

Lots of older code uses ``from gettext import gettext as _``.
Don't do that in new code: pull in this module as ``lib.gettext``.
Importing this module should still work from within lib/ if code
still uses a relative import, however.

"""

from gi.repository import GLib as _GLib


# Older code in lib imports these as "from gettext import gettext as _".
# Pull them in for backwards compat.
# Might change these to _Glib.dgettext/ngettext instead.

from gettext import gettext
from gettext import ngettext


# Newer code should use C_() even for simple cases, and provide contexts
# for translators.

def C_(context, msgid):
    """Translated string with supplied context.

    Convenience wrapper around g_dpgettext2. It's a function not a
    macro, but use it as if it was a C macro only.

    """
    return _GLib.dpgettext2("mypaint", context, msgid)
    # Explicit domain for the sake of running the tests on Travis-CI,
    # which uses an older version of GLib without [allow-none] as arg 0.
