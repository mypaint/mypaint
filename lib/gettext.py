# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

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

from __future__ import absolute_import, print_function

from warnings import warn
from gi.repository import GLib

# Set the default encoding like PyGTK
import sys
if sys.version_info < (3, 0):
    reload(sys)
    sys.setdefaultencoding("utf-8")

# Older code in lib imports these as "from gettext import gettext as _".
# Pull them in for backwards compat.
# Might change these to _Glib.dgettext/ngettext instead.

from gettext import gettext
from gettext import ngettext


# Newer code should use C_() even for simple cases, and provide contexts
# for translators.

def C_(context, msgid):
    """Mark a string for translation, with supplied context.

    :param str context: Disambiguating context. Use str literals.
    :param unicode msgid: String to translate. Unicode literals only.
    :returns: the translated string
    :rtype: unicode

    Convenience wrapper around g_dpgettext2. It's a function not a
    macro, but use it as if it was a C macro only: in other words, only
    use string literals so that the strings marked for translation can
    be extracted.

    Writing the context as a str literal and the string marked for
    translation as a unicode lteral makes the fake macro easier to read
    in the code.

    """
    g_dpgettext2 = GLib.dpgettext2
    try:
        result = g_dpgettext2(None, context, msgid)
    except TypeError as e:
        # Expect "Argument 0 does not allow None as a value" sometimes.
        # This is a known problem with Ubuntu Server 12.04 when testing
        # lib - that version of g_dpgettext2() does not appear to allow
        # NULL for its first arg.
        wtmpl = "C_(): g_dpgettext2() raised %r. Try a newer GLib?"
        warn(
            wtmpl % (e,),
            RuntimeWarning,
            stacklevel = 1,
        )
        return msgid
    else:
        assert isinstance(result, str), (
            "Expected g_dpgettext2() to return str, got %r instead"
            % (type(result),),
        )
        result = result.decode("utf-8")
        return result
