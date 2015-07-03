# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2007-2015 by Martin Renold <martinxyz@gmx.ch> and
# the MyPaint Developement Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Program meta-information: consts & display.

See also `lib.meta`.

"""

## Imports

from gi.repository import Gtk

from lib.gettext import C_
import lib.meta


## Program-related string constants

COPYRIGHT_STRING = C_(
    "About dialog: copyright statement",
    u"Copyright (C) 2005-2015\n"
    u"Martin Renold and the MyPaint Development Team"
)
WEBSITE_URI = "http://mypaint.org"
LICENSE_SUMMARY = C_(
    "About dialog: license summary",
    u"This program is free software; you can redistribute it and/or modify "
    u"it under the terms of the GNU General Public License as published by "
    u"the Free Software Foundation; either version 2 of the License, or "
    u"(at your option) any later version.\n"
    u"\n"
    u"This program is distributed in the hope that it will be useful, "
    u"but WITHOUT ANY WARRANTY. See the COPYING file for more details."
)

## Credits-related string constants

# Strings for specific tasks, all translated

_TASK_PROGRAMMING = C_(
    "About dialog: credits: tasks",
    u"programming"
)
_TASK_PORTING = C_(
    "About dialog: credits: tasks",
    u"portability"
)
_TASK_PROJECT_MANAGEMENT = C_(
    "About dialog: credits: tasks",
    u"project management"
)
_TASK_BRUSHES = C_(
    "About dialog: credits: tasks: brush presets and icons",
    u"brushes"
)
_TASK_PATTERNS = C_(
    "About dialog: credits: tasks: background paper textures",
    u"patterns"
)
_TASK_TOOL_ICONS = C_(
    "About dialog: credits: tasks: icons for internal tools",
    u"tool icons"
)
_TASK_APP_ICON = C_(
    "About dialog: credits: tasks: the main application icon",
    u"desktop icon"
)
_TASK_PALETTES = C_(
    "About dialog: credits: tasks: palettes",
    u"palettes"
)

_TASK_DOCS = C_(
    "About dialog: credits: tasks: docs, manuals and HOWTOs",
    u"documentation"
)
_TASK_SUPPORT = C_(
    "About dialog: credits: tasks: user support",
    u"support"
)
_TASK_OUTREACH = C_(
    "About dialog: credits: tasks: outreach (social media, ads?)",
    u"outreach"
)
_TASK_COMMUNITY = C_(
    "About dialog: credits: tasks: running or building a community",
    u"community"
)

_TASK_COMMA = C_(
    "About dialog: credits: tasks: joiner punctuation",
    u", ",
)

# List contributors in order of their appearance.
# The author's name is always written in their native script,
# and is not marked for translation. It may also have:
# transcriptions (Latin, English-ish) in brackets following, and/or
# a quoted ’nym in Latin script.
# For <given(s)> <surname(s)> combinations,
# a quoted publicly-known alias may go after the given name.

_AUTHOR_CREDITS = [
    u"Martin Renold (%s)" % _TASK_PROGRAMMING,
    u"Yves Combe (%s)" % _TASK_PORTING,
    u"Popolon (%s)" % _TASK_PROGRAMMING,
    u"Clement Skau (%s)" % _TASK_PROGRAMMING,
    u"Jon Nordby (%s)" % _TASK_PROGRAMMING,
    u"Álinson Santos (%s)" % _TASK_PROGRAMMING,
    u"Tumagonx (%s)" % _TASK_PORTING,
    u"Ilya Portnov (%s)" % _TASK_PROGRAMMING,
    u"Jonas Wagner (%s)" % _TASK_PROGRAMMING,
    u"Luka Čehovin (%s)" % _TASK_PROGRAMMING,
    u"Andrew Chadwick (%s)" % _TASK_PROGRAMMING,
    u"Till Hartmann (%s)" % _TASK_PROGRAMMING,
    u'David Grundberg (%s)' % _TASK_PROGRAMMING,
    u"Krzysztof Pasek (%s)" % _TASK_PROGRAMMING,
    u"Ben O’Steen (%s)" % _TASK_PROGRAMMING,
    u"Ferry Jérémie (%s)" % _TASK_PROGRAMMING,
    u"しげっち ‘sigetch’ (%s)" % _TASK_PROGRAMMING,
    u"Richard Jones (%s)" % _TASK_PROGRAMMING,
    u"David Gowers (%s)" % _TASK_PROGRAMMING,
]
_ARTIST_CREDITS = [
    u"Artis Rozentāls (%s)" % _TASK_BRUSHES,
    u"Popolon (%s)" % _TASK_BRUSHES,
    u"Marcelo ‘Tanda’ Cerviño (%s)" % _TASK_COMMA.join([
        _TASK_PATTERNS,
        _TASK_BRUSHES,
    ]),
    u"David Revoy (%s)" % _TASK_COMMA.join([
        _TASK_BRUSHES,
        _TASK_TOOL_ICONS,
    ]),
    u"Ramón Miranda (%s)" % _TASK_COMMA.join([
        _TASK_BRUSHES,
        _TASK_PATTERNS,
    ]),
    u"Enrico Guarnieri ‘Ico_dY’ (%s)" % _TASK_BRUSHES,
    u'Sebastian Kraft (%s)' % _TASK_APP_ICON,
    u"Nicola Lunghi (%s)" % _TASK_PATTERNS,
    u"Toni Kasurinen (%s)" % _TASK_BRUSHES,
    u"Сан Саныч ‘MrMamurk’ (%s)" % _TASK_PATTERNS,
    u"Andrew Chadwick (%s)" % _TASK_TOOL_ICONS,
    u"Ben O’Steen (%s)" % _TASK_TOOL_ICONS,
    u"Guillaume Loussarévian ‘Kaerhon’ (%s)" % _TASK_BRUSHES,
    u"Jakub Steiner ‘jimmac’ (%s)" % _TASK_COMMA.join([
        _TASK_APP_ICON,
        _TASK_PALETTES,
    ]),
]
_TRANSLATOR_CREDITS = C_(
    "About dialog: credits: translator credits (your name(s) here!)",
    u"translator-credits",
)


## About dialog for the app

def run_about_dialog(mainwin, app):
    """Runs MyPaint's about window as a transient modal dialog."""
    d = Gtk.AboutDialog()
    d.set_transient_for(mainwin)
    d.set_program_name(lib.meta.MYPAINT_PROGRAM_NAME)
    d.set_version(app.version)
    d.set_copyright(COPYRIGHT_STRING)
    d.set_website(WEBSITE_URI)
    d.set_logo(app.pixmaps.mypaint_logo)
    d.set_license(LICENSE_SUMMARY)
    d.set_wrap_license(True)
    d.set_authors(_AUTHOR_CREDITS)
    d.set_artists(_ARTIST_CREDITS)
    d.set_translator_credits(_TRANSLATOR_CREDITS)
    d.run()
    d.destroy()

