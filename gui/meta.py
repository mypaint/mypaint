# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2007-2015 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2015-2016 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Program meta-information: consts & display.

See also `lib.meta`.

"""

import os
import platform

## Imports
import sys

import cairo

import lib.meta
from lib.gettext import C_
from lib.gibindings import GdkPixbuf, GLib, Gtk
from lib.xml import escape

## Program-related string constants

COPYRIGHT_STRING = C_(
    "About dialog: copyright statement",
    "Copyright (C) 2005-2020\n" "Martin Renold and the MyPaint Development Team",
)
WEBSITE_URI = "http://mypaint.org"
LICENSE_SUMMARY = C_(
    "About dialog: license summary",
    "This program is free software; you can redistribute it and/or modify "
    "it under the terms of the GNU General Public License as published by "
    "the Free Software Foundation; either version 2 of the License, or "
    "(at your option) any later version.\n"
    "\n"
    "This program is distributed in the hope that it will be useful, "
    "but WITHOUT ANY WARRANTY. See the COPYING file for more details.",
)

## Credits-related string constants

# Strings for specific tasks, all translated

_TASK_PROGRAMMING = C_("About dialog: credits: tasks", "programming")
_TASK_PORTING = C_("About dialog: credits: tasks", "portability")
_TASK_PROJECT_MANAGEMENT = C_("About dialog: credits: tasks", "project management")
_TASK_BRUSHES = C_("About dialog: credits: tasks: brush presets and icons", "brushes")
_TASK_PATTERNS = C_(
    "About dialog: credits: tasks: background paper textures", "patterns"
)
_TASK_TOOL_ICONS = C_(
    "About dialog: credits: tasks: icons for internal tools", "tool icons"
)
_TASK_APP_ICON = C_(
    "About dialog: credits: tasks: the main application icon", "desktop icon"
)
_TASK_PALETTES = C_("About dialog: credits: tasks: palettes", "palettes")

_TASK_DOCS = C_(
    "About dialog: credits: tasks: docs, manuals and HOWTOs", "documentation"
)
_TASK_SUPPORT = C_("About dialog: credits: tasks: user support", "support")
_TASK_OUTREACH = C_(
    "About dialog: credits: tasks: outreach (social media, ads?)", "outreach"
)
_TASK_COMMUNITY = C_(
    "About dialog: credits: tasks: running or building a community",
    "community",
)

_TASK_COMMA = C_(
    "About dialog: credits: tasks: joiner punctuation",
    ", ",
)

# List contributors in order of their appearance.
# The author's name is always written in their native script,
# and is not marked for translation. It may also have:
# transcriptions (Latin, English-ish) in brackets following, and/or
# a quoted ’nym in Latin script.
# For <given(s)> <surname(s)> combinations,
# a quoted publicly-known alias may go after the given name.

# TODO: Simplify/unify how the dialog is built.
#  - This should really be built from a giant matrix.
#  - Each task type should determine a tab of the about dialog
#  - Contributors will still appear on multiple tabs,
#     - but that'd be automatic now
#  - Keep it reasonably simple, so that contributors can add themselves!
#  - Maybe get rid of the (%s) formatting junk?
#  - Split out ’nyms and transliterations too?

_AUTHOR_CREDITS = [
    "Martin Renold (%s)" % _TASK_PROGRAMMING,
    "Yves Combe (%s)" % _TASK_PORTING,
    "Popolon (%s)" % _TASK_PROGRAMMING,
    "Clement Skau (%s)" % _TASK_PROGRAMMING,
    "Jon Nordby (%s)" % _TASK_PROGRAMMING,
    "Álinson Santos (%s)" % _TASK_PROGRAMMING,
    "Tumagonx (%s)" % _TASK_PORTING,
    "Ilya Portnov (%s)" % _TASK_PROGRAMMING,
    "Jonas Wagner (%s)" % _TASK_PROGRAMMING,
    "Luka Čehovin (%s)" % _TASK_PROGRAMMING,
    "Andrew Chadwick (%s)"
    % _TASK_COMMA.join(
        [
            _TASK_PROGRAMMING,
            _TASK_PROJECT_MANAGEMENT,
            _TASK_PORTING,
        ]
    ),
    "Till Hartmann (%s)" % _TASK_PROGRAMMING,
    "David Grundberg (%s)" % _TASK_PROGRAMMING,
    "Krzysztof Pasek (%s)" % _TASK_PROGRAMMING,
    "Ben O’Steen (%s)" % _TASK_PROGRAMMING,
    "Ferry Jérémie (%s)" % _TASK_PROGRAMMING,
    "しげっち ‘sigetch’ (%s)" % _TASK_PROGRAMMING,
    "Richard Jones (%s)" % _TASK_PROGRAMMING,
    "David Gowers (%s)" % _TASK_PROGRAMMING,
    "Micael Dias (%s)" % _TASK_PROGRAMMING,
    "Anna Harren (%s)"
    % _TASK_COMMA.join(
        [
            _TASK_COMMUNITY,
            _TASK_PROGRAMMING,
            _TASK_DOCS,
        ]
    ),
    "Sebastien Leon (%s)" % _TASK_PROGRAMMING,
    "Ali Lown (%s)" % _TASK_PROGRAMMING,
    "Brien Dieterle (%s)" % _TASK_PROGRAMMING,
    "Jenny Wong (%s)" % _TASK_PROGRAMMING,
    "Dmitry Utkin ‘loentar’ (%s)" % _TASK_PROGRAMMING,
    "ShadowKyogre (%s)" % _TASK_PROGRAMMING,
    "Albert Westra (%s)"
    % _TASK_COMMA.join(
        [
            _TASK_COMMUNITY,
            _TASK_PROGRAMMING,
        ]
    ),
    "Cortexer (%s)" % _TASK_PROGRAMMING,
    "Elliott Sales de Andrade (%s)" % _TASK_PORTING,
    "Alberto Leiva Popper (%s)" % _TASK_PROGRAMMING,
    "Alinson Xavier (%s)" % _TASK_PROGRAMMING,
    "Jesper Lloyd (%s)"
    % _TASK_COMMA.join(
        [
            _TASK_PROGRAMMING,
            _TASK_PROJECT_MANAGEMENT,
        ]
    ),
]
_ARTIST_CREDITS = [
    "Artis Rozentāls (%s)" % _TASK_BRUSHES,
    "Popolon (%s)" % _TASK_BRUSHES,
    "Marcelo ‘Tanda’ Cerviño (%s)"
    % _TASK_COMMA.join(
        [
            _TASK_PATTERNS,
            _TASK_BRUSHES,
        ]
    ),
    "David Revoy (%s)"
    % _TASK_COMMA.join(
        [
            _TASK_BRUSHES,
            _TASK_TOOL_ICONS,
            _TASK_OUTREACH,
        ]
    ),
    "Ramón Miranda (%s)"
    % _TASK_COMMA.join(
        [
            _TASK_BRUSHES,
            _TASK_PATTERNS,
        ]
    ),
    "Enrico Guarnieri ‘Ico_dY’ (%s)" % _TASK_BRUSHES,
    "Sebastian Kraft (%s)" % _TASK_APP_ICON,
    "Nicola Lunghi (%s)" % _TASK_PATTERNS,
    "Toni Kasurinen (%s)" % _TASK_BRUSHES,
    "Сан Саныч ‘MrMamurk’ (%s)" % _TASK_PATTERNS,
    "Andrew Chadwick (%s)" % _TASK_TOOL_ICONS,
    "Ben O’Steen (%s)" % _TASK_TOOL_ICONS,
    "Guillaume Loussarévian ‘Kaerhon’ (%s)" % _TASK_BRUSHES,
    "Jakub Steiner ‘jimmac’ (%s)"
    % _TASK_COMMA.join(
        [
            _TASK_APP_ICON,
            _TASK_PALETTES,
        ]
    ),
    "ShadowKyogre (%s)" % _TASK_TOOL_ICONS,
    "Albert Westra (%s)" % _TASK_TOOL_ICONS,
    "Brien Dieterle (%s)" % _TASK_BRUSHES,
    "Jesper Lloyd (%s)" % _TASK_APP_ICON,
]
_TRANSLATOR_CREDITS = C_(
    "About dialog: credits: translator credits (your name(s) here!)",
    # TRANSLATORS: THIS SHOULD NOT BE TRANSLATED LITERALLY
    # TRANSLATORS: The "translation" of this string should be a list of names
    # TRANSLATORS: of the people who have contributed to the translation to
    # TRANSLATORS: this language. One name per line, optionally with an email
    # TRANSLATORS: address within angle brackets "<email@somewhere.com>", and
    # TRANSLATORS: optionally with a year or year range indicating when the
    # TRANSLATORS: contributions were made, e.g: 2005 or 2010-2012 etc.
    "translator-credits",
)


## About dialog for the app


def get_libs_version_string():
    """Get a string describing the versions of important libs.

    >>> type(get_libs_version_string()) == str
    True

    """
    versions = [
        (
            "Python",
            "{major}.{minor}.{micro}".format(
                major=sys.version_info.major,
                minor=sys.version_info.minor,
                micro=sys.version_info.micro,
            ),
        ),
        (
            "GTK",
            "{major}.{minor}.{micro}".format(
                major=Gtk.get_major_version(),
                minor=Gtk.get_minor_version(),
                micro=Gtk.get_micro_version(),
            ),
        ),
        ("GdkPixbuf", GdkPixbuf.PIXBUF_VERSION),
        ("Cairo", cairo.cairo_version_string()),  # NOT cairo.version
        (
            "GLib",
            "{major}.{minor}.{micro}".format(
                major=GLib.MAJOR_VERSION,
                minor=GLib.MINOR_VERSION,
                micro=GLib.MICRO_VERSION,
            ),
        ),
    ]
    return ", ".join([" ".join(t) for t in versions])


def run_about_dialog(mainwin, app):
    """Runs MyPaint's about window as a transient modal dialog."""
    d = Gtk.AboutDialog()
    d.set_transient_for(mainwin)
    d.set_program_name(lib.meta.MYPAINT_PROGRAM_NAME)
    p = lib.meta.MYPAINT_PROGRAM_NAME
    v_raw = app.version or lib.meta.MYPAINT_VERSION
    v = "{mypaint_version}\n\n<small>({libs_versions})</small>".format(
        mypaint_version=escape(v_raw),
        libs_versions=escape(get_libs_version_string()),
    )
    if os.name == "nt":
        # The architecture matters more on windows, to the extent that
        # we release two separate builds with differing names.
        bits_str, linkage = platform.architecture()
        bits_str = {
            "32bit": "w32",
            "64bit": "w64",
        }.get(bits_str, bits_str)
        p = "{progname} {w_bits}".format(
            progname=lib.meta.MYPAINT_PROGRAM_NAME,
            w_bits=bits_str,
        )
    # Some strings have markup characters escaped in GTK because
    # of standard markup being applied to that info section, noted below.

    # escapes input
    d.set_program_name(p)
    # does NOT escape input
    d.set_version(v)
    # escapes input
    d.set_copyright(COPYRIGHT_STRING)
    # only url (set_website_label escapes input)
    d.set_website(WEBSITE_URI)
    d.set_logo(app.pixmaps.mypaint_logo)
    # does NOT escape input
    d.set_license(LICENSE_SUMMARY)
    d.set_wrap_license(True)
    # Credits sections use some custom undocumented simple parsing
    # to produce markup for email links, website links etc.
    # NOTE: Said parsing does not care if input is escaped or not...
    d.set_authors(_AUTHOR_CREDITS)
    d.set_artists(_ARTIST_CREDITS)
    d.set_translator_credits(_TRANSLATOR_CREDITS)
    d.run()
    d.destroy()
