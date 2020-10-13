# This file is part of MyPaint.
# Copyright (C) 2007-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2013-2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
from __future__ import absolute_import

import os
import sys
import logging
# This imports the global gettext package, not lib/gettext
import gettext
import lib.config as config

logger = logging.getLogger(__file__)


def debug_locale_data(locale, locale_categories):
    for category in sorted(locale_categories):
        try:
            logger.debug(
                "getlocale(%s): %r",
                category,
                locale.getlocale(getattr(locale, category)),
            )
        except Exception:
            # TODO: Remove this when Py2 support is dropped
            logger.exception("Problem when getting locale (upstream)")


def init_gettext(localepath):
    """Initialize locales and gettext.

    This must be done before importing any translated python modules
    (to get global strings translated, especially brushsettings.py).

    """

    import locale
    import lib.i18n

    # Required in Windows for the "Region and Language" settings
    # to take effect.
    lib.i18n.set_i18n_envvars()
    lib.i18n.fixup_i18n_envvars()

    # Internationalization
    # Source of many a problem down the line, so lotsa debugging here.
    logger.debug("localepath: %r", localepath)
    logger.debug("getdefaultlocale(): %r", lib.i18n.getdefaultlocale())

    # Set the user's preferred locale.
    # https://docs.python.org/2/library/locale.html
    # Required in Windows for the "Region and Language" settings
    # to take effect.
    try:
        setlocale_result = locale.setlocale(locale.LC_ALL, '')
    except locale.Error:
        logger.exception("setlocale(LC_ALL, '') failed")
    else:
        logger.debug("setlocale(LC_ALL, ''): %r", setlocale_result)

    # More debugging: show the state after setlocale().
    logger.debug(
        "getpreferredencoding(): %r",
        locale.getpreferredencoding(do_setlocale=False),
    )

    if logger.isEnabledFor(logging.DEBUG):
        locale_categories = [
            s for s in dir(locale)
            if s.startswith("LC_") and s != "LC_ALL"
        ]
        debug_locale_data(locale, locale_categories)

    # Low-level bindtextdomain with paths.
    # This is still required to hook GtkBuilder up with translated
    # strings; the gettext() way doesn't cut it for external stuff
    # yanked in over GI.
    # https://bugzilla.gnome.org/show_bug.cgi?id=574520#c26
    bindtextdomain = None
    bind_textdomain_codeset = None
    textdomain = None

    # Try the POSIX/Linux way first.
    try:
        bindtextdomain = locale.bindtextdomain
        bind_textdomain_codeset = locale.bind_textdomain_codeset
        textdomain = locale.textdomain
    except AttributeError:
        logger.warning(
            "No bindtextdomain builtins found in module 'locale'."
        )
        logger.info(
            "Trying platform-specific fallback hacks to find "
            "bindtextdomain funcs.",
        )
        # Windows Python binaries tend not to expose bindtextdomain and
        # its buddies anywhere they can be called.
        if sys.platform == 'win32':
            libintl = None
            import ctypes
            libnames = [
                'libintl-8.dll',  # native for MSYS2's MINGW32
                'libintl.dll',  # no known cases, but a potential fallback
                'intl.dll',  # some old recipes off the internet
            ]
            for libname in libnames:
                try:
                    libintl = ctypes.cdll.LoadLibrary(libname)
                    bindtextdomain = libintl.bindtextdomain
                    bindtextdomain.argtypes = (
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                    )
                    bindtextdomain.restype = ctypes.c_char_p
                    bind_textdomain_codeset = libintl.bind_textdomain_codeset
                    bind_textdomain_codeset.argtypes = (
                        ctypes.c_char_p,
                        ctypes.c_char_p,
                    )
                    bind_textdomain_codeset.restype = ctypes.c_char_p
                    textdomain = libintl.textdomain
                    textdomain.argtypes = (
                        ctypes.c_char_p,
                    )
                    textdomain.restype = ctypes.c_char_p
                except Exception:
                    logger.exception(
                        "Windows: attempt to load bindtextdomain funcs "
                        "from %r failed (ctypes)",
                        libname,
                    )
                else:
                    logger.info(
                        "Windows: found working bindtextdomain funcs "
                        "in %r (ctypes)",
                        libname,
                    )
                    break
        else:
            logger.error(
                "No platform-specific fallback for locating bindtextdomain "
                "is known for %r",
                sys.platform,
            )
    # libmypaint's domain/path must be bound if its messages are not installed
    # in a location gettext is configured to look in (system-dependent and
    # set at compile time), which don't always include e.g:
    # /usr/local/share/locale/ or $HOME/.local/share/locale
    # It must also be set here for the appimage, where the path will always
    # be the same for libmypaint's and mypaint's message catalogs.
    localepath_libmypaint = config.libmypaint_locale_dir
    if not localepath_libmypaint:
        localepath_libmypaint = localepath

    # Bind text domains, i.e. tell libintl+GtkBuilder and Python's where
    # to find message catalogs containing translations.
    textdomains = [
        ("mypaint", localepath),
        (config.libmypaint_version, localepath_libmypaint),
    ]
    defaultdom = "mypaint"
    codeset = "UTF-8"
    for dom, path in textdomains:
        # Some people choose not to install any translation files.
        if not os.path.isdir(path):
            logger.warning(
                "No translations for %s. Missing locale dir %r.",
                dom, path,
            )
            continue
        # Only call the C library gettext setup funcs if there's a
        # complete set from the same source.
        # Required for translatable strings in GtkBuilder XML
        # to be translated.
        if bindtextdomain and bind_textdomain_codeset and textdomain:
            assert os.path.exists(path)
            assert os.path.isdir(path)
            if sys.platform == 'win32':
                p = bindtextdomain(dom.encode('utf-8'), path.encode('utf-8'))
                c = bind_textdomain_codeset(
                    dom.encode('utf-8'), codeset.encode('utf-8')
                )
            else:
                p = bindtextdomain(dom, path)
                c = bind_textdomain_codeset(dom, codeset)
            logger.debug("C bindtextdomain(%r, %r): %r", dom, path, p)
            logger.debug(
                "C bind_textdomain_codeset(%r, %r): %r",
                dom, codeset, c,
            )
        # Call the implementations in Python's standard gettext module
        # too.  This has proper cross-platform support, but it only
        # initializes the native Python "gettext" module.
        # Required for marked strings in Python source to be translated.
        # See http://docs.python.org/release/2.7/library/locale.html
        p = gettext.bindtextdomain(dom, path)
        c = gettext.bind_textdomain_codeset(dom, codeset)
        logger.debug("Python bindtextdomain(%r, %r): %r", dom, path, p)
        logger.debug(
            "Python bind_textdomain_codeset(%r, %r): %r",
            dom, codeset, c,
        )
    if bindtextdomain and bind_textdomain_codeset and textdomain:
        if sys.platform == 'win32':
            d = textdomain(defaultdom.encode('utf-8'))
        else:
            d = textdomain(defaultdom)
        logger.debug("C textdomain(%r): %r", defaultdom, d)
    d = gettext.textdomain(defaultdom)
    logger.debug("Python textdomain(%r): %r", defaultdom, d)
