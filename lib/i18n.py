# -*- coding: utf-8 -*-
# Copyright 2006 Joe Wreschnig
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation


"""I18n setup routines.

Stolen from QuodLibet (thanks, lazka!) <https://github.com/quodlibet
/quodlibet/blob/76bb75fa677a8a1db830fbe47d0000788184e903/quodlibet
/quodlibet/util/i18n.py#L58>.

"""

from __future__ import division, print_function

import os
import sys
import locale
import logging
logger = logging.getLogger(__name__)


def bcp47_to_language(code):
    """Takes a BCP 47 language identifier and returns a value suitable for the
    LANGUAGE env var.
    Only supports a small set of inputs and might return garbage..
    """

    if code == "zh-Hans":
        return "zh_CN"
    elif code == "zh-Hant":
        return "zh_TW"

    parts = code.split("-")

    # we only support ISO 639-1
    if not _is_iso_639_1(parts[0]):
        return parts[0].replace(":", "")
    lang_subtag = parts[0]

    region = ""
    if len(parts) >= 2 and _is_iso_639_1(parts[1]):
        region = parts[1]
    elif len(parts) >= 3 and _is_iso_639_1(parts[2]):
        region = parts[2]

    if region:
        return "%s_%s" % (lang_subtag, region)
    return lang_subtag


def _is_iso_639_1(s):
    return len(s) == 2 and s.isalpha()


def osx_locale_id_to_lang(id_):
    """Converts a NSLocale identifier to something suitable for LANG"""

    if "_" not in id_:
        return id_
    # id_ can be "zh-Hans_TW"
    parts = id_.rsplit("_", 1)
    ll = parts[0]
    ll = bcp47_to_language(ll).split("_")[0]
    return "%s_%s" % (ll, parts[1])


def set_i18n_envvars():
    """Set the LANG/LANGUAGE environment variables if not set.

    Covers the cases where the current platform doesn't use them by
    default (OS X, Windows). In these systems, it will pick up defaults
    by other OS-specific means.

    """

    if os.name == "nt":
        # This MyPaint code turns out to be quite portable.
        # On Win7 Professional, or Win7 Enterprise/Ultimate when there
        # are no UI language packs installed, it will pick up the user's
        # Region and Language → Formats setting when the code below
        # doesn't.
        langs = []
        deflocale = locale.getdefaultlocale()
        if deflocale:
            langs.append(deflocale[0])
        # Let the user's chosen UI language contribute too.
        # I think only the chosen UI language affects this.
        # Not sure what the priority order should be between the two.
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            for lang_k32 in [k32.GetUserDefaultUILanguage(),
                             k32.GetSystemDefaultUILanguage()]:
                lang = locale.windows_locale.get(lang_k32)
                if lang not in langs:
                    langs.append(lang)
        except:
            logger.exception("Windows: couldn't get UI language via ctypes")
            logger.warning(
                "Windows: falling back to Region & Language -> Formats, "
                "then POSIX mechanisms."
            )
        if langs:
            os.environ.setdefault('LANG', langs[0])
            os.environ.setdefault('LANGUAGE', ":".join(langs))
        logger.info("Windows: LANG=%r", os.environ.get("LANG"))
        logger.info("Windows: LANGUAGE=%r", os.environ.get("LANGUAGE"))

    elif sys.platform == "darwin":
        try:
            from AppKit import NSLocale
        except ImportError:
            logger.exception("OSX: failed to import AppKit.NSLocale")
            logger.warning("OSX: falling back to POSIX mechanisms.")
        else:
            logger.info(
                "OSX: imported AppKit.NSLocale OK. "
                "Will use for LANG, and for LANGUAGE order."
            )
            locale_id = NSLocale.currentLocale().localeIdentifier()
            lang = osx_locale_id_to_lang(locale_id)
            os.environ.setdefault('LANG', lang)
            preferred_langs = NSLocale.preferredLanguages()
            if preferred_langs:
                languages = map(bcp47_to_language, preferred_langs)
                os.environ.setdefault('LANGUAGE', ":".join(languages))
        logger.info("OSX: LANG=%r", os.environ.get("LANG"))
        logger.info("OSX: LANGUAGE=%r", os.environ.get("LANGUAGE"))

    else:
        logger.info("POSIX: LANG=%r", os.environ.get("LANG"))
        logger.info("POSIX: LANGUAGE=%r", os.environ.get("LANGUAGE"))


def fixup_i18n_envvars():
    """Sanitizes env vars before gettext can use them.

    LANGUAGE should support a priority list of languages with fallbacks,
    but doesn't work due to "en" not being known to gettext (This could
    be solved by providing a en.po in [QuodLibet] but all other
    libraries don't define it either). This tries to fix that.

    FIXME: check if MyPaint needs this - we have an en_GB after all!

    """

    try:
        langs = os.environ["LANGUAGE"].split(":")
    except KeyError:
        return

    # So, this seems to be an undocumented feature where C selects "no
    # translation". Insert it after the first en/en_XX so that if that
    # lect of English is not found it falls back to the source language,
    # which is US English.
    sanitized = []
    san_c_i = -1
    for lang in langs:
        sanitized.append(lang)
        if lang.startswith("en") and san_c_i < 0:
            san_c_i = len(sanitized)
    if san_c_i >= 0:
        sanitized.insert(san_c_i, "C")
    os.environ["LANGUAGE"] = ":".join(sanitized)
    logger.debug(
        "Sanitized the priority list in LANGUAGE to %r",
        os.environ["LANGUAGE"],
    )
