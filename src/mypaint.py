#!/usr/bin/env python
# This file is part of MyPaint.
# Copyright (C) 2007-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2013-2020 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Platform-dependent setup, and program launch.

This script does all the platform dependent stuff.
Its main task is to configure paths to access MyPaint's modules and
other resources, and set up paths for i18n message catalogs.

It then passes control to gui.main.main() for command line launching.

"""

## Imports (standard Python only at this point)

import sys
import os
import os.path
from os.path import join, isdir, dirname, abspath
import re
import logging

logger = logging.getLogger('mypaint')
if sys.version_info >= (3,):
    xrange = range
    unicode = str


## Logging classes

class ColorFormatter (logging.Formatter):
    """Minimal ANSI formatter, for use with non-Windows console logging."""

    # ANSI control sequences for various things
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    FG = 30
    BG = 40
    LEVELCOL = {
        "DEBUG": "\033[%02dm" % (FG+BLUE,),
        "INFO": "\033[%02dm" % (FG+GREEN,),
        "WARNING": "\033[%02dm" % (FG+YELLOW,),
        "ERROR": "\033[%02dm" % (FG+RED,),
        "CRITICAL": "\033[%02d;%02dm" % (FG+RED, BG+BLACK),
    }
    BOLD = "\033[01m"
    BOLDOFF = "\033[22m"
    ITALIC = "\033[03m"
    ITALICOFF = "\033[23m"
    UNDERLINE = "\033[04m"
    UNDERLINEOFF = "\033[24m"
    RESET = "\033[0m"

    def _replace_bold(self, m):
        return self.BOLD + m.group(0) + self.BOLDOFF

    def _replace_underline(self, m):
        return self.UNDERLINE + m.group(0) + self.UNDERLINEOFF

    def format(self, record):
        record = logging.makeLogRecord(record.__dict__)
        msg = record.msg
        token_formatting = [
            (re.compile(r'%r'), self._replace_bold),
            (re.compile(r'%s'), self._replace_bold),
            (re.compile(r'%\+?[0-9.]*d'), self._replace_bold),
            (re.compile(r'%\+?[0-9.]*f'), self._replace_bold),
        ]
        for token_re, repl in token_formatting:
            msg = token_re.sub(repl, msg)
        record.msg = msg
        record.reset = self.RESET
        record.bold = self.BOLD
        record.boldOff = self.BOLDOFF
        record.italic = self.ITALIC
        record.italicOff = self.ITALICOFF
        record.underline = self.UNDERLINE
        record.underlineOff = self.UNDERLINEOFF
        record.levelCol = ""
        if record.levelname in self.LEVELCOL:
            record.levelCol = self.LEVELCOL[record.levelname]
        return super(ColorFormatter, self).format(record)


## Helper functions


def win32_unicode_argv():
    # fix for https://gna.org/bugs/?17739
    # code mostly comes from http://code.activestate.com/recipes/572200/
    """Uses shell32.GetCommandLineArgvW to get sys.argv as a list of Unicode
    strings.

    Versions 2.x of Python don't support Unicode in sys.argv on
    Windows, with the underlying Windows API instead replacing multi-byte
    characters with '?'.
    """
    try:
        from ctypes import POINTER, byref, cdll, c_int, windll
        from ctypes.wintypes import LPCWSTR, LPWSTR

        get_cmd = cdll.kernel32.GetCommandLineW
        get_cmd.argtypes = []
        get_cmd.restype = LPCWSTR
        get_argv = windll.shell32.CommandLineToArgvW
        get_argv.argtypes = [LPCWSTR, POINTER(c_int)]

        get_argv.restype = POINTER(LPWSTR)
        cmd = get_cmd()
        argc = c_int(0)
        argv = get_argv(cmd, byref(argc))
        if argc.value > 0:
            # Remove Python executable if present
            if argc.value - len(sys.argv) == 1:
                start = 1
            else:
                start = 0
            return [argv[i] for i in xrange(start, argc.value)]
    except Exception:
        logger.exception(
            "Specialized Win32 argument handling failed. Please "
            "help us determine if this code is still needed, "
            "and submit patches if it's not."
        )
        logger.warning("Falling back to POSIX-style argument handling")


def get_paths():

    # Convert sys.argv to a list of unicode objects
    # (actually converting sys.argv confuses gtk, thus we add a new variable)
    # Post-Py3: almost certainly not needed, but check *all* platforms
    # before removing this stuff.

    sys.argv_unicode = None
    if sys.platform == 'win32':
        sys.argv_unicode = win32_unicode_argv()

    if sys.argv_unicode is None:
        argv_unicode = []
        for s in sys.argv:
            if hasattr(s, "decode"):
                s = s.decode(sys.getfilesystemencoding())
            argv_unicode.append(s)
        sys.argv_unicode = argv_unicode

    # Script and its location, in canonical absolute form
    prefix = dirname(abspath(sys.argv_unicode[0]))
    assert isinstance(prefix, unicode)

    # Usually, when installed with setup.py, MYPAINT_DIR_PATHS
    # is defined in the module, containing all paths we need to set up.
    if 'MYPAINT_DIR_PATHS' in globals():
        logger.info("Running from installed script...")
        global MYPAINT_DIR_PATHS
        paths = MYPAINT_DIR_PATHS
        for k, v in paths.items():
            paths[k] = abspath(join(prefix, v))
        logger.info("...using static relative paths")
        purelib_path = paths['purelib']
        platlib_path = paths['platlib']
        base_data_path = paths['data']
        locale_path = join(base_data_path, 'locale')
        icons_path = join(base_data_path, 'icons')
        data_path = join(base_data_path, 'mypaint')
    elif all(map(isdir, ['gui', 'lib', 'desktop'])):
        purelib_path = platlib_path = data_path = u'.'
        locale_path = join('build', 'locale')
        if not isdir(locale_path):
            logger.warning(
                'Locale files not found - translations will not work!')
        icons_path = u'desktop/icons'
    else:
        logger.critical("Installation layout: unknown!")
        raise RuntimeError("Unknown install type; could not determine paths")

    sys.path.insert(0, purelib_path)
    sys.path.insert(0, platlib_path)

    # There is no need to return the datadir of mypaint-data.
    # It will be set at build time. I still check brushes presence.
    import lib.config
    # Allow brushdir path to be set relative to the installation prefix
    # Use string-formatting *syntax*, but not actual formatting. This is
    # to not have to deal with the remote possibility of a legitimate
    # brushdir path with brace-enclosed components (legal UNIX-paths).
    brushdir_path = lib.config.mypaint_brushdir
    pref_key = "{installation-prefix}/"
    if brushdir_path.startswith(pref_key):
        logger.info("Using brushdir path relative to installation-prefix")
        brushdir_rel = brushdir_path[len(pref_key):]
        brushdir_path = abspath(join(prefix, "..", brushdir_rel))
        # When using a prefix-relative path, replace it with the absolute path
        lib.config.mypaint_brushdir = brushdir_path
    if not os.path.isdir(brushdir_path):
        logger.critical('Default brush collection not found!')
        logger.critical('It should have been here: %r', brushdir_path)
        sys.exit(1)

    # When using a prefix-relative path, replace it with the absolute path
    lib.config.mypaint_brushdir = brushdir_path

    old_confpath = check_old_style_config()
    assert isinstance(data_path, unicode)
    assert isinstance(icons_path, unicode)

    return data_path, icons_path, old_confpath, locale_path


def check_old_style_config():
    # Old style config file and user data locations.
    # Return None if using XDG will be correct.
    if sys.platform == 'win32':
        old_confpath = None
    else:
        from lib import fileutils
        homepath = fileutils.expanduser_unicode(u'~')
        old_confpath = join(homepath, '.mypaint/')

    if old_confpath:
        if not os.path.isdir(old_confpath):
            old_confpath = None
        else:
            wiki_base = "https://github.com/mypaint/mypaint/wiki/"
            wiki_page = wiki_base + "Migrating-settings-&-data-from-1.1"
            logger.info("There is an old-style configuration area in %r",
                        old_confpath)
            logger.info("Its contents can be migrated to $XDG_CONFIG_HOME "
                        "and $XDG_DATA_HOME if you wish.")
            logger.info("For further instructions, see: %s" % wiki_page)
    assert isinstance(old_confpath, unicode) or old_confpath is None
    return old_confpath


# Program launch

if __name__ == '__main__':
    # Console logging
    log_format = "%(levelname)s: %(name)s: %(message)s"
    console_handler = logging.StreamHandler(stream=sys.stderr)
    no_ansi_platforms = ["win32"]
    can_use_ansi_formatting = (
        (sys.platform not in no_ansi_platforms)
        and sys.stderr.isatty()
    )
    if can_use_ansi_formatting:
        log_format = (
            "%(levelCol)s%(levelname)s: "
            "%(bold)s%(name)s%(reset)s%(levelCol)s: "
            "%(message)s%(reset)s"
        )
        console_formatter = ColorFormatter(log_format)
    else:
        console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    debug = os.environ.get("MYPAINT_DEBUG", False)
    logging_level = logging.DEBUG if debug else logging.INFO
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging_level)
    if logging_level == logging.DEBUG:
        logger.info("Debugging output enabled via MYPAINT_DEBUG")

    # Path determination
    datapath, iconspath, old_confpath, localepath = get_paths()
    logger.debug('datapath: %r', datapath)
    logger.debug('iconspath: %r', iconspath)
    logger.debug('old_confpath: %r', old_confpath)
    logger.debug('localepath: %r', localepath)

    # Allow an override version string to be burned in during build.  Comes
    # from an active repository's git information and build timestamp, or
    # the release_info file from a tarball release.
    if 'MYPAINT_VERSION_CEREMONIAL' in globals():
        version = MYPAINT_VERSION_CEREMONIAL
    else:
        version = None

    # Start the app.
    from gui import main
    main.main(
        datapath,
        iconspath,
        localepath,
        old_confpath,
        version=version,
        debug=debug,
    )
