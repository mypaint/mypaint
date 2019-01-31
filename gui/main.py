# This file is part of MyPaint.
# Copyright (C) 2007-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2013-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Command-line handling - traditional main() function."""

## Imports *nothing involving mypaintlib at this point*

from __future__ import division, print_function

import os
import sys
import logging
import warnings

import lib.gichecks
from gi.repository import Gtk
from gi.repository import GdkPixbuf
from optparse import OptionParser

from lib.meta import MYPAINT_VERSION
import lib.glib

logger = logging.getLogger(__name__)


## Method defs

def _init_gtk_workarounds():
    """Initialize some workarounds for unoptimal GTK behavior"""
    # Via https://code.google.com/p/quodlibet/source/browse/quodlibet/
    logger.debug("Adding GTK workarounds...")

    # On windows the default variants only do ANSI paths, so replace them.
    # In some typelibs they are replaced by default, in some don't..
    if os.name == "nt":
        for name in ["new_from_file_at_scale", "new_from_file_at_size",
                     "new_from_file"]:
            cls = GdkPixbuf.Pixbuf
            func = getattr(cls, name + "_utf8", None)
            if func:
                logger.debug(
                    "Monkeypatching GdkPixbuf.Pixbuf.%s with %r",
                    name, func,
                )
                setattr(cls, name, func)

    logger.debug("GTK workarounds added.")


def main(datapath, iconspath, oldstyle_confpath=None, version=MYPAINT_VERSION):
    """Run MyPaint with `sys.argv_unicode`, called from the "mypaint" script.

    :param unicode datapath: The app's read-only data location.
    :param unicode iconspath: Extra search root for finding icons.
    :param unicode oldstyle_confpath: Old-style merged config folder.
    :param unicode version: full version string for the about box.

    The ``datapath`` parameter defines where MyPaint should find its
    static data, e.g. UI definition XML, backgrounds, and brush
    definitions. $PREFIX/share/mypaint is usual.

    The iconspath`` parameter tells MyPaint where to find its themeable UI
    icons. This will be used in addition to $XDG_DATA_DIRS for the
    purposes of icon lookup.  Normally it's $PREFIX/share/icons.

    If specified oldstyle_confpath must be a single directory path.  all
    user-specific data that MyPaint writes is written here.  If omitted,
    this data will be stored under the basedirs returned by
    GLib.get_user_config_dir() for settings, and by
    GLib.get_user_data_dir() for brushes and backgrounds. On Windows,
    these will be the same location.  On POSIX systems,
    $HOME/.config/mypaint and $HOME/.local/share/mypaint are a typical
    division.

    See `lib.meta` for details of what normally goes in `version`.

    """

    # Init the workaround before we start importing anything which
    # could still be using gtk2compat.
    _init_gtk_workarounds()

    # GLib user dirs: cache them now for greatest compatibility.
    # Importing mypaintlib before the 1st call to g_get_user*_dir()
    # breaks GLib for obscure reasons.
    # This needs to be done after i18n setup, or Windows configurations
    # with non-ASCII character in %USERPROFILE% will break.
    lib.glib.init_user_dir_caches()

    # mypaintlib import is performed first in gui.application now.
    from gui import application

    # Default logfile basename.
    # If it's relative, it's resolved relative to the user config path.
    default_logfile = None

    # Parse command line
    parser = OptionParser('usage: %prog [options] [FILE]')
    parser.add_option(
        '-c',
        '--config',
        metavar='DIR',
        default=oldstyle_confpath,
        help='use old-style merged config directory DIR, e.g. ~/.mypaint'
    )
    parser.add_option(
        '-l',
        '--logfile',
        metavar='FILE',
        default=default_logfile,
        help='log console messages to FILE (rel. to config location)'
    )
    parser.add_option(
        '-t',
        '--trace',
        action="store_true",
        help='print all executed Python statements'
    )
    parser.add_option(
        '-f',
        '--fullscreen',
        action="store_true",
        help='start in fullscreen mode'
    )
    parser.add_option(
        "-V",
        '--version',
        action="store_true",
        help='print version information and exit'
    )
    options, args = parser.parse_args(sys.argv_unicode[1:])

    # XDG support for new users on POSIX platforms
    if options.config is None:
        appsubdir = u"mypaint"
        basedir = lib.glib.get_user_data_dir()
        userdatapath = os.path.join(basedir, appsubdir)
        basedir = lib.glib.get_user_config_dir()
        userconfpath = os.path.join(basedir, appsubdir)
    else:
        userdatapath = options.config
        userconfpath = options.config

    # Log to the specified location
    # Prepend the data path if the user gave a relative location.
    if options.logfile:
        logfilepath = os.path.join(userdatapath, options.logfile)
        logdirpath, logfilebasename = os.path.split(logfilepath)
        if not os.path.isdir(logdirpath):
            os.makedirs(logdirpath)
        logger.info("Copying log messages to %r", logfilepath)
        logfile_handler = logging.FileHandler(
            logfilepath, mode="a",
            encoding="utf-8",
        )
        logfile_format = "%(asctime)s;%(levelname)s;%(name)s;%(message)s"
        logfile_handler.setFormatter(logging.Formatter(logfile_format))
        root_logger = logging.getLogger(None)
        root_logger.addHandler(logfile_handler)

    if os.environ.get("MYPAINT_DEBUG", False):
        logger.critical("Test critical message, please ignore")
        warnings.resetwarnings()
        logging.captureWarnings(True)

    if options.version:
        # Output (rather than log) the version
        print("MyPaint version %s" % (version, ))
        sys.exit(0)

    def run():
        logger.debug('user_datapath: %r', userdatapath)
        logger.debug('user_confpath: %r', userconfpath)

        app_state_dirs = application.StateDirs(
            app_data = datapath,
            app_icons = iconspath,
            user_data = userdatapath,
            user_config = userconfpath,
        )
        app = application.Application(
            filenames = args,
            state_dirs = app_state_dirs,
            version = version,
            fullscreen = options.fullscreen,
        )

        settings = Gtk.Settings.get_default()
        dark = app.preferences.get("ui.dark_theme_variant", True)
        settings.set_property("gtk-application-prefer-dark-theme", dark)

        from gui import gtkexcepthook
        func = app.filehandler.confirm_destructive_action
        gtkexcepthook.quit_confirmation_func = func

        # temporary workaround for gtk3 Ctrl-C bug:
        # https://bugzilla.gnome.org/show_bug.cgi?id=622084
        import signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        Gtk.main()

    if options.trace:
        import trace
        tracer = trace.Trace(trace=1, count=0)
        tracer.runfunc(run)
    else:
        run()
