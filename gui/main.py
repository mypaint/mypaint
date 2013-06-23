# This file is part of MyPaint.
# Copyright (C) 2007-2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os

from gui import gtk2compat
gobject = gtk2compat.gobject
import gtk
import glib

if not gtk2compat.USE_GTK3:
    required_pygtk = (2, 18, 0)
    required_glib = (2, 28, 0)
    assert glib.glib_version >= required_glib, (
      "You need to upgrade glib. At least version %d.%d.%d is required."
      % required_glib )
    assert gtk.ver >= required_pygtk, (
      "You need to upgrade PyGTK. At least version %d.%d.%d is required."
      % required_pygtk )

from gui import application
from optparse import OptionParser
import sys, time


#: Base version.
#: If this string ends with +git, it will be expanded by the wrapper script to
#: e.g. '1.1.0+gitSHORTCOMMITID' when built from inside a git repository.
MYPAINT_VERSION='1.1.0+git'
# ^ This is used by release.sh, so it must be a valid POSIX shell line too.


def main(datapath, extradata, oldstyle_confpath=None, version=MYPAINT_VERSION):
    """Run MyPaint with `sys.argv`, called from the "mypaint" script.

    :param datapath: The app's read-only data location.
      Where MyPaint should find its static data, e.g. UI definition XML,
      backgrounds, and brush definitions. $PREFIX/share/mypaint is usual.
    :param extradata: Extra search root for finding icons.
      Where to find the defaults for MyPaint's themeable UI icons. This will be
      used in addition to $XDG_DATA_DIRS for the purposes of icon lookup.
      Normally it's $PREFIX/share, to support unusual installations outside the
      usual locations. It should contain an icons/ subdirectory.
    :param oldstyle_confpath: Old-style merged config folder.
      If specified, all user-specific data that MyPaint writes is written here.
      If omitted, this data will be stored under the basedirs returned by
      glib.get_user_config_dir() for settings, and by glib.get_user_data_dir()
      for brushes and backgrounds. On Windows, these will be the same location.
      On POSIX systems, $HOME/.config/mypaint and $HOME/.local/share/mypaint
      are a typical division.
    :param version: full version string for display in the about box.

    The oldstyle_confpath parameter can also be overridden by command-line
    parameters. To support legacy MyPaint configuration dirs, call with
    oldstyle_confpath set to the expansion of ~/.mypaint.

    """

    # Default logfile basename.
    # If it's relative, it's resolved relative to the user config path.
    default_logfile = None
    if sys.platform == 'win32':   # http://gna.org/bugs/?17999
        default_logfile = "mypaint_error.log"

    # Parse command line
    parser = OptionParser('usage: %prog [options] [FILE]')
    parser.add_option('-c', '--config', metavar='DIR',
      default=oldstyle_confpath,
      help='use old-style merged config directory DIR, e.g. ~/.mypaint')
    parser.add_option('-l', '--logfile', metavar='FILE',
      default=default_logfile,
      help='append Python stdout and stderr to FILE (rel. to config location)')
    parser.add_option('-t', '--trace', action="store_true",
      help='print all executed Python statements')
    parser.add_option('-f', '--fullscreen', action="store_true",
      help='start in fullscreen mode')
    parser.add_option("-V", '--version', action="store_true",
      help='print version information and exit')
    options, args = parser.parse_args(sys.argv_unicode[1:])

    # XDG support for new users on POSIX platforms
    if options.config is None:
        encoding = 'utf-8'
        appsubdir = u"mypaint"
        basedir = glib.get_user_data_dir().decode(encoding)
        userdatapath = os.path.join(basedir, appsubdir)
        basedir = glib.get_user_config_dir().decode(encoding)
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
        print 'Python prints are redirected to %s after this one.' % logfilepath
        sys.stdout = sys.stderr = open(logfilepath, 'a', 1)
        print '--- mypaint log %s ---' % time.strftime('%Y-%m-%d %H:%M:%S')

    if options.version:
        print "MyPaint version %s" % (version,)
        sys.exit(0)

    def run():
        print 'DEBUG: user_datapath:', userdatapath
        print 'DEBUG: user_confpath:', userconfpath

        app = application.Application(args,
                app_datapath=datapath, app_extradatapath=extradata,
                user_datapath=userdatapath, user_confpath=userconfpath,
                version=version, fullscreen=options.fullscreen)

        # Recent gtk versions don't allow changing those menu shortcuts by
        # default. <rant>Sigh. This very useful feature used to be the
        # default behaviour even in the GIMP some time ago. I guess
        # assigning a keyboard shortcut without a complicated dialog
        # clicking marathon must have totally upset the people coming from
        # windows.</rant>
        gtksettings = gtk2compat.gtk.settings_get_default()
        gtksettings.set_property('gtk-can-change-accels', True)

        import gtkexcepthook
        func = app.filehandler.confirm_destructive_action
        gtkexcepthook.quit_confirmation_func = func

        # temporary workaround for gtk3 Ctrl-C bug:
        # https://bugzilla.gnome.org/show_bug.cgi?id=622084
        import signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        gtk.main()

    if options.trace:
        import trace
        tracer = trace.Trace(trace=1, count=0)
        tracer.runfunc(run)
    else:
        run()

