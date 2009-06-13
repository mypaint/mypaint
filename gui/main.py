# This file is part of MyPaint.
# Copyright (C) 2007-2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import sys, os
import gtk
from gui import application

# main entry, called from the "mypaint" script
def main(datapath, confpath):

    def usage_exit():
        print sys.argv[0], '[OPTION]... [FILENAME]'
        print 'Options:'
        print '  -c /path/to/config   use this directory instead of ~/.mypaint/'
        print '  -p                   profile (debug only; simulate some strokes and quit)'
        print '  -e                   use default pygtk exception handling (no dialog)'
        sys.exit(1)

    filename = None
    profile = False
    excepthook = True

    args = sys.argv[1:]
    while args:
        arg = args.pop(0)
        if arg == '-c':
            confpath = args.pop(0)
        elif arg == '-p':
            profile = True
        elif arg == '-e':
            excepthook = False
        elif arg.startswith('-'):
            usage_exit()
        else:
            if filename:
                print 'Cannot open more than one file!'
                sys.exit(2)
            filename = arg
            filename = filename.replace('file:///', '/') # some filebrowsers do this
            if not os.path.isfile(filename):
                print 'File', filename, 'does not exist!'
                sys.exit(2)

    print 'confpath =', confpath
    app = application.Application(datapath, confpath, filename, profile)

    # Recent gtk versions don't allow changing those menu shortcuts by
    # default. <rant>Sigh. This very useful feature used to be the
    # default behaviour even in the GIMP some time ago. I guess
    # assigning a keyboard shortcut without a complicated dialog
    # clicking marathon must have totally upset the people coming from
    # windows.</rant>
    gtksettings = gtk.settings_get_default()
    gtksettings.set_property('gtk-can-change-accels', True)

    if excepthook:
        import gtkexcepthook

    gtk.main()
