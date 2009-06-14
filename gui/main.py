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
from optparse import OptionParser

# main entry, called from the "mypaint" script
def main(datapath, confpath):

    parser = OptionParser()
    parser.add_option('-c', '--config', metavar='DIR',
                    help='use this config directory instead of ~/.mypaint/')
    parser.add_option('-p', '--profile', action='store_true', default = False,
                    help='(debug only; simulate some strokes and quit)')
    parser.add_option('-e', '--except-hook', action='store_true', default=True,
                    dest='excepthook', help='use default pygtk exception handling (no dialog')
    (options, args) = parser.parse_args()

    if not options.config:
        options.config = confpath 
    
    if len(args) == 0:
        filename = None
    elif len(args) == 1:
        filename = args[0]
        filename = filename.replace('file:///', '/') # some filebrowsers do this
        
        # fixme? testing for file existence in advance is unpythonic and opens up for a possible race condition
        # would have to solve it later, which could also be an advantage for GUI user feedback
        # also consider not killing the app when a file doesnt exist, but rather create the file
        if not os.path.isfile(filename): 
            print 'File', filename, 'does not exist!'
            sys.exit(2)
    else:
        print 'Cannot open more than one file!'
        sys.exit(2)

    print 'confpath =', options.config
    app = application.Application(datapath, options.config, filename, options.profile)

    # Recent gtk versions don't allow changing those menu shortcuts by
    # default. <rant>Sigh. This very useful feature used to be the
    # default behaviour even in the GIMP some time ago. I guess
    # assigning a keyboard shortcut without a complicated dialog
    # clicking marathon must have totally upset the people coming from
    # windows.</rant>
    gtksettings = gtk.settings_get_default()
    gtksettings.set_property('gtk-can-change-accels', True)

    if options.excepthook:
        print 'using gtkexcepthook'
        import gtkexcepthook

    gtk.main()
