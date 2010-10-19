# This file is part of MyPaint.
# Copyright (C) 2007-2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import pygtk
pygtk.require('2.0')
import gtk
required = (2, 16, 0)
assert gtk.ver >= required, 'You need to upgrade PyGTK, at least version %d.%d.%d is required.' % required

from gui import application
from optparse import OptionParser
import sys, time

# main entry, called from the "mypaint" script
def main(datapath, confpath):

    parser = OptionParser('usage: %prog [options] [FILE]')
    parser.add_option('-c', '--config', metavar='DIR', default=confpath,
                    help='use config directory DIR instead of ~/.mypaint/')
    parser.add_option('-l', '--logfile', metavar='FILE', default=None,
                    help='append python stdout and stderr to FILE')
    parser.add_option('-t', '--trace', action="store_true",
                    help='print all exectued python statements')
    options, args = parser.parse_args()

    if options.logfile:
        print 'Python prints are redirected to', options.logfile, 'after this one.'
        sys.stdout = sys.stderr = open(options.logfile, 'a', 1)
        print '--- mypaint log %s ---' % time.strftime('%Y-%m-%d %H:%M:%S')

    confpath = options.config.decode(sys.getfilesystemencoding())

    def run():
        print 'confpath =', options.config
        app = application.Application(datapath, confpath, args)

        # Recent gtk versions don't allow changing those menu shortcuts by
        # default. <rant>Sigh. This very useful feature used to be the
        # default behaviour even in the GIMP some time ago. I guess
        # assigning a keyboard shortcut without a complicated dialog
        # clicking marathon must have totally upset the people coming from
        # windows.</rant>
        gtksettings = gtk.settings_get_default()
        gtksettings.set_property('gtk-can-change-accels', True)

        import gtkexcepthook
        func = app.filehandler.confirm_destructive_action
        gtkexcepthook.quit_confirmation_func = func
        gtk.main()

    if options.trace:
        import trace
        tracer = trace.Trace(trace=1, count=0)
        tracer.runfunc(run)
    else:
        run()

