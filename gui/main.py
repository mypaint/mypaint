# This file is part of MyPaint.
# Copyright (C) 2007-2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import gettext
import gtk
from gui import application
from optparse import OptionParser

# main entry, called from the "mypaint" script
def main(datapath, confpath, localepath):

    parser = OptionParser('usage: %prog [options] [FILE]')
    parser.add_option('-c', '--config', metavar='DIR', default=confpath,
                    help='use this config directory instead of ~/.mypaint/')
    options, args = parser.parse_args()

    gettext.bindtextdomain("mypaint", localepath)
    gettext.textdomain("mypaint")

    print 'confpath =', options.config
    app = application.Application(datapath, options.config, args)

    # Recent gtk versions don't allow changing those menu shortcuts by
    # default. <rant>Sigh. This very useful feature used to be the
    # default behaviour even in the GIMP some time ago. I guess
    # assigning a keyboard shortcut without a complicated dialog
    # clicking marathon must have totally upset the people coming from
    # windows.</rant>
    gtksettings = gtk.settings_get_default()
    gtksettings.set_property('gtk-can-change-accels', True)

    import gtkexcepthook
    gtk.main()
