# This file is part of MyPaint.
# Copyright (C) 2007-2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""
This script does all the platform dependent stuff. Its main task is
to figure out where the python modules are.
"""

def get_paths():
    import sys, os
    join = os.path.join

    lib_shared='share/mypaint/'
    # note: some distros use lib64 instead, they have to edit this...
    lib_compiled='lib/mypaint/'

    scriptdir=os.path.dirname(sys.argv[0])

    # this script is installed as $prefix/bin. We just need $prefix to continue.
    #pwd=os.getcwd() # why????
    #dir_install=os.path.normpath(join(pwd,scriptdir)) # why????
    dir_install=scriptdir # same, except maybe if scriptdir is relative...

    if os.path.basename(dir_install) == 'bin':
        prefix=os.path.dirname(dir_install)
        libpath=join(prefix, lib_shared)
        libpath_compiled = join(prefix, lib_compiled)
        sys.path.insert(0, libpath)
        sys.path.insert(0, libpath_compiled)
        localepath = join(prefix, 'share/locale')
    else:
        # we are not installed
        prefix=None
        libpath='.'
        localepath = 'po'

    try: # just for a nice error message
        from lib import mypaintlib
    except ImportError:
        print
        print "We are not correctly installed or compiled!"
        print 'script: "%s"' % sys.argv[0]
        if prefix:
            print 'deduced prefix: "%s"' % prefix
            print 'lib_shared: "%s"' % libpath
            print 'lib_compiled: "%s"' % libpath_compiled
        print
        raise

    datapath = libpath
    if not os.path.isdir(join(datapath, 'brushes')):
        print 'Default brush collection not found! It should have been here:'
        print datapath
        raise sys.exit(1)

    homepath =  os.path.expanduser('~')
    if homepath == '~':
        confpath = join(prefix, 'UserData')
    else:
        confpath = join(homepath, '.mypaint/')

    return datapath, confpath, localepath

def psyco_opt():
    # This helps on slow PCs where the python overhead dominates.
    # (30% higher framerate measured on 533MHz CPU; startup slowdown below 20%)
    # Note: python -O -O does not help.
    import psyco
    psyco.full()
    print 'Psyco being used'


if __name__ == '__main__':
    try:
        psyco_opt()
    except ImportError:
        pass
    paths = get_paths()
    from gui import main
    main.main(*paths)
