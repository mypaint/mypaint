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
import sys, os

def get_paths():
    join = os.path.join

    lib_shared='share/mypaint/'
    # note: some distros use lib64 instead, they have to edit this...
    lib_compiled='lib/mypaint/'
    
    arg0 = unicode(sys.argv[0], sys.getfilesystemencoding())
    scriptdir=os.path.dirname(arg0)

    # this script is installed as $prefix/bin. We just need $prefix to continue.
    #pwd=os.getcwd() # why????
    #dir_install=os.path.normpath(join(pwd,scriptdir)) # why????
    dir_install=scriptdir # same, except maybe if scriptdir is relative...

    if os.path.basename(dir_install) == 'bin':
        prefix=os.path.dirname(dir_install)
        assert isinstance(prefix, unicode)
        libpath=join(prefix, lib_shared)
        libpath_compiled = join(prefix, lib_compiled)
        sys.path.insert(0, libpath)
        sys.path.insert(0, libpath_compiled)
        localepath = join(prefix, 'share/locale')
    elif sys.platform == 'win32':
        prefix=None
        # this is py2exe point of view, all executables in root of installdir
        # all path must be normalized to absolute path
        libpath = os.path.abspath(os.path.dirname(os.path.realpath(arg0)))
        sys.path.insert(0, libpath)
        localepath = join(libpath,'share/locale')
    else:
        # we are not installed
        prefix = None
        libpath = u'.'
        localepath = 'po'

    assert isinstance(libpath, unicode)

    try: # just for a nice error message
        from lib import mypaintlib
    except ImportError:
        print
        print "We are not correctly installed or compiled!"
        print 'script: "%s"' % arg0
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

    homepath =  os.path.expanduser(u'~')
    if homepath == '~':
        confpath = join(prefix, 'UserData')
    else:
        confpath = join(homepath, '.mypaint/')

    assert isinstance(datapath, unicode)
    assert isinstance(confpath, unicode)
    return datapath, confpath, localepath

def psyco_opt():
    # This helps on slow PCs where the python overhead dominates.
    # (30% higher framerate measured on 533MHz CPU; startup slowdown below 20%)
    # Note: python -O -O does not help.
    if os.name in ('nt', 'ce'):
        # reported to be broken on Windows
        return
    try:
        import psyco
        if sys.platform == 'win32':
            if psyco.hexversion >= 0x020000f0 :
                psyco.full()
                print 'Psyco being used'
            else:
                print "Need at least psyco 2.0 to run"
        else:
            psyco.full()
            print 'Psyco being used'
    except ImportError:
        pass

if __name__ == '__main__':
    psyco_opt()

    datapath, confpath, localepath = get_paths()

    # must be done before importing any translated python modules
    # (to get global strings translated, especially brushsettings.py)
    import gettext
    if sys.platform == 'win32':
        import locale
        os.environ['LANG'] = locale.getdefaultlocale()[0]
    gettext.bindtextdomain("mypaint", localepath)
    gettext.textdomain("mypaint")

    from gui import main
    main.main(datapath, confpath)
