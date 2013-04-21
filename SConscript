import os, sys
import time
from os.path import join, basename

Import('env', 'install_perms', 'install_tree')

# Clone the environment to not affect the common one
env = env.Clone()

mypaintlib = SConscript('lib/SConscript')
languages = SConscript('po/SConscript')

try:
    new_umask = 022
    old_umask = os.umask(new_umask)
    print "set umask to 0%03o (was 0%03o)" % (new_umask, old_umask)
except OSError:
    # Systems like Win32...
    pass

def burn_versions(target, source, env):
    # Burn versions into the generated Python target.
    # Make sure we run the python version that we built the extension
    # modules for:
    s =  '#!/usr/bin/env ' + env['python_binary'] + '\n'
    s += 5*'#\n'
    s += '# DO NOT EDIT - edit %s instead\n' % source[0]
    s += 5*'#\n'
    # Also burn in the last git revision number
    if os.path.isdir(".git"):
        git_in, git_out = os.popen2("git rev-parse --short HEAD")
        git_in.close()
        git_rev = str(git_out.read()).strip()
        git_out.close()
        s += "_MYPAINT_BUILD_GIT_REVISION = %r\n" % (git_rev,)
    else:
        s += "_MYPAINT_BUILD_GIT_REVISION = ''\n"
    # And a timestamp.
    now_utc = time.gmtime()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", now_utc)
    s += "_MYPAINT_BUILD_TIME_ISO = %r\n" % (timestamp,)
    s += "_MYPAINT_BUILD_GMTIME_TUPLE = %r\n" % (tuple(now_utc),)
    s += "\n\n"
    s += open(str(source[0])).read()
    f = open(str(target[0]), 'w')
    f.write(s)
    f.close()


## Build-time customization

# User-facing executable Python code
# MyPaint app
env.Command('mypaint', 'mypaint.py', [burn_versions, Chmod('$TARGET', 0755)])
AlwaysBuild('mypaint') # especially if the "python_binary" option was changed

# Thumbnailer script
env.Command('desktop/mypaint-ora-thumbnailer', 'desktop/mypaint-ora-thumbnailer.py', [burn_versions, Chmod('$TARGET', 0755)])
AlwaysBuild('desktop/mypaint-ora-thumbnailer')


## Additional cleanup

env.Clean('.', Glob('*.pyc'))
env.Clean('.', Glob('gui/*.pyc'))
env.Clean('.', Glob('gui/colors/*.pyc'))
env.Clean('.', Glob('lib/*.pyc'))


## Installation

# Painting resources
install_tree(env, '$prefix/share/mypaint', 'backgrounds')
install_tree(env, '$prefix/share/mypaint', 'pixmaps')
install_tree(env, '$prefix/share/mypaint', 'palettes')

# Desktop resources and themeable internal icons
install_tree(env, '$prefix/share', 'desktop/icons')
install_perms(env, '$prefix/share/applications', 'desktop/mypaint.desktop')
install_perms(env, '$prefix/bin', 'desktop/mypaint-ora-thumbnailer', perms=0755)
install_perms(env, '$prefix/share/thumbnailers', 'desktop/mypaint-ora.thumbnailer')

# location for achitecture-dependent modules
install_perms(env, '$prefix/lib/mypaint', mypaintlib)

# Program and supporting UI XML
install_perms(env, '$prefix/bin', 'mypaint', perms=0755)
install_perms(env, '$prefix/share/mypaint/gui', Glob('gui/*.xml'))
install_perms(env, '$prefix/share/mypaint/gui', Glob('gui/*.glade'))
install_perms(env, "$prefix/share/mypaint/lib",      Glob("lib/*.py"))
install_perms(env, "$prefix/share/mypaint/gui",      Glob("gui/*.py"))
install_perms(env, "$prefix/share/mypaint/gui/colors", Glob("gui/colors/*.py"))


Return('mypaintlib')

# vim:syntax=python
