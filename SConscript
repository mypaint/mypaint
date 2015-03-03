import os, sys
import time
from os.path import join, basename
from subprocess import check_output

# Pre-fight checks

if not os.path.exists('brushlib/SConscript'):
    print >>sys.stderr, """
----------------------------------------------------------------------

Missing submodule "brushlib"

Please run "git submodule update --init" to create it. See the README
for more information.

----------------------------------------------------------------------
"""
    sys.exit(2)

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
    s += "\n\n"
    if os.path.isfile("release_info"):
        # If we have release information from release.sh, use that
        s += open("release_info").read()
    else:
        # Glean it from the code and git, if we can
        sys.path.append(".")
        from lib.meta import MYPAINT_VERSION as base_version
        formal_version = base_version
        ceremonial_version = base_version
        if "-" in base_version:
            now_utc = time.gmtime()
            timestamp = time.strftime("%Y%m%d", now_utc)
            cmd = ['git', 'rev-parse', '--short', 'HEAD']
            try:
                git_rev = "+git." + str(check_output(cmd)).strip()
            except:
                git_rev = ""
            formal_version = "%s.%s" % (base_version, timestamp)
            ceremonial_version = "%s.%s%s" % (base_version, timestamp, git_rev)
        s += "# Auto-generated version info from SConscript\n"
        s += "MYPAINT_VERSION_BASE = %r\n" % (base_version,)
        s += "MYPAINT_VERSION_FORMAL = %r\n" % (formal_version,)
        s += "MYPAINT_VERSION_CEREMONIAL = %r\n" % (ceremonial_version,)
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
env.Clean('.', Glob('lib/layer/*.pyc'))


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
install_perms(env, '$prefix/share/appdata', 'desktop/mypaint.appdata.xml')

# location for achitecture-dependent modules
install_perms(env, '$prefix/lib/mypaint', mypaintlib)

# Program and supporting UI XML
install_perms(env, '$prefix/bin', 'mypaint', perms=0755)
install_perms(env, '$prefix/share/mypaint/gui', Glob('gui/*.xml'))
install_perms(env, '$prefix/share/mypaint/gui', Glob('gui/*.glade'))
install_perms(env, "$prefix/share/mypaint/lib",      Glob("lib/*.py"))
install_perms(env, "$prefix/share/mypaint/lib/layer", Glob("lib/layer/*.py"))
install_perms(env, "$prefix/share/mypaint/gui",      Glob("gui/*.py"))
install_perms(env, "$prefix/share/mypaint/gui/colors", Glob("gui/colors/*.py"))


Return('mypaintlib')

# vim:syntax=python
