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
    """Burn versions into the generated Python target."""
    # Make sure we run the python version that we built the extension
    # modules for:
    script_header =  '#!/usr/bin/env ' + env['python_binary'] + '\n'
    script_header += 5*'#\n'
    script_header += '# DO NOT EDIT - edit %s instead\n' % source[0]
    script_header += 5*'#\n'
    script_header += "\n\n"
    script_header += "# Auto-generated version info\n"
    sys.path.append(".")
    import lib.meta
    script_header += lib.meta._get_release_info_script(gitprefix="git")
    script_header += "\n\n"
    script_header += open(str(source[0])).read()
    f = open(str(target[0]), 'w')
    f.write(script_header)
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
