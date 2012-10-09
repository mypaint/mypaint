import os, sys
from os.path import join, basename

Import('env', 'python', 'install_perms', 'install_tree')

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

def burn_python_version(target, source, env):
    # make sure we run the python version that we built the extension modules for
    s =  '#!/usr/bin/env ' + python + '\n'
    s += 5*'#\n'
    s += '# DO NOT EDIT - edit %s instead\n' % source[0]
    s += 5*'#\n'
    s += open(str(source[0])).read()
    f = open(str(target[0]), 'w')
    f.write(s)
    f.close()

env.Command('mypaint', 'mypaint.py', [burn_python_version, Chmod('$TARGET', 0755)])

env.Clean('.', Glob('*.pyc'))
env.Clean('.', Glob('gui/*.pyc'))
env.Clean('.', Glob('gui/colors/*.pyc'))
env.Clean('.', Glob('lib/*.pyc'))

# Painting resources
install_tree(env, '$prefix/share/mypaint', 'backgrounds')
install_tree(env, '$prefix/share/mypaint', 'pixmaps')
install_tree(env, '$prefix/share/mypaint', 'palettes')

# Desktop resources and themeable internal icons
install_tree(env, '$prefix/share', 'desktop/icons')
install_perms(env, '$prefix/share/applications', 'desktop/mypaint.desktop')

# location for achitecture-dependent modules
install_perms(env, '$prefix/lib/mypaint', mypaintlib)

# Program and supporting UI XML
install_perms(env, '$prefix/bin', 'mypaint', perms=0755)
install_perms(env, '$prefix/share/mypaint/gui', Glob('gui/*.xml'))
install_perms(env, "$prefix/share/mypaint/lib",      Glob("lib/*.py"))
install_perms(env, "$prefix/share/mypaint/gui",      Glob("gui/*.py"))
install_perms(env, "$prefix/share/mypaint/gui/colors", Glob("gui/colors/*.py"))

Return('mypaintlib')
