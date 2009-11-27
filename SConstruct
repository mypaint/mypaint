import os, sys
from os.path import join, basename

EnsureSConsVersion(1, 0)

python = 'python%d.%d' % (sys.version_info[0], sys.version_info[1])
print 'Building for', python

try: 
    import numpy
except ImportError:
    print 'You need to have numpy installed.'
    print
    raise

SConsignFile() # no .scsonsign into $PREFIX please

# Does option parsing really screw up the win32 build? if no, remove comment
#if sys.platform == "win32":
#    env = Environment(ENV=os.environ)
#else:

opts = Variables()
opts.Add(PathVariable('prefix', 'autotools-style installation prefix', '/usr/local', validator=PathVariable.PathIsDirCreate))
opts.Add(BoolVariable('debug', 'enable HEAVY_DEBUG and disable optimizations', False))
env = Environment(ENV=os.environ, options=opts)
if sys.platform == "win32":
    env.Prepend(tools=['mingw'])
opts.Update(env)

env.ParseConfig('pkg-config --cflags --libs glib-2.0')

env.Append(CXXFLAGS=' -Wall -Wno-sign-compare -Wno-write-strings')

# Get the numpy include path (for numpy/arrayobject.h).
numpy_path = numpy.get_include()
env.Append(CPPPATH=numpy_path)


if sys.platform == "win32":
    env.ParseConfig('pkg-config --cflags --libs python25') # These two '.pc' files you probably have to make for yourself.
    env.ParseConfig('pkg-config --cflags --libs numpy')    # Place them among the other '.pc' files ( where the 'glib-2.0.pc' is located .. probably )
else:
    # some distros use python2.5-config, others python-config2.5
    try:
        env.ParseConfig(python + '-config --cflags --ldflags')
    except OSError:
        print 'going to try python-config instead'
        env.ParseConfig('python-config --cflags --ldflags')

if env.get('CPPDEFINES'):
    # make sure assertions are enabled
    env['CPPDEFINES'].remove('NDEBUG')

if env['debug']:
    env.Append(CPPDEFINES='HEAVY_DEBUG')
    env.Append(CCFLAGS='-O0', LINKFLAGS='-O0')

Export('env')
module = SConscript('lib/SConscript')
SConscript('brushlib/SConscript')
languages = SConscript('po/SConscript')

# Build mypaint.exe for running on windows
if sys.platform == "win32":
    env2 = Environment(tools=['mingw'], ENV=os.environ)
    env2.ParseConfig('pkg-config --cflags --libs python25')
    env2.Program('mypaint', ['mypaint_exe.c'])

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

env.Alias('install', env['prefix'])
def install(dst, pattern):
    env.Install(join(env['prefix'], dst), Glob(pattern))
install('bin', 'mypaint')
install('share/mypaint/brushes', 'brushes/*')
install('share/mypaint/backgrounds', 'backgrounds/*')
install('share/mypaint/pixmaps', 'pixmaps/*')

install('share', 'desktop/icons')
install('share/applications', 'desktop/mypaint.desktop')

# location for achitecture-dependent modules
env.Install(join(env['prefix'], 'lib/mypaint'), module)
install('share/mypaint/lib', 'lib/*.py')
install('share/mypaint/gui', 'gui/*.py')
install('share/mypaint/gui', 'gui/menu.xml')
install('share/mypaint/brushlib', 'brushlib/*.py')

# translations
for lang in languages:
    install('share/locale/%s/LC_MESSAGES' % lang, 'po/%s/LC_MESSAGES/mypaint.mo' % lang)
