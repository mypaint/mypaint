import os, sys
from os.path import join, basename

EnsureSConsVersion(1, 0)

# FIXME: sometimes it would be good to build for a different python
# version than the one running scons. (But how to find all paths then?)
python = 'python%d.%d' % (sys.version_info[0], sys.version_info[1])
print 'Building for', python

if sys.platform == "win32":
    python = 'python' # usually no versioned binaries on Windows

try: 
    import numpy
except ImportError:
    print 'You need to have numpy installed.'
    print
    raise

SConsignFile() # no .scsonsign into $PREFIX please

if sys.platform == "darwin":
    default_prefix = '/opt/local/'
else:
    default_prefix = '/usr/local/'

opts = Variables()
opts.Add(PathVariable('prefix', 'autotools-style installation prefix', default_prefix, validator=PathVariable.PathIsDirCreate))

opts.Add(BoolVariable('debug', 'enable HEAVY_DEBUG and disable optimizations', False))
env = Environment(ENV=os.environ, options=opts)
if sys.platform == "win32":
    # remove this mingw if trying VisualStudio
    env = Environment(tools=['mingw'], ENV=os.environ, options=opts)
opts.Update(env)

env.ParseConfig('pkg-config --cflags --libs glib-2.0')
env.ParseConfig('pkg-config --cflags --libs libpng')

env.Append(CXXFLAGS=' -Wall -Wno-sign-compare -Wno-write-strings')

# Get the numpy include path (for numpy/arrayobject.h).
numpy_path = numpy.get_include()
env.Append(CPPPATH=numpy_path)


if sys.platform == "win32":
    # official python shipped with no pc file on windows so get from current python
    from distutils import sysconfig
    pre,inc = sysconfig.get_config_vars('exec_prefix', 'INCLUDEPY')
    env.Append(CPPPATH=inc, LIBPATH=pre+'\libs', LIBS='python'+sys.version[0]+sys.version[2])
elif sys.platform == "darwin":
    env.ParseConfig('python-config --cflags')
    env.ParseConfig('python-config --ldflags')
else:
    # some distros use python2.5-config, others python-config2.5
    try:
        env.ParseConfig(python + '-config --cflags')
        env.ParseConfig(python + '-config --ldflags')
    except OSError:
        print 'going to try python-config instead'
        env.ParseConfig('python-config --ldflags')
        env.ParseConfig('python-config --cflags')

if env.get('CPPDEFINES'):
    # make sure assertions are enabled
    env['CPPDEFINES'].remove('NDEBUG')

if env['debug']:
    env.Append(CPPDEFINES='HEAVY_DEBUG')
    env.Append(CCFLAGS='-O0', LINKFLAGS='-O0')

Export('env', 'python')
module = SConscript('lib/SConscript')
SConscript('brushlib/SConscript')
languages = SConscript('po/SConscript')

# Build mypaint.exe for running on windows
if sys.platform == "win32":
    env2 = Environment(tools=['mingw'], ENV=os.environ)
    env2.Append(CPPPATH='-I'+inc, LIBPATH=pre+'\libs', LIBS='python'+sys.version[0]+sys.version[2])
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

env.Clean('.', Glob('*.pyc'))
env.Clean('.', Glob('gui/*.pyc'))
env.Clean('.', Glob('lib/*.pyc'))

env.Alias('install', env['prefix'])
def install(dst, pattern):
    files = Glob(pattern)
    assert files, "Glob expression did not match any files"
    env.Install(join(env['prefix'], dst), files)
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
