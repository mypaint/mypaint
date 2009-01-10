import os, sys

try: 
    Glob
except:
    # compatibility with SCons version 0.97
    from glob import glob as Glob
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

# Warn user of current set of build options.
if os.path.exists('options.cache'):
    optfile = file('options.cache')
    print "Saved options:", optfile.read().replace("\n", ", ")[:-2]
    optfile.close()
opts = Options('options.cache', ARGUMENTS)
opts.Add(PathOption('prefix', 'autotools-style installation prefix', '/usr/local', PathOption.PathAccept))
env = Environment(ENV=os.environ, options=opts)
opts.Update(env)
opts.Save('options.cache', env)

env.ParseConfig('pkg-config --cflags --libs glib-2.0')

env.Append(CXXFLAGS=' -Wall -Wno-sign-compare -Wno-write-strings')
#env.Append(CXXFLAGS=' -O0', LINKFLAGS=' -O0')
#env.Append(CXXFLAGS=' -O3', LINKFLAGS=' -O3')

# Get the numpy include path (for numpy/arrayobject.h).
numpy_path = numpy.get_include()
env.Append(CPPPATH=numpy_path)


if sys.platform == "win32":
    env.ParseConfig('pkg-config --cflags --libs python25') # These two '.pc' files you probably have to make for yourself.
    env.ParseConfig('pkg-config --cflags --libs numpy')    # Place them among the other '.pc' files ( where the 'glib-2.0.pc' is located .. probably )
else:
    env.ParseConfig('python-config --cflags --ldflags')

if env.get('CPPDEFINES'):
    # make sure assertions are enabled
    env['CPPDEFINES'].remove('NDEBUG')

module = SConscript('lib/SConscript', 'env')
SConscript('brushlib/SConscript', 'env')

# Build mypaint.exe for running on windows
if sys.platform == "win32":
    env2 = Environment(ENV=os.environ)
    env2.ParseConfig('pkg-config --cflags --libs python25')
    env2.Program('mypaint', ['mypaint_exe.c'])


env.Alias('install', env['prefix'])
def install(dst, pattern):
    env.Install(os.path.join(env['prefix'], dst), Glob(pattern))
install('bin', 'mypaint')
install('share/mypaint/brushes', 'brushes/*')
install('share/mypaint/backgrounds', 'backgrounds/*')

#install('share/mypaint/desktop', 'desktop/*')
# scons could recurse with Glob(), but it adds .svn directories when doing so
for dirpath, dirnames, filenames in os.walk('desktop'):
    if '.svn' in dirnames:
        dirnames.remove('.svn')
    env.Install(os.path.join(env['prefix'], 'share/mypaint', dirpath), [os.path.join(dirpath, s) for s in filenames])

# mypaint.desktop goes into /usr/share/applications (debian-only or standard?)
install('share/applications', 'desktop/mypaint.desktop')


#env.Install(module, '$PREFIX/lib/mypaint') # location for private compiled extensions
##env.Install(module, '$PREFIX/share/mypaint') # theoretical location for private pure python modules (meld uses $PREFIX/lib/meld)
# FIXME: and what if the python version changes?
# --> A program which requires a specific version of Python must begin with #!/usr/bin/pythonX.Y (or #!/usr/bin/env pythonX.Y). 
env.Install(os.path.join(env['prefix'], 'lib/mypaint'), module)
install('share/mypaint/lib', 'lib/*.py')
install('share/mypaint/gui', 'gui/*.py')
install('share/mypaint/brushlib', 'brushlib/*.py')

# debian python policy:
# "Private modules are installed in a directory such as /usr/share/packagename or /usr/lib/packagename."
# in general /usr/lib is for architecture-dependent stuff (compiled binaries or modules)
# and        /usr/share for independent


# normal python library:
# .deb on gettdeb.net:  /usr/share/mypaint/python

# autotools before:     /usr/lib/python2.5/site-packages/mypaint/mydrawwidget.so
# .deb on gettdeb.net:  /usr/lib/python2.5/site-packages/mypaint/mydrawwidget.so
# with foresight linux: /usr/lib64/python2.4/site-packages/mypaint/mydrawwidget.so
# (both of them probably just mirror autotools)

