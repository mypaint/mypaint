import os, sys
import numpy
SConsignFile() # no .scsonsign into $PREFIX please

if sys.platform == "win32":
	env = Environment(ENV=os.environ)
else:
	opts = Options('options.cache', ARGUMENTS)
	opts.Add(PathOption('PREFIX', 'Directory to install under', '/usr/local'))
	env = Environment(ENV=os.environ, options=opts)
	opts.Update(env)
	opts.Save('options.cache', env)

env.ParseConfig('pkg-config --cflags --libs glib-2.0')

env.Append(CXXFLAGS=' -Wall -Wno-sign-compare -Wno-write-strings')
#env.Append(CXXFLAGS=' -ggdb')
#env.Append(CXXFLAGS=' -O0', LINKFLAGS=' -O0')
#env.Append(CXXFLAGS=' -O3', LINKFLAGS=' -O3')
#env.Append(CXXFLAGS=' -pg', LINKFLAGS=' -pg')

# Get the numpy include path (for numpy/arrayobject.h).
numpy_path = numpy.get_include()
env.Append(CPPPATH=numpy_path)


if sys.platform == "win32":
	env.ParseConfig('pkg-config --cflags --libs python25') # These two '.pc' files you probably have to make for yourself.
	env.ParseConfig('pkg-config --cflags --libs numpy')    # Place them among the other '.pc' files ( where the 'glib-2.0.pc' is located .. probably )
else:
	env.ParseConfig('python-config --cflags --ldflags')

if env.get('CPPDEFINES'):
	env['CPPDEFINES'].remove('NDEBUG')

SConscript('lib/SConscript', 'env')
SConscript('brushlib/SConscript', 'env')

# Build mypaint.exe for running on windows
if sys.platform == "win32":
	env2 = Environment(ENV=os.environ)
	env2.ParseConfig('pkg-config --cflags --libs python25')
	env2.Program('mypaint', ['mypaint_exe.c'])

