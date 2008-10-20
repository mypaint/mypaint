import os
SConsignFile() # no .scsonsign into $PREFIX please

opts = Options('options.cache', ARGUMENTS)
opts.Add(PathOption('PREFIX', 'Directory to install under', '/usr/local'))
env = Environment(ENV=os.environ, options=opts)
opts.Update(env)
opts.Save('options.cache', env)

env.Append(CXXFLAGS=' -Wall -Wno-sign-compare -Wno-write-strings')
#env.Append(CXXFLAGS=' -ggdb')
#env.Append(CXXFLAGS=' -O3', LINKFLAGS=' -O3')
#env.Append(CXXFLAGS=' -pg', LINKFLAGS=' -pg')

env.ParseConfig('python-config --cflags --ldflags')
env.ParseConfig('pkg-config --cflags --libs glib-2.0')

# enable assertions (python-config defines NDEBUG)
env['CPPDEFINES'].remove('NDEBUG')

SConscript('lib/SConscript', 'env')


