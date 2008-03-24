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
env.ParseConfig('pkg-config --cflags --libs gtk+-2.0')

# code generator
#brushsettings = env.Command('brushsettings.h', ['generate.py', 'brushsettings.py'], './generate.py')
# For the record: I know that scons supports swig. But it doesn't scan for #include in the generated code.
# 
# I have given up. Scons just can't get the dependencies right with those
# code generators. Let's give scons a "normal" c++ project to dependency-scan.
env.Execute('./generate.py')
env.Clean('.', 'brushsettings.h')
env.Execute('swig -o mypaintlib_wrap.cpp -python -c++ mypaintlib.i')
env.Clean('.', 'mypaintlib_wrap.cc')
env.Clean('.', 'mypaintlib.py')

# python extension module
src = 'mypaintlib_wrap.cpp helpers.c mapping.c lfd.c'
module = env.LoadableModule('_mypaintlib', Split(src), SHLIBPREFIX="")


# installation

#env.Install(module, '$PREFIX/lib/mypaint') # location for private compiled extensions
##env.Install(module, '$PREFIX/share/mypaint') # theoretical location for private pure python modules (meld uses $PREFIX/lib/meld)
#env.Install(data, '$PREFIX/share/mypaint')

