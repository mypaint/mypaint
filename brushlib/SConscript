Import('env', 'python', 'install_perms')

# NOTE: We use a copy of the environment, to be able to both inherit common options,
# and also add our own specifics ones without affecting the other builds
brushlib_env = env.Clone()
env = brushlib_env

env.Append(CPPPATH='./')

env.ParseConfig('pkg-config --cflags --libs glib-2.0')

env.Execute(python + ' generate.py')
env.Clean('.', 'brushsettings.h')
env.Clean('.', Glob('*.pyc'))

module = env.SharedLibrary('../mypaint-brushlib', Glob("*.c"))

install_perms(env, '$prefix/lib/mypaint', module)
install_perms(env, '$prefix/include/mypaint', Glob("brushlib/mypaint-*.h"))

Return('module')
