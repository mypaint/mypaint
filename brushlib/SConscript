Import('env', 'python', 'install_perms')

# NOTE: We use a copy of the environment, to be able to both inherit common options,
# and also add our own specifics ones without affecting the other builds
top_env = env
env = env.Clone()

env.Append(CPPPATH='./')

env.ParseConfig('pkg-config --cflags --libs glib-2.0')

env.Execute(python + ' generate.py') # TODO: make a proper build rule
env.Clean('.', 'mypaint-brush-settings-gen.h')
env.Clean('.', Glob('*.pyc'))

module = env.SharedLibrary('../mypaint-brushlib', Glob("*.c"))

install_perms(env, '$prefix/lib/mypaint', module)
install_perms(env, '$prefix/include/mypaint', Glob("./mypaint-*.h"))

install_perms(env, "$prefix/share/mypaint/brushlib", Glob("./*.py"))
install_perms(env, "$prefix/share/mypaint/brushlib", "./brushsettings.json")

Return('module')
