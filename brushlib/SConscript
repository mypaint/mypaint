Import('env', 'python', 'install_perms')

# NOTE: We use a copy of the environment, to be able to both inherit common options,
# and also add our own specifics ones without affecting the other builds
top_env = env
env = env.Clone()
gegl_env = env.Clone()

env.Append(CPPPATH='./')

env.ParseConfig('pkg-config --cflags --libs glib-2.0')

env.Execute(python + ' generate.py') # TODO: make a proper build rule
env.Clean('.', 'mypaint-brush-settings-gen.h')
env.Clean('.', Glob('*.pyc'))

brushlib = env.SharedLibrary('../mypaint-brushlib', Glob("*.c"))

install_perms(env, '$prefix/lib/mypaint', brushlib)
install_perms(env, '$prefix/include/mypaint', Glob("./mypaint-*.h"))

install_perms(env, "$prefix/share/mypaint/brushlib", Glob("./*.py"))
install_perms(env, "$prefix/share/mypaint/brushlib", "./brushsettings.json")

# Optional: GEGL library
if env['enable_gegl']:
    gegl_env.ParseConfig('pkg-config --cflags --libs gegl-0.2')

    gegl_env.Append(LIBS="mypaint-brushlib")
    gegl_env.Append(LIBPATH="../")
    gegl_env.Append(CPPPATH='../brushlib/')

    brushlib_gegl = gegl_env.SharedLibrary('../mypaint-brushlib-gegl', Glob("./gegl/*.c"))
    install_perms(env, '$prefix/lib/mypaint', brushlib_gegl)
    install_perms(env, '$prefix/include/mypaint', Glob("./gegl/mypaint-gegl-*.h"))

Return('brushlib')
