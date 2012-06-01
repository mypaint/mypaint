Import('env', 'python', 'install_perms')

import os

# NOTE: We use a copy of the environment, to be able to both inherit common options,
# and also add our own specifics ones without affecting the other builds
top_env = env
env = env.Clone()
gegl_env = env.Clone()

env.Append(CPPPATH='./')

pkg_info = {}
pkg_info['@LIBNAME@'] = 'mypaint'
pkg_info['@REQUIRES@'] = 'glib-2.0'
pkg_info['@DESCRIPTION@'] = 'MyPaint brush engine library'
pkg_info['@PREFIX@'] = env['prefix']
pkg_info['@VERSION@'] = '0.1'
pkg_info['@LIBDIR@'] = os.path.join(env['prefix'], 'lib')
pkg_info['@INCLUDEDIR@'] = os.path.join(env['prefix'], 'include')
pc_file = env.Substfile("libmypaint.pc", "pkgconfig.pc.in", SUBST_DICT=pkg_info)
install_perms(env, '$prefix/lib/pkgconfig', pc_file)

env.Append(LIBS='m')
env.ParseConfig('pkg-config --cflags --libs glib-2.0')

env.Execute(python + ' generate.py') # TODO: make a proper build rule
env.Clean('.', 'mypaint-brush-settings-gen.h')
env.Clean('.', Glob('*.pyc'))

brushlib = env.SharedLibrary('../mypaint-brushlib', Glob("*.c"))

install_perms(env, '$prefix/lib/', brushlib)
install_perms(env, '$prefix/include/libmypaint', Glob("./mypaint-*.h"))

# FIXME: install to libmypaint
install_perms(env, "$prefix/share/mypaint/brushlib", Glob("./*.py"))
install_perms(env, "$prefix/share/mypaint/brushlib", "./brushsettings.json")

languages = SConscript('po/SConscript')

# Optional: GEGL library
if env['enable_gegl']:
    pkg_info = {}
    pkg_info['@LIBNAME@'] = 'mypaint-gegl'
    pkg_info['@REQUIRES@'] = 'gegl-0.2 libmypaint'
    pkg_info['@DESCRIPTION@'] = 'MyPaint brush engine library, with GEGL integration'
    pkg_info['@VERSION@'] = '0.1'
    pkg_info['@PREFIX@'] = env['prefix']
    pkg_info['@LIBDIR@'] = os.path.join(env['prefix'], 'lib')
    pkg_info['@INCLUDEDIR@'] = os.path.join(env['prefix'], 'include')
    pc_file = gegl_env.Substfile("libmypaint-gegl.pc", "pkgconfig.pc.in", SUBST_DICT=pkg_info)
    install_perms(env, '$prefix/lib/pkgconfig', pc_file)

    gegl_env.ParseConfig('pkg-config --cflags --libs gegl-0.2')

    gegl_env.Append(LIBS="mypaint-brushlib")
    gegl_env.Append(LIBPATH="../")
    gegl_env.Append(CPPPATH='../brushlib/')

    brushlib_gegl = gegl_env.SharedLibrary('../mypaint-brushlib-gegl', Glob("./gegl/*.c"))
    install_perms(env, '$prefix/lib/', brushlib_gegl)
    install_perms(env, '$prefix/include/libmypaint-gegl', Glob("./gegl/mypaint-gegl-*.h"))

Return('brushlib')
