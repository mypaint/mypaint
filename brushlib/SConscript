Import('env', 'python', 'install_perms')

import os

def add_gobject_introspection(env, gi_name, version,
                              func_prefix, type_prefix,
                              sources, includepath, library, pkgs):

    pkgs = ' '.join('--pkg=%s' % dep for dep in pkgs)
    library = library[0] # there should be only one Node in the list

    # Strip the library path to get the library name
    libname = os.path.basename(library.get_path())
    libname = os.path.splitext(libname)[0]
    if libname.startswith('lib'):
        libname = libname[3:]

    scanner_cmd = """LD_LIBRARY_PATH=./ g-ir-scanner -o $TARGET --warn-all \
        --namespace=%(gi_name)s --nsversion=%(version)s \
        --identifier-prefix=%(type_prefix)s --symbol-prefix=%(func_prefix)s \
        %(pkgs)s -I%(includepath)s \
        --library=%(libname)s $SOURCES""" % locals()

    gir_file = env.Command("%s.gir" % gi_name, sources, scanner_cmd)
    env.Depends(gir_file, library)
    typelib_file = env.Command("%s.typelib" % gi_name, gir_file,
                           "g-ir-compiler -o $TARGET $SOURCE")

    return (gir_file, typelib_file)

# NOTE: We use a copy of the environment, to be able to both inherit common options,
# and also add our own specifics ones without affecting the other builds
top_env = env
env = env.Clone()
gegl_env = env.Clone()

if env['enable_introspection']:
    env['use_glib'] = True
    print "Enabling glib because of enable_introspection=true"
else:
    env['use_glib'] = False

Export('env')
utils = SConscript('utils/SConscript')
tests = SConscript('tests/SConscript')

if env['enable_docs']:
    doc = SConscript('doc/SConscript')

env.Append(CPPPATH='./')

pkg_info = {}
pkg_info['@LIBNAME@'] = 'mypaint'

if env['use_glib']:
    pkg_info['@REQUIRES@'] = 'glib-2.0'
else:
    pkg_info['@REQUIRES@'] = ''

pkg_info['@DESCRIPTION@'] = 'MyPaint brush engine library'
pkg_info['@PREFIX@'] = env['prefix']
pkg_info['@VERSION@'] = '0.1'
pkg_info['@LIBDIR@'] = os.path.join(env['prefix'], 'lib')
pkg_info['@INCLUDEDIR@'] = os.path.join(env['prefix'], 'include')
pc_file = env.Substfile("libmypaint.pc", "pkgconfig.pc.in", SUBST_DICT=pkg_info)
install_perms(env, '$prefix/lib/pkgconfig', pc_file)

env.Append(LIBS='m')
env.ParseConfig('pkg-config --cflags --libs json')

config_defines = ''
if env['use_glib']:
    env.ParseConfig('pkg-config --cflags --libs gobject-2.0')
    config_defines += '#define MYPAINT_CONFIG_USE_GLIB 1\n'
else:
    config_defines += '#define MYPAINT_CONFIG_USE_GLIB 0\n'

config_file = env.Substfile("mypaint-config.h", "mypaint-config.h.in", SUBST_DICT={'@DEFINES@': config_defines})

env.Execute(python + ' generate.py') # TODO: make a proper build rule
env.Clean('.', 'mypaint-brush-settings-gen.h')
env.Clean('.', Glob('*.pyc'))

brushlib = env.SharedLibrary('../mypaint', Glob("*.c"))

if env['enable_introspection']:
    gir, typelib = add_gobject_introspection(env, "MyPaint", pkg_info["@VERSION@"],
                              "mypaint_", "MyPaint",
                              Glob("*.c") + Glob("mypaint-*.h") + Glob("glib/*.c") + Glob("glib/mypaint-*.h"),
                              './brushlib', brushlib, ['glib-2.0'])

    install_perms(env, '$prefix/share/gir-1.0', gir)
    install_perms(env, '$prefix/lib/girepository-1.0', typelib)

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

    gegl_env.Append(LIBS="mypaint")
    gegl_env.Append(LIBPATH="../")
    gegl_env.Append(CPPPATH='../brushlib/')

    brushlib_gegl = gegl_env.SharedLibrary('../mypaint-gegl', Glob("./gegl/*.c"))
    install_perms(env, '$prefix/lib/', brushlib_gegl)
    install_perms(env, '$prefix/include/libmypaint-gegl', Glob("./gegl/mypaint-gegl-*.h"))

    if gegl_env['enable_introspection']:
        gir, typelib = add_gobject_introspection(gegl_env, "MyPaintGegl", pkg_info["@VERSION@"],
                                  "mypaint_gegl", "MyPaintGegl",
                                  Glob("gegl/*.c") + Glob("gegl/mypaint-gegl-*.h"),
                                  './brushlib/gegl', brushlib_gegl, ['glib-2.0', 'gegl-0.2'])

        install_perms(env, '$prefix/share/gir-1.0', gir)
        install_perms(env, '$prefix/lib/girepository-1.0', typelib)

Return('brushlib')
