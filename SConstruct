import os, sys
from os.path import join, basename
from SCons.Script.SConscript import SConsEnvironment
import SCons.Util

EnsureSConsVersion(2, 1)

default_python_binary = 'python%d.%d' % (sys.version_info[0], sys.version_info[1])
default_python_config = 'python%d.%d-config' % (sys.version_info[0], sys.version_info[1])

if sys.platform == "win32":
    # Usually no versioned binaries on native Windows.
    default_python_binary = 'python'
    default_python_config = 'python-config'
elif sys.platform == "msys" and os.environ.get("MSYSTEM") != "MSYS":
    # Building from MINGW32 or MINGW64 shell using MSYS2 python+scons.
    # Defaults above will work fine.
    pass
elif os.path.exists('/etc/gentoo-release'):
     print('Gentoo: /etc/gentoo-release exists. Must be on a Gentoo based system.')
     default_python_config = 'python-config-%d.%d'  % (sys.version_info[0],sys.version_info[1])

SConsignFile() # no .scsonsign into $PREFIX please

default_prefix = '/usr/local/'
default_openmp = True

if sys.platform == "darwin":
    default_openmp = False

def isabs(key, dirname, env):
    assert os.path.isabs(dirname), "%r must have absolute path syntax" % (key,)

opts = Variables()
opts.Add(
    'prefix', 'autotools-style installation prefix',
    default=default_prefix,
    validator=isabs,
)
opts.Add(BoolVariable('debug', 'enable HEAVY_DEBUG and disable optimizations', False))
opts.Add(BoolVariable('enable_profiling', 'enable debug symbols for profiling purposes', True))
opts.Add(BoolVariable('enable_gegl', 'enable GEGL based code in build', False))
opts.Add(BoolVariable('enable_introspection', 'enable GObject introspection support', False))
opts.Add(BoolVariable('use_glib', 'enable glib (forced on by introspection)', False))
opts.Add(BoolVariable('enable_docs', 'enable documentation build', False))
opts.Add(BoolVariable('enable_gperftools', 'enable gperftools in build, for profiling', False))
opts.Add(BoolVariable('enable_openmp', 'enable OpenMP for multithreaded processing', default_openmp))
opts.Add('python_binary', 'python executable to build for', default_python_binary)
opts.Add('python_config', 'python-config to use', default_python_config)
opts.Add('numpy_include', 'override include dir for NumPy (where numpy/arrayobject.h lives)', None)

tools = ['default', 'textfile']

if sys.platform == "msys" and os.environ.get("MSYSTEM") != "MSYS":
    # Building from MINGW32 or MINGW64 shell using MSYS2 python+scons.
    # MSYS2 ship their own SCons.Tool.mingw_w64
    # https://github.com/Alexpux/MSYS2-packages/blob/master/scons
    tools.append("mingw_w64")
elif sys.platform == "win32":
    # Assume MinGW.org. This is untested outside of MSYS2.
    # You're welcome to try MSVC instead: if it works, please submit a
    # patch with a suitable options switch.
    tools.append("mingw")

env = Environment(ENV=os.environ, options=opts, tools=tools)

Help(opts.GenerateHelpText(env))

if not env.GetOption("help"):
    print('building for %r (use scons python_binary=xxx to change)'
          % env['python_binary'])
    print('using %r (use scons python_config=xxx to change)'
          % env['python_config'])

# Respect some standard build environment stuff
# See http://cgit.freedesktop.org/mesa/mesa/tree/scons/gallium.py
# See https://wiki.gentoo.org/wiki/SCons#Missing_CC.2C_CFLAGS.2C_LDFLAGS
if 'CC' in os.environ:
   env['CC'] = os.environ['CC']
if 'CFLAGS' in os.environ:
   env['CCFLAGS'] += SCons.Util.CLVar(os.environ['CFLAGS'])
if 'CXX' in os.environ:
   env['CXX'] = os.environ['CXX']
if 'CXXFLAGS' in os.environ:
   env['CXXFLAGS'] += SCons.Util.CLVar(os.environ['CXXFLAGS'])
if 'CPPFLAGS' in os.environ:
   env['CCFLAGS'] += SCons.Util.CLVar(os.environ['CPPFLAGS'])
   env['CXXFLAGS'] += SCons.Util.CLVar(os.environ['CPPFLAGS'])
if 'LDFLAGS' in os.environ:
    # LDFLAGS is omitted in SHLINKFLAGS, which is derived from LINKFLAGS
   env['LINKFLAGS'] += SCons.Util.CLVar(os.environ['LDFLAGS'])
if "$CCFLAGS" in env['CXXCOM']:
   env['CXXCOM'] = env['CXXCOM'].replace("$CCFLAGS","")

opts.Update(env)

env.Append(CXXFLAGS=' -Wall -Wno-sign-compare -Wno-write-strings')
env.Append(CCFLAGS='-Wall')
env.Append(CFLAGS='-std=c99')

# Define strdup() in string.h under glibc >= 2.10 (POSIX.1-2008)
env.Append(CFLAGS='-D_POSIX_C_SOURCE=200809L')

if env.get('CPPDEFINES'):
    # make sure assertions are enabled
    env['CPPDEFINES'].remove('NDEBUG')

if env['debug']:
    env.Append(CPPDEFINES='HEAVY_DEBUG')
    env.Append(CCFLAGS='-O0', LINKFLAGS='-O0')
else:
    # Overridable defaults
    env.Prepend(CCFLAGS='-O3', LINKFLAGS='-O3')

if env['enable_profiling'] or env['debug']:
    env.Append(CCFLAGS='-g')

#env.Append(CCFLAGS='-fno-inline', LINKFLAGS='-fno-inline')

if sys.platform == "darwin":
    pass
elif sys.platform == "win32":
    pass
elif sys.platform == "msys" and os.environ.get("MSYSTEM") != "MSYS":
    # Building from MINGW32 or MINGW64 shell using MSYS2 python+scons.
    pass
else: # Assume Linux.
    # Look up libraries dependencies relative to the library.
    env.Append(LINKFLAGS='-Wl,-z,origin')
    env.Append(RPATH = env.Literal(os.path.join('\\$$ORIGIN')))

# remove libraries produced by earlier versions, which are actually
# being used if they keep lying around, leading to mysterious bugs
env.Execute(Delete([
    'libmypaint-tests.so',
    'libmypaint-tests.so',
    'libmypaint.so',
    'libmypaintlib.so',
    'libmypaint.a',
    'libmypaint-tests.a',
    'lib/_mypaintlib.so',
    ]))


set_dir_postaction = {}
def install_perms(env, target, sources, perms=0o644, dirperms=0o755):
    """As a normal env.Install, but with Chmod postactions.

    The `target` parameter must be a string starting with ``$prefix``.
    The permissions in `perms` will be assigned to each file which was
    installed from `sources` in a post-install action.

    The `dirperms` permissions will be assigned to each created
    directory component which does not exist (at the time of calling
    this function). Each set of permission bits is assigned in its own
    postaction.

    """
    assert target.startswith('$prefix')
    install_targs = env.Install(target, sources)

    # Set file permissions, and defer directory permission setting
    for targ in install_targs:
        env.AddPostAction(targ, Chmod(targ, perms))
        d = os.path.dirname(os.path.normpath(targ.get_abspath()))
        d_prev = None
        while d != d_prev and not os.path.exists(d):
            if not d in set_dir_postaction:
                env.AddPostAction(targ, Chmod(d, dirperms))
                set_dir_postaction[d] = True
            d_prev = d
            d = os.path.dirname(d)

    # Return like Install()
    return install_targs


def install_tree(env, dest, path, perms=0o644, dirperms=0o755):
    assert os.path.isdir(path)
    target_root = join(dest, os.path.basename(path))
    for dirpath, dirnames, filenames in os.walk(path):
        reltarg = os.path.relpath(dirpath, path)
        target_dir = join(target_root, reltarg)
        target_dir = os.path.normpath(target_dir)
        filepaths = [join(dirpath, basename) for basename in filenames]
        install_perms(env, target_dir, filepaths, perms=perms, dirperms=dirperms)

def createStaticPicLibraryBuilder(env):
    """This is a utility function that creates the StaticExtLibrary Builder in
    an Environment if it is not there already.

    If it is already there, we return the existing one."""
    import SCons.Action

    try:
        static_extlib = env['BUILDERS']['StaticPicLibrary']
    except KeyError:
        action_list = [ SCons.Action.Action("$ARCOM", "$ARCOMSTR") ]
        if env.Detect('ranlib'):
            ranlib_action = SCons.Action.Action("$RANLIBCOM", "$RANLIBCOMSTR")
            action_list.append(ranlib_action)

    static_extlib = SCons.Builder.Builder(action = action_list,
                                          emitter = '$LIBEMITTER',
                                          prefix = '$LIBPREFIX',
                                          suffix = '$LIBSUFFIX',
                                          src_suffix = '$OBJSUFFIX',
                                          src_builder = 'SharedObject')

    env['BUILDERS']['StaticPicLibrary'] = static_extlib
    return static_extlib

createStaticPicLibraryBuilder(env)

# These hierarchies belong entirely to us, so unmake if asked.
env.Clean('$prefix', '$prefix/lib/mypaint')
env.Clean('$prefix', '$prefix/share/mypaint')

# Convenience alias for installing to $prefix
env.Alias('install', '$prefix')

Export('env', 'install_tree', 'install_perms')

# App and its library
SConscript('./SConscript')

# vim:syntax=python
