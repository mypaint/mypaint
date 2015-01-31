import os, sys
from os.path import join, basename
from SCons.Script.SConscript import SConsEnvironment
import SCons.Util

EnsureSConsVersion(2, 1)

default_python_binary = 'python%d.%d' % (sys.version_info[0], sys.version_info[1])
default_python_config = 'python%d.%d-config' % (sys.version_info[0], sys.version_info[1])

if sys.platform == "win32":
    # usually no versioned binaries on Windows
    default_python_binary = 'python'
    default_python_config = 'python-config'

if os.path.exists('/etc/gentoo-release'):
     print 'Gentoo: /etc/gentoo-release exists. Must be on a Gentoo based system.'
     default_python_config = 'python-config-%d.%d'  % (sys.version_info[0],sys.version_info[1])

SConsignFile() # no .scsonsign into $PREFIX please

default_prefix = '/usr/local/'
default_openmp = True

if sys.platform == "darwin":
    default_openmp = False

opts = Variables()
opts.Add(PathVariable('prefix', 'autotools-style installation prefix', default_prefix, validator=PathVariable.PathIsDirCreate))
opts.Add(BoolVariable('debug', 'enable HEAVY_DEBUG and disable optimizations', False))
opts.Add(BoolVariable('enable_profiling', 'enable debug symbols for profiling purposes', True))
opts.Add(BoolVariable('enable_gegl', 'enable GEGL based code in build', False))
opts.Add(BoolVariable('enable_introspection', 'enable GObject introspection support', False))
opts.Add(BoolVariable('use_sharedlib', 'build a shared library instead of a static library (forced on by introspection)', False))
opts.Add(BoolVariable('use_glib', 'enable glib (forced on by introspection)', False))
opts.Add(BoolVariable('enable_docs', 'enable documentation build', False))
opts.Add(BoolVariable('enable_gperftools', 'enable gperftools in build, for profiling', False))
opts.Add(BoolVariable('enable_openmp', 'enable OpenMP for multithreaded processing', default_openmp))
opts.Add('python_binary', 'python executable to build for', default_python_binary)
opts.Add('python_config', 'python-config to use', default_python_config)

tools = ['default', 'textfile']

env = Environment(ENV=os.environ, options=opts, tools=tools)

Help(opts.GenerateHelpText(env))

if not env.GetOption("help"):
    print('building for %r (use scons python_binary=xxx to change)'
          % env['python_binary'])
    print('using %r (use scons python_config=xxx to change)'
          % env['python_config'])

if sys.platform == "win32":
    # remove this mingw if trying VisualStudio
    env = Environment(tools=tools + ['mingw'], ENV=os.environ, options=opts)

# Respect some standard build environment stuff
# See http://cgit.freedesktop.org/mesa/mesa/tree/scons/gallium.py
# See https://wiki.gentoo.org/wiki/SCons#Missing_CC.2C_CFLAGS.2C_LDFLAGS
if os.environ.has_key('CC'):
   env['CC'] = os.environ['CC']
if os.environ.has_key('CFLAGS'):
   env['CCFLAGS'] += SCons.Util.CLVar(os.environ['CFLAGS'])
if os.environ.has_key('CXX'):
   env['CXX'] = os.environ['CXX']
if os.environ.has_key('CXXFLAGS'):
   env['CXXFLAGS'] += SCons.Util.CLVar(os.environ['CXXFLAGS'])
if os.environ.has_key('CPPFLAGS'):
   env['CCFLAGS'] += SCons.Util.CLVar(os.environ['CPPFLAGS'])
   env['CXXFLAGS'] += SCons.Util.CLVar(os.environ['CPPFLAGS'])
if os.environ.has_key('LDFLAGS'):
    # LDFLAGS is omitted in SHLINKFLAGS, which is derived from LINKFLAGS
   env['LINKFLAGS'] += SCons.Util.CLVar(os.environ['LDFLAGS'])
if "$CCFLAGS" in env['CXXCOM']:
   env['CXXCOM'] = env['CXXCOM'].replace("$CCFLAGS","")

opts.Update(env)

env.Append(CXXFLAGS=' -Wall -Wno-sign-compare -Wno-write-strings')
env.Append(CCFLAGS='-Wall')
env.Append(CFLAGS='-std=c99')

env['GEGL_VERSION'] = 0.3

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

# Look up libraries dependencies relative to the library
if sys.platform != "darwin" and sys.platform != "win32":
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
def install_perms(env, target, sources, perms=0644, dirperms=0755):
    """As a normal env.Install, but with Chmod postactions.

    The `target` parameter must be a string which starts with ``$prefix``.
    Unless this is a sandbox install, the permission bits `dirperms` will be
    set on every directory back to ``$prefix``, but not including it. `perms`
    will always be set on each installed file from `sources`.
    """
    assert target.startswith('$prefix')
    install_targs = env.Install(target, sources)
    sandboxed = False
    final_prefix = os.path.normpath(env["prefix"])

    # Set file permissions.
    for targ in install_targs:
        env.AddPostAction(targ, Chmod(targ, perms))
        targ_path = os.path.normpath(targ.get_path())
        if not targ_path.startswith(final_prefix):
            sandboxed = True

    if not sandboxed:
        # Set permissions on superdirs, back to $prefix (but not including it)
        # Not sure if this is necessary with the umask forcing. It might help
        # fix some broken installs.
        for file_targ in install_targs:
            d = os.path.normpath(target)
            d_prev = None
            while d != d_prev and d != '$prefix':
                d_prev = d
                if not set_dir_postaction.has_key(d):
                    env.AddPostAction(file_targ, Chmod(d, dirperms))
                    set_dir_postaction[d] = True
                d = os.path.dirname(d)

    return install_targs


def install_tree(env, dest, path, perms=0644, dirperms=0755):
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

# Common
install_tree(env, '$prefix/share/mypaint', 'brushes')

# These hierarchies belong entirely to us, so unmake if asked.
env.Clean('$prefix', '$prefix/lib/mypaint')
env.Clean('$prefix', '$prefix/share/mypaint')

# Convenience alias for installing to $prefix
env.Alias('install', '$prefix')

Export('env', 'install_tree', 'install_perms')

if not env.GetOption("help"):
    print "Enabling i18n for brushlib in full application build"
env['enable_i18n'] = True

# Brushlib
brushlib = SConscript('./brushlib/SConscript')
Export('brushlib')

# App and its library
SConscript('./SConscript')

# vim:syntax=python
