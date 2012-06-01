import os, sys
from os.path import join, basename
from SCons.Script.SConscript import SConsEnvironment

EnsureSConsVersion(1, 0)

# FIXME: sometimes it would be good to build for a different python
# version than the one running scons. (But how to find all paths then?)
python = 'python%d.%d' % (sys.version_info[0], sys.version_info[1])
print 'Building for', python

if sys.platform == "win32":
    python = 'python' # usually no versioned binaries on Windows

SConsignFile() # no .scsonsign into $PREFIX please

if sys.platform == "darwin":
    default_prefix = '/opt/local/'
else:
    default_prefix = '/usr/local/'

opts = Variables()
opts.Add(PathVariable('prefix', 'autotools-style installation prefix', default_prefix, validator=PathVariable.PathIsDirCreate))
opts.Add(BoolVariable('debug', 'enable HEAVY_DEBUG and disable optimizations', False))
opts.Add(BoolVariable('brushlib_only', 'only build and install brushlib/', False))
opts.Add(BoolVariable('enable_gegl', 'enable GEGL based code in build', False))

tools = ['default', 'textfile']

env = Environment(ENV=os.environ, options=opts, tools=tools)
if sys.platform == "win32":
    # remove this mingw if trying VisualStudio
    env = Environment(tools=tools + ['mingw'], ENV=os.environ, options=opts)
opts.Update(env)

env.Append(CXXFLAGS=' -Wall -Wno-sign-compare -Wno-write-strings')
env.Append(CCFLAGS='-Wall')
env.Append(CFLAGS='-std=c99')

if env.get('CPPDEFINES'):
    # make sure assertions are enabled
    env['CPPDEFINES'].remove('NDEBUG')

if env['debug']:
    env.Append(CPPDEFINES='HEAVY_DEBUG')
    env.Append(CCFLAGS='-O0', LINKFLAGS='-O0')

#env.Append(CCFLAGS='-fno-inline', LINKFLAGS='-fno-inline')

# Look up libraries dependencies relative to the library
env.Append(LINKFLAGS = Split('-z origin'))
env.Append(RPATH = env.Literal(os.path.join('\\$$ORIGIN')))

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



# Common
install_tree(env, '$prefix/share/mypaint', 'brushes')

# These hierarchies belong entirely to us, so unmake if asked.
env.Clean('$prefix', '$prefix/lib/mypaint')
env.Clean('$prefix', '$prefix/share/mypaint')

# Convenience alias for installing to $prefix
env.Alias('install', '$prefix')



Export('env', 'python', 'install_tree', 'install_perms')

brushlib = SConscript('./brushlib/SConscript')

if not env['brushlib_only']:
    application = SConscript('./SConscript')
