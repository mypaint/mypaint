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

try:
    import numpy
except ImportError:
    print 'You need to have numpy installed.'
    print
    raise

SConsignFile() # no .scsonsign into $PREFIX please

if sys.platform == "darwin":
    default_prefix = '/opt/local/'
else:
    default_prefix = '/usr/local/'

opts = Variables()
opts.Add(PathVariable('prefix', 'autotools-style installation prefix', default_prefix, validator=PathVariable.PathIsDirCreate))

opts.Add(BoolVariable('debug', 'enable HEAVY_DEBUG and disable optimizations', False))
env = Environment(ENV=os.environ, options=opts)
if sys.platform == "win32":
    # remove this mingw if trying VisualStudio
    env = Environment(tools=['mingw'], ENV=os.environ, options=opts)
opts.Update(env)

env.ParseConfig('pkg-config --cflags --libs glib-2.0')
env.ParseConfig('pkg-config --cflags --libs libpng')

env.Append(CXXFLAGS=' -Wall -Wno-sign-compare -Wno-write-strings')

# Get the numpy include path (for numpy/arrayobject.h).
numpy_path = numpy.get_include()
env.Append(CPPPATH=numpy_path)


if sys.platform == "win32":
    # official python shipped with no pc file on windows so get from current python
    from distutils import sysconfig
    pre,inc = sysconfig.get_config_vars('exec_prefix', 'INCLUDEPY')
    env.Append(CPPPATH=inc, LIBPATH=pre+'\libs', LIBS='python'+sys.version[0]+sys.version[2])
elif sys.platform == "darwin":
    env.ParseConfig(python + '-config --cflags')
    ldflags = env.backtick(python + '-config --ldflags').split()
    # scons does not seem to parse '-u' correctly
    # put all options after -u in LINKFLAGS
    if '-u' in ldflags:
        idx = ldflags.index('-u')
        env.Append(LINKFLAGS=ldflags[idx:])
        del ldflags[idx:]
    env.MergeFlags(' '.join(ldflags))
else:
    # some distros use python2.5-config, others python-config2.5
    try:
        env.ParseConfig(python + '-config --cflags')
        env.ParseConfig(python + '-config --ldflags')
    except OSError:
        print 'going to try python-config instead'
        env.ParseConfig('python-config --ldflags')
        env.ParseConfig('python-config --cflags')

if env.get('CPPDEFINES'):
    # make sure assertions are enabled
    env['CPPDEFINES'].remove('NDEBUG')

if env['debug']:
    env.Append(CPPDEFINES='HEAVY_DEBUG')
    env.Append(CCFLAGS='-O0', LINKFLAGS='-O0')

#env.Append(CCFLAGS='-fno-inline', LINKFLAGS='-fno-inline')

Export('env', 'python')
module = SConscript('lib/SConscript')
SConscript('brushlib/SConscript')
languages = SConscript('po/SConscript')

def burn_python_version(target, source, env):
    # make sure we run the python version that we built the extension modules for
    s =  '#!/usr/bin/env ' + python + '\n'
    s += 5*'#\n'
    s += '# DO NOT EDIT - edit %s instead\n' % source[0]
    s += 5*'#\n'
    s += open(str(source[0])).read()
    f = open(str(target[0]), 'w')
    f.write(s)
    f.close()

try:
    new_umask = 022
    old_umask = os.umask(new_umask)
    print "set umask to 0%03o (was 0%03o)" % (new_umask, old_umask)
except OSError:
    # Systems like Win32...
    pass

env.Command('mypaint', 'mypaint.py', [burn_python_version, Chmod('$TARGET', 0755)])

env.Clean('.', Glob('*.pyc'))
env.Clean('.', Glob('gui/*.pyc'))
env.Clean('.', Glob('lib/*.pyc'))


set_dir_postaction = {}
def install_perms(target, sources, perms=0644, dirperms=0755):
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


def install_tree(dest, path, perms=0644, dirperms=0755):
    assert os.path.isdir(path)
    target_root = join(dest, os.path.basename(path))
    for dirpath, dirnames, filenames in os.walk(path):
        reltarg = os.path.relpath(dirpath, path)
        target_dir = join(target_root, reltarg)
        target_dir = os.path.normpath(target_dir)
        filepaths = [join(dirpath, basename) for basename in filenames]
        install_perms(target_dir, filepaths, perms=perms, dirperms=dirperms)


# Painting resources
install_tree('$prefix/share/mypaint', 'brushes')
install_tree('$prefix/share/mypaint', 'backgrounds')
install_tree('$prefix/share/mypaint', 'pixmaps')

# Desktop resources and themeable internal icons
install_tree('$prefix/share', 'desktop/icons')
install_perms('$prefix/share/applications', 'desktop/mypaint.desktop')

# location for achitecture-dependent modules
install_perms('$prefix/lib/mypaint', module)

# Program and supporting UI XML
install_perms('$prefix/bin', 'mypaint', perms=0755)
install_perms('$prefix/share/mypaint/gui', Glob('gui/*.xml'))
install_perms("$prefix/share/mypaint/lib",      Glob("lib/*.py"))
install_perms("$prefix/share/mypaint/brushlib", Glob("brushlib/*.py"))
install_perms("$prefix/share/mypaint/gui",      Glob("gui/*.py"))

# translations
for lang in languages:
    install_perms('$prefix/share/locale/%s/LC_MESSAGES' % lang,
                 'po/%s/LC_MESSAGES/mypaint.mo' % lang)

# These hierarchies belong entirely to us, so unmake if asked.
env.Clean('$prefix', '$prefix/lib/mypaint')
env.Clean('$prefix', '$prefix/share/mypaint')

# Convenience alias for installing to $prefix
env.Alias('install', '$prefix')

