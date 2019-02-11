import os, sys
import time
from os.path import join, basename
from subprocess import check_output

# Pre-fight checks

Import('env', 'install_perms', 'install_tree')

# Clone the environment to not affect the common one
env = env.Clone()

mypaintlib = SConscript('lib/SConscript')
languages = SConscript('po/SConscript')

try:
    new_umask = 0o22
    old_umask = os.umask(new_umask)
    print("set umask to 0%03o (was 0%03o)" % (new_umask, old_umask))
except OSError:
    # Systems like Win32...
    pass

def CheckPKGConfig(context, version):
     context.Message( 'Checking for pkg-config... ' )
     ret = context.TryAction('pkg-config --atleast-pkgconfig-version=%s' % version)[0]
     context.Result( ret )
     return ret

def CheckPKG(context, name):
    context.Message( 'Checking for %s... ' % name )
    ret = context.TryAction('pkg-config --exists \'%s\'' % name)[0]
    context.Result( ret )
    return ret

# Generate a config.py file with build data.
def create_config_py(env):
    mypaint_brushdir = check_output(['pkg-config', '--variable=brushesdir', 'mypaint-brushes-2.0'])
    mypaint_brushdir = mypaint_brushdir.strip()
    config_info = {
        '@MYPAINT_BRUSHDIR@': mypaint_brushdir,
    }
    config_file = env.Substfile('config.py', 'config.py.in',
                                SUBST_DICT=config_info)
    install_perms(env, '$prefix/share/mypaint', config_file)

    return config_file

def burn_versions(env, target, source):
    """Pseudo-builder: bakes version info into the target Python script.

    This also burns in a suitable #! line for "python_binary".
    """
    def _burn_versions(target, source, env):
        sys.path.insert(0, ".")
        import lib.meta
        relinfo_script = lib.meta._get_release_info_script(gitprefix="git")
        header = "\n".join([
            "#!/usr/bin/env {python_binary}",
            "#",
            "# ***DO NOT EDIT THIS FILE***: edit {source} instead.",
            "#",
            "# Auto-generated version info:",
            "{relinfo_script}",
        ]).format(
            python_binary = env['python_binary'],
            relinfo_script = relinfo_script,
            source = source[0],
        )
        with open(unicode(target[0]), 'w') as output:
            output.write(header)
            with open(unicode(source[0])) as input:
                output.write(input.read())
    c = env.Command(target, source, [
        _burn_versions,
        Chmod(target, 0o755),
    ])
    d = env.Depends(target, env.Value(env["python_binary"]))
    # But don't depend on the git output:
    # users often run "scons prefix=/foo install" to install
    # after invoking "scons prefix=foo" to build.
    # In this case, we don't want git updating its index,
    # or scons to update any files.
    return [c, d]

env.AddMethod(burn_versions, "BurnVersions")


## Build-time customization

# User-facing executable Python code
# MyPaint app
env.BurnVersions('mypaint', 'mypaint.py')

# Thumbnailer script and .thumbnailer
# Only build or install on platforms where we might expect to find
# GNOME or Cinnamon.
if os.name == "posix":
    env.BurnVersions(
        'desktop/mypaint-ora-thumbnailer',
        'desktop/mypaint-ora-thumbnailer.py',
    )
    install_perms(env,
        '$prefix/bin',
        'desktop/mypaint-ora-thumbnailer',
        perms=0o755,
    )
    install_perms(env,
        '$prefix/share/thumbnailers',
        'desktop/mypaint-ora.thumbnailer',
    )


## Additional cleanup

env.Clean('.', Glob('*.pyc'))
env.Clean('.', Glob('gui/*.pyc'))
env.Clean('.', Glob('gui/colors/*.pyc'))
env.Clean('.', Glob('lib/*.pyc'))
env.Clean('.', Glob('lib/layer/*.pyc'))

## Configuration

conf = Configure(env, custom_tests = { 'CheckPKGConfig' : CheckPKGConfig,
                                       'CheckPKG' : CheckPKG })

if not conf.CheckPKGConfig('0.4.0'):
    # At least 0.4.0 is needed for option --variable.
    # Probably an even more recent version is preferred though.
    print 'pkg-config >= 0.4.0 not found.'
    Exit(1)

if not conf.CheckPKG('mypaint-brushes-2.0 >= 2.0'):
    print('mypaint-brushes-2.0 >= 2.0 not found.')
    Exit(1)

create_config_py(env)

## Installation

# Painting resources
install_tree(env, '$prefix/share/mypaint', 'backgrounds')
install_tree(env, '$prefix/share/mypaint', 'pixmaps')
install_tree(env, '$prefix/share/mypaint', 'palettes')

# Desktop resources and themeable internal icons
install_tree(env, '$prefix/share', 'desktop/icons')
install_perms(env, '$prefix/share/applications', 'desktop/mypaint.desktop')
install_perms(env, '$prefix/share/appdata', 'desktop/mypaint.appdata.xml')

# location for achitecture-dependent modules
install_perms(env, '$prefix/lib/mypaint', mypaintlib)

# Program and supporting UI XML
install_perms(env, '$prefix/bin', 'mypaint', perms=0o755)
install_perms(env, '$prefix/share/mypaint/gui', Glob('gui/*.xml'))
install_perms(env, '$prefix/share/mypaint/gui', Glob('gui/*.glade'))
install_perms(env, "$prefix/share/mypaint/lib",      Glob("lib/*.py"))
install_perms(env, "$prefix/share/mypaint/lib/layer", Glob("lib/layer/*.py"))
install_perms(env, "$prefix/share/mypaint/gui",      Glob("gui/*.py"))
install_perms(env, "$prefix/share/mypaint/gui/colors", Glob("gui/colors/*.py"))


Return('mypaintlib')

# vim:syntax=python
