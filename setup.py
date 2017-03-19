# This file is part of MyPaint.

# Imports:

from __future__ import print_function
import subprocess
import glob
import os
import os.path
import sys
import textwrap

from distutils.command.build import build
from distutils.command.clean import clean

from setuptools import setup
from setuptools import Extension
from setuptools import Command
from setuptools.command.build_ext import build_ext
from setuptools.command.install_scripts import install_scripts


# Helper classes and routines:

class BuildTranslations (Command):
    """Builds binary message catalogs for installation.

    This is declared as a subcommand of "build", but it can be invoked
    in its own right. The generated message catalogs are later installed
    as data files.

    """

    description = "build binary message catalogs (*.mo)"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        build = self.get_finalized_command("build")
        for src in glob.glob("po/*.po"):
            data_files = self._compile_message_catalog(src, build.build_temp)
            if not self.dry_run:
                self.distribution.data_files.extend(data_files)

    def _compile_message_catalog(self, src, temp):
        lang = os.path.basename(src)[:-3]
        targ_dir = os.path.join(temp, "locale", lang, "LC_MESSAGES")
        targ = os.path.join(targ_dir, "mypaint.mo")
        install_dir = os.path.join("locale", lang, "LC_MESSAGES")

        needs_update = True
        if os.path.exists(targ):
            if os.stat(targ).st_mtime >= os.stat(src).st_mtime:
                needs_update = False

        if needs_update:
            cmd = ("msgfmt", src, "-o", targ)
            if self.dry_run:
                self.announce("would run %s" % (" ".join(cmd),), level=2)
                return []
            self.announce("running %s" % (" ".join(cmd),), level=2)

            self.mkpath(targ_dir)
            subprocess.check_call(cmd)

        assert os.path.exists(targ)
        return [(install_dir, [targ])]


class Build (build):
    """Custom build (build_ext 1st for swig, run build_translations)

    distutils.command.build.build doesn't generate the extension.py for
    an _extension.so, unless the build_ext is done first or you install
    twice. The fix is to do the build_ext subcommand before build_py.
    In our case, swig runs first. Still needed as of Python 2.7.13.
    Fix adapted from https://stackoverflow.com/questions/17666018>.

    This build also ensures that build_translations is run.

    """

    sub_commands = (
        [(a, b) for (a, b) in build.sub_commands if a == 'build_ext'] +
        [(a, b) for (a, b) in build.sub_commands if a != 'build_ext'] +
        [("build_translations", None)]
    )


class BuildExt (build_ext):
    """Custom build_ext (extra --debug flags)."""

    def build_extension(self, ext):
        ccflags = ext.extra_compile_args
        linkflags = ext.extra_link_args

        if self.debug:
            for flag in ["-DNDEBUG"]:
                if flag in ccflags:
                    ccflags.remove(flag)
            ccflags.extend([
                "-O0",
                "-g",
                "-DHEAVY_DEBUG",
            ])
            linkflags.extend([
                "-O0",
            ])
        else:
            linkflags.append("-O3")
            ccflags.append("-O3")

        return build_ext.build_extension(self, ext)


class Clean (clean):
    """Custom clean: also remove swig-generated wrappers.

    distutils's clean has always left these lying around in the source,
    and they're a perpetual trip hazard when sharing the same source
    tree with a Windows VM.

    """

    def run(self):
        build_temp_files = glob.glob("lib/mypaintlib_wrap.c*")
        for file in build_temp_files:
            self.announce("removing %r" % (file,), level=2)
            os.unlink(file)
        return clean.run(self)


class Demo (Command):
    """Builds, and then does a MyPaint test run from the build."""

    description = "Rebuild, and then run inside the build tree."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        build = self.get_finalized_command("build")
        build.run()

        build_scripts = self.get_finalized_command("build_scripts")
        cmd = [
            build_scripts.executable,
            os.path.join(build.build_scripts, "mypaint.py"),
        ]

        self.announce("Running %r..." % (" ".join(cmd),), level=2)
        if self.dry_run:
            return

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.pathsep.join([
            os.path.abspath(build.build_lib),
            os.path.abspath(build.build_purelib),
            os.path.abspath(build.build_platlib),
        ])
        subprocess.check_call(
            cmd,
            env=env,
        )


class InstallScripts (install_scripts):
    """Install scripts with ".py" suffix removal and version headers.

    Bakes version information into each installed script.
    The .py suffix is also removed on most platforms we support.

    """

    def run(self):
        if not self.skip_build:
            self.run_command('build_scripts')

        sys.path.insert(0, ".")
        import lib.meta
        relinfo_script = lib.meta._get_release_info_script(gitprefix="git")

        header_tmpl = textwrap.dedent("""
            #
            # ***DO NOT EDIT THIS FILE***: edit {source} instead.
            #
            # Auto-generated version info follows.
            {relinfo_script}
        """)
        self.mkpath(self.install_dir)
        self.outfiles = []

        src_patt = os.path.join(self.build_dir, "*")
        for src in glob.glob(src_patt):
            header = header_tmpl.format(
                relinfo_script=relinfo_script,
                source=os.path.basename(src),
            )
            outfiles = self._install_script(src, header)
            self.outfiles.extend(outfiles)

    def _install_script(self, src, header):
        strip_ext = True
        set_mode = False
        if sys.platform == "win32":
            if "MSYSTEM" not in os.environ:  # and not MSYS2
                strip_ext = False
        targ_basename = os.path.basename(src)
        if strip_ext and targ_basename.endswith(".py"):
            targ_basename = targ_basename[:-3]
        targ = os.path.join(self.install_dir, targ_basename)
        self.announce("installing %s as %s" % (src, targ_basename), level=2)
        if self.dry_run:
            return []
        with open(src, "rU") as in_fp:
            with open(targ, "w") as out_fp:
                line = in_fp.readline().rstrip()
                if line.startswith("#!"):
                    print(line, file=out_fp)
                    print(header, file=out_fp)
                    if os.name == 'posix':
                        set_mode = True
                else:
                    print(header, file=out_fp)
                    print(line, file=out_fp)
                for line in in_fp.readlines():
                    line = line.rstrip()
                    print(line, file=out_fp)
        if set_mode:
            mode = ((os.stat(targ).st_mode) | 0o555) & 0o7777
            self.announce("changing mode of %s to %o" % (targ, mode), level=2)
            os.chmod(targ, mode)
        return [targ]


def uniq(items):
    """Order-preserving uniq()"""
    seen = set()
    result = []
    for i in items:
        if i in seen:
            continue
        seen.add(i)
        result.append(i)
    return result


def pkgconfig(packages, **kwopts):
    """Runs pkgconfig to update its args.

    Also returns the modified args dict. Recipe adapted from
    http://code.activestate.com/recipes/502261/

    """
    flag_map = {
        '-I': 'include_dirs',
        '-L': 'library_dirs',
        '-l': 'libraries',
    }
    extra_args_map = {
        "--libs": "extra_link_args",
        "--cflags": "extra_compile_args",
    }
    for (pc_arg, extras_key) in extra_args_map.items():
        cmd = ["pkg-config", pc_arg] + list(packages)
        for conf_arg in subprocess.check_output(cmd).split():
            flag = conf_arg[:2]
            flag_value = conf_arg[2:]
            flag_key = flag_map.get(flag)
            if flag_key:
                kw = flag_key
                val = flag_value
            else:
                kw = extras_key
                val = conf_arg
            kwopts.setdefault(kw, []).append(val)
    for kw, val in list(kwopts.items()):
        kwopts[kw] = uniq(val)
    return kwopts


def get_ext_modules():
    """Return a list of binary Extensions for setup() to process."""

    import numpy

    extra_compile_args = [
        '-Wall',
        '-Wno-sign-compare',
        '-Wno-write-strings',
        '-D_POSIX_C_SOURCE=200809L',
        "-DNO_TESTS",  # FIXME: we're building against shared libmypaint now
        '-g',  # always include symbols, for profiling
    ]
    extra_link_args = []

    if sys.platform != "darwin":
        extra_link_args.append("-fopenmp")
        extra_compile_args.append("-fopenmp")

    if sys.platform == "darwin":
        pass
    elif sys.platform == "win32":
        pass
    elif sys.platform == "msys":
        pass
    elif sys.platform == "linux2":
        # Look up libraries dependencies relative to the library.
        extra_link_args.append('-Wl,-z,origin')
        extra_link_args.append('-Wl,-rpath,$ORIGIN')

    mypaintlib_opts = pkgconfig(
        packages=[
            "pygobject-3.0",
            "glib-2.0",
            "libpng",
            "lcms2",
            "gtk+-3.0",
            "libmypaint",
        ],
        include_dirs=[
            numpy.get_include(),
        ],
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
    )

    mypaintlib_swig_opts = ['-Wall', '-noproxydel', '-c++']
    mypaintlib_swig_opts.extend([
        "-I" + d
        for d in mypaintlib_opts["include_dirs"]
    ])
    # FIXME: building against the new shared lib, omit old test code
    mypaintlib_swig_opts.extend(['-DNO_TESTS'])

    mypaintlib = Extension(
        '_mypaintlib',
        [
            'lib/mypaintlib.i',
            'lib/fill.cpp',
            'lib/gdkpixbuf2numpy.cpp',
            'lib/pixops.cpp',
            'lib/fastpng.cpp',
            'lib/brushsettings.cpp',
        ],
        swig_opts=mypaintlib_swig_opts,
        language='c++',
        **mypaintlib_opts
    )
    return [mypaintlib]


def get_data_files():
    """Return a list of data_files entries for setup() to process."""

    # Target paths are relative to $base/share, assuming setup.py's
    # default value for install-data.
    data_files = [
        # TARGDIR, SRCFILES
        ("appdata", ["desktop/mypaint.appdata.xml"]),
        ("applications", ["desktop/mypaint.desktop"]),
        ("thumbnailers", ["desktop/mypaint-ora.thumbnailer"]),
        ("mypaint/brushes", ["brushes/order.conf"]),
    ]

    # Paths which can only derived from globbing the source tree.
    data_file_patts = [
        # SRCDIR, SRCPATT, TARGDIR
        ("desktop/icons", "hicolor/*/*/*", "icons"),
        ("backgrounds", "*.*", "mypaint/backgrounds"),
        ("backgrounds", "*/*.*", "mypaint/backgrounds"),
        ("brushes", "*/*.*", "mypaint/brushes"),
        ("palettes", "*.gpl", "mypaint/palettes"),
        ("pixmaps", "*.png", "mypaint/pixmaps"),
    ]
    for (src_pfx, src_patt, targ_pfx) in data_file_patts:
        for src_file in glob.glob(os.path.join(src_pfx, src_patt)):
            file_rel = os.path.relpath(src_file, src_pfx)
            targ_dir = os.path.join(targ_pfx, os.path.dirname(file_rel))
            data_files.append((targ_dir, [src_file]))

    return data_files


# Setup script "main()":

setup(
    name='MyPaint',
    version='1.3.0a0',
    description='Simple painting program for use with graphics tablets.',
    author='Andrew Chadwick',
    author_email='a.t.chadwick@gmail.com',
    license="GPLv2+",
    url="http://mypaint.org",

    packages=['lib', 'lib.layer', 'gui', 'gui.colors'],
    package_data={
        "gui": ['*.xml', '*.glade'],
    },
    data_files=get_data_files(),
    cmdclass={
        "build": Build,
        "build_ext": BuildExt,
        "build_translations": BuildTranslations,
        "demo": Demo,
        "install_scripts": InstallScripts,
        "clean": Clean,
    },
    scripts=[
        "mypaint.py",
        "desktop/mypaint-ora-thumbnailer.py",
    ],
    ext_modules=get_ext_modules(),
)
