# This file is part of MyPaint.

# Imports:

from __future__ import print_function
from contextlib import contextmanager
import subprocess
import glob
import os
import os.path
import pprint
import sys
import textwrap
import tempfile
import shutil

from distutils.command.build import build
from distutils.command.clean import clean
from distutils.dir_util import mkpath

from setuptools import setup
from setuptools import Extension
from setuptools import Command
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.install_scripts import install_scripts

# Constants


# Some versions of clang requires different flag configurations than gcc
# to link correctly, so we enable configuration via environemnt variables.
OPENMP_CFLAG = os.getenv("OPENMP_CFLAG", "-fopenmp")
OPENMP_LDFLAG = os.getenv("OPENMP_LDFLAG", "-fopenmp")


# Helper classes and routines:


def pkgconf():
    """Returns the name used to execute pkg-config
    Uses the value of the PKG_CONFIG environment variable if it is set.
    """
    return os.getenv("PKG_CONFIG", "pkg-config")


def msgfmt():
    """Returns the name used to execute msgfmt
    """
    return os.getenv("MSGFMT", "msgfmt")


class BuildTranslations (Command):
    """Builds binary message catalogs for installation.

    This is declared as a subcommand of "build", but it can be invoked
    in its own right. The generated message catalogs are later installed
    as data files.

    """

    @staticmethod
    def all_locales():
        return [f[:-3] for f in os.listdir("./po/") if f.endswith('.po')]

    @staticmethod
    def get_translation_paths(command, lang_codes=None):
        """Returns paths for building and installing message catalogs

        The returned data is a tuple with two lists.
        The first contains (source, destination) pairs for building
        *.mo files from *.po files (relative paths).
        The second contains (gen_mo_src_dir, [mo_target_path]) tuples
        that are used by the 'install' command.
        """
        lang_codes = lang_codes or BuildTranslations.all_locales()
        tmpdir = command.get_finalized_command("build").build_temp

        msg_path_pairs = []
        data_path_pairs = []
        for lang in lang_codes:
            po_path = os.path.join("po", lang + ".po")
            mo_dir = os.path.join("locale", lang, "LC_MESSAGES")
            targ = os.path.join(tmpdir, mo_dir, "mypaint.mo")
            msg_path_pairs.append((po_path, targ))
            data_path_pairs.append((mo_dir, [targ]))
        return (msg_path_pairs, data_path_pairs)

    description = "build binary message catalogs (*.mo)"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        msg_paths = BuildTranslations.get_translation_paths(self)[0]
        for po_path, mo_path in msg_paths:
            self._compile_message_catalog(po_path, mo_path)
        tmp_dir = self.get_finalized_command("build").build_temp

        # Create a symlink to the locale directory to enable
        # use of translations when running from the source dir.
        # For now only enabled on OS's w. py symlinking available.
        if hasattr(os, "symlink"):
            base, tmp = os.path.split(tmp_dir)
            link_dest = os.path.join(base, "locale")
            link_src = os.path.join(tmp, "locale")
            if os.path.exists(link_dest):
                os.remove(link_dest)
            os.symlink(link_src, link_dest)

    def _compile_message_catalog(self, po_file_path, mo_file_path):
        needs_update = not (
            os.path.exists(mo_file_path) and
            os.stat(mo_file_path).st_mtime >=
            os.stat(po_file_path).st_mtime
        )

        if needs_update:
            cmd = (msgfmt(), po_file_path, "-o", mo_file_path)
            if self.dry_run:
                self.announce("would run %s" % (" ".join(cmd),), level=2)
            else:
                self.announce("running %s" % (" ".join(cmd),), level=2)
                self.mkpath(os.path.dirname(mo_file_path))
                subprocess.check_call(cmd)

                assert os.path.exists(mo_file_path)


class BuildConfig (Command):
    """Builds configuration file config.py.

    This allows python files to know where to find data when it is
    provided as a separate package, i.e. the brushes.

    It also handles which translation files will be included
    (all of them, for non-release builds)
    """

    description = "generate lib/config.py using fetched or provided values"
    user_options = [
        ("brushdir-path=", None,
         "use the provided argument as brush directory path"),
        ("translation-threshold=", None,
         "Limit translations to those with a completion percentage"
         "at or above the given threshold. Argument range: [0..100]"),
    ]

    LOCALE_CACHE = "locale_cache"
    COMPLETION_THRESHOLD = 75
    WARNING_TEMPLATE = (
        "# == THIS FILE IS GENERATED ==\n"
        "# DO NOT EDIT OR ADD TO VERSION CONTROL\n"
        "# The structure is defined in {input_file}\n"
        "# The generation is done by the {command_name} command in {script}\n"
        "\n"
    )

    def initialize_options(self):
        self.brushdir_path = None
        self.translation_threshold = 0

    def finalize_options(self):
        if self.brushdir_path and self.brushdir_path.strip()[0] == '/':
            print("WARNING: supplied brush directory path is not relative")
        self.translation_threshold = int(self.translation_threshold)
        assert(0 <= self.translation_threshold <= 100)

    def run(self):
        # Determine path to the brushes directory
        if self.brushdir_path is not None:
            mypaint_brushdir = self.brushdir_path
        else:
            mypaint_brushdir = self.pkgconf_brushdir_path()

        # Determine which locales are supported, based on existing po files
        # and optionally their level of completeness (% of strings translated)
        locales = self._get_locales()
        # Pretty print sorted locales to individual lines in list
        locstring = " " + pprint.pformat(sorted(locales), indent=4)[1:-1] + ","

        files = {
            'config.py.in': 'lib/config.py',
        }
        replacements = {
            '@MYPAINT_BRUSHDIR@': mypaint_brushdir,
            '@SUPPORTED_LOCALES@': locstring,
        }
        self.replace(files, replacements)

    @staticmethod
    def translation_completion_func():
        """Get a function for calculating translation completeness

        Tries to use polib if possible, but falls back to a shell script
        that only uses existing dependencies if polib is not installed.

        :return: Function calculating po translation completeness
        """
        try:
            import polib

            def py_completion(_, path, template=False):
                po = polib.pofile(path)
                if template:
                    return len(po)
                else:
                    return len(po.translated_entries())
            return py_completion
        except ImportError:
            print("polib not installed, falling back to shellscript!")

            def sh_completion(loc, _, template=False):
                cmd = [os.path.join("po", "num_strings_translated.sh")]
                if not template:
                    cmd.append(loc)
                return int(subprocess.check_output(cmd))
            return sh_completion

    def _get_locales(self):
        """Return a list of locales to use/install

        If no limitation is set (default) all locales are based solely on
        the existing *.po-files in the po directory. If limitation is enabled,
        the list is filtered based on the percentage of strings translated
        for each translation file.

        :return: A list of locales
        """
        locales = BuildTranslations.all_locales()
        if not self.translation_threshold:
            # No limit - just return all of them
            return locales

        # Get the completion percentage for translation files
        # and cache the result to allow for partial updates
        static_linguas_path = os.path.join("po", "STATIC_LINGUAS")
        with open(static_linguas_path, "r") as sl_file:
            static_linguas = sl_file.read().strip().split("\n")
        completion = BuildConfig.translation_completion_func()
        template_path = os.path.join("po", "mypaint.pot")
        total = float(completion(None, template_path, template=True))
        # Read/update cache
        with self._get_locale_data_cache() as cache:
            for loc in locales:
                # For static always-include locales, set a faux percentage of
                # 100, but set timestamp to 0 to force calculation on removal
                # from the STATIC_LINGUAS locale list.
                if loc in static_linguas:
                    cache[loc] = (100, 0)
                    continue
                po_path = os.path.join("po", loc + ".po")
                po_modified_time = os.stat(po_path).st_mtime
                if loc not in cache or cache[loc][1] < po_modified_time:
                    print("Updating cache:", loc)
                    num = completion(loc, po_path)
                    percentage = 100 * num / total
                    # Add some margin to avoid rounding problems
                    cache[loc] = (percentage, po_modified_time + 0.1)

        thresh = self.translation_threshold
        return [k for k, v in cache.items() if v[0] > thresh]

    @contextmanager
    def _get_locale_data_cache(self):
        """Retrieve/update locale info cache

        The locale cache holds cached information about the
        level of completion for individual locales, along
        with a timestamp for when this completion was last
        calculated.
        """
        build_tmp_dir = self.get_finalized_command("build").build_temp
        cache_file = os.path.join(build_tmp_dir, self.LOCALE_CACHE)
        if not os.path.isfile(cache_file):
            info_dict = dict()
        else:
            # Read in file to a list of lines
            with open(cache_file, "r") as f:
                info_lines = f.read().strip().split("\n")
            # Turn list of lines into a dict of (completion, timestamp)
            # tuples, keyed by the locale code.
            info_line_lists = [l.split(" ") for l in info_lines]
            info_dict = {loc: (float(completion), float(timestamp))
                         for loc, completion, timestamp in info_line_lists}
        yield info_dict
        # Turn the (modified) dict back into space-separated values
        # and overwrite the cache file (cache file timestamp is irrelevant)
        out = [str(loc) + " " + str(v[0]) + " " + str(v[1])
               for loc, v in info_dict.items()]
        mkpath(build_tmp_dir)
        with open(cache_file, "w") as f:
            f.write("\n".join(out))

    def pkgconf_brushdir_path(self):
        try:
            cmd = [
                pkgconf(), '--variable=brushesdir', 'mypaint-brushes-2.0'
            ]
            return subprocess.check_output(cmd).decode().strip()
        except subprocess.CalledProcessError as e:
            sys.stderr.write(
                str(e) +
                'pkg-config could not find package mypaint-brushes-2.0'
            )
            sys.exit(os.EX_CANTCREAT)

    def replace(self, files, replacements):
        for f in files:
            try:
                shutil.copyfile(f, files[f])
                fd = open(files[f], 'r+')
                contents = fd.read()
                # Make the necessary replacements.
                for r in replacements:
                    contents = contents.replace(r, replacements[r])
                warning = self.WARNING_TEMPLATE.format(
                    input_file=f,
                    command_name=self.__class__.__name__,
                    script=__file__
                )
                fd.truncate(0)
                fd.seek(0)
                fd.write(warning)
                fd.write(contents)
                fd.flush()
                fd.close()
            except IOError:
                sys.stderr.write(
                    'The script {} failed to update. '
                    'Check your permissions.'.format(f)
                )
                sys.exit(os.EX_CANTCREAT)


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
        [("build_config", None)] +
        [(a, b) for (a, b) in build.sub_commands if a == 'build_ext'] +
        [(a, b) for (a, b) in build.sub_commands if a != 'build_ext'] +
        [("build_translations", None)]
    )


class BuildExt (build_ext):
    """Custom build_ext.
    Adds additional behaviour to --debug option and
    adds an option to amend the rpath with library paths
    from dependencies found via pkg-config
    """

    user_options = [
        ("set-rpath", "f",
         "[MyPaint] Add dependency library paths from pkg-config "
         "to the rpath of mypaintlib (linux only)"),
        ("disable-openmp", None,
         "Don't use openmp, even if the platform supports it."),
    ] + build_ext.user_options

    def initialize_options(self):
        self.set_rpath = False
        self.disable_openmp = False
        build_ext.initialize_options(self)

    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.set_rpath and sys.platform.startswith("linux"):
            # The directories in runtime_library_dirs will be added
            # to the linker args as '-Wl,-R{dirs}' This _should_ be
            # compatible with the --rpath= build_ext option
            for ext in self.extensions:
                rt_libs = uniq(ext.library_dirs + ext.runtime_library_dirs)
                # Retain original list reference, just in case
                ext.runtime_library_dirs[:] = rt_libs

    def build_extension(self, ext):
        ccflags = ext.extra_compile_args
        linkflags = ext.extra_link_args

        if sys.platform != "darwin" and not self.disable_openmp:
            linkflags.append(OPENMP_CFLAG)
            ccflags.append(OPENMP_LDFLAG)

        if self.debug:
            skip = ["-DNDEBUG"]
            ccflags[:] = [f for f in ccflags if f not in skip]
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


class Install (install):
    """Custom install to handle translation files
    Same options as the regular install command.
    """

    def run(self):
        # lib.config is a module generated as part of the build process,
        # and may not exist when the setup script is run,
        # hence it should not (and often cannot) be a top-level import.
        import lib.config
        # We only install the locales added in the build_config step.
        locales = lib.config.supported_locales
        data_paths = BuildTranslations.get_translation_paths(self, locales)[1]
        self.distribution.data_files.extend(data_paths)
        install.run(self)


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
    """Builds, then do a test run from a temporary install tree"""

    description = "[MyPaint] build, install, and run in a throwaway folder"
    user_options = [
        ("temp-root=", None,
         "parent dir (retained) for the demo install (deleted)"),
    ]

    def initialize_options(self):
        self.temp_root = None

    def finalize_options(self):
        pass

    def run(self):
        if self.dry_run:
            self.announce(
                "The demo command can't do anything in dry-run mode",
                level=2,
            )
            return

        build = self.get_finalized_command("build")
        build.run()

        build_scripts = self.get_finalized_command("build_scripts")
        demo_cmd = [build_scripts.executable]

        temp_dir = tempfile.mkdtemp(
            prefix="demo-",
            suffix="",
            dir=self.temp_root,
        )
        try:
            inst = self.distribution.get_command_obj("install", 1)
            inst.root = temp_dir
            inst.prefix = ""
            inst.ensure_finalized()
            inst.run()

            script_path = None
            potential_script_names = ["mypaint.py", "mypaint"]
            for s in potential_script_names:
                p = os.path.join(inst.install_scripts, s)
                if os.path.exists(p):
                    script_path = p
                    break
            if not script_path:
                raise RuntimeError(
                    "Cannot locate installed script. "
                    "Tried all of %r in %r."
                    % (potential_script_names, inst.install_scripts)
                )

            config_dir = os.path.join(temp_dir, "_config")

            demo_cmd.extend([script_path, "-c", config_dir])

            self.announce("Demo: running %r..." % (demo_cmd,), level=2)
            subprocess.check_call(demo_cmd)
        except Exception:
            raise
        finally:
            self.announce("Demo: cleaning up %r..." % (temp_dir,), level=2)
            shutil.rmtree(temp_dir, ignore_errors=True)


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


class _ManagedInstBase (Command):
    """Base for commands that work on managed installs."""

    MANAGED_FILES_LIST = "managed_files.txt"

    # Common options:

    user_options = [
        ("prefix=", None, "POSIX-style install prefix, e.g. /usr"),
    ]

    def initialize_options(self):
        self.prefix = "/usr/local"

    def finalize_options(self):
        self.prefix = os.path.abspath(self.prefix)

    # Utility methods for subclasses:

    def is_installed(self):
        """True if an installation is present at the prefix."""
        inst = self.distribution.get_command_obj("install", 1)
        inst.root = self.prefix
        inst.prefix = "."
        inst.ensure_finalized()
        inst.record = os.path.join(inst.install_lib, self.MANAGED_FILES_LIST)
        return os.path.isfile(inst.record)

    def uninstall(self):
        """Uninstalls an existing install."""
        self.announce(
            "Uninstalling from %r..." % (self.prefix,),
            level=2,
        )

        inst = self.distribution.get_command_obj("install", 1)
        inst.root = self.prefix
        inst.prefix = "."
        inst.ensure_finalized()
        inst.record = os.path.join(inst.install_lib, self.MANAGED_FILES_LIST)

        self.announce(
            "Using the installation record in %r" % (inst.record,),
            level=2,
        )
        with open(inst.record, "r") as fp:
            for path_rel in fp:
                path_rel = path_rel.rstrip("\r\n")
                path = os.path.join("/", path_rel)
                self.rm(path)
        self.rm(inst.record)

        # Remove the trees that are entirely MyPaint's too
        # See setup.cfg's [install] and get_data_files()
        mypaint_data_trees = [
            inst.install_lib,
            os.path.join(inst.install_data, "mypaint"),
        ]
        for path in mypaint_data_trees:
            path = os.path.normpath(path)
            assert os.path.basename(path) == "mypaint", \
                "path %r does not end in mypaint" % (path,)
            self.rmtree(path)

    def rmtree(self, path):
        """Remove a tree recursively."""
        self.announce("recursively removing %r" % (path,), level=2)
        if not self.dry_run:
            shutil.rmtree(path, ignore_errors=True)

    def rm(self, path):
        """Remove a single file."""
        self.announce("removing %r" % (path,), level=2)
        if not self.dry_run:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.exists(path):
                raise RuntimeError(
                    "ERROR: %r exists, but it is no longer a file"
                    % (path,)
                )
            else:
                self.announce("remove: %r is already gone" % (path,), level=1)


class ManagedInstall (_ManagedInstBase):
    """Simplified "install" with a list of installed files.

    This command and ManagedUninstall are temporary hacks which we have
    to use because `pip {install,uninstall}` doesn't work yet. Once we
    have proper namespacing (prefixed `mypaint.{lib,gui}` modules), we
    may be able to drop these commands and use standard ones.

    """

    description = "[MyPaint] like install, but allow managed_uninstall"

    def run(self):

        if self.is_installed():
            self.announce("Already installed, uninstalling first...", level=2)
            self.uninstall()

        self.announce(
            "Installing (manageably) to %r" % (self.prefix,),
            level=2,
        )

        inst = self.distribution.get_command_obj("install", 1)
        inst.root = self.prefix
        inst.prefix = "."
        inst.ensure_finalized()
        inst.record = os.path.join(inst.install_lib, self.MANAGED_FILES_LIST)
        self.announce(
            "Record will be saved to %r" % (inst.record,),
            level=2,
        )

        if not self.dry_run:
            inst.run()
            assert os.path.isfile(inst.record)


class ManagedUninstall (_ManagedInstBase):
    """Removes the files created by ManagedInstall."""

    description = "[MyPaint] remove files that managed_install wrote"

    user_options = [
        ("prefix=", None, "same as used in your earlier managed_install"),
    ]

    def initialize_options(self):
        self.prefix = "/usr/local"

    def finalize_options(self):
        self.prefix = os.path.abspath(self.prefix)

    def run(self):
        self.uninstall()


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
        cmd = [pkgconf(), pc_arg] + list(packages)
        cmd_output = subprocess.check_output(
            cmd,
            universal_newlines=True,
        )
        for conf_arg in cmd_output.split():
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
        '--std=c++11',
        '-Wall',
        '-Wno-sign-compare',
        '-Wno-write-strings',
        '-D_POSIX_C_SOURCE=200809L',
        "-DNO_TESTS",  # FIXME: we're building against shared libmypaint now
        '-g',  # always include symbols, for profiling
    ]
    extra_link_args = []

    if sys.platform == "darwin":
        pass
    elif sys.platform == "win32":
        pass
    elif sys.platform == "msys":
        pass
    elif sys.platform.startswith("linux"):
        # Look up libraries dependencies relative to the library.
        extra_link_args.append('-Wl,-z,origin')
        extra_link_args.append('-Wl,-rpath,$ORIGIN')

    # Ensure that the lib path of libmypaint is added first
    # to the rpath when the --set-rpath option is used.
    # For most users this will be the case anyway, due to the
    # order of the pkg-config output, but this ensures it.
    mypaintlib_opts = pkgconfig(
        packages=[
            "libmypaint-2.0",
        ],
        include_dirs=[
            numpy.get_include(),
        ],
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
    )
    # Append the info from the rest of the dependencies
    mypaintlib_opts = pkgconfig(
        packages=[
            "pygobject-3.0",
            "glib-2.0",
            "libpng",
            "lcms2",
            "gtk+-3.0",
            "mypaint-brushes-2.0",
        ],
        **mypaintlib_opts
    )

    mypaintlib_swig_opts = ['-Wall', '-noproxydel', '-c++']
    mypaintlib_swig_opts.extend([
        "-I" + d
        for d in mypaintlib_opts["include_dirs"]
    ])
    # FIXME: building against the new shared lib, omit old test code
    mypaintlib_swig_opts.extend(['-DNO_TESTS'])

    mypaintlib = Extension(
        'lib._mypaintlib',
        [
            'lib/mypaintlib.i',
            'lib/gdkpixbuf2numpy.cpp',
            'lib/pixops.cpp',
            'lib/fastpng.cpp',
            'lib/brushsettings.cpp',
            'lib/fill/fill_common.cpp',
            'lib/fill/fill_constants.cpp',
            'lib/fill/floodfill.cpp',
            'lib/fill/gap_closing_fill.cpp',
            'lib/fill/gap_detection.cpp',
            'lib/fill/blur.cpp',
            'lib/fill/morphology.cpp',
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
        ("metainfo", ["desktop/mypaint.appdata.xml"]),
        ("applications", ["desktop/mypaint.desktop"]),
        ("thumbnailers", ["desktop/mypaint-ora.thumbnailer"]),
    ]

    # Paths which can only derived from globbing the source tree.
    data_file_patterns = [
        # SRCDIR, SRCPATT, TARGDIR
        ("desktop/icons", "hicolor/*/*/*", "icons"),
        ("backgrounds", "*.*", "mypaint/backgrounds"),
        ("backgrounds", "*/*.*", "mypaint/backgrounds"),
        ("palettes", "*.gpl", "mypaint/palettes"),
        ("pixmaps", "*.png", "mypaint/pixmaps"),
    ]
    for (src_prefix, src_pattern, targ_prefix) in data_file_patterns:
        for src_file in glob.glob(os.path.join(src_prefix, src_pattern)):
            file_rel = os.path.relpath(src_file, src_prefix)
            targ_dir = os.path.join(targ_prefix, os.path.dirname(file_rel))
            data_files.append((targ_dir, [src_file]))

    return data_files


# Setup script "main()":

setup(
    name='MyPaint',
    version='2.0.0a0',
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
        "build_config": BuildConfig,
        "demo": Demo,
        "install": Install,
        "managed_install": ManagedInstall,
        "managed_uninstall": ManagedUninstall,
        "install_scripts": InstallScripts,
        "clean": Clean,
    },
    scripts=[
        "mypaint.py",
        "desktop/mypaint-ora-thumbnailer.py",
    ],
    test_suite='tests',
    ext_modules=get_ext_modules(),
)
