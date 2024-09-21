# This file is part of MyPaint.
# Copyright (C) 2024 the MyPaint project
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# /src/lib_ext/meson.build

# Basic declarations
numpy_include = '-I' + run_command(numpy_include_cmd, check: true).stdout().strip()

# swig configuration

## Check command version and set python opts accordingly
# Returns: "\nSWIG Version 4.1.1\n\nCompiled with ..."
swig_version_cmd = run_command(swig, ['-version'], check: true)
swig_version = swig_version_cmd.stdout().split('\n')[1].split()[2].strip()
if swig_version.version_compare('<4.1.0')
  mypaintlib_swig_args = [swig, '-python', '-py3']
else
  mypaintlib_swig_args = [swig, '-python']
endif

## Other
mypaintlib_swig_args += ['-Wall', '-c++']
mypaintlib_swig_args += ['-DNO_TESTS'] # FIXME: Building against the new shared lib, omit old test code

## Extensions
mypaintlib_swig_args += [numpy_include]
dep_cflags = run_command(pkgconfig, dep_names, ['--cflags-only-I'], check: true).stdout().strip().split()
mypaintlib_swig_args += dep_cflags

# Run swig
mypaintlib_swig = custom_target(
  'mypaintlib.py',
  input: ['mypaintlib.i'],
  output:  ['mypaintlib.py', 'mypaintlib_wrap.cpp'],
  command: [mypaintlib_swig_args, '-o', '@OUTPUT1@', '@INPUT0@'],
  install: true,
  install_dir: [platlibdir + '/lib', false],
  )

# Shared library configuration
mypaintlib_compile_args = [
  numpy_include,
  '--std=c++11',
  '-Wall', # FIXME: this is duplicated in the final command
  '-Wno-sign-compare',
  '-Wno-write-strings',
  '-D_POSIX_C_SOURCE=200809L',
  '-g']  # always include symbols, for profiling, FIXME: this is duplicated in the final command.
mypaintlib_compile_args += ['-DNO_TESTS']  # FIXME: we're building against shared libmypaint now
mypaintlib_link_args = []

## Operating system
if host_os == 'darwin'
  mypaintlib_compile_args += ['-D_DARWIN_C_SOURCE']
endif

mypaintlib_sources = [
  mypaintlib_swig[1],
  'gdkpixbuf2numpy.cpp',
  'pixops.cpp',
  'fastpng.cpp',
  'brushsettings.cpp',
  'fill/fill_common.cpp',
  'fill/fill_constants.cpp',
  'fill/floodfill.cpp',
  'fill/gap_closing_fill.cpp',
  'fill/gap_detection.cpp',
  'fill/blur.cpp',
  'fill/morphology.cpp']

## openmp
## macOS doesn't ship with openmp by default, but there should be ways to work around it
## https://mac.r-project.org/openmp/
if not get_option('disable_openmp') and host_os != 'darwin'
  mypaintlib_compile_args += get_option('openmp_cflag')
  mypaintlib_link_args += get_option('openmp_ldflag')
endif

## Optimisations
if get_option('use_optimisations')
  optimised_builds = ['debugoptimized', 'release']
  if optimised_builds.contains(get_option('buildtype')) or get_option('debug_optimisations')
    mypaintlib_compile_args += [
      '-Ofast',
      '-ftree-vectorize',
      '-fopt-info-vec-optimized',
      '-march=native',
      '-mtune=native',
      '-funsafe-math-optimizations',
      '-funsafe-loop-optimizations']
  endif
endif

# Build shared library
shared_library(
  # The old build system produced "_mypaintlib.cpython-311-x86_64-linux-gnu.so". While we can try to match it, this is
  # close enough
  'mypaintlib',
  name_prefix: '_',
  sources: mypaintlib_sources,
  cpp_args: mypaintlib_compile_args,
  link_args: mypaintlib_link_args,
  dependencies: [dep_list, python_dep],
  link_depends: mypaintlib_swig,
  install: true,
  install_dir: platlibdir + '/lib')
