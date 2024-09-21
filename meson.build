# This file is part of MyPaint.
# Copyright (C) 2024 the MyPaint project
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# /meson.build -- The root buildfile for Meson

project('MyPaint', 'cpp',
  version : '2.1.0-alpha')

# Lint version number
ver_prerel_valid = ['', 'alpha', 'beta', 'rc']
ver_full = meson.project_version().split('.')
ver_major = ver_full[0]
ver_minor = ver_full[1]

## Find patch and prerelease values
if ver_full[2].contains('-')
  ver_patch_prerel = ver_full[2].split('-')
  ver_patch = ver_patch_prerel[0]
  ver_prerel = ver_patch_prerel[1]
else
  ver_patch = ver_full[2]
  ver_prerel = ''
endif

## Find prerelease number (if any)
if ver_full.length() == 4 and ver_prerel != ''
  ver_prerel_num = ver_full[3]
else
  ver_prerel_num = '0'
endif

if not ver_prerel_valid.contains(ver_prerel)
  error('Invalid prerelease type: ' + ver_prerel)
endif

# Build machine info
host_os = host_machine.system()

## Install directories
project_subdir = meson.project_name().to_lower()
platlibdir = get_option('libdir') / project_subdir
opt_purelibdir = get_option('libdir_noarch')
if opt_purelibdir == ''
  purelibdir = platlibdir
else
  purelibdir = opt_purelibdir / project_subdir
endif
bindir = get_option('bindir')
datadir = get_option('datadir')
project_datadir = datadir / project_subdir
localedir = get_option('localedir')
includedir = get_option('includedir')

# Modules
## Python
python_mod = import('python')
python_deps = ['cairo','gettext','gi','numpy', 'polib']
python = python_mod.find_installation('python3', modules: python_deps)
## Filesystem
fs = import('fs')
## i18n
i18n = import('i18n')

# Dependencies
python_dep = python.dependency(embed: true)
## First party dependencies
brushlib_dep = dependency('libmypaint-2.0')
brushes_dep = dependency('mypaint-brushes-2.0')
## Third party dependencies
cairo_dep = dependency('cairo', version: '>=1.4')
lcms2_dep = dependency('lcms2')
libpng_dep = dependency('libpng')
librsvg_dep = dependency('librsvg-2.0')
json_dep = dependency('json-c')
### GTK
gtk_dep = dependency('gtk+-3.0', version: '>=3.12')
glib_dep = dependency('glib-2.0')
gobject_dep = dependency('gobject-2.0')
pygobject_dep = dependency('pygobject-3.0')

## Array of extension deps
dep_list = [brushlib_dep, pygobject_dep, glib_dep, libpng_dep, lcms2_dep, gtk_dep, brushes_dep]
dep_names = []
foreach dep : dep_list
  dep_names += dep.name()
endforeach

# Helper commands
git = find_program('git', required : true)
linguas_cmd = find_program('build-aux/meson/get_locales.py', required : true)
numpy_include_cmd = find_program('build-aux/meson/numpy_include.py', required : true) # FIXME: meson issue #9598
pkgconfig = find_program('pkg-config', required: true)
swig = find_program('swig', required: true)

## Supported locales
opt_translation_threshold = get_option('translation_threshold').to_string()
supported_locales = run_command(linguas_cmd, ['data/po', opt_translation_threshold], check: true).stdout().strip().split( )

# Build & configure
subdir('src') # Configuration and binaries
subdir('src/platlib') # Build C++ extension
install_subdir('src/purelib', install_dir : purelibdir + '/lib', strip_directory : true) # Python library scripts
install_subdir('src/gui', install_dir : purelibdir + '/gui', strip_directory : true) # Python GUI scripts
subdir('data') # Data files
subdir('data/po') # Translations
