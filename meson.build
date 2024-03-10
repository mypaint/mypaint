project('MyPaint', 'cpp',
  version : 'v2.1.0',
  default_options : ['warning_level=3', 'cpp_std=c++14'])

# Build machine info
system_os = host_machine.system()

# Modules
## Python
py_mod = import('python')
py_deps = ['cairo','gettext','gi','numpy']
py = py_mod.find_installation('python3', modules: py_deps)

# Dependencies
python_dep = py.dependency()
## First party dependencies
brushlib_dep = dependency('libmypaint-2.0')
brush_dep = dependency('mypaint-brushes-2.0')

## Third party dependencies
cairo_dep = dependency('cairo', version: '>=1.4')
lcms2_dep = dependency('lcms2')
libpng_dep = dependency('libpng')
json_dep = dependency('json-c')
### GTK
gtk_dep = dependency('gtk+-3.0', version: '>=3.12')
glib_dep = dependency('glib-2.0')
gobject_dep = dependency('gobject-2.0')
pygobject_dep = dependency('pygobject-3.0')

# Build
## Swig
### Define swig
swig = find_program('swig', required: true)

### Define flags
swig_args = [ '--std=c++11',
  '-Wall',
  '-Wno-sign-compare',
  '-Wno-write-strings',
  '-D_POSIX_C_SOURCE=200809L',
  '-DNO_TESTS',  # FIXME: we're building against shared libmypaint now
  '-g',  # always include symbols, for profiling
  ]

if system_os == "darwin"
  swig_sys_args = '-D_DARWIN_C_SOURCE'
elif system_os == "linux" or system_os == "freebsd"
  # Look up libraries dependencies relative to the library.
  swig_sys_args = ['-Wl,-z,origin']
  swig_sys_args = ['-Wl,-rpath,$ORIGIN']
endif

### Run swig
r = run_command(swig, ['-version'], check: true) # Returns: "\nSWIG Version 4.1.1\n\nCompiled with ..."
swig_version = r.stdout().split('\n')[1].split()[2].strip()
if swig_version.version_compare('<4.1.0')
  swig_cmd = [swig, '-python', '-py3', '-o', '@OUTPUT1@', '@INPUT0@']
else
  swig_cmd = [swig, '-python', '-o', '@OUTPUT1@', '@INPUT0@']
endif

mypaintlib_swig = custom_target(
  'mypaintlib.py',
  input: ['lib/mypaintlib.i',
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
    'lib/fill/morphology.cpp'
  ],
  output:  ['mypaintlib.py', 'lib/mypaintlib_wrap.cpp'],
  command: swig_cmd,
  install: true,
  install_dir: [py.get_install_dir(pure: false, subdir: 'libnvme'), false],
  )

mypaintlib_clib = py.extension_module(
  '_mypaintlib',
  mypaintlib_swig[1],
  dependencies : [python_dep],
  include_directories: [incdir, internal_incdir],
  link_with: [libccan],
  install: true,
  subdir: 'libnvme',
  )


### Swig Flags
# mypaintlib_swig_opts = ['-Wall', '-noproxydel', '-c++']
# mypaintlib_swig_opts.extend([
#   "-I" + d
#   for d in mypaintlib_opts["include_dirs"]
# ])
# # FIXME: building against the new shared lib, omit old test code
# mypaintlib_swig_opts.extend(['-DNO_TESTS'])

## Extensions

## Options
# swig_opts=mypaintlib_swig_opts,
# **mypaintlib_opts

# Install
install_subdir('src/gui', install_dir : 'mypaint')
