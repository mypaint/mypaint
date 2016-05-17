## MyPaint

[![Translation Status](https://hosted.weblate.org/widgets/mypaint/mypaint/svg-badge.svg)](https://hosted.weblate.org/engage/mypaint/?utm_source=widget)
[![Build Status](https://travis-ci.org/mypaint/mypaint.png?branch=master)](https://travis-ci.org/mypaint/mypaint)

MyPaint is a simple drawing and painting program
that works well with Wacom-style graphics tablets.
Its main features are a highly configurable brush engine, speed,
and a fullscreen mode which allows artists to
fully immerse themselves in their work.

* Website: [mypaint.org](http://mypaint.org/)
* Twitter: [@MyPaintApp](https://twitter.com/MyPaintApp)
* Github:
  - [Development "master" branch](https://github.com/mypaint/mypaint)
  - [New issue tracker](https://github.com/mypaint/mypaint/issues)
* Other resources:
  - [Mailing list](https://mail.gna.org/listinfo/mypaint-discuss)
  - [Wiki](https://github.com/mypaint/mypaint/wiki)
  - [Forums](http://forum.intilinux.com/)
  - [Old bug tracker](http://gna.org/bugs/?group=mypaint)
    (patience please: we're migrating bugs across)
  - [Introductory docs for developers](https://github.com/mypaint/mypaint/wiki/Development)

MyPaint is written in Python, C++, and C.
It makes use of the GTK toolkit, version 3.x.
The source is maintained using [git](http://www.git-scm.com),
primarily on Github.

### Getting started

MyPaint has an associated library,
[libmypaint](https://github.com/mypaint/libmypaint),
which is distributed as a sister project on Github.

- libmypaint (>= 1.3.0-alpha.0)

There are several third-party dependencies too:

- scons (>= 2.1.0)
- pygobject
- gtk3 (>= 3.12)
- python (= 2.7) (OSX: python >= 2.7.4)
- swig
- numpy
- pycairo (>= 1.4)
- libpng
- lcms2
- libjson-c (>= 0.11, but the older "libjson" name at ~0.10 will work too)
- librsvg

Recommended: a pressure sensitive input device (graphic tablet)

### Build and Install

All systems differ.
The basic build documentation is divided by
broad class of operating system and software distribution.

* [README\_LINUX.md (chiefly Debian-based systems)](README_LINUX.md)
* [README\_WINDOWS.md (native WIN32/WIN64 using MSYS2)](README_WINDOWS.md)
* [README\_OSX.md (macports - needs review)](README_OSX.md)

### Contributing

The MyPaint project welcomes and encourages participation by everyone.
We want our community to be skilled and diverse,
and we want it to be a community that anybody can feel good about joining.
No matter who you are or what your background is, we welcome you.

Please note that MyPaint is released with a
[Contributor Code of Conduct](CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.

Please see the file [CONTRIBUTING.md](CONTRIBUTING.md)
for details of how you can begin contributing.

### Legal info

MyPaint is Free/Libre/Open Source software.
See [Licenses.md](Licenses.md) for a summary of its licensing.
A list of contributors can be found in the about dialog.
