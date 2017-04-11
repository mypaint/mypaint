# MyPaint [![Translation Status](https://hosted.weblate.org/widgets/mypaint/mypaint/svg-badge.svg)](https://hosted.weblate.org/engage/mypaint/?utm_source=widget) [![Travis Build Status](https://travis-ci.org/mypaint/mypaint.png?branch=master)](https://travis-ci.org/mypaint/mypaint) [![Tea-CI Build Status](https://tea-ci.org/api/badges/mypaint/mypaint/status.svg)](https://tea-ci.org/mypaint/mypaint)

MyPaint is a fast and dead-simple painting app for artists. It works
with many graphics tablets without any special configuration. You can
tweak its brushes to work like all kinds of real media. The full-screen
mode allows you to work without distractions.

## Features

* Infinite canvas
* Distraction-free fullscreen mode
* Extremely configurable brushes
* Speed, simplicity, and expressiveness

# Build/Test/Install

MyPaint depends on its brushstroke rendering library,
[libmypaint](https://github.com/mypaint/libmypaint).
If you have that installed, plus MyPaint's third party dependencies,
you can try it out without installing:

    git clone https://github.com/mypaint/mypaint.git
    cd mypaint
    python setup.py demo

If the demo works, you can install

    python setup.py managed_install
    python setup.py managed_uninstall

For more details, see the [Setup Instructions](BUILDING.md).

[1]:https://github.com/mypaint/libmypaint

# Contributing

The MyPaint project welcomes and encourages participation by everyone.
We want our community to be skilled and diverse, and we want it to be
a community that anybody can feel good about joining. No matter who you are
or what your background is, we welcome you.

Please see the [Contributing Guide](CONTRIBUTING.md) for full details of
how you can begin contributing.  All contributors to the MyPaint
project must abide by a [Code of Conduct](CODE_OF_CONDUCT.md).

## Community

* Website: [mypaint.org](http://mypaint.org/)
* Twitter: [@MyPaintApp](https://twitter.com/MyPaintApp)
* Github:
  - [Development "master" branch](https://github.com/mypaint/mypaint)
  - [Issue tracker](https://github.com/mypaint/mypaint/issues)
* Other resources:
  - [Wiki](https://github.com/mypaint/mypaint/wiki)
  - [Community Forums](https://community.mypaint.org)
  - [Introductory docs for developers](https://github.com/mypaint/mypaint/wiki/Development)

## Legal info

MyPaint is Free/Libre/Open Source software.  See [Licenses and
Copyrights](Licenses.md) for a summary of its licensing.  A list of
contributors can be found in the about dialog.
