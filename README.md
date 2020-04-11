<p align="center">
    <img src="pixmaps/mypaint_logo.png?raw=true" height="100px"/>
    <h1 align="center">MyPaint</h1>
    <h4 align="center">
      A fast and dead-simple painting app for artists
    </h4>
  <br>
</p>

[![Financial Contributors on Open Collective](https://opencollective.com/mypaint/all/badge.svg?label=financial+contributors)](https://opencollective.com/mypaint) [![Translation status](https://hosted.weblate.org/widgets/mypaint/-/mypaint/svg-badge.svg)](https://hosted.weblate.org/engage/mypaint/?utm_source=widget) [![Build status on Travis](https://travis-ci.org/mypaint/mypaint.svg?branch=master)](https://travis-ci.org/mypaint/mypaint) [![AppVeyor](https://ci.appveyor.com/api/projects/status/3s54192cipo2d4js/branch/master?svg=true)](https://ci.appveyor.com/project/achadwick/mypaint/branch/master) [![Packaging status](https://repology.org/badge/tiny-repos/mypaint.svg)](https://repology.org/project/mypaint/versions)


## Features

* Infinite canvas
* Extremely configurable brushes
* Distraction-free fullscreen mode
* Extensive graphic tablet support
* Speed, simplicity, and expressiveness
* Realistic paint-like pigment model
* 15 bit Rec 709 linear RGB colorspace
* Brush settings stored with each stroke on the canvas
* Layers, various modes, and layer groups

## Download

[Latest stable release.](https://github.com/mypaint/mypaint/releases/latest)

Releases and prereleases contain links to standalone packages
(and installers for full releases) for Windows, and AppImage files for Linux.

### Nightly releases

_Using alpha releases comes with its own risks.
Sometimes bugs can sneak in that causes crashes, so don't be too surprised by that.
If you come across any, please do [report those bugs][trackerlink] so they can be dealt with._

**Linux**

If you don't want to [build from source](#buildtestinstall),
the latest AppImage files can be found in a
[rolling release](https://github.com/mypaint/mypaint-appimage/releases/tag/continuous).
Just download the `.AppImage` file and make it executable.

**Windows**

The nightly installers and standalone archives can be downloaded from the
[AppVeyor CI](https://ci.appveyor.com/project/achadwick/mypaint)

Click on the link matching your architecture (32 or 64), then the tab named "Artifacts"
to get the file list. Only one of the files ending in `.exe` or `.7z` are needed.
Using the standalone archive (7z) is recommended.

### Chocolatey (windows)

If you prefer to use the Chocolatey repository, both
[stable releases][choco_prerel] and [pre-releases][choco_stable]
can be found there.

[choco_prerel]: https://chocolatey.org/packages/mypaint/
[choco_stable]: https://chocolatey.org/packages/mypaint/1.2.1

## Build/Test/Install

MyPaint depends on its brushstroke rendering library,
[libmypaint](https://github.com/mypaint/libmypaint), as well as
its brush library [mypaint-brushes](https://github.com/mypaint/mypaint-brushes).
If you have those installed, plus MyPaint's third party dependencies,
you can try it out without installing:

    git clone https://github.com/mypaint/mypaint.git
    cd mypaint
    python setup.py demo

If the demo works, you can install

    python setup.py managed_install
    python setup.py managed_uninstall

For more details, see the [Setup Instructions](BUILDING.md).

[1]:https://github.com/mypaint/libmypaint

## Contributing

The MyPaint project welcomes and encourages participation by everyone. We want our community to be skilled and diverse, and we want it to be a community that anybody can feel good about joining. No matter who you are or what your background is, we welcome you.

Please see the [Contributing Guide](CONTRIBUTING.md) for full details of how you can begin contributing.  All contributors to the MyPaint project must abide by a [Code of Conduct](CODE_OF_CONDUCT.md).

## Community

* Website: [mypaint.org](http://mypaint.org/)
* Twitter: [@MyPaintApp](https://twitter.com/MyPaintApp)
* Discord: [MyPaint](https://discord.gg/vbB434p)
* Github:
  - [Development "master" branch](https://github.com/mypaint/mypaint)
  - [Issue tracker][trackerlink]
* Other resources:
  - [Wiki](https://github.com/mypaint/mypaint/wiki)
  - [Community Forums](https://community.mypaint.org)
  - [Introductory docs for developers](https://github.com/mypaint/mypaint/wiki/Development)

## Legal info

MyPaint is Free/Libre/Open Source software.  See [Licenses and
Copyrights](Licenses.md) for a summary of its licensing.  A list of
contributors can be found in the about dialog.

[trackerlink]: https://github.com/mypaint/mypaint/issues

## Contributors

### Code Contributors

This project exists thanks to all the people who contribute. [[Contribute](CONTRIBUTING.md)].
<a href="https://github.com/mypaint/mypaint/graphs/contributors"><img src="https://opencollective.com/mypaint/contributors.svg?width=890&button=false" /></a>

### Financial Contributors

Become a financial contributor and help us sustain our community. [[Contribute](https://opencollective.com/mypaint/contribute)]

#### Individuals

<a href="https://opencollective.com/mypaint"><img src="https://opencollective.com/mypaint/individuals.svg?width=890"></a>

#### Organizations

Support this project with your organization. Your logo will show up here with a link to your website. [[Contribute](https://opencollective.com/mypaint/contribute)]

<a href="https://opencollective.com/mypaint/organization/0/website"><img src="https://opencollective.com/mypaint/organization/0/avatar.svg"></a>
<a href="https://opencollective.com/mypaint/organization/1/website"><img src="https://opencollective.com/mypaint/organization/1/avatar.svg"></a>
<a href="https://opencollective.com/mypaint/organization/2/website"><img src="https://opencollective.com/mypaint/organization/2/avatar.svg"></a>
<a href="https://opencollective.com/mypaint/organization/3/website"><img src="https://opencollective.com/mypaint/organization/3/avatar.svg"></a>
<a href="https://opencollective.com/mypaint/organization/4/website"><img src="https://opencollective.com/mypaint/organization/4/avatar.svg"></a>
<a href="https://opencollective.com/mypaint/organization/5/website"><img src="https://opencollective.com/mypaint/organization/5/avatar.svg"></a>
<a href="https://opencollective.com/mypaint/organization/6/website"><img src="https://opencollective.com/mypaint/organization/6/avatar.svg"></a>
<a href="https://opencollective.com/mypaint/organization/7/website"><img src="https://opencollective.com/mypaint/organization/7/avatar.svg"></a>
<a href="https://opencollective.com/mypaint/organization/8/website"><img src="https://opencollective.com/mypaint/organization/8/avatar.svg"></a>
<a href="https://opencollective.com/mypaint/organization/9/website"><img src="https://opencollective.com/mypaint/organization/9/avatar.svg"></a>
