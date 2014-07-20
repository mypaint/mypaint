MyPaint
=======

MyPaint is a simple drawing and painting program that works well with Wacom-style graphics tablets. Its primary features are its highly configurable brush engine, speed, and a fullscreen mode which allows artists to fully immerse themselves in their work.

* Website: [mypaint.info](http://mypaint.info/)
* Twitter: [@MyPaintApp](https://twitter.com/MyPaintApp)
* Github:
  - [Development "master" branch](https://github.com/mypaint/mypaint)
  - [New issue tracker](https://github.com/mypaint/mypaint/issues)
* Other resources:
  - [Mailing list](https://mail.gna.org/listinfo/mypaint-discuss)
  - [Wiki](http://wiki.mypaint.info/)
  - [Forums](http://forum.intilinux.com/)
  - [Old bug tracker](http://gna.org/bugs/?group=mypaint) (patience please: we're migrating bugs across)
  - [Introductory docs for developers](http://wiki.mypaint.info/index.php?title=Documentation/ForDevelopers)

MyPaint is written in Python, C++, and C, and it makes use of the GTK toolkit, version 3.x. The source is maintained using [git](http://www.git-scm.com), primarily on Github.

Getting started
===============

MyPaint has an associated library, [libmypaint](https://github.com/mypaint/libmypaint), which is distributed as a sister project on Github. If you pull the application's source, this dependency will be fetched automatically by the commands below as a relative [submodule](http://www.git-scm.com/book/en/Git-Tools-Submodules). There are several third-party dependencies too:

- scons (>= 2.1.0)
- pygobject
- gtk3
- python (= 2.7)
- swig
- numpy
- pycairo (>= 1.4)
- libpng
- lcms2
- libjson-c (>= 0.11, but the older "libjson" name at ~0.10 will work too)
- librsvg

Recommended: a pressure sensitive input device (graphic tablet)

Build & Install (Linux)
-----------------------

* **Install dependencies**: if you run Debian GNU/Linux or one of its derivatives like Linux Mint or Ubuntu, you can fetch the dependencies by running:

  ```sh
  $ sudo apt-get install g++ python-dev python-numpy \
    libgtk-3-dev python-gi-dev gir1.2-gtk-3.0 python-gi-cairo \ 
    swig scons gettext libpng12-dev liblcms2-dev libjson0-dev
  ```

* **Fetch the source**: start by cloning the source repository. This will create a directory named "mypaint". You should only need to do this initial step once.

  ```sh
  $ git clone https://github.com/mypaint/mypaint.git
  ```

* **Update submodules**: change into your cloned repository folder, and then update the "brushlib" submodule so that it contains _libmypaint_ at the correct version:

  ```sh
  $ cd mypaint
  $ git submodule update --init --force
  ```

* **Build & test**: starting from your cloned repository folder, run _scons_ to compile the C++ and C parts.  If the build was successful, run the generated script with a fresh throwaway configuration directory in order to test that the program works.

  ```sh
  $ scons
  $ ./mypaint -c /tmp/mypaint_cfgtmp_$$
  ```

* **Updating to the latest source** at a later date is trivial, but often requires the submodule to be bumped up to a new version, or a rebuild:

  ```sh
  $ cd path/to/mypaint
  $ scons --clean
  $ git pull
  $ git submodule update --init --force
  $ scons
  $ [... other commands as necessary ...]
  ```

* **To install** into the traditional `/usr/local` area so that it can be run from your desktop environment:

  ```sh
  $ cd path/to/mypaint
  $ sudo scons prefix=/usr/local install
  ```

* **To uninstall** the program from a given prefix, add the `--clean` option:

  ```sh
  $ sudo scons prefix=/usr/local install --clean
  ```


Build & Install (Windows)
-------------------------

TBD. [Please help us write this section](https://github.com/mypaint/mypaint/issues/48).

Starting point for up-to-date information: http://wiki.mypaint.info/Development/Packaging#Windows


Build & Install (Mac)
---------------------

TBD. [Please help us write this section](https://github.com/mypaint/mypaint/issues/49).

Starting point for up-to-date information: http://wiki.mypaint.info/Development/Packaging#OSX


Post-install: Linux
-------------------

* **(Advanced) people creating packages** for Linux distributions can install as if the prefix were /usr, but install the tree somewhere else. This can be done as an ordinary user.

  ```sh
  $ scons prefix=/usr --install-sandbox=`pwd`/path/to/sandbox
  ```

  **NOTE:** the sandbox location must be located under the current working directory, and be specified as an *absolute* path. Using `pwd` or your build environment's absolute path-to-here variable should achieve that. The above installs the main launch script (for example) as `./path/to/sandbox/usr/bin/mypaint`.  Use a symlink if that's too limiting.

* **(Troubleshooting) runtime linker**: you may need to update ld.so's caches and links after installation on some systems.

  ```sh
  $ sudo ldconfig
  ```

  Do this if you get any messages about MyPaint not being able to load `mypaintlib.so` when run on the command line.

  If you installed to a prefix other than the trusted locations, `/usr/lib` and `/lib`, you may need to add a line for it into `/etc/ld.so.conf` or `ld.so.conf.d` before running `ldconfig`. Scons won't do this for you because the need to perform the action varies by distribution, and package distributors need to be able to defer it to post-installation scripting.

* **(Troubleshooting) icon theme caches**: take care to update the icon theme cache for your prefix if you're installing mypaint to a location which has one of these files already. If you install new icons, any existing icon cache must be updated too, otherwise MyPaint won't be able to find its icons even if it looks in the right place. For example for an install into `/usr` (which has one on most systems), you should run:

  ```sh
  $ sudo gtk-update-icon-cache /usr/share/icons/hicolor
  $ sudo chmod a+r /usr/share/icons/hicolor/icon-theme.cache
  ```

  after installation to ensure that the cache is up to date. Scons won't do this for you because the cache file is optional.

  If you install to /usr/local, you may need to run this instead:

  ```sh
  $ gtk-update-icon-cache --ignore-theme-index /usr/local/share/icons/hicolor
  ```

Legal info
==========

The licenses vor various files are described in the file LICENSE.
Documentation can be found within the program and on the homepage:
http://mypaint.info/

A list of contributors can be found in the about dialog.
