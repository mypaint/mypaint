Building and Installing MyPaint on Linux
========================================

Your operating system distribution should have a stable version of
MyPaint available already. These instructions are for developers and
testers who want the bleeding edge.

Build
-----

* **Install dependencies**: if you run Debian GNU/Linux
  or one of its derivatives like Linux Mint or Ubuntu,
  you can fetch the dependencies by running:

  ```sh
  sudo apt-get install g++ python-dev python-numpy \
  libgtk-3-dev python-gi-dev gir1.2-gtk-3.0 python-gi-cairo \
  swig scons gettext libpng12-dev liblcms2-dev libjson-c-dev
  ```
  **Note**: Running `sudo apt-get build-dep mypaint` will install
  most (if not all) of the dependencies for you.

* **Fetch the source**: start by cloning the source repository.
  This will create a directory named "mypaint".
  You should only need to do this initial step once.

  ```sh
  git clone https://github.com/mypaint/mypaint.git
  ```

* **Update submodules**: change into your cloned repository folder,
  and then update the "brushlib" submodule
  so that it contains _libmypaint_ at the correct version:

  ```sh
  cd mypaint
  git submodule update --init --force
  ```

* **Build & test**: starting from your cloned repository folder,
  run _scons_ to compile the C++ and C parts.

  ```sh
  scons
  ```

* **Testing (interactive)**: if the build was successful,
  run the generated script with a clean temporary configuration area
  in order to test that the program works.

  ```sh
  rm -fr /tmp/mypaint_testconfig
  ./mypaint -c /tmp/mypaint_testconfig
  ```

* **Unit tests**: These are purely optional for most users,
  but they're useful for developers and people reporting bugs.
  Please run the unit tests before committing new code,
  and implement doctests for important new Python code.

  ```sh
  sudo apt-get install python-nose
  nosetests --with-doctest
  ```

  - If testing outside a graphical environment (anywhere Gdk refuses
    to initialize), limit the doctests to just `lib/` and `brushlib/`.
  - There are several interactive GUI tests in the `tests/` folder
    which `nosetests` does not run - quite intentionally -
    because their executable bit is set.

* **Updating to the latest source** at a later date is trivial,
  but doing this often means that you have to update the submodule
  or rebuild the compiled parts of the app:

  ```sh
  cd path/to/mypaint
  scons --clean
  git pull
  git submodule update --init --force
  scons
  # ... other commands as necessary ...
  ```

Installing the build
--------------------

* **To install** MyPaint into the traditional `/usr/local` area
  so that it can be run from your desktop environment:

  ```sh
  cd path/to/mypaint
  sudo scons prefix=/usr/local install
  ```

  - This usually results in entries in menus, launchers, Dashes
    and other desktop environment frippery.

* **To uninstall** the program from a given prefix,
  add the `--clean` option:

  ```sh
  sudo scons prefix=/usr/local install --clean
  ```

Post-install
------------

* **(Advanced) people creating packages** for Linux distributions
  can install as if the prefix were /usr,
  but install the tree somewhere else.
  This can be done as an ordinary user.

  ```sh
  scons prefix=/usr --install-sandbox=`pwd`/path/to/sandbox
  ```

  **NOTE:** the sandbox location must be located under
  the current working directory, and be specified as an *absolute* path.
  You can use ``pwd`` or your build system's absolute
  "path-to-here" variable to achieve that.
  The command above installs the main launch script (for example)
  as `./path/to/sandbox/usr/bin/mypaint`.
  Use a symlink if that's too limiting.

* **(Troubleshooting) runtime linker**: you may need to update
  the runtime linker's caches and links
  after installation on some systems.

  ```sh
  sudo ldconfig
  ```

  Do this if you get any messages about MyPaint
  not being able to load `mypaintlib.so` when run on the command line.

  If you installed to a prefix other than the trusted locations,
  which are typically `/usr/lib` and `/lib`,
  you may need to add a line for your prefix
  into `/etc/ld.so.conf` or `ld.so.conf.d`
  before running `ldconfig`.

  Scons currently won't do this for you because
  the need to perform the action varies by distribution,
  and package distributors need to be able to defer it
  to post-installation scripting.

* **(Troubleshooting) icon theme caches**: take care to update
  the icon theme cache for your prefix
  if you're installing mypaint to a location
  which has one of these files already.
  If you install new icons, any existing icon cache must be updated too,
  otherwise MyPaint won't be able to find its icons
  even if it looks in the right place.

  For example for an install into `/usr`,
  which has an icon cache on most systems,
  you should run:

  ```sh
  sudo gtk-update-icon-cache /usr/share/icons/hicolor
  sudo chmod a+r /usr/share/icons/hicolor/icon-theme.cache
  ```

  after installation to ensure that the cache is up to date.
  Scons currently won't do this for you
  because the cache file is optional.

  If you install to /usr/local, you may need to run this instead:

  ```sh
  gtk-update-icon-cache --ignore-theme-index /usr/local/share/icons/hicolor
  ```
