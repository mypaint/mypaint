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
  swig scons gettext libpng-dev liblcms2-dev libjson-c-dev
  ```
  **NOTE:** These are the package names
  which are used by the current Debian testing/unstable
  as of 2016-04-15.
  Some of the names have changed over time.
  If you have problems, try one of the following older names:
  `libjson0-dev`, `libpng12-dev`.

* **Install libmypaint**: we will make testing packages of libmypaint
  available for via the PPA.
  A package may also be in Debian, Ubuntu, or your derivative
  at the time you read this. If you have one of those systems,
  try this first:

  ```sh
  sudo apt-get install libmypaint-dev
  ```

  If that doesn't work, you will need to build libmypaint from git
  and install it. Follow the generic instructions here:
  https://github.com/mypaint/libmypaint/blob/master/README.md

  Alternatively, if you have a Debian/Ubuntu/derivative system,
  please try https://github.com/mypaint/libmypaint.deb

* **Fetch the source**: start by cloning the source repository.
  This will create a directory named "mypaint".
  You should only need to do this initial step once.

  ```sh
  git clone https://github.com/mypaint/mypaint.git
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
    to initialize), limit the doctests to just `lib/`.
  - There are several interactive GUI tests in the `tests/` folder
    which `nosetests` does not run - quite intentionally -
    because their executable bit is set.

* **Updating to the latest source** at a later date is trivial,
  but doing this often means that you have to
  rebuild the compiled parts of the app:

  ```sh
  cd path/to/mypaint
  scons --clean
  git pull
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

  - You *must* specify the `install` target.

  - This usually results in entries in menus, launchers, Dashes
    and other desktop environment frippery.

* **To uninstall** the program from a given prefix,
  add the `--clean` option:

  ```sh
  sudo scons prefix=/usr/local install --clean
  ```

You need to add the `install` target to these commands
if they are to do anything.
It's an alias for the current value of `prefix`,
and SCons always needs to be told what to build under.

The `prefix` path must be an absolute path,
but it doesn't have to exist.
It will be created if an install is requested.

Post-install
------------

* **(Advanced) people creating packages** for Linux distributions
  can install as if the prefix were /usr,
  but install the tree somewhere else.
  This can be done as an ordinary user.

  ```sh
  scons prefix=/usr --install-sandbox=/path/to/sandbox /path/to/sandbox
  ```

  **NOTE:** the sandbox location must
  be specified as an *absolute* path too.
  Note the need to repeat the sandbox path:
  remember, SCons always needs to be told where to put its stuff.

  The command above installs the main launch script (for example)
  as `/path/to/sandbox/usr/bin/mypaint`.
  Use a symlink if that's too limiting.

  The prefix must still be syntactically an absolute path,
  but it doesn't have to exist within the sandbox.
  Both the sandbox and the prefix within it will be created as needed.

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
