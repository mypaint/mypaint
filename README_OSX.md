Building MyPaint on OS X
========================

**IN PROGRESS**: Please help us improve this section.

Most users will want to grab `MyPaint-devel`
[from macports](https://www.macports.org/ports.php?by=name&substr=mypaint)
or stick with the stable `MyPaint` portfile already there.
Please report problems with the port via the MacPorts bug tracker
unless you're certain it's a MyPaint fault.
[Here's how](https://guide.macports.org/#project.tickets)!

For the adventurous, the following is reported to work on OS X 10.9:

* **Environment setup**: to use Frameworks Python (currently 2.7.8)
  while satisfying the other dependencies from Macports, use

  ```sh
  export PKG_CONFIG_PATH=/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/pkgconfig/
  export CFLAGS="-I/opt/local/include"
  ```

* **Install dependencies**:

  ```sh
  sudo port install gtk3
  sudo port install json-c
  sudo port install py27-numpy
  sudo port install py27-scipy
  sudo port install py27-pyobjc-cocoa    # optional, for i18n support
  sudo port install lcms2
  sudo port install py27-gobject3
  sudo port install hicolor-icon-theme
  ```

  You need to install [libmypaint](https://github.com/mypaint/libmypaint) too.
  At the time of writing, this is not available in MacPorts.
  Follow the instructions in libmypaint's
  [README.md](https://github.com/mypaint/libmypaint/blob/master/README.md)
  to install it.

* **Fetch source** just as for Linux:

  ```sh
  git clone https://github.com/mypaint/mypaint.git
  cd mypaint
  ```

* **Build and test**.
  The `sudo -E` should not be necessary - let us know if it isn't,
  or if it's unhelpful when running the build in test mode.

  ```sh
  sudo -E scons
  ./mypaint -c /tmp/mypaint_cfgtmp_$$
  ```
