#!/bin/sh
# Interactively install runtime dependencies.
# User-oriented convenience script for MSYS2 installations.
# This shouldn't be used as part of anybody's build scripting.

cat >&2 <<"__END__"
+++ About to install build deps for MyPaint

This script does the following on your behalf. Please read through the
list and press Return if you're happy. You will be able to uninstall
everything that gets installed with "pacman" afterwards. Only the files
in your MSYS2 installation will be affected.

This helper script needs your approval to...

1. Download and install Open Source software from the MSYS2 repositories:
   - all the runtime dependencies needed by MyPaint
   - all the necessary developer tools needed to build MyPaint
2. Download, build, and install Open Source software from GitHub:
   - libmypaint (MyPaint's C brush engine)

+++ Press Return to accept these changes, or Ctrl+C to quit now
__END__

read OK

echo >&2 "+++ Updating pacman cache, and installing dependencies"
pacman -Sy --noconfirm
windows/msys2-build.sh installdeps

echo >&2 "+++ Refreshing loaders.cache to solve missing SVG icons"
gdk-pixbuf-query-loaders --update-cache

cat >&2 <<__END__
+++ All done!

Dependencies installed for MSYSTEM=$MSYSTEM.
You should now be able to run the following commands to build and
test MyPaint from within this folder.

    python setup.py build
    python setup.py nosetests
    python setup.py test
    python setup.py clean --all

    python setup.py demo

+++ Happy developing!
__END__
