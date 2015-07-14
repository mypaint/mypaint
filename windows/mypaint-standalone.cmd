@echo off

cd %~dp0

set PATH=%CD%\mingw32\bin;%PATH%

rem Need to do this so that GdkPixbuf can find the rsvg loader.
rem It's really a postinst action though.
gdk-pixbuf-query-loaders > mingw32\lib\gdk-pixbuf-2.0\2.10.0\loaders.cache

rem GSettings schemas
glib-compile-schemas mingw32\share\glib-2.0\schemas

rem And, go!
rem We don't use -B or -O because that breaks ntpath.
start python2w mingw32\bin\mypaint
