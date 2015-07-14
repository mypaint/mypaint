@echo off
cd %~dp0\..
bin\gdk-pixbuf-query-loaders > lib\gdk-pixbuf-2.0\2.10.0\loaders.cache
bin\glib-compile-schemas share\glib-2.0\schemas
