@echo off
cd %~dp0\..
del /f/q lib\gdk-pixbuf-2.0\2.10.0\loaders.cache
del /f/q share\glib-2.0\schemas
