_:warning: These files are deprecated_

These files made up the old appimage build.
The new build system can be found [here][APPIMG_REPO].

[APPIMG_REPO]: https://github.com/mypaint/mypaint-appimage/

The only relevant file here is the `trigger_build.sh` script
that triggers the building of a new appimage as the final
step in a successful travis build. These appimages are uploaded
to the release tagged `continuous` in that same repo.
