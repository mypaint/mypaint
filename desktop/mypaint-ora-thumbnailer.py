#!/usr/bin/env python2
# Thumbnailer for GNOME/Cinnamon Nautilus, and compatible desktops.
#
# Copyright (c)  2010 Jon Nordby <jononor@gmail.com>
#           (c)  2013-2017 the MyPaint Development Team
# This program is distributed under the same terms as MyPaint itself.

# OpenRaster specification:
# http://freedesktop.org/wiki/Specifications/OpenRaster/Draft/FileLayout
#
# This can be used with earlier versions of GNOME by setting some gconf keys:
# http://library.gnome.org/devel/integration-guide/stable/thumbnailer.html.en
# However the MyPaint distribution does not do this in order to avoid a
# dependency on gconf.
#
# GNOME 3.0 thumbnailing requires a .thumbnailer file under $XDG_DATA_DIRS,
# subfolder "thumbnailers".
# http://www.ict.griffith.edu.au/anthony/info/X/Thumbnailing.txt Since this
# just requires files to be installed, we support that (and assume your
# installation prefix is one of the $XDG_DATA_DIRS:
# http://standards.freedesktop.org/basedir-spec/ )
#
# You will also need to tell your desktop environment about the
# image/openraster MIME type. Support is available in Debian-derived OSes in
# the shared-mime-info package, or see
# http://standards.freedesktop.org/shared-mime-info-spec/

import zipfile

import gi


gi.require_version("GdkPixbuf", "2.0")
try:
    from gi.repository import GdkPixbuf
except:
    raise


def ora_thumbnail(infile, outfile, size):
    """Extracts an OpenRaster file's thumbnail to PNG, with scaling."""

    # Extract a GdkPixbuf from the OpenRaster file
    with zipfile.ZipFile(infile) as zf:
        png_data = zf.read('Thumbnails/thumbnail.png')
    loader = GdkPixbuf.PixbufLoader()
    loader.write(png_data)
    loader.close()
    pixbuf = loader.get_pixbuf()

    # Scale if needed
    orig_w = pixbuf.get_width()
    orig_h = pixbuf.get_height()
    if orig_w > size or orig_h > size:
        scale_factor = float(size) / max(orig_w, orig_h)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        pixbuf = pixbuf.scale_simple(
            new_w, new_h,
            GdkPixbuf.InterpType.BILINEAR,
        )

    # Save. The output file is a temporary one created by
    # GNOME::ThumbnailFactory, which overwrites any options we add
    # with its own Thumb::MTime and Thumb::URI.
    pixbuf.savev(outfile, "png", [], [])
    # Hopefully the method name won't change for posix typelibs.


if __name__ == '__main__':
    import sys
    try:
        progname, infile, outfile, size = sys.argv
    except ValueError:
        sys.exit('Usage: %s <Input> <Output> <Size>' % sys.argv[0])
    ora_thumbnail(infile, outfile, int(size))
