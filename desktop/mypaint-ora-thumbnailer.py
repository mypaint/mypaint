# Thumbnailer for GNOME/Cinnamon Nautilus, and compatible desktops.
#
# Copyright (c)  2010 Jon Nordby <jononor@gmail.com>
#           (c)  2013 Andrew Chadwick <a.t.chadwick@gmail.com>
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
from gtk import gdk


def ora_thumbnail(infile, outfile, size):
    """Saves an OpenRaster file's thumbnail to a PNG file, with scaling.
    """

    # Extract a GdkPixbuf from the OpenRaster file
    png_data = zipfile.ZipFile(infile).read('Thumbnails/thumbnail.png')
    loader = gdk.PixbufLoader()
    loader.write(png_data)
    loader.close()
    pixbuf = loader.get_pixbuf()

    # Don't scale if not needed
    orig_w, orig_h = pixbuf.get_width(), pixbuf.get_height()
    if orig_w < size and orig_h < size:
        pixbuf.save(outfile, 'png')
        return

    # Scale and save
    scale_factor = float(size) / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
    scaled_pixbuf = pixbuf.scale_simple(new_w, new_h, gdk.INTERP_BILINEAR)
    scaled_pixbuf.save(outfile, 'png')


if __name__ == '__main__':
    import sys
    try:
        progname, infile, outfile, size = sys.argv
    except ValueError:
        sys.exit('Usage: %s <Input> <Output> <Size>' % sys.argv[0])
    ora_thumbnail(infile, outfile, int(size))

