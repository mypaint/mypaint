#!/usr/bin/env python
#
# References:
# http://library.gnome.org/devel/integration-guide/stable/thumbnailer.html.en
# http://create.freedesktop.org/wiki/OpenRaster/File_Layout_Specification


import sys, os, zipfile, tempfile
from gtk import gdk

def get_pixbuf(png_data_str):
    """Return a gdk.Pixbuf with the contents of png_data_str"""
    # We create a temporary file in the filesystem because gdk.Pixbuf
    # does not let us get a pixbuf of the png file in an easy way without it
    temp_dir = tempfile.mkdtemp()
    temp_path = temp_dir + "thumb.png"

    temp_file = open(temp_path, "w")
    temp_file.write(png_data_str)
    temp_file.close()

    pixbuf = gdk.pixbuf_new_from_file(temp_path)
    os.remove(temp_path)
    os.rmdir(temp_dir)

    return pixbuf

def output_thumbnail(pixbuf, output_path, max_size):
    """Output a thumbnail of pixbuf to output_path.
    The thumbnail will be scaled so that neither dimensions exceeds max_size"""
    orig_w, orig_h = pixbuf.get_width(), pixbuf.get_height()

    if orig_w < max_size and orig_h < max_size:
        # No need for scaling
        pixbuf.save(outfile, 'png')
        return

    scale_factor = float(max_size)/max(orig_w, orig_h)
    new_w, new_h = int(orig_w*scale_factor), int(orig_h*scale_factor)
    scaled_pixbuf = pixbuf.scale_simple(new_w, new_h, gdk.INTERP_BILINEAR)
    scaled_pixbuf.save(outfile, 'png')

if __name__ == '__main__':
    try:
        progname, infile, outfile, size = sys.argv
    except ValueError:
        sys.exit('Usage: %s <Input> <Output> <Size>' % sys.argv[0])

    png_data = zipfile.ZipFile(infile).read('Thumbnails/thumbnail.png')
    pixbuf = get_pixbuf(png_data)
    output_thumbnail(pixbuf, outfile, int(size))

