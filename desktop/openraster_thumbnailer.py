#!/usr/bin/env python
#
# Installation hack for nautilus (adapt the path as needed):
#
#   gconftool-2 --set /desktop/gnome/thumbnailers/image@openraster/command -t string "/usr/local/share/mypaint/desktop/openraster_thumbnailer.py %i %o %s"
#   gconftool-2 --set /desktop/gnome/thumbnailers/image@openraster/enable -t boolean "True"
# 
#   mkdir -p ~/.local/share/mime/packages
#   cp mime/mypaint.xml ~/.local/share/mime/packages/
#   update-mime-database ~/.local/share/mime
#
# ... and enjoy .ora thumbnails :-)
#
# References:
# http://library.gnome.org/devel/integration-guide/stable/thumbnailer.html.en
# http://create.freedesktop.org/wiki/OpenRaster/File_Layout_Specification


import sys, zipfile
import Image
import StringIO

if len(sys.argv) != 4:
    sys.exit('Usage: '+sys.argv[0]+' <Input> <Output> <Size>')

thumbnail = zipfile.ZipFile(sys.argv[1]).read('Thumbnails/thumbnail.png')

im = Image.open(StringIO.StringIO(thumbnail))
im.thumbnail( (int(sys.argv[3]), int(sys.argv[3])) )

im.save(sys.argv[2], 'png')

