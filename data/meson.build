# This file is part of MyPaint.
# Copyright (C) 2024 the MyPaint project
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# /data/meson.build

install_subdir('desktop/icons', install_dir : datadir + '/icons', strip_directory : true)

# Assets used by MyPaint
install_subdir('backgrounds', install_dir : project_datadir, strip_directory : false)
install_subdir('palettes', install_dir : project_datadir, strip_directory : false)
install_subdir('pixmaps', install_dir : project_datadir, strip_directory : false)

# For desktop environments
install_data('desktop/mypaint.desktop', install_dir : datadir + '/applications')
install_data('desktop/mypaint.appdata.xml', install_dir : datadir + '/metainfo')
install_data('desktop/mypaint-ora.thumbnailer', install_dir : datadir + '/thumbnailers')
