# This file is part of MyPaint.
# Copyright (C) 2012 by Jon Nordby <jononor@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import print_function

import sys
import os
import brush


def migrate_brushes_to_json(dirpath):

    files = os.listdir(dirpath)
    files = [os.path.join(dirpath, fn)for fn in files if os.path.splitext(fn)[1] == '.myb']

    for fpath in files:
        b = brush.BrushInfo(open(fpath, 'r').read())
        open(fpath, 'w').write(b.to_json())


if __name__ == '__main__':

    directories = sys.argv[1:]
    for dir in directories:
        migrate_brushes_to_json(dir)
