#!/usr/bin/env python3

# This file is part of MyPaint.
# Copyright (C) 2024 the MyPaint project
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later
# version.

# This script prints numpy's include directory. It shouldn't stay around forever. Replace it with a meson builtin after
# the following issue is closed:
# https://github.com/mesonbuild/meson/issues/9598

import numpy
print(numpy.get_include())
