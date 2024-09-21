# This file is part of MyPaint.
# Copyright (C) 2011-2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (C) 2007-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layers holding graphical data or other layers.

Users will normally interact with `PaintingLayer`s,
which contain pixel data, and expose drawing commands.
Other types of data layer exist.

Layers are arranged in ordered stacks,
which can be nested to form a tree structure.
Data layers form the leaves, and stacks form the branches.
Layer stacks are also layers in every sense, including the root one,
and are subject to the same constraints.
The root stack is owned by the document model.
Layers must be unique within the tree structure,
although this constraint is not enforced.

Layers emit a moderately fine-grained set of notifications
when the structure changes, or when the user draws something.
These can be listened to via the root layer stack.

"""

from __future__ import division, print_function

from .group import *
from .data import *
from .core import *
from .tree import *
