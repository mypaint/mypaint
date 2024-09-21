# This file is part of MyPaint.
# Copyright (C) 2011-2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (C) 2007-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function


def make_test_stack():
    """Makes a simple test RootLayerStack (2 branches of 3 leaves each)

    :return: The root stack, and a list of its leaves.
    :rtype: tuple

    """
    import lib.layer.group
    import lib.layer.data
    import lib.layer.tree
    root = lib.layer.tree.RootLayerStack(doc=None)
    layer0 = lib.layer.group.LayerStack(name='0')
    root.append(layer0)
    layer00 = lib.layer.data.PaintingLayer(name='00')
    layer0.append(layer00)
    layer01 = lib.layer.data.PaintingLayer(name='01')
    layer0.append(layer01)
    layer02 = lib.layer.data.PaintingLayer(name='02')
    layer0.append(layer02)
    layer1 = lib.layer.group.LayerStack(name='1')
    root.append(layer1)
    layer10 = lib.layer.data.PaintingLayer(name='10')
    layer1.append(layer10)
    layer11 = lib.layer.data.PaintingLayer(name='11')
    layer1.append(layer11)
    layer12 = lib.layer.data.PaintingLayer(name='12')
    layer1.append(layer12)
    return (root, [layer00, layer01, layer02, layer10, layer11, layer12])
