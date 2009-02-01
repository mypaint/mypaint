# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import tiledsurface

class Layer:
    # A layer contains a list of strokes, and possibly a background
    # pixmap. There is also a surface with those strokes rendered.
    #
    # Some history from the time when each snapshot required a
    # larger-than-screen pixbuf: The undo system used to work by
    # finding the closest snapshot (there were only two or three of
    # them) and rerendering the missing strokes. There used to be
    # quite some generic code here for this. The last svn revision
    # including this code was r242.
    #
    # Now the strokes are here just for possible future features, like
    # "pick brush from stroke", or "repaint layer with different
    # brush" for brush previews.
    #
    # The strokes don't take much memory compared to the snapshots
    # IMO, but this should better be measured...

    def __init__(self):
        self.surface = tiledsurface.Surface()
        self.strokes = []
        self.background = None

    def clear(self):
        self.background = None
        self.strokes = []
        self.surface.clear()

    def load_from_pixbuf(self, pixbuf):
        self.strokes = []
        self.background = pixbuf
        self.surface.load_from_data(pixbuf)

    def save_snapshot(self):
        return (self.strokes[:], self.background, self.surface.save_snapshot())

    def load_snapshot(self, data):
        strokes, background, data = data
        self.strokes = strokes[:]
        self.background = background
        self.surface.load_snapshot(data)

    def merge_into(self, dst):
        """
        Merges this layer into dst, modifying only dst.
        """
        src = self
        dst.background = None # hm... this breaks "full-rerender" capability, but should work fine... FIXME: redesign needed?
        dst.strokes = [] # this one too...
        for tx, ty in src.surface.get_tiles():
            src.surface.composite_tile_over(dst.surface.get_tile_memory(tx, ty, readonly=False), tx, ty)
