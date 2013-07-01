# This file is part of MyPaint.
# Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or


"""Color preview widget / current color indicator, for the status bar."""

# TODO: This *might* evolve to a color preview + alpha selector, possibly
# TODO:   with a history row taking up the bottom. For now let's draw it at
# TODO:   an aspect ratio of about 1:5 and see how users like it.


from colors import PreviousCurrentColorAdjuster


class BrushColorIndicator (PreviousCurrentColorAdjuster):
    """Previous/Current color adjuster bound to app.brush_color_manager"""

    __gtype_name__ = "MyPaintBrushColorIndicator"

    def __init__(self):
        PreviousCurrentColorAdjuster.__init__(self)
        self.connect("realize", self._init_color_manager)

    def _init_color_manager(self, widget):
        from application import get_app
        app = get_app()
        mgr = app.brush_color_manager
        assert mgr is not None
        self.set_color_manager(mgr)

