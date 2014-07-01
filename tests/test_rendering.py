#!/usr/bin/env python
import os
import sys
import time
import math
import cairo
from collections import namedtuple

os.chdir(os.path.dirname(sys.argv[0]))
sys.path.insert(0, '..')

import lib.helpers
import gui.tileddrawwidget
from lib.document import Document


TEST_BIGIMAGE = "bigimage.ora"

FakeAlloc = namedtuple("FakeAlloc", ["x", "y", "width", "height"])


class TiledDrawWidget (gui.tileddrawwidget.TiledDrawWidget):
    """Monkeypatched TDW for testing purposes"""

    def __init__(self, *args, **kwargs):
        gui.tileddrawwidget.TiledDrawWidget.__init__(self, *args, **kwargs)
        self.renderer.get_allocation = self._get_allocation

    def set_allocation(self, alloc):
        self._alloc = alloc

    def _get_allocation(self):
        return self._alloc


def test_scroll( tdw, model, width=1920, height=1080,
                 zoom=1.0, mirrored=False, rotation=0.0,
                 turns=8, turn_steps=8, turn_radius=0.3,
                 save_pngs=False,
                 ):
    """Test scroll performance

    Scroll around in a circle centred on the virtual display, testing
    the same sort of render that's used for display - albeit to an
    in-memory surface.

    This tests rendering and cache performance quite well, though it
    discounts Cairo acceleration.
    """
    radius = min(width, height) * turn_radius
    alloc = FakeAlloc(0, 0, width, height)
    tdw.set_allocation(alloc)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)

    tdw.set_rotation(rotation)
    tdw.set_zoom(zoom)
    tdw.set_mirrored(mirrored)
    tdw.recenter_document()

    start = time.clock()
    cx, cy = tdw.get_center()
    last_x = cx
    last_y = cy
    for turn_i in xrange(turns):
        for step_i in xrange(turn_steps):
            t = 2 * math.pi * (float(step_i)/turn_steps)
            x = cx + math.cos(t) * radius
            y = cy + math.sin(t) * radius
            dx = x - last_x
            dy = y - last_y
            cr = cairo.Context(surf)
            cr.rectangle(*alloc)
            cr.clip()
            tdw.scroll(dx, dy)
            tdw.renderer._repaint(cr)
            surf.flush()
            last_x = x
            last_y = y
            if save_pngs:
                filename = "/tmp/scroll-%03d-%03d.png" % (step_i, turn_i)
                surf.write_to_png(filename)
    return time.clock() - start


TESTS = [
    ("scroll_20%_1rev", test_scroll, dict(
            zoom=5.0,
            turns=1,
        )),
    ("scroll_20%_30revs", test_scroll, dict(
            zoom=5.0,
            turns=30,
        )),
    ("scroll_20%_1rev", test_scroll, dict(
            zoom=1.0,
            turns=1,
        )),
    ("scroll_100%_30revs", test_scroll, dict(
            zoom=1.0,
            turns=30,
        )),
    ("scroll_100%_small", test_scroll, dict(
            zoom=1.0,
            turn_radius=0.1,
        )),
    ("scroll_100%_big", test_scroll, dict(
            zoom=1.0,
            turn_radius=1,
        )),
    ("scroll_100%_bigger", test_scroll, dict(
            zoom=1.0,
            turn_radius=2,
        )),
    ("scroll_100%_huge", test_scroll, dict(
            zoom=1.0,
            turn_radius=5,
        )),
    ("scroll_1000%", test_scroll, dict(
            zoom=0.1,
        )),
    ("scroll_400%", test_scroll, dict(
            zoom=0.25,
        )),
    ("scroll_200%", test_scroll, dict(
            zoom=0.5,
        )),
    ("scroll_100%", test_scroll, dict(
            zoom=1.0,
        )),
    ("scroll_50%", test_scroll, dict(
            zoom=2.0,
        )),
    ("scroll_20%", test_scroll, dict(
            zoom=5.0,
        )),
    ("scroll_10%", test_scroll, dict(
            zoom=10.0,
        )),
    ]


def main():

    tdw = TiledDrawWidget()
    model = Document()
    try:
        model.load(TEST_BIGIMAGE)
        tdw.set_model(model)
        bbox = model.get_effective_bbox()
        for name, func, kwargs in TESTS:
            print "%s: %0.3f" % (name, func(tdw, model, **kwargs))

    finally:
        model.cleanup()

if __name__ == '__main__':
    main()
