#!/usr/bin/env python
from __future__ import division, print_function

import os
import sys
import time
import math
import cairo
from collections import namedtuple

os.chdir(os.path.dirname(sys.argv[0]))
sys.path.insert(0, '..')

import lib.gichecks
import lib.helpers
from lib import mypaintlib
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


def test_scroll(
    tdw, model, width=1920, height=1080,
    zoom=1.0, mirrored=False, rotation=0.0,
    turns=8, turn_steps=8, turn_radius=0.3,
    save_pngs=False,
    set_modes=None,
    use_background=True,
):
    """Test scroll performance

    Scroll around in a circle centred on the virtual display, testing
    the same sort of render that's used for display - albeit to an
    in-memory surface.

    This tests rendering and cache performance quite well, though it
    discounts Cairo acceleration.
    """
    num_undos_needed = 0
    if set_modes:
        for path, mode in set_modes.items():
            model.select_layer(path=path)
            num_undos_needed += 1
            model.set_current_layer_mode(mode)
            num_undos_needed += 1
            assert model.layer_stack.deepget(path, None).mode == mode
    model.layer_stack.background_visible = use_background
    model.layer_stack._render_cache.clear()

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
    nframes = 0
    for turn_i in xrange(turns):
        for step_i in xrange(turn_steps):
            t = 2 * math.pi * (step_i / turn_steps)
            x = cx + math.cos(t) * radius
            y = cy + math.sin(t) * radius
            dx = x - last_x
            dy = y - last_y
            cr = cairo.Context(surf)
            cr.rectangle(*alloc)
            cr.clip()
            tdw.scroll(dx, dy)
            tdw.renderer._draw_cb(tdw, cr)
            surf.flush()
            last_x = x
            last_y = y
            if save_pngs:
                filename = "/tmp/scroll-%03d-%03d.png" % (turn_i, step_i)
                surf.write_to_png(filename)
            nframes += 1
    dt = time.clock() - start
    for i in range(num_undos_needed):
        model.undo()
    if set_modes:
        for path in set_modes.keys():
            mode = model.layer_stack.deepget(path, None).mode
            assert mode == mypaintlib.CombineNormal
    return (nframes, dt)


TESTS = [
    ("scroll_5x_1rev", test_scroll, dict(
        zoom=5.0,
        turns=1,
    )),
    ("scroll_5x_30revs", test_scroll, dict(
        zoom=5.0,
        turns=30,
    )),
    ("scroll_1x_1rev", test_scroll, dict(
        zoom=1.0,
        turns=1,
    )),
    ("scroll_1x_30revs", test_scroll, dict(
        zoom=1.0,
        turns=30,
    )),
    # Circles using the defaults at different zooms
    # At the time of writing, the default radius is 0.3 times the height
    # of a typical 1920x1080 screen.
    ("scroll_0.10x", test_scroll, dict(
        zoom=0.10,
        # Figure is not clipped by the edgest of the screen
    )),
    ("scroll_0.25x", test_scroll, dict(
        zoom=0.25,
        # Figure is clipped at the top and bottom of the circle,
        # but "only just" (in reality, tens of tiles)
    )),
    ("scroll_0.5x", test_scroll, dict(
        zoom=0.5,
        # Figure fits comfortably within the width of the screen
        # at this zoom
    )),
    ("scroll_1x", test_scroll, dict(
        zoom=1.0,
        # No blank tiles visible onscreen at 100% zoom and above.
    )),
    ("scroll_2x", test_scroll, dict(
        zoom=2.0,
    )),
    ("scroll_8x", test_scroll, dict(
        zoom=8.0,
    )),
    ("scroll_16x", test_scroll, dict(
        zoom=16.0,
    )),
    ("scroll_32x", test_scroll, dict(
        zoom=32.0,
    )),
    ("scroll_64x", test_scroll, dict(
        zoom=64.0,
    )),
    # "lazy" means taking more steps, emulating the user panning more slowly
    # For this test it just means that more tiles from one frame to the
    # next have the same identity.
    ("scroll_1x_lazy_all_onscreen", test_scroll, dict(
        zoom=1.0,
        turn_steps=16, turns=3,  # 3 lazy turns
        turn_radius=0.1,
    )),
    ("scroll_1x_lazy_all_onscreen_masks", test_scroll, dict(
        zoom=1.0,
        turn_steps=16, turns=3,  # 3 lazy turns
        turn_radius=0.1,
        set_modes={
            (10,): mypaintlib.CombineDestinationIn,
            (5,): mypaintlib.CombineDestinationIn,
        },
    )),
    ("scroll_1x_lazy_all_onscreen_nobg", test_scroll, dict(
        zoom=1.0,
        turn_steps=16, turns=3,  # 3 lazy turns
        turn_radius=0.1,
        use_background=False,
    )),

    ("scroll_1x_lazy_mostly_onscreen", test_scroll, dict(
        zoom=1.0,
        turn_steps=16, turns=3,  # 3 lazy turns
        turn_radius=1,  # circles show some empty space
    )),
    ("scroll_1x_lazy_mostly_onscreen_masks", test_scroll, dict(
        zoom=1.0,
        turn_steps=16, turns=3,  # 3 lazy turns
        turn_radius=1,  # circles show some empty space
        set_modes={
            (10,): mypaintlib.CombineDestinationIn,
            (5,): mypaintlib.CombineDestinationIn,
        },
    )),
    ("scroll_1x_lazy_mostly_onscreen_nobg", test_scroll, dict(
        zoom=1.0,
        turn_steps=16, turns=3,  # 3 lazy turns
        turn_radius=1,  # circles show some empty space
        use_background=False,
    )),

    ("scroll_1x_lazy_mostly_offscreen", test_scroll, dict(
        zoom=1.0,
        turn_steps=16, turns=3,  # 3 lazy turns
        turn_radius=2,  # now mostly empty space outside the image
    )),
    ("scroll_1x_lazy_mostly_offscreen_masks", test_scroll, dict(
        zoom=1.0,
        turn_steps=16, turns=3,  # 3 lazy turns
        turn_radius=2,  # now mostly empty space outside the image
        set_modes={
            (10,): mypaintlib.CombineDestinationIn,
            (5,): mypaintlib.CombineDestinationIn,
        },
    )),
    ("scroll_1x_lazy_mostly_offscreen_nobg", test_scroll, dict(
        zoom=1.0,
        turn_steps=16, turns=3,  # 3 lazy turns
        turn_radius=2,  # now mostly empty space outside the image
        use_background=False,
    )),
]


def main():

    tdw = TiledDrawWidget()
    tdw.zoom_max = 64.0
    tdw.zoom_min = 1.0/16
    model = Document()
    try:
        model.load(TEST_BIGIMAGE)
        tdw.set_model(model)
        for name, func, kwargs in TESTS:
            nframes, dt = func(tdw, model, **kwargs)
            if dt <= 0:
                print("%s: 0s")
            else:
                print("%s: %0.3f seconds, %0.1f fps" % (name, dt,
                                                        nframes / dt))

    finally:
        model.cleanup()

if __name__ == '__main__':
    main()
