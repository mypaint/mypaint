#!/usr/bin/env python

# Imports:

from __future__ import division, print_function
from os.path import join
import sys
import time
import math
import cairo
from collections import namedtuple
import unittest

from . import paths
import lib.gichecks
from lib import mypaintlib
from lib.document import Document
from lib.pycompat import xrange


TEST_BIGIMAGE = "bigimage.ora"


# Helpers:

def _scroll(tdw, model, width=1920, height=1080,
            zoom=1.0, mirrored=False, rotation=0.0,
            turns=8, turn_steps=8, turn_radius=0.3,
            save_pngs=False,
            set_modes=None,
            use_background=True):
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
    fakealloc = namedtuple("FakeAlloc", ["x", "y", "width", "height"])
    alloc = fakealloc(0, 0, width, height)
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


# Test cases:

class Scroll (unittest.TestCase):
    """Not-quite headless raw panning/scrolling performance tests."""

    def test_5x_1rev(self):
        self._run_test(
            _scroll,
            zoom=5.0,
            turns=1,
        )

    def test_5x_30revs(self):
        self._run_test(
            _scroll,
            zoom=5.0,
            turns=30,
        )

    def test_1x_1rev(self):
        self._run_test(
            _scroll,
            zoom=1.0,
            turns=1,
        )

    def test_1x_30revs(self):
        self._run_test(
            _scroll,
            zoom=1.0,
            turns=30,
        )

    # Circles using the defaults at different zooms
    # At the time of writing, the default radius is 0.3 times the height
    # of a typical 1920x1080 screen.

    def test_0x10(self):
        # Figure is not clipped by the edges of the screen
        self._run_test(
            _scroll,
            zoom=0.10,
        )

    def test_0x25(self):
        # Figure is clipped at the top and bottom of the circle,
        # but "only just" (in reality, tens of tiles)
        self._run_test(
            _scroll,
            zoom=0.25,
        )

    def test_0x50(self):
        # Figure fits comfortably within the width of the screen
        # at this zoom
        self._run_test(
            _scroll,
            zoom=0.5,
        )

    def test_1x(self):
        # No blank tiles visible onscreen at 100% zoom and above.
        self._run_test(
            _scroll,
            zoom=1.0,
        ),

    def test_2x(self):
        self._run_test(
            _scroll,
            zoom=2.0,
        )

    def test_8x(self):
        self._run_test(
            _scroll,
            zoom=8.0,
        )

    def test_16x(self):
        self._run_test(
            _scroll,
            zoom=16.0,
        )

    def test_32x(self):
        self._run_test(
            _scroll,
            zoom=32.0,
        )

    def test_64x(self):
        self._run_test(
            _scroll,
            zoom=64.0,
        )

    # "lazy" means taking more steps, emulating the user panning more slowly
    # For this test it just means that more tiles from one frame to the
    # next have the same identity.

    def test_1x_lazy_all_onscreen(self):
        self._run_test(
            _scroll,
            zoom=1.0,
            turn_steps=16, turns=3,  # 3 lazy turns
            turn_radius=0.1,
        )

    def test_1x_lazy_all_onscreen_masks(self):
        self._run_test(
            _scroll,
            zoom=1.0,
            turn_steps=16, turns=3,  # 3 lazy turns
            turn_radius=0.1,
            set_modes={
                (10,): mypaintlib.CombineDestinationIn,
                (5,): mypaintlib.CombineDestinationIn,
            },
        )

    def test_1x_lazy_all_onscreen_nobg(self):
        self._run_test(
            _scroll,
            zoom=1.0,
            turn_steps=16, turns=3,  # 3 lazy turns
            turn_radius=0.1,
            use_background=False,
        )

    def test_1x_lazy_mostly_onscreen(self):
        self._run_test(
            _scroll,
            zoom=1.0,
            turn_steps=16, turns=3,  # 3 lazy turns
            turn_radius=1,  # circles show some empty space
        )

    def test_1x_lazy_mostly_onscreen_masks(self):
        self._run_test(
            _scroll,
            zoom=1.0,
            turn_steps=16, turns=3,  # 3 lazy turns
            turn_radius=1,  # circles show some empty space
            set_modes={
                (10,): mypaintlib.CombineDestinationIn,
                (5,): mypaintlib.CombineDestinationIn,
            },
        )

    def test_1x_lazy_mostly_onscreen_nobg(self):
        self._run_test(
            _scroll,
            zoom=1.0,
            turn_steps=16, turns=3,  # 3 lazy turns
            turn_radius=1,  # circles show some empty space
            use_background=False,
        )

    def test_1x_lazy_mostly_offscreen(self):
        self._run_test(
            _scroll,
            zoom=1.0,
            turn_steps=16, turns=3,  # 3 lazy turns
            turn_radius=2,  # now mostly empty space outside the image
        )

    def test_1x_lazy_mostly_offscreen_masks(self):
        self._run_test(
            _scroll,
            zoom=1.0,
            turn_steps=16, turns=3,  # 3 lazy turns
            turn_radius=2,  # now mostly empty space outside the image
            set_modes={
                (10,): mypaintlib.CombineDestinationIn,
                (5,): mypaintlib.CombineDestinationIn,
            },
        )

    def test_1x_lazy_mostly_offscreen_nobg(self):
        self._run_test(
            _scroll,
            zoom=1.0,
            turn_steps=16, turns=3,  # 3 lazy turns
            turn_radius=2,  # now mostly empty space outside the image
            use_background=False,
        )

    @classmethod
    def setUpClass(cls):
        # The tdw import below just segfaults on my system right now, if
        # there's no X11 display available. Be careful about proceeding.

        cls._tdw = None
        cls._model = None

        from gi.repository import Gdk
        if Gdk.Display.get_default() is None:
            return

        try:
            import gui.tileddrawwidget
        except Exception:
            return

        class TiledDrawWidget (gui.tileddrawwidget.TiledDrawWidget):
            """Monkeypatched TDW for testing purposes"""

            def __init__(self, *args, **kwargs):
                gui.tileddrawwidget.TiledDrawWidget\
                    .__init__(self, *args, **kwargs)
                self.renderer.get_allocation = self._get_allocation

            def set_allocation(self, alloc):
                self._alloc = alloc

            def _get_allocation(self):
                return self._alloc

        tdw = TiledDrawWidget()
        tdw.zoom_max = 64.0
        tdw.zoom_min = 1.0 / 16
        model = Document(painting_only=True)
        model.load(join(paths.TESTS_DIR, TEST_BIGIMAGE))
        tdw.set_model(model)
        cls._model = model
        cls._tdw = tdw

    @classmethod
    def tearDownClass(cls):
        if cls._model:
            cls._model.cleanup()

    def _run_test(self, func, **kwargs):
        if not (self._tdw and self._model):
            self.skipTest("no GUI or unable to import TDW class")
        nframes, dt = func(self._tdw, self._model, **kwargs)
        if dt <= 0:
            msg = "0s"
        else:
            msg = "%0.3fs, %0.1ffps" % (dt, nframes / dt)
        print(msg, end=", ", file=sys.stderr)


if __name__ == '__main__':
    assert(lib.gichecks)  # avoid a flake8 warning
    unittest.main()
