#!/usr/bin/env python
# Tests the layer compositing/blending code for correctness of its
# advertized optimization flags.

from __future__ import division, print_function
from random import random
import unittest

import numpy as np

from . import paths
from lib import mypaintlib
from lib.tiledsurface import N
from lib.modes import MODE_STRINGS
from lib.pycompat import xrange


ALPHA_VALUES = (0.0, 0.5, 1.0)
GREY_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
COLOR_COMPONENT_VALUES = (0.333, 0.666)
FIX15_ONE = 1 << 15


class Ops (unittest.TestCase):

    def tearDown(self):
        pass

    def setUp(self):
        if not mypaintlib.heavy_debug:
            self.skipTest("not compiled with HEAVY_DEBUG")

        # Unpremultiplied floating-point RGBA tuples
        self.sample_data = []

        # Make some alpha/grey/color combinations
        for a in ALPHA_VALUES:
            for v in GREY_VALUES:
                self.sample_data.append((v, v, v, a))
            for r in COLOR_COMPONENT_VALUES:
                for g in COLOR_COMPONENT_VALUES:
                    for b in COLOR_COMPONENT_VALUES:
                        self.sample_data.append((r, g, b, a))
        assert len(self.sample_data) < N

        # What the hell, have some random junk too
        for i in xrange(len(self.sample_data), N):
            fuzz = (random(), random(), random(), random())
            self.sample_data.append(fuzz)
        assert len(self.sample_data) == N

        # Prepare striped test data in a tile array
        self.src = np.empty((N, N, 4), dtype='uint16')
        self.dst_orig = np.empty((N, N, 4), dtype='uint16')
        for i, rgba1 in enumerate(self.sample_data):
            r1 = int(FIX15_ONE * rgba1[0] * rgba1[3])
            g1 = int(FIX15_ONE * rgba1[1] * rgba1[3])
            b1 = int(FIX15_ONE * rgba1[2] * rgba1[3])
            a1 = int(FIX15_ONE * rgba1[3])

            assert r1 <= FIX15_ONE
            assert r1 >= 0
            assert g1 <= FIX15_ONE
            assert g1 >= 0
            assert b1 <= FIX15_ONE
            assert b1 >= 0
            assert a1 <= FIX15_ONE
            assert a1 >= 0

            assert r1 <= a1
            assert g1 <= a1
            assert b1 <= a1
            for j in xrange(len(self.sample_data)):
                self.src[i, j, :] = (r1, g1, b1, a1)
                self.dst_orig[j, i, :] = (r1, g1, b1, a1)

    def test_all_modes_correctness(self):
        """Test that all modes work the way they advertise"""
        # Test each mode in turn
        for mode in xrange(mypaintlib.NumCombineModes):
            mode_info = mypaintlib.combine_mode_get_info(mode)
            mode_name = mode_info["name"]

            src = self.src
            dst = np.empty((N, N, 4), dtype='uint16')
            dst[...] = self.dst_orig[...]

            # Combine using the current mode
            mypaintlib.tile_combine(mode, src, dst, True, 1.0)

            # Tests
            zero_alpha_has_effect = False
            zero_alpha_clears_backdrop = True  # means: "*always* clears b."
            can_decrease_alpha = False
            for i in xrange(len(self.sample_data)):
                for j in xrange(len(self.sample_data)):
                    old = tuple(self.dst_orig[i, j])
                    new = tuple(dst[i, j])
                    if src[i][j][3] == 0:
                        if new[3] != 0 and old[3] != 0:
                            zero_alpha_clears_backdrop = False
                        if old != new:
                            zero_alpha_has_effect = True
                    if (not can_decrease_alpha) and (new[3] < old[3]):
                        can_decrease_alpha = True
                    self.assertFalse(
                        (new[0] > new[3] or
                         new[1] > new[3] or
                         new[2] > new[3]),
                        msg="%s isn't writing premultiplied data properly"
                            % (mode_name,),
                    )
                    self.assertFalse(
                        (new[0] > FIX15_ONE or new[1] > FIX15_ONE or
                         new[2] > FIX15_ONE or new[3] > FIX15_ONE),
                        msg="%s isn't writing fix15 data properly"
                            % (mode_name,),
                    )

            flag_test_results = [
                ("zero_alpha_has_effect", zero_alpha_has_effect),
                ("zero_alpha_clears_backdrop", zero_alpha_clears_backdrop),
                ("can_decrease_alpha", can_decrease_alpha),
            ]
            for info_str, tested_value in flag_test_results:
                current_value = bool(mode_info[info_str])
                self.assertEqual(
                    current_value, tested_value,
                    msg="%s's %r is wrong: should be %r, not %r"
                        % (mode_name, info_str, tested_value, current_value),
                )
            self.assertTrue(
                mode in MODE_STRINGS,
                msg="%s needs localizable UI strings" % (mode_name,),
            )


if __name__ == "__main__":
    assert paths  # to avoid a flake8 warning, nothing more
    unittest.main()
