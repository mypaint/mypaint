#!/usr/bin/env python
# Tests the layer compositing/blending code for correctness of its
# advertized optimization flags.

from __future__ import print_function

import os
import sys
from random import random

import numpy as np

os.chdir(os.path.dirname(sys.argv[0]))
sys.path.insert(0, '..')

from lib import mypaintlib
from lib.tiledsurface import N
from lib.layer import MODE_STRINGS

# Unpremultiplied floating-point RGBA tuples
SAMPLE_DATA = []

# Make some alpha/grey/color combinations
ALPHA_VALUES = (0.0, 0.5, 1.0)
GREY_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
COLOR_COMPONENT_VALUES = (0.333, 0.666)
for a in ALPHA_VALUES:
    for v in GREY_VALUES:
        SAMPLE_DATA.append((v, v, v, a))
    for r in COLOR_COMPONENT_VALUES:
        for g in COLOR_COMPONENT_VALUES:
            for b in COLOR_COMPONENT_VALUES:
                SAMPLE_DATA.append((r, g, b, a))
assert len(SAMPLE_DATA) < N

# What the hell, have some random junk too
for i in xrange(len(SAMPLE_DATA), N):
    fuzz = (random(), random(), random(), random())
    SAMPLE_DATA.append(fuzz)
assert len(SAMPLE_DATA) == N

# Need to turn the optimizations to be tested off...
assert mypaintlib.heavy_debug, \
    ("HEAVY_DEBUG must be set for this test code to work properly.\n"
     "Please recompile using 'scons debug=1', and re-run this script.")

# Prepare striped test data in a tile array
FIX15_ONE = 1 << 15
src = np.empty((N, N, 4), dtype='uint16')
dst_orig = np.empty((N, N, 4), dtype='uint16')
for i, rgba1 in enumerate(SAMPLE_DATA):
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
    for j in xrange(len(SAMPLE_DATA)):
        src[i, j, :] = (r1, g1, b1, a1)
        dst_orig[j, i, :] = (r1, g1, b1, a1)

# Test each mode in turn
for mode in xrange(mypaintlib.NumCombineModes):
    mode_info = mypaintlib.combine_mode_get_info(mode)
    mode_name = mode_info["name"]
    print(mode_name, end=' ')

    dst = np.empty((N, N, 4), dtype='uint16')
    dst[...] = dst_orig[...]

    # Combine using the current mode
    mypaintlib.tile_combine(mode, src, dst, True, 1.0)

    # Tests
    all_ok = True
    zero_alpha_has_effect = False
    zero_alpha_clears_backdrop = True   # meaning is "*always* clears b."
    can_decrease_alpha = False
    for i in xrange(len(SAMPLE_DATA)):
        for j in xrange(len(SAMPLE_DATA)):
            old = tuple(dst_orig[i, j])
            new = tuple(dst[i, j])
            if src[i][j][3] == 0:
                if new[3] != 0 and old[3] != 0:
                    zero_alpha_clears_backdrop = False
                if old != new:
                    zero_alpha_has_effect = True
            if (not can_decrease_alpha) and (new[3] < old[3]):
                can_decrease_alpha = True
            if all_ok:
                if new[0] > new[3] or new[1] > new[3] or new[2] > new[3]:
                    if all_ok:
                        print("**FAILED**")
                        all_ok = False
                    print ("  %s isn't writing premultiplied data properly"
                           % (mode_name,))
                if (new[0] > FIX15_ONE or new[1] > FIX15_ONE or
                        new[2] > FIX15_ONE or new[3] > FIX15_ONE):
                    if all_ok:
                        print("**FAILED**")
                        all_ok = False
                    print ("  %s isn't writing fix15 data properly"
                           % (mode_name,))

    flag_test_results = [
        ("zero_alpha_has_effect", zero_alpha_has_effect),
        ("zero_alpha_clears_backdrop", zero_alpha_clears_backdrop),
        ("can_decrease_alpha", can_decrease_alpha),
    ]
    for info_str, tested_value in flag_test_results:
        current_value = bool(mode_info[info_str])
        if current_value != tested_value:
            if all_ok:
                print("**FAILED**")
                all_ok = False
            print ("  %s's %r is wrong: should be %r, not %r"
                   % (mode_name, info_str, tested_value, current_value))
    if mode not in MODE_STRINGS:
        if all_ok:
            print("**FAILED**")
            all_ok = False
        print("  %s needs localizable UI strings" % (mode_name,))
    if all_ok:
        print("ok")
