#!/usr/bin/env python

# Imports:

from __future__ import division, print_function
from time import time
import os
import sys
from os.path import join
import unittest
import contextlib
import copy
from itertools import repeat, chain, product

from . import paths
from lib import mypaintlib
from lib import document
from lib import floodfill
from lib import fill_common
from lib import morphology

N = mypaintlib.TILE_SIZE


def exact_bbox(layer):
    """Expensively determine the bounding box
    coordinates of the actual pixel data
    :param layer: Layer to calculate bbox of
    :type layer: lib.layer.SimplePaintingLayer
    :rtype: lib.helpers.Rect
    """
    bbox = layer.get_bbox().copy()
    _x, _y, w, h = bbox
    tile_coordinates = layer.get_tile_coords()
    surf = layer._surface
    min_tx, min_ty = _x//N, _y//N
    max_tx, max_ty = (_x + w - 1)//N, (_y + h - 1)//N

    # The x and y bound searches are very similar,
    # but much more legible when written separately
    def y_bound(ty, bound, base, offset):
        for tx in range(min_tx, max_tx+1):
            if (tx, ty) not in tile_coordinates:
                continue
            with surf.tile_request(tx, ty, readonly=True) as tile:
                for (y, x) in product(range(base, bound, offset), range(N)):
                    if tile[y, x, 3] > 0:
                        bound = y
                        break
        return bound

    def x_bound(tx, bound, base, offset):
        for ty in range(min_ty, max_ty+1):
            if (tx, ty) not in tile_coordinates:
                continue
            with surf.tile_request(tx, ty, readonly=True) as tile:
                # Beware of awful (but convenient) memory access pattern below
                for (x, y) in product(range(base, bound, offset), range(N)):
                    if tile[y, x, 3] > 0:
                        bound = x
                        break
        return bound

    min_y = y_bound(min_ty, N-1, 0, 1)
    max_y = y_bound(max_ty, 0, N-1, -1)
    min_x = x_bound(min_tx, N-1, 0, 1)
    max_x = x_bound(max_tx, 0, N-1, -1)

    bbox.x += min_x
    bbox.y += min_y
    bbox.w += max_x - (min_x + N-1)
    bbox.h += max_y - (min_y + N-1)
    return bbox


def fill_test(test_func):
    """Decorator for fill tests; clears fill layers between tests"""
    def inner(self, *args, **kwargs):
        assert isinstance(self, FillTestsBase)
        self.clear_fill_layers()
        test_func(self, *args, **kwargs)
        self.clear_fill_layers()

    return inner


class FillTestsBase(unittest.TestCase):

    GAP_GROUP_PATH = (1,)
    FILL_GROUP_PATH = (3,)

    # Helpers
    @staticmethod
    def center(bbox):
        x, y, w, h = bbox
        return x + w//2, y + h//2

    @staticmethod
    def area(bbox):
        return bbox.w * bbox.h

    @staticmethod
    def layers_identical(l1, l2):
        """Check the layers are tile-for-tile identical"""
        if l1 is l2:
            return True
        s1, s2 = l1._surface, l2._surface
        t1, t2 = s1.get_tiles(), s2.get_tiles()
        tile_coordinates = set(t1).union(set(t2))
        for tx, ty in tile_coordinates:
            with s1.tile_request(tx, ty, readonly=True) as t1:
                with s2.tile_request(tx, ty, readonly=True) as t2:
                    if not (t1 is t2 or (t1 == t2).all()):
                        print(
                            "\nTile not matching={tile}!".format(
                                tile=(tx, ty)
                            )
                        )
                        return False
        return True

    def fill(
            self, src, dst, bbox=None, init_xy=None,
            tol=0.2, offset=0, feather=0, gc=None,
            framed=False
    ):
        """
        :param src: Outline goes here
        :type src: lib.layer.LayerStack
        :param dst: Fill goes here
        :type dst: lib.layer.LayerStack
        :param bbox: Bounding box limiting fill (default: root stack bbox)
        :param init_xy: Starting point of fill (default: src bbox center)
        :param tol: tolerance
        :param offset: Grow/Shrink fill by this amount [-64, 64]
        :param feather: Blur fill by this amount
        :param gc: gap closing parameters
        :type gc: lib.floodfill.GapClosingOptions
        :param framed: To be or not to be (framed)
        :type framed: bool
        """
        if bbox:
            x, y = self.center(bbox)
        else:
            bbox = self.root.get_bbox()
            x, y = self.center(src.get_bbox())
        if init_xy:
            x, y = init_xy
        col = (0.0, 0.0, 0.0)
        mode = mypaintlib.CombineNormal
        lock_alpha = False
        seeds = {(x, y)}
        opacity = 1.0
        args = floodfill.FloodFillArguments(
            (x, y), seeds, col, tol, offset,
            feather, gc, mode, lock_alpha,
            opacity, framed, bbox
        )
        handle = src.flood_fill(args, dst)
        handle.wait()

    @classmethod
    def setUpClass(cls):
        # Load test data
        doc = document.Document()
        doc.load(join(paths.TESTS_DIR, 'fill_outlines.ora'))
        root = doc.layer_stack

        # Set up references to test layers
        cls._fill_layers = []
        for fill_layer in root.deepget(cls.FILL_GROUP_PATH):
            cls._fill_layers.append(fill_layer)

        # Layers with no gaps
        cls.closed_small_s = root.deepget((0, 0))
        cls.closed_small_c = root.deepget((0, 1))
        cls.small = (
            cls.closed_small_s,
            cls.closed_small_c,
        )
        cls.closed_large_s = root.deepget((0, 2))
        cls.closed_large_c = root.deepget((0, 3))
        cls.large = (
            cls.closed_large_s,
            cls.closed_large_c,
        )
        cls.heavy = root.deepget((2, 0))
        cls.minimal = root.deepget((0, 4))
        cls.empty_layer = root.deepget((3,))

        # Layers with gaps
        cls.gap_layers = []
        for gap_layer in root.deepget(cls.GAP_GROUP_PATH):
            cls.gap_layers.append(gap_layer)

        cls.root = root

    def clear_fill_layers(self):
        """Clear all fill layers"""
        for layer in self._fill_layers:
            if len(layer._surface.get_tiles()) > 0:
                layer.clear()

    @contextlib.contextmanager
    def fill_layers(self):
        """Return the fill layers, guaranteeing
        they are clear before and after being used for testing.
        """
        self.clear_fill_layers()
        yield self._fill_layers
        self.clear_fill_layers()


class CorrectnessTests(FillTestsBase):

    @fill_test
    def test_fill_bbox(self):
        offsets = ((21, 17), (-21, 13), (-35, -59))
        dimensions = ((32, 32), (64, 64), (134, 367), (411, 631))
        bbox = self.empty_layer.get_bbox()
        for (dx, dy), (w, h) in product(offsets, dimensions):
            _bbox = bbox.copy()
            _bbox.x += dx
            _bbox.y += dy
            _bbox.w = w
            _bbox.h = h
            with self.fill_layers() as (f1, _):
                self.fill(
                    self.empty_layer, f1, bbox=_bbox
                )
                self.assertEqual(
                    exact_bbox(f1), _bbox,
                    msg="Filling with just a bbox should fill it completely"
                )

    @fill_test
    def test_basic_properties(self):
        for src in self.small:
            src_bb = exact_bbox(src)
            with self.fill_layers() as (f1, f2):
                self.fill(src, f1)
                f1_bb = exact_bbox(f1)
                self.assertGreater(
                    self.area(f1_bb), 0,
                    msg="Fill should not be empty!"
                    " layer={layer}".format(layer=src.name)
                )
                self.assertTrue(
                    src_bb.contains(f1_bb),
                    msg="Fill should be smaller than the outline!"
                    " layer={layer}".format(layer=src.name)
                )
                # This assertion relies on the fact that the outlines are
                # 1 pixel wide and do not stretch outside of the fill x,y
                # maximum and minimum ranges. This holds for the test data.
                src_bb.expand(-2)
                self.assertTrue(
                    f1_bb.contains(src_bb),
                    msg="Fill should reach the inner edges of the outline!"
                    " layer={layer}".format(layer=src.name)
                )
                # Starting from a connected point
                # should produce an identical result
                (x, y) = self.center(src_bb)
                self.fill(src, f2, init_xy=(x-10, y-10))
                self.assertTrue(
                    self.layers_identical(f1, f2),
                    msg="Fill results should be identical!"
                )

    @fill_test
    def test_gap_closing_fill(self):
        gap_size = 7
        avoid_seeping = False
        options = floodfill.GapClosingOptions(gap_size, avoid_seeping)
        for src in self.gap_layers:
            src_bb = src.get_bbox()
            # With seeping, permit a 1-pixel leak
            # outside of existing boundaries
            src_outer_bb = exact_bbox(src)
            src_inner_bb = src_outer_bb.copy()
            src_outer_bb.expand(2)
            src_inner_bb.expand(-2)
            with self.fill_layers() as (f1, f2):
                self.fill(src, f1, bbox=src_bb)
                f1_exact = exact_bbox(f1)
                self.assertEqual(
                    f1_exact, src_bb,
                    msg="Regular fill should seep through the gaps!"
                    " src='{layer}'".format(layer=src.name)
                )
                self.fill(src, f2, bbox=src_bb, gc=options)
                f2_exact = exact_bbox(f2)
                self.assertTrue(
                    src_outer_bb.contains(f2_exact),
                    msg="GC fill should not seep through the gaps!"
                    " src='{layer}'".format(layer=src.name)
                )
                self.assertTrue(
                    f2_exact.contains(src_inner_bb),
                    msg="GC fill should fill to the outline!"
                    " src='{layer}'".format(layer=src.name)
                )

    @fill_test
    def test_translation_invariant(self):
        offsets = ((0, 63), (-35, -77), (32, 21), (14, 26), (139, 64),)
        for src in (self.minimal,) + self.small:
            for (x, y) in offsets:
                rx, ry = -1*x, -1*y
                with self.fill_layers() as (f1, f2):
                    ix, iy = self.center(src.get_bbox())
                    self.fill(src, f1, init_xy=(ix, iy), tol=0)
                    src_copy = copy.deepcopy(src)
                    src_copy.translate(x, y)
                    self.fill(src_copy, f2, init_xy=(ix+x, iy+y), tol=0)
                    f2.translate(rx, ry)
                    self.assertTrue(
                        self.layers_identical(f1, f2),
                        msg="Fill should be invariant under translation!"
                        " src={layer} offset={offset}".format(
                            layer=src.name, offset=(x, y)
                        )
                    )

    @fill_test
    def test_erosion(self):
        # The SmallComplex outline has thin protrusions and internal
        # structures; hence we test it with a small offset
        offsets = (33, 1)
        for offs, src in zip(offsets, self.small):
            with self.fill_layers() as (f1, f2):
                self.fill(src, f1, offset=0)
                self.fill(src, f2, offset=-offs)
                normal = exact_bbox(f1)
                eroded = exact_bbox(f2)
                eroded.expand(offs)
                self.assertEqual(
                    normal, eroded,
                    msg="Eroded fill should be smaller than uneroded "
                        "counterpart! src={layer}!".format(layer=src.name)
                )

    @fill_test
    def test_dilation(self):
        offsets = (1, 21, 64)
        for src in self.small:
            with self.fill_layers() as (f1, f2):
                self.fill(src, f1, offset=0)
                normal = exact_bbox(f1)
                for offs in offsets:
                    f2.clear()
                    self.fill(src, f2, offset=offs)
                    dilated = exact_bbox(f2)
                    dilated.expand(-offs)
                    self.assertEqual(
                        normal, dilated,
                        msg="Dilated fill should be larger than undilated "
                        "counterpart! src={layer} offset={offs}".format(
                            layer=src.name, offs=offs
                        )
                    )


# Performance tests, not run as part of the standard test suite

@unittest.skipUnless(
    os.getenv('RUN_PERF'),
    "Set RUN_PERF envvar to run performance tests"
)
class PerformanceTests(FillTestsBase):
    """
    Performance tests for fill functions

    Tests performance for fill algorithms and morphological operations
    """

    def fill_perf(
            self, src, n, bbox=None, init_xy=None,
            tolerance=0.2, gap_closing_options=None
    ):
        """
        Run only the fill step, not the compositing step, n times
        """
        if bbox:
            x, y = self.center(bbox)
        else:
            bbox = fill_common.TileBoundingBox(self.root.get_bbox())
            x, y = self.center(src.get_bbox())
        if init_xy:
            x, y = init_xy

        seed_lists = floodfill.seeds_by_tile({(x, y)})
        src = src._surface
        init = floodfill.starting_coordinates(x, y)
        r, g, b, a = floodfill.get_target_color(src, *init)
        filler = mypaintlib.Filler(r, g, b, a, tolerance)

        fh = floodfill.FillHandler()

        if gap_closing_options:
            for _ in range(n-1):
                floodfill.gap_closing_fill(
                    fh, src, seed_lists, bbox, filler, gap_closing_options,
                )

            return floodfill.gap_closing_fill(
                fh, src, seed_lists, bbox, filler, gap_closing_options
            )
        else:
            for _ in range(n-1):
                floodfill.scanline_fill(fh, src, seed_lists, bbox, filler)

            return floodfill.scanline_fill(fh, src, seed_lists, bbox, filler)

    @fill_test
    def test_fill_full(self):
        """
        Test performance of regular filling, including
        final compositing into the destination layer
        """
        n_small = 10
        n_large = 5
        small = zip(self.small, repeat(n_small))
        large = zip(self.large, repeat(n_large))
        print("\n== Testing fill+comp performance ==")
        print("<layer>\t\t<runs>\t\t<avg time>")
        dst = self._fill_layers[0]
        for src, repeats in chain(small, large):
            t0 = time()
            for _ in range(repeats):
                self.fill(src, dst)
            avg_time = 1000 * (time() - t0) / repeats
            print(src.name, "\t", repeats, "\t\t%0.2fms" % avg_time)

    @fill_test
    def test_gc_fill_full(self):
        """
        Test performance of gap closing regular filling, including
        final compositing into the destination layer
        """
        repeats = 30
        gap_size = 7
        options = floodfill.GapClosingOptions(gap_size, False)
        dst = self._fill_layers[0]
        print("\n== Testing gap closing+comp performance ==", file=sys.stderr)
        print("<layer>\t\t<runs>\t\t<avg time>", file=sys.stderr)
        for src in self.gap_layers:
            t0 = time()
            for _ in range(repeats):
                self.fill(src, dst, gc=options, offset=-20)
            avg_time = 1000 * (time() - t0) / repeats
            print(src.name, "\t", repeats, "\t\t%0.2fms" % avg_time)

    @fill_test
    def test_fill_only(self):
        """Test performance of regular filling, omitting compositing"""
        repeats = 10
        print("\n== Testing fill performance ==")
        print("<layer>\t\t<runs>\t\t<avg time>")
        for src in chain(self.small, self.large, (self.heavy,)):
            t0 = time()
            self.fill_perf(src, repeats)
            avg_time = 1000 * (time() - t0) / repeats
            print(src.name, "\t", repeats, "\t\t%0.2fms" % avg_time)

    @fill_test
    def test_gc_fill_only(self):
        """Test performance of gap closing filling, omitting compositing"""
        options = floodfill.GapClosingOptions(7, False)
        repeats = 50
        print("\n== Testing gap closing performance ==", file=sys.stderr)
        print("<layer>\t\t<runs>\t\t<avg time>", file=sys.stderr)
        for src in self.gap_layers:
            t0 = time()
            self.fill_perf(src, repeats, gap_closing_options=options)
            avg_time = 1000 * (time() - t0) / repeats
            print(src.name, "\t", repeats, "\t\t%0.2fms" % avg_time)

    @unittest.skipUnless(
        os.getenv("MORPH_FULL"),
        "This is a fairly heavy test, run separately"
    )
    def test_morph_full(self):
        """
        Test performance of fill + morphing, including
        final compositing into the destination layer
        """
        n_small = 10
        n_large = 10
        small = zip(self.small, repeat(n_small))
        large = zip(self.large, repeat(n_large))
        offsets = (10, -10, 40, -40)
        print("\n== Testing morph operation performance ==", file=sys.stderr)
        print("<layer>\t\t<offset>\t<runs>\t\t<avg time>", file=sys.stderr)
        dst = self._fill_layers[0]
        for offs, (src, repeats) in product(offsets, chain(small, large)):
            t0 = time()
            for _ in range(repeats):
                self.fill(src, dst, offset=offs)
            avg_time = 1000 * (time() - t0) / repeats
            print(
                src.name, "\t", offs, "\t\t", repeats, "\t\t%0.2fms" % avg_time
            )
        dst.clear()

    def test_morph_only(self):
        offset = 64
        srcs = (self.closed_small_s, self.closed_large_s, self.closed_large_c)
        handler = floodfill.FillHandler()
        print("\nTesting morph performance, offset:", offset)
        for src in srcs:
            tiles = self.fill_perf(src, 1)
            t0 = time()
            morphology.morph(handler, offset, tiles)
            t = (time() - t0)
            print(src.name, "morph time (ms)", 1000*t)

    def test_blur_only(self):
        offset = 40
        srcs = (self.closed_small_s, self.closed_large_s, self.closed_large_c)
        handler = floodfill.FillHandler()
        print("\nTesting blur performance, radius:", offset)
        for src in srcs:
            tiles = self.fill_perf(src, 1)
            t0 = time()
            morphology.blur(handler, offset, tiles)
            t = (time() - t0)
            print(src.name, "blur time (ms)", 1000*t)


if __name__ == "__main__":
    unittest.main()
