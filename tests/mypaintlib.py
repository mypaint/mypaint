#!/usr/bin/env python

# Imports:

from __future__ import division, print_function
from time import time
from os.path import join
import unittest
import sys
import os
import tempfile
import shutil

import numpy as np

from . import paths
from lib import mypaintlib
from lib import tiledsurface
from lib import brush
from lib import document


N = mypaintlib.TILE_SIZE


# Test cases:

class TileConversions (unittest.TestCase):
    """Test the C tile conversion functions."""

    def test_fix15_to_int8_transparent(self):
        """Fully transparent tile stays fully transparent, without noise"""
        src = np.zeros((N, N, 4), 'uint16')
        dst = np.ones((N, N, 4), 'uint8')
        mypaintlib.tile_convert_rgba16_to_rgba8(src, dst, 2.2)
        self.assertFalse(dst.any(), msg="Not fully transparent")

    def test_fix15_to_uint8_opaque(self):
        """Fully opaque tile stays fully opaque"""
        src = np.zeros((N, N, 4), 'uint16')
        src[:, :, 3] = 1 << 15
        src[:, :, :3] = np.random.randint(0, 1 << 15, (N, N, 3))
        dst = np.zeros((N, N, 4), 'uint8')
        mypaintlib.tile_convert_rgba16_to_rgba8(src, dst, 2.2)
        self.assertTrue((dst[:, :, 3] == 255).all(), msg="Not fully opaque")


class Painting (unittest.TestCase):
    """Tests basic painting functionality."""

    @classmethod
    def setUpClass(cls):
        cls._old_cwd = os.getcwd()
        cls._temp_dir = tempfile.mkdtemp()
        os.chdir(cls._temp_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._old_cwd)
        shutil.rmtree(cls._temp_dir, ignore_errors=True)

    def test_direct_paint(self):
        """30s of painting at 1x with the default brush"""
        s = tiledsurface.Surface()
        events = np.loadtxt(join(paths.TESTS_DIR, 'painting30sec.dat'))

        t0 = time()
        s.begin_atomic()
        for t, x, y, pressure in events:
            r = g = b = 0.5 * (1.0 + np.sin(t))
            r *= 0.8
            s.draw_dab(x, y, 12, r, g, b, pressure, 0.6)
        s.end_atomic()
        s.save_as_png('test_directPaint.png')
        print('%0.4fs, ' % (time() - t0,), end="", file=sys.stderr)

    def test_brush_paint(self):
        """30s of painting at 4x with a charcoal brush"""
        s = tiledsurface.Surface()
        myb_path = join(paths.TESTS_DIR, 'brushes/v2/charcoal.myb')
        with open(myb_path, "r") as fp:
            bi = brush.BrushInfo(fp.read())
        b = brush.Brush(bi)

        events = np.loadtxt(join(paths.TESTS_DIR, 'painting30sec.dat'))

        bi.set_color_rgb((0.0, 0.9, 1.0))

        t0 = time()
        for i in range(10):
            t_old = events[0][0]
            for t, x, y, pressure in events:
                dtime = t - t_old
                t_old = t
                s.begin_atomic()
                b.stroke_to(
                    s.backend,
                    x * 4,
                    y * 4,
                    pressure,
                    0.0, 0.0,
                    dtime,
                    1.0,  # view zoom
                    0.0,  # view rotation
                    0.0,  # barrel rotation
                )
                s.end_atomic()
        print('%0.4fs, ' % (time() - t0,), end="", file=sys.stderr)
        # FIXME: why is this time so different each run?
        # print(s.get_bbox(), b.get_total_stroke_painting_time())

        s.save_as_png('test_brushPaint.png')


class DocPaint (unittest.TestCase):
    """Test document equality after saving and loading."""

    # Helpers:

    def files_equal(self, a, b):
        with open(a, 'rb') as af, open(b, 'rb') as bf:
            return af.read() == bf.read()

    def assert_files_equal(self, a, b):
        self.assertTrue(self.files_equal(a, b), "Files %r and %r differ")

    def assert_pngs_equal(self, a, b):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib not available")

        if self.files_equal(a, b):
            return

        equal_enough = True
        msg = None

        im_a = plt.imread(a) * 255.0
        im_b = plt.imread(b) * 255.0

        if equal_enough:
            if im_a.shape != im_b.shape:
                msg = ("%r and %r have different sizes (%r, %r)"
                       % (a, b, im_a.shape, im_b.shape))
                equal_enough = False

        if equal_enough:
            diff = im_b - im_a
            alpha = im_a.shape[-1] == 4
            if alpha:
                diff_alpha = diff[:, :, 3]

            # print(
            #   a, 'and', b,
            #   'are different, analyzing whether it is '
            #   'just the undefined colors...'
            # )
            # print('Average difference (255=white): (R, G, B, A)')
            # print(np.mean(np.mean(diff, 0), 0))

            diff = diff[:, :, 0:3]
            if alpha:
                diff *= plt.imread(a)[:, :, 3:4]

            res = np.mean(np.mean(diff, 0), 0)
            # dithering should make this value nearly zero...
            avgdiff = np.mean(res)
            if avgdiff > 0.07:
                msg = ("The average difference with premultiplied alpha "
                       "is too great: %r > 0.07 [255=white]." % (avgdiff,))
                equal_enough = False

        if equal_enough:
            res = np.amax(np.amax(abs(diff), 0), 0)
            maxdiff = max(abs(res))
            if maxdiff > 8.0:
                # This error will be visible
                # - smaller errors are hidden by the weak alpha
                #   but we should pay attention not to accumulate such
                #   errors at each load/save cycle.
                msg = ("The maximum abs difference with premultiplied alpha "
                       "is too great: %r > 8.0 [255=white]." % (maxdiff,))
                equal_enough = False

        self.assertTrue(equal_enough, msg=msg)

        if equal_enough:
            return True

        if False:
            print('Not equal enough! Visualizing error...')
            if alpha:
                plt.figure(1)
                plt.title('Alpha')
                plt.imshow(im_b[:, :, 3], interpolation='nearest')
                plt.colorbar()
            plt.figure(2)
            plt.title('Green Error (multiplied with alpha)')
            plt.imshow(diff[:, :, 1], interpolation='nearest')
            plt.colorbar()
            if alpha:
                plt.figure(3)
                plt.title('Alpha Error')
                plt.imshow(diff_alpha, interpolation='nearest')
                plt.colorbar()
            plt.show()

        return False

    # Unit test API:

    @classmethod
    def setUpClass(cls):
        cls._old_cwd = os.getcwd()
        cls._temp_dir = tempfile.mkdtemp()
        os.chdir(cls._temp_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._old_cwd)
        shutil.rmtree(cls._temp_dir, ignore_errors=True)

    def test_docpaint(self):
        """Saved and reloaded documents look identical"""

        # TODO: brushes should be tested in the new JSON (v3) format
        with open(join(paths.TESTS_DIR, 'brushes/v2/s008.myb')) as fp:
            b1 = brush.BrushInfo(fp.read())
        with open(join(paths.TESTS_DIR, 'brushes/v2/redbrush.myb')) as fp:
            b2 = brush.BrushInfo(fp.read())
        with open(join(paths.TESTS_DIR, 'brushes/v2/watercolor.myb')) as fp:
            b3 = brush.BrushInfo(fp.read())

        b = brush.BrushInfo()
        b.load_defaults()

        # test some actions
        doc = document.Document(b, painting_only=True)
        events = np.loadtxt(join(paths.TESTS_DIR, 'painting30sec.dat'))
        # events = events[:len(events) // 8]
        t_old = events[0][0]
        n = len(events)
        for i, (t, x, y, pressure) in enumerate(events):
            dtime = t - t_old
            t_old = t

            layer = doc.layer_stack.current
            layer.stroke_to(
                doc.brush,
                x / 4, y / 4,
                pressure,
                0.0, 0.0,
                dtime,
                1.0,  # view zoom
                0.0,  # view rotation
                0.0,  # barrel rotation
            )

            # Vary the colour so we know roughly where we are

            # Transition from one brush to another and occasionally make
            # some new layers.
            if i == 0:
                b.load_from_brushinfo(b1)
                hsv = (i / n, 1.0, 1.0)
                b.set_color_hsv(hsv)
            if i == int(n * 1 / 6):
                b.load_from_brushinfo(b2)
                hsv = (i / n, 1.0, 1.0)
                b.set_color_hsv(hsv)
            if i == int(n * 2 / 6):
                b.load_from_brushinfo(b3)
                hsv = (i / n, 1.0, 1.0)
                b.set_color_hsv(hsv)

            if i == int(n * 3 / 6):
                doc.add_layer([-1])
                b.load_from_brushinfo(b1)
                hsv = (i / n, 1.0, 1.0)
                b.set_color_hsv(hsv)
            if i == int(n * 4 / 6):
                b.load_from_brushinfo(b2)
                hsv = (i / n, 1.0, 1.0)
                b.set_color_hsv(hsv)
            if i == int(n * 5 / 6):
                b.load_from_brushinfo(b3)
                hsv = (i / n, 1.0, 1.0)
                b.set_color_hsv(hsv)

        # If there is an eraser (or smudging) at work, we might be
        # erasing tiles that are empty. Those tile get memory allocated
        # and affect the bounding box of the layer. This shouldn't be a
        # big issue, but they get dropped when loading a document, which
        # makes a comparison of the PNG files fail. The hack below is
        # to avoid that.
        for i, (path, layer) in enumerate(doc.layer_stack.walk()):
            layer._surface.remove_empty_tiles()

            png1a = 'test_doc1_layer%da.png' % (i,)
            png1b = 'test_doc1_layer%db.png' % (i,)
            layer.save_as_png(png1a)
            layer.save_as_png(png1b)

            # the resulting images will look slightly different because of
            # dithering
            self.assert_pngs_equal(png1a, png1b)

        # Whole doc save and load
        doc.save('test_doc1.ora')

        doc2 = document.Document()
        doc2.load('test_doc1.ora')

        # (We don't preserve the absolute position of the image, only
        # the size.)
        # assert doc.get_bbox() == doc2.get_bbox()

        # print('doc / doc2 bbox:', doc.get_bbox(), doc2.get_bbox())

        for i, (path, layer) in enumerate(doc2.layer_stack.walk()):
            png1a = 'test_doc1_layer%da.png' % (i,)
            png2a = 'test_doc2_layer%da.png' % (i,)
            png2b = 'test_doc2_layer%db.png' % (i,)
            layer.save_as_png(png2a)
            layer.save_as_png(png2b)
            self.assert_pngs_equal(png2a, png2b)
            self.assert_pngs_equal(png2a, png1a)

        doc2.save('test_doc2.ora')

        # check not possible, because PNGs not exactly equal:-
        # assert files_equal('test_f1.ora', 'test_f2.ora')

        # less strict test than above (just require load-save-load-save
        # not to alter the file)
        doc3 = document.Document()
        doc3.load('test_doc2.ora')
        self.assertTrue(doc2.get_bbox() == doc3.get_bbox())

        # check not possible, because PNGs not exactly equal:-
        # assert files_equal('test_f2.ora', 'test_f3.ora')

        # note: this is not supposed to be strictly reproducible because
        # of different random seeds [huh? what does that mean?]
        # bbox = doc.get_bbox()
        # print('document bbox is', bbox)

        # test for appearance changes (make sure they are intended)
        doc.save('test_docPaint_flat.png', alpha=False)
        doc.save('test_docPaint_alpha.png', alpha=True)
        self.assert_pngs_equal(
            'test_docPaint_flat.png',
            join(paths.TESTS_DIR, 'correct_docPaint_flat.png'),
        )
        self.assert_pngs_equal(
            'test_docPaint_alpha.png',
            join(paths.TESTS_DIR, 'correct_docPaint_alpha.png'),
        )


class Frame (unittest.TestCase):
    """Test frame saving"""

    @classmethod
    def setUpClass(cls):
        cls._old_cwd = os.getcwd()
        cls._temp_dir = tempfile.mkdtemp()
        os.chdir(cls._temp_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._old_cwd)
        shutil.rmtree(cls._temp_dir, ignore_errors=True)

    def test_save_doc_in_frame(self):
        """Test saving many different crops of bigimage.ora"""
        cnt = 0
        doc = document.Document()
        doc.load(join(paths.TESTS_DIR, 'bigimage.ora'))
        doc.set_frame_enabled(True)
        s = tiledsurface.Surface()

        t0 = time()
        positions = list(range(-1, +2))
        positions.extend(range(-N - 1, -N + 2))
        positions.extend(range(+N - 1, +N + 2))
        for x1 in positions:
            for x2 in positions:
                for y1 in positions:
                    for y2 in positions:
                        if x2 <= x1 or y2 <= y1:
                            continue
                        cnt += 1
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1
                        # print x, y, w, h
                        s.save_as_png('test_saveFrame_s.png', x, y, w, h)
                        doc.update_frame(x=x, y=y, width=w, height=h)
                        # doc.save('test_saveFrame_doc_%dx%d.png' % (w,h))
                        doc.save('test_saveFrame_doc.png')
                        doc.save('test_saveFrame_doc.jpg')
        print(
            "saved %d frames in %0.2fs, " % (cnt, time() - t0),
            end="",
            file=sys.stderr,
        )


if __name__ == "__main__":
    unittest.main()
    # Formerly:
    # # tileConversions()
    # # layerModes()
    # directPaint()
    # brushPaint()
    # # docPaint()
    # # saveFrame()
