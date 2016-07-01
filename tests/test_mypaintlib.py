#!/usr/bin/env python

from __future__ import print_function

from time import time
import sys
import os
import gc

import numpy as np

os.chdir(os.path.dirname(sys.argv[0]))
sys.path.insert(0, '..')

from lib import mypaintlib, tiledsurface, brush, document, command, helpers


def tileConversions():
    # fully transparent tile stays fully transparent (without noise)
    N = mypaintlib.TILE_SIZE
    src = np.zeros((N, N, 4), 'uint16')
    dst = np.ones((N, N, 4), 'uint8')
    mypaintlib.tile_convert_rgba16_to_rgba8(src, dst)
    assert not dst.any()
    # fully opaque tile stays fully opaque
    src[:, :, 3] = 1 << 15
    src[:, :, :3] = np.randint(0, 1 << 15, (N, N, 3))
    dst = np.zeros((N, N, 4), 'uint8')
    mypaintlib.tile_convert_rgba16_to_rgba8(src, dst)
    assert (dst[:, :, 3] == 255).all()


def layerModes():
    N = mypaintlib.TILE_SIZE

    dst = np.zeros((N, N, 4), 'uint16')  # rgbu
    dst_values = []
    r1 = range(0, 20)
    r2 = range((1 << 15)/2-10, (1 << 15)/2+10)
    r3 = range((1 << 15)-19, (1 << 15)+1)
    dst_values = r1 + r2 + r3

    src = np.zeros((N, N, 4), 'int64')
    alphas = np.hstack((
        np.arange(N/4),                     # low alpha
        (1 << 15)/2 - np.arange(N/4),       # 50% alpha
        (1 << 15) - np.arange(N/4),         # high alpha
        np.randint((1 << 15)+1, size=N/4),  # random alpha
        ))
    #plot(alphas); show()
    src[:, :, 3] = alphas.reshape(N, 1)  # alpha changes along y axis

    src[:, :, 0] = alphas  # red
    src[:, N*0/4:N*1/4, 0] = np.arange(N/4)  # dark colors
    src[:, N*1/4:N*2/4, 0] = alphas[N*1/4:N*2/4]/2 + np.arange(N/4) - N/2  # 50% lightness
    src[:, N*2/4:N*3/4, 0] = alphas[N*2/4:N*3/4] - np.arange(N/4)  # bright colors
    src[:, N*3/4:N*4/4, 0] = alphas[N*3/4:N*4/4] * np.random(N/4)  # random colors
    # clip away colors that are not possible due to low alpha
    src[:, :, 0] = np.minimum(src[:, :, 0], src[:, :, 3]).clip(0, 1 << 15)
    src = src.astype('uint16')

    #figure(1)
    #imshow(src[:,:,3], interpolation='nearest')
    #colorbar()
    #figure(2)
    #imshow(src[:,:,0], interpolation='nearest')
    #colorbar()
    #show()

    src[:, :, 1] = src[:, :, 0]  # green
    src[:, :, 2] = src[:, :, 0]  # blue

    for name in dir(mypaintlib):
        if not name.startswith('tile_composite_'):
            continue
        junk1, junk2, mode = name.split('_', 2)
        print('testing', name, 'for invalid output')
        f = getattr(mypaintlib, name)
        for dst_value in dst_values:
            for alpha in [1.0, 0.999, 0.99, 0.90, 0.51, 0.50, 0.49, 0.01, 0.001, 0.0]:
                dst[:] = dst_value
                dst_has_alpha = False
                src_opacity = alpha
                f(src, dst, dst_has_alpha, src_opacity)
                #imshow(dst[:,:,0], interpolation='nearest')
                #gray()
                #colorbar()
                #show()
                errors = dst > (1 << 15)
                assert not errors.any()
        print('passed')


def directPaint():

    s = tiledsurface.Surface()
    events = np.loadtxt('painting30sec.dat')

    s.begin_atomic()
    for t, x, y, pressure in events:
        r = g = b = 0.5 * (1.0 + np.sin(t))
        r *= 0.8
        s.draw_dab(x, y, 12, r, g, b, pressure, 0.6)
    s.end_atomic()
    s.save_as_png('test_directPaint.png')


def brushPaint():

    s = tiledsurface.Surface()
    with open('brushes/charcoal.myb') as fp:
        bi = brush.BrushInfo(fp.read())
    b = brush.Brush(bi)

    events = np.loadtxt('painting30sec.dat')

    bi.set_color_rgb((0.0, 0.9, 1.0))

    t0 = time()
    for i in range(10):
        t_old = events[0][0]
        for t, x, y, pressure in events:
            dtime = t - t_old
            t_old = t
            s.begin_atomic()
            b.stroke_to(s.backend, x*4, y*4, pressure, 0.0, 0.0, dtime)
            s.end_atomic()
    print('Brushpaint time:', time() - t0)
    # FIXME: why is this time so different each run?
    print(s.get_bbox(), b.get_total_stroke_painting_time())

    s.save_as_png('test_brushPaint.png')


def files_equal(a, b):
    with open(a, 'rb') as af, open(b, 'rb') as bf:
        return af.read() == bf.read()


def pngs_equal(a, b):
    import matplotlib.pyplot as plt

    if files_equal(a, b):
        print(a, 'and', b, 'are perfectly equal')
        return True
    im_a = plt.imread(a) * 255.0
    im_b = plt.imread(b) * 255.0
    if im_a.shape != im_b.shape:
        print(a, 'and', b, 'have different size:', im_a.shape, im_b.shape)
        return False
    diff = im_b - im_a
    alpha = im_a.shape[-1] == 4
    if alpha:
        diff_alpha = diff[:, :, 3]

    equal = True
    print(a, 'and', b,
          'are different, analyzing whether it is just the undefined colors...'
    )
    print('Average difference (255=white): (R, G, B, A)')
    print(np.mean(np.mean(diff, 0), 0))
    print('Average difference with premultiplied alpha (255=white): '
          '(R, G, B, A)')
    diff = diff[:, :, 0:3]
    if alpha:
        diff *= plt.imread(a)[:, :, 3:4]
    res = np.mean(np.mean(diff, 0), 0)
    print(res)
    if np.mean(res) > 0.01:
        # dithering should make this value nearly zero...
        equal = False
    print('Maximum abs difference with premultiplied alpha (255=white): '
          '(R, G, B, A)')
    res = np.amax(np.amax(abs(diff), 0), 0)
    print(res)
    if max(abs(res)) > 1.1:
        # this error will be visible
        # - smaller errors are hidden by the weak alpha
        #   (but we should pay attention not to accumulate such errors at each load/save cycle...)
        equal = False

    if not equal:
        print('Not equal enough!')
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

    return equal


def docPaint():
    with open('brushes/s008.myb') as fp:
        b1 = brush.BrushInfo(fp.read())
    with open('brushes/redbrush.myb') as fp:
        b2 = brush.BrushInfo(fp.read())
    b2.set_color_hsv((0.3, 0.4, 0.35))
    with open('brushes/watercolor.myb') as fp:
        b3 = brush.BrushInfo(fp.read())
    b3.set_color_hsv((0.9, 0.2, 0.2))

    b = brush.BrushInfo()
    b.load_defaults()

    # test some actions
    doc = document.Document(b)
    doc.undo()  # nop
    events = np.loadtxt('painting30sec.dat')
    events = events[:len(events)/8]
    t_old = events[0][0]
    n = len(events)
    layer = doc.layer_stack.current
    for i, (t, x, y, pressure) in enumerate(events):
        dtime = t - t_old
        t_old = t
        #print dtime
        layer.stroke_to(doc.brush, x, y, pressure, 0.0, 0.0, dtime)
        if i == n*1/8:
            b.load_from_brushinfo(b2)
        if i == n*2/8:
            doc.clear_layer()
            doc.undo()
            assert not doc.get_bbox().empty()
            doc.redo()
            assert doc.get_bbox().empty()
        if i == n*3/8:
            doc.undo()
            b.load_from_brushinfo(b3)
        if i == n*4/8:
            b.load_from_brushinfo(b2)
        if i == n*5/8:
            doc.undo()
            doc.redo()
        if i == n*6/8:
            b.load_from_brushinfo(b2)
        if i == n*7/8:
            doc.add_layer(1)

    # If there is an eraser (or smudging) at work, we might be erasing
    # tiles that are empty. Those tile get memory allocated and affect
    # the bounding box of the layer. This shouldn't be a big issue, but
    # they get dropped when loading a document, which makes a
    # comparision of the PNG files fail. The hack below is to avoid that.
    for l in doc.layer_stack:
        l._surface.remove_empty_tiles()

    doc.layer_stack[0].save_as_png('test_docPaint_a.png')
    doc.layer_stack[0].save_as_png('test_docPaint_a1.png')
    # the resulting images will look slightly different because of dithering
    assert pngs_equal('test_docPaint_a.png', 'test_docPaint_a1.png')

    # test save/load
    doc.save('test_f1.ora')
    doc2 = document.Document()
    doc2.load('test_f1.ora')

    # (We don't preserve the absolute position of the image, only the size.)
    #assert doc.get_bbox() == doc2.get_bbox()
    print('doc / doc2 bbox:', doc.get_bbox(), doc2.get_bbox())

    doc2.layer_stack[0].save_as_png('test_docPaint_b.png')
    assert pngs_equal('test_docPaint_a.png', 'test_docPaint_b.png')
    doc2.save('test_f2.ora')
    #check not possible, because PNGs not exactly equal:
    #assert files_equal('test_f1.ora', 'test_f2.ora')

    # less strict test than above (just require load-save-load-save not to alter the file)
    doc3 = document.Document()
    doc3.load('test_f2.ora')
    assert doc2.get_bbox() == doc3.get_bbox()
    doc3.layer_stack[0].save_as_png('test_docPaint_c.png')
    assert pngs_equal('test_docPaint_b.png', 'test_docPaint_c.png')
    doc2.save('test_f3.ora')
    #check not possible, because PNGs not exactly equal:
    #assert files_equal('test_f2.ora', 'test_f3.ora')

    # note: this is not supposed to be strictly reproducible because
    # of different random seeds [huh? what does that mean?]
    bbox = doc.get_bbox()
    print('document bbox is', bbox)

    # test for appearance changes (make sure they are intended)
    doc.save('test_docPaint_flat.png', alpha=False)
    doc.save('test_docPaint_alpha.png', alpha=True)
    assert pngs_equal('test_docPaint_flat.png', 'correct_docPaint_flat.png')
    assert pngs_equal('test_docPaint_alpha.png', 'correct_docPaint_alpha.png')


def saveFrame():
    print('test-saving various frame sizes...')
    cnt = 0
    doc = document.Document()
    #doc.load('bigimage.ora')
    doc.set_frame_enabled(True)
    s = tiledsurface.Surface()

    N = mypaintlib.TILE_SIZE
    positions = range(-1, +2) + range(-N-1, -N+2) + range(+N-1, +N+2)
    for x1 in positions:
        for x2 in positions:
            for y1 in positions:
                for y2 in positions:
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cnt += 1
                    x, y, w, h = x1, y1, x2-x1, y2-y1
                    #print x, y, w, h
                    s.save_as_png('test_saveFrame_s.png', x, y, w, h)
                    doc.update_frame(x=x, y=y, width=w, height=h)
                    #doc.save('test_saveFrame_doc_%dx%d.png' % (w,h))
                    doc.save('test_saveFrame_doc.png')
                    doc.save('test_saveFrame_doc.jpg')
    print('checked', cnt, 'different rectangles')

from optparse import OptionParser
parser = OptionParser('usage: %prog [options]')
options, tests = parser.parse_args()

#tileConversions()
#layerModes()
directPaint()
brushPaint()
#    docPaint()

#saveFrame()

print('Tests passed.')
