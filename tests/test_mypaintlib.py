#!/usr/bin/env python
from pylab import *
from time import time
import sys, os, gc

os.chdir(os.path.dirname(sys.argv[0]))
sys.path.insert(0, '..')

from lib import mypaintlib, tiledsurface, brush, document, command, helpers

def tileConversions():
    # fully transparent tile stays fully transparent (without noise)
    N = mypaintlib.TILE_SIZE
    src = zeros((N, N, 4), 'uint16')
    dst = ones((N, N, 4), 'uint8')
    mypaintlib.tile_convert_rgba16_to_rgba8(src, dst)
    assert not dst.any()
    # fully opaque tile stays fully opaque
    src[:,:,3] = 1<<15
    src[:,:,:3] = randint(0, 1<<15, (N, N, 3))
    dst = zeros((N, N, 4), 'uint8')
    mypaintlib.tile_convert_rgba16_to_rgba8(src, dst)
    assert (dst[:,:,3] == 255).all()

def directPaint():

    s = tiledsurface.Surface()
    events = loadtxt('painting30sec.dat.gz')

    s.begin_atomic()
    for t, x, y, pressure in events:
        r = g = b = 0.5*(1.0+sin(t))
        r *= 0.8
        s.draw_dab(x, y, 12, r, g, b, pressure, 0.6)
    s.end_atomic()
    s.save('test_directPaint.png')

def brushPaint():

    s = tiledsurface.Surface()
    bi = brush.BrushInfo(open('brushes/charcoal.myb').read())
    b = brush.Brush(bi)

    events = loadtxt('painting30sec.dat.gz')

    bi.set_color_rgb((0.0, 0.9, 1.0))

    t0 = time()
    for i in range(10):
        t_old = events[0][0]
        s.begin_atomic()
        for t, x, y, pressure in events:
            dtime = t - t_old
            t_old = t
            b.stroke_to (s, x, y, pressure, 0.0, 0.0, dtime)
        s.end_atomic()
    print 'Brushpaint time:', time()-t0
    print s.get_bbox(), b.stroke_total_painting_time # FIXME: why is this time so different each run?

    s.save('test_brushPaint.png')

def files_equal(a, b):
    return open(a, 'rb').read() == open(b, 'rb').read()

def pngs_equal(a, b):
    if files_equal(a, b):
        print a, 'and', b, 'are perfectly equal'
        return True
    im_a = imread(a)*255.0
    im_b = imread(b)*255.0
    if im_a.shape != im_b.shape:
        print a, 'and', b, 'have different size:', im_a.shape, im_b.shape
        return False
    diff = im_b - im_a
    alpha = im_a.shape[-1] == 4
    if alpha:
        diff_alpha = diff[:,:,3]

    equal = True
    print a, 'and', b, 'are different, analyzing whether it is just the undefined colors...'
    print 'Average difference (255=white): (R, G, B, A)'
    print mean(mean(diff, 0), 0)
    print 'Average difference with premultiplied alpha (255=white): (R, G, B, A)'
    diff = diff[:,:,0:3]
    if alpha:
        diff *= imread(a)[:,:,3:4]
    res = mean(mean(diff, 0), 0)
    print res
    if mean(res) > 0.01:
        # dithering should make this value nearly zero...
        equal = False
    print 'Maximum abs difference with premultiplied alpha (255=white): (R, G, B, A)'
    res = amax(amax(abs(diff), 0), 0)
    print res
    if max(abs(res)) > 1.1:
        # this error will be visible
        # - smaller errors are hidden by the weak alpha
        #   (but we should pay attention not to accumulate such errors at each load/save cycle...)
        equal = False

    if not equal:
        print 'Not equal enough!'
        if alpha:
            figure(1)
            title('Alpha')
            imshow(im_b[:,:,3], interpolation='nearest')
            colorbar()
        figure(2)
        title('Green Error (multiplied with alpha)')
        imshow(diff[:,:,1], interpolation='nearest')
        colorbar()
        if alpha:
            figure(3)
            title('Alpha Error')
            imshow(diff_alpha, interpolation='nearest')
            colorbar()
        show()

    return equal

def docPaint():
    b1 = brush.BrushInfo(open('brushes/s008.myb').read())
    b2 = brush.BrushInfo(open('brushes/redbrush.myb').read())
    b2.set_color_hsv((0.3, 0.4, 0.35))
    b3 = brush.BrushInfo(open('brushes/watercolor.myb').read())
    b3.set_color_hsv((0.9, 0.2, 0.2))

    b = brush.BrushInfo()
    b.load_defaults()

    # test some actions
    doc = document.Document(b)
    doc.undo() # nop
    events = loadtxt('painting30sec.dat.gz')
    events = events[:len(events)/8]
    t_old = events[0][0]
    n = len(events)
    for i, (t, x, y, pressure) in enumerate(events):
        dtime = t - t_old
        t_old = t
        #print dtime
        doc.stroke_to(dtime, x, y, pressure, 0.0, 0.0)
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
    for l in doc.layers:
        l.surface.remove_empty_tiles()

    doc.layers[0].surface.save('test_docPaint_a.png')
    doc.layers[0].surface.save('test_docPaint_a1.png')
    # the resulting images will look slightly different because of dithering
    assert pngs_equal('test_docPaint_a.png', 'test_docPaint_a1.png')

    # test save/load
    doc.save('test_f1.ora')
    doc2 = document.Document()
    doc2.load('test_f1.ora')

    # (We don't preserve the absolute position of the image, only the size.)
    #assert doc.get_bbox() == doc2.get_bbox()
    print 'doc / doc2 bbox:', doc.get_bbox(), doc2.get_bbox()

    doc2.layers[0].surface.save('test_docPaint_b.png')
    assert pngs_equal('test_docPaint_a.png', 'test_docPaint_b.png')
    doc2.save('test_f2.ora')
    #check not possible, because PNGs not exactly equal:
    #assert files_equal('test_f1.ora', 'test_f2.ora')

    # less strict test than above (just require load-save-load-save not to alter the file)
    doc3 = document.Document()
    doc3.load('test_f2.ora')
    assert doc2.get_bbox() == doc3.get_bbox()
    doc3.layers[0].surface.save('test_docPaint_c.png')
    assert pngs_equal('test_docPaint_b.png', 'test_docPaint_c.png')
    doc2.save('test_f3.ora')
    #check not possible, because PNGs not exactly equal:
    #assert files_equal('test_f2.ora', 'test_f3.ora')

    # note: this is not supposed to be strictly reproducible because
    # of different random seeds [huh? what does that mean?]
    bbox = doc.get_bbox()
    print 'document bbox is', bbox

    # test for appearance changes (make sure they are intended)
    doc.save('test_docPaint_flat.png', alpha=False)
    doc.save('test_docPaint_alpha.png', alpha=True)
    assert pngs_equal('test_docPaint_flat.png', 'correct_docPaint_flat.png')
    assert pngs_equal('test_docPaint_alpha.png', 'correct_docPaint_alpha.png')

from optparse import OptionParser
parser = OptionParser('usage: %prog [options]')
options, tests = parser.parse_args()

tileConversions()
directPaint()
brushPaint()
docPaint()

print 'Tests passed.'
