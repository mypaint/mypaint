#!/usr/bin/env python
from pylab import *
from time import time
import sys, os, gc

os.chdir(os.path.dirname(sys.argv[0]))
sys.path.insert(0, '..')

from lib import mypaintlib, tiledsurface, brush, document, command

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
    b = brush.Brush()
    #b.load_from_string(open('../brushes/s006.myb').read())
    b.load_from_string(open('../brushes/charcoal.myb').read())

    events = loadtxt('painting30sec.dat.gz')

    b.set_color_rgb((0.0, 0.9, 1.0))

    t0 = time()
    for i in range(10):
        t_old = events[0][0]
        s.begin_atomic()
        for t, x, y, pressure in events:
            dtime = t - t_old
            t_old = t
            b.stroke_to (s, x, y, pressure, dtime)
        s.end_atomic()
    print 'Brushpaint time:', time()-t0
    print s.get_bbox(), b.stroke_total_painting_time # FIXME: why is this time so different each run?

    s.save('test_brushPaint.png')

def files_equal(a, b):
    return open(a, 'rb').read() == open(b, 'rb').read()

def pngs_equal(a, b, exact=True):
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

    equal = False
    if not exact:
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
    if mean(res) > 0.001:
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
            imshow(im_b[:,:,3])
            colorbar()
        figure(2)
        title('Green Error (multiplied with alpha)')
        imshow(diff[:,:,1])
        colorbar()
        if alpha:
            figure(3)
            title('Alpha Error')
            imshow(diff_alpha)
            colorbar()
        show()

    return equal

def docPaint():
    b1 = brush.Brush()
    b1.load_from_string(open('../brushes/s008.myb').read())
    b2 = brush.Brush()
    b2.load_from_string(open('../brushes/redbrush.myb').read())
    b2.set_color_hsv((0.3, 0.4, 0.35))

    # test some actions
    doc = document.Document()
    doc.undo() # nop
    events = loadtxt('painting30sec.dat.gz')
    events = events[:len(events)/8]
    t_old = events[0][0]
    n = len(events)
    for i, (t, x, y, pressure) in enumerate(events):
        dtime = t - t_old
        t_old = t
        #print dtime
        doc.stroke_to(dtime, x, y, pressure)
        if i == n*1/8:
            doc.set_brush(b2)
        if i == n*2/8:
            doc.clear_layer()
            doc.undo()
            assert not doc.get_bbox().empty()
            doc.redo()
            assert doc.get_bbox().empty()
        if i == n*3/8:
            doc.undo()
            doc.set_brush(b1)
        if i == n*4/8:
            doc.set_brush(b2)
        if i == n*5/8:
            doc.undo()
            doc.redo()
        if i == n*6/8:
            doc.set_brush(b2)
        if i == n*7/8:
            doc.add_layer(1)

    doc.layers[0].surface.save('test_docPaint_a.png')
    doc.layers[0].surface.save('test_docPaint_a1.png')
    # the resulting images will look slightly different because of dithering
    assert pngs_equal('test_docPaint_a.png', 'test_docPaint_a1.png', exact=False)

    # test save/load
    doc.save('test_f1.ora')
    doc2 = document.Document()
    doc2.load('test_f1.ora')
    print doc.get_bbox(), doc2.get_bbox()
    # TODO: fix this one?!
    #assert doc.get_bbox() == doc2.get_bbox()
    doc2.layers[0].surface.save('test_docPaint_b.png')
    assert pngs_equal('test_docPaint_a.png', 'test_docPaint_b.png', exact=False)
    doc2.save('test_f2.ora')
    #check not possible, because PNGs not exactly equal:
    #assert files_equal('test_f1.ora', 'test_f2.ora')

    # less strict test than above (just require load-save-load-save not to alter the file)
    doc3 = document.Document()
    doc3.load('test_f2.ora')
    assert doc2.get_bbox() == doc3.get_bbox()
    doc3.layers[0].surface.save('test_docPaint_c.png')
    assert pngs_equal('test_docPaint_b.png', 'test_docPaint_c.png', exact=False) # TODO: exact=True please
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
    assert pngs_equal('test_docPaint_flat.png', 'correct_docPaint_flat.png', exact=True)
    assert pngs_equal('test_docPaint_alpha.png', 'correct_docPaint_alpha.png', exact=True)

def mem():
    gc.collect()
    return int(open('/proc/self/statm').read().split()[0])

def check_garbage(msg = 'uncollectable garbage left over from previous tests'):
    gc.collect()
    garbage = []
    for obj in gc.garbage:
        # ignore garbage generated by numpy loadtxt command
        # http://projects.scipy.org/numpy/ticket/1356
        if hasattr(obj, 'filename') and obj.filename == 'painting30sec.dat.gz':
            continue
        garbage.append(obj)
    assert not garbage, 'uncollectable garbage left over from previous tests: %s' % garbage

def leakTest_fast():
    check_garbage()
    s = tiledsurface.Surface()
    del s
    check_garbage('surface class leaks memory (regression)')

def leakTest_generic(func):
    print 'memory leak test', func.__name__
    check_garbage()

    doc = document.Document()
    #gc.set_debug(gc.DEBUG_LEAK)

    m = []
    N = 21
    for i in range(N):
        func(doc, i)
        m2 = mem()
        m.append(m2)
        print 'iteration %02d/%02d: %d pages used' % (i+1, N, m2)

    #import objgraph
    #from lib import strokemap
    #objgraph.show_refs(doc)
    #sys.exit(0)

    # note: if gc.DEBUG_LEAK is enabled above this is expected to fail
    check_garbage()

    print m
    # we also have oscillations for some tests
    cmp_1st = m[N*1/3:N*2/3]
    cmp_2nd = m[N*2/3:N*3/3]
    diff = abs(max(cmp_1st) - max(cmp_2nd))
    if diff == 1:
        print 'FIXME: known minor leak ignored'
    else:
        assert diff == 0, 'looks like a memory leak in ' + func.__name__

    print 'no leak found'

def leakTest_slow():

    def paint(doc):
        events = loadtxt('painting30sec.dat.gz')
        t_old = events[0][0]
        for i, (t, x, y, pressure) in enumerate(events):
            dtime = t - t_old
            t_old = t
            doc.stroke_to(dtime, x, y, pressure)

    def paint_and_clear(doc, iteration):
        paint(doc)
        doc.clear()

    def repeated_saving(doc, iteration):
        if iteration == 0:
            paint(doc)
        doc.save('test_leak.ora')
        doc.save('test_leak.png')
        doc.save('test_leak.jpg')

    def repeated_loading(doc, iteration):
        doc.load('bigimage.ora')

    def paint_save_clear(doc, iteration):
        paint(doc)
        doc.save('test_leak.ora')
        doc.clear()

    def provoke_leak(doc, iteration):
        # note: interestingly this leaky only shows in the later iterations
        #       (and smaller leaks will not be detected)
        setattr(gc, 'my_test_leak_%d' % iteration, zeros(50000))

    #leakTest_generic(provoke_leak)
    leakTest_generic(paint_and_clear)
    leakTest_generic(repeated_saving)
    leakTest_generic(repeated_loading)
    leakTest_generic(paint_save_clear)

directPaint()
brushPaint()
docPaint()
leakTest_fast()
if '--leak' in sys.argv:
    leakTest_slow()
else:
    print
    print 'Skipping slow memory leak tests (use --leak to run them).'

print 'Tests passed.'
