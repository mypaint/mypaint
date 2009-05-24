#!/usr/bin/env python
from pylab import *
from StringIO import StringIO
from time import time

from lib import mypaintlib, tiledsurface, brushsettings, brush, document, command

def directPaint():

    s = tiledsurface.Surface()
    events = load('painting30sec.dat.gz')

    s.begin_atomic()
    for t, x, y, pressure in events:
        r = g = b = 0.5*(1.0+sin(t))
        r *= 0.8
        s.draw_dab(x, y, 12, r, g, b, pressure, 0.6)
    s.end_atomic()
    s.save('test_directPaint.png')

def brushPaint():

    s = tiledsurface.Surface()
    b = brush.Brush_Lowlevel()
    #b.load_from_string(open('brushes/s006.myb').read())
    b.load_from_string(open('brushes/charcoal.myb').read())

    events = load('painting30sec.dat.gz')

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

def pngs_equal(a, b):
    if files_equal(a, b):
        return True
    diff = imread(b) - imread(a)

    print a, 'and', b, 'are different, analyzing whether it is just the undefined colors...'
    print 'Average difference (255=white): (R, G, B, A)'
    print mean(mean(diff, 0), 0) * 255
    print 'Average difference with premultiplied alpha (255=white): (R, G, B, A)'
    diff2 = diff[:,:,0:3] * imread(a)[:,:,3:4]
    res = mean(mean(diff2, 0), 0) * 255
    print res

    res = max(abs(res)) < 0.1
    if res == False:
        print 'Not equal enough!'
        figure(1)
        title('Combined')
        imshow(diff)
        figure(2)
        title('Green')
        colorbar()
        imshow(diff[:,:,1])
        figure(3)
        title('Alpha')
        colorbar()
        imshow(diff[:,:,3])
        show()

    return res

def docPaint():
    b1 = brush.Brush_Lowlevel()
    b1.load_from_string(open('brushes/s008.myb').read())
    b2 = brush.Brush_Lowlevel()
    b2.load_from_string(open('brushes/redbrush.myb').read())
    b2.set_color_hsv((0.3, 0.4, 0.35))

    # test some actions
    doc = document.Document()
    doc.undo() # nop
    events = load('painting30sec.dat.gz')
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

    doc.layers[0].surface.save('test_docPaint_a.png')
    doc.layers[0].surface.save('test_docPaint_a1.png')
    assert pngs_equal('test_docPaint_a.png', 'test_docPaint_a1.png')

    # test save/load
    f1 = StringIO()
    doc.save('test_f1.ora')
    doc2 = document.Document()
    doc2.load('test_f1.ora')
    print doc.get_bbox(), doc2.get_bbox()
    # TODO: fix this one?!
    #assert doc.get_bbox() == doc2.get_bbox()
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


directPaint()
brushPaint()
docPaint()

print 'tests done'
