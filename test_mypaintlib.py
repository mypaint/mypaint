#!/usr/bin/env python
from pylab import *
from StringIO import StringIO

from lib import mypaintlib, tiledsurface, brushsettings, brush, document, command

def directPaint():

    s = tiledsurface.TiledSurface()
    events = load('painting30sec.dat.gz')

    for t, x, y, pressure in events:
        r = g = b = 0.5*(1.0+sin(t))
        r *= 0.8
        s.draw_dab(x, y, 12, r, g, b, pressure, 0.6)
    s.save('test_directPaint.png')

def brushPaint():

    s = tiledsurface.TiledSurface()
    b = brush.Brush_Lowlevel()
    b.load_from_string(open('brushes/s006.myb').read())

    events = load('painting30sec.dat.gz')

    b.set_color_rgb((0.0, 0.9, 1.0))

    t_old = events[0][0]
    for t, x, y, pressure in events:
        dtime = t - t_old
        t_old = t
        b.tiled_surface_stroke_to (s, x, y, pressure, dtime)
    print s.get_bbox(), b.stroke_total_painting_time # FIXME: why is this time so different each run?

    s.save('test_brushPaint.png')

def files_equal(a, b):
    return open(a, 'rb').read() == open(b, 'rb').read()

def docPaint():
    b1 = brush.Brush_Lowlevel()
    b1.load_from_string(open('brushes/s006.myb').read())
    b2 = brush.Brush_Lowlevel()
    b2.load_from_string(open('brushes/s023.myb').read())

    # test all actions
    doc = document.Document()
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

    # test save/load
    f1 = StringIO()
    doc.save('test_f1.myp', compress=False)
    doc2 = document.Document()
    doc2.load('test_f1.myp', decompress=False)
    print doc.get_bbox(), doc2.get_bbox()
    assert doc.get_bbox() == doc2.get_bbox()
    doc2.save('test_f2.myp', compress=False)
    assert files_equal('test_f1.myp', 'test_f2.myp')
    doc2.layers[0].surface.save('test_docPaint_b.png')
    assert files_equal('test_docPaint_a.png', 'test_docPaint_b.png')
    while doc2.undo():
        pass
    assert doc2.get_bbox().empty()
    while doc2.redo():
        pass
    doc2.layers[0].surface.save('test_docPaint_c.png')
    assert files_equal('test_docPaint_a.png', 'test_docPaint_c.png')
    doc2.save('test_f3.myp', compress=False)
    assert files_equal('test_f1.myp', 'test_f3.myp')
    # TODO: add checks for the rendered buffers (random seed should be equal)

    # note: this is not supposed to be strictly reproducible because of different random seeds
    bbox = doc.get_bbox()
    print 'document bbox is', bbox


directPaint()
brushPaint()
docPaint()

print 'tests done'
