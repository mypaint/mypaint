from pylab import *
from mypaintlib import document, command, brush

doc = document.Document()

#b = brush.Brush_Lowlevel()
#b.load_from_string(open('brushes/s006.myb').read())

events = load('painting30sec.dat.gz')
t_old = events[0][0]
for t, x, y, pressure in events:
    dtime = t - t_old
    t_old = t
    doc.stroke_to(dtime, x, y, pressure)


print doc.get_bbox()
#x, y, w, h = doc.get_bbox()
#doc.render()

s = doc.layers[0].surface
s.save('docPaint.png')

