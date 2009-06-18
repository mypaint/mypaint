#!/usr/bin/env python

def renderperf():
    import gtk
    from time import time
    import pylab
    
    #moved out from gui/drawwindow.py class Window
    def start_profiling(inst): #where inst = instance of drawwindow.Window
        def autopaint():
            events = pylab.load('painting30sec.dat.gz')
            events[:,0] *= 0.3
            events = list(events)
            t0 = time()
            t_old = 0.0
            for t, x, y, pressure in events:
                sleeptime = t-(time()-t0)
                if sleeptime > 0.001:
                    yield sleeptime
                dtime = t - t_old
                t_old = t
                inst.doc.stroke_to(dtime, x, y, pressure)
            print 'replay done.'
            print inst.repaints, 'repaints'
            gtk.main_quit()
            yield 10.0

        import gobject
        p = autopaint()
        def timer_cb():
            gobject.timeout_add(int(p.next()*1000.0), timer_cb)

        inst.repaints = 0
        oldfunc=inst.tdw.repaint
        def count_repaints(*args, **kwargs):
            inst.repaints += 1
            return oldfunc(*args, **kwargs)
        inst.tdw.repaint = count_repaints
        timer_cb()

        import math
        inst.tdw.rotate(46.0/360*2*math.pi)


    from gui import main
    import mypaint
    data, conf = mypaint.get_paths()
    conf = '/tmp/randomdir228283' #fresh config
    app = main.main(data, conf, standalone = False)
    start_profiling(app.drawWindow)
    gtk.main()

def strokeperf():
    #from library
    raise NotImplementedError

def startuptime():
    #take a similar approach as renderperf() ?
    raise NotImplementedError

if __name__ == '__main__':
    renderperf()
