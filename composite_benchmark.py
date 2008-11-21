from scipy import *
#from pylab import *
from time import time
import sys
sys.path.insert(0, 'lib')
import mypaintlib

import gtk
gdk = gtk.gdk

iterations=1000
N=64

def benchmarkGdkPixbuf():
    print 'gdkPixbuf blitting 8bit RGBA on RGB'
    src = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, True, 8, N, N)  
    dst = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, N, N)  

    src.get_pixels_array()[:,:,:] = (rand(N,N,4)*255).astype('uint8')
    dst.get_pixels_array()[:,:,:] = (rand(N,N,3)*255).astype('uint8')

    t = time()
    for i in xrange(iterations):
        src.composite(dst, 0, 0, N, N, 0, 0, 1, 1, gtk.gdk.INTERP_NEAREST, 255)
    return time() - t

def benchmarkSciPy(t='float32'):
    print 'benchmarkSciPy', t
    src = rand(N,N,4).astype(t)
    dst = rand(N,N,4).astype(t)

    src_rgb = src[:,:,0:3].copy()
    src_a   = src[:,:,3: ].copy()
    dst_rgb = dst[:,:,0:3].copy()
    dst_a   = dst[:,:,3: ].copy()

    t = time()
    for i in xrange(iterations):
        src_a_ = 1.0-src_a
        dst_rgb = src_rgb * src_a + dst_rgb * src_a_
        dst_a = src_a + dst_a * src_a_
    return time() - t

def benchmarkSciPyPremulSlice(t='float32'):
    print 'benchmarkSciPyPremulSlice', t
    src = rand(N,N,4).astype(t)
    dst = rand(N,N,4).astype(t)

    t = time()
    for i in xrange(iterations):
        dst = src + dst - dst[:,:,3:]*dst
    return time() - t

def benchmarkSciPyPremul(t='float32'):
    print 'benchmarkSciPyPremul', t
    src = rand(N,N,4).astype(t)
    dst = rand(N,N,4).astype(t)

    src_rgb = src[:,:,0:3].copy()
    src_a   = src[:,:,3: ].copy()
    dst_rgb = dst[:,:,0:3].copy()
    dst_a   = dst[:,:,3: ].copy()

    t = time()
    for i in xrange(iterations):
        # resultColor = topColor + (1.0 - topAlpha) * bottomColor
        dst_rgb = src_rgb + dst_rgb - src_a*dst_rgb
        # resultAlpha = topAlpha + (1.0 - topAlpha) * bottomAlpha
        dst_a = src_a + dst_a - src_a*dst_a
    return time() - t

def benchmarkSciPyPremulOpt(t='float32'):
    print 'benchmarkSciPyPremulOpt', t
    src = rand(N,N,4).astype(t)
    dst = rand(N,N,4).astype(t)

    src_rgb = src[:,:,0:3].copy()
    src_a   = src[:,:,3: ].copy()
    dst_rgb = dst[:,:,0:3].copy()
    dst_a   = dst[:,:,3: ].copy()

    t = time()
    for i in xrange(iterations):
        # resultColor = topColor + (1.0 - topAlpha) * bottomColor
        dst_rgb += src_rgb
        dst_rgb -= src_a*dst_rgb
        # resultAlpha = topAlpha + (1.0 - topAlpha) * bottomAlpha
        dst_a += src_a
        dst_a -= src_a*dst_a
    return time() - t

def benchmark16bitPremulC():
    print 'benchmark16bitPremulC'
    src = (rand(N,N,4)*65535).astype('uint16')
    dst = (rand(N,N,3)*255).astype('uint8')

    t = time()
    for i in xrange(iterations):
        mypaintlib.composite_tile_over_rgb8(src, dst)
    return time() - t


a = benchmarkGdkPixbuf()
print a
b = benchmarkSciPyPremulOpt()
print b, b/a
c = benchmark16bitPremulC()
print c, c/a, c/b
