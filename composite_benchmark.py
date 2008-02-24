from scipy import *
#from pylab import *
from time import time

import gtk
gdk = gtk.gdk

iterations=100
N=64

def benchmarkGdkPixbuf():
    src = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, True, 8, N, N)  
    dst = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, True, 8, N, N)  

    src.get_pixels_array()[:,:,:] = (rand(N,N,4)*255).astype('uint8')
    dst.get_pixels_array()[:,:,:] = (rand(N,N,4)*255).astype('uint8')

    t = time()
    for i in xrange(iterations):
        src.composite(dst, 0, 0, N, N, 0, 0, 1, 1, gtk.gdk.INTERP_NEAREST, 255)
    return time() - t

def benchmarkSciPy(t='float32'):
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
    src = rand(N,N,4).astype(t)
    dst = rand(N,N,4).astype(t)

    t = time()
    for i in xrange(iterations):
        dst = src + dst - dst[:,:,3:]*dst
    return time() - t

def benchmarkSciPyPremul(t='float32'):
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
    print dst_a.dtype
    return time() - t

def benchmarkSciPyPremulOpt(t='float32'):
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
    print dst_a.dtype
    return time() - t


a = benchmarkGdkPixbuf()
print a
b = benchmarkSciPyPremul()
print b, b/a
c = benchmarkSciPyPremulOpt()
print c, c/a, c/b
