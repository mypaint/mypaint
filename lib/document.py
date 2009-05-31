# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""
Design thoughts:
A stroke:
- is a list of motion events
- knows everything needed to draw itself (brush settings / initial brush state)
- has fixed brush settings (only brush states can change during a stroke)

A layer:
- is a container of several strokes (strokes can be removed)
- can be rendered as a whole
- can contain cache bitmaps, so it doesn't have to retrace all strokes all the time

A document:
- contains several layers
- knows the active layer and the current brush
- manages the undo history
- must be altered via undo/redo commands (except painting)
"""

import mypaintlib, helpers, tiledsurface, pixbufsurface
import command, stroke, layer, serialize
import brush # FIXME: the brush module depends on gtk and everything, but we only need brush_lowlevel
import gzip, os, zipfile, tempfile, numpy, time
join = os.path.join
import xml.etree.ElementTree as ET
from gtk import gdk

N = tiledsurface.N

class Document():
    """
    This is the "model" in the Model-View-Controller design.
    (The "view" would be ../gui/tileddrawwidget.py.)
    It represenst everything that the user would want to save.


    The "controller" mostly in drawwindow.py.
    It should be possible to use it without any GUI attached.
    
    Undo/redo is part of the model. The whole undo/redo stack can be
    saved to disk (planned) and can be used to reconstruct
    everything else.
    """
    # Please note the following difficulty:
    #
    #   Most of the time there is an unfinished (but already rendered)
    #   stroke pending, which has to be turned into a command.Action
    #   or discarded as empty before any other action is possible.
    #
    # TODO: the document should allow to "playback" (redo) a stroke
    # partially and examine its timing (realtime playback / calculate
    # total painting time) ?using half-done commands?

    def __init__(self):
        self.brush = brush.Brush_Lowlevel()
        self.stroke = None
        self.canvas_observers = []
        self.layer_observers = []  # callback arguments: x, y, w, h
        self.stroke_observers = [] # callback arguments: stroke, brush (brush is a temporary read-only convenience object)
        self.clear(True)

    def clear(self, init=False):
        self.split_stroke()
        if not init:
            bbox = self.get_bbox()
        # throw everything away, including undo stack
        self.command_stack = command.CommandStack()
        self.set_background((255, 255, 255))
        self.layers = []
        self.layer_idx = None
        self.add_layer(0)
        # disallow undo of the first layer
        self.command_stack.clear()
        self.unsaved_painting_time = 0.0

        if not init:
            for f in self.canvas_observers:
                f(*bbox)

    def get_current_layer(self):
        return self.layers[self.layer_idx]
    layer = property(get_current_layer)

    def split_stroke(self):
        if not self.stroke: return
        self.stroke.stop_recording()
        if not self.stroke.empty:
            self.layer.strokes.append(self.stroke)
            before = self.snapshot_before_stroke
            after = self.layer.save_snapshot()
            self.command_stack.do(command.Stroke(self, self.stroke, before, after))
            self.snapshot_before_stroke = after
            self.unsaved_painting_time += self.stroke.total_painting_time
            for f in self.stroke_observers:
                f(self.stroke, self.brush)
        self.stroke = None

    def select_layer(self, idx):
        self.do(command.SelectLayer(self, idx))

    def clear_layer(self):
        self.do(command.ClearLayer(self))

    def stroke_to(self, dtime, x, y, pressure):
        if not self.stroke:
            self.stroke = stroke.Stroke()
            self.stroke.start_recording(self.brush)
            self.snapshot_before_stroke = self.layer.save_snapshot()
        self.stroke.record_event(dtime, x, y, pressure)

        l = self.layer
        l.surface.begin_atomic()
        split = self.brush.stroke_to (l.surface, x, y, pressure, dtime)
        l.surface.end_atomic()

        if split:
            self.split_stroke()

    def layer_modified_cb(self, *args):
        # for now, any layer modification is assumed to be visible
        for f in self.canvas_observers:
            f(*args)

    def invalidate_all(self):
        for f in self.canvas_observers:
            f(0, 0, 0, 0)

    def undo(self):
        self.split_stroke()
        while 1:
            cmd = self.command_stack.undo()
            if not cmd or not cmd.automatic_undo:
                return cmd

    def redo(self):
        self.split_stroke()
        while 1:
            cmd = self.command_stack.redo()
            if not cmd or not cmd.automatic_undo:
                return cmd

    def do(self, cmd):
        self.split_stroke()
        self.command_stack.do(cmd)

    def set_brush(self, brush):
        self.split_stroke()
        self.brush.copy_settings_from(brush)

    def get_bbox(self):
        res = helpers.Rect()
        for layer in self.layers:
            # OPTIMIZE: only visible layers...
            # careful: currently saving assumes that all layers are included
            bbox = layer.surface.get_bbox()
            res.expandToIncludeRect(bbox)
        return res

    def blit_tile_into(self, dst, tx, ty, layers=None, background_memory=None):
        if layers is None:
            layers = self.layers
        if background_memory is None:
            background_memory = self.background_memory

        # render solid or tiled background
        #dst[:] = background_memory # 13 times slower than below, with some bursts having the same speed as below (huh?)
        # note: optimization for solid colors is not worth it any more now, even if it gives 2x speedup (at best)
        mypaintlib.tile_blit_rgb8_into_rgb8(background_memory, dst)

        for layer in layers:
            surface = layer.surface
            surface.composite_tile_over(dst, tx, ty)
            
    def add_layer(self, insert_idx):
        self.do(command.AddLayer(self, insert_idx))

    def remove_layer(self):
        self.do(command.RemoveLayer(self))

    def merge_layer(self, dst_idx):
        self.do(command.MergeLayer(self, dst_idx))

    def load_layer_from_pixbuf(self, pixbuf, x=0, y=0):
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        self.do(command.LoadLayer(self, arr, x, y))

    def set_background(self, obj):
        # This is not an undoable action. One reason is that dragging
        # on the color chooser would get tons of undo steps.
        try:
            obj = helpers.gdkpixbuf2numpy(obj)
        except:
            # it was already an array
            pass
        if len(obj) > 3:
            # simplify single-color pixmaps
            color = obj[0,0,:]
            if (obj == color).all():
                obj = tuple(color)
        self.background = obj

        # optimization
        self.background_memory = numpy.zeros((N, N, 3), dtype='uint8')
        self.background_memory[:,:,:] = self.background

        self.invalidate_all()

    def get_background_pixbuf(self):
        pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, N, N)
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        arr[:,:,:] = self.background
        return pixbuf

    def load_from_pixbuf(self, pixbuf):
        self.clear()
        self.load_layer_from_pixbuf(pixbuf)

    def is_layered(self):
        count = 0
        for l in self.layers:
            if not l.surface.is_empty():
                count += 1
        return count > 1

    def save(self, filename, **kwargs):
        trash, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        save = getattr(self, 'save_' + ext, self.unsupported)
        save(filename, **kwargs)
        self.unsaved_painting_time = 0.0

    def load(self, filename):
        trash, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        load = getattr(self, 'load_' + ext, self.unsupported)
        load(filename)
        self.command_stack.clear()
        self.unsaved_painting_time = 0.0

    def unsupported(self, filename):
        raise ValueError, 'Unkwnown file format extension: ' + repr(filename)

    def render_as_pixbuf(self, *args):
        return pixbufsurface.render_as_pixbuf(self, *args)

    def save_png(self, filename, compression=2, alpha=False):
        if alpha:
            tmp_layer = layer.Layer()
            for l in self.layers:
                l.merge_into(tmp_layer)
            pixbuf = tmp_layer.surface.render_as_pixbuf()
        else:
            pixbuf = self.render_as_pixbuf()
        pixbuf.save(filename, 'png', {'compression':str(compression)})

    def load_png(self, filename):
        self.load_from_pixbuf(gdk.pixbuf_new_from_file(filename))

    def load_jpg(self, filename):
        self.load_from_pixbuf(gdk.pixbuf_new_from_file(filename))
    load_jpeg = load_jpg

    def save_jpg(self, filename, quality=90):
        pixbuf = self.render_as_pixbuf()
        pixbuf.save(filename, 'jpeg', options={'quality':str(quality)})
    save_jpeg = save_jpg

    def save_ora(self, filename, options=None):
        print 'save_ora:'
        t0 = time.time()
        tempdir = tempfile.mkdtemp('mypaint')
        # use .tmp extension, so we don't overwrite a valid file if there is an exception
        z = zipfile.ZipFile(filename + '.tmpsave', 'w', compression=zipfile.ZIP_STORED)
        # work around a permission bug in the zipfile library: http://bugs.python.org/issue3394
        def write_file_str(filename, data):
            zi = zipfile.ZipInfo(filename)
            zi.external_attr = 0100644 << 16
            z.writestr(zi, data)
        write_file_str('mimetype', 'image/openraster') # must be the first file
        image = ET.Element('image')
        stack = ET.SubElement(image, 'stack')
        x0, y0, w0, h0 = self.get_bbox()
        a = image.attrib
        a['x'] = str(0)
        a['y'] = str(0)
        a['w'] = str(w0)
        a['h'] = str(h0)

        def store_pixbuf(pixbuf, name):
            tmp = join(tempdir, 'tmp.png')
            t1 = time.time()
            pixbuf.save(tmp, 'png', {'compression':'2'})
            print '  %.3fs saving %s compression 2' % (time.time() - t1, name)
            z.write(tmp, name)
            os.remove(tmp)

        def add_layer(x, y, pixbuf, name):
            layer = ET.Element('layer')
            stack.append(layer)
            store_pixbuf(pixbuf, name)
            a = layer.attrib
            a['src'] = name
            a['x'] = str(x)
            a['y'] = str(y)

        for idx, l in enumerate(reversed(self.layers)):
            if l.surface.is_empty():
                continue
            x, y, w, h = l.surface.get_bbox()
            pixbuf = l.surface.render_as_pixbuf()
            add_layer(x-x0, y-y0, pixbuf, 'data/layer%03d.png' % idx)

        # save background as layer (solid color or tiled)
        s = pixbufsurface.Surface(0, 0, w0, h0)
        s.fill(self.background)
        add_layer(0, 0, s.pixbuf, 'data/background.png')

        # preview
        t2 = time.time()
        print '  starting to render image for thumbnail...'
        pixbuf = self.render_as_pixbuf()
        w, h = pixbuf.get_width(), pixbuf.get_height()
        if w > h:
            w, h = 256, max(h*256/w, 1)
        else:
            w, h = max(w*256/h, 1), 256
        t1 = time.time()
        pixbuf = pixbuf.scale_simple(w, h, gdk.INTERP_BILINEAR)
        print '  %.3fs scaling thumbnail' % (time.time() - t1)
        store_pixbuf(pixbuf, 'Thumbnails/thumbnail.png')
        print '  total %.3fs spent on thumbnail' % (time.time() - t2)

        xml = ET.tostring(image, encoding='UTF-8')

        write_file_str('stack.xml', xml)
        z.close()
        os.rmdir(tempdir)
        os.rename(filename + '.tmpsave', filename)

        print '%.3fs save_ora total' % (time.time() - t0)

    def load_ora(self, filename):
        print 'load_ora:'
        t0 = time.time()
        tempdir = tempfile.mkdtemp('mypaint')
        z = zipfile.ZipFile(filename)
        print 'mimetype:', z.read('mimetype').strip()
        xml = z.read('stack.xml')
        image = ET.fromstring(xml)
        stack = image.find('stack')

        self.clear() # this leaves one empty layer
        for layer in stack:
            if layer.tag != 'layer':
                print 'Warning: ignoring unsupported tag:', layer.tag
                continue
            a = layer.attrib
            src = a.get('src', '')
            if not src.lower().endswith('.png'):
                print 'Warning: ignoring non-png layer'
                continue

            tmp = join(tempdir, 'tmp.png')
            f = open(tmp, 'wb')
            f.write(z.read(src))
            f.close()
            t1 = time.time()
            pixbuf = gdk.pixbuf_new_from_file(tmp)
            print '  %.3fs loading %s' % (time.time() - t1, src)
            os.remove(tmp)

            x = int(a.get('x', '0'))
            y = int(a.get('y', '0'))
            self.add_layer(insert_idx=0)
            last_pixbuf = pixbuf
            t1 = time.time()
            self.load_layer_from_pixbuf(pixbuf, x, y)
            print '  %.3fs converting pixbuf to layer format' % (time.time() - t1)

        os.rmdir(tempdir)

        if len(self.layers) == 1:
            raise ValueError, 'Could not load any layer.'

        # recognize solid or tiled background layers, at least those that mypaint saves
        # (OpenRaster will probably get generator layers for this some day)
        t1 = time.time()
        p = last_pixbuf
        if not p.get_has_alpha() and p.get_width() % N == 0 and p.get_height() % N == 0:
            tiles = self.layers[0].surface.tiledict.values()
            if len(tiles) > 1:
                all_equal = True
                for tile in tiles[1:]:
                    if (tile.rgba != tiles[0].rgba).any():
                        all_equal = False
                        break
                if all_equal:
                    arr = helpers.gdkpixbuf2numpy(p)
                    tile = arr[0:N,0:N,:]
                    self.set_background(tile.copy())
                    self.select_layer(0)
                    self.remove_layer()
        print '  %.3fs recognizing tiled background' % (time.time() - t1)

        if len(self.layers) > 1:
            # remove the still present initial empty top layer
            self.select_layer(len(self.layers)-1)
            self.remove_layer()
            # this leaves the topmost layer selected

        print '%.3fs load_ora total' % (time.time() - t0)
