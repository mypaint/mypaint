# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

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
import gzip, os, zipfile, tempfile
join = os.path.join
import xml.etree.ElementTree as ET
from gtk import gdk

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
        self.layer_observers = []

        self.clear(True)

    def clear(self, init=False):
        self.split_stroke()
        if not init:
            bbox = self.get_bbox()
        # throw everything away, including undo stack
        self.command_stack = command.CommandStack()
        self.background = (255, 255, 255)
        self.layers = []
        self.layer_idx = None
        self.add_layer(0)
        # disallow undo of the first layer (TODO: deleting the last layer should clear it instead)
        self.command_stack = command.CommandStack()

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
            self.layer.new_stroke_rendered_on_surface(self.stroke)
            self.command_stack.do(command.Stroke(self, self.stroke))
        self.stroke = None

    def select_layer(self, idx):
        self.do(command.SelectLayer(self, idx))

    def clear_layer(self):
        self.do(command.ClearLayer(self))

    def stroke_to(self, dtime, x, y, pressure):
        if not self.stroke:
            self.stroke = stroke.Stroke()
            self.stroke.start_recording(self.brush)
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

    def blit_tile_into(self, dst, tx, ty, layers=None):
        if layers is None:
            layers = self.layers

        # render solid white background (planned: something like self.background.blit_tile())
        assert dst.shape[2] == 3, 'RGB destination expected'
        N = tiledsurface.N
        dst[:N,:N,:] = self.background

        for layer in layers:
            surface = layer.surface
            surface.composite_tile_over(dst, tx, ty)
            
    def get_total_painting_time(self):
        t = 0.0
        for cmd in self.command_stack.undo_stack:
            if isinstance(cmd, command.Stroke):
                t += cmd.stroke.total_painting_time
        return t

    def add_layer(self, insert_idx):
        self.do(command.AddLayer(self, insert_idx))

    def remove_layer(self):
        self.do(command.RemoveLayer(self))

    def load_layer_from_pixbuf(self, pixbuf, x=0, y=0):
        arr = pixbuf.get_pixels_array()
        arr = mypaintlib.gdkpixbuf2numpy(arr)
        self.do(command.LoadLayer(self, arr, x, y))

    def set_background(self, obj):
        # This is not an undoable action. One reason is that dragging
        # on the color chooser would get tons of undo steps.
        try:
            obj = obj.get_pixels_array()
            obj = mypaintlib.gdkpixbuf2numpy(obj)
        except:
            # it was already an array
            pass
        if len(obj) > 3:
            # simplify single-color pixmaps
            color = obj[0,0,:]
            if (obj == color).all():
                obj = list(color)
        self.background = obj
        self.invalidate_all()

    def get_background_pixbuf(self):
        N = tiledsurface.N
        pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, N, N)
        arr = pixbuf.get_pixels_array()
        arr = mypaintlib.gdkpixbuf2numpy(arr)
        arr[:,:,:] = self.background
        return pixbuf

    def load_from_pixbuf(self, pixbuf):
        self.clear()
        self.load_layer_from_pixbuf(pixbuf)

    def save(self, filename):
        trash, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        print ext
        save = getattr(self, 'save_' + ext, self.unsupported)
        save(filename)

    def load(self, filename):
        trash, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        load = getattr(self, 'load_' + ext, self.unsupported)
        load(filename)

    def unsupported(self, filename):
        raise ValueError, 'Unkwnown file format extension: ' + repr(filename)

    def render_as_pixbuf(self, *args):
        return pixbufsurface.render_as_pixbuf(self, *args)

    def save_png(self, filename):
        pixbuf = self.render_as_pixbuf()
        pixbuf.save(filename, 'png')

    def load_png(self, filename):
        self.load_from_pixbuf(gdk.pixbuf_new_from_file(filename))

    def save_ora(self, filename):
        tempdir = tempfile.mkdtemp('mypaint')
        z = zipfile.ZipFile(filename, 'w', compression=zipfile.ZIP_STORED)
        # work around a permission bug in the zipfile library: http://bugs.python.org/issue3394
        def write_file_str(filename, data):
            zi = zipfile.ZipInfo(filename)
            zi.external_attr = 0100644 << 16
            z.writestr(zi, data)
        write_file_str('mimetype', 'image/openraster') # Mime type must be the first object stored. FIXME: what should go here?
        image = ET.Element('image')
        stack = ET.SubElement(image, 'stack')
        x0, y0, w0, h0 = self.get_bbox()
        a = image.attrib
        a['x'] = str(0)
        a['y'] = str(0)
        a['w'] = str(w0)
        a['h'] = str(h0)

        def add_layer(x, y, pixbuf, name):
            layer = ET.Element('layer')
            stack.append(layer)

            tmp = join(tempdir, 'tmp.png')
            pixbuf.save(tmp, 'png')
            z.write(tmp, name)
            os.remove(tmp)

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

        xml = ET.tostring(image, encoding='UTF-8')

        write_file_str('stack.xml', xml)
        z.close()
        os.rmdir(tempdir)

    def load_ora(self, filename):
        tempdir = tempfile.mkdtemp('mypaint')
        z = zipfile.ZipFile(filename)
        print 'mimetype:', z.read('mimetype').strip()
        xml = z.read('stack.xml')
        image = ET.fromstring(xml)
        stack = image.find('stack')

        self.clear()
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
            f = open(tmp, 'w')
            f.write(z.read(src))
            f.close()
            pixbuf = gdk.pixbuf_new_from_file(tmp)
            os.remove(tmp)

            x = int(a.get('x', '0'))
            y = int(a.get('y', '0'))
            self.add_layer(insert_idx=0)
            last_pixbuf = pixbuf
            self.load_layer_from_pixbuf(pixbuf, x, y)

        os.rmdir(tempdir)

        if len(self.layers) == 1:
            raise ValueError, 'Could not load any layer.'

        # recognize solid or tiled background layers (at least those that mypaint saves)
        # (OpenRaster will probably get generator layers for this some day)
        N = tiledsurface.N
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
                    arr = p.get_pixels_array()
                    arr = mypaintlib.gdkpixbuf2numpy(arr)
                    tile = arr[0:N,0:N,:]
                    self.set_background(tile.copy())
                    self.select_layer(0)
                    self.remove_layer()

        if len(self.layers) > 1:
            # select the still present initial empty top layer
            # hm, should this better be removed?
            self.select_layer(len(self.layers)-1)

