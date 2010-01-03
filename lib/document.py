# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os, zipfile, tempfile, time
join = os.path.join
import xml.etree.ElementTree as ET
from gtk import gdk
import gobject, numpy

import helpers, tiledsurface, pixbufsurface, backgroundsurface, mypaintlib
import command, stroke, layer
import brush
N = tiledsurface.N

class SaveLoadError(Exception):
    """Expected errors on loading or saving, like missing permissions or non-existing files."""
    pass

class Document():
    """
    This is the "model" in the Model-View-Controller design.
    (The "view" would be ../gui/tileddrawwidget.py.)
    It represents everything that the user would want to save.


    The "controller" mostly in drawwindow.py.
    It is possible to use it without any GUI attached (see ../tests/)
    """
    # Please note the following difficulty with the undo stack:
    #
    #   Most of the time there is an unfinished (but already rendered)
    #   stroke pending, which has to be turned into a command.Action
    #   or discarded as empty before any other action is possible.
    #   (split_stroke)

    def __init__(self):
        self.brush = brush.Brush()
        self.stroke = None
        self.canvas_observers = []
        self.stroke_observers = [] # callback arguments: stroke, brush (brush is a temporary read-only convenience object)
        self.doc_observers = []
        self.clear(True)

    def call_doc_observers(self):
        for f in self.doc_observers:
            f(self)
        return True

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

        self.call_doc_observers()

    def get_current_layer(self):
        return self.layers[self.layer_idx]
    layer = property(get_current_layer)

    def split_stroke(self):
        if not self.stroke: return
        self.stroke.stop_recording()
        if not self.stroke.empty:
            self.command_stack.do(command.Stroke(self, self.stroke, self.snapshot_before_stroke))
            del self.snapshot_before_stroke
            self.unsaved_painting_time += self.stroke.total_painting_time
            for f in self.stroke_observers:
                f(self.stroke, self.brush)
        self.stroke = None

    def select_layer(self, idx):
        self.do(command.SelectLayer(self, idx))

    def move_layer(self, was_idx, new_idx):
        self.do(command.MoveLayer(self, was_idx, new_idx))

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

    def straight_line(self, src, dst):
        self.split_stroke()
        # TODO: undo last stroke if it was very short... (but not at document level?)
        real_brush = self.brush
        self.brush = brush.Brush()
        self.brush.copy_settings_from(real_brush)

        duration = 3.0
        pressure = 0.3
        N = 1000
        x = numpy.linspace(src[0], dst[0], N)
        y = numpy.linspace(src[1], dst[1], N)
        # rest the brush in src for a minute, to avoid interpolation
        # from the upper left corner (states are zero) (FIXME: the
        # brush should handle this on its own, maybe?)
        self.stroke_to(60.0, x[0], y[0], 0.0)
        for i in xrange(N):
            self.stroke_to(duration/N, x[i], y[i], pressure)
        self.split_stroke()
        self.brush = real_brush


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

    def get_last_command(self):
        self.split_stroke()
        return self.command_stack.get_last_command()

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

    def blit_tile_into(self, dst_8bit, tx, ty, mipmap=0, layers=None, background=None):
        if layers is None:
            layers = self.layers
        if background is None:
            background = self.background

        assert dst_8bit.dtype == 'uint8'
        dst = numpy.empty((N, N, 3), dtype='uint16')

        background.blit_tile_into(dst, tx, ty, mipmap)

        for layer in layers:
            surface = layer.surface
            surface.composite_tile_over(dst, tx, ty, mipmap_level=mipmap, opacity=layer.opacity)

        mypaintlib.tile_convert_rgb16_to_rgb8(dst, dst_8bit)

    def add_layer(self, insert_idx=None, after=None):
        self.do(command.AddLayer(self, insert_idx, after))

    def remove_layer(self,layer=None):
        self.do(command.RemoveLayer(self,layer))

    def merge_layer(self, dst_idx):
        self.do(command.MergeLayer(self, dst_idx))

    def load_layer_from_pixbuf(self, pixbuf, x=0, y=0):
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        self.do(command.LoadLayer(self, arr, x, y))

    def set_layer_opacity(self, opacity):
        cmd = self.get_last_command()
        if isinstance(cmd, command.SetLayerOpacity):
            self.undo()
        self.do(command.SetLayerOpacity(self, opacity))

    def set_background(self, obj):
        # This is not an undoable action. One reason is that dragging
        # on the color chooser would get tons of undo steps.

        if not isinstance(obj, backgroundsurface.Background):
            obj = backgroundsurface.Background(obj)
        self.background = obj

        self.invalidate_all()

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
        self.split_stroke()
        trash, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        save = getattr(self, 'save_' + ext, self.unsupported)
        try:        
            save(filename, **kwargs)
        except gobject.GError, e:
            if  e.code == 5:
                #add a hint due to a very consfusing error message when there is no space left on device
                raise SaveLoadError, 'Unable to save: ' + e.message +  '\nDo you have enough space left on the device?'
            else:
                raise SaveLoadError, 'Unable to save: ' + e.message
        except IOError, e:
            raise SaveLoadError, 'Unable to save: ' + e.strerror
        self.unsaved_painting_time = 0.0

    def load(self, filename):
        if not os.path.isfile(filename):
            raise SaveLoadError, 'File does not exist: ' + repr(filename)
        if not os.access(filename,os.R_OK):
            raise SaveLoadError, 'You do not have the necessary permissions to open file: ' + repr(filename)
        trash, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        load = getattr(self, 'load_' + ext, self.unsupported)
        load(filename)
        self.command_stack.clear()
        self.unsaved_painting_time = 0.0
        self.call_doc_observers()

    def unsupported(self, filename):
        raise SaveLoadError, 'Unknown file format extension: ' + repr(filename)

    def render_as_pixbuf(self, *args):
        return pixbufsurface.render_as_pixbuf(self, *args)

    def save_png(self, filename, compression=2, alpha=False, multifile=False):
        if multifile:
            self.save_multifile_png(filename, compression)
        else:
            if alpha:
                tmp_layer = layer.Layer()
                for l in self.layers:
                    l.merge_into(tmp_layer)
                pixbuf = tmp_layer.surface.render_as_pixbuf()
            else:
                pixbuf = self.render_as_pixbuf()
            pixbuf.save(filename, 'png', {'compression':str(compression)})

    def save_multifile_png(self, filename, compression=2, alpha=False):
        prefix, ext = os.path.splitext(filename)
        # if we have a number already, strip it
        l = prefix.rsplit('.', 1)
        if l[-1].isdigit():
            prefix = l[0]
        doc_bbox = self.get_bbox()
        for i, l in enumerate(self.layers):
            filename = '%s.%03d%s' % (prefix, i+1, ext)
            l.surface.save(filename, *doc_bbox)

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
        a['w'] = str(w0)
        a['h'] = str(h0)

        def store_pixbuf(pixbuf, name):
            tmp = join(tempdir, 'tmp.png')
            t1 = time.time()
            pixbuf.save(tmp, 'png', {'compression':'2'})
            print '  %.3fs saving %s compression 2' % (time.time() - t1, name)
            z.write(tmp, name)
            os.remove(tmp)

        def add_layer(x, y, opac, pixbuf, name, layer_name):
            layer = ET.Element('layer')
            stack.append(layer)
            store_pixbuf(pixbuf, name)
            a = layer.attrib
            if layer_name:
                a['name'] = layer_name
            a['src'] = name
            a['x'] = str(x)
            a['y'] = str(y)
            a['opacity'] = str(opac)
            return layer

        for idx, l in enumerate(reversed(self.layers)):
            if l.surface.is_empty():
                continue
            opac = l.opacity
            x, y, w, h = l.surface.get_bbox()
            pixbuf = l.surface.render_as_pixbuf()

            el = add_layer(x-x0, y-y0, opac, pixbuf, 'data/layer%03d.png' % idx, l.name)
            # strokemap
            data = l.save_strokemap_to_string(-x, -y)
            name = 'data/layer%03d_strokemap.dat' % idx
            el.attrib['mypaint_strokemap'] = name
            write_file_str(name, data)


        # save background as layer (solid color or tiled)
        s = pixbufsurface.Surface(x0, y0, w0, h0)
        s.fill(self.background)
        l = add_layer(0, 0, 1.0, s.pixbuf, 'data/background.png', 'background')
        bg = self.background
        x, y, w, h = bg.get_pattern_bbox()
        pixbuf = pixbufsurface.render_as_pixbuf(bg, x+x0, y+y0, w, h, alpha=False)
        store_pixbuf(pixbuf, 'data/background_tile.png')
        l.attrib['background_tile'] = 'data/background_tile.png'

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

        helpers.indent_etree(image)
        xml = ET.tostring(image, encoding='UTF-8')

        write_file_str('stack.xml', xml)
        z.close()
        os.rmdir(tempdir)
        if os.path.exists(filename):
            os.remove(filename) # windows needs that
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

        def get_pixbuf(filename):
            t1 = time.time()
            tmp = join(tempdir, 'tmp.png')
            f = open(tmp, 'wb')
            f.write(z.read(filename))
            f.close()
            res = gdk.pixbuf_new_from_file(tmp)
            os.remove(tmp)
            print '  %.3fs loading %s' % (time.time() - t1, filename)
            return res

        def get_layers_list(root, x=0,y=0):
            res = []
            for item in root:
                if item.tag == 'layer':
                    if 'x' in item.attrib:
                        item.attrib['x'] = int(item.attrib['x']) + x
                    if 'y' in item.attrib:
                        item.attrib['y'] = int(item.attrib['y']) + y
                    res.append(item)
                elif item.tag == 'stack':
                    stack_x = int( item.attrib.get('x', 0) )
                    stack_y = int( item.attrib.get('y', 0) )
                    res += get_layers_list(item, stack_x, stack_y)
                else:
                    print 'Warning: ignoring unsupported tag:', item.tag
            return res

        self.clear() # this leaves one empty layer
        no_background = True
        for layer in get_layers_list(stack):
            a = layer.attrib

            if 'background_tile' in a:
                assert no_background
                try:
                    print a['background_tile']
                    self.set_background(get_pixbuf(a['background_tile']))
                    no_background = False
                    continue
                except backgroundsurface.BackgroundError, e:
                    print 'ORA background tile not usable:', e

            src = a.get('src', '')
            if not src.lower().endswith('.png'):
                print 'Warning: ignoring non-png layer'
                continue
            pixbuf = get_pixbuf(src)
            name = a.get('name', '')

            x = int(a.get('x', '0'))
            y = int(a.get('y', '0'))
            opac = float(a.get('opacity', '1.0'))
            self.add_layer(insert_idx=0)
            last_pixbuf = pixbuf
            t1 = time.time()
            self.load_layer_from_pixbuf(pixbuf, x, y)
            layer = self.layers[0]
            layer.name = name
            layer.opacity = helpers.clamp(opac, 0.0, 1.0)
            print '  %.3fs converting pixbuf to layer format' % (time.time() - t1)
            # strokemap
            fname = a.get('mypaint_strokemap', None)
            if fname:
                if x % N or y % N:
                    print 'Warning: dropping non-aligned strokemap'
                else:
                    data = z.read(fname)
                    self.layers[0].load_strokemap_from_string(data, x, y)

        os.rmdir(tempdir)

        if len(self.layers) == 1:
            raise ValueError, 'Could not load any layer.'

        if no_background:
            # recognize solid or tiled background layers, at least those that mypaint <= 0.7.1 saves
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
