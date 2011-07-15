# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os, zipfile, tempfile, time, traceback
join = os.path.join
from cStringIO import StringIO
import xml.etree.ElementTree as ET
from gtk import gdk
import gobject, numpy
from gettext import gettext as _

import helpers, tiledsurface, pixbufsurface, backgroundsurface, mypaintlib
import command, stroke, layer
import brush

N = tiledsurface.N
LOAD_CHUNK_SIZE = 64*1024

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

    def __init__(self, brushinfo=None):
        if not brushinfo:
            brushinfo = brush.BrushInfo()
            brushinfo.load_defaults()
        self.brush = brush.Brush(brushinfo)
        self.stroke = None
        self.canvas_observers = []
        self.stroke_observers = [] # callback arguments: stroke, brush (brush is a temporary read-only convenience object)
        self.doc_observers = []
        self.frame_observers = []
        self.clear(True)

        self._frame = [0, 0, 0, 0]
        self._frame_enabled = False
        # Used by move_frame() to accumulate values
        self._frame_dx = 0.0
        self._frame_dy = 0.0

    def get_frame(self):
        return self._frame

    def move_frame(self, dx=0.0, dy=0.0):
        """Move the frame. Accumulates changes and moves the frame once
        the accumulated change reaches the minimum move step."""
        # FIXME: Should be 1 (pixel aligned), not tile aligned
        # This is due to PNG saving having to be tile aligned
        min_step = N

        def round_to_n(value, n):
            return int(round(value/n)*n)

        x, y, w, h = self.get_frame()

        self._frame_dx += dx
        self._frame_dy += dy
        step_x = round_to_n(self._frame_dx, min_step)
        step_y = round_to_n(self._frame_dy, min_step)

        if step_x:
            self.set_frame(x=x+step_x)
            self._frame_dx -= step_x

        if step_y:
            self.set_frame(y=y+step_y)
            self._frame_dy -= step_y

    def set_frame(self, x=None, y=None, width=None, height=None):
        """Set the size of the frame. Pass None to indicate no-change."""

        for i, var in enumerate([x, y, width, height]):
            if not var is None:
                # FIXME: must be aligned to tile size due to PNG saving
                assert not var % N, "Frame size must be aligned to tile size"
                self._frame[i] = var

        for f in self.frame_observers: f()

    def get_frame_enabled(self):
        return self._frame_enabled

    def set_frame_enabled(self, enabled):
        self._frame_enabled = enabled
        for f in self.frame_observers: f()
    frame_enabled = property(get_frame_enabled)

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

    def move_layer(self, was_idx, new_idx, select_new=False):
        self.do(command.MoveLayer(self, was_idx, new_idx, select_new))

    def reorder_layers(self, new_layers):
        self.do(command.ReorderLayers(self, new_layers))

    def clear_layer(self):
        if not self.layer.surface.is_empty():
            self.do(command.ClearLayer(self))

    def stroke_to(self, dtime, x, y, pressure, xtilt,ytilt):
        if not self.stroke:
            self.stroke = stroke.Stroke()
            self.stroke.start_recording(self.brush)
            self.snapshot_before_stroke = self.layer.save_snapshot()
        self.stroke.record_event(dtime, x, y, pressure, xtilt,ytilt)

        l = self.layer
        l.surface.begin_atomic()
        split = self.brush.stroke_to (l.surface, x, y, pressure, xtilt,ytilt, dtime)
        l.surface.end_atomic()

        if split:
            self.split_stroke()

    def straight_line(self, src, dst):
        self.split_stroke()
        self.brush.reset() # reset dynamic states (eg. filtered velocity)

        duration = 3.0
        pressure = 0.3
        N = 1000
        x = numpy.linspace(src[0], dst[0], N)
        y = numpy.linspace(src[1], dst[1], N)
        # rest the brush in src for a minute, to avoid interpolation
        # from the upper left corner (states are zero) (FIXME: the
        # brush should handle this on its own, maybe?)
        self.stroke_to(60.0, x[0], y[0], 0.0, 0.0, 0.0)
        for i in xrange(N):
            self.stroke_to(duration/N, x[i], y[i], pressure, 0.0, 0.0)
        self.split_stroke()
        self.brush.reset()


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

    def get_bbox(self):
        res = helpers.Rect()
        for layer in self.layers:
            # OPTIMIZE: only visible layers...
            # careful: currently saving assumes that all layers are included
            bbox = layer.surface.get_bbox()
            res.expandToIncludeRect(bbox)
        return res

    def get_effective_bbox(self):
        """Return the effective bounding box of the document.
        If the frame is enabled, this is the bounding box of the frame, 
        else the (dynamic) bounding box of the document."""
        return self.get_frame() if self.frame_enabled else self.get_bbox()

    def blit_tile_into(self, dst_8bit, tx, ty, mipmap_level=0, layers=None, background=None):
        if layers is None:
            layers = self.layers
        if background is None:
            background = self.background

        assert dst_8bit.dtype == 'uint8'
        dst = numpy.empty((N, N, 3), dtype='uint16')

        background.blit_tile_into(dst, tx, ty, mipmap_level)

        for layer in layers:
            surface = layer.surface
            surface.composite_tile_over(dst, tx, ty, mipmap_level=mipmap_level, opacity=layer.effective_opacity)

        mypaintlib.tile_convert_rgb16_to_rgb8(dst, dst_8bit)

    def add_layer(self, insert_idx=None, after=None, name=''):
        self.do(command.AddLayer(self, insert_idx, after, name))

    def remove_layer(self,layer=None):
        if len(self.layers) > 1:
            self.do(command.RemoveLayer(self,layer))
        else:
            self.clear_layer()

    def merge_layer_down(self):
        dst_idx = self.layer_idx - 1
        if dst_idx < 0:
            return False
        self.do(command.MergeLayer(self, dst_idx))
        return True

    def load_layer_from_pixbuf(self, pixbuf, x=0, y=0):
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        self.do(command.LoadLayer(self, arr, x, y))

    def set_layer_visibility(self, visible, layer):
        cmd = self.get_last_command()
        if isinstance(cmd, command.SetLayerVisibility) and cmd.layer is layer:
            self.undo()
        self.do(command.SetLayerVisibility(self, visible, layer))

    def set_layer_locked(self, locked, layer):
        cmd = self.get_last_command()
        if isinstance(cmd, command.SetLayerLocked) and cmd.layer is layer:
            self.undo()
        self.do(command.SetLayerLocked(self, locked, layer))

    def set_layer_opacity(self, opacity, layer=None):
        """Sets the opacity of a layer. If layer=None, works on the current layer"""
        cmd = self.get_last_command()
        if isinstance(cmd, command.SetLayerOpacity):
            self.undo()
        self.do(command.SetLayerOpacity(self, opacity, layer))

    def set_background(self, obj):
        # This is not an undoable action. One reason is that dragging
        # on the color chooser would get tons of undo steps.

        if not isinstance(obj, backgroundsurface.Background):
            obj = backgroundsurface.Background(obj)
        self.background = obj

        self.invalidate_all()

    def load_from_pixbuf(self, pixbuf):
        """Load a document from a pixbuf."""
        self.clear()
        self.load_layer_from_pixbuf(pixbuf)
        self.set_frame(*self.get_bbox())

    def is_layered(self):
        count = 0
        for l in self.layers:
            if not l.surface.is_empty():
                count += 1
        return count > 1

    def is_empty(self):
        return len(self.layers) == 1 and self.layer.surface.is_empty()

    def save(self, filename, **kwargs):
        self.split_stroke()
        trash, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        save = getattr(self, 'save_' + ext, self.unsupported)
        try:        
            save(filename, **kwargs)
        except gobject.GError, e:
            traceback.print_exc()
            if e.code == 5:
                #add a hint due to a very consfusing error message when there is no space left on device
                raise SaveLoadError, _('Unable to save: %s\nDo you have enough space left on the device?') % e.message
            else:
                raise SaveLoadError, _('Unable to save: %s') % e.message
        except IOError, e:
            traceback.print_exc()
            raise SaveLoadError, _('Unable to save: %s') % e.strerror
        self.unsaved_painting_time = 0.0

    def load(self, filename, **kwargs):
        if not os.path.isfile(filename):
            raise SaveLoadError, _('File does not exist: %s') % repr(filename)
        if not os.access(filename,os.R_OK):
            raise SaveLoadError, _('You do not have the necessary permissions to open file: %s') % repr(filename)
        trash, ext = os.path.splitext(filename)
        ext = ext.lower().replace('.', '')
        load = getattr(self, 'load_' + ext, self.unsupported)
        try:
            load(filename, **kwargs)
        except gobject.GError, e:
            traceback.print_exc()
            raise SaveLoadError, _('Error while loading: GError %s') % e
        except IOError, e:
            traceback.print_exc()
            raise SaveLoadError, _('Error while loading: IOError %s') % e
        self.command_stack.clear()
        self.unsaved_painting_time = 0.0
        self.call_doc_observers()

    def unsupported(self, filename, *args, **kwargs):
        raise SaveLoadError, _('Unknown file format extension: %s') % repr(filename)

    def render_as_pixbuf(self, *args, **kwargs):
        return pixbufsurface.render_as_pixbuf(self, *args, **kwargs)

    def render_thumbnail(self):
        t0 = time.time()
        x, y, w, h = self.get_effective_bbox()
        mipmap_level = 0
        while mipmap_level < tiledsurface.MAX_MIPMAP_LEVEL and max(w, h) >= 512:
            mipmap_level += 1
            x, y, w, h = x/2, y/2, w/2, h/2

        pixbuf = self.render_as_pixbuf(x, y, w, h, mipmap_level=mipmap_level)
        assert pixbuf.get_width() == w and pixbuf.get_height() == h
        pixbuf = helpers.scale_proportionally(pixbuf, 256, 256)
        print 'Rendered thumbnail in', time.time() - t0, 'seconds.'
        return pixbuf

    def save_png(self, filename, alpha=False, multifile=False, **kwargs):
        doc_bbox = self.get_effective_bbox()
        if multifile:
            self.save_multifile_png(filename, **kwargs)
        else:
            if alpha:
                tmp_layer = layer.Layer()
                for l in self.layers:
                    l.merge_into(tmp_layer)
                tmp_layer.surface.save(filename, *doc_bbox)
            else:
                pixbufsurface.save_as_png(self, filename, *doc_bbox, alpha=False, **kwargs)

    def save_multifile_png(self, filename, alpha=False, **kwargs):
        prefix, ext = os.path.splitext(filename)
        # if we have a number already, strip it
        l = prefix.rsplit('.', 1)
        if l[-1].isdigit():
            prefix = l[0]
        doc_bbox = self.get_effective_bbox()
        for i, l in enumerate(self.layers):
            filename = '%s.%03d%s' % (prefix, i+1, ext)
            l.surface.save(filename, *doc_bbox, **kwargs)

    @staticmethod
    def _pixbuf_from_stream(fp, feedback_cb=None):
        loader = gdk.PixbufLoader()
        while True:
            if feedback_cb is not None:
                feedback_cb()
            buf = fp.read(LOAD_CHUNK_SIZE)
            if buf == '':
                break
            loader.write(buf)
        loader.close()
        return loader.get_pixbuf()

    def load_from_pixbuf_file(self, filename, feedback_cb=None):
        fp = open(filename, 'rb')
        pixbuf = self._pixbuf_from_stream(fp, feedback_cb)
        fp.close()
        self.load_from_pixbuf(pixbuf)

    load_png = load_from_pixbuf_file
    load_jpg = load_from_pixbuf_file
    load_jpeg = load_from_pixbuf_file

    def save_jpg(self, filename, quality=90, **kwargs):
        doc_bbox = self.get_effective_bbox()
        pixbuf = self.render_as_pixbuf(*doc_bbox, **kwargs)
        pixbuf.save(filename, 'jpeg', options={'quality':str(quality)})

    save_jpeg = save_jpg

    def save_ora(self, filename, options=None, **kwargs):
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
        x0, y0, w0, h0 = self.get_effective_bbox()
        a = image.attrib
        a['w'] = str(w0)
        a['h'] = str(h0)

        def store_pixbuf(pixbuf, name):
            tmp = join(tempdir, 'tmp.png')
            t1 = time.time()
            pixbuf.save(tmp, 'png')
            print '  %.3fs pixbuf saving %s' % (time.time() - t1, name)
            z.write(tmp, name)
            os.remove(tmp)

        def store_surface(surface, name, rect=[]):
            tmp = join(tempdir, 'tmp.png')
            t1 = time.time()
            surface.save(tmp, *rect, **kwargs)
            print '  %.3fs surface saving %s' % (time.time() - t1, name)
            z.write(tmp, name)
            os.remove(tmp)

        def add_layer(x, y, opac, surface, name, layer_name, visible=True, rect=[]):
            layer = ET.Element('layer')
            stack.append(layer)
            store_surface(surface, name, rect)
            a = layer.attrib
            if layer_name:
                a['name'] = layer_name
            a['src'] = name
            a['x'] = str(x)
            a['y'] = str(y)
            a['opacity'] = str(opac)
            if visible:
                a['visibility'] = 'visible'
            else:
                a['visibility'] = 'hidden'
            return layer

        for idx, l in enumerate(reversed(self.layers)):
            if l.surface.is_empty():
                continue
            opac = l.opacity
            x, y, w, h = l.surface.get_bbox()
            el = add_layer(x-x0, y-y0, opac, l.surface, 'data/layer%03d.png' % idx, l.name, l.visible, rect=(x, y, w, h))
            # strokemap
            sio = StringIO()
            l.save_strokemap_to_file(sio, -x, -y)
            data = sio.getvalue(); sio.close()
            name = 'data/layer%03d_strokemap.dat' % idx
            el.attrib['mypaint_strokemap_v2'] = name
            write_file_str(name, data)

        # save background as layer (solid color or tiled)
        bg = self.background
        # save as fully rendered layer
        x, y, w, h = self.get_bbox()
        l = add_layer(x-x0, y-y0, 1.0, bg, 'data/background.png', 'background', rect=(x,y,w,h))
        x, y, w, h = bg.get_pattern_bbox()
        # save as single pattern (with corrected origin)
        store_surface(bg, 'data/background_tile.png', rect=(x+x0, y+y0, w, h))
        l.attrib['background_tile'] = 'data/background_tile.png'

        # preview (256x256)
        t2 = time.time()
        print '  starting to render full image for thumbnail...'

        thumbnail_pixbuf = self.render_thumbnail()
        store_pixbuf(thumbnail_pixbuf, 'Thumbnails/thumbnail.png')
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

        return thumbnail_pixbuf

    def load_ora(self, filename, feedback_cb=None):
        """Loads from an OpenRaster file"""
        print 'load_ora:'
        t0 = time.time()
        z = zipfile.ZipFile(filename)
        print 'mimetype:', z.read('mimetype').strip()
        xml = z.read('stack.xml')
        image = ET.fromstring(xml)
        stack = image.find('stack')

        w = int(image.attrib['w'])
        h = int(image.attrib['h'])

        def round_up_to_n(value, n):
            assert value >= 0, "function undefined for negative numbers"

            residual = value % n
            if residual:
                value = value - residual + n
            return int(value)

        def get_pixbuf(filename):
            t1 = time.time()

            try:
                fp = z.open(filename, mode='r')
            except KeyError:
                # support for bad zip files (saved by old versions of the GIMP ORA plugin)
                fp = z.open(filename.encode('utf-8'), mode='r')
                print 'WARNING: bad OpenRaster ZIP file. There is an utf-8 encoded filename that does not have the utf-8 flag set:', repr(filename)

            res = self._pixbuf_from_stream(fp, feedback_cb)
            fp.close()
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
        # FIXME: don't require tile alignment for frame
        self.set_frame(width=round_up_to_n(w, N), height=round_up_to_n(h, N))

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
            visible = not 'hidden' in a.get('visibility', 'visible')
            self.add_layer(insert_idx=0, name=name)
            last_pixbuf = pixbuf
            t1 = time.time()
            self.load_layer_from_pixbuf(pixbuf, x, y)
            layer = self.layers[0]

            self.set_layer_opacity(helpers.clamp(opac, 0.0, 1.0), layer)
            self.set_layer_visibility(visible, layer)
            print '  %.3fs converting pixbuf to layer format' % (time.time() - t1)
            # strokemap
            fname = a.get('mypaint_strokemap_v2', None)
            if fname:
                if x % N or y % N:
                    print 'Warning: dropping non-aligned strokemap'
                else:
                    sio = StringIO(z.read(fname))
                    layer.load_strokemap_from_file(sio, x, y)
                    sio.close()

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
