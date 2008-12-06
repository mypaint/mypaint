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

import mypaintlib, helpers, tiledsurface, command, stroke, layer, serialize
import brush # FIXME: the brush module depends on gtk and everything, but we only need brush_lowlevel
import random, gc, gzip, os
import numpy
import gtk

class Document():
    # This is the "model" in the Model-View-Controller design.
    # It should be possible to use it without any GUI attached.
    #
    # Undo/redo is part of the model. The whole undo/redo stack can be
    # saved to disk (planned) and can be used to reconstruct
    # everything else.
    #
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
        self.layers = []
        self.layer_idx = None
        self.add_layer(0)
        # disallow undo of the first layer (TODO: deleting the last layer should clear it instead)
        self.command_stack = command.CommandStack()

        if not init:
            for f in self.canvas_observers:
                f(*bbox)

    def split_stroke(self):
        if not self.stroke: return
        self.stroke.stop_recording()
        if not self.stroke.empty:
            l = self.layers[self.layer_idx]
            l.rendered.strokes.append(self.stroke)
            l.populate_cache()
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

        layer = self.layers[self.layer_idx]
        layer.surface.begin_atomic()
        split = self.brush.stroke_to (layer.surface, x, y, pressure, dtime)
        layer.surface.end_atomic()

        if split:
            self.split_stroke()

    def layer_modified_cb(self, *args):
        # for now, any layer modification is assumed to be visible
        for f in self.canvas_observers:
            f(*args)

    def change_brush(self, brush):
        self.split_stroke()
        assert not self.stroke
        self.brush.copy_settings_from(brush)

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

    # rendering etc.
    def get_bbox(self):
        res = helpers.Rect()
        for layer in self.layers:
            # OPTIMIZE: only visible layers...
            bbox = layer.surface.get_bbox()
            res.expandToIncludeRect(bbox)
        return res

    def get_tiles(self):
        # OPTIMIZE: this is used for rendering, so, only visible tiles?
        #           on the other hand, visibility can be checked later too
        tiles = set()
        for l in self.layers:
            tiles.update(l.get_tiles())
        return tiles

    def composite_tile(self, dst, tx, ty, layers=None):
        # OPTIMIZE: should use some caching strategy for the results somewhere, probably not here
        if layers is None:
            layers = self.layers
        for layer in layers:
            surface = layer.surface
            surface.composite_tile(dst, tx, ty)
            
    def render(self, dst, px, py, layers=None):
        assert dst.shape[2] == 3, 'RGB only for now'
        assert px == 0 and py == 0, 'not implemented'
        N = tiledsurface.N
        h, w, trash = dst.shape
        # FIXME: code duplication with tileddrawwidget.repaint()
        for tx, ty in self.get_tiles():
            x = tx*N
            y = ty*N
            if x < 0 or x+N > w: continue
            if y < 0 or y+N > h: continue
            self.composite_tile(dst[y:y+N,x:x+N], tx, ty, layers)

    def get_total_painting_time(self):
        t = 0.0
        for cmd in self.command_stack.undo_stack:
            if isinstance(cmd, command.Stroke):
                t += cmd.stroke.total_painting_time
        return t

    def render_as_pixbuf(self, x, y, w, h, layers=None):
        from gtk import gdk
        pixbuf = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, w, h)
        pixbuf.fill(0xffffffff)
        arr = pixbuf.get_pixels_array()
        arr = mypaintlib.gdkpixbuf2numpy(arr)
        self.render(arr, -x, -y, layers)
        return pixbuf

    def render_current_layer_as_pixbuf(self):
        l = self.layers[self.layer_idx]
        bbox = list(l.surface.get_bbox())
        return self.render_as_pixbuf(*bbox + [[l]])

    def add_layer(self, insert_idx=None):
        if insert_idx is None:
            insert_idx = self.layer_idx+1
        self.do(command.AddLayer(self, insert_idx))

    def load_layer_from_data(self, data):
        self.do(command.LoadLayer(self, data))

    def load_layer_from_pixbuf(self, pixbuf):
        arr = pixbuf.get_pixels_array()
        arr = mypaintlib.gdkpixbuf2numpy(arr)
        self.load_layer_from_data(arr)

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

    def save_png(self, filename):
        self.render_as_pixbuf().save(filename)

    def load_png(self, filename):
        self.load_from_pixbuf(gtk.gdk.pixbuf_new_from_file(filename))

    def save_myp(self, filename, compress=True):
        print 'WARNING: save/load file format is experimental'
        NEEDS_REWRITE
        self.split_stroke()
        if compress:
            f = gzip.GzipFile(filename, 'wb')
        else:
            f = open(filename, 'wb')
        f.write('MyPaint document\n1\n\n')
        #self.command_stack.serialize(f)
        for cmd in self.command_stack.undo_stack:
            # FIXME: ugly design
            # FIXME: do we really want to stay backwards compatible with all those internals on the undo stack?
            #        (and in the brush dab rendering code, etc.)
            if isinstance(cmd, command.Stroke):
                f.write('Stroke\n')
                serialize.save(cmd.stroke, f)
            elif isinstance(cmd, command.ClearLayer):
                f.write('ClearLayer\n')
            elif isinstance(cmd, command.AddLayer):
                f.write('AddLayer %d\n' % cmd.insert_idx)
            else:
                assert False, 'save not implemented for %s' % cmd
        f.close()

    def load_myp(self, filename, decompress=True):
        print 'WARNING: save/load file format is experimental'
        NEEDS_REWRITE
        self.clear()
        if decompress:
            f = gzip.GzipFile(filename, 'rb')
        else:
            f = open(filename, 'rb')
        assert f.readline() == 'MyPaint document\n'
        version = f.readline()
        assert version == '1\n'
        # skip lines to allow backwards compatible extensions
        while f.readline() != '\n':
            pass

        while 1:
            cmd = f.readline()
            if not cmd:
                break
            cmd, parts = cmd.split()[0], cmd.split()[1:]
            if cmd == 'Stroke':
                # FIXME: this code should probably be in command.py
                stroke_ = stroke.Stroke()
                serialize.load(stroke_, f)
                cmd = command.Stroke(self, stroke_)
                self.command_stack.do(cmd)
            elif cmd == 'ClearLayer':
                layer_idx = int(parts[0])
                cmd = command.ClearLayer(self, layer_idx)
                self.command_stack.do(cmd)
            elif cmd == 'AddLayer':
                insert_idx = int(parts[0])
                cmd = command.AddLayer(self, insert_idx)
                self.command_stack.do(cmd)
            else:
                assert False, 'unknown command %s' % cmd
        assert not f.read()
