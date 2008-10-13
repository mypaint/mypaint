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

import helpers, tiledsurface, command, stroke, layer, serialize
import brush # FIXME: the brush module depends on gtk and everything, but we only need brush_lowlevel
import random, gc
import numpy

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

        self.clear()

    def clear(self):
        self.split_stroke()
        # throw everything away, including undo stack
        self.layer = layer.Layer()
        self.layer.surface.observers.append(self.layer_modified_cb)
        self.layers = [self.layer]
        self.command_stack = command.CommandStack()

    def split_stroke(self):
        if not self.stroke: return
        self.stroke.stop_recording()
        if not self.stroke.empty:
            self.layer.rendered.strokes.append(self.stroke)
            self.layer.populate_cache()
            self.command_stack.do(command.Stroke(self, self.layers.index(self.layer), self.stroke))
        self.stroke = None

    def reset(self):
        assert False, 'depreciated, use clear'

    def clear_layer(self):
        self.do(command.ClearLayer(self, self.layers.index(self.layer)))

    def stroke_to(self, dtime, x, y, pressure):
        if not self.stroke:
            self.stroke = stroke.Stroke()
            self.stroke.start_recording(self.brush)
        self.stroke.record_event(dtime, x, y, pressure)

        split = self.brush.tiled_surface_stroke_to (self.layer.surface, x, y, pressure, dtime)

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

    def get_last_command(self): 
        XXX # or check, without side-effect?
        self.split_stroke()
        return self.command_stack.get_last_command()

    def undo(self):
        self.split_stroke()
        return self.command_stack.undo()

    def redo(self):
        self.split_stroke()
        return self.command_stack.redo()


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

    def get_total_painting_time(self):
        t = 0.0
        for cmd in self.command_stack.undo_stack:
            if isinstance(cmd, command.Stroke):
                t += cmd.stroke.total_painting_time
        return t

    def save(self, f):
        self.split_stroke()
        if isinstance(f, str):
            f = open(f, 'wb')
        f.write('MyPaint document\n1\n\n')
        #self.command_stack.serialize(f)
        for cmd in self.command_stack.undo_stack:
            # FIXME: ugly design
            if isinstance(cmd, command.Stroke):
                f.write('Stroke %d\n' % cmd.layer_idx)
                serialize.save(cmd.stroke, f)
            elif isinstance(cmd, command.ClearLayer):
                f.write('ClearLayer %d\n' % cmd.layer_idx)
            else:
                assert False, 'save not implemented for %s' % cmd

    def load(self, f):
        self.clear()
        if isinstance(f, str):
            f = open(f, 'rb')
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
                layer_idx = int(parts[0])
                stroke_ = stroke.Stroke()
                serialize.load(stroke_, f)
                cmd = command.Stroke(self, layer_idx, stroke_)
                self.command_stack.do(cmd)
            elif cmd == 'ClearLayer':
                layer_idx = int(parts[0])
                cmd = command.ClearLayer(self, layer_idx)
                self.command_stack.do(cmd)
            else:
                assert False, 'unknown command %s' % cmd
        assert not f.read()
