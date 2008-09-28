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

import helpers, tilelib, command, stroke, layer
import brush # FIXME: the brush module depends on gtk and everything, but we only need brush_lowlevel
import random, gc
import numpy

class Document():
    # This is the "model" in the Model-View-Controller design.
    # It should be possible to use it without any GUI attached.
    #
    # - the document allows to "playback" a stroke partially and examine its timing
    #
    # Please note the following difficulties:
    #
    #   It is allowed to have "unfinished" actions on the stack. In
    #   particular a stroke action can be partially done.
    #
    #   We allow painting without pressure, and pressure without
    #   painting. To find out if some motion event starts a new
    #   undoable stroke, we have to let the brush execute it on the
    #   surface. To record the stroke, we need to save the dynamic
    #   states of the brush /before/ it modifies the surface.

    def __init__(self):
        self.brush = None
        self.layer = layer.Layer()
        self.layer.surface.observers.append(self.layer_modified_cb)
        self.layers = [self.layer]

        self.brush = brush.Brush_Lowlevel()

        # NOTE: does recording / replay code really belong to the model?
        #       should it better be part of the controller?
        #       ==> No. It is required also when loading a painting without GUI.
        #           And since the model includes stroke as "list-of-events",
        #           we might as well record them here.
        #       But how will playback work?
        #       ==> Basically like redo-redo-redo but splitting each stroke
        #           into tiny bits.
        #           Add a command that allows "redo 0.091 seconds".
        #           The command is then "half-redone" and will

        self.stroke = stroke.Stroke()
        self.stroke.start_recording(self.brush)

        self.command_stack = command.CommandStack()

        self.canvas_observers = []

    def stroke_to(self, dtime, x, y, pressure):
        if not self.stroke:
            self.stroke = stroke.Stroke()
            self.stroke.start_recording(self.brush)
        self.stroke.record_event(dtime, x, y, pressure)

        split_stroke = self.brush.tiled_surface_stroke_to (self.layer.surface, x, y, pressure, dtime)

        if split_stroke:
            self.split_stroke()

    def layer_modified_cb(self, *args):
        # for now, any layer modification is assumed to be visible
        for f in self.canvas_observers:
            f(*args)

    def split_stroke(self):
        if not self.stroke: return
        self.stroke.stop_recording()
        if not self.stroke.empty:
            self.command_stack.add(command.Stroke(self.layer, self.stroke))
        self.stroke = None

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
        self.command_stack.add(cmd)


    def set_brush(self, brush):
        self.split_stroke()
        self.brush.copy_settings_from(brush)


