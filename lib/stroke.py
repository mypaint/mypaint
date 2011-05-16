# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import brush
import numpy

class Stroke:
    """
    This class stores all information required to replay a stroke with
    the brush engine, event by event. This includes the RNG seed etc.
    """
    # A "finished" stroke object is immutable, except right after
    # creation (when it has never been fully rendered).  To modify an
    # existing stroke, the old one must be deleted and a new Stroke
    # instance must be used to replace it.
    serial_number = 0
    def __init__(self):
        self.finished = False
        Stroke.serial_number += 1
        self.serial_number = Stroke.serial_number

    def start_recording(self, brush):
        assert not self.finished

        self.brush_settings = brush.brushinfo.save_to_string() # fast (brush caches this string)

        states = brush.get_state()
        assert states.dtype == 'float32'
        self.brush_state = states.tostring()

        self.brush = brush
        self.brush.new_stroke() # this just resets the stroke_* members of the brush

        self.tmp_event_list = []

    def record_event(self, dtime, x, y, pressure, xtilt,ytilt):
        assert not self.finished
        self.tmp_event_list.append((dtime, x, y, pressure, xtilt,ytilt))

    def stop_recording(self):
        assert not self.finished
        # OPTIMIZE 
        # - for space: just gzip? use integer datatypes?
        # - for time: maybe already use array storage while recording?
        data = numpy.array(self.tmp_event_list, dtype='float64')
        data = data.tostring()
        version = '2'
        self.stroke_data = version + data

        self.total_painting_time = self.brush.stroke_total_painting_time
        #if not self.empty:
        #    print 'Recorded', len(self.stroke_data), 'bytes. (painting time: %.2fs)' % self.total_painting_time
        #print 'Compressed size:', len(zlib.compress(self.stroke_data)), 'bytes.'
        del self.brush, self.tmp_event_list
        self.finished = True

    def is_empty(self):
        return self.total_painting_time == 0
    empty = property(is_empty)
        
    def render(self, surface):
        assert self.finished

        # OPTIMIZE: check if parsing of settings is a performance bottleneck
        b = brush.Brush(brush.BrushInfo(self.brush_settings))

        states = numpy.fromstring(self.brush_state, dtype='float32')
        b.set_state(states)

        #b.set_print_inputs(1)
        #print 'replaying', len(self.stroke_data), 'bytes'

        version, data = self.stroke_data[0], self.stroke_data[1:]
        assert version == '2'
        data = numpy.fromstring(data, dtype='float64')
        data.shape = (len(data)/6, 6)

        surface.begin_atomic()
        for dtime, x, y, pressure, xtilt,ytilt in data:
            b.stroke_to (surface, x, y, pressure, xtilt,ytilt, dtime)
        surface.end_atomic()

    def copy_using_different_brush(self, brush):
        assert self.finished
        s = Stroke()
        s.__dict__.update(self.__dict__)
        s.brush_settings = brush.save_to_string()
        # note: we keep self.brush_state intact, even if the new brush
        # has different meanings for the states. This should cause
        # fewer glitches than resetting the initial state to zero.
        return s
