# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

import helpers, brush
import numpy
import random

class Stroke:
    # A "finished" stroke object is immutable, except right after creation (when it has never been fully rendered).
    serial_number = 0
    def __init__(self):
        self.finished = False
        Stroke.serial_number += 1
        self.serial_number = Stroke.serial_number

    def start_recording(self, brush):
        assert not self.finished

        self.brush_settings = brush.save_to_string() # fast (brush caches this string)

        self.brush_state = brush.get_state()

        self.seed = random.randrange(0x10000)
        self.brush = brush
        brush.srandom(self.seed)
        self.brush.new_stroke() # this only resets the stroke_* members

        self.tmp_event_list = []

    def record_event(self, dtime, x, y, pressure):
        assert not self.finished
        self.tmp_event_list.append((dtime, x, y, pressure))

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
        self.empty = self.total_painting_time == 0
        #if not self.empty:
        #    print 'Recorded', len(self.stroke_data), 'bytes. (painting time: %.2fs)' % self.total_painting_time
        #print 'Compressed size:', len(zlib.compress(self.stroke_data)), 'bytes.'
        del self.brush, self.tmp_event_list
        self.finished = True
        
    def render(self, surface):
        assert self.finished

        b = brush.Brush_Lowlevel()
        b.load_from_string(self.brush_settings) # OPTIMIZE: check if this is a performance bottleneck
        b.set_state(self.brush_state)
        b.srandom(self.seed)
        #b.set_print_inputs(1)
        #print 'replaying', len(self.stroke_data), 'bytes'

        version, data = self.stroke_data[0], self.stroke_data[1:]
        assert version == '2'
        data = numpy.fromstring(data, dtype='float64')
        data.shape = (len(data)/4, 4)
        for dtime, x, y, pressure in data: # FIXME: only iterable if more than one?
            b.tiled_surface_stroke_to (surface, x, y, pressure, dtime)

    def copy_using_different_brush(self, brush):
        assert self.finished
        s = Stroke()
        s.__dict__.update(self.__dict__)
        s.brush_settings = brush.save_to_string()
        # note: we keep self.brush_state intact, even if the new brush
        # has different meanings for the states. This should cause
        # fewer glitches than resetting the initial state to zero.
        return s
