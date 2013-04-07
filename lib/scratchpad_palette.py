# This file is part of MyPaint.
# Copyright (C) 2011 by Ben O'Steen <bosteen@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import re
import os

from helpers import rgb_to_hsv
from helpers import hsv_to_rgb


def squiggle(off_x=0.0, off_y=0.0, scale=20.0):
    events = []
    events.append((0.0, off_x, off_y, 0.0))
    events.append((0.008, off_x, off_y, 0.0))
    t = 0.016
    pressure = 1.0
    for dx in xrange(3):
        x = dx % 2
        pressure -= 0.2
        for y in xrange(2):
            events.append((t, scale*(float(x))+off_x, scale*(float(y))+off_y, pressure))
            t += 0.008
    events.append((t, scale*(float(x))+off_x, scale*(float(y))+off_y, 0.0))
    return events

def slash_squiggle(off_x = 0.0, off_y=0.0, scale = 20.0):
    events = []
    events.append((0.0, off_x, off_y, 0.0))
    events.append((0.0, off_x, off_y, 1.0))
    events.append((0.0, off_x+scale, off_y+scale, 1.0))
    events.append((0.0, off_x, off_y, 0.0))
    return events

def box_squiggle(off_x = 0.0, off_y=0.0, scale = 20.0):
    events = []
    events.append((0.0, off_x, off_y, 0.0))
    events.append((0.008, off_x+scale, off_y, 1.0))
    events.append((0.016, off_x+scale, off_y+scale, 1.0))
    events.append((0.024, off_x, off_y+scale, 1.0))
    events.append((0.032, off_x, off_y, 1.0))
    events.append((0.040, off_x, off_y, 0.0))
    return events

def hatch_squiggle(off_x = 0.0, off_y=0.0, scale = 20.0):
    events = []
    t=0.8
    events.append((0.0, off_x, off_y, 0.0))
    slice_width = scale / 3.0
    for u in xrange(3):
        # Horizontal stripes
        events.append((t, off_x, (u * slice_width) + off_y, 1.0))
        t += 0.08
        events.append((t, scale+off_x, (u * slice_width) + off_y, 1.0))
        t += 0.08
        events.append((t, scale+off_x, (u * slice_width) + off_y, 0.0))
        t += 0.08
        # vertical stripes
        events.append((t, (u * slice_width) + off_x, off_y, 1.0))
        t += 0.08
        events.append((t, (u * slice_width) + off_x, scale + off_y, 1.0))
        t += 0.08
        events.append((t, (u * slice_width) + off_x, scale + off_y, 0.0))
        t += 0.08
    events.append((t, off_x, off_y, 0.0))
    return events

def draw_palette(app, palette, doc, columns=8, grid_size = 30.0, scale=13.0,
                 offset_x = 0.0, offset_y = 0.0,
                 swatch_method=squiggle):
    # store the current brush colour:
    brush_colour = app.brush.get_color_rgb()
    off_y = offset_y
    for colour_idx in xrange(len(palette)):
        off_x = (colour_idx % columns) * grid_size + offset_x
        if not (colour_idx % columns) and colour_idx:
            off_y += grid_size
        gen_events = swatch_method(off_x, off_y, scale=scale)
        # Set the color
        app.brush.set_color_rgb(palette.rgb(colour_idx))
        # simulate strokes on scratchpad
        for t, x, y, pressure in gen_events:
            x, y = doc.tdw.display_to_model(x, y)
            doc.model.stroke_to(0.008, x, y, pressure, 0.0, 0.0)
        doc.model.split_stroke()
    app.brush.set_color_rgb(brush_colour)

class GimpPalette(list):
    # loads a given gimp palette and makes it queriable
    # Would 'save' functionality be useful at some stage?

    def __init__(self, filename=None):
        self.columns = 0
        self.scheme = "RGB"
        if filename:
            self.load(filename)

    def load(self, filename):
        if os.path.isfile(filename):
            color_number = len(self)
            fp = open(filename, "r")
            header = fp.readline()
            if header[:12] != "GIMP Palette":
                raise SyntaxError, "not a valid GIMP palette"

            limit = 500    # not sure what the max colours are in a Gimp Palette

            while (limit != 0):
                color_line = fp.readline()

                if not color_line:
                    # Empty line = EOF?
                    break
                # Skip comments
                if re.match("#", color_line):
                    continue

                # Name: value pairs
                if re.match("\w+:", color_line):
                    tokens = color_line.split(":")
                    if len(tokens) == 2:
                        if tokens[0].lower().startswith("columns"):
                            try:
                                val = int(tokens[1].strip())
                                self.columns = val
                            except ValueError, e:
                                print "Bad Column value: %s" % tokens[1]
                    continue
                try:
                    triple = tuple(map(int, re.split("\s+", color_line.strip())[:3]))
                    if len(triple) != 3:
                        # could be index
                        print "Is index?"
                        raise ValueError
                    self.append(triple)
                except ValueError,e:
                    # Bad Data will not parse as Int
                    print "Bad line in palette: '%s'" % color_line[:-1]

                limit -= 1
            fp.close()
            print "Palette size:%s - Loaded %s new colors from palette %s" % (len(self), len(self) - color_number, filename)

    def hsv(self, index):
        if index < 0 or index > (len(self)-1):
            return None  # should be Exception perhaps?
        else:
            return rgb_to_hsv(*map(lambda x: x / 255.0, self[index]))

    def rgb(self, index):
        if index < 0 or index > (len(self)-1):
            return None  # should be Exception perhaps?
        else:
            return map(lambda x: x / 255.0, self[index])

    def append_hsv(self, *hsvvals):
        h,s,v = hsvvals
        self.append(map(lambda x: int(x * 255), hsv_to_rgb(h,s,v)))

    def append_rgb(self, *rgbvals):
        self.append(map(lambda x: int(x * 255), rgbvals))

    def append_hue_spectrum(self, rgbbase):
        h,s,v = rgb_to_hsv(*rgbbase)
        for hue_idx in xrange(20):
            hue = (hue_idx*0.05)
            self.append_hsv(hue, s,v)

    def append_sat_spectrum(self, hsv, number=8):
        h,s,v = hsv
        step = 1.0 / float(number)
        for sat_idx in xrange(number):
            sat = (sat_idx*step)
            self.append_hsv(h, sat, v)

