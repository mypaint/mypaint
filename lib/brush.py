# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import mypaintlib
from brushlib import brushsettings
import helpers

current_brushfile_version = 2

# points = [(x1, y1), (x2, y2), ...] (at least two points, or None)
class Setting:
    "a specific setting for a specific brush"
    def __init__(self, setting, parent_brush, observers):
        self.setting = setting
        self.brush = parent_brush
        self.observers = observers
        self.base_value = None
        self.points = [[] for i in xrange(len(brushsettings.inputs))]
        self.restore_defaults()
    def restore_defaults(self):
        self.set_base_value(self.setting.default)
        for i in brushsettings.inputs:
            if self.setting.cname == 'opaque_multiply' and i.name == 'pressure':
                # make opaque depend on pressure by default
                self.set_points(i, [(0.0, 0.0), (1.0, 1.0)])
            else:
                self.set_points(i, [])
    def set_base_value(self, value):
        if self.base_value == value: return
        self.base_value = value
        self.brush.set_base_value(self.setting.index, value)
        for f in self.observers: f()
    def has_only_base_value(self):
        for i in brushsettings.inputs:
            if self.has_input(i):
                return False
        return True
    def has_input(self, input):
        return self.points[input.index]
    def has_input_nonlinear(self, input):
        points = self.points[input.index]
        if not points: return False
        if len(points) > 2: return True
        # also if it is linear but the x-axis was changed (hm, bad function name)
        if abs(points[0][0] - input.soft_min) > 0.001: return True
        if abs(points[1][0] - input.soft_max) > 0.001: return True
        return False

    def set_points(self, input, points):
        assert len(points) != 1
        if self.points[input.index] == points: return
        #if len(points) > 2:
        #    print 'set_points[%s](%s, %s)' % (self.setting.cname, input.name, points)

        self.brush.set_mapping_n(self.setting.index, input.index, len(points))
        for i, (x, y) in enumerate(points):
            self.brush.set_mapping_point(self.setting.index, input.index, i, x, y)

        self.points[input.index] = points[:] # copy
        for f in self.observers: f()

    def save_to_string(self):
        s = str(self.base_value)
        for i in brushsettings.inputs:
            points = self.points[i.index]
            if points:
                s += ' | ' + i.name + ' ' + ', '.join(['(%f %f)' % xy for xy in points])
        return s
    def load_from_string(self, s, version):
        error = None
        parts = s.split('|')
        self.set_base_value(float(parts[0]))
        for i in brushsettings.inputs:
            self.set_points(i, [])
        for part in parts[1:]:
            command, args = part.strip().split(' ', 1)
            if version <= 1 and command == 'speed': command = 'speed1'
            i = brushsettings.inputs_dict.get(command)
            if i:
                if version <= 1:
                    points_old = [float(f) for f in args.split()]
                    points = [(0, 0)]
                    while points_old:
                        x = points_old.pop(0)
                        y = points_old.pop(0)
                        if x == 0: break
                        assert x > points[-1][0]
                        points.append((x, y))
                else:
                    points = []
                    for s in args.split(', '):
                        s = s.strip()
                        if not (s.startswith('(') and s.endswith(')') and ' ' in s):
                            return '(x y) expected, got "%s"' % s
                        s = s[1:-1]
                        try:
                            x, y = [float(ss) for ss in s.split(' ')]
                        except:
                            print s
                            raise
                        points.append((x, y))
                assert len(points) >= 2
                self.set_points(i, points)
            else:
                error = 'unknown input "%s"' % command
        return error
    def transform_y(self, func):
        # useful for migration from a earlier version
        self.set_base_value(func(self.base_value))
        for i in brushsettings.inputs:
            if not self.points[i.index]: continue
            points = self.points[i.index]
            points = [(x, func(y)) for x, y in points]
            self.set_points(i, points)

class Brush(mypaintlib.Brush):
    def __init__(self):
        mypaintlib.Brush.__init__(self)
        self.settings_observers = []
        self.settings_observers_hidden = []
        self.settings = []
        for s in brushsettings.settings:
            self.settings.append(Setting(s, self, self.settings_observers))

        self.saved_string = None
        self.settings_observers.append(self.invalidate_saved_string)

    def invalidate_saved_string(self):
        self.saved_string = None

    def begin_atomic(self):
        self.settings_observers_hidden.append(self.settings_observers[:])
        del self.settings_observers[:]

    def end_atomic(self):
        self.settings_observers[:] = self.settings_observers_hidden.pop()
        for f in self.settings_observers: f()


    def get_stroke_bbox(self):
        bbox = self.stroke_bbox
        return bbox.x, bbox.y, bbox.w, bbox.h

    def setting_by_cname(self, cname):
        s = brushsettings.settings_dict[cname]
        return self.settings[s.index]

    def save_to_string(self):
        if self.saved_string: return self.saved_string
        res  = '# mypaint brush file\n'
        res += '# you can edit this file and then select the brush in mypaint (again) to reload\n'
        res += 'version %d\n' % current_brushfile_version
        for s in brushsettings.settings:
            res += s.cname + ' ' + self.settings[s.index].save_to_string() + '\n'
        self.saved_string = res
        return res

    def load_from_string(self, s):
        self.begin_atomic()
        for setting in self.settings:
            setting.restore_defaults()
        num_found = 0
        errors = []
        version = 1 # for files without a 'version' field
        for line in s.split('\n'):
            line = line.strip()
            if line.startswith('#'): continue
            if not line: continue
            try:
                command, rest = line.split(' ', 1)
                error = None

                if command in brushsettings.settings_dict:
                    setting = self.setting_by_cname(command)
                    error = setting.load_from_string(rest, version)
                elif command in brushsettings.settings_migrate:
                    command_new, transform_func = brushsettings.settings_migrate[command]
                    setting = self.setting_by_cname(command_new)
                    error = setting.load_from_string(rest, version)
                    if transform_func:
                        setting.transform_y(transform_func)
                elif command == 'version':
                    version = int(rest)
                    if version > current_brushfile_version:
                        error = 'this brush was saved with a more recent version of mypaint'
                elif version <= 1 and command == 'color':
                    self.set_color_rgb([int(s)/255.0 for s in rest.split()])
                elif version <= 1 and command == 'change_radius':
                    if rest != '0.0': error = 'change_radius is not supported any more'
                elif version <= 2 and command == 'adapt_color_from_image':
                    if rest != '0.0': error = 'adapt_color_from_image is obsolete, ignored; use smudge and smudge_length instead'
                elif version <= 1 and command == 'painting_time':
                    pass
                else:
                    error = 'unknown command, line ignored'

                if error:
                    errors.append((line, error))

            except Exception, e:
                errors.append((line, str(e)))
            else:
                num_found += 1
        if num_found == 0:
            errors.append(('', 'there was only garbage in this file, using defaults'))
        self.end_atomic()

        if not errors:
            # speedup for self.save_to_string()
            self.saved_string = s

        return errors

    def copy_settings_from(self, other):
        self.load_from_string(other.save_to_string())

    def get_color_hsv(self):
        h = self.setting_by_cname('color_h').base_value
        s = self.setting_by_cname('color_s').base_value
        v = self.setting_by_cname('color_v').base_value
        return (h, s, v)

    def set_color_hsv(self, hsv):
        self.begin_atomic()
        h, s, v = hsv
        self.setting_by_cname('color_h').set_base_value(h)
        self.setting_by_cname('color_s').set_base_value(s)
        self.setting_by_cname('color_v').set_base_value(v)
        self.end_atomic()

    def set_color_rgb(self, rgb):
        self.set_color_hsv(helpers.rgb_to_hsv(*rgb))

    def get_color_rgb(self):
        hsv = self.get_color_hsv()
        return helpers.hsv_to_rgb(*hsv)

    def is_eraser(self):
        return self.setting_by_cname('eraser').base_value > 0.9

