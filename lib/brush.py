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
import urllib
import copy
import math
import json

string_value_settings = set(("parent_brush_name", "group"))
current_brushfile_version = 2

brush_settings = set([s.cname for s in brushsettings.settings])
all_settings = brush_settings.union(string_value_settings)

def brushinfo_quote(string):
    """Quote a string for serialisation of brushes.

    >>> brushinfo_quote(u'foo')
    'foo'
    >>> brushinfo_quote(u'foo/bar blah')
    'foo%2Fbar%20blah'
    >>> brushinfo_quote(u'Have a nice day \u263A')
    'Have%20a%20nice%20day%20%E2%98%BA'
    """
    string = unicode(string)
    u8bytes = string.encode("utf-8")
    return str(urllib.quote(u8bytes, safe=''))


def brushinfo_unquote(quoted):
    """Unquote a serialised string value from a brush field.

    >>> brushinfo_unquote("foo")
    u'foo'
    >>> brushinfo_unquote("foo%2fbar%20blah")
    u'foo/bar blah'
    >>> expected = u'Have a nice day \u263A'
    >>> brushinfo_unquote('Have%20a%20nice%20day%20%E2%98%BA') == expected
    True
    """
    quoted = str(quoted)
    u8bytes = urllib.unquote(quoted)
    return unicode(u8bytes.decode("utf-8"))


class BrushInfo:
    """Fully parsed description of a brush.
    """

    def __init__(self, string=None):
        """Construct a BrushInfo object, optionally parsing it."""
        self.settings = {}
        self.cache_str = None
        self.observers = [self.settings_changed_cb]
        self.observers_hidden = []
        self.pending_updates = set()
        if string:
            self.load_from_string(string)

    def settings_changed_cb(self, settings):
        self.cache_str = None

    def clone(self):
        """Returns a deep-copied duplicate."""
        res = BrushInfo()
        res.load_from_brushinfo(self)
        return res

    def load_from_brushinfo(self, other):
        """Updates the brush's Settings from (a clone of) ``brushinfo``."""
        self.settings = copy.deepcopy(other.settings)
        for f in self.observers:
            f(all_settings)
        self.cache_str = other.cache_str

    def load_defaults(self):
        """Load default brush settings, dropping all current settings."""
        self.begin_atomic()
        self.settings = {}
        for s in brushsettings.settings:
            self.reset_setting(s.cname)
        self.end_atomic()

    def reset_setting(self, cname):
        basevalue = brushsettings.settings_dict[cname].default
        if cname == 'opaque_multiply':
            # make opaque depend on pressure by default
            input_points = {'pressure': [(0.0, 0.0), (1.0, 1.0)]}
        else:
            input_points = {}
        self.settings[cname] = [basevalue, input_points]
        for f in self.observers:
            f(set([cname]))

    class ParseError(Exception):
        pass

    class Obsolete(ParseError):
        pass

    def to_json(self):
        settings = dict(self.settings)

        # Parent brush is not really a brush (engine) setting
        parent_brush_name = settings.pop('parent_brush_name', '')
        # Neither is group
        brush_group = settings.pop('group', '')

        # Make the contents of each setting a bit more explicit
        for k, v in settings.items():
            base_value, inputs = v
            settings[k] = {'base_value': base_value, 'inputs': inputs}

        document = {'version': 3,
                    'comment': """MyPaint brush file""",
                    'parent_brush_name': parent_brush_name,
                    'settings': settings,
                    'group': brush_group,
                   }
        return json.dumps(document, sort_keys=True, indent=4)

    def from_json(self, json_string):
        brush_def = json.loads(json_string)
        if brush_def['version'] != 3:
            raise BrushInfo.ParseError, 'brush is not compatible with this version of mypaint (json version=%r)' % brush_def['version']

        # settings not in json_string must still be present in self.settings
        self.load_defaults()

        # MyPaint expects that each setting has an array, where
        # index 0 is base value, and index 1 is inputs
        for k, v in brush_def['settings'].items():
            base_value, inputs = v['base_value'], v['inputs']
            if k not in self.settings:
                print 'ignoring unknown brush setting %r' % k
                continue
            self.settings[k] = [base_value, inputs]

        # MyPaint expects parent brush as a setting
        self.settings['parent_brush_name'] = brush_def['parent_brush_name']
        # FIXME: who uses this? brush groups are stored externally in order.conf, is this redundant?
        self.settings['group'] = brush_def['group']

    def load_from_string(self, settings_str):
        """Load a setting string, overwriting all current settings."""

        if settings_str.startswith('{'):
            # new json-based brush format
            self.from_json(settings_str)
        elif settings_str.startswith('#'):
            # old brush format
            self._load_old_format(settings_str)
        else:
            raise BrushInfo.ParseError, 'brush format not recognized'

        for f in self.observers:
            f(all_settings)
        self.cache_str = settings_str   # Maybe. It could still be old format...

    def _load_old_format(self, settings_str):

        def parse_value(rawvalue, cname, version):
            """Parses a setting value, for a given setting name and brushfile version."""
            if cname in string_value_settings:
                string = brushinfo_unquote(rawvalue)
                return [(cname, string)]
            elif version <= 1 and cname == 'color':
                rgb = [int(c)/255.0 for c in rawvalue.split(" ")]
                h, s, v = helpers.rgb_to_hsv(*rgb)
                return [('color_h', [h, {}]), ('color_s', [s, {}]), ('color_v', [v, {}])]
            elif version <= 1 and cname == 'change_radius':
                if rawvalue == '0.0':
                    return []
                raise Obsolete, 'change_radius is not supported any more'
            elif version <= 2 and cname == 'adapt_color_from_image':
                if rawvalue == '0.0':
                    return []
                raise Obsolete, 'adapt_color_from_image is obsolete, ignored;' + \
                                ' use smudge and smudge_length instead'
            elif version <= 1 and cname == 'painting_time':
                return []
 
            if version <= 1 and cname == 'speed':
                cname = 'speed1'
            parts = rawvalue.split('|')
            basevalue = float(parts[0])
            input_points = {}
            for part in parts[1:]:
                inputname, rawpoints = part.strip().split(' ', 1)
                if version <= 1:
                    points = parse_points_v1(rawpoints)
                else:
                    points = parse_points_v2(rawpoints)
                assert len(points) >= 2
                input_points[inputname] = points
            return [(cname, [float(basevalue), input_points])]

        def parse_points_v1(rawpoints):
            """Parses the points list format from versions prior to version 2."""
            points_seq = [float(f) for f in rawpoints.split()]
            points = [(0, 0)]
            while points_seq:
                x = points_seq.pop(0)
                y = points_seq.pop(0)
                if x == 0: break
                assert x > points[-1][0]
                points.append((x, y))
            return points

        def parse_points_v2(rawpoints):
            """Parses the newer points list format of v2 and beyond."""
            points = []
            for s in rawpoints.split(', '):
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
            return points

        def transform_y(valuepair, func):
            """Used during migration from earlier versions."""
            basevalue, input_points = valuepair
            basevalue = func(basevalue)
            input_points_new = {}
            for inputname, points in input_points.iteritems():
                points_new = [(x, func(y)) for x, y in points]
                input_points_new[inputname] = points_new
            return [basevalue, input_points_new]

        # Split out the raw settings and grab the version we're dealing with
        rawsettings = []
        errors = []
        version = 1 # for files without a 'version' field
        for line in settings_str.split('\n'):
            try:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                cname, rawvalue = line.split(' ', 1)
                if cname == 'version':
                    version = int(rawvalue)
                    if version > current_brushfile_version:
                        raise BrushInfo.ParseError, 'this brush was saved with a more recent version of mypaint'
                else:
                    rawsettings.append((cname, rawvalue))
            except Exception, e:
                errors.append((line, str(e)))

        # Parse each pair
        self.load_defaults()
        # compatibility hack: keep disabled for old brushes, but still use non-zero default
        self.settings['anti_aliasing'][0] = 0.0
        num_parsed = 0
        for rawcname, rawvalue in rawsettings:
            try:
                cnamevaluepairs = parse_value(rawvalue, rawcname, version)
                num_parsed += 1
                for cname, value in cnamevaluepairs:
                    if cname in brushsettings.settings_migrate:
                        cname, func = brushsettings.settings_migrate[cname]
                        if func:
                            value = transform_y(value, func)
                    self.settings[cname] = value
            except Exception, e:
                line = "%s %s" % (rawcname, rawvalue)
                errors.append((line, str(e)))
        if errors:
            for error in errors:
                print error
        if num_parsed == 0:
            raise BrushInfo.ParseError, 'old brush file format parser did not find any brush settings in this file'


    def save_to_string(self):
        """Serialise brush information to a string. Result is cached."""
        if self.cache_str:
            return self.cache_str

        res = self.to_json()

        self.cache_str = res
        return res

    def _save_old_format(self):
        res = '# mypaint brush file\n'
        res += '# you can edit this file and then select the brush in mypaint (again) to reload\n'
        res += 'version %d\n' % current_brushfile_version

        for cname, data in self.settings.iteritems():
            if cname in string_value_settings:
                if data is not None:
                    res += cname + " " + brushinfo_quote(data)
            else:
                res += cname + " "
                basevalue, input_points = data
                res += str(basevalue)
                if input_points:
                    for inputname, points in input_points.iteritems():
                        res += " | " + inputname + ' '
                        res += ', '.join(['(%f %f)' % xy for xy in points])
            res += "\n"

        return res

    def get_base_value(self, cname):
        return self.settings[cname][0]

    def get_points(self, cname, input, readonly=False):
        res = self.settings[cname][1].get(input, ())
        if not readonly: # slow
            res = copy.deepcopy(res)
        return res

    def set_base_value(self, cname, value):
        assert cname in brush_settings
        assert not math.isnan(value)
        assert not math.isinf(value)
        if self.settings[cname][0] != value:
            self.settings[cname][0] = value
            for f in self.observers:
                f(set([cname]))

    def set_points(self, cname, input, points):
        assert cname in brush_settings
        points = tuple(points)
        d = self.settings[cname][1]
        if points:
            d[input] = copy.deepcopy(points)
        elif input in d:
            d.pop(input)

        for f in self.observers:
            f(set([cname]))

    def set_setting(self, cname, value):
        self.settings[cname] = copy.deepcopy(value)
        for f in self.observers:
            f(set([cname]))

    def get_setting(self, cname):
        return copy.deepcopy(self.settings[cname])

    def get_string_property(self, name):
        return self.settings.get(name, None)

    def set_string_property(self, name, value):
        assert name in string_value_settings
        if value is None:
            self.settings.pop(name, None)
        else:
            assert isinstance(value, str) or isinstance(value, unicode)
            self.settings[name] = value
        for f in self.observers:
            f(set([name]))

    def has_only_base_value(self, cname):
        """Return whether a setting is constant for this brush."""
        for i in brushsettings.inputs:
            if self.has_input(cname, i.name):
                return False
        return True

    def has_large_base_value(self, cname, threshold=0.9):
        return self.get_base_value(cname) > threshold

    def has_small_base_value(self, cname, threshold=0.1):
        return self.get_base_value(cname) < threshold

    def has_input(self, cname, input):
        """Return whether a given input is used by some setting."""
        points = self.get_points(cname, input, readonly=True)
        return bool(points)

    def begin_atomic(self):
        self.observers_hidden.append(self.observers[:])
        del self.observers[:]
        self.observers.append(self.add_pending_update)

    def add_pending_update(self, settings):
        self.pending_updates.update(settings)

    def end_atomic(self):
        self.observers[:] = self.observers_hidden.pop()
        pending = self.pending_updates.copy()
        if pending:
            self.pending_updates.clear()
            for f in self.observers:
                f(pending)

    def get_color_hsv(self):
        h = self.get_base_value('color_h')
        s = self.get_base_value('color_s')
        v = self.get_base_value('color_v')
        assert not math.isnan(h)
        return (h, s, v)

    def set_color_hsv(self, hsv):
        self.begin_atomic()
        try:
            h, s, v = hsv
            self.set_base_value('color_h', h)
            self.set_base_value('color_s', s)
            self.set_base_value('color_v', v)
        finally:
            self.end_atomic()

    def set_color_rgb(self, rgb):
        self.set_color_hsv(helpers.rgb_to_hsv(*rgb))

    def get_color_rgb(self):
        hsv = self.get_color_hsv()
        return helpers.hsv_to_rgb(*hsv)

    def is_eraser(self):
        return self.has_large_base_value("eraser")

    def is_alpha_locked(self):
        return self.has_large_base_value("lock_alpha")

    def is_colorize(self):
        return self.has_large_base_value("colorize")

    def matches(self, other, ignore=["color_h", "color_s", "color_v", "parent_brush_name"]):
        s1 = self.settings.copy()
        s2 = other.settings.copy()
        for k in ignore:
            s1.pop(k, None)
            s2.pop(k, None)
        return s1 == s2

class Brush(mypaintlib.PythonBrush):
    """
    Low-level extension of the C brush class, propagating all changes of
    a brushinfo instance down into the C code.
    """
    def __init__(self, brushinfo):
        mypaintlib.PythonBrush.__init__(self)
        self.brushinfo = brushinfo
        brushinfo.observers.append(self.update_brushinfo)
        self.update_brushinfo(all_settings)

        # override some mypaintlib.Brush methods with special wrappers
        # from python_brush.hpp
        self.get_state = self.python_get_state
        self.set_state = self.python_set_state
        self.stroke_to = self.python_stroke_to

    def update_brushinfo(self, settings):
        """Mirror changed settings into the BrushInfo tracking this Brush."""

        for cname in settings:
            setting = brushsettings.settings_dict.get(cname)
            if not setting:
                continue

            base = self.brushinfo.get_base_value(cname)
            self.set_base_value(setting.index, base)

            for input in brushsettings.inputs:
                points = self.brushinfo.get_points(cname, input.name, readonly=True)

                assert len(points) != 1
                #if len(points) > 2:
                #    print 'set_points[%s](%s, %s)' % (cname, input.name, points)

                self.set_mapping_n(setting.index, input.index, len(points))
                for i, (x, y) in enumerate(points):
                    self.set_mapping_point(setting.index, input.index, i, x, y)

    def get_stroke_bbox(self):
        bbox = self.stroke_bbox
        return bbox.x, bbox.y, bbox.w, bbox.h


if __name__ == "__main__":
    import doctest
    doctest.testmod()
