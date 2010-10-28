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

string_value_settings = set(("parent_brush_name", "group"))
current_brushfile_version = 2


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


class BrushInfo(dict):
    """Lightweight but fully parsed description of a brush.

    Just the strings, numbers and inputs/points hashes in a dict-based wrapper
    without any special interpretation other than a free upgrade to the newest
    brush format."""

    def __init__(self, string=None):
        """Construct a BrushInfo object, optionally parsing it."""
        dict.__init__(self)
        self._cache_str = None
        if string is not None:
            self.parse(string)

    def clone(self):
        """Returns a deep-copied duplicate."""
        return copy.deepcopy(self)

    class ParseError(Exception):
        pass

    class Obsolete(ParseError):
        pass

    def parse(self, settings_str):
        """Load a setting string, overwriting all current settings."""

        def parse_value(rawvalue, cname, version):
            """Parses a setting value, for a given setting name and brushfile version."""
            if cname in string_value_settings:
                string = brushinfo_unquote(rawvalue)
                return [(cname, string)]
            elif version <= 1 and cname == 'color':
                rgb = [int(c)/255.0 for c in rawvalue.split(" ")]
                h, s, v = helpers.rgb_to_hsv(*rgb)
                return [('color_h', (h, {})), ('color_s', (s, {})), ('color_v', (v, {}))]
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
                input_points.update({ inputname: points })
            return [(cname, (float(basevalue), input_points))]

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
            return (basevalue, input_points_new)

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
        self.clear()
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
                    self[cname] = value
            except Exception, e:
                line = "%s %s" % (rawcname, rawvalue)
                errors.append((line, str(e)))
        if num_parsed == 0:
            errors.append(('', 'there was only garbage in this file, using defaults'))
        if errors:
            for error in errors:
                print error
        self._cache_str = settings_str   # Maybe. It could still be old format...

    def serialize(self):
        """Serialise brush information to a string. Result is cached."""
        if self._cache_str is not None:
            return self._cache_str
        res = '# mypaint brush file\n'
        res += '# you can edit this file and then select the brush in mypaint (again) to reload\n'
        res += 'version %d\n' % current_brushfile_version
        for cname, data in self.iteritems():
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
        self._cache_str = res
        return res

    def clear(self):
        """Erases all keys of the underlying dict, invalidating cached strings."""
        self._cache_str = None
        dict.clear(self)

    def __setitem__(self, key, value):
        """Set an item via the dict interface, also invalidating cached strings.

        BrushInfo objects should map setting names to tuples of the form
        (base_value, input_points) where input_points is a hash mapping input
        names to lists of (x, y) pairs and at least length 2.
        """
        self._cache_str = None
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        """Delete an item via the dict interface, also invalidating cached strings."""
        self._cache_str = None
        dict.__delitem__(self, key, value)

    def engine_settings_iter(self):
        """Iterate across the settings that are meaningful to the brush engine.

        Yields tuples of the form (cname, base_value, input_points)."""
        for cname, data in self.iteritems():
            if cname in string_value_settings:
                continue
            basevalue, input_points = data
            yield (cname, basevalue, input_points)

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
        self.brush.brushinfo_pending_changes[self.setting.cname] = self.save_to_tuple()
        for f in self.observers:
            f()

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
        for f in self.observers:
            f()
        self.brush.brushinfo_pending_changes[self.setting.cname] = self.save_to_tuple()

    def save_to_tuple(self):
        # Equivalent of the old save_to_string().
        input_hash = {}
        for i in brushsettings.inputs:
            points = self.points[i.index]
            if points:
                input_vector = []
                for x, y in points:
                    input_vector.append((x, y))
                input_hash[i.name] = input_vector
        return (self.base_value, input_hash)

class Brush(mypaintlib.Brush):
    def __init__(self):
        mypaintlib.Brush.__init__(self)
        self.brushinfo = BrushInfo()
        self.brushinfo_pending_changes = {}
        self.settings_observers = []
        self.settings_observers_hidden = []
        self.settings = []
        for s in brushsettings.settings:
            self.settings.append(Setting(s, self, self.settings_observers))
        self.settings_observers.append(self._update_brushinfo)

    def _update_brushinfo(self):
        """Mirror changed settings into the BrushInfo tracking this Brush."""
        for cname, tup in self.brushinfo_pending_changes.iteritems():
            self.brushinfo[cname] = tup
        self.brushinfo_pending_changes = {}

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
        return self.brushinfo.serialize()

    def load_from_string(self, s):
        self.load_from_brushinfo(BrushInfo(s))

    def load_from_brushinfo(self, brushinfo):
        """Updates the brush's Settings from (a clone of) ``brushinfo``."""
        self.begin_atomic()
        try:
            self.brushinfo = brushinfo.clone()
            for setting in self.settings:
                setting.restore_defaults()
            engine_settings = self.brushinfo.engine_settings_iter()
            for cname, basevalue, input_points in engine_settings:
                setting = self.setting_by_cname(cname)
                setting.set_base_value(basevalue)
                for inputname, points in input_points.iteritems():
                    brushinput = brushsettings.inputs_dict.get(inputname)
                    setting.set_points(brushinput, points)
        finally:
            self.end_atomic()

    def copy_settings_from(self, other):
        self.load_from_brushinfo(other.brushinfo)

    def get_color_hsv(self):
        h = self.setting_by_cname('color_h').base_value
        s = self.setting_by_cname('color_s').base_value
        v = self.setting_by_cname('color_v').base_value
        return (h, s, v)

    def set_color_hsv(self, hsv):
        self.begin_atomic()
        try:
            h, s, v = hsv
            self.setting_by_cname('color_h').set_base_value(h)
            self.setting_by_cname('color_s').set_base_value(s)
            self.setting_by_cname('color_v').set_base_value(v)
        finally:
            self.end_atomic()

    def set_color_rgb(self, rgb):
        self.set_color_hsv(helpers.rgb_to_hsv(*rgb))

    def get_color_rgb(self):
        hsv = self.get_color_hsv()
        return helpers.hsv_to_rgb(*hsv)

    def is_eraser(self):
        return self.setting_by_cname('eraser').base_value > 0.9


if __name__ == "__main__":
    import doctest
    doctest.testmod()
