# This file is part of MyPaint.
# Copyright (C) 2007-2018 by the MyPaint Development Team
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function
import logging
import copy
import math
import json

from lib import mypaintlib
from lib import helpers
from lib import brushsettings
from lib.pycompat import unicode
from lib.pycompat import PY3

if PY3:
    from urllib.parse import quote_from_bytes as url_quote
    from urllib.parse import unquote_to_bytes as url_unquote
else:
    from urllib import quote as url_quote
    from urllib import unquote as url_unquote

logger = logging.getLogger(__name__)


# Module constants:

STRING_VALUE_SETTINGS = set((
    "parent_brush_name",
    "group",  # Possibly obsolete group field (replaced by order.conf?)
    "comment",  # MyPaint uses this to explanation what the file is
    "notes",  # Brush developer's notes field, multiline
    "description",  # Short, user-facing description field, single line
))
OLDFORMAT_BRUSHFILE_VERSION = 2

BRUSH_SETTINGS = set([s.cname for s in brushsettings.settings])
ALL_SETTINGS = BRUSH_SETTINGS.union(STRING_VALUE_SETTINGS)

_BRUSHINFO_MATCH_IGNORES = [
    "color_h", "color_s", "color_v",
    "parent_brush_name",
]


# Helper funcs for quoting and unquoting:

def brushinfo_quote(string):
    """Quote a string for serialisation of brushes.

    >>> brushinfo_quote(u'foo') == b'foo'
    True
    >>> brushinfo_quote(u'foo/bar blah') == b'foo%2Fbar%20blah'
    True
    >>> expected = b'Have%20a%20nice%20day%20%E2%98%BA'
    >>> brushinfo_quote(u'Have a nice day \u263A') == expected
    True

    """
    string = unicode(string)
    u8bytes = string.encode("utf-8")
    return url_quote(u8bytes, safe='').encode("ascii")


def brushinfo_unquote(quoted):
    """Unquote a serialised string value from a brush field.

    >>> brushinfo_unquote(b"foo") == u'foo'
    True
    >>> brushinfo_unquote(b"foo%2fbar%20blah") == u'foo/bar blah'
    True
    >>> expected = u'Have a nice day \u263A'
    >>> brushinfo_unquote(b'Have%20a%20nice%20day%20%E2%98%BA') == expected
    True

    """
    if not isinstance(quoted, bytes):
        raise ValueError("Cann")
    u8bytes = url_unquote(quoted)
    return unicode(u8bytes.decode("utf-8"))


# Exceptions raised during brush parsing:

class ParseError (Exception):
    pass


class Obsolete (ParseError):
    pass


# Helper functions for parsing the old brush format:

def _oldfmt_parse_value(rawvalue, cname, version):
    """Parses a raw setting value.

    This code handles a format that changed over time, so the
    parse is for a given setting name and brushfile version.

    """
    if cname in STRING_VALUE_SETTINGS:
        string = brushinfo_unquote(rawvalue)
        return [(cname, string)]
    elif version <= 1 and cname == 'color':
        rgb = [int(c) / 255.0 for c in rawvalue.split(" ")]
        h, s, v = helpers.rgb_to_hsv(*rgb)
        return [
            ('color_h', [h, {}]),
            ('color_s', [s, {}]),
            ('color_v', [v, {}]),
        ]
    elif version <= 1 and cname == 'change_radius':
        if rawvalue == '0.0':
            return []
        raise Obsolete('change_radius is not supported any more')
    elif version <= 2 and cname == 'adapt_color_from_image':
        if rawvalue == '0.0':
            return []
        raise Obsolete(
            'adapt_color_from_image is obsolete, ignored;'
            ' use smudge and smudge_length instead'
        )
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
            points = _oldfmt_parse_points_v1(rawpoints)
        else:
            points = _oldfmt_parse_points_v2(rawpoints)
        assert len(points) >= 2
        input_points[inputname] = points
    return [(cname, [float(basevalue), input_points])]


def _oldfmt_parse_points_v1(rawpoints):
    """Parses the points list format from v1"""
    points_seq = [float(f) for f in rawpoints.split()]
    points = [(0, 0)]
    while points_seq:
        x = points_seq.pop(0)
        y = points_seq.pop(0)
        if x == 0:
            break
        assert x > points[-1][0]
        points.append((x, y))
    return points


def _oldfmt_parse_points_v2(rawpoints):
    """Parses the newer points list format of v2 and beyond."""
    points = []
    for s in rawpoints.split(', '):
        s = s.strip()
        if not (s.startswith('(') and s.endswith(')') and ' ' in s):
            return '(x y) expected, got "%s"' % s
        s = s[1:-1]
        x, y = [float(ss) for ss in s.split(' ')]
        points.append((x, y))
    return points


def _oldfmt_transform_y(valuepair, func):
    """Used during migration from earlier versions."""
    basevalue, input_points = valuepair
    basevalue = func(basevalue)
    input_points_new = {}
    for inputname, points in input_points.items():
        points_new = [(x, func(y)) for x, y in points]
        input_points_new[inputname] = points_new
    return [basevalue, input_points_new]


# Class defs:

class BrushInfo (object):
    """Fully parsed description of a brush.
    """

    def __init__(self, string=None):
        """Construct a BrushInfo object, optionally parsing it."""
        super(BrushInfo, self).__init__()
        self.settings = {}
        self.cache_str = None
        self.observers = []
        for s in brushsettings.settings:
            self.reset_setting(s.cname)
        self.observers.append(self.settings_changed_cb)
        self.observers_hidden = []
        self.pending_updates = set()
        if string:
            self.load_from_string(string)
        from gui.application import get_app
        self.app = get_app()
        try:
            self.EOTF = self.app.preferences['display.colorspace_EOTF']
        except: 
            self.EOTF = 2.2

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
            f(ALL_SETTINGS)
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

    def to_json(self):
        settings = dict(self.settings)

        # Fields we save that aren't really brush engine settings
        parent_brush_name = settings.pop('parent_brush_name', '')
        brush_group = settings.pop('group', '')
        description = settings.pop('description', '')
        notes = settings.pop('notes', '')

        # The comment we save is always the same
        settings.pop('comment', '')

        # Make the contents of each setting a bit more explicit
        for k, v in list(settings.items()):
            base_value, inputs = v
            settings[k] = {'base_value': base_value, 'inputs': inputs}

        document = {
            'version': 3,
            'comment': """MyPaint brush file""",
            'parent_brush_name': parent_brush_name,
            'settings': settings,
            'group': brush_group,
            'notes': notes,
            'description': description,
        }
        return json.dumps(document, sort_keys=True, indent=4)

    def from_json(self, json_string):
        """Loads settings from a JSON string.

        >>> from glob import glob
        >>> for p in glob("tests/brushes/v3/*.myb"):
        ...     with open(p, "rb") as fp:
        ...         bstr = fp.read()
        ...         ustr = bstr.decode("utf-8")
        ...     b1 = BrushInfo()
        ...     b1.from_json(bstr)
        ...     b1 = BrushInfo()
        ...     b1.from_json(ustr)

        See also load_from_string(), which can handle the old v2 format.

        Accepts both unicode and byte strings. Byte strings are assumed
        to be encoded as UTF-8 when any decoding's needed.

        """

        # Py3: Ubuntu Trusty's 3.4.3 json.loads() requires unicode strs.
        # Layer Py3, and Py2 is OK with either.
        if not isinstance(json_string, unicode):
            if not isinstance(json_string, bytes):
                raise ValueError("Need either a str or a bytes object")
            json_string = json_string.decode("utf-8")

        brush_def = json.loads(json_string)
        if brush_def.get('version', 0) < 3:
            raise BrushInfo.ParseError(
                'brush is not compatible with this version of mypaint '
                '(json file version=%r)' % (brush_def.get('version'),)
            )

        # settings not in json_string must still be present in self.settings
        self.load_defaults()

        # MyPaint expects that each setting has an array, where
        # index 0 is base value, and index 1 is inputs
        for k, v in brush_def['settings'].items():
            base_value, inputs = v['base_value'], v['inputs']
            if k not in self.settings:
                logger.warning('ignoring unknown brush setting %r', k)
                continue
            self.settings[k] = [base_value, inputs]

        # Non-libmypaint string fields
        for cname in STRING_VALUE_SETTINGS:
            self.settings[cname] = brush_def.get(cname, '')
        # FIXME: Who uses "group"?
        # FIXME: Brush groups are stored externally in order.conf,
        # FIXME: is that one redundant?

    def load_from_string(self, settings_str):
        """Load a setting string, overwriting all current settings."""

        settings_unicode = settings_str
        if not isinstance(settings_unicode, unicode):
            if not isinstance(settings_unicode, bytes):
                raise ValueError("Need either a str or a bytes object")
            settings_unicode = settings_unicode.decode("utf-8")

        if settings_unicode.startswith(u'{'):
            # new json-based brush format
            self.from_json(settings_str)
        elif settings_unicode.startswith(u'#'):
            # old brush format
            self._load_old_format(settings_str)
        else:
            raise BrushInfo.ParseError('brush format not recognized')

        for f in self.observers:
            f(ALL_SETTINGS)
        self.cache_str = settings_str

    def _load_old_format(self, settings_str):
        """Loads brush settings in the old (v2) format.

        >>> from glob import glob
        >>> for p in glob("tests/brushes/v2/*.myb"):
        ...     with open(p, "rb") as fp:
        ...         bstr = fp.read()
        ...         ustr = bstr.decode("utf-8")
        ...     b1 = BrushInfo()
        ...     b1._load_old_format(bstr)
        ...     b2 = BrushInfo()
        ...     b2._load_old_format(ustr)

        Accepts both unicode and byte strings. Byte strings are assumed
        to be encoded as UTF-8 when any decoding's needed.

        """

        # Py2 is happy natively comparing unicode with str, no encode
        # needed. For Py3, need to parse as str so that updated dict
        # keys can be compared sensibly with stuff written by other
        # code.

        if not isinstance(settings_str, unicode):
            if not isinstance(settings_str, bytes):
                raise ValueError("Need either a str or a bytes object")
            if PY3:
                settings_str = settings_str.decode("utf-8")

        # Split out the raw settings and grab the version we're dealing with
        rawsettings = []
        errors = []
        version = 1  # for files without a 'version' field
        for line in settings_str.split('\n'):
            try:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                cname, rawvalue = line.split(' ', 1)
                if cname == 'version':
                    version = int(rawvalue)
                    if version > OLDFORMAT_BRUSHFILE_VERSION:
                        raise BrushInfo.ParseError(
                            "This brush is not in the old format "
                            "supported (version > {})".format(
                                OLDFORMAT_BRUSHFILE_VERSION,
                            )
                        )
                else:
                    rawsettings.append((cname, rawvalue))
            except Exception as e:
                errors.append((line, str(e)))

        # Parse each pair
        self.load_defaults()
        # compatibility hack: keep disabled for old brushes,
        # but still use non-zero default
        self.settings['anti_aliasing'][0] = 0.0
        num_parsed = 0
        for rawcname, rawvalue in rawsettings:
            try:
                cnamevaluepairs = _oldfmt_parse_value(
                    rawvalue,
                    rawcname,
                    version,
                )
                num_parsed += 1
                for cname, value in cnamevaluepairs:
                    if cname in brushsettings.settings_migrate:
                        cname, func = brushsettings.settings_migrate[cname]
                        if func:
                            value = _oldfmt_transform_y(value, func)
                    self.settings[cname] = value
            except Exception as e:
                line = "%s %s" % (rawcname, rawvalue)
                errors.append((line, str(e)))
        if errors:
            for error in errors:
                logger.warning(error)
        if num_parsed == 0:
            raise BrushInfo.ParseError(
                "old brush file format parser did not find "
                "any brush settings in this file",
            )

    def save_to_string(self):
        """Serialise brush information to a string. Result is cached."""
        if self.cache_str:
            return self.cache_str

        res = self.to_json()

        self.cache_str = res
        return res

    def get_base_value(self, cname):
        return self.settings[cname][0]

    def get_points(self, cname, input, readonly=False):
        res = self.settings[cname][1].get(input, ())
        if not readonly:  # slow
            res = copy.deepcopy(res)
        return res

    def set_base_value(self, cname, value):
        assert cname in BRUSH_SETTINGS
        assert not math.isnan(value)
        assert not math.isinf(value)
        if self.settings[cname][0] != value:
            self.settings[cname][0] = value
            for f in self.observers:
                f(set([cname]))

    def set_points(self, cname, input, points):
        assert cname in BRUSH_SETTINGS
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
        value = self.settings.get(name, None)
        if value is None:
            return None
        return unicode(value)

    def set_string_property(self, name, value):
        assert name in STRING_VALUE_SETTINGS
        if value is None:
            self.settings.pop(name, None)
        else:
            assert isinstance(value, str) or isinstance(value, unicode)
            self.settings[name] = unicode(value)
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
        tf = self.EOTF
        h = self.get_base_value('color_h')
        s = self.get_base_value('color_s')
        v = self.get_base_value('color_v')
        rgb = helpers.hsv_to_rgb(h, s, v)
        rgb = rgb[0]**(1 / tf), rgb[1]**(1 / tf), rgb[2]**(1 / tf)
        hsv = helpers.rgb_to_hsv(*rgb)
        h, s, v = hsv
        assert not math.isnan(h)
        return (h, s, v)

    def set_color_hsv(self, hsv):
        tf = self.EOTF
        if not hsv:
            return
        self.begin_atomic()
        try:
            rgb = helpers.hsv_to_rgb(*hsv)
            rgb = rgb[0]**tf, rgb[1]**tf, rgb[2]**tf
            hsv = helpers.rgb_to_hsv(*rgb)
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

    def matches(self, other, ignore=_BRUSHINFO_MATCH_IGNORES):
        s1 = self.settings.copy()
        s2 = other.settings.copy()
        for k in ignore:
            s1.pop(k, None)
            s2.pop(k, None)
        return s1 == s2


class Brush (mypaintlib.PythonBrush):
    """A brush, capable of painting to a surface

    Low-level extension of the C++ brush class, propagating all changes
    made to a BrushInfo instance down into the C brush struct.

    """

    def __init__(self, brushinfo):
        super(Brush, self).__init__()
        self.brushinfo = brushinfo
        brushinfo.observers.append(self._update_from_brushinfo)
        self._update_from_brushinfo(ALL_SETTINGS)

    def _update_from_brushinfo(self, settings):
        """Updates changed low-level settings from the BrushInfo"""
        for cname in settings:
            self._update_setting_from_brushinfo(cname)

    def _update_setting_from_brushinfo(self, cname):
        setting = brushsettings.settings_dict.get(cname)
        if not setting:
            return
        base = self.brushinfo.get_base_value(cname)
        self.set_base_value(setting.index, base)
        for input in brushsettings.inputs:
            points = self.brushinfo.get_points(cname, input.name,
                                               readonly=True)
            assert len(points) != 1
            self.set_mapping_n(setting.index, input.index, len(points))
            for i, (x, y) in enumerate(points):
                self.set_mapping_point(setting.index, input.index, i, x, y)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
