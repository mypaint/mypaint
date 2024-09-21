# This file is part of MyPaint.
# Copyright (C) 2016-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Exposes information about the settings and inputs libmypaint uses."""


# Imports:

from __future__ import division, print_function

import lib.mypaintlib as _mypaintlib


# Public variables:

#: List of info about all brush engine inputs known to libmypaint.
#: Elements are `BrushInputInfo` objects.
inputs = []

#: Contents of `inputs`, organized by internal name.
inputs_dict = {}

#: List of info about all brush settings known to libmypaint.
#: Elements are `BrushSettingInfo` objects.
settings = []

#: Contents of `settings`, organized by internal name.
settings_dict = {}

#: List of only the settings that should be visible in a brush editor UI.
settings_visible = []

#: Migration plan and calculations for old setting names.
settings_migrate = {
    # old cname              new cname        scale function
    'color_hue': ('change_color_h', lambda y: y * 64.0 / 360.0),
    'color_saturation': ('change_color_hsv_s', lambda y: y * 128.0 / 256.0),
    'color_value': ('change_color_v', lambda y: y * 128.0 / 256.0),
    'speed_slowness': ('speed1_slowness', None),
    'change_color_s': ('change_color_hsv_s', None),
    'stroke_treshold': ('stroke_threshold', None),
}


# Class definitions:

class BrushInputInfo:
    """Information about an input known by the libmypaint brush engine.

    Inputs are reflections of the data your tablet pen sends, and
    include things like tilt, pressure, and position. There are special
    inputs too, such as the speed and direction of drawing, a random
    input, and one which can be specified by the user to contain the
    effect of one or more settings.

    >>> for bi in inputs:
    ...     assert 0 <= bi.index < len(inputs)
    ...     assert bi.name in inputs_dict
    ...     for f in BrushInputInfo._FIELDS:
    ...         assert getattr(bi, f) is not None
    ...     assert bi.hard_max > bi.hard_min
    ...     assert bi.soft_max > bi.soft_min
    ...     assert bi.hard_max >= bi.soft_max
    ...     assert bi.hard_min <= bi.soft_min

    :ivar str name: Internal setting name.
    :ivar float hard_min:
    :ivar float soft_min:
    :ivar float normal:
    :ivar float hard_max:
    :ivar float soft_max:
    :ivar unicode dname: Localized human-readable name.
    :ivar unicode tooltip: Localized short technical description.
    :ivar int index: Index of the object in `inputs`.

    """

    _FIELDS = (
        "name",
        "hard_min", "soft_min",
        "normal",
        "hard_max", "soft_max",
        "dname", "tooltip",
        "index",
    )

    def __init__(self, **kwargs):
        for k in self._FIELDS:
            setattr(self, k, kwargs.get(k))

    def __repr__(self):
        return u"<{classname} {fields}>".format(
            classname = self.__class__.__name__,
            fields = {k: getattr(self, k, None) for k in self._FIELDS},
        )


class BrushSettingInfo:
    """Representation of a brush setting known to libmypaint.

    Settings are what the user tweaks in the brush editor. They include
    things like the radius or the smudge factor. Think of them as
    user-defined functions that use the current value of zero or more
    BrushInputs to form a single output value.

    >>> for bs in settings:
    ...      assert 0 <= bs.index < len(settings)
    ...      assert bs.cname in settings_dict
    ...      for f in BrushSettingInfo._FIELDS:
    ...          assert getattr(bs, f) is not None
    ...      assert bs.max > bs.min

    :ivar str cname: Internal name.
    :ivar str name: Localized, short, human-readable name.
    :ivar bool constant:
    :ivar float min:
    :ivar float default:
    :ivar float max:
    :ivar unicode tooltip: Localized, human-readable technical description.
    :ivar int index: Index of the object in `settings`.

    """

    _FIELDS = [
        "cname",
        "name",
        "constant",
        "min", "default", "max",
        "tooltip",
        "index",
    ]

    def __init__(self, **kwargs):
        for k in self._FIELDS:
            setattr(self, k, kwargs.get(k))


# Module initialization:

for i, info_dict in enumerate(_mypaintlib.get_libmypaint_brush_inputs()):
    info_dict["index"] = i
    input = BrushInputInfo(**info_dict)
    inputs.append(input)
    inputs_dict[input.name] = input

for i, info_dict in enumerate(_mypaintlib.get_libmypaint_brush_settings()):
    info_dict["index"] = i
    setting = BrushSettingInfo(**info_dict)
    settings.append(setting)
    settings_dict[setting.cname] = setting

_settings_hidden = set(["color_h", "color_s", "color_v"])
for s in settings:
    if s.cname in _settings_hidden:
        continue
    settings_visible.append(s)


# Module testing:

def _test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _test()
