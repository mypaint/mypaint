# -*- coding: utf-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2019 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


class CompatFileBehaviorConfig:

    # Key for the behavior setting in user preferences
    SETTING = 'compat_behavior_when_unknown'

    # Setting options
    ALWAYS_1X = 'always-1.x'
    ALWAYS_2X = 'always-2.x'
    UNLESS_PIGMENT_LAYER_1X = 'unless-pigment-layer-1.x'


CFBC = CompatFileBehaviorConfig

# Keys for settings in the user preferences
DEFAULT_COMPAT = 'default_compatibility_mode'
COMPAT_SETTINGS = 'compability_settings'

# Keys for compat mode sub-options in the user preferences
PIGMENT_BY_DEFAULT = 'pigment_on_by_default'
PIGMENT_LAYER_BY_DEFAULT = 'pigment_layer_is_default'

C1X = '1.x'
C2X = '2.x'

# Default compatibility settings
DEFAULT_CONFIG = {
    CFBC.SETTING:
    CFBC.UNLESS_PIGMENT_LAYER_1X,
    DEFAULT_COMPAT: C2X,
    COMPAT_SETTINGS: {
        C1X: {
            PIGMENT_BY_DEFAULT: False,
            PIGMENT_LAYER_BY_DEFAULT: False,
        },
        C2X: {
            PIGMENT_BY_DEFAULT: True,
            PIGMENT_LAYER_BY_DEFAULT: True,
        },
    },
}
