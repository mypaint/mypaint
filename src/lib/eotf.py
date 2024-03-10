# Copyright (C) 2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""A configurable eotf value
"""

# Overall, this is a bit ugly, but due to the number of call chains
# that would need to be updated with the same copy-pasted code, this
# is still probably the best we can do as a post-fix, since nothing
# was built around using an eotf to begin with.

DEFAULT_EOTF = 2.2
__EOTF = dict()


# Value used to transform color channels on
# load/save and other color operations.

def set_eotf(value):
    """Set the EOTF value to be used for color transforms"""
    assert isinstance(value, float)
    __EOTF['current'] = value


def eotf():
    """Get the current EOTF value, used for color transforms"""
    return __EOTF.get('current', DEFAULT_EOTF)


# Session-constant base EOTF - the one to fall back to
# when inverting already applied color transforms.

def set_base_eotf(value):
    """Set the base EOTF value used to invert transforms"""
    assert isinstance(value, float)
    __EOTF['base'] = value


def base_eotf():
    """Get the base eotf, configured in the user preferences"""
    return __EOTF.get('base', DEFAULT_EOTF)
