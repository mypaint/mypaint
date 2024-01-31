# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Display filter effects."""


## Imports
from __future__ import division, print_function

import numpy as np


## Constants

# sRGB coefficients: the usual sRGB / Rec. 709 ones

_LUMA_COEFFS = (0.2126, 0.7152, 0.0722)

# Conversion matrices for simulating common color vision deficiencies
#
# The coefficients come from Loren Petrich's work, formerly at
# http://homepage.mac.com/lpetrich/ColorBlindnessSim/ColorBlindnessSim.html
# and Max Novakovic's Javascript implementation, formerly at
# http://disturbmedia.com/max/colour-blindness.html

_SIM_DEUTERANOPIA_R_COEFFS = (0.43, 0.72, -0.15)
_SIM_DEUTERANOPIA_G_COEFFS = (0.34, 0.57, 0.09)
_SIM_DEUTERANOPIA_B_COEFFS = (-0.02, 0.03, 1)

_SIM_PROTANOPIA_R_COEFFS = (0.2, 0.99, -0.19)
_SIM_PROTANOPIA_G_COEFFS = (0.16, 0.79, 0.04)
_SIM_PROTANOPIA_B_COEFFS = (0.01, 0.01, 1)

_SIM_TRITANOPIA_R_COEFFS = (0.972, 0.112, -0.084)
_SIM_TRITANOPIA_G_COEFFS = (0.022, 0.818, 0.160)
_SIM_TRITANOPIA_B_COEFFS = (-0.063, 0.881, 0.182)


## Filter functions


def luma_only(dst):
    """Convert an NxNx3 array to show only luma (brightness)"""
    luma = (dst[...,0:3] * _LUMA_COEFFS).sum(axis=2)
    dst[..., 0:3] = luma[..., np.newaxis]


def invert_colors(dst):
    """Invert each RGB channel in an RGB array"""
    dst[...,0:3] = 255 - dst[...,0:3]


def sim_deuteranopia(dst):
    """Simulate deuteranopia (insensitivity to red)"""
    r = (dst[...,0:3] * _SIM_DEUTERANOPIA_R_COEFFS).sum(axis=2)
    g = (dst[...,0:3] * _SIM_DEUTERANOPIA_G_COEFFS).sum(axis=2)
    b = (dst[...,0:3] * _SIM_DEUTERANOPIA_B_COEFFS).sum(axis=2)
    np.clip(r, 0, 255, dst[..., 0])
    np.clip(g, 0, 255, dst[..., 1])
    np.clip(b, 0, 255, dst[..., 2])


def sim_protanopia(dst):
    """Simulate protanopia (insensitivity to green)"""
    r = (dst[...,0:3] * _SIM_PROTANOPIA_R_COEFFS).sum(axis=2)
    g = (dst[...,0:3] * _SIM_PROTANOPIA_G_COEFFS).sum(axis=2)
    b = (dst[...,0:3] * _SIM_PROTANOPIA_B_COEFFS).sum(axis=2)
    np.clip(r, 0, 255, dst[..., 0])
    np.clip(g, 0, 255, dst[..., 1])
    np.clip(b, 0, 255, dst[..., 2])


def sim_tritanopia(dst):
    """Simulate tritanopia (insensitivity to green)"""
    r = (dst[...,0:3] * _SIM_TRITANOPIA_R_COEFFS).sum(axis=2)
    g = (dst[...,0:3] * _SIM_TRITANOPIA_G_COEFFS).sum(axis=2)
    b = (dst[...,0:3] * _SIM_TRITANOPIA_B_COEFFS).sum(axis=2)
    np.clip(r, 0, 255, dst[..., 0])
    np.clip(g, 0, 255, dst[..., 1])
    np.clip(b, 0, 255, dst[..., 2])
