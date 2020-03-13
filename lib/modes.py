# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layer mode constants"""

from __future__ import division, print_function

from gettext import gettext as _
import lib.mypaintlib

#: Additional pass-through mode for layer groups (not saved, but reflected
#: into other flags which are saved)
PASS_THROUGH_MODE = -1


#: Valid modes for all layers
STANDARD_MODES = tuple(range(lib.mypaintlib.NumCombineModes))


#: Extra modes valid only for sub-stacks (groups)
STACK_MODES = (PASS_THROUGH_MODE,)


#: The default layer combine mode - overridable
_DEFAULT_MODE = lib.mypaintlib.CombineSpectralWGM


def set_default_mode(mode):
    assert mode in STANDARD_MODES
    global _DEFAULT_MODE
    _DEFAULT_MODE = mode


def default_mode():
    return _DEFAULT_MODE


#: UI strings (label, tooltip) for the layer modes
MODE_STRINGS = {
    # Group modes
    PASS_THROUGH_MODE: (
        _("Pass-through"),
        _("Group contents apply directly to the group's backdrop")),
    # Standard blend modes (using src-over compositing)
    lib.mypaintlib.CombineNormal: (
        _("Normal"),
        _("The top layer only, without blending colors.")),
    lib.mypaintlib.CombineSpectralWGM: (
        _("Pigment"),
        _("Similar to mixing actual pigments by upsampling "
          "to 10 spectral channels.")),
    lib.mypaintlib.CombineMultiply: (
        _("Multiply"),
        _("Similar to loading two slides into a projector and "
          "projecting the combined result.")),
    lib.mypaintlib.CombineScreen: (
        _("Screen"),
        _("Like shining two separate slide projectors onto a screen "
          "simultaneously. This is the inverse of 'Multiply'.")),
    lib.mypaintlib.CombineOverlay: (
        _("Overlay"),
        _("Overlays the backdrop with the top layer, preserving the "
          "backdrop's highlights and shadows. This is the inverse "
          "of 'Hard Light'.")),
    lib.mypaintlib.CombineDarken: (
        _("Darken"),
        _("The top layer is used where it is darker than "
          "the backdrop.")),
    lib.mypaintlib.CombineLighten: (
        _("Lighten"),
        _("The top layer is used where it is lighter than "
          "the backdrop.")),
    lib.mypaintlib.CombineColorDodge: (
        _("Dodge"),
        _("Brightens the backdrop using the top layer. The effect is "
          "similar to the photographic darkroom technique of the same "
          "name which is used for improving contrast in shadows.")),
    lib.mypaintlib.CombineColorBurn: (
        _("Burn"),
        _("Darkens the backdrop using the top layer. The effect looks "
          "similar to the photographic darkroom technique of the same "
          "name which is used for reducing over-bright highlights.")),
    lib.mypaintlib.CombineHardLight: (
        _("Hard Light"),
        _("Similar to shining a harsh spotlight onto the backdrop.")),
    lib.mypaintlib.CombineSoftLight: (
        _("Soft Light"),
        _("Like shining a diffuse spotlight onto the backdrop.")),
    lib.mypaintlib.CombineDifference: (
        _("Difference"),
        _("Subtracts the darker color from the lighter of the two.")),
    lib.mypaintlib.CombineExclusion: (
        _("Exclusion"),
        _("Similar to the 'Difference' mode, but lower in contrast.")),
    # Nonseparable blend modes (with src-over compositing)
    lib.mypaintlib.CombineHue: (
        _("Hue"),
        _("Combines the hue of the top layer with the saturation and "
          "luminosity of the backdrop.")),
    lib.mypaintlib.CombineSaturation: (
        _("Saturation"),
        _("Applies the saturation of the top layer's colors to the "
          "hue and luminosity of the backdrop.")),
    lib.mypaintlib.CombineColor: (
        _("Color"),
        _("Applies the hue and saturation of the top layer to the "
          "luminosity of the backdrop.")),
    lib.mypaintlib.CombineLuminosity: (
        _("Luminosity"),
        _("Applies the luminosity of the top layer to the hue and "
          "saturation of the backdrop.")),
    # Compositing operators (using normal blend mode)
    lib.mypaintlib.CombineLighter: (
        _("Plus"),
        _("This layer and its backdrop are simply added together.")),
    lib.mypaintlib.CombineDestinationIn: (
        _("Destination In"),
        _("Uses the backdrop only where this layer covers it. "
          "Everything else is ignored.")),
    lib.mypaintlib.CombineDestinationOut: (
        _("Destination Out"),
        _("Uses the backdrop only where this layer doesn't cover it. "
          "Everything else is ignored.")),
    lib.mypaintlib.CombineSourceAtop: (
        _("Source Atop"),
        _("Source which overlaps the destination, replaces the destination. "
          "Destination is placed elsewhere.")),
    lib.mypaintlib.CombineDestinationAtop: (
        _("Destination Atop"),
        _("Destination which overlaps the source replaces the source. "
          "Source is placed elsewhere.")),
}
for mode in STANDARD_MODES + STACK_MODES:
    assert mode in MODE_STRINGS


#: Name to layer combine mode lookup used when loading OpenRaster
ORA_MODES_BY_OPNAME = {
    lib.mypaintlib.combine_mode_get_info(mode)["name"]: mode
    for mode in range(lib.mypaintlib.NumCombineModes)
}


#: Layer modes which sometimes lower the alpha of their backdrop
MODES_DECREASING_BACKDROP_ALPHA = {
    m for m in range(lib.mypaintlib.NumCombineModes)
    if lib.mypaintlib.combine_mode_get_info(m).get("can_decrease_alpha")
}


#: Layer modes which, even with alpha==0, sometimes alter their backdrops
MODES_EFFECTIVE_AT_ZERO_ALPHA = {
    m for m in range(lib.mypaintlib.NumCombineModes)
    if lib.mypaintlib.combine_mode_get_info(m).get("zero_alpha_has_effect")
}


#: Layer modes which *always* set the backdrop's alpha to zero
#: if their own alpha is zero.
MODES_CLEARING_BACKDROP_AT_ZERO_ALPHA = {
    m for m in range(lib.mypaintlib.NumCombineModes)
    if lib.mypaintlib.combine_mode_get_info(m).get(
        "zero_alpha_clears_backdrop"
    )
}
