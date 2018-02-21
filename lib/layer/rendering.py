# This file is part of MyPaint.
# Copyright (C) 2017 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Abstract definitions of how to render layer trees.

Rendering is performed by RootLayerStack objects: see lib.layer.tree
and its "render_*()" methods.

"""

# Imports:

import abc


# Public constants:

class Opcode:
    """Opcode values for Renderable.get_render_ops()."""

    #: Composite some data into the backdrop.
    #: Format: (COMPOSITE, <TileCompositable>, modenum, opacity)
    COMPOSITE = 1

    #: Blit some data into the backdrop.
    #: Format: (BLIT, <TileBlittable>, None, None)
    BLIT = 2

    #: Push an empty tile onto the isolation stack.
    #: This creates a new isolated backdrop for later ops.
    #: Format: (PUSH, None, None, opacity)
    PUSH = 3

    #: Pop the isolation stack & composite the removed tile.
    #: The removed backdrop is composited onto the previous one.
    #: Format: (POP, None, modenum, opacity)
    POP = 4


# Classes and interfaces:

class Spec (object):
    """Selection criteria for Renderable.get_render_ops()."""

    def __init__(self, **kwargs):
        """Initialize with optional field info from **kwargs."""
        super(Spec, self).__init__()

        #: Limitation: render *only* these layers, if specified.
        #: If a child layer is present here,
        #: its entire parent chain must be present too.
        #: Default: None.
        #: Type: set() of Renderables.
        self.layers = kwargs.get("layers", None)

        #: The currently active layer.
        #: Renderables may choose to render themselves differently
        #: if they are the current layer. Other flags may apply too.
        #: Default: None.
        self.current = kwargs.get("current", None)

        #: Flag: the layer in "current" is being shown in solo mode.
        #: The "layers" field is expected to hold all the parents.
        #: Default: False.
        self.solo = kwargs.get("solo", False)

        #: Flag: the layer in "current" is being shown as a preview effect.
        #: The "layers" field is expected to hold all the parents.
        #: Default: False.
        self.previewing = kwargs.get("previewing", False)

        #: Effect: a Renderable to be rendered on top of everything.
        #: Default: None.
        self.global_overlay = kwargs.get("global_overlay", None)

        #: Effect: a Renderable to be rendered on top of "current".
        #: See also "global_overlay".
        #: Default: None.
        self.current_overlay = kwargs.get("current_overlay", None)

        #: Tristate flag: override for the the visible state of the
        #: builtin background layer. Leave this one unset if possible.
        #: Default: None (use the background layer's own visibility flag)
        flagval = kwargs.get("background", None)
        if flagval is not None:
            flagval = bool(flagval)
        self.background = flagval

    def cacheable(self):
        """True if render methods can usefully cache output for this spec."""
        result = (self.current_overlay is None)
        result &= (self.global_overlay is None)
        result &= not bool(self.solo)
        result &= not bool(self.previewing)
        result &= (self.layers is None)
        result &= (self.background is None)
        return result


class Renderable:
    """Abstract interface for elements that can be rendered."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_render_ops(self, spec):
        """Returns a flat sequence of rendering ops.

        Implementations of this function are expected
        to perform all decisions about what to render,
        and it is reasonable to make quite complex decisions here.
        This method will be called on the entire layer stack
        before any tiles are accessed and composited.
        It's expected that it will visit descendents recursively
        and serve up a small program in rendering order.

        :param lib.layer.rendering.Spec spec: what to render.
        :rtype: sequence
        :returns: sequence of ops

        The "spec" parameter works pretty much as a query.
        The tree of renderables being searched/visited
        may choose to return different things
        based on what it contains.
        See: lib.layer.rendering.Spec.

        The returned sequence of ops needs to be completely flat.
        Each op is a tuple, [(OPCODE, DATA, MODE, OPACITY), ...]

        OPCODE is an Opcode constant.
        The DATA, MODE, and OPACITY members will vary
        according to the specific operation.
        See lib.layer.rendering.Opcode for details.

        """
