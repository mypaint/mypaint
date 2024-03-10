# This file is part of MyPaint.
# -*- encoding: utf-8 -*-
# Copyright (C) 2011-2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Whole-tree-level layer classes and functions"""


## Imports

from __future__ import division, print_function

import re
import logging
from copy import copy
from copy import deepcopy
import os.path
from warnings import warn
import contextlib

from lib.gibindings import GdkPixbuf
from lib.gibindings import GLib
import numpy as np

from lib.eotf import eotf
from lib.gettext import C_
import lib.mypaintlib
import lib.tiledsurface as tiledsurface
from lib.tiledsurface import TileAccessible
from lib.tiledsurface import TileBlittable
import lib.helpers as helpers
from lib.observable import event
import lib.pixbuf
import lib.cache
from lib.modes import PASS_THROUGH_MODE
from lib.modes import MODES_DECREASING_BACKDROP_ALPHA
from . import data
from . import group
from . import core
from . import rendering
import lib.feedback
import lib.naming
from lib.pycompat import xrange


logger = logging.getLogger(__name__)


## Class defs


class PlaceholderLayer (group.LayerStack):
    """Trivial temporary placeholder layer, used for moves etc.

    The layer stack architecture does not allow for the same layer
    appearing twice in a tree structure. Layer operations therefore
    require unique placeholder layers occasionally, typically when
    swapping nodes in the tree or handling drags.
    """

    DEFAULT_NAME = C_(
        "layer default names",
        # TRANSLATORS: Short default name for temporary placeholder layers.
        # TRANSLATORS: (The user should never see this except in error cases)
        u"Placeholder",
    )


class RootLayerStack (group.LayerStack):
    """Specialized document root layer stack

    In addition to the basic lib.layer.group.LayerStack implementation,
    this class's methods and properties provide:

     * the document's background, using an internal BackgroundLayer;
     * tile rendering for the doc via the regular rendering interface;
     * special viewing modes (solo, previewing);
     * the currently selected layer;
     * path-based access to layers in the tree;
     * a global symmetry axis for painting;
     * manipulation of layer paths; and
     * convenient iteration over the tree structure.

    In other words, root layer stacks handle anything that needs
    document-scale oversight of the tree structure to operate.  An
    instance of this instantiated for the running app as part of the
    primary `lib.document.Document` object.  The descendent layers of
    this object are those that are presented as user-addressable layers
    in the Layers panel.

    Be careful to maintain global uniqueness of layers within the root
    layer stack. If this isn't respected, then replacing an instance of
    item which exists in two or more places in the tree will break that
    layer's root reference and cause it to silently stop emitting
    updates. Use a `PlaceholderLayer` to work around this, or just
    reinstate the root ref when you're done juggling layers.

    """

    ## Class constants

    DEFAULT_NAME = C_(
        "layer default names",
        u"Root",
    )
    INITIAL_MODE = lib.mypaintlib.CombineNormal
    PERMITTED_MODES = {INITIAL_MODE}

    ## Initialization

    def __init__(self, doc=None,
                 cache_size=lib.cache.DEFAULT_CACHE_SIZE,
                 **kwargs):
        """Construct, as part of a model

        :param doc: The model document. May be None for testing.
        :type doc: lib.document.Document
        :param cache_size: size of the layer render cache
        :type cache_size: int
        """
        super(RootLayerStack, self).__init__(**kwargs)
        self.doc = doc
        self._render_cache = lib.cache.LRUCache(capacity=cache_size)
        # Background
        default_bg = (255, 255, 255)
        self._default_background = default_bg
        self._background_layer = data.BackgroundLayer(default_bg)
        self._background_visible = True
        # Symmetry - the `unset` flag is used to decide on whether to reset
        # the symmetry state - to e.g. set the center based on the viewport,
        # or the document bounds.
        self._symmetry_unset = True
        self._symmetry_active = False
        self._symmetry_type = lib.mypaintlib.SymmetryVertical
        self._symmetry_center = (0, 0)
        self._symmetry_angle = 0
        self._symmetry_lines = 2
        # Special rendering state
        self._current_layer_solo = False
        self._current_layer_previewing = False
        # Current layer
        self._current_path = ()
        # Temporary overlay for the current layer
        self._current_layer_overlay = None
        # Self-observation
        self.layer_content_changed += self._render_cache_clear_area
        self.layer_properties_changed += self._render_cache_clear
        self.layer_deleted += self._render_cache_clear
        self.layer_inserted += self._render_cache_clear
        # Layer thumbnail updates
        self.layer_content_changed += self._mark_layer_for_rethumb
        self._rethumb_layers = []
        self._rethumb_layers_timer_id = None

    # Render cache management:

    def _render_cache_get(self, key1, key2):
        try:
            cache2 = self._render_cache[key1]
            return cache2[key2]
        except KeyError:
            pass
        return None

    def _render_cache_set(self, key1, key2, data):
        try:
            cache2 = self._render_cache[key1]
        except KeyError:
            cache2 = dict()  # it'll have ~MAX_MIPMAP_LEVEL items
            self._render_cache[key1] = cache2
        cache2[key2] = data

    def _render_cache_clear_area(self, root, layer, x, y, w, h):
        """Clears rendered tiles from the cache in a specific area."""

        if (w <= 0) or (h <= 0):  # update all notifications
            self._render_cache_clear()
            return

        n = lib.mypaintlib.TILE_SIZE
        tx_min = x // n
        tx_max = ((x + w) // n)
        ty_min = y // n
        ty_max = ((y + h) // n)
        mipmap_level_max = lib.mypaintlib.MAX_MIPMAP_LEVEL

        for tx in range(tx_min, tx_max + 1):
            for ty in range(ty_min, ty_max + 1):
                for level in range(0, mipmap_level_max + 1):
                    fac = 2 ** level
                    key = ((tx // fac), (ty // fac), level)
                    self._render_cache.pop(key, None)

    def _render_cache_clear(self, *_ignored):
        """Clears all rendered tiles from the cache."""
        self._render_cache.clear()

    # Global ops:

    def clear(self):
        """Clear the layer and set the default background"""
        super(RootLayerStack, self).clear()
        self.set_background(self._default_background)
        self.current_path = ()
        self._render_cache_clear()

    def ensure_populated(self, layer_class=None):
        """Ensures that the stack is non-empty by making a new layer if needed

        :param layer_class: The class of layer to add, if necessary
        :type layer_class: LayerBase
        :returns: The new layer instance, or None if nothing was created

        >>> root = RootLayerStack(None); root
        <RootLayerStack len=0>
        >>> root.ensure_populated(layer_class=group.LayerStack); root
        <LayerStack len=0>
        <RootLayerStack len=1>

        The default `layer_class` is the regular painting layer.

        >>> root.clear(); root
        <RootLayerStack len=0>
        >>> root.ensure_populated(); root
        <PaintingLayer>
        <RootLayerStack len=1>

        """
        if layer_class is None:
            layer_class = data.PaintingLayer
        layer = None
        if len(self) == 0:
            layer = layer_class()
            self.append(layer)
            self._current_path = (0,)
        return layer

    def remove_empty_tiles(self):
        """Removes empty tiles in all layers backed by a tiled surface.

        :returns: Stats about the removal: (nremoved, ntotal)
        :rtype: tuple

        """
        removed, total = (0, 0)
        for path, layer in self.walk():
            try:
                remove_method = layer.remove_empty_tiles
            except AttributeError:
                continue
            r, t = remove_method()
            removed += r
            total += t
        logger.debug(
            "remove_empty_tiles: removed %d of %d tiles",
            removed, total,
        )
        return (removed, total)

    ## Terminal root access

    @property
    def root(self):
        """Layer stack root: itself, in this case"""
        return self

    @root.setter
    def root(self, newroot):
        raise ValueError("Cannot set the root of the root layer stack")

    ## Info methods

    def get_names(self):
        """Returns the set of unique names of all descendents"""
        return set((l.name for l in self.deepiter()))

    ## Rendering: root stack API

    def _get_render_background(self, spec):
        """True if render() should render the internal background

        :rtype: bool

        This reflects the background visibility flag normally,
        but the layer-previewing flag inverts its effect.
        This has the effect of making the current layer
        blink very appreciably when changing layers.

        Solo mode never shows the background, currently.
        If this changes, layer normalization and thumbnails will break.

        See also: background_visible, current_layer_previewing.

        """
        if spec.background is not None:
            return spec.background
        if spec.previewing:
            return not self._background_visible
        elif spec.solo:
            return False
        else:
            return self._background_visible

    def get_render_is_opaque(self, spec=None):
        """True if the rendering is known to be 100% opaque

        :rtype: bool

        The UI should draw its own checquered background if this is
        false, and expect `render()` to write RGBA data with lots
        of transparent areas.

        Even if the special background layer is enabled, it may be
        knocked out by certain compositing modes of layers above it.

        """
        if spec is None:
            spec = rendering.Spec(
                current=self.current,
                previewing=self._current_layer_previewing,
                solo=self._current_layer_solo,
            )
        if not self._get_render_background(spec):
            return False
        for path, layer in self.walk(bghit=True, visible=True):
            if layer.mode in MODES_DECREASING_BACKDROP_ALPHA:
                return False
        return True

    def layers_along_path(self, path):
        """Yields all layers along a path, not including the root"""
        if not path:
            return
        unused_path = list(path)
        layer = self
        while len(unused_path) > 0:
            if not isinstance(layer, group.LayerStack):
                break
            idx = unused_path.pop(0)
            if not (0 <= idx < len(layer)):
                break
            layer = layer[idx]
            yield layer

    def layers_along_or_under_path(self, path, no_hidden_descendants=False):
        """All parents, and all descendents of a path."""
        path = tuple(path)
        hidden_paths = set()
        for p, layer in self.walk():
            if not (path[0:len(p)] == p or    # ancestor of p, or p itself
                    p[0:len(path)] == path):  # descendent of p
                continue
            # Conditionally exclude hidden child layers
            if no_hidden_descendants and len(p) > len(path):
                if not layer.visible or p[:-1] in hidden_paths:
                    if isinstance(layer, group.LayerStack):
                        hidden_paths.add(p)
                    continue
            yield layer


    def _get_render_spec(self, respect_solo=True, respect_previewing=True):
        """Get a specification object for rendering the current state.

        The returned spec object captures the current layer and all the
        fiddly bits about layer preview and solo state, and exactly what
        layers to render when one of those modes is active.

        """
        spec = rendering.Spec(current=self.current)

        if respect_solo:
            spec.solo = self._current_layer_solo
        if respect_previewing:
            spec.previewing = self._current_layer_previewing
        if spec.solo or spec.previewing:
            path = self.get_current_path()
            spec.layers = set(self.layers_along_or_under_path(path))

        return spec

    def _get_backdrop_render_spec_for_layer(self, path):
        """Get a render spec for the backdrop of a layer.

        This method returns a spec object expressing the natural
        rendering of the backdrop to a specific layer path. This is used
        for extracts and subtractions.

        The backdrop consists of all layers underneath the layer in
        question, plus all of their parents.

        """
        seen_srclayer = False
        backdrop_layers = set()
        for p, layer in self.walk():
            if path_startswith(p, path):
                seen_srclayer = True
            elif seen_srclayer or isinstance(layer, group.LayerStack):
                backdrop_layers.add(layer)
        # For the backdrop, use a default rendering, respecting
        # all but transient effects.
        bd_spec = self._get_render_spec(respect_previewing=False)
        if bd_spec.layers is not None:
            backdrop_layers.intersection_update(bd_spec.layers)
        bd_spec.layers = backdrop_layers
        return bd_spec

    def render(self, surface, tiles, mipmap_level, overlay=None,
               opaque_base_tile=None, filter=None, spec=None,
               progress=None, background=None, alpha=None, **kwargs):
        """Render a batch of tiles into a tile-addressable surface.

        :param TileAccesible surface: The target surface.
        :param iterable tiles: The tile indices to render into "surface".
        :param int mipmap_level: downscale degree. Ensure tile indices match.
        :param lib.layer.core.LayerBase overlay: A global overlay layer.
        :param callable filter: Display filter (8bpc tile array mangler).
        :param lib.layer.rendering.Spec spec: Explicit rendering spec.
        :param lib.feedback.Progress progress: Feedback object.
        :param bool background: Render the background? (None means natural).
        :param bool alpha: Deprecated alias for "background" (reverse sense).
        :param **kwargs: Extensibility.

        This API may evolve to use only the "spec" argument rather than
        the explicit overlay etc.

        """
        if progress is None:
            progress = lib.feedback.Progress()
        tiles = list(tiles)
        progress.items = len(tiles)
        if len(tiles) == 0:
            progress.close()
            return

        if background is None and alpha is not None:
            warn("Use 'background' instead of 'alpha'", DeprecationWarning)
            background = not alpha

        if spec is None:
            spec = self._get_render_spec()
        if overlay is not None:
            spec.global_overlay = overlay
        if background is not None:
            spec.background = bool(background)

        dst_has_alpha = not self.get_render_is_opaque(spec=spec)
        ops = self.get_render_ops(spec)

        target_surface_is_8bpc = False
        use_cache = False
        tx, ty = tiles[0]
        with surface.tile_request(tx, ty, readonly=True) as sample_tile:
            target_surface_is_8bpc = (sample_tile.dtype == 'uint8')
            if target_surface_is_8bpc:
                use_cache = spec.cacheable()
        key2 = (id(opaque_base_tile), dst_has_alpha)

        # Rendering loop.
        # Keep this as tight as possible, and consider C++ parallelization.
        tiledims = (tiledsurface.N, tiledsurface.N, 4)
        dst_has_alpha_orig = dst_has_alpha
        for tx, ty in tiles:
            dst_8bpc_orig = None
            dst_has_alpha = dst_has_alpha_orig
            key1 = (tx, ty, mipmap_level)
            cache_hit = False

            with surface.tile_request(tx, ty, readonly=False) as dst:

                # Twirl out any 8bpc target here,
                # if the render cache is empty for this tile.
                if target_surface_is_8bpc:
                    dst_8bpc_orig = dst
                    dst = None
                    if use_cache:
                        dst = self._render_cache_get(key1, key2)

                    if dst is None:
                        dst = np.zeros(tiledims, dtype='uint16')
                    else:
                        cache_hit = True  # note: dtype is now uint8

                if not cache_hit:
                    # Render to dst.
                    # dst is a fix15 rgba tile

                    dst_over_opaque_base = None
                    if dst_has_alpha and opaque_base_tile is not None:
                        dst_over_opaque_base = dst
                        lib.mypaintlib.tile_copy_rgba16_into_rgba16(
                            opaque_base_tile,
                            dst_over_opaque_base,
                        )
                        dst = np.zeros(tiledims, dtype='uint16')

                    # Process the ops list.
                    self._process_ops_list(
                        ops,
                        dst, dst_has_alpha,
                        tx, ty, mipmap_level,
                    )

                    if dst_over_opaque_base is not None:
                        dst_has_alpha = False
                        lib.mypaintlib.tile_combine(
                            lib.mypaintlib.CombineNormal,
                            dst, dst_over_opaque_base,
                            False, 1.0,
                        )
                        dst = dst_over_opaque_base

                # If the target tile is fix15 already, we're done.
                if dst_8bpc_orig is None:
                    continue

                # Untwirl into the target 8bpc tile.
                if not cache_hit:
                    # Rendering just happened.
                    # Convert to 8bpc, and maybe store.
                    if dst_has_alpha:
                        conv = lib.mypaintlib.tile_convert_rgba16_to_rgba8
                    else:
                        conv = lib.mypaintlib.tile_convert_rgbu16_to_rgbu8
                    conv(dst, dst_8bpc_orig, eotf())

                    if use_cache:
                        self._render_cache_set(key1, key2, dst_8bpc_orig)
                else:
                    # An already 8pbc dst was loaded from the cache.
                    # It will match dst_has_alpha already.
                    dst_8bpc_orig[:] = dst

                dst = dst_8bpc_orig

                # Display filtering only happens when rendering
                # 8bpc for the screen.
                if filter is not None:
                    filter(dst)

            # end tile_request
            progress += 1
        progress.close()

    def render_layer_preview(self, layer, size=256, bbox=None, **options):
        """Render a standardized thumbnail/preview of a specific layer.

        :param lib.layer.core.LayerBase layer: The layer to preview.
        :param int size: Size of the output pixbuf.
        :param tuple bbox: Rectangle to render (x, y, w, h).
        :param **options: Passed to render().
        :rtype: GdkPixbuf.Pixbuf

        """
        x, y, w, h = self._validate_layer_bbox_arg(layer, bbox)

        mipmap_level = 0
        while mipmap_level < lib.tiledsurface.MAX_MIPMAP_LEVEL:
            if max(w, h) <= size:
                break
            mipmap_level += 1
            x //= 2
            y //= 2
            w //= 2
            h //= 2
        w = max(1, w)
        h = max(1, h)

        spec = self._get_render_spec_for_layer(layer)

        surface = lib.pixbufsurface.Surface(x, y, w, h)
        surface.pixbuf.fill(0x00000000)
        tiles = list(surface.get_tiles())
        self.render(surface, tiles, mipmap_level, spec=spec, **options)

        pixbuf = surface.pixbuf
        assert pixbuf.get_width() == w
        assert pixbuf.get_height() == h
        if not ((w == size) or (h == size)):
            pixbuf = helpers.scale_proportionally(pixbuf, size, size)
        return pixbuf

    def render_layer_as_pixbuf(self, layer, bbox=None, **options):
        """Render a layer as a GdkPixbuf.

        :param lib.layer.core.LayerBase layer: The layer to preview.
        :param tuple bbox: Rectangle to render (x, y, w, h).
        :param **options: Passed to render().
        :rtype: GdkPixbuf.Pixbuf

        The "layer" param must be a descendent layer or the root layer
        stack itself.

        The "bbox" parameter defaults to the natural data bounding box
        of "layer", and has a minimum size of one tile.

        """
        x, y, w, h = self._validate_layer_bbox_arg(layer, bbox)
        spec = self._get_render_spec_for_layer(layer)

        surface = lib.pixbufsurface.Surface(x, y, w, h)
        surface.pixbuf.fill(0x00000000)
        tiles = list(surface.get_tiles())
        self.render(surface, tiles, 0, spec=spec, **options)

        pixbuf = surface.pixbuf
        assert pixbuf.get_width() == w
        assert pixbuf.get_height() == h
        return pixbuf

    def render_layer_to_png_file(self, layer, filename, bbox=None, **options):
        """Render out to a PNG file. Used by LayerGroup.save_as_png()."""
        bbox = self._validate_layer_bbox_arg(layer, bbox)
        spec = self._get_render_spec_for_layer(layer)
        spec.background = options.get("render_background")
        rendering = _TileRenderWrapper(self, spec, use_cache=False)
        if "alpha" not in options:
            options["alpha"] = True
        lib.surface.save_as_png(rendering, filename, *bbox, **options)

    def get_tile_accessible_layer_rendering(self, layer):
        """Get a TileAccessible temporary rendering of a sublayer.

        :returns: A temporary rendering object with inbuilt tile cache.

        The result is used to implement flood_fill for layer types
        which don't contain their own tile-accessible data.

        """
        spec = self._get_render_spec_for_layer(
            layer, no_hidden_descendants=True
        )
        rendering = _TileRenderWrapper(self, spec)
        return rendering

    def _get_render_spec_for_layer(self, layer, no_hidden_descendants=False):
        """Get a standardized rendering spec for a single layer.

        :param layer: The layer to render, can be the RootLayerStack.
        :rtype: lib.layer.rendering.Spec

        This method prepares a standardized rendering spec that shows a
        specific sublayer by itself, or the root stack complete with
        background. The spec returned does not introduce any special
        effects, and ignores any special viewing modes. It is suitable
        for the standardized "render_layer_*()" methods.

        """
        spec = self._get_render_spec(
            respect_solo=False,
            respect_previewing=False,
        )
        if layer is not self:
            layer_path = self.deepindex(layer)
            if layer_path is None:
                raise ValueError(
                    "Layer is not a descendent of this RootLayerStack.",
                )
            layers = self.layers_along_or_under_path(
                layer_path, no_hidden_descendants)
            spec.layers = set(layers)
            spec.current = layer
            spec.solo = True
        return spec

    def render_single_tile(self, dst, dst_has_alpha,
                           tx, ty, mipmap_level=0,
                           layer=None, spec=None, ops=None):
        """Render one tile in a standardized way (by default).

        It's used in fix15 mode for enabling flood fill when the source
        is a group, or when sample_merged is turned on.

        """
        if ops is None:
            if spec is None:
                if layer is None:
                    layer = self.current
                spec = self._get_render_spec_for_layer(layer)
            ops = self.get_render_ops(spec)

        dst_is_8bpc = (dst.dtype == 'uint8')
        if dst_is_8bpc:
            dst_8bpc_orig = dst
            tiledims = (tiledsurface.N, tiledsurface.N, 4)
            dst = np.zeros(tiledims, dtype='uint16')

        self._process_ops_list(ops, dst, dst_has_alpha, tx, ty, mipmap_level)

        if dst_is_8bpc:
            if dst_has_alpha:
                conv = lib.mypaintlib.tile_convert_rgba16_to_rgba8
            else:
                conv = lib.mypaintlib.tile_convert_rgbu16_to_rgbu8
            conv(dst, dst_8bpc_orig, eotf())
            dst = dst_8bpc_orig

    def _validate_layer_bbox_arg(self, layer, bbox,
                                 min_size=lib.tiledsurface.TILE_SIZE):
        """Check a bbox arg, defaulting it to the data size of a layer."""
        min_size = int(min_size)
        if bbox is not None:
            x, y, w, h = (int(n) for n in bbox)
        else:
            x, y, w, h = layer.get_bbox()
        if w == 0 or h == 0:
            x = 0
            y = 0
            w = 1
            h = 1
        w = max(min_size, w)
        h = max(min_size, h)
        return (x, y, w, h)

    @staticmethod
    def _process_ops_list(ops, dst, dst_has_alpha, tx, ty, mipmap_level):
        """Process a list of ops to render a tile. fix15 data only!"""
        # FIXME: should this be expanded to cover caching and 8bpc
        # targets? It would save on some code duplication elsewhere.
        # On the other hand, this is sort of what a parallelized,
        # GIL-holding C++ loop body might look like.

        stack = []
        for (opcode, opdata, mode, opacity) in ops:
            if opcode == rendering.Opcode.COMPOSITE:
                opdata.composite_tile(
                    dst, dst_has_alpha, tx, ty,
                    mipmap_level=mipmap_level,
                    mode=mode, opacity=opacity,
                )
            elif opcode == rendering.Opcode.BLIT:
                opdata.blit_tile_into(
                    dst, dst_has_alpha, tx, ty,
                    mipmap_level,
                )
            elif opcode == rendering.Opcode.PUSH:
                stack.append((dst, dst_has_alpha))
                tiledims = (tiledsurface.N, tiledsurface.N, 4)
                dst = np.zeros(tiledims, dtype='uint16')
                dst_has_alpha = True
            elif opcode == rendering.Opcode.POP:
                src = dst
                (dst, dst_has_alpha) = stack.pop(-1)
                lib.mypaintlib.tile_combine(
                    mode,
                    src, dst, dst_has_alpha,
                    opacity,
                )
            else:
                raise RuntimeError(
                    "Unknown lib.layer.rendering.Opcode: %r",
                    opcode,
                )
        if len(stack) > 0:
            raise ValueError(
                "Ops list contains more PUSH operations "
                "than POPs. Rendering is incomplete."
            )

    ## Renderable implementation

    def get_render_ops(self, spec):
        """Get rendering instructions."""
        ops = []
        if self._get_render_background(spec):
            bg_opcode = rendering.Opcode.BLIT
            bg_surf = self._background_layer._surface
            ops.append((bg_opcode, bg_surf, None, None))
        for child_layer in reversed(self):
            ops.extend(child_layer.get_render_ops(spec))
        if spec.global_overlay is not None:
            ops.extend(spec.global_overlay.get_render_ops(spec))
        return ops

    # Symmetry state

    @property
    def symmetry_unset(self):
        return self._symmetry_unset

    @symmetry_unset.setter
    def symmetry_unset(self, unset):
        self._symmetry_unset = bool(unset)

    @property
    def symmetry_active(self):
        """Whether symmetrical painting is active.

        This is a convenience property for part of
        the state managed by `set_symmetry_state()`.
        """
        return self._symmetry_active

    @symmetry_active.setter
    def symmetry_active(self, active):
        self.set_symmetry_state(active)

    @property
    def symmetry_center(self):
        return self._symmetry_center

    @symmetry_center.setter
    def symmetry_center(self, center):
        self.set_symmetry_state(True, center=center)

    @property
    def symmetry_x(self):
        return self._symmetry_center[0]

    @symmetry_x.setter
    def symmetry_x(self, x):
        self.set_symmetry_state(True, center=(x, self._symmetry_center[1]))

    @property
    def symmetry_y(self):
        return self._symmetry_center[1]

    @symmetry_y.setter
    def symmetry_y(self, y):
        self.set_symmetry_state(True, center=(self._symmetry_center[0], y))

    @property
    def symmetry_type(self):
        return self._symmetry_type

    @symmetry_type.setter
    def symmetry_type(self, symmetry_type):
        self.set_symmetry_state(True, symmetry_type=symmetry_type)

    @property
    def symmetry_lines(self):
        return self._symmetry_lines

    @symmetry_lines.setter
    def symmetry_lines(self, symmetry_lines):
        self.set_symmetry_state(True, symmetry_lines=symmetry_lines)

    @property
    def symmetry_angle(self):
        return self._symmetry_angle

    @symmetry_angle.setter
    def symmetry_angle(self, symmetry_angle):
        self.set_symmetry_state(True, angle=symmetry_angle)

    def set_symmetry_state(
            self, active=None, center=None,
            symmetry_type=None, symmetry_lines=None, angle=None):
        """Set the central, propagated, symmetry state.

        The root layer stack specialization manages a central state,
        which is propagated to the current layer automatically.

        See `LayerBase.set_symmetry_state` for the params.
        This override allows the shared `center_x` to be ``None``:
        see `symmetry_x` for what that means.

        """
        if active is not None:
            active = bool(active)
            self._symmetry_active = active
        if center is not None:
            center = int(round(center[0])), int(round(center[1]))
            self._symmetry_center = center
        if symmetry_type is not None:
            symmetry_type = int(symmetry_type)
            self._symmetry_type = symmetry_type
        if symmetry_lines is not None:
            symmetry_lines = int(symmetry_lines)
            self._symmetry_lines = symmetry_lines
        if angle is not None:
            self._symmetry_angle = angle

        current = self.get_current()
        if current is not self:
            self._propagate_symmetry_state(current)
        self.symmetry_state_changed(
            active, center, symmetry_type, symmetry_lines, angle,
        )

    def _propagate_symmetry_state(self, layer):
        """Copy the symmetry state to the a descendant layer"""
        assert layer is not self
        layer.set_symmetry_state(
            self._symmetry_active,
            self._symmetry_center,
            self._symmetry_type,
            self._symmetry_lines,
            self._symmetry_angle,
        )

    @event
    def symmetry_state_changed(
            self, active, center, symmetry_type, symmetry_lines, angle):
        """Event: symmetry state changed

        An argument value of None means that the state value has not changed,
        allowing for granular updates but also making it necessary to add
        None-checks if the values are to be used.

        :param bool active: whether symmetry is enabled or not
        :param tuple center: the (x, y) coordinates of the symmetry center
        :param int symmetry_type: the symmetry type
        :param int symmetry_lines: new number of symmetry lines
        :param float angle: the angle of the symmetry line(s)
        """

    ## Current layer

    def get_current_path(self):
        """Get the current layer's path

        :rtype: tuple

        If the current path was set to a path which was invalid at the
        time of setting, the returned value is always an empty tuple for
        convenience of casting. This is however an invalid path for
        addressing sub-layers.
        """
        if not self._current_path:
            return ()
        return self._current_path

    def set_current_path(self, path):
        """Set the current layer path

        :param path: The path to use; will be trimmed until it fits
        :type path: tuple
        """
        if len(self) == 0:
            self._current_path = None
            self.current_path_updated(())
            return
        path = tuple(path)
        while len(path) > 0:
            layer = self.deepget(path)
            if layer is not None:
                self._propagate_symmetry_state(layer)
                break
            path = path[:-1]
        if len(path) == 0:
            path = None
        self._current_path = path
        self.current_path_updated(path)

    current_path = property(get_current_path, set_current_path)

    def get_current(self):
        """Get the current layer (also exposed as a read-only property)

        This returns the root layer stack itself if the current path
        doesn't address a sub-layer.
        """
        return self.deepget(self.get_current_path(), self)

    current = property(get_current)

    ## The background layer

    @property
    def background_layer(self):
        """The background layer (accessor)"""
        return self._background_layer

    def set_background(self, obj, make_default=False):
        """Set the background layer's surface from an object

        :param obj: Background object
        :type obj: layer.data.BackgroundLayer or tuple or numpy array
        :param make_default: make this the default bg for clear()
        :type make_default: bool

        The background object argument `obj` can be a background layer,
        or an RGB triple (uint8), or a HxWx4 or HxWx3 numpy array which
        can be either uint8 or uint16.

        Setting the background issues a full redraw for the root layer,
        and also issues the `background_changed` event. The background
        will also be made visible if it isn't already.
        """
        if isinstance(obj, data.BackgroundLayer):
            obj = obj._surface
        if not isinstance(obj, tiledsurface.Background):
            if isinstance(obj, GdkPixbuf.Pixbuf):
                obj = helpers.gdkpixbuf2numpy(obj)
            obj = tiledsurface.Background(obj)
        self._background_layer.set_surface(obj)
        if make_default:
            self._default_background = obj
        self.background_changed()
        if not self._background_visible:
            self._background_visible = True
            self.background_visible_changed()
        self.layer_content_changed(self, 0, 0, 0, 0)

    @event
    def background_changed(self):
        """Event: background layer data has changed"""

    @property
    def background_visible(self):
        """Whether the background is visible

        Accepts only values which can be converted to bool.  Changing
        the background visibility flag issues a full redraw for the root
        layer, and also issues the `background_changed` event.
        """
        return bool(self._background_visible)

    @background_visible.setter
    def background_visible(self, value):
        value = bool(value)
        old_value = self._background_visible
        self._background_visible = value
        if value != old_value:
            self.background_visible_changed()
            self.layer_content_changed(self, 0, 0, 0, 0)

    @event
    def background_visible_changed(self):
        """Event: the background visibility flag has changed"""

    ## Temporary overlays for the current layer (not saved)

    @property
    def current_layer_overlay(self):
        """A temporary overlay layer for the current layer.

        This isn't saved as part of the document, and strictly speaking
        it exists outside the doument tree. If it is present, then
        during rendering it is composited onto the current painting
        layer in isolation. The effect is as if the overlay were part of
        the current painting layer.

        The intent of this layer type is to collect together and preview
        sets of updates to the current layer in response to user input.
        The updates can then be applied all together by an action.
        Another possibility might be for brush preview special effects.

        The current layer overlay can be a group, which allows capture
        of masked drawing. If you want updates to propagate back to the
        root, the group needs to be set as the ``current_layer_overlay``
        first. Otherwise, ``root``s won't be hooked up and managed in
        the right order.

        >>> root = RootLayerStack()
        >>> root.append(data.SimplePaintingLayer())
        >>> root.append(data.SimplePaintingLayer())
        >>> root.set_current_path([1])
        >>> ovgroup = group.LayerStack()
        >>> root.current_layer_overlay = ovgroup
        >>> ovdata1 = data.SimplePaintingLayer()
        >>> ovdata2 = data.SimplePaintingLayer()
        >>> ovgroup.append(ovdata1)
        >>> ovgroup.append(ovdata2)
        >>> change_count = 0
        >>> def changed(*a):
        ...     global change_count
        ...     change_count += 1
        >>> root.layer_content_changed += changed
        >>> ovdata1.clear()
        >>> ovdata2.clear()
        >>> change_count
        2

        Setting the overlay or setting it to None generates content
        change notifications too.

        >>> root.current_layer_overlay = None
        >>> root.current_layer_overlay = data.SimplePaintingLayer()
        >>> change_count
        4

        """
        return self._current_layer_overlay

    @current_layer_overlay.setter
    def current_layer_overlay(self, overlay):
        old_overlay = self._current_layer_overlay
        self._current_layer_overlay = overlay
        self.current_layer_overlay_changed(old_overlay)

        updates = []
        if old_overlay is not None:
            old_overlay.root = None
            updates.append(old_overlay.get_full_redraw_bbox())
        if overlay is not None:
            overlay.root = self  # for redraw announcements
            updates.append(overlay.get_full_redraw_bbox())

        if updates:
            update_bbox = tuple(core.combine_redraws(updates))
            self.layer_content_changed(self, *update_bbox)

    @event
    def current_layer_overlay_changed(self, old):
        """Event: current_layer_overlay was altered"""

    ## Layer Solo toggle (not saved)

    @property
    def current_layer_solo(self):
        """Layer-solo state for the document

        Accepts only values which can be converted to bool.
        Altering this property issues the `current_layer_solo_changed`
        event, and a full `layer_content_changed` for the root stack.
        """
        return self._current_layer_solo

    @current_layer_solo.setter
    def current_layer_solo(self, value):
        # TODO: make this undoable
        value = bool(value)
        old_value = self._current_layer_solo
        self._current_layer_solo = value
        if value != old_value:
            self.current_layer_solo_changed()
            self.layer_content_changed(self, 0, 0, 0, 0)

    @event
    def current_layer_solo_changed(self):
        """Event: current_layer_solo was altered"""

    ## Current layer temporary preview state (not saved, used for blink)

    @property
    def current_layer_previewing(self):
        """Layer-previewing state, as used when blinking a layer

        Accepts only values which can be converted to bool.  Altering
        this property calls `current_layer_previewing_changed` and also
        issues a full `layer_content_changed` for the root stack.
        """
        return self._current_layer_previewing

    @current_layer_previewing.setter
    def current_layer_previewing(self, value):
        """Layer-previewing state, as used when blinking a layer"""
        value = bool(value)
        old_value = self._current_layer_previewing
        self._current_layer_previewing = value
        if value != old_value:
            self.current_layer_previewing_changed()
            self.layer_content_changed(self, 0, 0, 0, 0)

    @event
    def current_layer_previewing_changed(self):
        """Event: current_layer_previewing was altered"""

    ## Layer naming within the tree

    def get_unique_name(self, layer):
        """Get a unique name for a layer to use

        :param LayerBase layer: Any layer
        :rtype: unicode
        :returns: A unique name

        The returned name is guaranteed not to occur in the tree.  This
        method can be used before or after the layer is inserted into
        the stack.
        """
        existing = {l.name for path, l in self.walk()
                    if l is not layer
                    and l.name is not None}
        blank = re.compile(r'^\s*$')
        newname = layer._name
        if newname is None or blank.match(newname):
            newname = layer.DEFAULT_NAME
        return lib.naming.make_unique_name(newname, existing)

    ## Layer path manipulation

    def path_above(self, path, insert=False):
        """Return the path for the layer stacked above a given path

        :param path: a layer path
        :type path: list or tuple
        :param insert: get an insertion path
        :type insert: bool
        :return: the layer above `path` in walk order
        :rtype: tuple

        Normally this is used for locating the layer above a given node
        in the layers stack as the user sees it in a typical user
        interface:

          >>> root = RootLayerStack(doc=None)
          >>> for p, l in [ ([0], data.PaintingLayer()),
          ...               ([1], group.LayerStack()),
          ...               ([1, 0], group.LayerStack()),
          ...               ([1, 0, 0], group.LayerStack()),
          ...               ([1, 0, 0, 0], data.PaintingLayer()),
          ...               ([1, 1], data.PaintingLayer()) ]:
          ...    root.deepinsert(p, l)
          >>> root.path_above([1])
          (0,)

        Ascending the stack using this method can enter and leave
        subtrees:

          >>> root.path_above([1, 1])
          (1, 0, 0, 0)
          >>> root.path_above([1, 0, 0, 0])
          (1, 0, 0)

        There is no existing path above the topmost node in the stack:

          >>> root.path_above([0]) is None
          True

        This method can also be used to get a path for use with
        `deepinsert()` which will allow insertion above a particular
        existing layer.  Normally this is the same path as the input,

          >>> root.path_above([0, 1], insert=True)
          (0, 1)

        however for nonexistent paths, you're guaranteed to get back a
        valid insertion path:

          >>> root.path_above([42, 1, 101], insert=True)
          (0,)

        which of necessity is the insertion point for a new layer at the
        very top of the stack.
        """
        path = tuple(path)
        if len(path) == 0:
            raise ValueError("Path identifies the root stack")
        if insert:
            # Same sanity checks as for path_below()
            parent_path, index = path[:-1], path[-1]
            parent = self.deepget(parent_path, None)
            if parent is None:
                return (0,)
            else:
                index = max(0, index)
                return tuple(list(parent_path) + [index])
        p_prev = None
        for p, l in self.walk():
            p = tuple(p)
            if path == p:
                return p_prev
            p_prev = p
        return None

    def path_below(self, path, insert=False):
        """Return the path for the layer stacked below a given path

        :param path: a layer path
        :type path: list or tuple
        :param insert: get an insertion path
        :type insert: bool
        :return: the layer below `path` in walk order
        :rtype: tuple or None

        This method is the inverse of `path_above()`: it normally
        returns the tree path below its `path` as the user would see it
        in a typical user interface:

          >>> root = RootLayerStack(doc=None)
          >>> for p, l in [ ([0], data.PaintingLayer()),
          ...               ([1], group.LayerStack()),
          ...               ([1, 0], group.LayerStack()),
          ...               ([1, 0, 0], group.LayerStack()),
          ...               ([1, 0, 0, 0], data.PaintingLayer()),
          ...               ([1, 1], data.PaintingLayer()) ]:
          ...    root.deepinsert(p, l)
          >>> root.path_below([0])
          (1,)

        Descending the stack using this method can enter and leave
        subtrees:

          >>> root.path_below([1, 0])
          (1, 0, 0)
          >>> root.path_below([1, 0, 0, 0])
          (1, 1)

        There is no path below the lowest addressable layer:

          >>> root.path_below([1, 1]) is None
          True

        Asking for an insertion path tries to get you somewhere to put a
        new layer that would make intuitive sense.  For most kinds of
        layer, that means one at the same level as the reference point

          >>> root.path_below([0], insert=True)
          (1,)
          >>> root.path_below([1, 1], insert=True)
          (1, 2)
          >>> root.path_below([1, 0, 0, 0], insert=True)
          (1, 0, 0, 1)

        However for sub-stacks, the insert-path "below" the stack is
        that for a new node as the stack's top child node

          >>> root.path_below([1, 0], insert=True)
          (1, 0, 0)

        Another exception to the general rule is that invalid paths
        always have an insertion path "below" them:

          >> root.path_below([999, 42, 67], insert=True)
          (2,)

        although this of necessity returns the insertion point for a new
        layer at the very bottom of the stack.
        """
        path = tuple(path)
        if len(path) == 0:
            raise ValueError("Path identifies the root stack")
        if insert:
            parent_path, index = path[:-1], path[-1]
            parent = self.deepget(parent_path, None)
            if parent is None:
                return (len(self),)
            elif isinstance(self.deepget(path, None), group.LayerStack):
                return path + (0,)
            else:
                index = min(len(parent), index + 1)
                return parent_path + (index,)
        p_prev = None
        for p, l in self.walk():
            p = tuple(p)
            if path == p_prev:
                return p
            p_prev = p
        return None

    ## Layer bubbling

    def _bubble_layer(self, path, upstack):
        """Move a layer up or down, preserving the tree structure

        Parameters and return values are the same as for the public
        methods (`bubble_layer_up()`, `bubble_layer_down()`), with the
        following addition:

        :param upstack: true to bubble up, false to bubble down
        """
        path = tuple(path)
        if len(path) == 0:
            raise ValueError("Cannot reposition the root of the stack")

        parent_path, index = path[:-1], path[-1]
        parent = self.deepget(parent_path, self)
        assert index < len(parent)
        assert index > -1

        # Collapse sub-stacks when bubbling them (not sure about this)
        if False:
            layer = self.deepget(path)
            assert layer is not None
            if isinstance(layer, group.LayerStack) and len(path) > 0:
                self.collapse_layer(path)

        # The layer to be moved may already be at the end of its stack
        # in the direction we want; if so, remove it then insert it
        # one place beyond its parent in the bubble direction.
        end_index = 0 if upstack else (len(parent) - 1)
        if index == end_index:
            if parent is self:
                return False
            grandparent_path = parent_path[:-1]
            grandparent = self.deepget(grandparent_path, self)
            parent_index = grandparent.index(parent)
            layer = parent.pop(index)
            beyond_parent_index = parent_index
            if not upstack:
                beyond_parent_index += 1
            if len(grandparent_path) > 0:
                self.expand_layer(grandparent_path)
            grandparent.insert(beyond_parent_index, layer)
            return True

        # Move the layer within its current parent
        new_index = index + (-1 if upstack else 1)
        if new_index < len(parent) and new_index > -1:
            # A sibling layer is already at the intended position
            sibling = parent[new_index]
            if isinstance(sibling, group.LayerStack):
                # Ascend: remove layer & put it at the near end
                # of the sibling stack
                sibling_path = parent_path + (new_index,)
                self.expand_layer(sibling_path)
                layer = parent.pop(index)
                if upstack:
                    sibling.append(layer)
                else:
                    sibling.insert(0, layer)
                return True
            else:
                # Swap positions with the sibling layer.
                # Use a placeholder, otherwise the root ref will be
                # lost.
                layer = parent[index]
                placeholder = PlaceholderLayer(name="swap")
                parent[index] = placeholder
                parent[new_index] = layer
                parent[index] = sibling
                return True
        else:
            # Nothing there, move to the end of this branch
            layer = parent.pop(index)
            if upstack:
                parent.insert(0, layer)
            else:
                parent.append(layer)
            return True

    @event
    def collapse_layer(self, path):
        """Event: request that the UI collapse a given path"""

    @event
    def expand_layer(self, path):
        """Event: request that the UI expand a given path"""

    def bubble_layer_up(self, path):
        """Move a layer up through the stack

        :param path: Layer path identifying the layer to move
        :returns: True if the stack structure was modified

        Bubbling follows the layout of the tree and preserves its
        structure apart from the layers touched by the move, so it can
        be driven by the keyboard usefully. `bubble_layer_down()` is the
        exact inverse of this operation.

        These methods assume the existence of a UI which lays out layers
        from top to bottom down the page, and which shows nodes or rows
        for LayerStacks (groups) before their contents.  If the path
        identifies a substack, the substack is moved as a whole.

        Bubbling layers may issue several layer_inserted and
        layer_deleted events depending on what's moved, and may alter
        the current path too (see current_path_changed).
        """
        old_current = self.current
        modified = self._bubble_layer(path, True)
        if modified and old_current:
            self.current_path = self.canonpath(layer=old_current)
        return modified

    def bubble_layer_down(self, path):
        """Move a layer down through the stack

        :param path: Layer path identifying the layer to move
        :returns: True if the stack structure was modified

        This is the inverse operation to bubbling a layer up.
        Parameters, notifications, and return values are the same as
        those for `bubble_layer_up()`.
        """
        old_current = self.current
        modified = self._bubble_layer(path, False)
        if modified and old_current:
            self.current_path = self.canonpath(layer=old_current)
        return modified

    ## Simplified tree storage and access

    # We use a path concept that's similar to GtkTreePath's, but almost like a
    # key/value store if this is the root layer stack.

    def walk(self, visible=None, bghit=None):
        """Walks the tree, listing addressable layers & their paths

        The parameters control how the walk operates as well as limiting
        its generated output.

        :param visible: Only visible layers
        :type visible: bool
        :param bghit: Only layers compositing directly on the background
        :type bghit: bool
        :returns: Iterator yielding ``(path, layer)`` tuples
        :rtype: collections.Iterable

        Layer substacks are listed before their contents, but the root
        of the walk is always excluded::

            >>> from . import data
            >>> root = RootLayerStack(doc=None)
            >>> for p, l in [([0], data.PaintingLayer()),
            ...              ([1], group.LayerStack(name="A")),
            ...              ([1,0], data.PaintingLayer(name="B")),
            ...              ([1,1], data.PaintingLayer()),
            ...              ([2], data.PaintingLayer(name="C"))]:
            ...     root.deepinsert(p, l)
            >>> walk = list(root.walk())
            >>> root in {l for p, l in walk}
            False
            >>> walk[1]  # doctest: +ELLIPSIS
            ((1,), <LayerStack len=2 ...'A'>)
            >>> walk[2]  # doctest: +ELLIPSIS
            ((1, 0), <PaintingLayer ...'B'>)

        The default behaviour is to return all layers.  If `visible`
        is true, hidden layers are excluded.  This excludes child layers
        of invisible layer stacks as well as the invisible stacks
        themselves.

            >>> root.deepget([0]).visible = False
            >>> root.deepget([1]).visible = False
            >>> list(root.walk(visible=True))  # doctest: +ELLIPSIS
            [((2,), <PaintingLayer ...'C'>)]

        If `bghit` is true, layers which could never affect the special
        background layer are excluded from the listing.  Specifically,
        all children of isolated layers are excluded, but not the
        isolated layers themselves.

            >>> root.deepget([1]).mode = lib.mypaintlib.CombineMultiply
            >>> walk = list(root.walk(bghit=True))
            >>> root.deepget([1]) in {l for p, l in walk}
            True
            >>> root.deepget([1, 0]) in {l for p, l in walk}
            False

        The special background layer itself is never returned by walk().
        """
        queue = [((i,), c) for i, c in enumerate(self)]
        while len(queue) > 0:
            path, layer = queue.pop(0)
            if visible and not layer.visible:
                continue
            yield (path, layer)
            if not isinstance(layer, group.LayerStack):
                continue
            if bghit and (layer.mode != PASS_THROUGH_MODE):
                continue
            queue[:0] = [(path + (i,), c) for i, c in enumerate(layer)]

    def deepiter(self, visible=None):
        """Iterates across all descendents of the stack

        >>> from . import test
        >>> stack, leaves = test.make_test_stack()
        >>> len(list(stack.deepiter()))
        8
        >>> len(set(stack.deepiter())) == len(list(stack.deepiter())) # no dups
        True
        >>> stack not in stack.deepiter()
        True
        >>> () not in stack.deepiter()
        True
        >>> leaves[0] in stack.deepiter()
        True
        """
        return (t[1] for t in self.walk(visible=visible))

    def deepget(self, path, default=None):
        """Gets a layer based on its path

        >>> from . import test
        >>> stack, leaves = test.make_test_stack()
        >>> stack.deepget(()) is stack
        True
        >>> stack.deepget((0,1))
        <PaintingLayer '01'>
        >>> stack.deepget((0,))
        <LayerStack len=3 '0'>

        If the layer cannot be found, None is returned; however a
        different default value can be specified::

        >>> stack.deepget((42,0), None)
        >>> stack.deepget((0,11), default="missing")
        'missing'

        """
        if path is None:
            return default
        if len(path) == 0:
            return self
        unused_path = list(path)
        layer = self
        while len(unused_path) > 0:
            idx = unused_path.pop(0)
            if abs(idx) > (len(layer) - 1):
                return default
            layer = layer[idx]
            if unused_path:
                if not isinstance(layer, group.LayerStack):
                    return default
            else:
                return layer
        return default

    def deepinsert(self, path, layer):
        """Inserts a layer before the final index in path

        :param path: an insertion path: see below
        :type path: iterable of integers
        :param layer: the layer to insert
        :type layer: LayerBase

        Deepinsert cannot create sub-stacks. Every element of `path`
        before the final element must be a valid `list`-style ``[]``
        index into an existing stack along the chain being addressed,
        starting with the root.  The final element may be any index
        which `list.insert()` accepts.  Negative final indices, and
        final indices greater than the number of layers in the addressed
        stack are quite valid in `path`::

        >>> from . import data
        >>> from . import test
        >>> stack, leaves = test.make_test_stack()
        >>> layer = data.PaintingLayer(name='foo')
        >>> stack.deepinsert((0,9999), layer)
        >>> stack.deepget((0,-1)) is layer
        True
        >>> layer = data.PaintingLayer(name='foo')
        >>> stack.deepinsert([0], layer)
        >>> stack.deepget([0]) is layer
        True

        Inserting a layer using this method gives it a unique name
        within the tree::

        >>> layer.name != 'foo'
        True
        """
        if len(path) == 0:
            raise IndexError('Cannot insert after the root')
        unused_path = list(path)
        stack = self
        while len(unused_path) > 0:
            idx = unused_path.pop(0)
            if not isinstance(stack, group.LayerStack):
                raise IndexError("All nonfinal elements of %r must "
                                 "identify a stack" % (path,))
            if unused_path:
                stack = stack[idx]
            else:
                stack.insert(idx, layer)
                layer.name = self.get_unique_name(layer)
                self._propagate_symmetry_state(layer)
                return
        assert (len(unused_path) > 0), ("deepinsert() should never "
                                        "exhaust the path")

    def deeppop(self, path):
        """Removes a layer by its path

        >>> from . import test
        >>> stack, leaves = test.make_test_stack()
        >>> stack.deeppop(())
        Traceback (most recent call last):
        ...
        IndexError: Cannot pop the root stack
        >>> stack.deeppop([0])
        <LayerStack len=3 '0'>
        >>> stack.deeppop((0,1))
        <PaintingLayer '11'>
        >>> stack.deeppop((0,2))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        IndexError: ...
        """
        if len(path) == 0:
            raise IndexError("Cannot pop the root stack")
        parent_path = path[:-1]
        child_index = path[-1]
        if len(parent_path) == 0:
            parent = self
        else:
            parent = self.deepget(parent_path)
        old_current = self.current_path
        removed = parent.pop(child_index)
        self.current_path = old_current  # i.e. nearest remaining
        return removed

    def deepremove(self, layer):
        """Removes a layer from any of the root's descendents

        >>> from . import test
        >>> stack, leaves = test.make_test_stack()
        >>> stack.deepremove(leaves[3])
        >>> stack.deepremove(leaves[2])
        >>> stack.deepremove(stack.deepget([0]))
        >>> stack
        <RootLayerStack len=1>
        >>> stack.deepremove(leaves[3])
        Traceback (most recent call last):
        ...
        ValueError: Layer is not in the root stack or any descendent
        """
        if layer is self:
            raise ValueError("Cannot remove the root stack")
        old_current = self.current_path
        for path, descendent_layer in self.walk():
            assert len(path) > 0
            if descendent_layer is not layer:
                continue
            parent_path = path[:-1]
            if len(parent_path) == 0:
                parent = self
            else:
                parent = self.deepget(parent_path)
            parent.remove(layer)
            self.current_path = old_current  # i.e. nearest remaining
            return None
        raise ValueError("Layer is not in the root stack or "
                         "any descendent")

    def deepindex(self, layer):
        """Return a path for a layer by searching the stack tree

        >>> from . import test
        >>> stack, leaves = test.make_test_stack()
        >>> stack.deepindex(stack)
        ()
        >>> [stack.deepindex(l) for l in leaves]
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        """
        if layer is self:
            return ()
        for path, ly in self.walk():
            if ly is layer:
                return tuple(path)
        return None

    ## Convenience methods for commands

    def canonpath(self, index=None, layer=None, path=None,
                  usecurrent=False, usefirst=False):
        """Verify and return the path for a layer from various criteria

        :param index: index of the layer in walk() order
        :param layer: a layer, which must be a descendent of this root
        :param path: a layer path
        :param usecurrent: if true, use the current path as fallback
        :param usefirst: if true, use the first path as fallback
        :return: a new, verified path referring to an existing layer
        :rtype: tuple

        The returned path is guaranteed to refer to an existing layer
        other than the root, and be the path in its most canonical
        form::

          >>> root = RootLayerStack(doc=None)
          >>> root.deepinsert([0], data.PaintingLayer())
          >>> root.deepinsert([1], group.LayerStack())
          >>> root.deepinsert([1, 0], data.PaintingLayer())
          >>> layer = data.PaintingLayer()
          >>> root.deepinsert([1, 1], layer)
          >>> root.deepinsert([1, 2], data.PaintingLayer())
          >>> root.canonpath(layer=layer)
          (1, 1)
          >>> root.canonpath(path=(-1, -2))
          (1, 1)
          >>> root.canonpath(index=3)
          (1, 1)

        Fallbacks can be specified for times when the regular criteria
        don't work::

          >>> root.current_path = (1, 1)
          >>> root.canonpath(usecurrent=True)
          (1, 1)
          >>> root.canonpath(usefirst=True)
          (0,)

        If no matching layer exists, a ValueError is raised::

          >>> root.clear()
          >>> root.canonpath(usecurrent=True)
          ... # doctest: +ELLIPSIS
          Traceback (most recent call last):
          ...
          ValueError: ...
          >>> root.canonpath(usefirst=True)
          ... # doctest: +ELLIPSIS
          Traceback (most recent call last):
          ...
          ValueError: ...
        """
        if path is not None:
            layer = self.deepget(path)
            if layer is self:
                raise ValueError("path=%r is root: must be descendent" %
                                 (path,))
            if layer is not None:
                path = self.deepindex(layer)
                assert self.deepget(path) is layer
                return path
            elif not usecurrent:
                raise ValueError("layer not found with path=%r" %
                                 (path,))
        elif index is not None:
            if index < 0:
                raise ValueError("negative layer index %r" % (index,))
            for i, (path, layer) in enumerate(self.walk()):
                if i == index:
                    assert self.deepget(path) is layer
                    return path
            if not usecurrent:
                raise ValueError("layer not found with index=%r" %
                                 (index,))
        elif layer is not None:
            if layer is self:
                raise ValueError("layer is root stack: must be "
                                 "descendent")
            path = self.deepindex(layer)
            if path is not None:
                assert self.deepget(path) is layer
                return path
            elif not usecurrent:
                raise ValueError("layer=%r not found" % (layer,))
        # Criterion failed. Try fallbacks.
        if usecurrent:
            path = self.get_current_path()
            layer = self.deepget(path)
            if layer is not None:
                if layer is self:
                    raise ValueError("The current layer path refers to "
                                     "the root stack")
                path = self.deepindex(layer)
                assert self.deepget(path) is layer
                return path
            if not usefirst:
                raise ValueError("Invalid current path; usefirst "
                                 "might work but not specified")
        if usefirst:
            if len(self) > 0:
                path = (0,)
                assert self.deepget(path) is not None
                return path
            else:
                raise ValueError("Invalid current path; stack is empty")
        raise TypeError("No layer/index/path criterion, and "
                        "no fallback criteria")

    ## Layer merging

    def layer_new_normalized(self, path):
        """Copy a layer to a normal painting layer that looks the same

        :param tuple path: Path to normalize
        :returns: New normalized layer
        :rtype: lib.layer.data.PaintingLayer

        The normalize operation does whatever is needed to convert a
        layer of any type into a normal painting layer with full opacity
        and Normal combining mode, while retaining its appearance at the
        current time. This may mean:

        * Just a simple copy
        * Merging all of its visible sublayers into the copy
        * Removing the effect the backdrop has on its appearance

        The returned painting layer is not inserted into the tree
        structure, and nothing in the tree structure is changed by this
        operation. The layer returned is always fully opaque, visible,
        and has normal mode. Its strokemap is constructed from all
        visible and tangible painting layers in the original, and it has
        the same name as the original, initially.

        >>> from . import test
        >>> root, leaves = test.make_test_stack()
        >>> orig_walk = list(root.walk())
        >>> orig_layers = {l for (p,l) in orig_walk}
        >>> for path, layer in orig_walk:
        ...     normized = root.layer_new_normalized(path)
        ...     assert normized not in orig_layers  # always a new layer
        >>> assert list(root.walk()) == orig_walk  # structure unchanged

        """
        srclayer = self.deepget(path)
        if not srclayer:
            raise ValueError("Path %r not found", path)

        # Simplest case
        if not srclayer.visible:
            return data.PaintingLayer(name=srclayer.name)

        if isinstance(srclayer, data.PaintingLayer):
            if srclayer.mode == lib.mypaintlib.CombineSpectralWGM:
                return deepcopy(srclayer)

        # Backdrops need removing if they combine with this layer's data.
        # Surface-backed layers' tiles can just be used as-is if they're
        # already fairly normal.
        needs_backdrop_removal = True
        if (srclayer.mode == lib.mypaintlib.CombineNormal
            and srclayer.opacity == 1.0):

            # Optimizations for the tiled-surface types
            if isinstance(srclayer, data.PaintingLayer):
                return deepcopy(srclayer)  # include strokes
            elif isinstance(srclayer, data.SurfaceBackedLayer):
                return data.PaintingLayer.new_from_surface_backed_layer(
                    srclayer
                )

            # Otherwise we're gonna have to render the source layer,
            # but we can skip the background removal *most* of the time.
            if isinstance(srclayer, group.LayerStack):
                needs_backdrop_removal = (srclayer.mode == PASS_THROUGH_MODE)
            else:
                needs_backdrop_removal = False
        # Begin building output, collecting tile indices and strokemaps.
        dstlayer = data.PaintingLayer()
        dstlayer.name = srclayer.name
        if srclayer.mode == lib.mypaintlib.CombineSpectralWGM:
            dstlayer.mode = srclayer.mode
        else:
            dstlayer.mode = lib.mypaintlib.CombineNormal
        tiles = set()
        for p, layer in self.walk():
            if not path_startswith(p, path):
                continue
            tiles.update(layer.get_tile_coords())
            if (isinstance(layer, data.PaintingLayer)
                    and not layer.locked
                    and not layer.branch_locked):
                dstlayer.strokes[:0] = layer.strokes

        # Might need to render the backdrop, in order to subtract it.
        bd_ops = []
        if needs_backdrop_removal:
            bd_spec = self._get_backdrop_render_spec_for_layer(path)
            bd_ops = self.get_render_ops(bd_spec)

        # Need to render the layer to be normalized too.
        # The ops are processed on top of the tiles bd_ops will render.
        src_spec = rendering.Spec(
            current=srclayer,
            solo=True,
            layers=set(self.layers_along_or_under_path(path))
        )
        src_ops = self.get_render_ops(src_spec)

        # Process by tile.
        # This is like taking before/after pics from a normal render(),
        # then subtracting the before from the after.
        logger.debug("Normalize: bd_ops = %r", bd_ops)
        logger.debug("Normalize: src_ops = %r", src_ops)
        dstsurf = dstlayer._surface
        tiledims = (tiledsurface.N, tiledsurface.N, 4)
        for tx, ty in tiles:
            bd = np.zeros(tiledims, dtype='uint16')
            with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                self._process_ops_list(bd_ops, bd, True, tx, ty, 0)
                lib.mypaintlib.tile_copy_rgba16_into_rgba16(bd, dst)
                self._process_ops_list(src_ops, dst, True, tx, ty, 0)
                if bd_ops:
                    dst[:, :, 3] = 0  # minimize alpha (discard original)
                    lib.mypaintlib.tile_flat2rgba(dst, bd)

        return dstlayer

    def get_merge_down_target(self, path):
        """Returns the target path for Merge Down, after checks

        :param tuple path: Source path for the Merge Down
        :returns: Target path for the merge, if it exists
        :rtype: tuple (or None)
        """
        if not path:
            return None

        source = self.deepget(path)
        if (source is None
                or source.locked
                or source.branch_locked
                or not source.get_mode_normalizable()):
            return None

        target_path = path[:-1] + (path[-1] + 1,)

        target = self.deepget(target_path)
        if (target is None
                or target.locked
                or target.branch_locked
                or not target.get_mode_normalizable()):
            return None

        return target_path

    def layer_new_merge_down(self, path):
        """Create a new layer containing the Merge Down of two layers

        :param tuple path: Path to the top layer to Merge Down
        :returns: New merged layer
        :rtype: lib.layer.data.PaintingLayer

        The current layer and the one below it are merged into a new
        layer, if that is possible, and the new layer is returned.
        Nothing is inserted or removed from the stack.  Any merged layer
        will contain a combined strokemap based on the input layers -
        although locked layers' strokemaps are not merged.

        You get what you see. This means that both layers must be
        visible to be used in the output.

        >>> from . import test
        >>> root, leaves = test.make_test_stack()
        >>> orig_walk = list(root.walk())
        >>> orig_layers = {l for (p,l) in orig_walk}
        >>> n_merged = 0
        >>> n_not_merged = 0
        >>> for path, layer in orig_walk:
        ...     try:
        ...         merged = root.layer_new_merge_down(path)
        ...     except ValueError:   # expect this
        ...         n_not_merged += 1
        ...         continue
        ...     assert merged not in orig_layers  # always a new layer
        ...     n_merged += 1
        >>> assert list(root.walk()) == orig_walk  # structure unchanged
        >>> assert n_merged > 0
        >>> assert n_not_merged > 0

        """
        target_path = self.get_merge_down_target(path)
        if not target_path:
            raise ValueError("Invalid path for Merge Down")
        # Normalize input
        merge_layers = []
        for p in [target_path, path]:
            assert p is not None
            layer = self.layer_new_normalized(p)
            merge_layers.append(layer)
        assert None not in merge_layers
        # Build output strokemap, determine set of data tiles to merge
        dstlayer = data.PaintingLayer()
        srclayer = self.deepget(path)
        if srclayer.mode == lib.mypaintlib.CombineSpectralWGM:
            dstlayer.mode = srclayer.mode 
        else:
            dstlayer.mode = lib.mypaintlib.CombineNormal
        tiles = set()
        for layer in merge_layers:
            tiles.update(layer.get_tile_coords())
            assert isinstance(layer, data.PaintingLayer)
            assert not layer.locked
            assert not layer.branch_locked
            dstlayer.strokes[:0] = layer.strokes
        # Build a (hopefully sensible) combined name too
        names = [l.name for l in reversed(merge_layers)
                 if l.has_interesting_name()]
        name = C_(
            "layer default names: joiner punctuation for merged layers",
            u", ",
        ).join(names)
        if name != '':
            dstlayer.name = name
        logger.debug("Merge Down: normalized source=%r", merge_layers)
        # Rendering loop
        dstsurf = dstlayer._surface
        for tx, ty in tiles:
            with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                for layer in merge_layers:
                    mode = layer.mode
                    if mode != lib.mypaintlib.CombineSpectralWGM:
                        mode = lib.mypaintlib.CombineNormal
                    layer._surface.composite_tile(
                        dst, True,
                        tx, ty, mipmap_level=0,
                        mode=mode, opacity=layer.opacity
                    )
        return dstlayer

    def layer_new_merge_visible(self):
        """Create and return the merge of all currently visible layers

        :returns: New merged layer
        :rtype: lib.layer.data.PaintingLayer

        All visible layers are merged into a new PaintingLayer, which is
        returned. Nothing is inserted or removed from the stack.  The
        merged layer will contain a combined strokemap based on those
        layers which are visible but not locked.

        You get what you see. If the background layer is visible at the
        time of the merge, then many modes will pick up an image of it.
        It will be "subtracted" from the result of the merge so that the
        merge result can be stacked above the same background.

        >>> from . import test
        >>> root, leaves = test.make_test_stack()
        >>> orig_walk = list(root.walk())
        >>> orig_layers = {l for (p,l) in orig_walk}
        >>> merged = root.layer_new_merge_visible()
        >>> assert list(root.walk()) == orig_walk  # structure unchanged
        >>> assert merged not in orig_layers   # layer is a new object

        See also: `walk()`, `background_visible`.
        """

        # Extract tile indices, names, and strokemaps.
        tiles = set()
        strokes = []
        names = []
        for path, layer in self.walk(visible=True):
            tiles.update(layer.get_tile_coords())
            if (isinstance(layer, data.StrokemappedPaintingLayer)
                    and not layer.locked
                    and not layer.branch_locked):
                strokes[:0] = layer.strokes
            if layer.has_interesting_name():
                names.append(layer.name)

        # Start making the output layer.
        dstlayer = data.PaintingLayer()
        dstlayer.mode = lib.mypaintlib.CombineNormal
        dstlayer.strokes = strokes
        name = C_(
            "layer default names: joiner punctuation for merged layers",
            u", ",
        ).join(names)
        if name != '':
            dstlayer.name = name
        dstsurf = dstlayer._surface

        # Render the entire tree, mostly normally.
        # Solo mode counts as normal, previewing mode does not.
        spec = self._get_render_spec(respect_previewing=False)
        self.render(dstsurf, tiles, 0, spec=spec)

        # Then subtract the background surface if it was rendered.
        # This leaves a ghost image.
        # Sure, we could render isolated for the case where all layers
        # that hit the background composite with src-over.
        # But that makes an exception and Exceptions Are Bad™.
        # Especially if they're really non-obvious to the user, like this.
        # Maybe it'd be better to split this op into two variants,
        # "Remove Background" and "Ignore Background"?
        if self._get_render_background(spec):
            bgsurf = self._background_layer._surface
            for tx, ty in tiles:
                with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                    with bgsurf.tile_request(tx, ty, readonly=True) as bg:
                        dst[:, :, 3] = 0  # minimize alpha (discard original)
                        lib.mypaintlib.tile_flat2rgba(dst, bg)

        return dstlayer

    ## Layer uniquifying (sort of the opposite of Merge Down)

    def uniq_layer(self, path, pixels=False):
        """Uniquify a painting layer's tiles or pixels."""
        targ_path = path
        targ_layer = self.deepget(path)

        if targ_layer is None:
            logger.error("uniq: target layer not found")
            return
        if not isinstance(targ_layer, data.PaintingLayer):
            logger.error("uniq: target layer is not a painting layer")
            return

        # Extract ops lists for the target and its backdrop
        bd_spec = self._get_backdrop_render_spec_for_layer(targ_path)
        bd_ops = self.get_render_ops(bd_spec)

        targ_only_spec = rendering.Spec(
            current=targ_layer,
            solo=True,
            layers=set(self.layers_along_or_under_path(targ_path))
        )
        targ_only_ops = self.get_render_ops(targ_only_spec)

        # Process by tile, like Normalize's backdrop removal.
        logger.debug("uniq: bd_ops = %r", bd_ops)
        logger.debug("uniq: targ_only_ops = %r", targ_only_ops)
        targ_surf = targ_layer._surface
        tile_dims = (tiledsurface.N, tiledsurface.N, 4)
        unchanged_tile_indices = set()
        zeros = np.zeros(tile_dims, dtype='uint16')
        for tx, ty in targ_surf.get_tiles():
            bd_img = copy(zeros)
            self._process_ops_list(bd_ops, bd_img, True, tx, ty, 0)
            targ_img = copy(bd_img)
            self._process_ops_list(targ_only_ops, targ_img, True, tx, ty, 0)
            equal_channels = (targ_img == bd_img)   # NxNn4 dtype=bool
            if equal_channels.all():
                unchanged_tile_indices.add((tx, ty))
            elif pixels:
                equal_px = equal_channels.all(axis=2, keepdims=True)  # NxNx1
                with targ_surf.tile_request(tx, ty, readonly=False) as targ:
                    targ *= np.invert(equal_px)

        targ_surf.remove_tiles(unchanged_tile_indices)

    def refactor_layer_group(self, path, pixels=False):
        """Factor common stuff out of a group's child layers."""
        targ_path = path
        targ_group = self.deepget(path)

        if targ_group is None:
            logger.error("refactor: target group not found")
            return
        if not isinstance(targ_group, group.LayerStack):
            logger.error("refactor: target group is not a LayerStack")
            return
        if targ_group.mode == PASS_THROUGH_MODE:
            logger.error("refactor: target group is not isolated")
            return
        if len(targ_group) == 0:
            return

        # Normalize each child layer that needs it.
        # Refactoring can cope with some opacity variations.
        for i, child in enumerate(targ_group):
            if child.mode == lib.mypaintlib.CombineNormal:
                continue
            child_path = tuple(list(targ_path) + [i])
            child = self.layer_new_normalized(child_path)
            targ_group[i] = child

        # Extract ops list fragments for the child layers.
        normalized_child_layers = list(targ_group)
        child_ops = {}
        union_tiles = set()
        for i, child in enumerate(normalized_child_layers):
            child_path = tuple(list(targ_path) + [i])
            spec = rendering.Spec(
                current=child,
                solo=True,
                layers=set(self.layers_along_or_under_path(child_path))
            )
            ops = self.get_render_ops(spec)
            child_ops[child] = ops
            union_tiles.update(child.get_tile_coords())

        # Insert a layer to contain all the common pixels or tiles
        common_layer = data.PaintingLayer()
        common_layer.mode = lib.mypaintlib.CombineNormal
        common_layer.name = C_(
            "layer default names: refactor: name of the common areas layer",
            u"Common",
        )
        common_surf = common_layer._surface
        targ_group.append(common_layer)

        # Process by tile
        n = tiledsurface.N
        zeros_rgba = np.zeros((n, n, 4), dtype='uint16')
        ones_bool = np.ones((n, n, 1), dtype='bool')
        common_data_tiles = set()
        child0 = normalized_child_layers[0]
        child0_surf = child0._surface
        for tx, ty in union_tiles:
            common_px = copy(ones_bool)
            rgba0 = None
            for child in normalized_child_layers:
                ops = child_ops[child]
                rgba = copy(zeros_rgba)
                self._process_ops_list(ops, rgba, True, tx, ty, 0)
                if rgba0 is None:
                    rgba0 = rgba
                else:
                    common_px &= (rgba0 == rgba).all(axis=2, keepdims=True)

            if common_px.all():
                with common_surf.tile_request(tx, ty, readonly=False) as d:
                    with child0_surf.tile_request(tx, ty, readonly=True) as s:
                        d[:] = s
                common_data_tiles.add((tx, ty))

            elif pixels and common_px.any():
                with common_surf.tile_request(tx, ty, readonly=False) as d:
                    with child0_surf.tile_request(tx, ty, readonly=True) as s:
                        d[:] = s * common_px
                for child in normalized_child_layers:
                    surf = child._surface
                    if (tx, ty) in surf.get_tiles():
                        with surf.tile_request(tx, ty, readonly=False) as d:
                            d *= np.invert(common_px)

        # Remove the remaining complete common tiles.
        for child in normalized_child_layers:
            surf = child._surface
            surf.remove_tiles(common_data_tiles)

    ## Loading

    def load_from_openraster(self, orazip, elem, cache_dir, progress,
                             x=0, y=0, **kwargs):
        """Load the root layer stack from an open .ora file

        >>> root = RootLayerStack(None)
        >>> import zipfile
        >>> import tempfile
        >>> import xml.etree.ElementTree as ET
        >>> import shutil
        >>> tmpdir = tempfile.mkdtemp()
        >>> assert os.path.exists(tmpdir)
        >>> with zipfile.ZipFile("tests/bigimage.ora") as orazip:
        ...     image_elem = ET.fromstring(orazip.read("stack.xml"))
        ...     stack_elem = image_elem.find("stack")
        ...     root.load_from_openraster(
        ...         orazip=orazip,
        ...         elem=stack_elem,
        ...         cache_dir=tmpdir,
        ...         progress=None,
        ...     )
        >>> len(list(root.walk())) > 0
        True
        >>> shutil.rmtree(tmpdir)
        >>> assert not os.path.exists(tmpdir)

        """
        self._no_background = True
        super(RootLayerStack, self).load_from_openraster(
            orazip,
            elem,
            cache_dir,
            progress,
            x=x, y=y,
            **kwargs
        )
        del self._no_background
        self._set_current_path_after_ora_load()
        self._mark_all_layers_for_rethumb()

    def _set_current_path_after_ora_load(self):
        """Set a suitable working layer after loading from oradir/orazip"""
        # Select a suitable working layer from the user-accesible ones.
        # Try for the uppermost layer marked as initially selected,
        # fall back to the uppermost immediate child of the root stack.
        num_loaded = 0
        selected_path = None
        uppermost_child_path = None
        for path, loaded_layer in self.walk():
            if not selected_path and loaded_layer.initially_selected:
                selected_path = path
            if not uppermost_child_path and len(path) == 1:
                uppermost_child_path = path
            num_loaded += 1
        logger.debug("Loaded %d layer(s)" % num_loaded)
        num_layers = num_loaded
        if num_loaded == 0:
            logger.error('Could not load any layer, document is empty.')
            if self.doc and self.doc.CREATE_PAINTING_LAYER_IF_EMPTY:
                logger.info('Adding an empty painting layer')
                self.ensure_populated()
                selected_path = [0]
                num_layers = len(self)
                assert num_layers > 0
            else:
                logger.warning("No layers, and doc debugging flag is active")
                return
        if not selected_path:
            selected_path = uppermost_child_path
        selected_path = tuple(selected_path)
        logger.debug("Selecting %r after load", selected_path)
        self.set_current_path(selected_path)

    def _load_child_layer_from_orazip(self, orazip, elem, cache_dir,
                                      progress, x=0, y=0, **kwargs):
        """Loads and appends a single child layer from an open .ora file"""
        attrs = elem.attrib
        # Handle MyPaint's special background tile notation
        # MyPaint will support reading .ora files using the legacy
        # background tile attribute until v2.0.0.
        bg_src_attrs = [
            data.BackgroundLayer.ORA_BGTILE_ATTR,
            data.BackgroundLayer.ORA_BGTILE_LEGACY_ATTR,
        ]
        for bg_src_attr in bg_src_attrs:
            bg_src = attrs.get(bg_src_attr, None)
            if not bg_src:
                continue
            logger.debug(
                "Found bg tile %r in %r",
                bg_src,
                bg_src_attr,
            )
            assert self._no_background, "Only one background is permitted"
            try:
                bg_pixbuf = lib.pixbuf.load_from_zipfile(
                    datazip=orazip,
                    filename=bg_src,
                    progress=progress,
                )
                self.set_background(bg_pixbuf)
                self._no_background = False
                return
            except tiledsurface.BackgroundError as e:
                logger.warning('ORA background tile not usable: %r', e)
        super(RootLayerStack, self)._load_child_layer_from_orazip(
            orazip,
            elem,
            cache_dir,
            progress,
            x=x, y=y,
            **kwargs
        )

    def load_from_openraster_dir(self, oradir, elem, cache_dir, progress,
                                 x=0, y=0, **kwargs):
        """Loads layer flags and data from an OpenRaster-style dir"""
        self._no_background = True
        super(RootLayerStack, self).load_from_openraster_dir(
            oradir,
            elem,
            cache_dir,
            progress,
            x=x, y=y,
            **kwargs
        )
        del self._no_background
        self._set_current_path_after_ora_load()
        self._mark_all_layers_for_rethumb()

    def _load_child_layer_from_oradir(self, oradir, elem, cache_dir,
                                      progress, x=0, y=0, **kwargs):
        """Loads and appends a single child layer from an open .ora file"""
        attrs = elem.attrib
        # Handle MyPaint's special background tile notation
        # MyPaint will support reading .ora files using the legacy
        # background tile attribute until v2.0.0.
        bg_src_attrs = [
            data.BackgroundLayer.ORA_BGTILE_ATTR,
            data.BackgroundLayer.ORA_BGTILE_LEGACY_ATTR,
        ]
        for bg_src_attr in bg_src_attrs:
            bg_src = attrs.get(bg_src_attr, None)
            if not bg_src:
                continue
            logger.debug(
                "Found bg tile %r in %r",
                bg_src,
                bg_src_attr,
            )
            assert self._no_background, "Only one background is permitted"
            try:
                bg_pixbuf = lib.pixbuf.load_from_file(
                    filename = os.path.join(oradir, bg_src),
                    progress = progress,
                )
                self.set_background(bg_pixbuf)
                self._no_background = False
                return
            except tiledsurface.BackgroundError as e:
                logger.warning('ORA background tile not usable: %r', e)
        super(RootLayerStack, self)._load_child_layer_from_oradir(
            oradir,
            elem,
            cache_dir,
            progress,
            x=x, y=y,
            **kwargs
        )

    ## Saving

    def save_to_openraster(self, orazip, tmpdir, path, canvas_bbox,
                           frame_bbox, progress=None, **kwargs):
        """Saves the stack's data into an open OpenRaster ZipFile"""
        if not progress:
            progress = lib.feedback.Progress()
        progress.items = 10

        # First 90%: save the stack contents normally.
        stack_elem = super(RootLayerStack, self).save_to_openraster(
            orazip, tmpdir, path, canvas_bbox,
            frame_bbox,
            progress=progress.open(9),
            **kwargs
        )

        # Remaining 10%: save the special background layer too.
        bg_layer = self.background_layer
        bg_layer.initially_selected = False
        bg_path = (len(self),)
        bg_elem = bg_layer.save_to_openraster(
            orazip, tmpdir, bg_path,
            canvas_bbox, frame_bbox,
            progress=progress.open(1),
            **kwargs
        )
        stack_elem.append(bg_elem)

        progress.close()
        return stack_elem

    def queue_autosave(self, oradir, taskproc, manifest, bbox, **kwargs):
        """Queues the layer for auto-saving"""
        stack_elem = super(RootLayerStack, self).queue_autosave(
            oradir, taskproc, manifest, bbox,
            **kwargs
        )
        # Queue background layer
        bg_layer = self.background_layer
        bg_elem = bg_layer.queue_autosave(
            oradir, taskproc, manifest, bbox,
            **kwargs
        )
        stack_elem.append(bg_elem)
        return stack_elem

    ## Notification mechanisms

    @event
    def layer_content_changed(self, *args):
        """Event: notifies that sub-layer's pixels have changed"""

    def _notify_layer_properties_changed(self, layer, changed):
        if layer is self:
            return
        assert layer.root is self
        path = self.deepindex(layer)
        assert path is not None, "Unable to find layer which was changed"
        self.layer_properties_changed(path, layer, changed)

    @event
    def layer_properties_changed(self, path, layer, changed):
        """Event: notifies that a sub-layer's properties have changed"""

    def _notify_layer_deleted(self, parent, oldchild, oldindex):
        assert parent.root is self
        assert oldchild.root is not self
        path = self.deepindex(parent)
        if path is None:  # e.g. layers within current_layer_overlay
            return
        path = path + (oldindex,)
        self.layer_deleted(path)

    @event
    def layer_deleted(self, path):
        """Event: notifies that a sub-layer has been deleted"""

    def _notify_layer_inserted(self, parent, newchild, newindex):
        assert parent.root is self
        assert newchild.root is self
        path = self.deepindex(newchild)
        if path is None:  # e.g. layers within current_layer_overlay
            return
        assert len(path) > 0
        self.layer_inserted(path)

    @event
    def layer_inserted(self, path):
        """Event: notifies that a sub-layer has been added"""
        pass

    @event
    def current_path_updated(self, path):
        """Event: notifies that the layer selection has been updated"""
        pass

    def save_snapshot(self):
        """Snapshots the state of the layer, for undo purposes"""
        return RootLayerStackSnapshot(self)

    ## Layer preview thumbnails

    def _mark_all_layers_for_rethumb(self):
        self._rethumb_layers[:] = []
        for path, layer in self.walk():
            self._rethumb_layers.append(layer)
        self._restart_rethumb_timer()

    def _mark_layer_for_rethumb(self, root, layer, *_ignored):
        if layer not in self._rethumb_layers:
            self._rethumb_layers.append(layer)
        self._restart_rethumb_timer()

    def _restart_rethumb_timer(self):
        timer_id = self._rethumb_layers_timer_id
        if timer_id is not None:
            GLib.source_remove(timer_id)
        timer_id = GLib.timeout_add(
            priority=GLib.PRIORITY_LOW,
            interval=100,
            function=self._rethumb_layers_timer_cb,
        )
        self._rethumb_layers_timer_id = timer_id

    def _rethumb_layers_timer_cb(self):
        if len(self._rethumb_layers) >= 1:
            layer0 = self._rethumb_layers.pop(-1)
            path0 = self.deepindex(layer0)
            if not path0:
                return True
            layer0.update_thumbnail()
            self.layer_thumbnail_updated(path0, layer0)
            # Queue parent layers too
            path = path0[:-1]
            parents = []
            while len(path) > 0:
                layer = self.deepget(path)
                if layer not in self._rethumb_layers:
                    parents.append(layer)
                path = path[:-1]
            self._rethumb_layers.extend(reversed(parents))
            return True
        # Stop the timer when there is nothing more to be done.
        self._rethumb_layers_timer_id = None
        return False

    @event
    def layer_thumbnail_updated(self, path, layer):
        """Event: a layer thumbnail was updated.

        :param tuple path: The path to _layer_.
        :param lib.layer.core.LayerBase layer: The layer that was updated.

        See lib.layer.core.LayerBase.thumbnail

        """
        pass


class RootLayerStackSnapshot (group.LayerStackSnapshot):
    """Snapshot of a root layer stack's state"""

    def __init__(self, layer):
        super(RootLayerStackSnapshot, self).__init__(layer)
        self.bg_sshot = layer.background_layer.save_snapshot()
        self.bg_visible = layer.background_visible
        self.current_path = layer.current_path

    def restore_to_layer(self, layer):
        super(RootLayerStackSnapshot, self).restore_to_layer(layer)
        layer.background_layer.load_snapshot(self.bg_sshot)
        layer.background_visible = self.bg_visible
        layer.current_path = self.current_path


class _TileRenderWrapper (TileAccessible, TileBlittable):
    """Adapts a RootLayerStack to support RO tile_request()s.

    The wrapping is very minimal.
    Tiles are rendered into empty buffers on demand and cached.
    The tile request interface is therefore read only,
    and these wrappers should be used only as temporary objects.

    """

    def __init__(self, root, spec, use_cache=True):
        """Adapt a renderable object to support "tile_request()".

        :param RootLayerStack root: root of a tree.
        :param lib.layer.rendering.Spec spec: How to render it.
        :param bool use_cache: Cache rendered output.

        """
        super(_TileRenderWrapper, self).__init__()
        self._root = root
        self._spec = spec
        self._ops = root.get_render_ops(spec)
        self._use_cache = bool(use_cache)
        self._cache = {}

        # Store the subset of layers that are visible, as a list.
        # If this is a solo layer, only filter from its sub-hierarchy.
        if spec.solo:
            self._visible_layers = spec.layers
        else:
            self._visible_layers = list(root.deepiter(visible=True))

    @contextlib.contextmanager
    def tile_request(self, tx, ty, readonly):
        """Context manager that fetches a single tile as fix15 RGBA data.

        :param int tx: Location to access (X coordinate).
        :param int ty: Location to access (Y coordinate).
        :param bool readonly: Must be True.
        :yields: One NumPy tile array.

        To be used with the 'with' statement.

        """
        if not readonly:
            raise ValueError("Only readonly tile requests are supported")
        dst = None
        if self._use_cache:
            dst = self._cache.get((tx, ty), None)
        if dst is None:
            bg_hidden = not self._root.root.background_visible
            if (self._spec.solo or bg_hidden) and self._all_empty(tx, ty):
                dst = tiledsurface.transparent_tile.rgba
            else:
                tiledims = (tiledsurface.N, tiledsurface.N, 4)
                dst = np.zeros(tiledims, 'uint16')
                self._root.render_single_tile(
                    dst, True,
                    tx, ty, 0,
                    ops=self._ops,
                )
            if self._use_cache:
                self._cache[(tx, ty)] = dst
        yield dst

    def _all_empty(self, tx, ty):
        """Check that no tile exists at (tx, ty) in any visible layer"""
        tc = (tx, ty)
        for layer in self._visible_layers:
            if tc in layer.get_tile_coords():
                return False
        return True

    def get_bbox(self):
        """Explicit passthrough of get_bbox"""
        return self._root.get_bbox()

    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, **kwargs):
        """Copy a rendered tile into a fix15 or 8bpp array."""
        assert dst.dtype == 'uint8'
        with self.tile_request(tx, ty, readonly=True) as src:
            assert src.dtype == 'uint16'
            if dst_has_alpha:
                conv = lib.mypaintlib.tile_convert_rgba16_to_rgba8
            else:
                conv = lib.mypaintlib.tile_convert_rgbu16_to_rgbu8
            conv(src, dst, eotf())

    def __getattr__(self, attr):
        """Pass through calls to other methods"""
        return getattr(self._root, attr)


## Layer path tuple functions


def path_startswith(path, prefix):
    """Returns whether one path starts with another

    :param tuple path: Path to be tested
    :param tuple prefix: Prefix path to be tested against

    >>> path_startswith((1,2,3), (1,2))
    True
    >>> path_startswith((1,2,3), (1,2,3,4))
    False
    >>> path_startswith((1,2,3), (1,0))
    False
    """
    if len(prefix) > len(path):
        return False
    for i in xrange(len(prefix)):
        if path[i] != prefix[i]:
            return False
    return True


## Module testing


def _test():
    """Run doctest strings"""
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _test()
