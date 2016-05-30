# This file is part of MyPaint.
# Copyright (C) 2011-2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Whole-tree-level layer classes and functions"""


## Imports
from __future__ import print_function

from gi.repository import GdkPixbuf

import re
import logging
logger = logging.getLogger(__name__)
from warnings import warn
from copy import deepcopy
import os.path

import numpy as np

from lib.gettext import C_
import lib.mypaintlib
import lib.tiledsurface as tiledsurface
import lib.helpers as helpers
from lib.observable import event
import lib.pixbuf
import lib.cache
from lib.modes import *
import data
import group


## Class defs


class PlaceholderLayer (group.LayerStack):
    """Trivial temporary placeholder layer, used for moves etc.

    The layer stack architecture does not allow for the same layer
    appearing twice in a tree structure. Layer operations therefore
    require unique placeholder layers occasionally, typically when
    swapping nodes in the tree or handling drags.
    """

    #TRANSLATORS: Short default name for temporary placeholder layers.
    #TRANSLATORS: (The user should never see this except in error cases)
    DEFAULT_NAME = C_(
        "layer default names",
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

    def __init__(self, doc=None, **kwargs):
        """Construct, as part of a model

        :param doc: The model document. May be None for testing.
        :type doc: lib.document.Document
        """
        super(RootLayerStack, self).__init__(**kwargs)
        self.doc = doc
        self._render_cache = lib.cache.LRUCache()
        # Background
        default_bg = (255, 255, 255)
        self._default_background = default_bg
        self._background_layer = data.BackgroundLayer(default_bg)
        self._background_visible = True
        # Symmetry
        self._symmetry_axis = None
        self._symmetry_active = False
        # Special rendering state
        self._current_layer_solo = False
        self._current_layer_previewing = False
        # Current layer
        self._current_path = ()
        # Self-observation
        self.layer_content_changed += self._clear_render_cache
        self.layer_properties_changed += self._clear_render_cache
        self.layer_deleted += self._clear_render_cache
        self.layer_inserted += self._clear_render_cache

    def _clear_render_cache(self, *_ignored):
        self._render_cache.clear()

    def clear(self):
        """Clear the layer and set the default background"""
        super(RootLayerStack, self).clear()
        self.set_background(self._default_background)
        self.current_path = ()
        self._clear_render_cache()

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

    def _get_render_background(self):
        """True if render_into should render the internal background

        :rtype: bool

        This reflects the background visibility flag normally,
        but the layer-previewing flag inverts its effect.
        This has the effect of making the current layer
        blink very appreciably when changing layers.

        See also: background_visible, current_layer_previewing.

        """
        if self._current_layer_previewing:
            return not self._background_visible
        else:
            return self._background_visible

    def get_render_is_opaque(self):
        """True if the rendering is known to be 100% opaque

        :rtype: bool

        The UI should draw its own checquered background if this is
        false, and expect `render_into()` to write RGBA data with lots
        of transparent areas.

        Even if the background is enabled, it may be knocked out by
        certain compositing modes of layers above onto it.
        """
        if not self._get_render_background():
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

    def render_into(self, surface, tiles, mipmap_level, overlay=None,
                    opaque_base_tile=None, filter=None):
        """Tiled rendering: used for display only

        :param surface: target rgba8 surface
        :type surface: lib.pixbufsurface.Surface
        :param tiles: tile coords, (tx, ty), to render
        :type tiles: list
        :param mipmap_level: layer and surface mipmap level to use
        :type mipmap_level: int
        :param overlay: overlay layer to render (stroke highlighting)
        :type overlay: SurfaceBackedLayer
        :param array opaque_base_tile: optional fallback base tile
        :param callable filter: display filter

        Rendering for the display may write non-opaque tiles
        to the target surface.
        This is determined by the combined effect of
        layer modes, rendering flags,
        and the background layer's visibility.
        See `get_render_is_opaque()` for an external test.
        Rendering non-opaque data is noticably slower in Cairo:
        see https://github.com/mypaint/mypaint/issues/21.

        As a workaround for the slowdown,
        an opaque base tile can be used as a fallback
        for all tiles rendered in the workflow,
        to be used when non-opaque rendering happens.
        This can contain an alpha check image,
        though the results won't look as nice as
        using a real background checquerboard pattern.
        Using the fallback guarantees that output is opaque,
        assuming it really does contain opaque RGBA data.

        * IN FLUX: the opaque base may change to a surface or a layer
        """
        # Decide a rendering mode
        render_background = self._get_render_background()
        dst_has_alpha = not self.get_render_is_opaque()
        layers = None
        if self._current_layer_previewing or self._current_layer_solo:
            path = self.get_current_path()
            layers = set(self.layers_along_path(path))
        previewing = None
        solo = None
        if self._current_layer_previewing:
            previewing = self.current
        if self._current_layer_solo:
            solo = self.current
        # Blit loop. Could this be done in C++?
        for tx, ty in tiles:
            with surface.tile_request(tx, ty, readonly=False) as dst:
                self.composite_tile(
                    dst, dst_has_alpha, tx, ty,
                    mipmap_level,
                    layers=layers,
                    render_background=render_background,
                    overlay=overlay,
                    previewing=previewing,
                    solo=solo,
                    opaque_base_tile=opaque_base_tile,
                )
                if filter:
                    filter(dst)

    def render_thumbnail(self, bbox, **options):
        """Renders a 256x256 thumbnail of the stack

        :param bbox: Bounding box to make a thumbnail of
        :type bbox: tuple
        :param **options: Passed to `render_as_pixbuf()`.
        :rtype: GtkPixbuf
        """
        x, y, w, h = bbox
        if w == 0 or h == 0:
            # workaround to save empty documents
            x, y, w, h = 0, 0, tiledsurface.N, tiledsurface.N
        mipmap_level = 0
        while (mipmap_level < tiledsurface.MAX_MIPMAP_LEVEL and
               max(w, h) >= 512):
            mipmap_level += 1
            x, y, w, h = x/2, y/2, w/2, h/2
        pixbuf = self.render_as_pixbuf(x, y, w, h,
                                       mipmap_level=mipmap_level,
                                       **options)
        assert pixbuf.get_width() == w and pixbuf.get_height() == h
        return helpers.scale_proportionally(pixbuf, 256, 256)

    ## Rendering: common layer API

    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       **kwargs):
        """Unconditionally copy one tile's data into an array

        The root layer stack implementation just uses `composite_tile()`
        due to its lack of conditionality.
        """
        self.composite_tile(dst, dst_has_alpha, tx, ty,
                            mipmap_level=mipmap_level, **kwargs)

    def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       layers=None, render_background=None, overlay=None,
                       opaque_base_tile=None,
                       **kwargs):
        """Composite a tile's data, respecting flags/layers list

        The root layer stack implementation accepts the parameters
        documented in `BaseLayer.composite_tile()`, and also consumes:

        :param bool render_background: Render the internal bg layer
        :param BaseLayer overlay: Overlay layer
        :param array opaque_base_tile: Fallback base tile

        The root layer has flags which ensure it is always visible, so the
        result is generally indistinguishable from `blit_tile_into()`.
        However the rendering loop, `render_into()`, calls this method
        an as

        The overlay layer is optional. If present, it is drawn on top.
        Overlay layers must support 15-bit scaled-int tile compositing.

        The base tile is used under the results of rendering, with the
        results drawn over it with simple alpha compositing.

        As a further extension to the base API, `dst` may be an 8bpp
        array. A temporary 15-bit scaled int array is used for
        compositing in this case, and the output is converted to 8bpp.
        """
        if render_background is None:
            render_background = self._get_render_background()
        if render_background:
            background_surface = self._background_layer._surface
        else:
            background_surface = self._blank_bg_surface
        assert dst.shape[-1] == 4

        N = tiledsurface.N

        cache_key = None
        cache_hit = False
        if dst.dtype == 'uint8':
            dst_8bit = dst
            dst = None
            using_cache = (
                layers is None
                and overlay is None
                and not (kwargs.get("solo") or kwargs.get("previewing"))
            )
            if using_cache:
                cache_key = (tx, ty, dst_has_alpha, mipmap_level,
                             render_background, id(opaque_base_tile))
                dst = self._render_cache.get(cache_key)
            if dst is None:
                dst = np.empty((N, N, 4), dtype='uint16')
            else:
                cache_hit = True
        else:
            dst_8bit = None

        if not cache_hit:
            dst_over_opaque_base = None
            if dst_has_alpha and opaque_base_tile is not None:
                dst_over_opaque_base = dst
                lib.mypaintlib.tile_copy_rgba16_into_rgba16(
                    opaque_base_tile,
                    dst_over_opaque_base,
                )
                dst = np.empty((N, N, 4), dtype='uint16')

            background_surface.blit_tile_into(dst, dst_has_alpha, tx, ty,
                                              mipmap_level)
            for layer in reversed(self):
                layer.composite_tile(dst, dst_has_alpha, tx, ty,
                                     mipmap_level, layers=layers, **kwargs)
            if overlay:
                overlay.composite_tile(dst, dst_has_alpha, tx, ty,
                                       mipmap_level, layers=set([overlay]),
                                       **kwargs)

            if dst_over_opaque_base is not None:
                dst_has_alpha = False
                lib.mypaintlib.tile_combine(
                    lib.mypaintlib.CombineNormal,
                    dst, dst_over_opaque_base,
                    dst_has_alpha, 1.0,
                )
                dst = dst_over_opaque_base

            if cache_key is not None:
                self._render_cache[cache_key] = dst

        if dst_8bit is not None:
            if dst_has_alpha:
                lib.mypaintlib.tile_convert_rgba16_to_rgba8(dst, dst_8bit)
            else:
                lib.mypaintlib.tile_convert_rgbu16_to_rgbu8(dst, dst_8bit)

    ## Symmetry axis

    @property
    def symmetry_active(self):
        """Whether symmetrical painting is active.

        This is a convenience property for part of
        the state managed by `set_symmetry_state()`.
        """
        return self._symmetry_active

    @symmetry_active.setter
    def symmetry_active(self, active):
        if self._symmetry_axis is None:
            raise ValueError(
                "UI code must set a non-Null symmetry_axis "
                "before activating symmetrical painting."
            )
        self.set_symmetry_state(active, self._symmetry_axis)

    @property
    def symmetry_axis(self):
        """The active painting symmetry X axis value

        The `symmetry_axis` property may be set to None.
        This indicates the initial state of a document when
        it has been newly created, or newly opened from a file.

        Setting the property to a value forces `symmetry_active` on,
        and setting it to ``None`` forces `symmetry_active` off.
        In both bases, only one `symmetry_state_changed` gets emitted.

        This is a convenience property for part of
        the state managed by `set_symmetry_state()`.
        """
        return self._symmetry_axis

    @symmetry_axis.setter
    def symmetry_axis(self, x):
        if x is None:
            self.set_symmetry_state(False, None)
        else:
            self.set_symmetry_state(True, x)

    def set_symmetry_state(self, active, center_x):
        """Set the central, propagated, symmetry axis and active flag.

        The root layer stack specialization manages a central state,
        which is propagated to the current layer automatically.

        See `LayerBase.set_symmetry_state` for the params.
        This override allows the shared `center_x` to be ``None``:
        see `symmetry_axis` for what that means.

        """
        active = bool(active)
        if center_x is not None:
            center_x = round(float(center_x))
        oldstate = (self._symmetry_active, self._symmetry_axis)
        newstate = (active, center_x)
        if oldstate == newstate:
            return
        self._symmetry_active = active
        self._symmetry_axis = center_x
        current = self.get_current()
        if current is not self:
            self._propagate_symmetry_state(current)
        self.symmetry_state_changed(active, center_x)

    def _propagate_symmetry_state(self, layer):
        """Copy the symmetry state to the a descendant layer"""
        assert layer is not self
        if self._symmetry_axis is None:
            return
        layer.set_symmetry_state(
            self._symmetry_active,
            self._symmetry_axis,
        )

    @event
    def symmetry_state_changed(self, active, x):
        """Event: symmetry axis was changed, or was toggled

        :param bool active: updated `symmetry_active` value
        :param int x: updated `symmetry_active` flag
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
        # If it has a nonblank name that's unique, that's fine
        existing = {d.name for d in self.deepiter()
                    if d is not layer and d.name is not None}
        blank = re.compile(r'^\s*$')
        newname = layer._name
        if newname is None or blank.match(newname):
            newname = layer.DEFAULT_NAME
        if newname not in existing:
            return newname
        # Map name prefixes to the max of their numeric suffixes
        existing_base2num = {}
        for existing_name in existing:
            match = layer.UNIQUE_NAME_REGEX.match(existing_name)
            if match is not None:
                base = unicode(match.group(1))
                num = int(match.group(2))
            else:
                base = unicode(existing_name)
                num = 0
            num = max(num, existing_base2num.get(base, 0))
            existing_base2num[base] = num
        # Construct a new unique name that fits the prefix/suffix req.
        match = layer.UNIQUE_NAME_REGEX.match(newname)
        if match is not None:
            base = unicode(match.group(1))
        else:
            base = unicode(newname)
        num = existing_base2num.get(base, 0) + 1
        newname = layer.UNIQUE_NAME_TEMPLATE % {
            "name": base,
            "number": num,
        }
        assert layer.UNIQUE_NAME_REGEX.match(newname)
        assert newname not in existing
        return newname

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
        for p, l in self.deepenumerate():
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
                index = min(len(parent), index+1)
                return parent_path + (index,)
        p_prev = None
        for p, l in self.deepenumerate():
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

            >>> import data
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
            >>> walk[1]
            ((1,), <LayerStack len=2 u'A'>)
            >>> walk[2]
            ((1, 0), <PaintingLayer u'B'>)

        The default behaviour is to return all layers.  If `visible`
        is true, hidden layers are excluded.  This excludes child layers
        of invisible layer stacks as well as the invisible stacks
        themselves.

            >>> root.deepget([0]).visible = False
            >>> root.deepget([1]).visible = False
            >>> list(root.walk(visible=True))
            [((2,), <PaintingLayer u'C'>)]

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

    def deepiter(self):
        """Iterates across all descendents of the stack

        >>> import test
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
        return (t[1] for t in self.walk())

    def deepenumerate(self):
        """Enumerates the structure of a stack, from top to bottom

        >>> import test
        >>> stack, leaves = test.make_test_stack()
        >>> [a[0] for a in stack.deepenumerate()]
        [(0,), (0, 0), (0, 1), (0, 2), (1,), (1, 0), (1, 1), (1, 2)]
        >>> set(leaves) - set([a[1] for a in stack.deepenumerate()])
        set([])

        This method is pending deprecation: it is the same as `walk()`
        with its default arguments::

        >>> list(stack.walk()) == list(stack.deepenumerate())
        True

        But `walk()` is more versatile and shorter to type out.
        """
        warn("walk() is more versatile, please use that instead",
             PendingDeprecationWarning, stacklevel=2)
        return self.walk()

    def deepget(self, path, default=None):
        """Gets a layer based on its path

        >>> import test
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
            if abs(idx) > len(layer)-1:
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

        >>> import test
        >>> from data import PaintingLayer
        >>> stack, leaves = test.make_test_stack()
        >>> layer = PaintingLayer(name='foo')
        >>> stack.deepinsert((0,9999), layer)
        >>> stack.deepget((0,-1)) is layer
        True
        >>> layer = PaintingLayer(name='foo')
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

        >>> import test
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

        >>> import test
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
        for path, descendent_layer in self.deepenumerate():
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

        >>> import test
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

        :param index: index of the layer in deepenumerate() order
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
            for i, (path, layer) in enumerate(self.deepenumerate()):
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

    def _get_backdrop(self, path):
        """Returns the backdrop layers underlying a path

        :param tuple path: The addressed path
        :returns: Its backdrop, as a list of layers
        :rtype: list

        These are the layers forming the isolated part of the backdop
        beneath the identified layer.  The returned list may start with
        the backdrop layer and contain others, or be empty.

            >>> root = RootLayerStack(doc=None)
            >>> for path, layer in [
            ...     ([0], data.PaintingLayer(name="notpart")),
            ...     ([1], group.LayerStack(name="ggp")),
            ...     ([1,0], group.LayerStack(name="gp")),
            ...     ([1,0,0], group.LayerStack(name="p")),
            ...     ([1,0,0,0], data.PaintingLayer(name="notpart")),
            ...     ([1,0,0,1], data.PaintingLayer(name="addressed")),
            ...     ([1,0,0,2], data.PaintingLayer(name="invis.:notpart")),
            ...     ([1,0,0,3], data.PaintingLayer(name="A")),
            ...     ([1,0,1], data.PaintingLayer(name="B")),
            ...     ([1,0,2], group.LayerStack(name="C")),
            ...     ([1,0,2,0], data.PaintingLayer(name="notpart")),
            ...     ([1,0,3], data.PaintingLayer(name="D")),
            ...     ([1,1], data.PaintingLayer(name="E")),
            ...     ([2], data.PaintingLayer(name="F")),
            ...     ]:
            ...     root.deepinsert(path, layer)
            >>> root.deepget([1,0,0,2]).visible = False
            >>> root.deepget([1,0,0]).mode = PASS_THROUGH_MODE
            >>> root.deepget([1,0]).mode = PASS_THROUGH_MODE
            >>> root.deepget([1]).mode = PASS_THROUGH_MODE

        If there are no isolated groups (other than the root stack
        itself), you'll get all the underlying layers including the
        internal `background_layer`, if that's currently visible:

            >>> path = [1,0,0,1]
            >>> [b.name for b in root._get_backdrop(path)]
            [u'background', u'F', u'E', u'D', u'C', u'B', u'A']

        The nearest isolated group to the addressed path which is not
        the addressed layer truncates the backdrop sequence:

            >>> [b.name for b in root._get_backdrop([1])]
            [u'background', u'F']
            >>> root.deepget([1]).mode = lib.mypaintlib.CombineNormal
            >>> [b.name for b in root._get_backdrop(path)]
            [u'E', u'D', u'C', u'B', u'A']
            >>> [b.name for b in root._get_backdrop([1])]
            [u'background', u'F']
            >>> root.deepget([1,0,0]).mode = lib.mypaintlib.CombineScreen
            >>> [b.name for b in root._get_backdrop(path)]
            [u'A']

        If the layer being addressed is the lowest one in an isolated
        group, its backdrop is blank:

            >>> [b.name for b in root._get_backdrop([1,0,0,3])]
            []

        Compositing the returned list in order over a zero-alpha
        starting point reproduces the pixels that the addressed layer
        would be composited over when rendering normally without any
        special layer visibility modes.
        """
        backdrop = []
        if self._background_visible:
            backdrop.append(self._background_layer)
        stack = self
        for i, idx in enumerate(path):
            if idx < 0:
                raise ValueError("Negative index in path %r" % (path,))
            underlying = []
            for layer in stack[idx+1:]:
                if layer.visible:
                    underlying.append(layer)
            backdrop.extend(reversed(underlying))
            stack = stack[idx]
            if i == len(path)-1:
                break
            if stack.mode != PASS_THROUGH_MODE:
                backdrop = []
        return backdrop

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

        >>> from lib.layer.test import make_test_stack
        >>> root, leaves = make_test_stack()
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
        # Simple case
        if not srclayer.visible:
            return data.PaintingLayer(name=srclayer.name)
        # Backdrops need removing if they combine with this layer's data.
        # Surface-backed layers' tiles can just be used as-is if they're
        # already fairly normal.
        needs_backdrop_removal = True
        backdrop_layers = []
        if srclayer.mode == DEFAULT_MODE and srclayer.opacity == 1.0:
            # Optimizations for the tiled-surface types
            if isinstance(srclayer, data.PaintingLayer):
                return deepcopy(srclayer)  # include strokes
            elif isinstance(srclayer, data.SurfaceBackedLayer):
                return data.PaintingLayer.new_from_surface_backed_layer(srclayer)
            # Otherwise we're gonna have to render, but we can skip the
            # background removal most of the time.
            if isinstance(srclayer, group.LayerStack):
                needs_backdrop_removal = (srclayer.mode == PASS_THROUGH_MODE)
            else:
                needs_backdrop_removal = False
        if needs_backdrop_removal:
            backdrop_layers = self._get_backdrop(path)
        # Begin building output, and enumerate set of tiles to render
        dstlayer = data.PaintingLayer()
        dstlayer.name = srclayer.name
        tiles = set()
        for p, layer in self.walk():
            if not path_startswith(p, path):
                continue
            tiles.update(layer.get_tile_coords())
            if isinstance(layer, data.PaintingLayer) and not layer.locked:
                dstlayer.strokes[:0] = layer.strokes
        # Render loop
        logger.debug("Normalize: render using backdrop %r", backdrop_layers)
        dstsurf = dstlayer._surface
        N = tiledsurface.N
        for tx, ty in tiles:
            bd = np.zeros((N, N, 4), dtype='uint16')
            for layer in backdrop_layers:
                if layer is self._background_layer:
                    surf = self._background_layer._surface
                    surf.blit_tile_into(bd, True, tx, ty, 0)
                    # FIXME: shouldn't need this special case
                else:
                    layer.composite_tile(bd, True, tx, ty, mipmap_level=0)
            with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                lib.mypaintlib.tile_copy_rgba16_into_rgba16(bd, dst)
                srclayer.composite_tile(dst, True, tx, ty, mipmap_level=0)
                if backdrop_layers:
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
        if not source:
            return None
        target_path = path[:-1] + (path[-1] + 1,)
        target = self.deepget(target_path)
        if not target:
            return None
        if not (source.get_mode_normalizable() and
                target.get_mode_normalizable()):
            return None
        if target.locked or source.locked:
            return None
        return target_path

    def layer_new_merge_down(self, path):
        """Create a new layer containg the Merge Down of two layers

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

        >>> from lib.layer.test import make_test_stack
        >>> root, leaves = make_test_stack()
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
        backdrop_layers = self._get_backdrop(target_path)
        # Normalize input
        merge_layers = []
        for p in [target_path, path]:
            assert p is not None
            layer = self.layer_new_normalized(p)
            merge_layers.append(layer)
        assert None not in merge_layers
        # Build output strokemap, determine set of data tiles to merge
        dstlayer = data.PaintingLayer()
        tiles = set()
        for layer in merge_layers:
            tiles.update(layer.get_tile_coords())
            assert isinstance(layer, data.PaintingLayer) and not layer.locked
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
        logger.debug("Merge Down: backdrop=%r", backdrop_layers)
        logger.debug("Merge Down: normalized source=%r", merge_layers)
        # Rendering loop
        dstsurf = dstlayer._surface
        for tx, ty in tiles:
            with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                for layer in merge_layers:
                    layer.composite_tile(dst, True, tx, ty, mipmap_level=0)
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

        >>> from lib.layer.test import make_test_stack
        >>> root, leaves = make_test_stack()
        >>> orig_walk = list(root.walk())
        >>> orig_layers = {l for (p,l) in orig_walk}
        >>> merged = root.layer_new_merge_visible()
        >>> assert list(root.walk()) == orig_walk  # structure unchanged
        >>> assert merged not in orig_layers   # layer is a new object

        See also: `walk()`, `background_visible`.
        """
        # What to render (+ strokemap)
        tiles = set()
        strokes = []
        names = []
        for path, layer in self.walk(visible=True):
            tiles.update(layer.get_tile_coords())
            if isinstance(layer, data.PaintingLayer) and not layer.locked:
                strokes[:0] = layer.strokes
            if layer.has_interesting_name():
                names.append(layer.name)
        dstlayer = data.PaintingLayer()
        dstlayer.strokes = strokes
        name = C_(
            "layer default names: joiner punctuation for merged layers",
            u", ",
        ).join(names)
        if name != '':
            dstlayer.name = name
        # Render & subtract backdrop (= the background, if visible)
        dstsurf = dstlayer._surface
        bgsurf = self._background_layer._surface
        for tx, ty in tiles:
            with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                self.composite_tile(
                    dst, True, tx, ty, mipmap_level=0,
                    render_background=self._background_visible
                )
                if self._background_visible:
                    with bgsurf.tile_request(tx, ty, readonly=True) as bg:
                        dst[:, :, 3] = 0  # minimize alpha (discard original)
                        lib.mypaintlib.tile_flat2rgba(dst, bg)
        return dstlayer

    ## Loading

    def load_from_openraster(self, orazip, elem, cache_dir, feedback_cb,
                             x=0, y=0, **kwargs):
        """Load the root layer stack from an open .ora file

        >>> root = RootLayerStack(None)
        >>> import zipfile
        >>> import tempfile
        >>> import xml.etree.ElementTree as ET
        >>> import shutil
        >>> tmpdir = tempfile.mkdtemp()
        >>> assert os.path.exists(tmpdir)
        >>> orazip = zipfile.ZipFile("tests/bigimage.ora")
        >>> image_elem = ET.fromstring(orazip.read("stack.xml"))
        >>> stack_elem = image_elem.find("stack")
        >>> root.load_from_openraster(
        ...    orazip=orazip,
        ...    elem=stack_elem,
        ...    cache_dir=tmpdir,
        ...    feedback_cb=None,
        ... )
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
            feedback_cb,
            x=x, y=y,
            **kwargs
        )
        del self._no_background
        self._set_current_path_after_ora_load()

    def _set_current_path_after_ora_load(self):
        """Set a suitable working layer after loading from oradir/orazip"""
        # Select a suitable working layer from the user-accesible ones.
        # Try for the uppermost layer marked as initially selected,
        # fall back to the uppermost immediate child of the root stack.
        num_loaded = 0
        selected_path = None
        uppermost_child_path = None
        for path, loaded_layer in self.deepenumerate():
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
                                      feedback_cb, x=0, y=0, **kwargs):
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
                    feedback_cb=feedback_cb,
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
            feedback_cb,
            x=x, y=y,
            **kwargs
        )

    def load_from_openraster_dir(self, oradir, elem, cache_dir, feedback_cb,
                                 x=0, y=0, **kwargs):
        """Loads layer flags and data from an OpenRaster-style dir"""
        self._no_background = True
        super(RootLayerStack, self).load_from_openraster_dir(
            oradir,
            elem,
            cache_dir,
            feedback_cb,
            x=x, y=y,
            **kwargs
        )
        del self._no_background
        self._set_current_path_after_ora_load()

    def _load_child_layer_from_oradir(self, oradir, elem, cache_dir,
                                      feedback_cb, x=0, y=0, **kwargs):
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
                    feedback_cb = feedback_cb,
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
            feedback_cb,
            x=x, y=y,
            **kwargs
        )

    ## Saving

    def save_to_openraster(self, orazip, tmpdir, path, canvas_bbox,
                           frame_bbox, **kwargs):
        """Saves the stack's data into an open OpenRaster ZipFile"""
        stack_elem = super(RootLayerStack, self).save_to_openraster(
            orazip, tmpdir, path, canvas_bbox,
            frame_bbox, **kwargs
        )
        # Save background
        bg_layer = self.background_layer
        bg_layer.initially_selected = False
        bg_path = (len(self),)
        bg_elem = bg_layer.save_to_openraster(
            orazip, tmpdir, bg_path,
            canvas_bbox, frame_bbox,
            **kwargs
        )
        stack_elem.append(bg_elem)
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
        assert path is not None, "Unable to find parent of deleted child"
        path = path + (oldindex,)
        self.layer_deleted(path)

    @event
    def layer_deleted(self, path):
        """Event: notifies that a sub-layer has been deleted"""

    def _notify_layer_inserted(self, parent, newchild, newindex):
        assert parent.root is self
        assert newchild.root is self
        path = self.deepindex(newchild)
        assert path is not None, "Unable to find child which was inserted"
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

