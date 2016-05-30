# This file is part of MyPaint.
# Copyright (C) 2011-2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
# Copyright (C) 2007-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layer group classes (stacks)"""


## Imports
from __future__ import print_function

import logging
logger = logging.getLogger(__name__)

import numpy as np

from lib.gettext import C_
import lib.mypaintlib
import lib.tiledsurface as tiledsurface
import lib.pixbufsurface
import lib.helpers as helpers
import lib.fileutils
from lib.modes import *
import core
import data
import lib.layer.error
import lib.surface
import lib.autosave


## Class defs

class LayerStack (core.LayerBase, lib.autosave.Autosaveable):
    """Ordered stack of layers, linear but nestable

    A stack's sub-layers are stored in the reverse order to that used by
    rendering: the first element in the sequence, index ``0``, is the
    topmost layer. As it happens, this makes both interoperability with
    GTK tree paths and loading from OpenRaster simpler: both use the
    same top-to-bottom order.

    Layer stacks support list-like access to their child layers.  Using
    the `insert()`, `pop()`, `remove()` methods or index-based access
    and assignment maintains the root stack reference sensibly (most of
    the time) when sublayers are added or deleted from a LayerStack
    which is part of a tree structure. Permuting the structure this way
    announces the changes to any registered observer methods for these
    events.

    """

    ## Class constants

    #TRANSLATORS: Short default name for layer groups.
    DEFAULT_NAME = C_(
        "layer default names",
        "Group",
    )

    PERMITTED_MODES = set(STANDARD_MODES + STACK_MODES)
    INITIAL_MODE = lib.mypaintlib.CombineNormal

    ## Construction and other lifecycle stuff

    def __init__(self, **kwargs):
        """Initialize, with no sub-layers"""
        self._layers = []  # must be done before supercall
        super(LayerStack, self).__init__(**kwargs)
        # Blank background, for use in rendering
        N = tiledsurface.N
        blank_arr = np.zeros((N, N, 4), dtype='uint16')
        self._blank_bg_surface = tiledsurface.Background(blank_arr)

    def load_from_openraster(self, orazip, elem, cache_dir, feedback_cb,
                             x=0, y=0, **kwargs):
        """Load this layer from an open .ora file"""
        if elem.tag != "stack":
            raise lib.layer.error.LoadingFailed("<stack/> expected")
        super(LayerStack, self).load_from_openraster(
            orazip,
            elem,
            cache_dir,
            feedback_cb,
            x=x, y=y,
            **kwargs
        )
        self.clear()
        x += int(elem.attrib.get("x", 0))
        y += int(elem.attrib.get("y", 0))

        # The only combination which can result in a non-isolated mode
        # under the OpenRaster and W3C definition. Represented
        # internally with a special mode to make the UI prettier.
        isolated_flag = unicode(elem.attrib.get("isolation", "auto"))
        is_pass_through = (self.mode == DEFAULT_MODE
                           and self.opacity == 1.0
                           and (isolated_flag.lower() == "auto"))
        if is_pass_through:
            self.mode = PASS_THROUGH_MODE

        # Document order is the same as _layers, bottom layer to top.
        for child_elem in elem.findall("./*"):
            assert child_elem is not elem
            self._load_child_layer_from_orazip(
                orazip,
                child_elem,
                cache_dir,
                feedback_cb,
                x=x, y=y,
                **kwargs
            )

    def _load_child_layer_from_orazip(self, orazip, elem, cache_dir,
                                      feedback_cb, x=0, y=0, **kwargs):
        """Loads a single child layer element from an open .ora file"""
        try:
            child = _layer_new_from_orazip(
                orazip,
                elem,
                cache_dir,
                feedback_cb,
                self.root,
                x=x, y=y,
                **kwargs
            )
        except lib.layer.error.LoadingFailed:
            logger.warning("Skipping non-loadable layer")
        else:
            self.append(child)

    def load_from_openraster_dir(self, oradir, elem, cache_dir, feedback_cb,
                                 x=0, y=0, **kwargs):
        """Loads layer flags and data from an OpenRaster-style dir"""
        if elem.tag != "stack":
            raise lib.layer.error.LoadingFailed("<stack/> expected")
        super(LayerStack, self).load_from_openraster_dir(
            oradir,
            elem,
            cache_dir,
            feedback_cb,
            x=x, y=y,
            **kwargs
        )
        self.clear()
        x += int(elem.attrib.get("x", 0))
        y += int(elem.attrib.get("y", 0))
        # Convert normal+nonisolated to the internal pass-thru mode
        isolated_flag = unicode(elem.attrib.get("isolation", "auto"))
        is_pass_through = (self.mode == DEFAULT_MODE
                           and self.opacity == 1.0
                           and (isolated_flag.lower() == "auto"))
        if is_pass_through:
            self.mode = PASS_THROUGH_MODE
        # Delegate loading of child layers
        for child_elem in elem.findall("./*"):
            assert child_elem is not elem
            self._load_child_layer_from_oradir(
                oradir,
                child_elem,
                cache_dir,
                feedback_cb,
                x=x, y=y,
                **kwargs
            )

    def _load_child_layer_from_oradir(self, oradir, elem, cache_dir,
                                      feedback_cb, x=0, y=0, **kwargs):
        """Loads a single child layer element from an open .ora file

        Child classes can override this, but otherwise it's an internal
        method.

        """
        try:
            child = _layer_new_from_oradir(
                oradir,
                elem,
                cache_dir,
                feedback_cb,
                self.root,
                x=x, y=y,
                **kwargs
            )
        except lib.layer.error.LoadingFailed:
            logger.warning("Skipping non-loadable layer")
        else:
            self.append(child)

    def clear(self):
        """Clears the layer, and removes any child layers"""
        super(LayerStack, self).clear()
        removed = list(self._layers)
        self._layers[:] = []
        for i, layer in reversed(list(enumerate(removed))):
            self._notify_disown(layer, i)

    def __repr__(self):
        """String representation of a stack

        >>> repr(LayerStack(name='test'))
        "<LayerStack len=0 'test'>"
        """
        if self.name:
            return '<%s len=%d %r>' % (self.__class__.__name__, len(self),
                                       self.name)
        else:
            return '<%s len=%d>' % (self.__class__.__name__, len(self))

    ## Notification

    def _notify_disown(self, orphan, oldindex):
        """Recursively process a removed child (root reset, notify)"""
        # Reset root and notify. No actual tree permutations.
        root = self.root
        orphan.root = None
        # Recursively disown all descendents of the orphan first
        if isinstance(orphan, LayerStack):
            for i, child in reversed(list(enumerate(orphan))):
                orphan._notify_disown(child, i)
        # Then notify, now all descendents are gone
        if root is not None:
            root._notify_layer_deleted(self, orphan, oldindex)

    def _notify_adopt(self, adoptee, newindex):
        """Recursively process an added child (set root, notify)"""
        # Set root and notify. No actual tree permutations.
        root = self.root
        adoptee.root = root
        # Notify for the newly adopted layer first
        if root is not None:
            root._notify_layer_inserted(self, adoptee, newindex)
        # Recursively adopt all descendents of the adoptee after
        if isinstance(adoptee, LayerStack):
            for i, child in enumerate(adoptee):
                adoptee._notify_adopt(child, i)

    ## Basic list-of-layers access

    def __len__(self):
        """Return the number of layers in the stack

        >>> stack = LayerStack()
        >>> len(stack)
        0
        >>> stack.append(core.LayerBase())
        >>> len(stack)
        1
        """
        return len(self._layers)

    def __iter__(self):
        """Iterates across child layers in the order used by append()"""
        return iter(self._layers)

    def append(self, layer):
        """Appends a layer (notifies root)"""
        newindex = len(self)
        self._layers.append(layer)
        self._notify_adopt(layer, newindex)
        self._content_changed(*layer.get_full_redraw_bbox())

    def remove(self, layer):
        """Removes a layer by equality (notifies root)"""
        oldindex = self._layers.index(layer)
        assert oldindex is not None
        removed = self._layers.pop(oldindex)
        assert removed is not None
        self._notify_disown(removed, oldindex)
        self._content_changed(*removed.get_full_redraw_bbox())

    def pop(self, index=None):
        """Removes a layer by index (notifies root)"""
        if index is None:
            index = len(self._layers)-1
            removed = self._layers.pop()
        else:
            index = self._normidx(index)
            removed = self._layers.pop(index)
        self._notify_disown(removed, index)
        self._content_changed(*removed.get_full_redraw_bbox())
        return removed

    def _normidx(self, i, insert=False):
        """Normalize an index for array-like access

        >>> group = LayerStack()
        >>> group.append(data.PaintingLayer())
        >>> group.append(data.PaintingLayer())
        >>> group.append(data.PaintingLayer())
        >>> group._normidx(-4, insert=True)
        0
        >>> group._normidx(1)
        1
        >>> group._normidx(999)
        999
        >>> group._normidx(999, insert=True)
        3
        """
        if i < 0:
            i = len(self) + i
        if insert:
            return max(0, min(len(self), i))
        return i

    def insert(self, index, layer):
        """Adds a layer before an index (notifies root)"""
        index = self._normidx(index, insert=True)
        self._layers.insert(index, layer)
        self._notify_adopt(layer, index)
        self._content_changed(*layer.get_full_redraw_bbox())

    def __setitem__(self, index, layer):
        """Replaces the layer at an index (notifies root)"""
        index = self._normidx(index)
        oldlayer = self._layers[index]
        self._layers[index] = layer
        self._notify_disown(oldlayer, index)
        updates = [oldlayer.get_full_redraw_bbox()]
        self._notify_adopt(layer, index)
        updates.append(layer.get_full_redraw_bbox())
        self._content_changed(*tuple(core.combine_redraws(updates)))

    def __getitem__(self, index):
        """Fetches the layer at an index"""
        return self._layers[index]

    def index(self, layer):
        """Fetches the index of a child layer, by equality"""
        return self._layers.index(layer)

    ## Info methods

    def get_bbox(self):
        """Returns the inherent (data) bounding box of the stack"""
        result = helpers.Rect()
        for layer in self._layers:
            result.expandToIncludeRect(layer.get_bbox())
        return result

    def get_full_redraw_bbox(self):
        """Returns the full update notification bounding box of the stack"""
        result = super(LayerStack, self).get_full_redraw_bbox()
        if result.w == 0 or result.h == 0:
            return result
        for layer in self._layers:
            bbox = layer.get_full_redraw_bbox()
            if bbox.w == 0 or bbox.h == 0:
                return bbox
            result.expandToIncludeRect(bbox)
        return result

    def is_empty(self):
        return len(self._layers) == 0

    @property
    def effective_opacity(self):
        """The opacity used when compositing a layer: zero if invisible"""
        # Mirror what composite_tile does.
        if self.visible:
            return self.opacity
        else:
            return 0.0

    ## Rendering

    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       **kwargs):
        """Unconditionally copy one tile's data into an array"""
        N = tiledsurface.N
        tmp = np.zeros((N, N, 4), dtype='uint16')
        for layer in reversed(self._layers):
            layer.composite_tile(tmp, True, tx, ty, mipmap_level,
                                 layers=None, **kwargs)
        if dst.dtype == 'uint16':
            lib.mypaintlib.tile_copy_rgba16_into_rgba16(tmp, dst)
        elif dst.dtype == 'uint8':
            if dst_has_alpha:
                lib.mypaintlib.tile_convert_rgba16_to_rgba8(tmp, dst)
            else:
                lib.mypaintlib.tile_convert_rgbu16_to_rgbu8(tmp, dst)
        else:
            raise ValueError('Unsupported destination buffer type %r' %
                             dst.dtype)

    def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       layers=None, previewing=None, solo=None, **kwargs):
        """Composite a tile's data into an array, respecting flags/layers list"""

        mode = self.mode
        opacity = self.opacity
        if layers is not None:
            if self not in layers:
                return
            # If this is the layer to be previewed, show all child layers
            # as the layer data.
            if self in (previewing, solo):
                layers.update(self._layers)
        elif not self.visible:
            return

        # Render each child layer in turn
        isolate = (self.mode != PASS_THROUGH_MODE)
        if isolate and previewing and self is not previewing:
            isolate = False
        if isolate and solo and self is not solo:
            isolate = False
        if isolate:
            N = tiledsurface.N
            tmp = np.zeros((N, N, 4), dtype='uint16')
            for layer in reversed(self._layers):
                p = (self is previewing) and layer or previewing
                s = (self is solo) and layer or solo
                layer.composite_tile(tmp, True, tx, ty, mipmap_level,
                                     layers=layers, previewing=p, solo=s,
                                     **kwargs)
            if previewing or solo:
                mode = DEFAULT_MODE
                opacity = 1.0
            lib.mypaintlib.tile_combine(
                mode, tmp,
                dst, dst_has_alpha,
                opacity,
            )
        else:
            for layer in reversed(self._layers):
                p = (self is previewing) and layer or previewing
                s = (self is solo) and layer or solo
                layer.composite_tile(dst, dst_has_alpha, tx, ty, mipmap_level,
                                     layers=layers, previewing=p, solo=s,
                                     **kwargs)

    def render_as_pixbuf(self, *args, **kwargs):
        return lib.pixbufsurface.render_as_pixbuf(self, *args, **kwargs)

    ## Flood fill

    def flood_fill(self, x, y, color, bbox, tolerance, dst_layer=None):
        """Fills a point on the surface with a color (into other only!)

        See `PaintingLayer.flood_fill() for parameters and semantics. Layer
        stacks only support flood-filling into other layers because they are
        not surface backed.
        """
        assert dst_layer is not self
        assert dst_layer is not None
        src = lib.surface.TileRequestWrapper(self)
        dst = dst_layer._surface
        tiledsurface.flood_fill(src, x, y, color, bbox, tolerance, dst)

    def get_fillable(self):
        """False! Stacks can't be filled interactively or directly."""
        return False

    ## Moving

    def get_move(self, x, y):
        """Get a translation/move object for this layer"""
        return LayerStackMove(self, x, y)

    ## Saving

    @lib.fileutils.via_tempfile
    def save_as_png(self, filename, *rect, **kwargs):
        """Save to a named PNG file"""
        if 'alpha' not in kwargs:
            kwargs['alpha'] = True
        lib.surface.save_as_png(self, filename, *rect, **kwargs)

    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        """Saves the stack's data into an open OpenRaster ZipFile"""

        # MyPaint uses the same origin internally for all data layers,
        # meaning the internal stack objects don't impose any offsets on
        # their children. Any x or y attrs which were present when the
        # stack was loaded from .ORA were accounted for back then.
        stack_elem = self._get_stackxml_element("stack")

        # Recursively save out the stack's child layers
        for layer_idx, layer in list(enumerate(self)):
            layer_path = tuple(list(path) + [layer_idx])
            layer_elem = layer.save_to_openraster(orazip, tmpdir, layer_path,
                                                  canvas_bbox, frame_bbox,
                                                  **kwargs)
            stack_elem.append(layer_elem)

        # OpenRaster has no pass-through composite op: need to override.
        # MyPaint's "Pass-through" mode is internal shorthand for the
        # default behaviour of OpenRaster.
        isolation = "isolate"
        if self.mode == PASS_THROUGH_MODE:
            stack_elem.attrib.pop("opacity", None)  # => 1.0
            stack_elem.attrib.pop("composite-op", None)  # => svg:src-over
            isolation = "auto"
        stack_elem.attrib["isolation"] = isolation

        return stack_elem

    def queue_autosave(self, oradir, taskproc, manifest, bbox, **kwargs):
        """Queues the layer for auto-saving"""
        # Build a layers.xml element: no x or y for stacks
        stack_elem = self._get_stackxml_element("stack")
        for layer in self._layers:
            layer_elem = layer.queue_autosave(
                oradir, taskproc, manifest, bbox,
                **kwargs
            )
            stack_elem.append(layer_elem)
        # Convert the internal pass-through composite op to its
        # OpenRaster equivalent: the default, non-isolated src-over.
        isolation = "isolate"
        if self.mode == PASS_THROUGH_MODE:
            stack_elem.attrib.pop("opacity", None)  # => 1.0
            stack_elem.attrib.pop("composite-op", None)  # => svg:src-over
            isolation = "auto"
        stack_elem.attrib["isolation"] = isolation
        return stack_elem

    ## Snapshotting

    def save_snapshot(self):
        """Snapshots the state of the layer, for undo purposes"""
        return LayerStackSnapshot(self)

    ## Trimming

    def trim(self, rect):
        """Trim the layer to a rectangle, discarding data outside it"""
        for layer in self:
            layer.trim(rect)

    ## Type-specific action

    def activate_layertype_action(self):
        root = self.root
        if root is None:
            return
        path = root.deepindex(self)
        if path and len(path) > 0:
            root.expand_layer(path)

    def get_icon_name(self):
        return "mypaint-layer-group-symbolic"


class LayerStackSnapshot (core.LayerBaseSnapshot):
    """Snapshot of a layer stack's state"""

    def __init__(self, layer):
        super(LayerStackSnapshot, self).__init__(layer)
        self.layer_snaps = [l.save_snapshot() for l in layer._layers]
        self.layer_classes = [l.__class__ for l in layer._layers]

    def restore_to_layer(self, layer):
        super(LayerStackSnapshot, self).restore_to_layer(layer)
        layer._layers = []
        for layer_class, snap in zip(self.layer_classes,
                                     self.layer_snaps):
            child = layer_class()
            child.load_snapshot(snap)
            layer._layers.append(child)


class LayerStackMove (object):
    """Move object wrapper for layer stacks"""

    def __init__(self, layers, x, y):
        super(LayerStackMove, self).__init__()
        self._moves = []
        for layer in layers:
            self._moves.append(layer.get_move(x, y))

    def update(self, dx, dy):
        for move in self._moves:
            move.update(dx, dy)

    def cleanup(self):
        for move in self._moves:
            move.cleanup()

    def process(self, n=200):
        n = max(20, int(n / len(self._moves)))
        incomplete = False
        for move in self._moves:
            incomplete = move.process(n=n) or incomplete
        return incomplete


## Layer factory func

_LAYER_LOADER_CLASS_ORDER = [
    LayerStack,
    data.PaintingLayer,
    data.VectorLayer,
    data.FallbackBitmapLayer,
    data.FallbackDataLayer,
]


def _layer_new_from_orazip(orazip, elem, cache_dir, feedback_cb,
                           root, x=0, y=0, **kwargs):
    """New layer from an OpenRaster zipfile (factory)"""
    for layer_class in _LAYER_LOADER_CLASS_ORDER:
        try:
            return layer_class.new_from_openraster(
                orazip,
                elem,
                cache_dir,
                feedback_cb,
                root,
                x=x, y=y,
                **kwargs
            )
        except lib.layer.error.LoadingFailed:
            pass
    raise lib.layer.error.LoadingFailed(
        "No delegate class willing to load %r" % (elem,)
    )


def _layer_new_from_oradir(oradir, elem, cache_dir, feedback_cb,
                           root, x=0, y=0, **kwargs):
    """New layer from a dir with an OpenRaster-like layout (factory)"""
    for layer_class in _LAYER_LOADER_CLASS_ORDER:
        try:
            return layer_class.new_from_openraster_dir(
                oradir,
                elem,
                cache_dir,
                feedback_cb,
                root,
                x=x, y=y,
                **kwargs
            )
        except lib.layer.error.LoadingFailed:
            pass
    raise lib.layer.error.LoadingFailed(
        "No delegate class willing to load %r" % (elem,)
    )


## Module testing


def _test():
    """Run doctest strings"""
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _test()
