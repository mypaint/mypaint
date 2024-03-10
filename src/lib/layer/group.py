# This file is part of MyPaint.
# Copyright (C) 2011-2019 by the MyPaint Development Team
# Copyright (C) 2007-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layer group classes (stacks)"""


## Imports

from __future__ import division, print_function

import logging
from copy import copy

from lib.gettext import C_
import lib.mypaintlib
import lib.pixbufsurface
import lib.helpers as helpers
import lib.fileutils
from lib.modes import STANDARD_MODES
from lib.modes import STACK_MODES
from lib.modes import PASS_THROUGH_MODE
from . import core
from . import data
import lib.layer.error
import lib.surface
import lib.autosave
import lib.feedback
import lib.layer.core
from .rendering import Opcode
from lib.pycompat import unicode

logger = logging.getLogger(__name__)


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

    DEFAULT_NAME = C_(
        "layer default names",
        # TRANSLATORS: Short default name for layer groups.
        "Group",
    )

    TYPE_DESCRIPTION = C_(
        "layer type descriptions",
        u"Layer Group",
    )

    PERMITTED_MODES = set(STANDARD_MODES + STACK_MODES)

    ## Construction and other lifecycle stuff

    def __init__(self, **kwargs):
        """Initialize, with no sub-layers.

        Despite an empty layer stack having a zero length, it never
        tests as False under any circumstances. All layers and layer
        groups work this way.

        >>> g = LayerStack()
        >>> len(g)
        0
        >>> if not g:
        ...    raise ValueError("len=0 group tests as False, incorrectly")
        >>> bool(g)
        True

        """
        self._layers = []  # must be done before supercall
        super(LayerStack, self).__init__(**kwargs)

    def load_from_openraster(self, orazip, elem, cache_dir, progress,
                             x=0, y=0, **kwargs):
        """Load this layer from an open .ora file"""
        if elem.tag != "stack":
            raise lib.layer.error.LoadingFailed("<stack/> expected")

        if not progress:
            progress = lib.feedback.Progress()
        progress.items = 1 + len(list(elem.findall("./*")))

        # Item 1: supercall
        super(LayerStack, self).load_from_openraster(
            orazip,
            elem,
            cache_dir,
            progress.open(),
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
        # TODO: Check if this applies to CombineSpectralWGM as well
        is_pass_through = (self.mode == lib.mypaintlib.CombineNormal
                           and self.opacity == 1.0
                           and (isolated_flag.lower() == "auto"))
        if is_pass_through:
            self.mode = PASS_THROUGH_MODE

        # Items 2..n: child elements.
        # Document order is the same as _layers, bottom layer to top.
        for child_elem in elem.findall("./*"):
            assert child_elem is not elem
            self._load_child_layer_from_orazip(
                orazip,
                child_elem,
                cache_dir,
                progress.open(),
                x=x, y=y,
                **kwargs
            )
        progress.close()

    def _load_child_layer_from_orazip(self, orazip, elem, cache_dir,
                                      progress, x=0, y=0, **kwargs):
        """Loads a single child layer element from an open .ora file"""
        try:
            child = _layer_new_from_orazip(
                orazip,
                elem,
                cache_dir,
                progress,
                self.root,
                x=x, y=y,
                **kwargs
            )
        except lib.layer.error.LoadingFailed:
            logger.warning("Skipping non-loadable layer")
        else:
            self.append(child)

    def load_from_openraster_dir(self, oradir, elem, cache_dir, progress,
                                 x=0, y=0, **kwargs):
        """Loads layer flags and data from an OpenRaster-style dir"""
        if elem.tag != "stack":
            raise lib.layer.error.LoadingFailed("<stack/> expected")
        super(LayerStack, self).load_from_openraster_dir(
            oradir,
            elem,
            cache_dir,
            progress,
            x=x, y=y,
            **kwargs
        )
        self.clear()
        x += int(elem.attrib.get("x", 0))
        y += int(elem.attrib.get("y", 0))
        # Convert normal+nonisolated to the internal pass-thru mode
        isolated_flag = unicode(elem.attrib.get("isolation", "auto"))
        # TODO: Check if this applies to CombineSpectralWGM as well
        is_pass_through = (self.mode == lib.mypaintlib.CombineNormal
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
                progress,
                x=x, y=y,
                **kwargs
            )

    def _load_child_layer_from_oradir(self, oradir, elem, cache_dir,
                                      progress, x=0, y=0, **kwargs):
        """Loads a single child layer element from an open .ora file

        Child classes can override this, but otherwise it's an internal
        method.

        """
        try:
            child = _layer_new_from_oradir(
                oradir,
                elem,
                cache_dir,
                progress,
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
        orphan.group = None
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
        adoptee.group = self
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
        >>> from . import data
        >>> stack.append(data.PaintingLayer())
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
            result.expand_to_include_rect(layer.get_bbox())
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
            result.expand_to_include_rect(bbox)
        return result

    def is_empty(self):
        return len(self._layers) == 0

    @property
    def effective_opacity(self):
        """The opacity used when compositing a layer: zero if invisible"""
        if self.visible:
            return self.opacity
        else:
            return 0.0

    ## Renderable implementation

    def get_render_ops(self, spec):
        """Get rendering instructions."""

        # Defaults, which might be overridden
        visible = self.visible
        mode = self.mode
        opacity = self.opacity

        # Respect the explicit layers list.
        if spec.layers is not None:
            if self not in spec.layers:
                return []

        if spec.previewing:
            # Previewing mode is a quick flash of how the layer data
            # looks, unaffected by any modes or visibility settings.
            visible = True
            mode = PASS_THROUGH_MODE
            opacity = 1

        elif spec.solo:
            # Solo mode shows how the current layer looks by itself when
            # its visible flag is true, along with any of its child
            # layers.  Child layers use their natural visibility.
            # However solo layers are unaffected by any ancestor layers'
            # modes or visibilities.

            try:
                ancestor = not spec.__descendent_of_current
            except AttributeError:
                ancestor = True

            if self is spec.current:
                spec = copy(spec)
                spec.__descendent_of_current = True
                visible = True
            elif ancestor:
                mode = PASS_THROUGH_MODE
                opacity = 1.0
                visible = True

        if not visible:
            return []

        isolate_child_layers = (mode != PASS_THROUGH_MODE)

        ops = []
        if isolate_child_layers:
            ops.append((Opcode.PUSH, None, None, None))
        for child_layer in reversed(self._layers):
            ops.extend(child_layer.get_render_ops(spec))
        if isolate_child_layers:
            ops.append((Opcode.POP, None, mode, opacity))

        return ops

    ## Flood fill

    def flood_fill(self, fill_args, dst_layer=None):
        """Fills a point on the surface with a color (into other only!)

        See `PaintingLayer.flood_fill() for parameters and semantics. Layer
        stacks only support flood-filling into other layers because they are
        not surface backed.

        """
        assert dst_layer is not self
        assert dst_layer is not None

        root = self.root
        if root is None:
            raise ValueError(
                "Cannot flood_fill() into a layer group which is not "
                "a descendent of a RootLayerStack."
            )
        src = root.get_tile_accessible_layer_rendering(self)
        dst = dst_layer._surface
        return lib.tiledsurface.flood_fill(src, fill_args, dst)

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
        """Save to a named PNG file.

        For a layer stack (including the special root one), this works
        by rendering the stack and its contents in solo mode.

        """
        root = self.root
        if root is None:
            raise ValueError(
                "Cannot flood_fill() into a layer group which is not "
                "a descendent of a RootLayerStack."
            )
        root.render_layer_to_png_file(self, filename, bbox=rect, **kwargs)

    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, progress=None,
                           **kwargs):
        """Saves the stack's data into an open OpenRaster ZipFile"""

        if not progress:
            progress = lib.feedback.Progress()
        progress.items = 1 + len(self)

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
                                                  progress=progress.open(),
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

        progress += 1

        progress.close()

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
        self.layer_snaps = [l.save_snapshot() for l in layer]
        self.layer_classes = [l.__class__ for l in layer]

    def restore_to_layer(self, layer):
        super(LayerStackSnapshot, self).restore_to_layer(layer)
        layer.clear()
        for layer_class, snap in zip(self.layer_classes, self.layer_snaps):
            child = layer_class()
            child.load_snapshot(snap)
            layer.append(child)


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
        if len(self._moves) < 1:
            return False
        n = max(20, int(n // len(self._moves)))
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


def _layer_new_from_orazip(orazip, elem, cache_dir, progress,
                           root, x=0, y=0, **kwargs):
    """New layer from an OpenRaster zipfile (factory)"""
    for layer_class in _LAYER_LOADER_CLASS_ORDER:
        try:
            return layer_class.new_from_openraster(
                orazip,
                elem,
                cache_dir,
                progress,
                root,
                x=x, y=y,
                **kwargs
            )
        except lib.layer.error.LoadingFailed:
            pass
    raise lib.layer.error.LoadingFailed(
        "No delegate class willing to load %r" % (elem,)
    )


def _layer_new_from_oradir(oradir, elem, cache_dir, progress,
                           root, x=0, y=0, **kwargs):
    """New layer from a dir with an OpenRaster-like layout (factory)"""
    for layer_class in _LAYER_LOADER_CLASS_ORDER:
        try:
            return layer_class.new_from_openraster_dir(
                oradir,
                elem,
                cache_dir,
                progress,
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
