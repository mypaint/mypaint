# This file is part of MyPaint.
# Copyright (C) 2007-2008 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Layers holding graphical data or other layers

Users will normally interact with `PaintingLayer`s, which contain pixel
data and expose drawing commands. Other types of data layer may be added
in future.

Layers are arranged in a tree structure consisting of ordered stacks,
growing from a single root stack belonging to the document model. The
data layers form the leaves. This tree must only contain a single
instance of any layer at a given time, although this is not enforced.
Layer (sub-)stacks are also layers in every sense, and are subject to
the same constraints.

Layers emit a moderately fine-grained set of notifications whenever the
structure changes or the user draws something. These can be listened to
via the root layer stack.
"""

## Imports

import gi
from gi.repository import GdkPixbuf

import re
import struct
import zlib
import numpy
from numpy import *
import logging
import os
from cStringIO import StringIO
import time
import zipfile
logger = logging.getLogger(__name__)
import tempfile
import shutil
import xml.etree.ElementTree as ET
import weakref
from warnings import warn
from copy import deepcopy

from gettext import gettext as _

import tiledsurface
import pixbufsurface
import strokemap
import mypaintlib
import helpers
from observable import event

from tiledsurface import OPENRASTER_COMBINE_MODES
from tiledsurface import DEFAULT_COMBINE_MODE
from tiledsurface import COMBINE_MODE_STRINGS


## Module constants

LOAD_CHUNK_SIZE = 64*1024


#: Layer modes which can lower the alpha of their backdrop
MODES_DECREASING_BACKDROP_ALPHA = {
    m for m in range(mypaintlib.NumCombineModes)
    if mypaintlib.combine_mode_get_info(m).get("can_decrease_alpha")}

#: Layer modes which, even with alpha==0, can alter their backdrops
MODES_EFFECTIVE_AT_ZERO_ALPHA = {
    m for m in range(mypaintlib.NumCombineModes)
    if mypaintlib.combine_mode_get_info(m).get("zero_alpha_has_effect")}


## Class defs

## Basic interface for a renderable layer & docs


class LayerBase (object):
    """Base class defining the layer API

    Layers support two similar tile-based methods which are used for two
    distinct rendering cases: blitting and compositing.  "Blitting" is
    an unconditional copying of the layer's content into a tile buffer
    without any consideration of the layer's own rendering flags.
    "Compositing" is a conditional alpha-compositing which respects the
    layer's own flags like opacity and layer mode.  Aggregated rendering
    for the display is supported using the compositing pathway and is
    coordinated via the `RootLayerStack`.  Exporting individual layers
    is handled via the blitting pathway, which for layer stacks involves
    compositing the stacks' contents together to render an effective
    image.

    Layers are minimally aware of the tree structure they reside in in
    that they contain a reference to the root of their tree for
    signalling purposes.  Updates to the tree structure and to layers'
    graphical contents are announced via the `RootLayerStack` object
    representing the base of the tree.
    """

    ## Class constants

    #TRANSLATORS: Default name for new (base class) layers
    DEFAULT_NAME = _(u"Layer")

    #TRANSLATORS: Template for creating unique names
    UNIQUE_NAME_TEMPLATE = _(u'%(name)s %(number)d')

    #TRANSLATORS: Regex matching suffix numbers in assigned unique names.
    UNIQUE_NAME_REGEX = re.compile(_('^(.*?)\\s*(\\d+)$'))


    assert UNIQUE_NAME_REGEX.match(UNIQUE_NAME_TEMPLATE % {
                                     "name": DEFAULT_NAME,
                                     "number": 42,
                                   })


    ## Construction, loading, other lifecycle stuff

    def __init__(self, name=None, **kwargs):
        """Construct a new layer

        :param name: The name for the new layer.
        :param **kwargs: Ignored.

        All layer subclasses must permit construction without
        parameters.
        """
        super(LayerBase, self).__init__()
        # Defaults for the notifiable properties
        self._opacity = 1.0
        self._name = name
        self._visible = True
        self._locked = False
        self._mode = DEFAULT_COMBINE_MODE
        self._root_ref = None  # or a weakref to the root
        #: True if the layer was marked as selected when loaded.
        self.initially_selected = False

    @classmethod
    def new_from_openraster(cls, orazip, elem, tempdir, feedback_cb,
                            root, x=0, y=0, **kwargs):
        """Reads and returns a layer from an OpenRaster zipfile

        This implementation just creates a new instance of its class and
        calls `load_from_openraster()` on it. This should suffice for
        all subclasses which support parameterless construction.
        """

        layer = cls()
        layer.load_from_openraster(orazip, elem, tempdir, feedback_cb,
                                   x=x, y=y, **kwargs)
        return layer


    def load_from_openraster(self, orazip, elem, tempdir, feedback_cb,
                             x=0, y=0, **kwargs):
        """Loads layer data from an open OpenRaster zipfile

        :param orazip: An OpenRaster zipfile, opened for extracting
        :type orazip: zipfile.ZipFile
        :param elem: <layer/> or <stack/> element to load (stack.xml)
        :type elem: xml.etree.ElementTree.Element
        :param tempdir: A temporary working directory
        :param feedback_cb: Callback invoked to provide feedback to the user
        :param x: X offset of the top-left point for image data
        :param y: Y offset of the top-left point for image data
        :param **kwargs: Extensibility

        The base implementation loads the common layer flags from a `<layer/>`
        or `<stack/>` element, but does nothing more than that. Loading layer
        data from the zipfile or recursing into stack contents is deferred to
        subclasses.
        """
        attrs = elem.attrib
        self.name = unicode(attrs.get('name', ''))
        self.opacity = helpers.clamp(float(attrs.get('opacity', '1.0')),
                                      0.0, 1.0)

        compop = str(attrs.get('composite-op', ''))
        self.mode = OPENRASTER_COMBINE_MODES.get(compop, DEFAULT_COMBINE_MODE)

        visible = attrs.get('visibility', 'visible').lower()
        self.visible = (visible != "hidden")

        locked = attrs.get("edit-locked", 'false').lower()
        self.locked = helpers.xsd2bool(locked)

        selected = attrs.get("selected", 'false').lower()
        self.initially_selected = helpers.xsd2bool(selected)


    def __deepcopy__(self, memo):
        """Returns an independent copy of the layer, for Duplicate Layer

        >>> from copy import deepcopy
        >>> orig = LayerBase()
        >>> dup = deepcopy(orig)

        Everything about the returned layer must be a completely
        independent copy of the original layer.  If the copy can be
        worked on, working on it must leave the original unaffected.
        This base implementation can be reused/extended by subclasses if
        they support zero-argument construction. It will use the derived
        class's snapshotting implementation (see `save_snapshot()` and
        `load_snapshot()`) to populate the copy.
        """
        layer = self.__class__()
        layer.load_snapshot(self.save_snapshot())
        return layer


    def clear(self):
        """Clears the layer"""
        pass


    ## Properties

    @property
    def root(self):
        """The root of the layer tree structure

        Only `RootLayerStack` instances or None are accepted.
        You won't normally need to adjust this unless you're doing
        something fancy: it's automatically maintained by intermediate
        and root `LayerStack` elements in the tree whenever layers are
        added or removed from a rooted tree structure.
        """
        if self._root_ref is not None:
            return self._root_ref()
        return None

    @root.setter
    def root(self, newroot):
        if newroot is None:
            self._root_ref = None
        else:
            assert isinstance(newroot, RootLayerStack)
            self._root_ref = weakref.ref(newroot)

    @property
    def opacity(self):
        """Opacity of the layer

        Values must permit conversion to a `float` in [0, 1].
        Changing this property issues ``layer_properties_changed`` and
        appropriate ``layer_content_changed`` notifications via the root
        layer stack if the layer is within a tree structure.
        """
        return self._opacity

    @opacity.setter
    def opacity(self, opacity):
        opacity = helpers.clamp(float(opacity), 0.0, 1.0)
        if opacity == self._opacity:
            return
        self._opacity = opacity
        self._properties_changed(["opacity"])
        bbox = tuple(self.get_full_redraw_bbox())
        self._content_changed(*bbox)

    @property
    def name(self):
        """The layer's name, for display purposes

        Values must permit conversion to a (unicode) string.  If the
        layer is part of a tree structure, ``layer_properties_changed``
        notifications will be issued via the root layer stack. In
        addition, assigned names may be corrected to be unique within
        the tree.
        """
        return self._name

    @name.setter
    def name(self, name):
        if name is not None:
            name = unicode(name)
        else:
            name = self.DEFAULT_NAME
        oldname = self._name
        self._name = name
        root = self.root
        if root is not None:
            self._name = root.get_unique_name(self)
        if self._name != oldname:
            self._properties_changed(["name"])

    @property
    def visible(self):
        """Whether the layer is visible

        Values must permit conversion to a `bool`.
        Changing this property issues ``layer_properties_changed`` and
        appropriate ``layer_content_changed`` notifications via the root
        layer stack if the layer is within a tree structure.
        """
        return self._visible

    @visible.setter
    def visible(self, visible):
        visible = bool(visible)
        if visible == self._visible:
            return
        redraws = [self.get_full_redraw_bbox()]
        self._visible = visible
        self._properties_changed(["visible"])
        redraws.append(self.get_full_redraw_bbox())
        self._content_changed_aggregated(redraws)


    @property
    def locked(self):
        """Whether the layer is locked (immutable)"""
        return self._locked

    @locked.setter
    def locked(self, locked):
        locked = bool(locked)
        if locked != self._locked:
            self._locked = locked
            self._properties_changed(["locked"])

    @property
    def mode(self):
        """How this layer combines with its backdrop

        Values must permit conversion to an int, and should be within
        the range of the underlying C enumeration.
        Changing this property issues ``layer_properties_changed`` and
        appropriate ``layer_content_changed`` notifications via the root
        layer stack if the layer is within a tree structure.
        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        mode = int(mode)
        if mode < 0 or mode >= tiledsurface.NUM_COMBINE_MODES:
            mode = tiledsurface.DEFAULT_COMBINE_MODE
        if mode == self._mode:
            return
        redraws = [self.get_full_redraw_bbox()]
        self._mode = mode
        self._properties_changed(["mode"])
        redraws.append(self.get_full_redraw_bbox())
        self._content_changed_aggregated(redraws)


    ## Notifications

    def _content_changed(self, *args):
        """Notifies the root's content observers

        If this layer's root stack is defined, i.e. if it is part of a
        tree structure, the root's `layer_content_changed()` event
        method will be invoked with this layer and the supplied
        arguments. This reflects a region of pixels in the document
        changing.
        """
        root = self.root
        if root is not None:
            root.layer_content_changed(self, *args)

    def _content_changed_aggregated(self, bboxes):
        """Aggregated content-change notification"""
        if self.root is None:
            return
        redraw_bbox = helpers.Rect()
        for bbox in bboxes:
            if bbox.w == 0 and bbox.h == 0:
                redraw_bbox = bbox
                break
            redraw_bbox.expandToIncludeRect(bbox)
        self._content_changed(*redraw_bbox)

    def _properties_changed(self, properties):
        """Notifies the root's layer properties observers

        If this layer's root stack is defined, i.e. if it is part of a
        tree structure, the root's `layer_properties_changed()` event
        method will be invoked with the layer and the supplied
        arguments. This reflects details about the layer like its name
        or its locked status changing.
        """
        root = self.root
        if root is not None:
            root._notify_layer_properties_changed(self, set(properties))


    ## Info methods

    def get_icon_name(self):
        """The name of the icon to display for the layer

        Ideally symbolic. A value of `None` means that no icon should be
        displayed.
        """
        return None

    @property
    def effective_opacity(self):
        """The opacity used when compositing a layer: zero if invisible

        This must match the appearance given by `composite_tile()` when it is
        called with no `layers` list, even if that method uses other means to
        determine how or whether to write its output. The base class's
        effective opacity is zero because the base `composite_tile()` does
        nothing.
        """
        return 0.0

    def get_alpha(self, x, y, radius):
        """Gets the average alpha within a certain radius at a point

        :param x: model X coordinate
        :param y: model Y coordinate
        :param radius: radius over which to average
        :rtype: float

        The return value is not affected by the layer opacity, effective or
        otherwise. This is used by `Document.pick_layer()` and friends to test
        whether there's anything significant present at a particular point.
        The default alpha at a point is zero.
        """
        return 0.0

    def get_bbox(self):
        """Returns the inherent (data) bounding box of the layer

        :rtype: lib.helpers.Rect

        The returned rectangle is tile-aligned. It's just a default (zero-size)
        rect in the base implementation.
        """
        return helpers.Rect()

    def get_full_redraw_bbox(self):
        """Returns the full update notification bounding box of the layer

        :rtype: lib.helpers.Rect

        This is the bounding box which should be used for redrawing if a
        layer-wide property like opacity or combining mode changes.
        Normally this is the layer's data bounding box, which allows the
        GUI to skip empty tiles when redrawing the layer stack. If
        instead the layer's compositing mode means that an opacity of
        zero affects the backdrop regardless, then the returned bbox is
        a zero-size rectangle, which is the signal for a full redraw.
        """
        if self.mode in MODES_EFFECTIVE_AT_ZERO_ALPHA:
            return helpers.Rect()
        else:
            return self.get_bbox()

    def is_empty(self):
        """Tests whether the surface is empty

        Always true in the base implementation.
        """
        return True

    def get_paintable(self):
        """True if this layer currently accepts painting brushstrokes

        Always false in the base implementation.
        """
        return False

    def get_fillable(self):
        """True if this layer currently accepts flood fill

        Always false in the base implementation.
        """
        return False

    def get_stroke_info_at(self, x, y):
        """Return the brushstroke at a given point

        :param x: X coordinate to pick from, in model space.
        :param y: Y coordinate to pick from, in model space.
        :rtype: lib.strokemap.StrokeShape or None

        Returns None for the base class.
        """
        return None

    def get_last_stroke_info(self):
        """Return the most recently painted stroke

        :rtype lib.strokemap.StrokeShape or None

        Returns None for the base class.
        """
        return None

    def get_mode_normalizable(self):
        """True if this layer can be normalized"""
        unsupported = set(MODES_EFFECTIVE_AT_ZERO_ALPHA)
        # Normalizing would have to make an infinite number of tiles
        unsupported.update(MODES_DECREASING_BACKDROP_ALPHA)
        # Normal mode cannot decrease the bg's alpha
        return self.mode not in unsupported

    def get_trimmable(self):
        """True if this layer currently accepts trim()"""
        return False

    def has_interesting_name(self):
        """True if the layer looks as if it has a user-assigned name

        Interesting means non-blank, and not the default name or a
        numbered version of it. This is used when merging layers: Merge
        Down is used on temporary layers a lot, and those probably have
        boring names.
        """
        name = self._name
        if name is None or name.strip() == '':
            return False
        if name == self.DEFAULT_NAME:
            return False
        match = self.UNIQUE_NAME_REGEX.match(name)
        if match is not None:
            base = unicode(match.group(1))
            if base == self.DEFAULT_NAME:
                return False
        return True


    ## Flood fill

    def flood_fill(self, x, y, color, bbox, tolerance, dst_layer=None):
        """Fills a point on the surface with a colour

        See `PaintingLayer.flood_fill() for parameters and semantics. The base
        implementation does nothing.
        """
        pass


    ## Rendering

    def get_tile_coords(self):
        """Returns all data tiles in this layer

        :returns: All tiles with data
        :rtype: sequence

        This method should return a sequence listing the coordinates for
        all tiles with data in this layer.

        It is used when computing layer merges.  Tile coordinates must
        be returned as ``(tx, ty)`` pairs.

        The base implementation returns an empty sequence.
        """
        return []

    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       **kwargs):
        """Unconditionally copy one tile's data into an array without options

        The visibility, opacity, and compositing mode flags of this layer must
        be ignored, or take default values. If the layer has sub-layers, they
        must be composited together as an isolated group (i.e. over an empty
        backdrop) using their `composite_tile()` method. It is the result of
        this compositing which is blitted, ignoring this layer's visibility,
        opacity, and compositing mode flags.

        :param dst: Target tile array (uint16, NxNx4, 15-bit scaled ints)
        :type dst: numpy.ndarray
        :param dst_has_alpha: the alpha channel in dst should be preserved
        :type dst_has_alpha: bool
        :param tx: Tile X coordinate, in model tile space
        :type tx: int
        :param ty: Tile Y coordinate, in model tile space
        :type ty: int
        :param mipmap_level: layer mipmap level to use
        :type mipmap_level: int
        :param **kwargs: extensibility...

        The base implementation does nothing.
        """
        pass

    def composite_tile( self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                        layers=None, previewing=None, **kwargs ):
        """Composite a tile's data into an array, respecting flags/layers list

        Unlike `blit_tile_into()`, the visibility, opacity, and compositing
        mode flags of this layer must be respected.  It otherwise works just
        like `blit_tile_into()`, but may make a local decision about whether
        to render as an isolated group.  This method uses the same parameters
        as `blit_tile_into()`, with two additions:

        :param layers: the set of layers to render
        :type layers: set of layers, or None
        :param previewing: the layer currently being previewed
        :type previewing: layer

        If `layers` is defined, it identifies the layers which are to be
        rendered: certain special rendering modes require this. For layers
        other then the root stack, layers should not render themselves if
        omitted from a defined `layers`.

        When `previewing` is set, `layers` must still be obeyed.  The preview
        layer should be rendered with normal blending and compositing modes,
        and with full opacity. This rendering mode is used for layer blink
        previewing.

        The base implementation does nothing.
        """
        pass


    def render_as_pixbuf(self, *rect, **kwargs):
        """Renders this layer as a pixbuf

        :param *rect: rectangle to save, as a 4-tuple
        :param **kwargs: passed to pixbufsurface.render_as_pixbuf()
        :rtype: Gdk.Pixbuf
        """
        raise NotImplementedError


    ## Translation

    def get_move(self, x, y):
        """Get a translation/move object for this layer

        :param x: Model X position of the start of the move
        :param y: Model X position of the start of the move
        :returns: A move object
        """
        raise NotImplementedError


    def translate(self, dx, dy):
        """Translate a layer non-interactively

        :param dx: Horizontal offset in model coordinates
        :param dy: Vertical offset in model coordinates
        :returns: full redraw bboxes for the move: ``[before, after]``
        :rtype: list

        The base implementation uses `get_move()` and the object it returns.
        """
        update_bboxes = [self.get_full_redraw_bbox()]
        move = self.get_move(0, 0)
        move.update(dx, dy)
        move.process(n=-1)
        move.cleanup()
        update_bboxes.append(self.get_full_redraw_bbox())
        return update_bboxes


    ## Standard stuff

    def __repr__(self):
        """Simplified repr() of a layer"""
        if self.name:
            return "<%s %r>" % (self.__class__.__name__, self.name)
        else:
            return "<%s>" % (self.__class__.__name__)

    def __nonzero__(self):
        """Layers are never false"""
        return True

    def __eq__(self, layer):
        """Two layers are only equal if they are the same object

        This is meaningful during layer repositions in the GUI, where
        shallow copies are used.
        """
        return self is layer


    ## Saving


    def save_as_png(self, filename, *rect, **kwargs):
        """Save to a named PNG file

        :param filename: filename to save to
        :param *rect: rectangle to save, as a 4-tuple
        :param **kwargs: passed to pixbufsurface.save_as_png()
        :rtype: Gdk.Pixbuf

        The base implementation does nothing.
        """
        pass


    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        """Saves the layer's data into an open OpenRaster ZipFile

        :param orazip: a `zipfile.ZipFile` open for write
        :param tmpdir: path to a temp dir, removed after the save
        :param path: Unique path of the layer, for encoding in filenames
        :type path: tuple of ints
        :param canvas_bbox: Bounding box of all layers, absolute coords
        :type canvas_bbox: tuple
        :param frame_bbox: Bounding box of the image being saved
        :type frame_bbox: tuple
        :param **kwargs: Keyword args used by the save implementation
        :returns: element describing data written
        :rtype: xml.etree.ElementTree.Element

        There are three bounding boxes which need to considered. The
        inherent bbox of the layer as returned by `get_bbox()` is always
        tile aligned and refers to absolute model coordinates, as is
        `canvas_bbox`.

        All of the above bbox's coordinates are defined relative to the
        canvas origin. However, when saving, the data written must be
        translated so that `frame_bbox`'s top left corner defines the
        origin (0, 0), of the saved OpenRaster file. The width and
        height of `frame_bbox` determine the saved image's dimensions.

        More than one file may be written to the zipfile. The etree
        element returned should describe everything that was written.

        Paths must be unique sequences of ints, but are not necessarily
        valid RootLayerStack paths. It's faked for the normally
        unaddressable background layer right now, for example.
        """
        raise NotImplementedError


    def _get_stackxml_element(self, frame_bbox, tag):
        """Internal: basic layer info for .ora saving as an etree Element"""
        x0, y0 = frame_bbox[0:2]
        bx, by, bw, bh = self.get_bbox()
        elem = ET.Element(tag)
        attrs = elem.attrib
        if self.name:
            attrs["name"] = str(self.name)
        attrs["x"] = str(bx - x0)
        attrs["y"] = str(by - y0)
        attrs["opacity"] = str(self.opacity)
        if self.initially_selected:
            attrs["selected"] = "true"
        if self.locked:
            attrs["edit-locked"] = "true"
        if self.visible:
            attrs["visibility"] = "visible"
        else:
            attrs["visibility"] = "hidden"
        compop = mypaintlib.combine_mode_get_info(self.mode).get("name")
        if compop is not None:
            attrs["composite-op"] = str(compop)
        return elem


    ## Painting symmetry axis


    def set_symmetry_axis(self, center_x):
        """Sets the surface's painting symmetry axis

        :param center_x: Model X coordinate of the axis of symmetry. Set
               to None to remove the axis of symmetry
        :type x: `float` or `None`

        This is only useful for paintable layers. Received strokes are
        reflected in the symmetry axis when it is set.

        the base implementation does nothing.
        """
        pass


    ## Snapshot


    def save_snapshot(self):
        """Snapshots the state of the layer, for undo purposes

        The returned data should be considered opaque, useful only as a
        memento to be restored with load_snapshot().
        """
        return _LayerBaseSnapshot(self)


    def load_snapshot(self, sshot):
        """Restores the layer from snapshot data"""
        redraws = [self.get_full_redraw_bbox()]
        sshot.restore_to_layer(self)
        redraws.append(self.get_full_redraw_bbox())
        self._content_changed_aggregated(redraws)


    ## Trimming


    def trim(self, rect):
        """Trim the layer to a rectangle, discarding data outside it

        :param rect: A trimming rectangle in model coordinates
        :type rect: tuple (x, y, w, h)

        The base implementation does nothing.
        """
        pass


    ## Type-specific actions

    def activate_layertype_action(self):
        """Perform the special action associated with this layer type

        This corresponds to the user clicking on the layer's type icon, as
        returned by `self.get_icon_name()`. The default action does nothing.
        """
        pass


class _LayerBaseSnapshot (object):
    """Base snapshot implementation

    Snapshots are stored in commands, and used to implement undo and redo.
    They must be independent copies of the data, although copy-on-write
    semantics are fine. Snapshot objects must be complete enough clones of the
    layer's data for duplication to work.
    """

    def __init__(self, layer):
        super(_LayerBaseSnapshot, self).__init__()
        self.opacity = layer.opacity
        self.name = layer.name
        self.mode = layer.mode
        self.opacity = layer.opacity
        self.visible = layer.visible
        self.locked = layer.locked

    def restore_to_layer(self, layer):
        layer.opacity = self.opacity
        layer.name = self.name
        layer.mode = self.mode
        layer.opacity = self.opacity
        layer.visible = self.visible
        layer.locked = self.locked


class LoadError (Exception):
    """Raised when loading to indicate that a layer cannot be loaded"""
    pass


## Stacks of layers


class LayerStack (LayerBase):
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

    Be careful to maintain global uniqueness of layers within the root
    layer stack. If this isn't respected, then replacing an instance of
    item which exists in two or more places in the tree will break that
    layer's root reference and cause it to silently stop emitting
    updates. Use a `PlaceholderLayer` to work around this, or just
    reinstate the root ref when you're done juggling layers.
    """

    ## Class constants

    #TRANSLATORS: Short default name for layer groups.
    DEFAULT_NAME = _("Group")


    ## Construction and other lifecycle stuff

    def __init__(self, **kwargs):
        """Initialize, with no sub-layers"""
        self._layers = []  # must be done before supercall
        super(LayerStack, self).__init__(**kwargs)
        # Defaults for properties with notifications
        self._isolated = False
        # Blank background, for use in rendering
        N = tiledsurface.N
        blank_arr = numpy.zeros((N, N, 4), dtype='uint16')
        self._blank_bg_surface = tiledsurface.Background(blank_arr)


    def load_from_openraster(self, orazip, elem, tempdir, feedback_cb,
                             x=0, y=0, **kwargs):
        """Load this layer from an open .ora file"""
        if elem.tag != "stack":
            raise LoadError, "<stack/> expected"
        super(LayerStack, self) \
            .load_from_openraster(orazip, elem, tempdir, feedback_cb,
                                  x=x, y=y, **kwargs)
        self.clear()
        x += int(elem.attrib.get("x", 0))
        y += int(elem.attrib.get("y", 0))
        isolated_flag = unicode(elem.attrib.get("isolation", "auto"))
        self.isolated = (isolated_flag.lower() != "auto")
        # Document order is the same as _layers, bottom layer to top.
        for child_elem in elem.findall("./*"):
            assert child_elem is not elem
            self.load_child_layer_from_openraster(orazip, child_elem,
                            tempdir, feedback_cb, x=x, y=y, **kwargs)


    def load_child_layer_from_openraster(self, orazip, elem, tempdir,
                                         feedback_cb,
                                         x=0, y=0, **kwargs):
        """Loads a single child layer element from an open .ora file"""
        try:
            child = layer_new_from_openraster(orazip, elem, tempdir,
                                              feedback_cb, self.root,
                                              x=x, y=y, **kwargs)
        except LoadError:
            logger.warning("Skipping non-loadable layer")
        if child is None:
            logger.warning("Skipping empty layer")
            return
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


    ## Properties

    @property
    def isolated(self):
        """Explicit group isolation flag"""
        return self._isolated

    @isolated.setter
    def isolated(self, isolated):
        isolated = bool(isolated)
        if isolated == self._isolated:
            return
        updates = [self.get_full_redraw_bbox()]
        self._isolated = isolated
        self._properties_changed(["isolated"])
        updates.append(self.get_full_redraw_bbox())
        self._content_changed_aggregated(updates)

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
        >>> stack.append(LayerBase())
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
        >>> group.append(PaintingLayer())
        >>> group.append(PaintingLayer())
        >>> group.append(PaintingLayer())
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
        self._content_changed_aggregated(updates)

    def __getitem__(self, index):
        """Fetches the layer at an index"""
        return self._layers[index]

    def index(self, layer):
        """Fetches the index of a child layer, by equality"""
        return self._layers.index(layer)


    ## Info methods

    def get_auto_isolation(self):
        """Whether the stack implicitly needs isolated group rendering

        :returns: Stack must always be rendered as an isolated group
        :rtype: bool

        Auto-isolation means that the group's own opacity or mode
        properties require it to always be rendered as an isolated
        group, regardless of the value of its explicit isolation flag.

        The layer model recommended by Compositing and Blending Level 1
        implies that group isolation happens automatically when group
        invariance breaks due to particular properties of the group
        element alone.

        ref: http://www.w3.org/TR/compositing-1/#csscompositingrules_SVG
        """
        return self.opacity < 1.0 or self.mode != DEFAULT_COMBINE_MODE

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

    def blit_tile_into( self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                        **kwargs ):
        """Unconditionally copy one tile's data into an array"""
        N = tiledsurface.N
        tmp = numpy.zeros((N, N, 4), dtype='uint16')
        for layer in reversed(self._layers):
            layer.composite_tile(tmp, True, tx, ty, mipmap_level,
                                 layers=None, **kwargs)
        if dst.dtype == 'uint16':
            mypaintlib.tile_copy_rgba16_into_rgba16(tmp, dst)
        elif dst.dtype == 'uint8':
            if dst_has_alpha:
                mypaintlib.tile_convert_rgba16_to_rgba8(tmp, dst)
            else:
                mypaintlib.tile_convert_rgbu16_to_rgbu8(tmp, dst)
        else:
            raise ValueError, ('Unsupported destination buffer type %r' %
                               dst.dtype)


    def composite_tile( self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
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
        isolate = self.isolated or self.get_auto_isolation()
        if isolate and previewing and self is not previewing:
            isolate = False
        if isolate and solo and self is not solo:
            isolate = False
        if isolate:
            N = tiledsurface.N
            tmp = numpy.zeros((N, N, 4), dtype='uint16')
            for layer in reversed(self._layers):
                p = (self is previewing) and layer or previewing
                s = (self is solo) and layer or solo
                layer.composite_tile(tmp, True, tx, ty, mipmap_level,
                                     layers=layers, previewing=p, solo=s,
                                     **kwargs)
            if previewing or solo:
                mode = DEFAULT_COMBINE_MODE
                opacity = 1.0
            mypaintlib.tile_combine(mode, tmp, dst, dst_has_alpha, opacity)
        else:
            for layer in reversed(self._layers):
                p = (self is previewing) and layer or previewing
                s = (self is solo) and layer or solo
                layer.composite_tile(dst, dst_has_alpha, tx, ty, mipmap_level,
                                     layers=layers, previewing=p, solo=s,
                                     **kwargs)

    def render_as_pixbuf(self, *args, **kwargs):
        return pixbufsurface.render_as_pixbuf(self, *args, **kwargs)


    ## Flood fill

    def flood_fill(self, x, y, color, bbox, tolerance, dst_layer=None):
        """Fills a point on the surface with a colour (into other only!)

        See `PaintingLayer.flood_fill() for parameters and semantics. Layer
        stacks only support flood-filling into other layers because they are
        not surface backed.
        """
        assert dst_layer is not self
        assert dst_layer is not None
        src = tiledsurface.TileRequestWrapper(self)
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

    def save_as_png(self, filename, *rect, **kwargs):
        """Save to a named PNG file"""
        if 'alpha' not in kwargs:
            kwargs['alpha'] = True
        pixbufsurface.save_as_png(self, filename, *rect, **kwargs)


    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        """Saves the stack's data into an open OpenRaster ZipFile"""
        stack_elem = self._get_stackxml_element(frame_bbox, "stack")

        # Saving uses the same origin for all layers regardless of nesting
        # depth, that of the frame. It's more compatible, and closer to
        # MyPaint's internal model. Sub-stacks therefore get the default offset,
        # which is zero.
        del stack_elem.attrib["x"]
        del stack_elem.attrib["y"]

        for layer_idx, layer in list(enumerate(self)):
            layer_path = tuple(list(path) + [layer_idx])
            layer_elem = layer.save_to_openraster(orazip, tmpdir, layer_path,
                                                  canvas_bbox, frame_bbox,
                                                  **kwargs)
            stack_elem.append(layer_elem)

        isolated = "isolate" if self.isolated else "auto"
        stack_elem.attrib["isolation"] = isolated

        return stack_elem


    ## Snapshotting

    def save_snapshot(self):
        """Snapshots the state of the layer, for undo purposes"""
        return _LayerStackSnapshot(self)


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
        return "mypaint-layers-symbolic"


class _LayerStackSnapshot (_LayerBaseSnapshot):
    """Snapshot of a layer stack's state"""

    def __init__(self, layer):
        super(_LayerStackSnapshot, self).__init__(layer)
        self.isolated = layer.isolated
        self.layer_snaps = [l.save_snapshot() for l in layer._layers]
        self.layer_classes = [l.__class__ for l in layer._layers]

    def restore_to_layer(self, layer):
        super(_LayerStackSnapshot, self).restore_to_layer(layer)
        layer.isolated = self.isolated
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


class PlaceholderLayer (LayerStack):
    """Trivial temporary placeholder layer, used for moves etc.

    The layer stack architecture does not allow for the same layer
    appearing twice in a tree structure. Layer operations therefore
    require unique placeholder layers occasionally, typically when
    swapping nodes in the tree or handling drags.
    """

    #TRANSLATORS: Short default name for temporary placeholder layers.
    #TRANSLATORS: (The user should never see this except in error cases)
    DEFAULT_NAME = _("Placeholder")


class RootLayerStack (LayerStack):
    """Specialized document root layer stack

    In addition to the basic `LayerStack` implementation, this class's
    methods and properties provide:

     * the document's background, using an internal BackgroundLayer;
     * tile rendering for the doc via the regular rendering interface;
     * special viewing modes (solo, previewing);
     * the currently selected layer;
     * path-based access to layers in the tree;
     * manipulation of layer paths; and
     * convenient iteration over the tree structure.

    In other words, root layer stacks handle anything that needs
    document-scale oversight of the tree structure to operate.  An
    instance of this instantiated for the running app as part of the
    primary `lib.document.Document` object.  The descendent layers of
    this object are those that are presented as user-addressable layers
    in the Layers panel.
    """

    ## Initialization

    def __init__(self, doc, **kwargs):
        """Construct, as part of a model

        :param doc: The model document. May be None for testing.
        :type doc: lib.document.Document
        """
        super(RootLayerStack, self).__init__(**kwargs)
        self._doc = doc
        # Background
        default_bg = (255, 255, 255)
        self._default_background = default_bg
        self._background_layer = BackgroundLayer(default_bg)
        self._background_visible = True
        # Special rendering state
        self._current_layer_solo = False
        self._current_layer_previewing = False
        # Current layer
        self._current_path = ()

    def clear(self):
        """Clear the layer and set the default background"""
        super(RootLayerStack, self).clear()
        self.set_background(self._default_background)

    def ensure_populated(self, layer_class=None):
        """Ensures that the stack is non-empty by making a new layer if needed

        :param layer_class: The class of layer to add, if necessary
        :type layer_class: LayerBase
        :returns: The new layer instance, or None if nothing was created

        The default `layer_class` is the regular painting layer.
        """
        if layer_class is None:
            layer_class = PaintingLayer
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
        """
        # Layer-solo mode should probably *not* render without the
        # background.  While it's intended to be used for showing what a
        # layer contains by itself, part of that involves showing what
        # effect the the layer's mode has. Layer-solo over real alpha
        # checks doesn't permit that.
        return ((self._current_layer_solo or self._background_visible) and
                not self._current_layer_previewing)
        # Conversely, current-layer-preview is intended to *blink* very
        # visibly to notify the user, so always turn off the background
        # for that.

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

    def _layers_along_path(self, path):
        """Yields all layers along a path, not including the root"""
        if not path:
            return
        unused_path = list(path)
        layer = self
        while len(unused_path) > 0:
            if not isinstance(layer, LayerStack):
                break
            idx = unused_path.pop(0)
            if not (0 <= idx < len(layer)):
                break
            layer = layer[idx]
            yield layer

    def render_into(self, surface, tiles, mipmap_level, overlay=None):
        """Tiled rendering: used for display only

        :param surface: target rgba8 surface
        :type surface: lib.pixbufsurface.Surface
        :param tiles: tile coords, (tx, ty), to render
        :type tiles: list
        :param mipmap_level: layer and surface mipmap level to use
        :type mipmap_level: int
        :param overlay: overlay layer to render (stroke highlighting)
        :type overlay: SurfaceBackedLayer
        """
        # Decide a rendering mode
        background = self._get_render_background()
        dst_has_alpha = not self.get_render_is_opaque()
        layers = None
        if self._current_layer_previewing or self._current_layer_solo:
            path = self.get_current_path()
            layers = set(self._layers_along_path(path))
        previewing = None
        solo = None
        if self._current_layer_previewing:
            previewing = self.current
        if self._current_layer_solo:
            solo = self.current
        # Blit loop. Could this be done in C++?
        for tx, ty in tiles:
            with surface.tile_request(tx, ty, readonly=False) as dst:
                self.composite_tile(dst, dst_has_alpha, tx, ty,
                                    mipmap_level, layers=layers,
                                    background=background,
                                    overlay=overlay,
                                    previewing=previewing, solo=solo)

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
        while ( mipmap_level < tiledsurface.MAX_MIPMAP_LEVEL and
                max(w, h) >= 512 ):
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
        self.composite_tile( dst, dst_has_alpha, tx, ty,
                             mipmap_level=mipmap_level, **kwargs )


    def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       layers=None, background=None, overlay=None,
                       **kwargs):
        """Composite a tile's data, respecting flags/layers list

        The root layer stack implementation accepts the parameters
        documented in `BaseLayer.composite_tile()`, and also consumes:

        :param background: Whether to render the background layer
        :type background: bool or None
        :param overlay: Overlay layer
        :type overlay: BaseLayer

        If `background` is None, an internal default will be used. The
        root layer has flags which ensure it is always visible, so the
        result is generally indistinguishable from `blit_tile_into()`.
        However the rendering loop, `render_into()`, calls this method
        and sometimes passes in a zero-alpha `background` for special
        rendering modes which need isolated rendering.

        The overlay layer is optional. If present, it is drawn on top.
        Overlay layers must support 15-bit scaled-int tile compositing.

        As a further extension to the base API, `dst` may be an 8bpp
        array. A temporary 15-bit scaled int array is used for
        compositing in this case, and the output is converted to 8bpp.
        """
        if background is None:
            background = self._get_render_background()
        if background:
            background_surface = self._background_layer._surface
        else:
            background_surface = self._blank_bg_surface
        assert dst.shape[-1] == 4
        if dst.dtype == 'uint8':
            dst_8bit = dst
            N = tiledsurface.N
            dst = numpy.empty((N, N, 4), dtype='uint16')
        else:
            dst_8bit = None
        background_surface.blit_tile_into(dst, dst_has_alpha, tx, ty,
                                          mipmap_level)
        for layer in reversed(self):
            layer.composite_tile(dst, dst_has_alpha, tx, ty,
                                 mipmap_level, layers=layers, **kwargs)
        if overlay:
            overlay.composite_tile(dst, dst_has_alpha, tx, ty,
                                   mipmap_level, layers=set([overlay]),
                                   **kwargs)
        if dst_8bit is not None:
            if dst_has_alpha:
                mypaintlib.tile_convert_rgba16_to_rgba8(dst, dst_8bit)
            else:
                mypaintlib.tile_convert_rgbu16_to_rgbu8(dst, dst_8bit)


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
            return
        path = tuple(path)
        while len(path) > 0:
            layer = self.deepget(path)
            if layer is not None:
                new_current_path = path
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
        :type obj: layer.BackgroundLayer or tuple or numpy array
        :param make_default: make this the default bg for clear()
        :type make_default: bool

        The background object argument `obj` can be a background layer,
        or an RGB triple (uint8), or a HxWx4 or HxWx3 numpy array which
        can be either uint8 or uint16.

        Setting the background issues a full redraw for the root layer,
        and also issues the `background_changed` event. The background
        will also be made visible if it isn't already.
        """
        if isinstance(obj, BackgroundLayer):
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
        existing = { d.name for d in self.deepiter()
                     if d is not layer and d.name is not None }
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
        newname = layer.UNIQUE_NAME_TEMPLATE % { "name": base,
                                                 "number": num, }
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
          >>> for p, l in [ ([0], PaintingLayer()),
          ...               ([1], LayerStack()),
          ...               ([1, 0], LayerStack()),
          ...               ([1, 0, 0], LayerStack()),
          ...               ([1, 0, 0, 0], PaintingLayer()),
          ...               ([1, 1], PaintingLayer()) ]:
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
          >>> for p, l in [ ([0], PaintingLayer()),
          ...               ([1], LayerStack()),
          ...               ([1, 0], LayerStack()),
          ...               ([1, 0, 0], LayerStack()),
          ...               ([1, 0, 0, 0], PaintingLayer()),
          ...               ([1, 1], PaintingLayer()) ]:
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
            elif isinstance(self.deepget(path, None), LayerStack):
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
            if isinstance(layer, LayerStack) and len(path) > 0:
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
            if isinstance(sibling, LayerStack):
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

            >>> root = RootLayerStack(doc=None)
            >>> for p, l in [([0], PaintingLayer()),
            ...              ([1], LayerStack(name="A")),
            ...              ([1,0], PaintingLayer(name="B")),
            ...              ([1,1], PaintingLayer()),
            ...              ([2], PaintingLayer(name="C"))]:
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

            >>> root.deepget([1]).isolated = True
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
            if not isinstance(layer, LayerStack):
                continue
            if bghit and (layer.isolated or layer.get_auto_isolation()):
                continue
            queue[:0] = [(path + (i,), c) for i, c in enumerate(layer)]


    def deepiter(self):
        """Iterates across all descendents of the stack

        >>> stack, leaves = _make_test_stack()
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

        >>> stack, leaves = _make_test_stack()
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

        >>> stack, leaves = _make_test_stack()
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
                if not isinstance(layer, LayerStack):
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

        >>> stack, leaves = _make_test_stack()
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
            if not isinstance(stack, LayerStack):
                raise IndexError("All nonfinal elements of %r must "
                                 "identify a stack" % (path,))
            if unused_path:
                stack = stack[idx]
            else:
                stack.insert(idx, layer)
                layer.name = self.get_unique_name(layer)
                return
        assert (len(unused_path) > 0), ("deepinsert() should never "
                                        "exhaust the path")

    def deeppop(self, path):
        """Removes a layer by its path

        >>> stack, leaves = _make_test_stack()
        >>> stack.deeppop(())
        Traceback (most recent call last):
        ...
        IndexError: Cannot pop the root stack
        >>> stack.deeppop([0])
        <LayerStack len=3 '0'>
        >>> stack.deeppop((0,1))
        <PaintingLayer '11'>
        >>> stack.deeppop((0,2))
        Traceback (most recent call last):
        ...
        IndexError: ...
        """
        if len(path) == 0:
            raise IndexError, "Cannot pop the root stack"
        parent_path = path[:-1]
        child_index = path[-1]
        if len(parent_path) == 0:
            parent = self
        else:
            parent = self.deepget(parent_path)
        old_current = self.current_path
        removed = parent.pop(child_index)
        self.current_path = old_current # i.e. nearest remaining
        return removed


    def deepremove(self, layer):
        """Removes a layer from any of the root's descendents

        >>> stack, leaves = _make_test_stack()
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
            self.current_path = old_current # i.e. nearest remaining
            return None
        raise ValueError("Layer is not in the root stack or "
                         "any descendent")


    def deepindex(self, layer):
        """Return a path for a layer by searching the stack tree

        >>> stack, leaves = _make_test_stack()
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
          >>> root.deepinsert([0], PaintingLayer())
          >>> root.deepinsert([1], LayerStack())
          >>> root.deepinsert([1, 0], PaintingLayer())
          >>> layer = PaintingLayer()
          >>> root.deepinsert([1, 1], layer)
          >>> root.deepinsert([1, 2], PaintingLayer())
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
          Traceback (most recent call last):
          ...
          ValueError: ...
          >>> root.canonpath(usefirst=True)
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
            ...     ([0], PaintingLayer(name="notpart")),
            ...     ([1], LayerStack(name="ggp")),
            ...     ([1,0], LayerStack(name="gp")),
            ...     ([1,0,0], LayerStack(name="p")),
            ...     ([1,0,0,0], PaintingLayer(name="notpart")),
            ...     ([1,0,0,1], PaintingLayer(name="addressed")),
            ...     ([1,0,0,2], PaintingLayer(name="invis.:notpart")),
            ...     ([1,0,0,3], PaintingLayer(name="A")),
            ...     ([1,0,1], PaintingLayer(name="B")),
            ...     ([1,0,2], LayerStack(name="C")),
            ...     ([1,0,2,0], PaintingLayer(name="notpart")),
            ...     ([1,0,3], PaintingLayer(name="D")),
            ...     ([1,1], PaintingLayer(name="E")),
            ...     ([2], PaintingLayer(name="F")),
            ...     ]:
            ...     root.deepinsert(path, layer)
            >>> root.deepget([1,0,0,2]).visible = False

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
            >>> root.deepget([1]).isolated = True
            >>> [b.name for b in root._get_backdrop(path)]
            [u'E', u'D', u'C', u'B', u'A']
            >>> [b.name for b in root._get_backdrop([1])]
            [u'background', u'F']

        Auto-isolation counts for this, so the opacities and modes of
        parent stacks don't need to be considered when merging
        backdrops:

            >>> root.deepget([1,0]).opacity = 0.5
            >>> [b.name for b in root._get_backdrop(path)]
            [u'D', u'C', u'B', u'A']
            >>> root.deepget([1,0,0]).mode = mypaintlib.CombineScreen
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
            if stack.isolated or stack.get_auto_isolation():
                backdrop = []
        return backdrop

    def layer_new_normalized(self, path):
        """Copy a layer to a normal painting layer that looks the same

        :param tuple path: Path to normalize
        :returns: New normalized layer
        :rtype: lib.layer.PaintingLayer

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
        """
        srclayer = self.deepget(path)
        if not srclayer:
            raise ValueError("Path %r not found", path)
        # We need a backdrop sometimes, for layer removal
        needs_backdrop_removal = True
        backdrop_layers = []
        if srclayer.mode == DEFAULT_COMBINE_MODE and srclayer.opacity == 1.0:
            if isinstance(srclayer, LayerStack):
                needs_backdrop_removal = not srclayer.isolated
            elif isinstance(srclayer, PaintingLayer): # optimization for merge
                if srclayer.visible:
                    return deepcopy(srclayer)
                else:
                    return PaintingLayer(name=srclayer.name)
            elif already_normal:
                needs_backdrop_removal = False
        if needs_backdrop_removal:
            backdrop_layers = self._get_backdrop(path)
        # Begin building output, and enumerate set of tiles to render
        dstlayer = PaintingLayer()
        dstlayer.name = srclayer.name
        tiles = set()
        for p, layer in self.walk():
            if not path_startswith(p, path):
                continue
            tiles.update(layer.get_tile_coords())
            if isinstance(layer, PaintingLayer) and not layer.locked:
                dstlayer.strokes[:0] = layer.strokes
        # Render loop
        logger.debug("Normalize: render using backdrop %r", backdrop_layers)
        dstsurf = dstlayer._surface
        N = tiledsurface.N
        for tx, ty in tiles:
            bd = numpy.zeros((N, N, 4), dtype='uint16')
            for layer in backdrop_layers:
                if layer is self._background_layer:
                    surf = self._background_layer._surface
                    surf.blit_tile_into(bd, True, tx, ty, 0)
                    # FIXME: shouldn't need this special case
                else:
                    layer.composite_tile(bd, True, tx, ty, mipmap_level=0)
            with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                mypaintlib.tile_copy_rgba16_into_rgba16(bd, dst)
                srclayer.composite_tile(dst, True, tx, ty, mipmap_level=0)
                if backdrop_layers:
                    dst[:,:,3] = 0 # minimize alpha (discard original)
                    mypaintlib.tile_flat2rgba(dst, bd)
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
        return target_path

    def layer_new_merge_down(self, path):
        """Create a new layer containg the Merge Down of two layers

        :param tuple path: Path to the top layer to Merge Down
        :returns: New merged layer
        :rtype: lib.layer.PaintingLayer

        The current layer and the one below it are merged into a new
        layer, if that is possible, and the new layer is returned.
        Nothing is inserted or removed from the stack.  Any merged layer
        will contain a combined strokemap based on the input layers -
        although locked layers' strokemaps are not merged.

        You get what you see. This means that both layers must be
        visible to be used in the output.
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
        dstlayer = PaintingLayer()
        tiles = set()
        strokes = []
        for layer in merge_layers:
            tiles.update(layer.get_tile_coords())
            assert isinstance(layer, PaintingLayer) and not layer.locked
            dstlayer.strokes[:0] = layer.strokes
        # Build a (hopefully sensible) combined name too
        names = [l.name for l in reversed(merge_layers)
                 if l.has_interesting_name()]
        #TRANSLATORS: name combining punctuation for Merge Down
        name = _(u", ").join(names)
        if name != '':
            dstlayer.name = name
        logger.debug("Merge Down: backdrop=%r", backdrop_layers)
        logger.debug("Merge Down: normalized source=%r", merge_layers)
        # Rendering loop
        N = tiledsurface.N
        dstsurf = dstlayer._surface
        for tx, ty in tiles:
            with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                for layer in merge_layers:
                    layer.composite_tile(dst, True, tx, ty, mipmap_level=0)
        return dstlayer

    def layer_new_merge_visible(self):
        """Create and return the merge of all currently visible layers

        :returns: New merged layer
        :rtype: lib.layer.PaintingLayer

        All visible layers are merged into a new PaintingLayer, which is
        returned. Nothing is inserted or removed from the stack.  The
        merged layer will contain a combined strokemap based on those
        layers which are visible but not locked.

        You get what you see. If the background layer is visible at the
        time of the merge, then many modes will pick up an image of it.
        It will be "subtracted" from the result of the merge so that the
        merge result can be stacked above the same background.

        See also: `walk()`, `background_visible`.
        """
        # What to render (+ strokemap)
        tiles = set()
        strokes = []
        for path, layer in self.walk(visible=True):
            tiles.update(layer.get_tile_coords())
            if isinstance(layer, PaintingLayer) and not layer.locked:
                strokes[:0] = layer.strokes
        dstlayer = PaintingLayer()
        dstlayer.strokes = strokes
        # Render & subtract backdrop (= the background, if visible)
        dstsurf = dstlayer._surface
        bgsurf = self._background_layer._surface
        for tx, ty in tiles:
            with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                self.composite_tile(
                    dst, True, tx, ty, mipmap_level=0,
                    background=self._background_visible
                )
                if self._background_visible:
                    with bgsurf.tile_request(tx, ty, readonly=True) as bg:
                        dst[:,:,3] = 0 # minimize alpha (discard original)
                        mypaintlib.tile_flat2rgba(dst, bg)
        return dstlayer


    ## Loading

    def load_from_openraster(self, orazip, elem, tempdir, feedback_cb,
                             x=0, y=0, **kwargs):
        """Load this layer from an open .ora file"""
        self._no_background = True
        super(RootLayerStack, self) \
            .load_from_openraster(orazip, elem, tempdir, feedback_cb,
                                  x=x, y=y, **kwargs)
        del self._no_background
        # Select a suitable working layer
        num_loaded = 0
        selected_path = None
        for path, loaded_layer in self.deepenumerate():
            if loaded_layer.initially_selected:
                selected_path = path
            num_loaded += 1
        logger.debug("Loaded %d layer(s)" % num_loaded)
        if num_loaded == 0:
            logger.error('Could not load any layer, document is empty.')
            logger.info('Adding an empty painting layer')
            empty_layer = PaintingLayer()
            self.append(empty_layer)
            selected_path = [0]
        num_layers = len(self)
        assert num_layers > 0
        if not selected_path:
            selected_path = [max(0, num_layers-1)]
        self.set_current_path(selected_path)


    def load_child_layer_from_openraster(self, orazip, elem, tempdir,
                                         feedback_cb, x=0, y=0, **kwargs):
        """Loads and appends a single child layer from an open .ora file"""
        attrs = elem.attrib
        # Handle MyPaint's special background tile notation
        bg_src = attrs.get('background_tile', None)
        if bg_src:
            assert self._no_background, "Only one background is permitted"
            try:
                logger.debug("background tile: %r", bg_src)
                bg_pixbuf = pixbuf_from_zipfile(orazip, bg_src, feedback_cb)
                self.set_background(bg_pixbuf)
                self._no_background = False
                return
            except tiledsurface.BackgroundError, e:
                logger.warning('ORA background tile not usable: %r', e)
        super(RootLayerStack, self) \
            .load_child_layer_from_openraster(orazip, elem, tempdir,
                                              feedback_cb, x=x, y=y, **kwargs)

    ## Saving

    def save_to_openraster(self, orazip, tmpdir, path, canvas_bbox,
                           frame_bbox, **kwargs):
        """Saves the stack's data into an open OpenRaster ZipFile"""
        stack_elem = super(RootLayerStack, self) \
            .save_to_openraster( orazip, tmpdir, path, canvas_bbox,
                                 frame_bbox, **kwargs )
        # Save background
        bg_layer = self.background_layer
        bg_layer.initially_selected = False
        bg_path = (len(self),)
        bg_elem = bg_layer.save_to_openraster( orazip, tmpdir, bg_path,
                                               canvas_bbox, frame_bbox,
                                               **kwargs )
        stack_elem.append(bg_elem)

        isolated = "isolated" if self.isolated else "auto"
        stack_elem.attrib["isolation"] = isolated

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


## Layers with data


class SurfaceBackedLayer (LayerBase):
    """Minimal Surface-backed layer implementation

    This minimal implementation is backed by a surface, which is used
    for rendering by by the main application; subclasses are free to
    choose whether they consider the surface to be the canonical source
    of layer data or something else with the surface being just a
    preview.
    """

    ## Class constants: capabilities

    #: Whether the surface can be painted to (if not locked)
    IS_PAINTABLE = False

    #: Whether the surface can be filled (if not locked)
    IS_FILLABLE = False

    #: Suffixes allowed in load_from_openraster()
    ALLOWED_SUFFIXES = []


    ## Initialization

    def __init__(self, surface=None, **kwargs):
        """Construct a new SurfaceBackedLayer

        :param surface: Surface to use, overriding the default.
        :param **kwargs: passed to superclass.

        If `surface` is specified, content observers will not be attached, and
        the layer will not be cleared during construction. The default is to
        instantiate and use a new, observed, `tiledsurface.Surface`.
        """
        super(SurfaceBackedLayer, self).__init__(**kwargs)

        # Pluggable surface implementation
        # Only connect observers if using the default tiled surface
        if surface is None:
            self._surface = tiledsurface.Surface()
            self._surface.observers.append(self._content_changed)
        else:
            self._surface = surface

    def load_from_surface(self, surface):
        """Load the backing surface image's tiles from another surface"""
        self._surface.load_from_surface(surface)

    def load_from_strokeshape(self, strokeshape):
        """Load image tiles from a strokemap.StrokeShape"""
        strokeshape.render_to_surface(self._surface)


    ## Loading

    def load_from_openraster(self, orazip, elem, tempdir, feedback_cb,
                             x=0, y=0, extract_and_keep=False, **kwargs):
        """Loads layer flags and bitmap/surface data from a .ora zipfile

        :param extract_and_keep: Set to true to extract and keep a copy

        The normal behaviour is to load the data file directly from `orazip`
        without using a temporary file.  If `extract_and_keep` is set, an
        alternative method is used which extracts

            os.path.join(tempdir, elem.attrib["src"])

        and reads from that. The caller is then free to do what it likes with
        this file.
        """
        # Load layer flags
        super(SurfaceBackedLayer, self) \
            .load_from_openraster(orazip, elem, tempdir, feedback_cb,
                                  x=x, y=y, **kwargs)
        # Read bitmap content into the surface
        attrs = elem.attrib
        src = attrs.get("src", None)
        src_rootname, src_ext = os.path.splitext(src)
        src_rootname = os.path.basename(src_rootname)
        src_ext = src_ext.lower()
        x += int(attrs.get('x', 0))
        y += int(attrs.get('y', 0))
        logger.debug("Loading %r at %+d%+d", src_rootname, x, y)
        t0 = time.time()
        suffixes = self.ALLOWED_SUFFIXES
        if src_ext not in suffixes:
            logger.error("Cannot load SurfaceBackedLayers from a %r", src_ext)
            raise LoadError, "Only %r are supported" % (suffixes,)
        if extract_and_keep:
            orazip.extract(src, path=tempdir)
            tmp_filename = os.path.join(tempdir, src)
            self.load_surface_from_pixbuf_file(tmp_filename, x, y, feedback_cb)
        else:
            pixbuf = pixbuf_from_zipfile(orazip, src, feedback_cb=feedback_cb)
            self.load_surface_from_pixbuf(pixbuf, x=x, y=y)
        t1 = time.time()
        logger.debug('%.3fs loading and converting src %r for %r',
                     t1 - t0, src_ext, src_rootname)


    def load_surface_from_pixbuf_file(self, filename, x=0, y=0,
                                      feedback_cb=None):
        """Loads the layer's surface from any file which GdkPixbuf can open"""
        fp = open(filename, 'rb')
        pixbuf = pixbuf_from_stream(fp, feedback_cb)
        fp.close()
        return self.load_surface_from_pixbuf(pixbuf, x, y)


    def load_surface_from_pixbuf(self, pixbuf, x=0, y=0):
        """Loads the layer's surface from a GdkPixbuf"""
        arr = helpers.gdkpixbuf2numpy(pixbuf)
        surface = tiledsurface.Surface()
        bbox = surface.load_from_numpy(arr, x, y)
        self.load_from_surface(surface)
        return bbox

    def clear(self):
        """Clears the layer"""
        self._surface.clear()

    ## Info methods

    @property
    def effective_opacity(self):
        """The opacity used when compositing a layer: zero if invisible"""
        # Mirror what composite_tile does.
        if self.visible:
            return self.opacity
        else:
            return 0.0

    def get_alpha(self, x, y, radius):
        """Gets the average alpha within a certain radius at a point"""
        return self._surface.get_alpha(x, y, radius)


    def get_bbox(self):
        """Returns the inherent bounding box of the surface, tile aligned"""
        return self._surface.get_bbox()


    def is_empty(self):
        """Tests whether the surface is empty"""
        return self._surface.is_empty()


    def get_paintable(self):
        """True if this layer currently accepts painting brushstrokes"""
        return self.IS_PAINTABLE and not self.locked


    def get_fillable(self):
        """True if this layer currently accepts flood fill"""
        return self.IS_FILLABLE and not self.locked

    ## Flood fill

    def flood_fill(self, x, y, color, bbox, tolerance, dst_layer=None):
        """Fills a point on the surface with a colour

        See `PaintingLayer.flood_fill() for parameters and semantics. This
        implementation does nothing.
        """
        pass


    ## Rendering

    def get_tile_coords(self):
        return self._surface.get_tiles().keys()

    def blit_tile_into(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       **kwargs):
        """Unconditionally copy one tile's data into an array without options

        The minimal surface-based implementation composites one tile of the
        backing surface over the array dst, modifying only dst.
        """
        self._surface.composite_tile( dst, dst_has_alpha, tx, ty,
                                      mipmap_level=mipmap_level,
                                      opacity=1, mode=DEFAULT_COMBINE_MODE )


    def composite_tile(self, dst, dst_has_alpha, tx, ty, mipmap_level=0,
                       layers=None, previewing=None, solo=None, **kwargs):
        """Composite a tile's data into an array, respecting flags/layers list

        The minimal surface-based implementation composites one tile of the
        backing surface over the array dst, modifying only dst.
        """
        mode = self.mode
        opacity = self.opacity
        if layers is not None:
            if self not in layers:
                return
        elif not self.visible:
            return
        if self is previewing:  # not solo though - we show the effect of that
            mode = DEFAULT_COMBINE_MODE
            opacity = 1.0
        self._surface.composite_tile( dst, dst_has_alpha, tx, ty,
                                      mipmap_level=mipmap_level,
                                      opacity=opacity, mode=mode )

    def render_as_pixbuf(self, *rect, **kwargs):
        """Renders this layer as a pixbuf"""
        return self._surface.render_as_pixbuf(*rect, **kwargs)


    ## Translating


    def get_move(self, x, y):
        """Get a translation/move object for this layer

        :param x: Model X position of the start of the move
        :param y: Model X position of the start of the move
        :returns: A move object

        Subclasses should extend this minimal implementation to provide
        additional functionality for moving things other than the surface tiles
        around.
        """
        return self._surface.get_move(x, y)


    ## Saving


    def save_as_png(self, filename, *rect, **kwargs):
        """Save to a named PNG file

        :param filename: filename to save to
        :param *rect: rectangle to save, as a 4-tuple
        :param **kwargs: passed to pixbufsurface.save_as_png()
        :rtype: Gdk.Pixbuf
        """
        self._surface.save_as_png(filename, *rect, **kwargs)


    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        """Saves the layer's data into an open OpenRaster ZipFile"""
        rect = self.get_bbox()
        return self._save_rect_to_ora( orazip, tmpdir, "layer", path,
                                       frame_bbox, rect, **kwargs )

    @staticmethod
    def _make_refname(prefix, path, suffix, sep='-'):
        """Internal: standardized filename for something wiith a path"""
        assert "." in suffix
        path_ref = sep.join([("%02d" % (n,)) for n in path])
        if not suffix.startswith("."):
            suffix = sep + suffix
        return "".join([prefix, sep, path_ref, suffix])


    def _save_rect_to_ora( self, orazip, tmpdir, prefix, path,
                           frame_bbox, rect, **kwargs ):
        """Internal: saves a rectangle of the surface to an ORA zip"""
        # Write PNG data via a tempfile
        pngname = self._make_refname(prefix, path, ".png")
        pngpath = os.path.join(tmpdir, pngname)
        t0 = time.time()
        self.save_as_png(pngpath, *rect, **kwargs)
        t1 = time.time()
        logger.debug('%.3fs surface saving %r', t1-t0, pngname)
        # Archive and remove
        storepath = "data/%s" % (pngname,)
        orazip.write(pngpath, storepath)
        os.remove(pngpath)
        # Return details
        elem = self._get_stackxml_element(frame_bbox, "layer")
        elem.attrib["src"] = storepath
        return elem


    ## Painting symmetry axis


    def set_symmetry_axis(self, center_x):
        """Sets the surface's painting symmetry axis"""
        if center_x is None:
            self._surface.set_symmetry_state(False, 0.0)
        else:
            self._surface.set_symmetry_state(True, center_x)


    ## Snapshots


    def save_snapshot(self):
        """Snapshots the state of the layer, for undo purposes"""
        return _SurfaceBackedLayerSnapshot(self)


    ## Trimming

    def get_trimmable(self):
        return True

    def trim(self, rect):
        """Trim the layer to a rectangle, discarding data outside it

        :param rect: A trimming rectangle in model coordinates
        :type rect: tuple (x, y, w, h)

        Only complete tiles are discarded by this method.
        """
        self._surface.trim(rect)


class _SurfaceBackedLayerSnapshot (_LayerBaseSnapshot):
    """Minimal layer implementation's snapshot

    Snapshots are stored in commands, and used to implement undo and redo.
    They must be independent copies of the data, although copy-on-write
    semantics are fine. Snapshot objects don't have to be _full and exact_
    clones of the layer's data, but they do need to capture _inherent_
    qualities of the layer. Mere metadata can be ignored. For the base
    layer implementation, this means the surface tiles and the layer's
    opacity.
    """

    def __init__(self, layer):
        super(_SurfaceBackedLayerSnapshot, self).__init__(layer)
        self.surface_sshot = layer._surface.save_snapshot()

    def restore_to_layer(self, layer):
        super(_SurfaceBackedLayerSnapshot, self).restore_to_layer(layer)
        layer._surface.load_snapshot(self.surface_sshot)


class BackgroundLayer (SurfaceBackedLayer):
    """Background layer, with a repeating tiled image

    By convention only, there is just a single non-editable background
    layer in any document, hidden behind an API in the document's
    RootLayerStack. In the MyPaint application, the working document's
    background layer cannot be manipulated by the user except through
    the background dialog.
    """

    # This could be generalized as a repeating tile for general use in
    # the layers stack, extending the ExternalLayer concept.  Think
    # textures!

    def __init__(self, bg, **kwargs):
        if isinstance(bg, tiledsurface.Background):
            surface = bg
        else:
            surface = tiledsurface.Background(bg)
        super(BackgroundLayer, self).__init__(name=u"background",
                                              surface=surface, **kwargs)
        self.locked = False
        self.visible = True
        self.opacity = 1.0

    def copy(self):
        raise NotImplementedError, "BackgroundLayer cannot be copied yet"

    def save_snapshot(self):
        raise NotImplementedError, "BackgroundLayer cannot be snapshotted yet"

    def load_snapshot(self):
        raise NotImplementedError, "BackgroundLayer cannot be snapshotted yet"

    def set_surface(self, surface):
        """Sets the surface from a tiledsurface.Background"""
        assert isinstance(surface, tiledsurface.Background)
        self._surface = surface

    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        # Save as a regular layer for other apps.
        # Background surfaces repeat, so this will fit the frame.
        # XXX But we use the canvas bbox and always have. Why?
        # XXX - Presumably it's for origin alignment.
        # XXX - Inefficient for small frames.
        # XXX - I suspect rect should be redone with (w,h) granularity
        # XXX   and be based on the frame bbox.
        rect = canvas_bbox
        elem = super(BackgroundLayer, self)\
            ._save_rect_to_ora( orazip, tmpdir, "background", path,
                                frame_bbox, rect, **kwargs )
        # Also save as single pattern (with corrected origin)
        x0, y0 = frame_bbox[0:2]
        x, y, w, h = self.get_bbox()
        rect = (x+x0, y+y0, w, h)

        pngname = self._make_refname("background", path, "tile.png")
        tmppath = os.path.join(tmpdir, pngname)
        t0 = time.time()
        self._surface.save_as_png(tmppath, *rect, **kwargs)
        t1 = time.time()
        storename = 'data/%s' % (pngname,)
        logger.debug('%.3fs surface saving %s', t1 - t0, storename)
        orazip.write(tmppath, storename)
        os.remove(tmppath)
        elem.attrib['background_tile'] = storename
        return elem


class ExternalLayer (SurfaceBackedLayer):
    """A layer which is stored as a tempfile in a non-MyPaint format

    External layers add the name of the tempfile to the base implementation.
    The internal surface is used to display a bitmap preview of the layer, but
    this cannot be edited.
    """

    ## Class constants

    IS_FILLABLE = False
    IS_PAINTABLE = False
    ALLOWED_SUFFIXES = []


    ## Construction

    def __init__(self, **kwargs):
        """Construct, with blank internal fields"""
        super(ExternalLayer, self).__init__(**kwargs)
        self._basename = None
        self._workdir = None
        self._x = None
        self._y = None

    def set_workdir(self, workdir):
        """Sets the working directory (i.e. to doc's tempdir)

        This is where working copies are created and cached during operation.
        """
        self._workdir = workdir


    def load_from_openraster(self, orazip, elem, tempdir, feedback_cb,
                             x=0, y=0, **kwargs):
        """Loads layer data and attrs from an OpenRaster zipfile

        Using this method also sets the working directory for the layer to
        tempdir.
        """
        # Load layer flags and raster data
        super(ExternalLayer, self) \
            .load_from_openraster(orazip, elem, tempdir, feedback_cb,
                                  x=x, y=y, extract_and_keep=True, **kwargs)
        # Use the extracted file as the zero revision, and record layer
        # working parameters.
        attrs = elem.attrib
        src = attrs.get("src", None)
        src_rootname, src_ext = os.path.splitext(src)
        src_ext = src_ext.lower()
        tmp_filename = os.path.join(tempdir, src)
        if not os.path.exists(tmp_filename):
            raise LoadError, ("tmpfile missing after extract_and_keep: %r" %
                              (tmp_filename,))
        rev0_fd, rev0_filename = tempfile.mkstemp(suffix=src_ext, dir=tempdir)
        os.close(rev0_fd)
        os.rename(tmp_filename, rev0_filename)
        self._basename = os.path.basename(rev0_filename)
        self._workdir = tempdir
        self._x = x + int(attrs.get('x', 0))
        self._y = y + int(attrs.get('y', 0))


    ## Snapshots

    def save_snapshot(self):
        """Snapshots the state of the layer and its strokemap for undo"""
        return _ExternalLayerSnapshot(self)

    ## Moving


    def get_move(self, x, y):
        """Start a new move for the external layer"""
        surface_move = super(ExternalLayer, self).get_move(x, y)
        return ExternalLayerMove(self, surface_move)


    ## Trimming (no-op for external layers)

    def get_trimmable(self):
        return False

    def trim(self, rect):
        """Override: external layers have no useful trim(), so do nothing"""
        pass


    ## Saving


    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        """Saves the working file to an OpenRaster zipfile"""
        # No supercall in this override, but the base implementation's
        # attributes method is useful.
        elem = self._get_stackxml_element(frame_bbox, "layer")
        attrs = elem.attrib
        # Store the managed layer position rather than one based on the
        # surface's tiles bbox, however.
        x0, y0 = frame_bbox[0:2]
        attrs["x"] = str(self._x - x0)
        attrs["y"] = str(self._y - y0)
        # Pick a suitable name to store under.
        src_path = os.path.join(self._workdir, self._basename)
        src_rootname, src_ext = os.path.splitext(src_path)
        src_ext = src_ext.lower()
        storename = self._make_refname("layer", path, src_ext)
        storepath = "data/%s" % (storename,)
        # Archive (but do not remove) the managed tempfile
        t0 = time.time()
        orazip.write(src_path, storepath)
        t1 = time.time()
        # Return details of what was written.
        attrs["src"] = unicode(storepath)
        return elem


class _ExternalLayerSnapshot (_SurfaceBackedLayerSnapshot):
    """Snapshot subclass for external layers"""

    def __init__(self, layer):
        super(_ExternalLayerSnapshot, self).__init__(layer)
        self.basename = self._copy_working_file( layer._basename,
                                                 layer._workdir )
        self.workdir = layer._workdir
        self.x = layer._x
        self.y = layer._y

    def restore_to_layer(self, layer):
        super(_ExternalLayerSnapshot, self).restore_to_layer(layer)
        layer._basename = self._copy_working_file( self.basename,
                                                   self.workdir )
        layer._workdir = self.workdir
        layer._x = self.x
        layer._y = self.y

    @staticmethod
    def _copy_working_file(old_basename, workdir):
        old_filename = os.path.join(workdir, old_basename)
        rootname, ext = os.path.splitext(old_basename)
        old_fp = open(old_filename, 'rb')
        new_fp = tempfile.NamedTemporaryFile( dir=workdir,
                                              suffix=ext, mode="w+b",
                                              delete=False )
        shutil.copyfileobj(old_fp, new_fp)
        new_basename = os.path.basename(new_fp.name)
        logger.debug( "Copied %r to %r within %r...",
                      old_basename, new_basename, workdir )
        new_fp.close()
        old_fp.close()
        return new_basename

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        sshot_copy = os.path.join(self.workdir, self.basename)
        if os.path.exists(sshot_copy):
            logger.debug("Cleanup: removing %r from %r",
                         self.basename, self.workdir)
            os.remove(sshot_copy)
        else:
            logger.debug("Cleanup: %r was already removed from %r",
                         self.basename, self.workdir)


class ExternalLayerMove (object):
    """Move object wrapper for external layers"""

    def __init__(self, layer, surface_move):
        super(ExternalLayerMove, self).__init__()
        self._wrapped = surface_move
        self._layer = layer
        self._start_x = layer._x
        self._start_y = layer._y

    def update(self, dx, dy):
        self._layer._x = int(round(self._start_x + dx))
        self._layer._y = int(round(self._start_y + dy))
        self._wrapped.update(dx, dy)

    def cleanup(self):
        self._wrapped.cleanup()

    def process(self, n=200):
        return self._wrapped.process(n)



class VectorLayer (ExternalLayer):
    """SVG-based vector layer"""

    #TRANSLATORS: Short default name for vector (SVG/Inkscape) layers
    DEFAULT_NAME = _(u"Vector Layer")

    ALLOWED_SUFFIXES = [".svg"]

    # activate_layertype_action() should invoke inkscape. Modally?

    def get_icon_name(self):
        return "mypaint-layer-vector-symbolic"


class PaintingLayer (SurfaceBackedLayer):
    """A paintable, bitmap layer

    Painting layers add a strokemap to the base implementation. The
    stroke map is a stack of `strokemap.StrokeShape` objects in painting
    order, allowing strokes and their associated brush and color
    information to be picked from the canvas.
    """

    ## Class constants

    IS_PAINTABLE = True
    IS_FILLABLE = True
    ALLOWED_SUFFIXES = [".png"]

    #TRANSLATORS: Default name for new normal, paintable layers
    DEFAULT_NAME = _(u"Layer")


    ## Initializing & resetting


    def __init__(self, **kwargs):
        super(PaintingLayer, self).__init__(**kwargs)
        #: Stroke map.
        #: List of strokemap.StrokeShape instances (not stroke.Stroke), ordered
        #: by depth.
        self.strokes = []


    def clear(self):
        """Clear both the surface and the strokemap"""
        super(PaintingLayer, self).clear()
        self.strokes = []


    def load_from_surface(self, surface):
        """Load the surface image's tiles from another surface"""
        super(PaintingLayer, self).load_from_surface(surface)
        self.strokes = []


    def load_from_openraster(self, orazip, elem, tempdir, feedback_cb,
                             x=0, y=0, **kwargs):
        """Loads layer flags, PNG data, and strokemap from a .ora zipfile"""
        # Load layer flags
        super(PaintingLayer, self) \
            .load_from_openraster(orazip, elem, tempdir, feedback_cb,
                                  x=x, y=y, allowed_suffixes=[".png"],
                                  **kwargs)
        # Strokemap too
        attrs = elem.attrib
        x += int(attrs.get('x', 0))
        y += int(attrs.get('y', 0))
        strokemap_name = attrs.get('mypaint_strokemap_v2', None)
        if strokemap_name is not None:
            t2 = time.time()
            sio = StringIO(orazip.read(strokemap_name))
            self.load_strokemap_from_file(sio, x, y)
            sio.close()
            t3 = time.time()
            logger.debug('%.3fs loading strokemap %r',
                         t3 - t2, strokemap_name)


    ## Flood fill

    def flood_fill(self, x, y, color, bbox, tolerance, dst_layer=None):
        """Fills a point on the surface with a colour

        :param x: Starting point X coordinate
        :param y: Starting point Y coordinate
        :param color: an RGB color
        :type color: tuple
        :param bbox: Bounding box: limits the fill
        :type bbox: lib.helpers.Rect or equivalent 4-tuple
        :param tolerance: how much filled pixels are permitted to vary
        :type tolerance: float [0.0, 1.0]
        :param dst_layer: Optional target layer (default is self!)
        :type dst_layer: SurfaceBackedLayer

        The `tolerance` parameter controls how much pixels are permitted to
        vary from the starting colour.  We use the 4D Euclidean distance from
        the starting point to each pixel under consideration as a metric,
        scaled so that its range lies between 0.0 and 1.0.

        The default target layer is `self`. This method invalidates the filled
        area of the target layer's surface, queueing a redraw if it is part of
        a visible document.
        """
        if dst_layer is None:
            dst_layer = self
        self._surface.flood_fill(x, y, color, bbox, tolerance,
                                 dst_surface=dst_layer._surface)


    ## Painting


    def stroke_to(self, brush, x, y, pressure, xtilt, ytilt, dtime):
        """Render a part of a stroke to the canvas surface

        :param brush: The brush to use for rendering dabs
        :type brush: lib.brush.Brush
        :param x: Input event's X coord, translated to document coords
        :param y: Input event's Y coord, translated to document coords
        :param pressure: Input event's pressure
        :param xtilt: Input event's tilt component in the document X direction
        :param ytilt: Input event's tilt component in the document Y direction
        :param dtime: Time delta, in seconds
        :returns: whether the stroke should now be split
        :rtype: bool

        This method renders zero or more dabs to the surface of this layer,
        but does not affect the strokemap. Use this for the incremental
        painting of segments of a stroke sorresponding to single input events.
        The return value decides whether to finalize the lib.stroke.Stroke
        which is currently recording the user's input, and begin recording a
        new one.
        """
        self._surface.begin_atomic()
        split = brush.stroke_to(self._surface.backend, x, y,
                                    pressure, xtilt, ytilt, dtime)
        self._surface.end_atomic()
        return split

    def render_stroke(self, stroke):
        """Render a whole captured stroke to the canvas

        :param stroke: The stroke to render
        :type stroke: lib.stroke.Stroke
        """
        stroke.render(self._surface)

    def add_stroke_shape(self, stroke, before):
        """Adds a rendered stroke's shape to the strokemap

        :param stroke: the stroke sequence which has been rendered
        :type stroke: lib.stroke.Stroke
        :param before: layer snapshot taken before the stroke started
        :type before: lib.layer._PaintingLayerSnapshot

        The StrokeMap is a stack of lib.strokemap.StrokeShape objects which
        encapsulate the shape of a rendered stroke, and the brush settings
        which were used to render it.  The shape of the rendered stroke is
        determined by visually diffing snapshots taken before the stroke
        started and now.
        """
        shape = strokemap.StrokeShape()
        after_sshot = self._surface.save_snapshot()
        shape.init_from_snapshots(before.surface_sshot, after_sshot)
        shape.brush_string = stroke.brush_settings
        self.strokes.append(shape)


    ## Snapshots


    def save_snapshot(self):
        """Snapshots the state of the layer and its strokemap for undo"""
        return _PaintingLayerSnapshot(self)


    ## Translating

    def get_move(self, x, y):
        """Get an interactive move object for the surface and its strokemap"""
        surface_move = super(PaintingLayer, self).get_move(x, y)
        return PaintingLayerMove(self, surface_move)


    ## Trimming


    def trim(self, rect):
        """Trim the layer and its strokemap"""
        super(PaintingLayer, self).trim(rect)
        empty_strokes = []
        for stroke in self.strokes:
            if not stroke.trim(rect):
                empty_strokes.append(stroke)
        for stroke in empty_strokes:
            logger.debug("Removing emptied stroke %r", stroke)
            self.strokes.remove(stroke)

    ## Strokemap


    def load_strokemap_from_file(self, f, translate_x, translate_y):
        assert not self.strokes
        brushes = []
        N = tiledsurface.N
        x = int(translate_x//N) * N
        y = int(translate_y//N) * N
        dx = translate_x % N
        dy = translate_y % N
        while True:
            t = f.read(1)
            if t == 'b':
                length, = struct.unpack('>I', f.read(4))
                tmp = f.read(length)
                brushes.append(zlib.decompress(tmp))
            elif t == 's':
                brush_id, length = struct.unpack('>II', f.read(2*4))
                stroke = strokemap.StrokeShape()
                tmp = f.read(length)
                stroke.init_from_string(tmp, x, y)
                stroke.brush_string = brushes[brush_id]
                # Translate non-aligned strokes
                if (dx, dy) != (0, 0):
                    stroke.translate(dx, dy)
                self.strokes.append(stroke)
            elif t == '}':
                break
            else:
                assert False, 'invalid strokemap'


    def get_stroke_info_at(self, x, y):
        """Get the stroke at the given point"""
        x, y = int(x), int(y)
        for s in reversed(self.strokes):
            if s.touches_pixel(x, y):
                return s


    def get_last_stroke_info(self):
        if not self.strokes:
            return None
        return self.strokes[-1]


    ## Saving

    @staticmethod
    def _write_file_str(z, filename, data):
        """Helper: write data to a zipfile with the right permissions"""
        # Work around a permission bug in the zipfile library:
        # http://bugs.python.org/issue3394
        zi = zipfile.ZipInfo(filename)
        zi.external_attr = 0100644 << 16
        z.writestr(zi, data)

    def _save_strokemap_to_file(self, f, translate_x, translate_y):
        brush2id = {}
        for stroke in self.strokes:
            s = stroke.brush_string
            # save brush (if not already known)
            if s not in brush2id:
                brush2id[s] = len(brush2id)
                s = zlib.compress(s)
                f.write('b')
                f.write(struct.pack('>I', len(s)))
                f.write(s)
            # save stroke
            s = stroke.save_to_string(translate_x, translate_y)
            f.write('s')
            f.write(struct.pack('>II', brush2id[stroke.brush_string], len(s)))
            f.write(s)
        f.write('}')


    def save_to_openraster(self, orazip, tmpdir, path,
                           canvas_bbox, frame_bbox, **kwargs):
        """Save the strokemap too, in addition to the base implementation"""
        # Save the layer normally

        elem = super(PaintingLayer, self)\
            .save_to_openraster( orazip, tmpdir, path,
                                 canvas_bbox, frame_bbox, **kwargs )
        # Store stroke shape data too
        x, y, w, h = self.get_bbox()
        sio = StringIO()
        t0 = time.time()
        self._save_strokemap_to_file(sio, -x, -y)
        t1 = time.time()
        data = sio.getvalue()
        sio.close()
        datname = self._make_refname("layer", path, "strokemap.dat")
        logger.debug("%.3fs strokemap saving %r", t1-t0, datname)
        storepath = "data/%s" % (datname,)
        self._write_file_str(orazip, storepath, data)
        # Return details
        elem.attrib['mypaint_strokemap_v2'] = storepath
        return elem


class _PaintingLayerSnapshot (_SurfaceBackedLayerSnapshot):
    """Snapshot subclass for painting layers"""

    def __init__(self, layer):
        super(_PaintingLayerSnapshot, self).__init__(layer)
        self.strokes = layer.strokes[:]

    def restore_to_layer(self, layer):
        super(_PaintingLayerSnapshot, self).restore_to_layer(layer)
        layer.strokes = self.strokes[:]


class PaintingLayerMove (object):
    """Move object wrapper for painting layers"""

    def __init__(self, layer, surface_move):
        super(PaintingLayerMove, self).__init__()
        self._wrapped = surface_move
        self._layer = layer
        self._final_dx = 0
        self._final_dy = 0

    def update(self, dx, dy):
        self._final_dx = dx
        self._final_dy = dy
        return self._wrapped.update(dx, dy)

    def cleanup(self):
        result = self._wrapped.cleanup()
        dx = self._final_dx
        dy = self._final_dy
        # Arrange for the strokemap to be moved too;
        # this happens in its own background idler.
        for stroke in self._layer.strokes:
            stroke.translate(dx, dy)
            # Minor problem: huge strokemaps take a long time to move, and the
            # translate must be forced to completion before drawing or any
            # further layer moves. This can cause apparent hangs for no
            # reason later on. Perhaps it would be better to process them
            # fully in this hourglass-cursor phase after all?

    def process(self, n=200):
        return self._wrapped.process(n)


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


## Helper functions


_LAYER_NEW_CLASSES = [LayerStack, PaintingLayer, VectorLayer]


def layer_new_from_openraster(orazip, elem, tempdir, feedback_cb,
                              root, x=0, y=0, **kwargs):
    """Construct and return a new layer from a .ora file (factory)"""
    for layer_class in _LAYER_NEW_CLASSES:
        try:
            return layer_class.new_from_openraster(orazip, elem, tempdir,
                                                   feedback_cb, root,
                                                   x=x, y=y, **kwargs)
        except LoadError:
            pass
    raise LoadError, "No delegate class willing to load %r" % (elem,)


def pixbuf_from_stream(fp, feedback_cb=None):
    """Extract and return a GdkPixbuf from file-like object"""
    loader = GdkPixbuf.PixbufLoader()
    while True:
        if feedback_cb is not None:
            feedback_cb()
        buf = fp.read(LOAD_CHUNK_SIZE)
        if buf == '':
            break
        loader.write(buf)
    loader.close()
    return loader.get_pixbuf()


def pixbuf_from_zipfile(datazip, filename, feedback_cb=None):
    """Extract and return a GdkPixbuf from a zipfile entry"""
    try:
        datafp = datazip.open(filename, mode='r')
    except KeyError:
        # Support for bad zip files (saved by old versions of the
        # GIMP ORA plugin)
        datafp = datazip.open(filename.encode('utf-8'), mode='r')
        logger.warning('Bad ZIP file. There is an utf-8 encoded '
                       'filename that does not have the utf-8 '
                       'flag set: %r', filename)
    pixbuf = pixbuf_from_stream(datafp, feedback_cb=feedback_cb)
    datafp.close()
    return pixbuf


## Module testing


def _test():
    """Run doctest strings"""
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)


def _make_test_stack():
    """Makes a simple test RootLayerStack (2 branches of 3 leaves each)

    :return: The root stack, and a list of its leaves.
    :rtype: tuple
    """
    root = RootLayerStack(doc=None)
    layer0 = LayerStack(name='0'); root.append(layer0)
    layer00 = PaintingLayer(name='00'); layer0.append(layer00)
    layer01 = PaintingLayer(name='01'); layer0.append(layer01)
    layer02 = PaintingLayer(name='02'); layer0.append(layer02)
    layer1 = LayerStack(name='1'); root.append(layer1)
    layer10 = PaintingLayer(name='10'); layer1.append(layer10)
    layer11 = PaintingLayer(name='11'); layer1.append(layer11)
    layer12 = PaintingLayer(name='12'); layer1.append(layer12)
    return (root, [layer00,layer01,layer02, layer10,layer11,layer12])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _test()
