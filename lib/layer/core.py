# This file is part of MyPaint.
# Copyright (C) 2011-2018 by the MyPaint Development Team.
# Copyright (C) 2007-2012 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Core layer classes etc."""


## Imports

import logging
import os
import xml.etree.ElementTree as ET
import weakref
from warnings import warn
import abc

from lib.gettext import C_
import lib.mypaintlib
import lib.strokemap
import lib.helpers as helpers
import lib.fileutils
import lib.pixbuf
from lib.modes import PASS_THROUGH_MODE
from lib.modes import STANDARD_MODES
from lib.modes import ORA_MODES_BY_OPNAME
from lib.modes import MODES_EFFECTIVE_AT_ZERO_ALPHA
from lib.modes import MODES_DECREASING_BACKDROP_ALPHA
import lib.modes
import lib.xml
import lib.tiledsurface
from .rendering import Renderable

logger = logging.getLogger(__name__)


## Base class defs


class LayerBase(Renderable):
    """Base class defining the layer API
    
    Layers support the Renderable interface, and are rendered with the
    "render_*()" methods of their root layer stack.
    
    Layers are minimally aware of the tree structure they reside in, in
    that they contain a reference to the root of their tree for
    signalling purposes.  Updates to the tree structure and to layers'
    graphical contents are announced via the RootLayerStack object
    representing the base of the tree.

    Args:

    Returns:

    Raises:

    """

    ## Class constants

    #: Forms the default name, may be suffixed per lib.naming consts.
    DEFAULT_NAME = C_(
        "layer default names",
        "Layer",
    )

    #: A string for the layer type.
    TYPE_DESCRIPTION = None

    PERMITTED_MODES = set(STANDARD_MODES)

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
        self._mode = lib.modes.default_mode()
        self._group_ref = None
        self._root_ref = None
        self._thumbnail = None
        #: True if the layer was marked as selected when loaded.
        self.initially_selected = False

    @classmethod
    def new_from_openraster(
        cls, orazip, elem, cache_dir, progress, root, x=0, y=0, **kwargs
    ):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Reads and returns a layer from an OpenRaster zipfile
        
        This implementation just creates a new instance of its class and
        calls `load_from_openraster()` on it. This should suffice for
        all subclasses which support parameterless construction.

        Args:
            orazip: 
            elem: 
            cache_dir: 
            progress: 
            root: 
            x:  (Default value = 0)
            y:  (Default value = 0)
            **kwargs: 

        Returns:

        Raises:

        """

        layer = cls()
        layer.load_from_openraster(
            orazip, elem, cache_dir, progress, x=x, y=y, **kwargs
        )
        return layer

    @classmethod
    def new_from_openraster_dir(
        cls, oradir, elem, cache_dir, progress, root, x=0, y=0, **kwargs
    ):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Reads and returns a layer from an OpenRaster-like folder
        
        This implementation just creates a new instance of its class and
        calls `load_from_openraster_dir()` on it. This should suffice
        for all subclasses which support parameterless construction.

        Args:
            oradir: 
            elem: 
            cache_dir: 
            progress: 
            root: 
            x:  (Default value = 0)
            y:  (Default value = 0)
            **kwargs: 

        Returns:

        Raises:

        """
        layer = cls()
        layer.load_from_openraster_dir(
            oradir, elem, cache_dir, progress, x=x, y=y, **kwargs
        )
        return layer

    def load_from_openraster(
        self, orazip, elem, cache_dir, progress, x=0, y=0, **kwargs
    ):
        """Loads layer data from an open OpenRaster zipfile

        Args:
            orazip (zipfile.ZipFile): An OpenRaster zipfile, opened for extracting
            elem (xml.etree.ElementTree.Element): <layer/> or <stack/> element to load (stack.xml)
            cache_dir: Cache root dir for this document
            progress (lib.feedback.Progress or None): Provides feedback to the user.
            x: X offset of the top-left point for image data (Default value = 0)
            y: Y offset of the top-left point for image data (Default value = 0)
            **kwargs: Extensibility
        
        The base implementation loads the common layer flags from a `<layer/>`
        or `<stack/>` element, but does nothing more than that. Loading layer
        data from the zipfile or recursing into stack contents is deferred to
        subclasses.

        Returns:

        Raises:

        """
        self._load_common_flags_from_ora_elem(elem)

    def load_from_openraster_dir(
        self, oradir, elem, cache_dir, progress, x=0, y=0, **kwargs
    ):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Loads layer data from an OpenRaster-style folder.
        
        Parameters are the same as for load_from_openraster, with the
        following exception (replacing ``orazip``):

        Args:
            oradir: Folder with a .ORA-like tree structure.
            elem: 
            cache_dir: 
            progress: 
            x:  (Default value = 0)
            y:  (Default value = 0)
            **kwargs: 

        Returns:

        Raises:

        """
        self._load_common_flags_from_ora_elem(elem)

    def _load_common_flags_from_ora_elem(self, elem):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """

        Args:
            elem: 

        Returns:

        Raises:

        """
        attrs = elem.attrib
        self.name = str(attrs.get("name", ""))
        compop = str(attrs.get("composite-op", ""))
        self.mode = ORA_MODES_BY_OPNAME.get(compop, lib.modes.default_mode())
        self.opacity = helpers.clamp(float(attrs.get("opacity", "1.0")), 0.0, 1.0)
        visible = attrs.get("visibility", "visible").lower()
        self.visible = visible != "hidden"
        locked = attrs.get("edit-locked", "false").lower()
        self.locked = lib.xml.xsd2bool(locked)
        selected = attrs.get("selected", "false").lower()
        self.initially_selected = lib.xml.xsd2bool(selected)

    def __deepcopy__(self, memo):
        """Returns an independent copy of the layer, for Duplicate Layer

        >>> from copy import deepcopy
        >>> orig = _StubLayerBase()
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
    def group(self):
        """The group of the current layer.
        
        Returns None if the layer is not in a group.

        Args:

        Returns:

        Raises:

        >>> from . import group
        >>> outer = group.LayerStack()
        >>> inner = group.LayerStack()
        >>> scribble = _StubLayerBase()
        >>> outer.append(inner)
        >>> inner.append(scribble)
        >>> outer.group is None
        True
        >>> inner.group == outer
        True
        >>> scribble.group == inner
        True
        """
        if self._group_ref is not None:
            return self._group_ref()
        return None

    @group.setter
    def group(self, group):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """

        Args:
            group: 

        Returns:

        Raises:

        """
        if group is None:
            self._group_ref = None
        else:
            self._group_ref = weakref.ref(group)

    @property
    def root(self):
        """The root of the layer tree structure
        
        Only RootLayerStack instances or None are permitted.
        You won't normally need to adjust this unless you're doing
        something fancy: it's automatically maintained by intermediate
        and root `LayerStack` elements in the tree whenever layers are
        added or removed from a rooted tree structure.

        Args:

        Returns:

        Raises:

        >>> from . import tree
        >>> root = tree.RootLayerStack(doc=None)
        >>> layer = _StubLayerBase()
        >>> root.append(layer)
        >>> layer.root                 #doctest: +ELLIPSIS
        <RootLayerStack...>
        >>> layer.root is root
        True
        """
        if self._root_ref is not None:
            return self._root_ref()
        return None

    @root.setter
    def root(self, newroot):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """

        Args:
            newroot: 

        Returns:

        Raises:

        """
        if newroot is None:
            self._root_ref = None
        else:
            self._root_ref = weakref.ref(newroot)

    @property
    def opacity(self):
        """Opacity multiplier for the layer.
        
        Values must permit conversion to a `float` in [0, 1].
        Changing this property issues ``layer_properties_changed`` and
        appropriate ``layer_content_changed`` notifications via the root
        layer stack if the layer is within a tree structure.
        
        Layers with a `mode` of `PASS_THROUGH_MODE` have immutable
        opacities: the value is always 100%. This restriction only
        applies to `LayerStack`s - i.e. layer groups - because those are
        the only kinds of layer which can be put into pass-through mode.

        Args:

        Returns:

        Raises:

        """
        return self._opacity

    @opacity.setter
    def opacity(self, opacity):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """

        Args:
            opacity: 

        Returns:

        Raises:

        """
        opacity = helpers.clamp(float(opacity), 0.0, 1.0)
        if opacity == self._opacity:
            return
        if self.mode == PASS_THROUGH_MODE:
            warn(
                "Cannot change the change the opacity multiplier "
                "of a layer group in PASS_THROUGH_MODE",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self._opacity = opacity
        self._properties_changed(["opacity"])
        # Note: not the full_redraw_bbox here.
        # Changing a layer's opacity multiplier alone cannot change the
        # calculated alpha of an outlying empty tile in the layer.
        # Those are always zero. Even if the layer has a fancy masking
        # mode, that won't affect redraws arising from mere opacity
        # multiplier updates.
        bbox = tuple(self.get_bbox())
        self._content_changed(*bbox)

    @property
    def name(self):
        """The layer's name, for display purposes
        
        Values must permit conversion to a str.  If the
        layer is part of a tree structure, ``layer_properties_changed``
        notifications will be issued via the root layer stack. In
        addition, assigned names may be corrected to be unique within
        the tree.

        Args:

        Returns:

        Raises:

        """
        return self._name

    @name.setter
    def name(self, name):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """

        Args:
            name: 

        Returns:

        Raises:

        """
        if name is not None:
            name = str(name)
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
        """Whether the layer has a visible effect on its backdrop.
        
        Some layer modes normally have an effect even if the calculated
        alpha of a pixel is zero. This switch turns that off too.
        
        Values must permit conversion to a `bool`.
        Changing this property issues ``layer_properties_changed`` and
        appropriate ``layer_content_changed`` notifications via the root
        layer stack if the layer is within a tree structure.

        Args:

        Returns:

        Raises:

        """
        return self._visible

    @visible.setter
    def visible(self, visible):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """

        Args:
            visible: 

        Returns:

        Raises:

        """
        visible = bool(visible)
        if visible == self._visible:
            return
        self._visible = visible
        self._properties_changed(["visible"])
        # Toggling the visibility flag always causes the mode to stop
        # or start having its normal effect. Need the full redraw bbox
        # so that outlying empty tiles will be updated properly.
        bbox = tuple(self.get_full_redraw_bbox())
        self._content_changed(*bbox)

    @property
    def branch_visible(self):
        """Check whether the layer's branch is visible.
        
        Returns True if the layer's group and all of its parents are visible,
        False otherwise.
        
        Returns True if the layer is not in a group.

        Args:

        Returns:

        Raises:

        >>> from . import group
        >>> outer = group.LayerStack()
        >>> inner = group.LayerStack()
        >>> scribble = _StubLayerBase()
        >>> outer.append(inner)
        >>> inner.append(scribble)
        >>> outer.branch_visible
        True
        >>> inner.branch_visible
        True
        >>> scribble.branch_visible
        True
        >>> outer.visible = False
        >>> outer.branch_visible
        True
        >>> inner.branch_visible
        False
        >>> scribble.branch_visible
        False
        """
        group = self.group
        if group is None:
            return True

        return group.visible and group.branch_visible

    @property
    def locked(self):
        """Whether the layer is locked (immutable).
        
        Values must permit conversion to a `bool`.
        Changing this property issues `layer_properties_changed` via the
        root layer stack if the layer is within a tree structure.

        Args:

        Returns:

        Raises:

        """
        return self._locked

    @locked.setter
    def locked(self, locked):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """

        Args:
            locked: 

        Returns:

        Raises:

        """
        locked = bool(locked)
        if locked != self._locked:
            self._locked = locked
            self._properties_changed(["locked"])

    @property
    def branch_locked(self):
        """Check whether the layer's branch is locked.
        
        Returns True if the layer's group or at least one of its parents
        is locked, False otherwise.
        
        Returns False if the layer is not in a group.

        Args:

        Returns:

        Raises:

        >>> from . import group
        >>> outer = group.LayerStack()
        >>> inner = group.LayerStack()
        >>> scribble = _StubLayerBase()
        >>> outer.append(inner)
        >>> inner.append(scribble)
        >>> outer.branch_locked
        False
        >>> inner.branch_locked
        False
        >>> scribble.branch_locked
        False
        >>> outer.locked = True
        >>> outer.branch_locked
        False
        >>> inner.branch_locked
        True
        >>> scribble.branch_locked
        True
        """
        group = self.group
        if group is None:
            return False

        return group.locked or group.branch_locked

    @property
    def mode(self):
        """How this layer combines with its backdrop.
        
        Values must permit conversion to an int, and must be permitted
        for the mode's class.
        
        Changing this property issues ``layer_properties_changed`` and
        appropriate ``layer_content_changed`` notifications via the root
        layer stack if the layer is within a tree structure.
        
        In addition to the modes supported by the base implementation,
        layer groups permit `lib.modes.PASS_THROUGH_MODE`, an
        additional mode where group contents are rendered as if their
        group were not present. Setting the mode to this value also
        sets the opacity to 100%.
        
        For layer groups, "Normal" mode implies group isolation
        internally. These semantics differ from those of OpenRaster and
        the W3C, but saving and loading applies the appropriate
        transformation.
        
        See also: PERMITTED_MODES.

        Args:

        Returns:

        Raises:

        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """

        Args:
            mode: 

        Returns:

        Raises:

        """
        mode = int(mode)
        if mode not in self.PERMITTED_MODES:
            mode = lib.modes.default_mode()
        if mode == self._mode:
            return
        # Forcing the opacity for layer groups here allows a redraw to
        # be subsumed. Only layer groups permit PASS_THROUGH_MODE.
        propchanges = []
        if mode == PASS_THROUGH_MODE:
            self._opacity = 1.0
            propchanges.append("opacity")
        # When changing the mode, the before and after states may have
        # different treatments of outlying empty tiles. Need the full
        # redraw bboxes of both states to ensure correct redraws.
        redraws = [self.get_full_redraw_bbox()]
        self._mode = mode
        redraws.append(self.get_full_redraw_bbox())
        self._content_changed(*tuple(combine_redraws(redraws)))
        propchanges.append("mode")
        self._properties_changed(propchanges)

    ## Notifications

    def _content_changed(self, *args):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Notifies the root's content observers
        
        If this layer's root stack is defined, i.e. if it is part of a
        tree structure, the root's `layer_content_changed()` event
        method will be invoked with this layer and the supplied

        Args:
            *args: 

        Returns:

        Raises:

        """
        root = self.root
        if root is not None:
            root.layer_content_changed(self, *args)

    def _properties_changed(self, properties):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Notifies the root's layer properties observers
        
        If this layer's root stack is defined, i.e. if it is part of a
        tree structure, the root's `layer_properties_changed()` event
        method will be invoked with the layer and the supplied

        Args:
            properties: 

        Returns:

        Raises:

        """
        root = self.root
        if root is not None:
            root._notify_layer_properties_changed(self, set(properties))

    ## Info methods

    def get_icon_name(self):
        """The name of the icon to display for the layer
        
        Ideally symbolic. A value of `None` means that no icon should be
        displayed.

        Args:

        Returns:

        Raises:

        """
        return None

    @property
    def effective_opacity(self):
        """The opacity used when rendering a layer: zero if invisible
        
        This must match the appearance produced by the layer's
        Renderable.get_render_ops() implementation when it is called
        with no explicit "layers" specification. The base class's
        effective opacity is zero because the base get_render_ops() is
        unimplemented.

        Args:

        Returns:

        Raises:

        """
        return 0.0

    def get_alpha(self, x, y, radius):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Gets the average alpha within a certain radius at a point

        Args:
            x: model X coordinate
            y: model Y coordinate
            radius: radius over which to average
        :rtype: float
        
        The return value is not affected by the layer opacity, effective or
        otherwise. This is used by `Document.pick_layer()` and friends to test
        whether there's anything significant present at a particular point.
        The default alpha at a point is zero.

        Returns:

        Raises:

        """
        return 0.0

    def get_bbox(self):
        """Returns the inherent (data) bounding box of the layer"""
        return helpers.Rect()

    def get_full_redraw_bbox(self):
        """Gets the full update notification bounding box of the layer"""
        if self.mode in MODES_EFFECTIVE_AT_ZERO_ALPHA:
            return helpers.Rect()
        else:
            return self.get_bbox()

    def is_empty(self):
        """Tests whether the surface is empty
        
        Always true in the base implementation.

        Args:

        Returns:

        Raises:

        """
        return True

    def get_paintable(self):
        """True if this layer currently accepts painting brushstrokes
        
        Always false in the base implementation.

        Args:

        Returns:

        Raises:

        """
        return False

    def get_fillable(self):
        """True if this layer currently accepts flood fill
        
        Always false in the base implementation.

        Args:

        Returns:

        Raises:

        """
        return False

    def get_stroke_info_at(self, x, y):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Return the brushstroke at a given point

        Args:
            x: X coordinate to pick from, in model space.
            y: Y coordinate to pick from, in model space.
        :rtype: lib.strokemap.StrokeShape or None
        
        Returns None for the base class.

        Returns:

        Raises:

        """
        return None

    def get_last_stroke_info(self):
        """Return the most recently painted stroke"""
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

        Args:

        Returns:

        Raises:

        """
        name = self._name
        if name is None or name.strip() == "":
            return False
        if name == self.DEFAULT_NAME:
            return False
        match = lib.naming.UNIQUE_NAME_REGEX.match(name)
        if match is not None:
            base = match.group("name")
            if base == self.DEFAULT_NAME:
                return False
        return True

    ## Flood fill

    def flood_fill(self, fill_args, dst_layer=None):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Fills a point on the surface with a color
        
        See PaintingLayer.flood_fill() for parameters and semantics.
        The base implementation does nothing.

        Args:
            fill_args: 
            dst_layer:  (Default value = None)

        Returns:

        Raises:

        """
        pass

    ## Rendering

    def get_tile_coords(self):
        """Returns all data tiles in this layer

        Args:

        Returns:
            sequence

This method should return a sequence listing the coordinates for
all tiles with data in this layer.

It is used when computing layer merges.  Tile coordinates must
be returned as ``(tx, ty)`` pairs.

The base implementation returns an empty sequence.: All tiles with data

        Raises:

        """
        return []

    ## Translation

    def get_move(self, x, y):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Get a translation/move object for this layer

        Args:
            x: Model X position of the start of the move
            y: Model X position of the start of the move

        Returns:
            A move object

        Raises:

        """
        raise NotImplementedError

    def translate(self, dx, dy):
        # type: (Types.ELLIPSIS) -> list
        """Translate a layer non-interactively

        Args:
            dx: Horizontal offset in model coordinates
            dy: Vertical offset in model coordinates

The base implementation uses `get_move()` and the object it returns.: full redraw bboxes for the move: ``[before, after]``

        Raises:

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

    def __bool__(self):
        """Layers are never false.

        >>> sample = _StubLayerBase()
        >>> bool(sample)
        True

        """
        return True

    def __eq__(self, layer):
        """Two layers are only equal if they are the same object

        This is meaningful during layer repositions in the GUI, where
        shallow copies are used.
        """
        return self is layer

    def __hash__(self):
        """Return a hash for the layer (identity only)"""
        return id(self)

    ## Saving

    def save_as_png(self, filename, *rect, **kwargs):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Save to a named PNG file

        Args:
            filename: filename to save to
            *rect: rectangle to save, as a 4-tuple
            **kwargs: passthrough opts for underlying implementations
        :rtype: Gdk.Pixbuf
        
        The base implementation does nothing.

        Returns:

        Raises:

        """
        pass

    def save_to_openraster(
        self, orazip, tmpdir, path, canvas_bbox, frame_bbox, **kwargs
    ):
        """Saves the layer's data into an open OpenRaster ZipFile

        Args:
            orazip: a `zipfile.ZipFile` open for write
            tmpdir: path to a temp dir, removed after the save
            path (tuple of ints): Unique path of the layer, for encoding in filenames
            canvas_bbox (tuple): Bounding box of all layers, absolute coords
            frame_bbox (tuple): Bounding box of the image being saved
            **kwargs: Keyword args used by the save implementation

        Returns:
            xml.etree.ElementTree.Element

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
unaddressable background layer right now, for example.: element describing data written

        Raises:

        """
        raise NotImplementedError

    def _get_stackxml_element(self, tag, x=None, y=None):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Internal: get a basic etree Element for .ora saving

        Args:
            tag: 
            x:  (Default value = None)
            y:  (Default value = None)

        Returns:

        Raises:

        """

        elem = ET.Element(tag)
        attrs = elem.attrib
        if self.name:
            attrs["name"] = str(self.name)
        if x is not None:
            attrs["x"] = str(x)
        if y is not None:
            attrs["y"] = str(y)
        attrs["opacity"] = str(self.opacity)
        if self.initially_selected:
            attrs["selected"] = "true"
        if self.locked:
            attrs["edit-locked"] = "true"
        if self.visible:
            attrs["visibility"] = "visible"
        else:
            attrs["visibility"] = "hidden"
        # NOTE: This *will* be wrong for the PASS_THROUGH_MODE case.
        # NOTE: LayerStack will need to override this attr.
        mode_info = lib.mypaintlib.combine_mode_get_info(self.mode)
        if mode_info is not None:
            compop = mode_info.get("name")
            if compop is not None:
                attrs["composite-op"] = str(compop)
        return elem

    ## Painting symmetry axis

    def set_symmetry_state(self, active, center, symmetry_type, symmetry_lines, angle):
        """Set the surface's painting symmetry axis and active flag.

        Args:
            active (bool): Whether painting should be symmetrical.
            center (tuple): (x, y) coordinates of the center of symmetry
            symmetry_type (int): symmetry type that will be applied if active
            symmetry_lines (int): number of rotational
        symmetry lines for angle dependent symmetry modes.
            angle (float): The angle of the symmetry line(s)
        
        The symmetry axis is only meaningful to paintable layers.
        Received strokes are reflected along the line ``x=center_x``
        when symmetrical painting is active.
        
        This method is used by RootLayerStack only,
        propagating a central shared flag and value to all layers.
        
        The base implementation does nothing.

        Returns:

        Raises:

        """
        pass

    ## Snapshot

    def save_snapshot(self):
        """Snapshots the state of the layer, for undo purposes
        
        The returned data should be considered opaque, useful only as a
        memento to be restored with load_snapshot().

        Args:

        Returns:

        Raises:

        """
        return LayerBaseSnapshot(self)

    def load_snapshot(self, sshot):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Restores the layer from snapshot data

        Args:
            sshot: 

        Returns:

        Raises:

        """
        sshot.restore_to_layer(self)

    ## Thumbnails

    @property
    def thumbnail(self):
        """The layer's cached preview thumbnail."""
        return self._thumbnail

    def update_thumbnail(self):
        """Safely updates the cached preview thumbnail.
        
        This method updates self.thumbnail using render_thumbnail() and
        the data bounding box, and eats any NotImplementedErrors.
        
        This is used by the layer stack to keep the preview thumbnail up
        to date. It is called automatically after layer data is changed
        and stable for a bit, so there is normally no need to call it in
        client code.

        Args:

        Returns:

        Raises:

        """
        try:
            self._thumbnail = self.render_thumbnail(
                self.get_bbox(),
                alpha=True,
            )
        except NotImplementedError:
            self._thumbnail = None

    def render_thumbnail(self, bbox, **options):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """Renders a 256x256 thumb of the layer in an arbitrary bbox.

        Args:
            bbox: Bounding box to make a thumbnail of.
            **options: Passed to RootLayerStack.render_layer_preview().
        :rtype: GtkPixbuf or None
        
        Use the thumbnail property if you just want a reasonably
        up-to-date preview thumbnail for a single layer.
        
        See also: RootLayerStack.render_layer_preview().

        Returns:

        Raises:

        """
        root = self.root
        if root is None:
            return None
        return root.render_layer_preview(self, bbox=bbox, **options)

    ## Trimming

    def trim(self, rect):
        """Trim the layer to a rectangle, discarding data outside it

        Args:
            rect (tuple (x, y, w, h)): A trimming rectangle in model coordinates
        
        The base implementation does nothing.

        Returns:

        Raises:

        """
        pass


class _StubLayerBase(LayerBase):
    """An instantiable (but broken) LayerBase, for testing."""

    def get_render_ops(self, *argv, **kwargs):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """

        Args:
            *argv: 
            **kwargs: 

        Returns:

        Raises:

        """
        pass


class LayerBaseSnapshot:
    """Base snapshot implementation
    
    Snapshots are stored in commands, and used to implement undo and redo.
    They must be independent copies of the data, although copy-on-write
    semantics are fine. Snapshot objects must be complete enough clones of the
    layer's data for duplication to work.

    Args:

    Returns:

    Raises:

    """

    def __init__(self, layer):
        super(LayerBaseSnapshot, self).__init__()
        self.name = layer.name
        self.mode = layer.mode
        self.opacity = layer.opacity
        self.visible = layer.visible
        self.locked = layer.locked

    def restore_to_layer(self, layer):
        # type: (Types.ELLIPSIS) -> Types.NONE
        """

        Args:
            layer: 

        Returns:

        Raises:

        """
        layer.name = self.name
        layer.mode = self.mode
        layer.opacity = self.opacity
        layer.visible = self.visible
        layer.locked = self.locked


class ExternallyEditable(metaclass=abc.ABCMeta):
    """Interface for layers which can be edited in an external app"""

    _EDITS_SUBDIR = "edits"

    @abc.abstractmethod
    def new_external_edit_tempfile(self):
        """Get a tempfile for editing in an external app

        Args:

        Returns:
            Absolute path to a newly-created tempfile for editing
            
            The returned tempfiles are only expected to persist on disk
            until a subsequent call to this method is made.

        Raises:

        """

    @abc.abstractmethod
    def load_from_external_edit_tempfile(self, tempfile_path):
        # type: (str) -> Types.NONE
        """Load content from an external-edit tempfile

        Args:
            tempfile_path: Tempfile to load.

        Returns:

        Raises:

        """

    @property
    def external_edits_dir(self):
        """Directory to use for external edit files"""
        cache_dir = self.root.doc.cache_dir
        edits_dir = os.path.join(cache_dir, self._EDITS_SUBDIR)
        if not os.path.isdir(edits_dir):
            os.makedirs(edits_dir)
        return edits_dir


## Helper functions


def combine_redraws(bboxes):
    """Combine multiple rectangles representing redraw areas into one

    Args:
        bboxes (iterable): Sequence of redraw bboxes (lib.helpers.Rect)

    Returns:
        lib.helpers.Rect

This is best used for small, related redraws, since the GUI may have
better ways of combining rectangles into update regions.  Pairs of
before and after states are good candidates for using this.

If any of the input bboxes have zero size, the first such bbox is
returned. Zero-size update bboxes are the conventional way of
requesting a full-screen update.: A single redraw bbox.

    Raises:

    """
    redraw_bbox = helpers.Rect()
    for bbox in bboxes:
        if bbox.w == 0 and bbox.h == 0:
            return bbox
        redraw_bbox.expand_to_include_rect(bbox)
    return redraw_bbox


## Module testing


def _test():
    """Run doctest strings"""
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _test()
