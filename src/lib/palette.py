# This file is part of MyPaint.
# Copyright (C) 2013-2018 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


"""Palette: user-defined lists of color swatches"""

# TODO: Make palettes part of the model, save as part of ORA documents.


## Imports

from __future__ import division, print_function

import re
from copy import copy
import logging

from lib.helpers import clamp
from lib.observable import event
from lib.color import RGBColor
from lib.color import YCbCrColor
from lib.color import UIColor  # noqa
from lib.pycompat import unicode
from lib.pycompat import xrange
from lib.pycompat import PY3
from io import open

logger = logging.getLogger(__name__)

## Class and function defs


class Palette (object):
    """A flat list of color swatches, compatible with the GIMP

    As a (sideways-compatible) extension to the GIMP's format, MyPaint supports
    empty slots in the palette. These slots are represented by pure black
    swatches with the name ``__NONE__``.

    Palette objects expose the position within the palette of a current color
    match, which can be declared to be approximate or exact. This is used for
    highlighting the user concept of the "current color" in the GUI.

    Palette objects can be serialized in the GIMP's file format (the regular
    `unicode()` function on a Palette will do this too), or converted to and
    from a simpler JSON-ready representation for storing in the MyPaint prefs.
    Support for loading and saving via modal dialogs is defined here too.

    """

    ## Class-level constants
    _EMPTY_SLOT_ITEM = RGBColor(-1, -1, -1)
    _EMPTY_SLOT_NAME = "__NONE__"

    ## Construction, loading and saving

    def __init__(self, filehandle=None, filename=None, colors=None):
        """Instantiate, from a file or a sequence of colors

        :param filehandle: Filehandle to load.
        :param filename: Name of a file to load.
        :param colors: Iterable sequence of colors (lib.color.UIColor).

        The constructor arguments are mutually exclusive.  With no args
        specified, you get an empty palette.

          >>> Palette()
          <Palette colors=0, columns=0, name=None>

        Palettes can be generated from interpolations, which is handy for
        testing, at least.

          >>> cols = RGBColor(1,1,0).interpolate(RGBColor(1,0,1), 10)
          >>> Palette(colors=cols)
          <Palette colors=10, columns=0, name=None>

        """
        super(Palette, self).__init__()

        #: Number of columns. 0 means "natural flow"
        self._columns = 0
        #: List of named colors
        self._colors = []
        #: Name of the palette as a Unicode string, or None
        self._name = None
        #: Current position in the palette. None=no match; integer=index.
        self._match_position = None
        #: True if the current match is approximate
        self._match_is_approx = False

        # Clear and initialize
        self.clear(silent=True)
        if colors:
            for col in colors:
                col = self._copy_color_in(col)
                self._colors.append(col)
        elif filehandle:
            self.load(filehandle, silent=True)
        elif filename:
            with open(filename, "r", encoding="utf-8", errors="replace") as fp:
                self.load(fp, silent=True)

    def clear(self, silent=False):
        """Resets the palette to its initial state.

          >>> grey16 = RGBColor(1,1,1).interpolate(RGBColor(0,0,0), 16)
          >>> p = Palette(colors=grey16)
          >>> p.name = "Greyscale"
          >>> p.columns = 3
          >>> p                               # doctest: +ELLIPSIS
          <Palette colors=16, columns=3, name=...'Greyscale'>
          >>> p.clear()
          >>> p
          <Palette colors=0, columns=0, name=None>

        Fires the `info_changed()`, `sequence_changed()`, and `match_changed()`
        events, unless the `silent` parameter tests true.
        """
        self._colors = []
        self._columns = 0
        self._name = None
        self._match_position = None
        self._match_is_approx = False
        if not silent:
            self.info_changed()
            self.sequence_changed()
            self.match_changed()

    def load(self, filehandle, silent=False):
        """Load contents from a file handle containing a GIMP palette.

        :param filehandle: File-like object (.readline, line iteration)
        :param bool silent: If true, don't emit any events.

        >>> pal = Palette()
        >>> with open("palettes/MyPaint_Default.gpl", "r") as fp:
        ...     pal.load(fp)
        >>> len(pal) > 1
        True

        If the file format is incorrect, a RuntimeError will be raised.

        """
        comment_line_re = re.compile(r'^#')
        field_line_re = re.compile(r'^(\w+)\s*:\s*(.*)$')
        color_line_re = re.compile(r'^(\d+)\s+(\d+)\s+(\d+)\s*(?:\b(.*))$')
        fp = filehandle
        self.clear(silent=True)   # method fires events itself
        line = fp.readline()
        if line.strip() != "GIMP Palette":
            raise RuntimeError("Not a valid GIMP Palette")
        header_done = False
        line_num = 0
        for line in fp:
            line = line.strip()
            line_num += 1
            if line == '':
                continue
            if comment_line_re.match(line):
                continue
            if not header_done:
                match = field_line_re.match(line)
                if match:
                    key, value = match.groups()
                    key = key.lower()
                    if key == 'name':
                        self._name = value.strip()
                    elif key == 'columns':
                        self._columns = int(value)
                    else:
                        logger.warning("Unknown 'key:value' pair %r", line)
                    continue
                else:
                    header_done = True
            match = color_line_re.match(line)
            if not match:
                logger.warning("Expected 'R G B [Name]', not %r", line)
                continue
            r, g, b, col_name = match.groups()
            col_name = col_name.strip()
            r = clamp(int(r), 0, 0xff) / 0xff
            g = clamp(int(g), 0, 0xff) / 0xff
            b = clamp(int(b), 0, 0xff) / 0xff
            if r == g == b == 0 and col_name == self._EMPTY_SLOT_NAME:
                self.append(None)
            else:
                col = RGBColor(r, g, b)
                col.__name = col_name
                self._colors.append(col)
        if not silent:
            self.info_changed()
            self.sequence_changed()
            self.match_changed()

    def save(self, filehandle):
        """Saves the palette to an open file handle.

        :param filehandle: File-like object (.write suffices)

        >>> from lib.pycompat import PY3
        >>> if PY3:
        ...     from io import StringIO
        ... else:
        ...     from cStringIO import StringIO
        >>> fp = StringIO()
        >>> cols = RGBColor(1,.7,0).interpolate(RGBColor(.1,.1,.5), 16)
        >>> pal = Palette(colors=cols)
        >>> pal.save(fp)
        >>> fp.getvalue() == unicode(pal)
        True

        The file handle is not flushed, and is left open after the
        write.

        >>> fp.flush()
        >>> fp.close()

        """
        filehandle.write(unicode(self))

    def update(self, other):
        """Updates all details of this palette from another palette.

        Fires the `info_changed()`, `sequence_changed()`, and `match_changed()`
        events.
        """
        self.clear(silent=True)
        for col in other._colors:
            col = self._copy_color_in(col)
            self._colors.append(col)
        self._name = other._name
        self._columns = other._columns
        self.info_changed()
        self.sequence_changed()
        self.match_changed()

    ## Palette size and metadata

    def get_columns(self):
        """Get the number of columns (0 means unspecified)."""
        return self._columns

    def set_columns(self, n):
        """Set the number of columns (0 means unspecified)."""
        self._columns = int(n)
        self.info_changed()

    def get_name(self):
        """Gets the palette's name."""
        return self._name

    def set_name(self, name):
        """Sets the palette's name."""
        if name is not None:
            name = unicode(name)
        self._name = name
        self.info_changed()

    def __bool__(self):
        """Palettes never test false, regardless of their length.

        >>> p = Palette()
        >>> bool(p)
        True

        """
        return True

    def __len__(self):
        """Palette length is the number of color slots within it."""
        return len(self._colors)

    ## PY2/PY3 compat

    __nonzero__ = __bool__

    ## Match position marker

    def get_match_position(self):
        """Return the position of the current match (int or None)"""
        return self._match_position

    def set_match_position(self, i):
        """Sets the position of the current match (int or None)

        Fires `match_changed()` if the value is changed."""
        if i is not None:
            i = int(i)
            if i < 0 or i >= len(self):
                i = None
        if i != self._match_position:
            self._match_position = i
            self.match_changed()

    def get_match_is_approx(self):
        """Returns whether the current match is approximate."""
        return self._match_is_approx

    def set_match_is_approx(self, approx):
        """Sets whether the current match is approximate

        Fires match_changed() if the boolean value changes."""
        approx = bool(approx)
        if approx != self._match_is_approx:
            self._match_is_approx = approx
            self.match_changed()

    def match_color(self, col, exact=False, order=None):
        """Moves the match position to the color closest to the argument.

        :param col: The color to match.
        :type col: lib.color.UIColor
        :param exact: Only consider exact matches, and not near-exact or
                approximate matches.
        :type exact: bool
        :param order: a search order to use. Default is outwards from the
                match position, or in order if the match is unset.
        :type order: sequence or iterator of integer color indices.
        :returns: Whether the match succeeded.
        :rtype: bool

        By default, the matching algorithm favours exact or near-exact matches
        which are close to the current position. If the current position is
        unset, this search starts at 0. If there are no exact or near-exact
        matches, a looser approximate match will be used, again favouring
        matches with nearby positions.

          >>> red2blue = RGBColor(1, 0, 0).interpolate(RGBColor(0, 1, 1), 5)
          >>> p = Palette(colors=red2blue)
          >>> p.match_color(RGBColor(0.45, 0.45, 0.45))
          True
          >>> p.match_position
          2
          >>> p.match_is_approx
          True
          >>> p[p.match_position]
          <RGBColor r=0.5000, g=0.5000, b=0.5000>
          >>> p.match_color(RGBColor(0.5, 0.5, 0.5))
          True
          >>> p.match_is_approx
          False
          >>> p.match_color(RGBColor(0.45, 0.45, 0.45), exact=True)
          False
          >>> p.match_color(RGBColor(0.5, 0.5, 0.5), exact=True)
          True

        Fires the ``match_changed()`` event when changes happen.
        """
        if order is not None:
            search_order = order
        elif self.match_position is not None:
            search_order = _outwards_from(len(self), self.match_position)
        else:
            search_order = xrange(len(self))
        bestmatch_i = None
        bestmatch_d = None
        is_approx = True
        for i in search_order:
            c = self._colors[i]
            if c is self._EMPTY_SLOT_ITEM:
                continue
            # Closest exact or near-exact match by index distance (according to
            # the search_order). Considering near-exact matches as equivalent
            # to exact matches improves the feel of PaletteNext and
            # PalettePrev.
            if exact:
                if c == col:
                    bestmatch_i = i
                    is_approx = False
                    break
            else:
                d = _color_distance(col, c)
                if c == col or d < 0.06:
                    bestmatch_i = i
                    is_approx = False
                    break
                if bestmatch_d is None or d < bestmatch_d:
                    bestmatch_i = i
                    bestmatch_d = d
            # Measuring over a blend into solid equiluminant 0-chroma
            # grey for the orange #DA5D2E with an opaque but feathered
            # brush made huge, and picking just inside the point where the
            # palette widget begins to call it approximate:
            #
            # 0.05 is a difference only discernible (to me) by tilting LCD
            # 0.066 to 0.075 appears slightly greyer for large areas
            # 0.1 and above is very clearly distinct

        # If there are no exact or near-exact matches, choose the most similar
        # color anywhere in the palette.
        if bestmatch_i is not None:
            self._match_position = bestmatch_i
            self._match_is_approx = is_approx
            self.match_changed()
            return True
        return False

    def move_match_position(self, direction, refcol):
        """Move the match position in steps, matching first if needed.

        :param direction: Direction for moving, positive or negative
        :type direction: int:, ``1`` or ``-1``
        :param refcol: Reference color, used for initial matching when needed.
        :type refcol: UIColor
        :returns: the color newly matched, if the match position has changed
        :rtype: UIColor|NoneType

        Invoking this method when there's no current match position will select
        the color that's closest to the reference color, just like
        `match_color()`

        >>> greys = RGBColor(1,1,1).interpolate(RGBColor(0,0,0), 16)
        >>> pal = Palette(colors=greys)
        >>> refcol = RGBColor(0.5, 0.55, 0.45)
        >>> pal.move_match_position(-1, refcol)
        >>> pal.match_position
        7
        >>> pal.match_is_approx
        True

        When the current match is defined, but only an approximate match, this
        method converts it to an exact match but does not change its position.

          >>> pal.move_match_position(-1, refcol) is None
          False
          >>> pal.match_position
          7
          >>> pal.match_is_approx
          False

        When the match is initially exact, its position is stepped in the
        direction indicated, either by +1 or -1. Blank palette entries are
        skipped.

          >>> pal.move_match_position(-1, refcol) is None
          False
          >>> pal.match_position
          6
          >>> pal.match_is_approx
          False

        Fires ``match_position_changed()`` and ``match_is_approx_changed()`` as
        appropriate.  The return value is the newly matched color whenever this
        method produces a new exact match.

        """
        # Normalize direction
        direction = int(direction)
        if direction < 0:
            direction = -1
        elif direction > 0:
            direction = 1
        else:
            return None
        # If nothing is selected, pick the closest match without changing
        # the managed color.
        old_pos = self._match_position
        if old_pos is None:
            self.match_color(refcol)
            return None
        # Otherwise, refine the match, or step it in the requested direction.
        new_pos = None
        if self._match_is_approx:
            # Make an existing approximate match concrete.
            new_pos = old_pos
        else:
            # Index reflects a close or identical match.
            # Seek in the requested direction, skipping empty entries.
            pos = old_pos
            assert direction != 0
            pos += direction
            while pos < len(self._colors) and pos >= 0:
                if self._colors[pos] is not self._EMPTY_SLOT_ITEM:
                    new_pos = pos
                    break
                pos += direction
        # Update the palette index and the managed color.
        result = None
        if new_pos is not None:
            col = self._colors[new_pos]
            if col is not self._EMPTY_SLOT_ITEM:
                result = self._copy_color_out(col)
            self.set_match_position(new_pos)
            self.set_match_is_approx(False)
        return result

    ## Property-style access for setters and getters

    columns = property(get_columns, set_columns)
    name = property(get_name, set_name)
    match_position = property(get_match_position, set_match_position)
    match_is_approx = property(get_match_is_approx, set_match_is_approx)

    ## Color access

    def _copy_color_out(self, col):
        if col is self._EMPTY_SLOT_ITEM:
            return None
        result = RGBColor(color=col)
        result.__name = col.__name
        return result

    def _copy_color_in(self, col, name=None):
        if col is self._EMPTY_SLOT_ITEM or col is None:
            result = self._EMPTY_SLOT_ITEM
        else:
            if name is None:
                try:
                    name = col.__name
                except AttributeError:
                    pass
            if name is not None:
                name = unicode(name)
            result = RGBColor(color=col)
            result.__name = name
        return result

    def append(self, col, name=None, unique=False, match=False):
        """Appends a color, optionally setting a name for it.

        :param col: The color to append.
        :param name: Name of the color to insert.
        :param unique: If true, don't append if the color already exists
                in the palette. Only exact matches count.
        :param match: If true, set the match position to the
                appropriate palette entry.
        """
        col = self._copy_color_in(col, name)
        if unique:
            # Find the final exact match, if one is present
            for i in xrange(len(self._colors)-1, -1, -1):
                if col == self._colors[i]:
                    if match:
                        self._match_position = i
                        self._match_is_approx = False
                        self.match_changed()
                    return
        # Append new color, and select it if requested
        end_i = len(self._colors)
        self._colors.append(col)
        if match:
            self._match_position = end_i
            self._match_is_approx = False
            self.match_changed()
        self.sequence_changed()

    def insert(self, i, col, name=None):
        """Inserts a color, setting an optional name for it.

        :param i: Target index. `None` indicates appending a color.
        :param col: Color to insert. `None` indicates an empty slot.
        :param name: Name of the color to insert.

          >>> grey16 = RGBColor(1, 1, 1).interpolate(RGBColor(0, 0, 0), 16)
          >>> p = Palette(colors=grey16)
          >>> p.insert(5, RGBColor(1, 0, 0), name="red")
          >>> p
          <Palette colors=17, columns=0, name=None>
          >>> p[5]
          <RGBColor r=1.0000, g=0.0000, b=0.0000>

        Fires the `sequence_changed()` event. If the match position changes as
        a result, `match_changed()` is fired too.

        """
        col = self._copy_color_in(col, name)
        if i is None:
            self._colors.append(col)
        else:
            self._colors.insert(i, col)
            if self.match_position is not None:
                if self.match_position >= i:
                    self.match_position += 1
        self.sequence_changed()

    def reposition(self, src_i, targ_i):
        """Moves a color, or copies it to empty slots, or moves it the end.

        :param src_i: Source color index.
        :param targ_i: Source color index, or None to indicate the end.

        This operation performs a copy if the target is an empty slot, and a
        remove followed by an insert if the target slot contains a color.

          >>> grey16 = RGBColor(1, 1, 1).interpolate(RGBColor(0, 0, 0), 16)
          >>> p = Palette(colors=grey16)
          >>> p[5] = None           # creates an empty slot
          >>> p.match_position = 8
          >>> p[5] == p[0]
          False
          >>> p.reposition(0, 5)
          >>> p[5] == p[0]
          True
          >>> p.match_position
          8
          >>> p[5] = RGBColor(1, 0, 0)
          >>> p.reposition(14, 5)
          >>> p.match_position     # continues pointing to the same color
          9
          >>> len(p)       # repositioning doesn't change the length
          16

        Fires the `color_changed()` event for copies to empty slots, or
        `sequence_changed()` for moves. If the match position changes as a
        result, `match_changed()` is fired too.

        """
        assert src_i is not None
        if src_i == targ_i:
            return
        try:
            col = self._colors[src_i]
            assert col is not None  # just in case we change the internal repr
        except IndexError:
            return

        # Special case: just copy if the target is an empty slot
        match_pos = self.match_position
        if targ_i is not None:
            targ = self._colors[targ_i]
            if targ is self._EMPTY_SLOT_ITEM:
                self._colors[targ_i] = self._copy_color_in(col)
                self.color_changed(targ_i)
                # Copying from the matched color moves the match position.
                # Copying to the match position clears the match.
                if match_pos == src_i:
                    self.match_position = targ_i
                elif match_pos == targ_i:
                    self.match_position = None
                return

        # Normal case. Remove...
        self._colors.pop(src_i)
        moving_match = False
        updated_match = False
        if match_pos is not None:
            # Moving rightwards. Adjust for the pop().
            if targ_i is not None and targ_i > src_i:
                targ_i -= 1
            # Similar logic for the match position, but allow it to follow
            # the move if it started at the src position.
            if match_pos == src_i:
                match_pos = None
                moving_match = True
                updated_match = True
            elif match_pos > src_i:
                match_pos -= 1
                updated_match = True
        # ... then append or insert.
        if targ_i is None:
            self._colors.append(col)
            if moving_match:
                match_pos = len(self._colors) - 1
                updated_match = True
        else:
            self._colors.insert(targ_i, col)
            if match_pos is not None:
                if moving_match:
                    match_pos = targ_i
                    updated_match = True
                elif match_pos >= targ_i:
                    match_pos += 1
                    updated_match = True
        # Announce changes
        self.sequence_changed()
        if updated_match:
            self.match_position = match_pos
            self.match_changed()

    def pop(self, i):
        """Removes a color, returning it.

        Fires the `match_changed()` event if the match index changes as a
        result of the removal, and `sequence_changed()` if a color was removed,
        prior to its return.
        """
        i = int(i)
        try:
            col = self._colors.pop(i)
        except IndexError:
            return
        if self.match_position == i:
            self.match_position = None
        elif self.match_position > i:
            self.match_position -= 1
        self.sequence_changed()
        return self._copy_color_out(col)

    def get_color(self, i):
        """Looks up a color by its list index."""
        if i is None:
            return None
        try:
            col = self._colors[i]
            return self._copy_color_out(col)
        except IndexError:
            return None

    def __getitem__(self, i):
        return self.get_color(i)

    def __setitem__(self, i, col):
        self._colors[i] = self._copy_color_in(col, None)
        self.color_changed(i)

    ## Color name access

    def get_color_name(self, i):
        """Looks up a color's name by its list index."""
        try:
            col = self._colors[i]
        except IndexError:
            return
        if col is self._EMPTY_SLOT_ITEM:
            return
        return col.__name

    def set_color_name(self, i, name):
        """Sets a color's name by its list index."""
        try:
            col = self._colors[i]
        except IndexError:
            return
        if col is self._EMPTY_SLOT_ITEM:
            return
        col.__name = name
        self.color_changed(i)

    def get_color_by_name(self, name):
        """Looks up the first color with the given name.

          >>> pltt = Palette()
          >>> pltt.append(RGBColor(1,0,1), "Magenta")
          >>> pltt.get_color_by_name("Magenta")
          <RGBColor r=1.0000, g=0.0000, b=1.0000>

        """
        for col in self:
            if col.__name == name:
                return RGBColor(color=col)

    def __iter__(self):
        return self.iter_colors()

    def iter_colors(self):
        """Iterates across the palette's colors."""
        for col in self._colors:
            if col is self._EMPTY_SLOT_ITEM:
                yield None
            else:
                yield col

    ## Observable events

    @event
    def info_changed(self):
        """Event: palette name, or number of columns was changed."""

    @event
    def match_changed(self):
        """Event: either match position or match_is_approx was updated."""

    @event
    def sequence_changed(self):
        """Event: the color ordering or palette length was changed."""

    @event
    def color_changed(self, i):
        """Event: the color in the given slot, or its name, was modified."""

    ## Dumping and cloning

    def __unicode__(self):
        """Py2-era serialization as a Unicode string.

        Used by the Py3 __str__() while we are in transition.

        """
        result = u"GIMP Palette\n"
        if self._name is not None:
            result += u"Name: %s\n" % self._name
        if self._columns > 0:
            result += u"Columns: %d\n" % self._columns
        result += u"#\n"
        for col in self._colors:
            if col is self._EMPTY_SLOT_ITEM:
                col_name = self._EMPTY_SLOT_NAME
                r = g = b = 0
            else:
                col_name = col.__name
                r, g, b = [clamp(int(c*0xff), 0, 0xff) for c in col.get_rgb()]
            if col_name is None:
                result += u"%d %d %d\n" % (r, g, b)
            else:
                result += u"%d %d %d    %s\n" % (r, g, b, col_name)
        return result

    def __str__(self):
        """Py3: serialize as str (=Unicode). Py2: as bytes (lossy!)."""
        s = self.__unicode__()
        if not PY3:
            s = s.encode("utf-8", errors="replace")
        return s

    def __copy__(self):
        clone = Palette()
        clone.set_name(self.get_name())
        clone.set_columns(self.get_columns())
        for col in self._colors:
            if col is self._EMPTY_SLOT_ITEM:
                clone.append(None)
            else:
                clone.append(copy(col), col.__name)
        return clone

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __repr__(self):
        return "<Palette colors=%d, columns=%d, name=%r>" % (
            len(self._colors),
            self._columns,
            self._name,
        )

    ## Conversion to/from simple dict representation

    def to_simple_dict(self):
        """Converts the palette to a simple dict form used in the prefs."""
        simple = {}
        simple["name"] = self.get_name()
        simple["columns"] = self.get_columns()
        entries = []
        for col in self.iter_colors():
            if col is None:
                entries.append(None)
            else:
                name = col.__name
                entries.append((col.to_hex_str(), name))
        simple["entries"] = entries
        return simple

    @classmethod
    def new_from_simple_dict(cls, simple):
        """Constructs and returns a palette from the simple dict form."""
        pal = cls()
        pal.set_name(simple.get("name", None))
        pal.set_columns(simple.get("columns", None))
        for entry in simple.get("entries", []):
            if entry is None:
                pal.append(None)
            else:
                s, name = entry
                col = RGBColor.new_from_hex_str(s)
                pal.append(col, name)
        return pal


## Helper functions

def _outwards_from(n, i):
    """Search order within the palette, outwards from a given index.

    Defined for a sequence of len() `n`, outwards from index `i`.
    """
    assert i < n and i >= 0
    yield i
    for j in xrange(n):
        exhausted = True
        if i - j >= 0:
            yield i - j
            exhausted = False
        if i + j < n:
            yield i + j
            exhausted = False
        if exhausted:
            break


def _color_distance(c1, c2):
    """Distance metric for color matching in the palette.

    Use a geometric YCbCr distance, as recommended by Graphics Programming with
    Perl, chapter 1, Martien Verbruggen. If we want to give the chrominance
    dimensions a different weighting to luma, we can.

    """
    c1 = YCbCrColor(color=c1)
    c2 = YCbCrColor(color=c2)
    d_cb = c1.Cb - c2.Cb
    d_cr = c1.Cr - c2.Cr
    d_y = c1.Y - c2.Y
    return ((d_cb**2) + (d_cr**2) + (d_y)**2) ** (1.0/3)


## Module testing


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import doctest
    doctest.testmod()
