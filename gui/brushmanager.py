# This file is part of MyPaint.
# -*- coding: utf-8 -*-
# Copyright (C) 2009-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2010-2018 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""File management for brushes and brush groups."""

## Imports

from __future__ import division, print_function

from itertools import chain
import os
import zipfile
from os.path import basename
from warnings import warn
import logging
import shutil
import uuid
import contextlib

from lib.gettext import gettext as _
from lib.gettext import C_
from lib.helpers import utf8

from lib.gibindings import Gtk
from lib.gibindings import GdkPixbuf

from . import dialogs
from lib.brush import BrushInfo
from lib.observable import event
import lib.pixbuf
from . import drawutils
import gui.mode
import lib.config
from lib.pycompat import unicode
from lib.pycompat import xrange
from lib.pycompat import PY3

if PY3:
    import urllib.parse
else:
    import urllib


## Public module constants

PREVIEW_W = 128   #: Width of brush preview images
PREVIEW_H = 128   #: Height of brush preview images

FOUND_BRUSHES_GROUP = 'lost&found'   #: Orphaned brushes found at startup
DELETED_BRUSH_GROUP = 'deleted'   #: Orphaned brushes go here after group del
FAVORITES_BRUSH_GROUP = u'favorites'  #: User's favourites
NEW_BRUSH_GROUP = 'new'  #: Home for newly created brushes

## Internal module constants

_DEFAULT_STARTUP_GROUP = 'set#2'  # Suggestion only (FIXME: no effect?)
_DEFAULT_BRUSH = 'Dieterle/Fan#1'  # TODO: phase out and use heuristics?
_DEFAULT_ERASER = 'deevad/kneaded_eraser_large'  # TODO: -----------"---------
_DEVBRUSH_NAME_PREFIX = "devbrush_"
_BRUSH_HISTORY_NAME_PREFIX = "history_"
_BRUSH_HISTORY_SIZE = 5
_NUM_BRUSHKEYS = 10

_BRUSHPACK_README = "readme.txt"
_BRUSHPACK_ORDERCONF = "order.conf"

_DEVICE_NAME_NAMESPACE = uuid.UUID('169eaf8a-554e-45b8-8295-fc09b10031cc')

_TEST_BRUSHPACK_PY27 = u"tests/brushpacks/saved-with-py2.7.zip"

logger = logging.getLogger(__name__)


## Helper functions

def _device_name_uuid(device_name):
    """Return UUID5 string for a given device name

    >>> result = _device_name_uuid(u'Wacom Intuos5 touch S Pen stylus')
    >>> result == u'e97830e9-f9f9-50a5-8fff-68bead1a7021'
    True
    >>> type(result) == type(u'')
    True

    """
    if not PY3:
        device_name = utf8(unicode(device_name))
    return unicode(uuid.uuid5(_DEVICE_NAME_NAMESPACE, device_name))


def _quote_device_name(device_name):
    """Converts a device name to something safely storable on the disk

    Quotes an arbitrary device name for use as the basename of a
    device-specific brush.

    >>> result = _quote_device_name(u'Heavy Metal Umlaut D\u00ebvice')
    >>> result == 'Heavy+Metal+Umlaut+D%C3%ABvice'
    True
    >>> type(result) == type(u'')
    True
    >>> result = _quote_device_name(u'unsafe/device\\\\name')
    >>> result == 'unsafe%2Fdevice%5Cname'
    True
    >>> type(result) == type(u'')
    True

    Hopefully this is OK for Windows, UNIX and Mac OS X names.
    """
    device_name = unicode(device_name)
    if PY3:
        quoted = urllib.parse.quote_plus(
            device_name, safe='',
            encoding="utf-8",
        )
    else:
        u8bytes = device_name.encode("utf-8")
        quoted = urllib.quote_plus(u8bytes, safe='')
    return unicode(quoted)


def translate_group_name(name):
    """Translates a group name from a disk name to a display name."""
    d = {FOUND_BRUSHES_GROUP: _('Lost & Found'),
         DELETED_BRUSH_GROUP: _('Deleted'),
         FAVORITES_BRUSH_GROUP: _('Favorites'),
         'ink': _('Ink'),
         'classic': _('Classic'),
         'set#1': _('Set#1'),
         'set#2': _('Set#2'),
         'set#3': _('Set#3'),
         'set#4': _('Set#4'),
         'set#5': _('Set#5'),
         'experimental': _('Experimental'),
         'new': _('New'),
         }
    return d.get(name, name)


def _parse_order_conf(file_content):
    """Parse order.conf file data.

    :param bytes file_content: data from an order.conf (encoded UTF-8)
    :returns: a group dict

    The returned dict is of the form "{u'group1' : [u'brush1',
    u'brush2'], u'group2' : [u'brush3']}".

    """
    groups = {}
    try:
        file_content = file_content.decode("utf-8")
    except UnicodeDecodeError:
        # This handles order.conf files saved with the wrong encoding
        # on Windows (encoding was previously not explicitly specified).
        logger.warning("order.conf file not encoded with utf-8")
        file_content = file_content.decode('latin-1')
    curr_group = FOUND_BRUSHES_GROUP
    lines = file_content.replace(u'\r', u'\n').split(u'\n')
    for line in lines:
        name = line.strip()
        if name.startswith(u'#') or not name:
            continue
        if name.startswith(u'Group: '):
            curr_group = name[7:]
            if curr_group not in groups:
                groups[curr_group] = []
            continue
        groups.setdefault(curr_group, [])
        if name in groups[curr_group]:
            logger.warning(
                '%r: brush appears twice in the same group, ignored',
                name,
            )
            continue
        groups[curr_group].append(name)
    return groups


## Class definitions


class BrushManager (object):
    """Brush manager, responsible for groups of brushes."""

    ## Initialization

    def __init__(self, stock_brushpath, user_brushpath, app=None):
        """Initialize, with paths and a ref to the main app.

        :param unicode|str stock_brushpath: MyPaint install's RO brushes.
        :param unicode|str user_brushpath: User-writable brush library.
        :param gui.application.Application app: Main app (use None for test).

        The user_brushpath folder will be created if it does not yet exist.

        >>> from tempfile import mkdtemp
        >>> from shutil import rmtree
        >>> tmpdir = mkdtemp(u".brushes")
        >>> bm = BrushManager(lib.config.mypaint_brushdir, tmpdir, app=None)
        >>> len(bm.groups) > 0
        True
        >>> all([isinstance(k, unicode) for k in bm.groups.keys()])
        True
        >>> all([isinstance(v, list) for v in bm.groups.values()])
        True
        >>> rmtree(tmpdir)

        """
        super(BrushManager, self).__init__()

        # Default pigment setting when not specified by the brush
        self.pigment_by_default = None

        self.stock_brushpath = stock_brushpath
        self.user_brushpath = user_brushpath
        self.app = app

        #: The selected brush, as a ManagedBrush. Its settings are
        #: automatically reflected into the working brush engine brush when
        #: it changes.
        self.selected_brush = None

        self.groups = {}  #: Lists of ManagedBrushes, keyed by group name
        self.contexts = []  # Brush keys, indexed by keycap digit number
        self._brush_by_device = {}  # Device name to brush mapping.

        #: Slot used elsewhere for storing the ManagedBrush corresponding to
        #: the most recently saved or restored "context", a.k.a. brush key.
        self.selected_context = None

        if not os.path.isdir(self.user_brushpath):
            os.mkdir(self.user_brushpath)
        self._init_groups()

        # Brush order saving when that changes.
        self.brushes_changed += self._brushes_modified_cb

        # Update the history at the end of each definite input stroke.
        if app is not None:
            app.doc.input_stroke_ended += self._input_stroke_ended_cb

        # Make sure the user always gets a brush tool when they pick a brush
        # preset.
        self.brush_selected += self._brush_selected_cb

    @classmethod
    @contextlib.contextmanager
    def _mock(cls):
        """Context-managed mock BrushManager object for tests.

        Brushes are imported from the shipped brushes subfolder,
        and the user temp area is a temporary directory that's
        cleaned up by the context manager.

        Body yields (BrushManager_instance, tmp_dir_path).

        Please ensure that there are no open files in the tmpdir after
        use to that it can be rmtree()d. On Windows, that means closing
        any zipfile.Zipfile()s you open, even for read.

        """
        from tempfile import mkdtemp
        from shutil import rmtree

        dist_brushes = lib.config.mypaint_brushdir
        tmp_user_brushes = mkdtemp(suffix=u"_brushes")
        try:
            bm = cls(dist_brushes, tmp_user_brushes, app=None)
            yield (bm, tmp_user_brushes)
        finally:
            rmtree(tmp_user_brushes)

    def _load_brush(self, brush_cache, name, **kwargs):
        """Load a ManagedBrush from disk by name, via a cache."""
        if name not in brush_cache:
            b = ManagedBrush(self, name, persistent=True, **kwargs)
            brush_cache[name] = b
        return brush_cache[name]

    def _load_ordered_groups(self, brush_cache, filename):
        try:
            return self._load_ordered_groups_inner(brush_cache, filename)
        except Exception:
            logger.exception("Failed to load groups from %s" % filename)
            return {}

    def _load_ordered_groups_inner(self, brush_cache, filename):
        """Load a groups dict from an order.conf file."""
        groups = {}
        if os.path.exists(filename):
            with open(filename, "rb") as fp:
                groups = _parse_order_conf(fp.read())
            # replace brush names with ManagedBrush instances
            for group, names in list(groups.items()):
                brushes = []
                for name in names:
                    try:
                        b = self._load_brush(brush_cache, name)
                    except IOError as e:
                        logger.warn('%r: %r (removed from group)', name, e)
                        continue
                    brushes.append(b)
                groups[group] = brushes
        return groups

    def _init_ordered_groups(self, brush_cache):
        """Initialize the ordered subset of available brush groups.

        The ordered subset consists of those brushes which are listed in
        the stock and user brush directories' `order.conf` files.  This
        method safely merges upstream changes into the user's ordering.

        """
        join = os.path.join
        base_order_conf = join(self.user_brushpath, 'order_default.conf')
        our_order_conf = join(self.user_brushpath, 'order.conf')
        their_order_conf = join(self.stock_brushpath, 'order.conf')

        # Three-way-merge of brush groups (for upgrading)
        base = self._load_ordered_groups(brush_cache, base_order_conf)
        our = self._load_ordered_groups(brush_cache, our_order_conf)
        their = self._load_ordered_groups(brush_cache, their_order_conf)

        if not our:
            # order.conf missing, restore stock order even
            # if order_default.conf exists
            base = {}

        if base == their:
            self.groups = our
        else:
            logger.info('Merging upstream brush changes into your collection.')
            groups = set(base).union(our).union(their)
            for group in groups:
                # treat the non-existing groups as if empty
                base_brushes = base.setdefault(group, [])
                our_brushes = our.setdefault(group, [])
                their_brushes = their.setdefault(group, [])
                # add new brushes
                insert_index = 0
                for b in their_brushes:
                    if b in our_brushes:
                        insert_index = our_brushes.index(b) + 1
                    else:
                        if b not in base_brushes:
                            our_brushes.insert(insert_index, b)
                            insert_index += 1
                # remove deleted brushes
                for b in base_brushes:
                    if b not in their_brushes and b in our_brushes:
                        our_brushes.remove(b)
                # remove empty groups (except for the favorites)
                if not our_brushes and group != FAVORITES_BRUSH_GROUP:
                    del our[group]
            # finish
            self.groups = our
            self.save_brushorder()
            shutil.copy(their_order_conf, base_order_conf)

    def _list_brushes(self, path):
        """Recursively list the brushes within a directory.

        Return a list of brush names relative to path, using slashes
        for subdirectories on all platforms.

        """
        path += '/'
        result = []
        assert isinstance(path, unicode)  # make sure we get unicode filenames
        for name in os.listdir(path):
            assert isinstance(name, unicode)
            if name.endswith('.myb'):
                result.append(name[:-4])
            elif os.path.isdir(path + name):
                for name2 in self._list_brushes(path + name):
                    result.append(name + '/' + name2)
        return result

    def _init_unordered_groups(self, brush_cache):
        """Initialize the unordered subset of available brushes+groups.

        The unordered subset consists of all brushes that are not listed
        in an `order.conf` file. It includes brushkey brushes,
        per-device brushes, brushes in the painting history.

        This method trawls the stock and user brush directories for
        brushes which aren't listed in in an existing group, and adds
        them to the Lost & Found group, creating it if necessary. It
        should therefore be called after `_init_ordered_groups()`.

        """
        listbrushes = self._list_brushes
        for name in (listbrushes(self.stock_brushpath)
                     + listbrushes(self.user_brushpath)):
            if name.startswith(_DEVBRUSH_NAME_PREFIX):
                # Device brushes are lazy-loaded in fetch_brush_for_device()
                continue

            try:
                b = self._load_brush(brush_cache, name)
            except IOError as e:
                logger.warn("%r: %r (ignored)", name, e)
                continue
            if name.startswith('context'):
                i = int(name[-2:])
                self.contexts[i] = b
            elif name.startswith(_BRUSH_HISTORY_NAME_PREFIX):
                i_str = name.replace(_BRUSH_HISTORY_NAME_PREFIX, '')
                i = int(i_str)
                if 0 <= i < _BRUSH_HISTORY_SIZE:
                    self.history[i] = b
                else:
                    logger.warning(
                        "Brush history item %s "
                        "(entry %d): index outside of history range (0-%d)!",
                        name, i,
                        _BRUSH_HISTORY_SIZE - 1
                    )
            else:
                if not self.is_in_brushlist(b):
                    logger.info("Unassigned brush %r: assigning to %r",
                                name, FOUND_BRUSHES_GROUP)
                    brushes = self.groups.setdefault(FOUND_BRUSHES_GROUP, [])
                    brushes.insert(0, b)

    def _init_default_brushkeys_and_history(self):
        """Assign sensible defaults for brushkeys and history.

        Operates by filling in the gaps after `_init_unordered_groups()`
        has had a chance to populate the two lists.

        """

        # Try the default startup group first.
        default_group = self.groups.get(_DEFAULT_STARTUP_GROUP, None)

        # Otherwise, use the biggest group to minimise the chance
        # of repetition.
        if default_group is None:
            groups_by_len = sorted((len(g), n, g)
                                   for n, g in self.groups.items())
            _len, _name, default_group = groups_by_len[-1]

        # Populate blank entries.
        for i in xrange(_NUM_BRUSHKEYS):
            if self.contexts[i] is None:
                idx = (i + 9) % 10  # keyboard order
                c_name = unicode('context%02d') % i
                c = ManagedBrush(self, name=c_name, persistent=False)
                group_idx = idx % len(default_group)
                b = default_group[group_idx]
                b.clone_into(c, c_name)
                self.contexts[i] = c
        for i in xrange(_BRUSH_HISTORY_SIZE):
            if self.history[i] is None:
                h_name = unicode('%s%d') % (_BRUSH_HISTORY_NAME_PREFIX, i)
                h = ManagedBrush(self, name=h_name, persistent=False)
                group_i = i % len(default_group)
                b = default_group[group_i]
                b.clone_into(h, h_name)
                self.history[i] = h

    def _init_groups(self):
        """Initialize brush groups, loading them from disk."""

        self.contexts = [None for i in xrange(_NUM_BRUSHKEYS)]
        self.history = [None for i in xrange(_BRUSH_HISTORY_SIZE)]

        brush_cache = {}
        self._init_ordered_groups(brush_cache)
        self._init_unordered_groups(brush_cache)
        self._init_default_brushkeys_and_history()

        # clean up legacy stuff
        fn = os.path.join(self.user_brushpath, 'deleted.conf')
        if os.path.exists(fn):
            os.remove(fn)

    ## Observable events

    @event
    def brushes_changed(self, brushes):
        """Event: brushes changed (within their groups).

        Each observer is called with the following args:

        :param self: this BrushManager object
        :param brushes: Affected brushes
        :type brushes: list of ManagedBrushes

        This event is used to notify about brush ordering changes or brushes
        being moved between groups.
        """

    @event
    def groups_changed(self):
        """Event: brush groups changed (deleted, renamed, created)

        Observer callbacks are invoked with no args (other than a ref to the
        brushgroup).  This is used when the "set" of groups change, e.g. when a
        group is renamed, deleted, or created.  It's invoked when self.groups
        changes.
        """

    @event
    def brush_selected(self, brush, info):
        """Event: a different brush was selected.

        Observer callbacks are invoked with the newly selected ManagedBrush and
        its corresponding BrushInfo.
        """

    ## Initial and default brushes

    def select_initial_brush(self):
        """Select the initial brush using saved app preferences.
        """
        initial_brush = None
        # If we recorded which devbrush was last in use, restore it and assume
        # that most of the time the user will continue to work with the same
        # brush and its settings.
        app = self.app
        if app is not None:
            prefs = app.preferences
            last_used_devbrush = prefs.get('devbrush.last_used')
            initial_brush = self.fetch_brush_for_device(last_used_devbrush)
            # Otherwise, initialise from the old selected_brush setting
            if initial_brush is None:
                last_active_name = prefs.get('brushmanager.selected_brush')
                if last_active_name is not None:
                    initial_brush = self.get_brush_by_name(last_active_name)
        # Fallback
        if initial_brush is None:
            initial_brush = self.get_default_brush()
        self.select_brush(initial_brush)

    def _get_matching_brush(self, name=None, keywords=None,
                            favored_group=_DEFAULT_STARTUP_GROUP,
                            fallback_eraser=0.0):
        """Gets a brush robustly by name, by partial name, or a default.

        If a brush named `name` exists, use that. Otherwise search though all
        groups, `favored_group` first, for brushes with any of `keywords`
        in their name. If that fails, construct a new default brush and use
        a given value for its 'eraser' property.
        """
        if name is not None:
            brush = self.get_brush_by_name(name)
            if brush is not None:
                return brush
        if keywords is not None:
            group_names = sorted(self.groups.keys())
            if favored_group in self.groups:
                group_names.remove(favored_group)
                group_names.insert(0, favored_group)
            for group_name in group_names:
                for brush in self.groups[group_name]:
                    for keyword in keywords:
                        if keyword in brush.name:
                            return brush
        # Fallback
        name = 'fallback-default'
        if fallback_eraser != 0.0:
            name += '-eraser'
        brush = ManagedBrush(self, name)
        brush.brushinfo.set_base_value("eraser", fallback_eraser)
        return brush

    def get_default_brush(self):
        """Returns a suitable default drawing brush."""
        drawing = ["pencil", "charcoal", "sketch"]
        return self._get_matching_brush(name=_DEFAULT_BRUSH, keywords=drawing)

    def get_default_eraser(self):
        """Returns a suitable default eraser brush."""
        erasing = ["eraser", "kneaded", "smudge"]
        return self._get_matching_brush(name=_DEFAULT_ERASER, keywords=erasing,
                                        fallback_eraser=1.0)

    def set_pigment_by_default(self, pigment_by_default):
        """Change the default pigment setting to on/off

        This updates loaded managed brushes as well, if they
        do not have the pigment setting set explicitly.
        """
        if self.pigment_by_default != pigment_by_default:
            msg = "Switching default pigment setting to {state}"
            logger.info(msg.format(
                state="On" if pigment_by_default else "Off"))
            self.pigment_by_default = pigment_by_default
            self._reset_pigment_setting()

    def default_pigment_setting(self, setting_info):
        """Pigment (paint_mode) setting override
        """
        if self.pigment_by_default:
            return setting_info.default
        else:
            return setting_info.min

    def _reset_pigment_setting(self):
        appbrush = ()
        if self.app:
            appbrush = (self.app.brush,)
        # Reset the pigment setting for any cached brushes
        # that may have been loaded with the other default - this
        # will not affect brushes that have the pigment setting
        # defined explicitly.
        to_reset = chain(
            # Alter both the current working brush and the selected brush
            appbrush,
            (self.selected_brush.get_brushinfo(),),
            # Also the brush history
            [mb.get_brushinfo() for mb in self.history
             if mb is not None and mb.loaded()],
            # And any other loaded brush
            [mb.get_brushinfo()
             for v in self.groups.values()
             for mb in v if mb.loaded()]
        )
        for bi in to_reset:
            bi.reset_if_undefined('paint_mode')


    ## Brushpack import and export

    def import_brushpack(self, path, window=None):
        """Import a brushpack from a zipfile, with confirmation dialogs.

        :param path: Brush pack zipfile path
        :type path: str
        :param window: Parent window, for dialogs to set.
        :type window: GtkWindow or None
        :returns: Set of imported group names
        :rtype: set

        Confirmation dialogs are only shown if "window" is a suitable
        toplevel to attach the dialogs to.

        >>> with BrushManager._mock() as (bm, tmpdir):
        ...     imp = bm.import_brushpack(_TEST_BRUSHPACK_PY27, window=None)
        ...     py27_g = bm.groups.get(list(imp)[0])
        >>> py27_g[0]  # doctest: +ELLIPSIS
        <ManagedBrush...>
        >>> g_names = set(b.name for b in py27_g)
        >>> u'brushlib-test/basic' in g_names
        True
        >>> u'brushlib-test/fancy_\U0001f308\U0001f984\u2728' in g_names
        True

        """

        with zipfile.ZipFile(path) as zf:

            # In Py2, when the entry was saved without the 0x800 flag
            # namelist() will return it as bytes, not unicode.  We only
            # want Unicode strings.
            names = []
            for name in zf.namelist():
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                names.append(name)

            readme = None
            if _BRUSHPACK_README in names:
                readme = zf.read(_BRUSHPACK_README).decode("utf-8")

            if _BRUSHPACK_ORDERCONF not in names:
                raise InvalidBrushpack(C_(
                    "brushpack import failure messages",
                    u"No file named “{order_conf_file}”. "
                    u"This is not a brushpack."
                ).format(
                    order_conf_file = _BRUSHPACK_ORDERCONF,
                ))
            groups = _parse_order_conf(zf.read(_BRUSHPACK_ORDERCONF))

            new_brushes = []
            for brushes in groups.values():
                for brush in brushes:
                    if brush not in new_brushes:
                        new_brushes.append(brush)
            logger.info(
                "%d different brushes found in %r of brushpack",
                len(new_brushes),
                _BRUSHPACK_ORDERCONF,
            )

            # Validate file content. The names in order.conf and the
            # brushes found in the zip must match. This should catch
            # encoding screwups, everything should be a unicode object.
            for brush in new_brushes:
                if brush + '.myb' not in names:
                    raise InvalidBrushpack(C_(
                        "brushpack import failure messages",
                        u"Brush “{brush_name}” is "
                        u"listed in “{order_conf_file}”, "
                        u"but it does not exist in the zipfile."
                    ).format(
                        brush_name = brush,
                        order_conf_file = _BRUSHPACK_ORDERCONF,
                    ))
            for name in names:
                if name.endswith('.myb'):
                    brush = name[:-4]
                    if brush not in new_brushes:
                        raise InvalidBrushpack(C_(
                            "brushpack import failure messages",
                            u"Brush “{brush_name}” exists in the zipfile, "
                            u"but it is not listed in “{order_conf_file}”."
                        ).format(
                            brush_name = brush,
                            order_conf_file = _BRUSHPACK_ORDERCONF,
                        ))
            if readme and window:
                answer = dialogs.confirm_brushpack_import(
                    basename(path), window, readme,
                )
                if answer == Gtk.ResponseType.REJECT:
                    return set()

            do_overwrite = False
            do_ask = True
            renamed_brushes = {}
            imported_groups = set()
            for groupname, brushes in groups.items():
                managed_brushes = self.get_group_brushes(groupname)
                if managed_brushes:
                    answer = dialogs.DONT_OVERWRITE_THIS
                    if window:
                        answer = dialogs.confirm_rewrite_group(
                            window, translate_group_name(groupname),
                            translate_group_name(DELETED_BRUSH_GROUP),
                        )
                    if answer == dialogs.CANCEL:
                        return set()
                    elif answer == dialogs.OVERWRITE_THIS:
                        self.delete_group(groupname)
                    elif answer == dialogs.DONT_OVERWRITE_THIS:
                        i = 0
                        old_groupname = groupname
                        while groupname in self.groups:
                            i += 1
                            groupname = old_groupname + '#%d' % i
                    managed_brushes = self.get_group_brushes(groupname)
                imported_groups.add(groupname)

                for brushname in brushes:
                    # extract the brush from the zip
                    assert (brushname + '.myb') in zf.namelist()
                    # Support for utf-8 ZIP filenames that don't have
                    # the utf-8 bit set.
                    brushname_utf8 = utf8(brushname)
                    try:
                        myb_data = zf.read(brushname + u'.myb')
                    except KeyError:
                        myb_data = zf.read(brushname_utf8 + b'.myb')
                    try:
                        preview_data = zf.read(brushname + u'_prev.png')
                    except KeyError:
                        preview_data = zf.read(brushname_utf8 + b'_prev.png')
                    # in case we have imported that brush already in a
                    # previous group, but decided to rename it
                    if brushname in renamed_brushes:
                        brushname = renamed_brushes[brushname]
                    # possibly ask how to import the brush file
                    # (if we didn't already)
                    b = self.get_brush_by_name(brushname)
                    if brushname in new_brushes:
                        new_brushes.remove(brushname)
                        if b:
                            existing_preview_pixbuf = b.preview
                            if do_ask and window:
                                answer = dialogs.confirm_rewrite_brush(
                                    window, brushname, existing_preview_pixbuf,
                                    preview_data,
                                )
                                if answer == dialogs.CANCEL:
                                    break
                                elif answer == dialogs.OVERWRITE_ALL:
                                    do_overwrite = True
                                    do_ask = False
                                elif answer == dialogs.OVERWRITE_THIS:
                                    do_overwrite = True
                                    do_ask = True
                                elif answer == dialogs.DONT_OVERWRITE_THIS:
                                    do_overwrite = False
                                    do_ask = True
                                elif answer == dialogs.DONT_OVERWRITE_ANYTHING:
                                    do_overwrite = False
                                    do_ask = False
                            # find a new name (if requested)
                            brushname_old = brushname
                            i = 0
                            while not do_overwrite and b:
                                i += 1
                                brushname = brushname_old + u'#%d' % i
                                renamed_brushes[brushname_old] = brushname
                                b = self.get_brush_by_name(brushname)

                        if not b:
                            b = ManagedBrush(self, brushname)

                        # write to disk and reload brush (if overwritten)
                        prefix = b._get_fileprefix(saving=True)
                        with open(prefix + '.myb', 'wb') as myb_f:
                            myb_f.write(myb_data)
                        with open(prefix + '_prev.png', 'wb') as preview_f:
                            preview_f.write(preview_data)
                        b.load()
                    # finally, add it to the group
                    if b not in managed_brushes:
                        managed_brushes.append(b)
                    self.brushes_changed(managed_brushes)

        if DELETED_BRUSH_GROUP in self.groups:
            # remove deleted brushes that are in some group again
            self.delete_group(DELETED_BRUSH_GROUP)
        return imported_groups

    def export_group(self, group, filename):
        """Exports a group to a brushpack zipfile.

        :param unicode|str group: Name of the group to save.
        :param unicode|str filename: Path to a .zip file to create.

        >>> with BrushManager._mock() as (bm, tmpdir):
        ...     group = list(bm.groups)[0]
        ...     zipname = os.path.join(tmpdir, group + u"zip")
        ...     bm.export_group(group, zipname)
        ...     with zipfile.ZipFile(zipname, mode="r") as zf:
        ...         assert len(zf.namelist()) > 0
        ...         assert u"order.conf" in zf.namelist()

        """
        brushes = self.get_group_brushes(group)
        order_conf = b'Group: %s\n' % utf8(group)
        with zipfile.ZipFile(filename, mode='w') as zf:
            for brush in brushes:
                prefix = brush._get_fileprefix()
                zf.write(prefix + u'.myb', brush.name + u'.myb')
                zf.write(prefix + u'_prev.png', brush.name + u'_prev.png')
                order_conf += utf8(brush.name + "\n")
            zf.writestr(u'order.conf', order_conf)

    ## Brush lookup / access

    def get_brush_by_name(self, name):
        """Gets a ManagedBrush by its name.

        Slow method, should not be called too often.

        >>> with BrushManager._mock() as (bm, tmpdir):
        ...     brush1 = bm.get_brush_by_name(u"classic/pen")
        >>> brush1 # doctest: +ELLIPSIS
        <ManagedBrush...>

        """
        # FIXME: speed up, use a dict.
        for group, brushes in self.groups.items():
            for b in brushes:
                if b.name == name:
                    return b

    def is_in_brushlist(self, brush):
        """Returns whether this brush is in some brush group's list."""
        for group, brushes in self.groups.items():
            if brush in brushes:
                return True
        return False

    def get_parent_brush(self, brush=None, brushinfo=None):
        """Gets the parent `ManagedBrush` for a brush or a `BrushInfo`.
        """
        if brush is not None:
            brushinfo = brush.brushinfo
        if brushinfo is None:
            raise RuntimeError("One of `brush` or `brushinfo` must be defined")
        parent_name = brushinfo.get_string_property("parent_brush_name")
        if parent_name is None:
            return None
        else:
            parent_brush = self.get_brush_by_name(parent_name)
            if parent_brush is None:
                return None
            return parent_brush

    ## Brush order within groups, order.conf

    def _brushes_modified_cb(self, bm, brushes):
        """Saves the brush order when it changes."""
        self.save_brushorder()

    def save_brushorder(self):
        """Save the user's chosen brush order to disk.

        >>> with BrushManager._mock() as (bm, tmpdir):
        ...     bm.save_brushorder()

        """

        path = os.path.join(self.user_brushpath, u'order.conf')
        with open(path, 'wb') as f:
            f.write(utf8(u'# this file saves brush groups and order\n'))
            for group, brushes in self.groups.items():
                f.write(utf8(u'Group: {}\n'.format(group)))
                for b in brushes:
                    f.write(utf8(b.name + u'\n'))

    ## The selected brush

    def select_brush(self, brush):
        """Selects a ManagedBrush, highlights it, & updates the live brush.

        :param brush: brush to select
        :type brush: BrushInfo

        """
        if brush is None:
            brush = self.get_default_brush()

        brushinfo = brush.brushinfo
        if not self.is_in_brushlist(brush):
            # select parent brush instead, but keep brushinfo
            parent = self.get_parent_brush(brush=brush)
            if parent is not None:
                brush = parent

        self.selected_brush = brush
        if self.app is not None:
            self.app.preferences['brushmanager.selected_brush'] = brush.name

        # Notify subscribers. Takes care of updating the live
        # brush, amongst other things
        self.brush_selected(brush, brushinfo)

    def clone_selected_brush(self, name):
        """Clones the current and selected brush into a new `BrushInfo`.

        Creates a new ManagedBrush based on the selected brush in the brushlist
        and the currently active lib.brush. The brush settings are copied from
        the active brush, and the preview is copied from the currently selected
        BrushInfo.

        """
        if self.app is None:
            raise ValueError("No app. BrushManager in test mode?")
        clone = ManagedBrush(self, name, persistent=False)
        clone.brushinfo = self.app.brush.clone()
        clone.preview = self.selected_brush.preview
        parent = self.selected_brush.name
        clone.brushinfo.set_string_property("parent_brush_name", parent)
        return clone

    def _brush_selected_cb(self, bm, brush, brushinfo):
        """Internal callback: User just picked a brush preset.

        Called when the user changes to a brush preset somehow (e.g.
        from a shortcut or the brush panel). Makes sure a
        brush-dependant tool (e.g. Freehand, Connected Lines, etc.) is
        selected.

        """
        if not self.app:
            return
        self.app.doc.modes.pop_to_behaviour(gui.mode.Behavior.PAINT_BRUSH)

    ## Device-specific brushes

    def store_brush_for_device(self, device_name, managed_brush):
        """Records a brush as associated with an input device.

        :param device_name: name of an input device
        :type device_name: str
        :param managed_brush: the brush to associate
        :type managed_brush: ManagedBrush

        Normally the brush will be cloned first, since it will be given a new
        name. However, if the brush has a 'name' attribute of None, it will
        *not* be cloned and just modified in place and stored.

        """
        brush = managed_brush
        if brush.name is not None:
            brush = brush.clone()
        brush.name = unicode(
            _DEVBRUSH_NAME_PREFIX + _device_name_uuid(device_name))
        self._brush_by_device[device_name] = brush

    def fetch_brush_for_device(self, device_name):
        """Fetches the brush associated with an input device."""
        if not device_name:
            return None

        if device_name not in self._brush_by_device:
            self._brush_by_device[device_name] = None

            names = (
                _device_name_uuid(device_name),
                _quote_device_name(device_name),  # for backward compatibility
            )
            for name in names:
                path = os.path.join(
                    self.user_brushpath, _DEVBRUSH_NAME_PREFIX + name + '.myb')
                if not os.path.isfile(path):
                    continue

                try:
                    b = ManagedBrush(
                        self, unicode(_DEVBRUSH_NAME_PREFIX + name),
                        persistent=True)
                except IOError as e:
                    logger.warn("%r: %r (ignored)", name, e)
                else:
                    self._brush_by_device[device_name] = b

                break

        assert device_name in self._brush_by_device
        return self._brush_by_device[device_name]

    def save_brushes_for_devices(self):
        """Saves the device/brush associations to disk."""
        for devbrush in self._brush_by_device.values():
            if devbrush is not None:
                devbrush.save()

    ## Brush history

    def _input_stroke_ended_cb(self, doc, event):
        """Update brush usage history at the end of an input stroke."""
        if self.app is None:
            raise ValueError("No app. BrushManager in test mode?")
        wb_info = self.app.brush
        wb_parent_name = wb_info.settings.get("parent_brush_name")
        # Remove the to-be-added brush from the history if it's already in it
        if wb_parent_name:
            # Favour "same parent" as the main measure of identity,
            # when it's defined.
            for i, hb in enumerate(self.history):
                hb_info = hb.brushinfo
                hb_parent_name = hb_info.settings.get("parent_brush_name")
                if wb_parent_name == hb_parent_name:
                    del self.history[i]
                    break
        else:
            # Otherwise, fall back to matching on the brush dynamics.
            # Many old .ORA files have pickable strokes in their layer map
            # which don't nominate a parent.
            for i, hb in enumerate(self.history):
                hb_info = hb.brushinfo
                if wb_info.matches(hb_info):
                    del self.history[i]
                    break
        # Append the working brush to the history, and trim it to length
        nb = ManagedBrush(self, name=None, persistent=False)
        nb.brushinfo = wb_info.clone()
        nb.preview = self.selected_brush.preview
        self.history.append(nb)
        while len(self.history) > _BRUSH_HISTORY_SIZE:
            del self.history[0]
        # Rename the history brushes so they save to the right files.
        for i, hb in enumerate(self.history):
            hb.name = u"%s%d" % (_BRUSH_HISTORY_NAME_PREFIX, i)

    def save_brush_history(self):
        """Saves the brush usage history to disk."""
        for brush in self.history:
            brush.save()

    ## Brush groups

    def get_group_brushes(self, group):
        """Get a group's active brush list.

        If the group does not exist, it will be created.

        :param str group: Name of the group to fetch
        :returns: The active list of `ManagedBrush`es.
        :rtype: list

        The returned list is owned by the BrushManager. You can modify
        it, but you'll have to do your own notifications.

        See also: groups_changed(), brushes_changed().

        """
        if group not in self.groups:
            brushes = []
            self.groups[group] = brushes
            self.groups_changed()
            self.save_brushorder()
        return self.groups[group]

    def create_group(self, new_group):
        """Creates a new brush group

        :param group: Name of the group to create
        :type group: str
        :rtype: empty list, owned by the BrushManager

        Returns the newly created group as a(n empty) list.

        """
        return self.get_group_brushes(new_group)

    def rename_group(self, old_group, new_group):
        """Renames a group.

        :param old_group: Name of the group to assign the new name to.
        :type old_group: str
        :param new_group: New name for the group.
        :type new_group: str

        """
        brushes = self.create_group(new_group)
        brushes += self.groups[old_group]
        self.delete_group(old_group)

    def delete_group(self, group):
        """Deletes a group.

        :param group: Name of the group to delete
        :type group: str

        Orphaned brushes will be placed into `DELETED_BRUSH_GROUP`, which
        will be created if necessary.

        """

        homeless_brushes = self.groups[group]
        del self.groups[group]

        for brushes in self.groups.values():
            for b2 in brushes:
                if b2 in homeless_brushes:
                    homeless_brushes.remove(b2)

        if homeless_brushes:
            deleted_brushes = self.get_group_brushes(DELETED_BRUSH_GROUP)
            for b in homeless_brushes:
                deleted_brushes.insert(0, b)
            self.brushes_changed(deleted_brushes)
        self.brushes_changed(homeless_brushes)
        self.groups_changed()
        self.save_brushorder()


class ManagedBrush(object):
    """User-facing representation of a brush's settings.

    Managed brushes have a name, a preview image, and brush settings.
    The settings and the preview are loaded on demand.
    They cannot be selected or painted with directly,
    but their settings can be loaded into the running app:
    see `Brushmanager.select_brush()`.

    """

    def __init__(self, brushmanager, name=None, persistent=False):
        """Construct, with a ref back to its BrushManager.

        Normally clients won't construct ManagedBrushes directly.
        Instead, use the groups dict in the BrushManager for access to
        all the brushes loaded from the user and stock brush folders.

        >>> with BrushManager._mock() as (bm, tmpdir):
        ...     for gname, gbrushes in bm.groups.items():
        ...         for b in gbrushes:
        ...             assert isinstance(b, ManagedBrush)
        ...             b.load()

        """

        super(ManagedBrush, self).__init__()
        self.bm = brushmanager
        self._preview = None
        self._brushinfo = BrushInfo(default_overrides={
            'paint_mode': self.bm.default_pigment_setting
        })

        #: The brush's relative filename, sans extension.
        self.name = name

        #: If True, this brush is stored in the filesystem.
        self.persistent = persistent

        # If True, this brush is fully initialized, ready to paint with.
        self._settings_loaded = False

        # Change detection for on-disk files.
        self._settings_mtime = None
        self._preview_mtime = None

        # Files are loaded later,
        # but throw an exception now if they don't exist.
        if persistent:
            self._get_fileprefix()
            assert self.name is not None

    def loaded(self):
        return self._settings_loaded

    ## Preview image: loaded on demand

    def get_preview(self):
        """Gets a preview image for the brush

        For persistent brushes, this loads the disk preview; otherwise a
        fairly slow automated brush preview is used.

        >>> with BrushManager._mock() as (bm, tmpdir):
        ...     b = ManagedBrush(bm, name=None, persistent=False)
        ...     b.get_preview()   # doctest: +ELLIPSIS
        <GdkPixbuf.Pixbuf...>

        The results are cached in RAM.

        >>> with BrushManager._mock() as (bm, tmpdir):
        ...     imported = bm.import_brushpack(_TEST_BRUSHPACK_PY27)
        ...     assert(imported)
        ...     pixbufs1 = []
        ...     for gn in sorted(bm.groups.keys()):
        ...         gbs = bm.groups[gn]
        ...         for b in gbs:
        ...             pixbufs1.append(b)
        ...     pixbufs2 = []
        ...     for gn in sorted(bm.groups.keys()):
        ...         gbs = bm.groups[gn]
        ...         for b in gbs:
        ...             pixbufs2.append(b)
        >>> len(pixbufs1) == len(pixbufs2)
        True
        >>> all([p1 is p2 for (p1, p2) in zip(pixbufs1, pixbufs2)])
        True
        >>> pixbufs1 == pixbufs2
        True

        """
        if self._preview is None and self.name:
            self._load_preview()
        if self._preview is None:
            brushinfo = self.get_brushinfo()
            self._preview = drawutils.render_brush_preview_pixbuf(brushinfo)
        return self._preview

    def set_preview(self, pixbuf):
        self._preview = pixbuf

    preview = property(get_preview, set_preview)

    ## Text fields

    @property
    def description(self):
        """Short, user-facing tooltip description for the brush.

        >>> with BrushManager._mock() as (bm, tmpdir):
        ...     for gn, gbs in bm.groups.items():
        ...         for b in gbs:
        ...             assert isinstance(b.description, unicode)
        ...             b.description = u"junk"
        ...             assert isinstance(b.description, unicode)
        ...             b.save()

        """
        return self.brushinfo.get_string_property("description")

    @description.setter
    def description(self, s):
        self.brushinfo.set_string_property("description", s)

    @property
    def notes(self):
        """Longer, brush developer's notes field for a brush.

        >>> with BrushManager._mock() as (bm, tmpdir):
        ...     imp = bm.import_brushpack(_TEST_BRUSHPACK_PY27)
        ...     imp_g = list(imp)[0]
        ...     for b in bm.groups[imp_g]:
        ...         assert isinstance(b.notes, unicode)
        ...         b.notes = u"junk note"
        ...         assert isinstance(b.notes, unicode)
        ...         b.save()

        """
        return self.brushinfo.get_string_property("notes")

    @notes.setter
    def notes(self, s):
        self.brushinfo.set_string_property("notes", s)

    ## Brush settings: loaded on demand

    def get_brushinfo(self):
        self._ensure_settings_loaded()
        return self._brushinfo

    def set_brushinfo(self, brushinfo):
        self._brushinfo = brushinfo

    brushinfo = property(get_brushinfo, set_brushinfo)

    ## Display

    def __repr__(self):
        if self._brushinfo.settings:
            pname = self._brushinfo.get_string_property("parent_brush_name")
            return "<ManagedBrush %r p=%s>" % (self.name, pname)
        else:
            return "<ManagedBrush %r (settings not loaded yet)>" % self.name

    def get_display_name(self):
        """Gets a displayable name for the brush."""
        if self.bm.is_in_brushlist(self):  # FIXME: get rid of this check
            dname = self.name
        else:
            dname = self.brushinfo.get_string_property("parent_brush_name")
        if dname is None:
            return _("Unknown Brush")
        return dname.replace("_", " ")

    ## Cloning

    def clone(self, name):
        """Clone this brush, and give it a new name.

        Creates a new brush with all the settings of this brush,
        assigning it a new name

        """
        clone = ManagedBrush(self.bm)
        self.clone_into(clone, name=name)
        return clone

    def clone_into(self, target, name):
        "Copies all brush settings into another brush, giving it a new name"
        self._ensure_settings_loaded()
        target.brushinfo = self.brushinfo.clone()
        if self.bm.is_in_brushlist(self):  # FIXME: get rid of this check!
            target.brushinfo.set_string_property(
                "parent_brush_name", self.name,
            )
        target.preview = self.preview
        target.name = name

    ## File save/load helpers

    def _get_fileprefix(self, saving=False):
        """Returns the filesystem prefix to use when saving or loading.

        :param saving: caller wants a prefix to save to
        :type saving: bool
        :rtype: unicode

        This assigns ``self.name`` if it isn't defined.

        Files are stored with the returned prefix,
        with the extension ".myb" for brush data
        and "_prev.myb" for preview images.

        If `saving` is true, intermediate directories will be created,
        and the returned prefix will always contain the user brushpath.
        Otherwise the prefix you get depends on
        whether a stock brush exists and
        whether a user brush with the same name does not.

        See also `delete_from_disk()`.

        """
        prefix = u'b'
        user_bp = os.path.realpath(self.bm.user_brushpath)
        stock_bp = os.path.realpath(self.bm.stock_brushpath)
        if user_bp == stock_bp:
            # working directly on brush collection, use different prefix
            prefix = u's'

        # Construct a new, unique name if the brush is not yet named
        if not self.name:
            i = 0
            while True:
                self.name = u'%s%03d' % (prefix, i)
                a = os.path.join(self.bm.user_brushpath, self.name + u'.myb')
                b = os.path.join(self.bm.stock_brushpath, self.name + u'.myb')
                if not os.path.isfile(a) and not os.path.isfile(b):
                    break
                i += 1
        assert isinstance(self.name, unicode)

        # Always save to the user brush path.
        prefix = os.path.join(self.bm.user_brushpath, self.name)
        if saving:
            if u'/' in self.name:
                d = os.path.dirname(prefix)
                if not os.path.isdir(d):
                    os.makedirs(d)
            return prefix

        # Loading: try user first, then stock
        if not os.path.isfile(prefix + u'.myb'):
            prefix = os.path.join(self.bm.stock_brushpath, self.name)
        if not os.path.isfile(prefix + u'.myb'):
            raise IOError('brush "%s" not found' % self.name)
        return prefix

    def _remember_mtimes(self):
        prefix = self._get_fileprefix()
        try:
            preview_file = prefix + '_prev.png'
            self._preview_mtime = os.path.getmtime(preview_file)
        except OSError:
            logger.exception("Failed to update preview file access time")
            self._preview_mtime = None
        try:
            settings_file = prefix + '.myb'
            self._settings_mtime = os.path.getmtime(settings_file)
        except OSError:
            logger.exception("Failed to update settings file access time")
            self._settings_mtime = None

    ## Saving and deleting

    def save(self):
        """Saves the brush's settings and its preview"""
        prefix = self._get_fileprefix(saving=True)
        # Save preview
        if self.preview.get_has_alpha():
            # Remove alpha
            # Previous mypaint versions would display an empty image
            w, h = PREVIEW_W, PREVIEW_H
            tmp = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, False,
                                       8, w, h)
            tmp.fill(0xffffffff)
            self.preview.composite(tmp, 0, 0, w, h, 0, 0, 1, 1,
                                   GdkPixbuf.InterpType.BILINEAR, 255)
            self.preview = tmp
        preview_filename = prefix + '_prev.png'
        logger.debug("Saving brush preview to %r", preview_filename)
        lib.pixbuf.save(self.preview, preview_filename, "png")
        # Save brush settings
        brushinfo = self.brushinfo.clone()
        settings_filename = prefix + '.myb'
        logger.debug("Saving brush settings to %r", settings_filename)
        with open(settings_filename, 'w') as settings_fp:
            settings_fp.write(brushinfo.save_to_string())
        # Record metadata
        self._remember_mtimes()

    def delete_from_disk(self):
        """Tries to delete the files for this brush from disk.

        :rtype: bool

        Returns True if the disk files can no longer be loaded. Stock brushes
        cannot be deleted, but if a user brush is hiding a stock brush with the
        same name, then although this method will remove the files describing
        the user brush, the stock brush is left intact. In this case, False is
        returned (because a load() attempt will now load the stock brush - and
        in fact has just done so).

        """

        prefix = os.path.join(self.bm.user_brushpath, self.name)
        if os.path.isfile(prefix + '.myb'):
            os.remove(prefix + '_prev.png')
            os.remove(prefix + '.myb')
            try:
                self.load()
            except IOError:
                # Files are no longer there, and no stock files with the
                # same name could be loaded.
                return True
            else:
                # User brush was hiding a stock brush with the same name.
                return False
        # Stock brushes cannot be deleted.
        return False

    ## Loading and reloading

    def load(self):
        """Loads the brush's preview and settings from disk."""
        if self.name is None:
            warn("Attempt to load an unnamed brush, don't do that.",
                 RuntimeWarning, 2)
            return
        self._load_preview()
        self._load_settings()

    def _load_preview(self):
        """Loads the brush preview as pixbuf into the brush."""
        assert self.name
        prefix = self._get_fileprefix()
        filename = prefix + '_prev.png'
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
        except Exception:
            logger.exception("Failed to load preview pixbuf, will fall back "
                             "to default")
            pixbuf = None
        self._preview = pixbuf
        self._remember_mtimes()

    def _load_settings(self):
        """Loads the brush settings/dynamics from disk."""
        prefix = self._get_fileprefix()
        filename = prefix + '.myb'
        with open(filename) as fp:
            brushinfo_str = fp.read()
        try:
            self._brushinfo.load_from_string(brushinfo_str)
        except Exception as e:
            logger.warning('Failed to load brush %r: %s', filename, e)
            self._brushinfo.load_defaults()
        self._remember_mtimes()
        self._settings_loaded = True
        if self.bm.is_in_brushlist(self):  # FIXME: get rid of this check
            self._brushinfo.set_string_property("parent_brush_name", None)
        self.persistent = True

    def _has_changed_on_disk(self):
        prefix = self._get_fileprefix()
        if self._preview_mtime != os.path.getmtime(prefix + '_prev.png'):
            return True
        if self._settings_mtime != os.path.getmtime(prefix + '.myb'):
            return True
        return False

    def reload_if_changed(self):
        if self._settings_mtime is None:
            return
        if self._preview_mtime is None:
            return
        if not self.name:
            return
        if not self._has_changed_on_disk():
            return False
        logger.info('Brush %r has changed on disk, reloading it.',
                    self.name)
        self.load()
        return True

    def _ensure_settings_loaded(self):
        """Ensures the brush's settings are loaded, if persistent"""
        if self.persistent and not self._settings_loaded:
            logger.debug("Loading %r...", self)
            self.load()
            assert self._settings_loaded


class InvalidBrushpack (Exception):
    """Raised when brushpacks cannot be imported."""


## Module testing

if __name__ == '__main__':
    import doctest
    doctest.testmod()
