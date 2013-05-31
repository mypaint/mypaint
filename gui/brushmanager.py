# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""File management for brushes and brush groups.
"""

## Imports

import os, zipfile
from os.path import basename
import urllib
from warnings import warn
import logging

from gettext import gettext as _
import gtk2compat
import gtk
from gtk import gdk # only for gdk.pixbuf

import dialogs
from lib.brush import BrushInfo
from lib.observable import event


## Public module constants

PREVIEW_W = 128
PREVIEW_H = 128
FOUND_BRUSHES_GROUP = 'lost&found'
DELETED_BRUSH_GROUP = 'deleted'
FAVORITES_BRUSH_GROUP = 'favorites'


## Internal module constants

_DEFAULT_STARTUP_GROUP = 'set#2'  # Suggestion only (FIXME: no effect?)
_DEFAULT_BRUSH = 'deevad/2B_pencil'  # TODO: phase out and use heuristics?
_DEFAULT_ERASER = 'deevad/kneaded_eraser_large' # TODO: -----------"---------
_DEVBRUSH_NAME_PREFIX = "devbrush_"
_BRUSH_HISTORY_NAME_PREFIX = "history_"
_BRUSH_HISTORY_SIZE = 5
_NUM_BRUSHKEYS = 10

logger = logging.getLogger(__name__)

## Helper functions

def _devbrush_quote(device_name, prefix=_DEVBRUSH_NAME_PREFIX):
    """Converts a device name to something safely storable on the disk

    Quotes an arbitrary device name for use as the basename of a
    device-specific brush.

        >>> _devbrush_quote(u'Heavy Metal Umlaut D\u00ebvice')
        'devbrush_Heavy+Metal+Umlaut+D%C3%ABvice'
        >>> _devbrush_quote(u'unsafe/device\u005Cname') # U+005C == backslash
        'devbrush_unsafe%2Fdevice%5Cname'

    Hopefully this is OK for Windows, UNIX and Mac OS X names.
    """
    device_name = unicode(device_name)
    u8bytes = device_name.encode("utf-8")
    quoted = urllib.quote_plus(u8bytes, safe='')
    return prefix + quoted


def _devbrush_unquote(devbrush_name, prefix=_DEVBRUSH_NAME_PREFIX):
    """Unquotes a device name

    Unquotes the basename of a devbrush for use when matching device names.

        >>> expected = "My sister was bitten by a m\u00f8\u00f8se..."
        >>> quoted = 'devbrush_My+sister+was+bitten+by+a+m%5Cu00f8%5Cu00f8se...'
        >>> _devbrush_unquote(quoted) == expected
        True
    """
    devbrush_name = str(devbrush_name)
    assert devbrush_name.startswith(prefix)
    quoted = devbrush_name[len(prefix):]
    u8bytes = urllib.unquote_plus(quoted)
    return unicode(u8bytes.decode("utf-8"))


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
         }
    return d.get(name, name)

def _parse_order_conf(file_content):
    """Parse order.conf file data.

    Returns a dict of the form ``{'group1' : ['brush1', 'brush2'],
    'group2' : ['brush3']}``.

    """
    groups = {}
    curr_group = FOUND_BRUSHES_GROUP
    lines = file_content.replace('\r', '\n').split('\n')
    for line in lines:
        name = line.strip().decode('utf-8')
        if name.startswith('#') or not name:
            continue
        if name.startswith('Group: '):
            curr_group = name[7:]
            if curr_group not in groups:
                groups[curr_group] = []
            continue
        groups.setdefault(curr_group, [])
        if name in groups[curr_group]:
            logger.warning('%r: brush appears twice in the same group, ignored'
                           % (name,))
            continue
        groups[curr_group].append(name)
    return groups


## Class definitions


class BrushManager (object):
    """Brush manager, responsible for groups of brushes."""

    ## Initialization

    def __init__(self, stock_brushpath, user_brushpath, app):
        """Initialize, with paths and a ref to the main app."""
        super(BrushManager, self).__init__()
        self.stock_brushpath = stock_brushpath
        self.user_brushpath = user_brushpath
        self.app = app

        #: The selected brush, as a ManagedBrush. Its settings are
        #: automatically reflected into the working brush engine brush when
        #: it changes.
        self.selected_brush = None

        self.groups = {} #: Lists of ManagedBrushes, keyed by group name
        self.contexts = [] #: Brush keys, indexed by keycap digit number
        self.active_groups = [] #: Active groups: a list of group names
        self.brush_by_device = {} #: Device name to brish mapping.

        #: Slot used elsewhere for storing the ManagedBrush corresponding to
        #: the most recently saved or restored "context", a.k.a. brush key.
        self.selected_context = None

        if not os.path.isdir(self.user_brushpath):
            os.mkdir(self.user_brushpath)
        self._load_groups()

        # Retrieve which groups were last open, or default to a nice/sane set.
        last_active_groups = app.preferences['brushmanager.selected_groups']
        if not last_active_groups:
            if _DEFAULT_STARTUP_GROUP in self.groups:
                last_active_groups = [_DEFAULT_STARTUP_GROUP]
            elif self.groups:
                group_names = self.groups.keys()
                group_names.sort()
                last_active_groups = [group_names[0]]
            else:
                last_active_groups = []
        for group in reversed(last_active_groups):
            if group in self.groups:
                brushes = self.get_group_brushes(group, make_active=True)

        # Brush order saving when that changes.
        self.brushes_changed += self._brushes_modified_cb

        # Update the history at the end of each definite input stroke.
        stroke_end_cb = self._input_stroke_ended_cb
        self.app.doc.input_stroke_ended_observers.append(stroke_end_cb)

    def _load_groups(self):
        """Initial loading of groups from disk, initializing them.

        Handles initial loading of the brushkey brushes, the painting gistory
        brushes, and all groups.

        """
        self.contexts = [None for i in xrange(_NUM_BRUSHKEYS)]
        self.history = [None for i in xrange(_BRUSH_HISTORY_SIZE)]

        brush_by_name = {}
        def get_brush(name, **kwargs):
            if name not in brush_by_name:
                b = ManagedBrush(self, name, persistent=True, **kwargs)
                brush_by_name[name] = b
            return brush_by_name[name]

        def read_groups(filename):
            groups = {}
            if os.path.exists(filename):
                groups = _parse_order_conf(open(filename).read())
                # replace brush names with ManagedBrush instances
                for group, names in groups.items():
                    brushes = []
                    for name in names:
                        try:
                            b = get_brush(name)
                        except IOError, e:
                            logger.warn('%r (removed from group)' % (e,))
                            continue
                        brushes.append(b)
                    groups[group] = brushes
            return groups

        # Three-way-merge of brush groups (for upgrading)
        base = read_groups(os.path.join(self.user_brushpath, 'order_default.conf'))
        our = read_groups(os.path.join(self.user_brushpath, 'order.conf'))
        their = read_groups(os.path.join(self.stock_brushpath, 'order.conf'))

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
            data = open(os.path.join(self.stock_brushpath, 'order.conf')).read()
            open(os.path.join(self.user_brushpath,  'order_default.conf'), 'w').write(data)

        # check for brushes that are in the brush directory, but not in any group

        def listbrushes(path):
            # Return a list of brush names relative to path, using
            # slashes for subirectories on all platforms.
            path += '/'
            l = []
            assert isinstance(path, unicode) # make sure we get unicode filenames 
            for name in os.listdir(path):
                assert isinstance(name, unicode)
                if name.endswith('.myb'):
                    l.append(name[:-4])
                elif os.path.isdir(path+name):
                    for name2 in listbrushes(path+name):
                        l.append(name + '/' + name2)
            return l

        # Distinguish between brushes in the brushlist and those that are not;
        # handle lost-and-found ones.
        for name in listbrushes(self.stock_brushpath) \
                  + listbrushes(self.user_brushpath):
            if name.startswith('context'):
                b = get_brush(name)
                i = int(name[-2:])
                self.contexts[i] = b
            elif name.startswith(_DEVBRUSH_NAME_PREFIX):
                b = get_brush(name)
                device_name = _devbrush_unquote(name)
                self.brush_by_device[device_name] = b
            elif name.startswith(_BRUSH_HISTORY_NAME_PREFIX):
                b = get_brush(name)
                i_str = name.replace(_BRUSH_HISTORY_NAME_PREFIX, '')
                i = int(i_str)
                self.history[i] = b
            else:
                # normal brush that will appear in the brushlist
                b = get_brush(name)
                if not [True for group in our.itervalues() if b in group]:
                    brushes = self.groups.setdefault(FOUND_BRUSHES_GROUP, [])
                    brushes.insert(0, b)

        # Sensible defaults for brushkeys and history: clone the first few
        # brushes from a normal group if we need to and if we can.
        # Try the default startup group first.
        default_group = self.groups.get(_DEFAULT_STARTUP_GROUP, None)

        # Otherwise, use the biggest group to minimise the chance
        # of repetition.
        if default_group is None:
            groups_by_len = [(len(g),n,g) for n,g in self.groups.items()]
            groups_by_len.sort()
            _len, _name, default_group = groups_by_len[-1]

        # Populate blank entries.
        for i in xrange(_NUM_BRUSHKEYS):
            if self.contexts[i] is None:
                idx = (i+9) % 10 # keyboard order
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
        group is renamed, deleted, or created.  It's invoked when EITHER
        self.group OR self.active_groups change.
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
        last_used_devbrush = app.preferences.get('devbrush.last_used', None)
        initial_brush = self.brush_by_device.get(last_used_devbrush, None)
        # Otherwise, initialise from the old selected_brush setting
        if initial_brush is None:
            last_active_name = app.preferences['brushmanager.selected_brush']
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
            group_names = self.groups.keys()
            group_names.sort()
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


    ## Brushpack import and export


    def import_brushpack(self, path, window):
        """Import a brushpack from a zipfile, with confirmation dialogs.

        :param path: Brush pack zipfile path
        :type path: str
        :param window: Parent window, for dialogs to set.
        :type window: GtkWindow

        """

        zip = zipfile.ZipFile(path)
        names = zip.namelist()
        # zipfile does utf-8 decoding on its own; this is just to make
        # sure we have only unicode objects as brush names.
        names = [s.decode('utf-8') for s in names]

        readme = None
        if 'readme.txt' in names:
            readme = zip.read('readme.txt')

        assert 'order.conf' in names, 'invalid brushpack: order.conf missing'
        groups = _parse_order_conf(zip.read('order.conf'))

        new_brushes = []
        for brushes in groups.itervalues():
            for brush in brushes:
                if brush not in new_brushes:
                    new_brushes.append(brush)
        logger.info("%d different brushes found in order.conf of brushpack"
                    % (len(new_brushes),))

        # Validate file content. The names in order.conf and the
        # brushes found in the zip must match. This should catch
        # encoding screwups, everything should be an unicode object.
        for brush in new_brushes:
            assert brush + '.myb' in names, 'invalid brushpack: brush %r in order.conf does not exist in zip' % brush
        for name in names:
            if name.endswith('.myb'):
                brush = name[:-4]
                assert brush in new_brushes, 'invalid brushpack: brush %r exists in zip, but not in order.conf' % brush

        if readme:
            answer = dialogs.confirm_brushpack_import(basename(path), window, readme)
            if answer == gtk.RESPONSE_REJECT:
                return

        do_overwrite = False
        do_ask = True
        renamed_brushes = {}
        final_groups = []
        for groupname, brushes in groups.iteritems():
            managed_brushes = self.get_group_brushes(groupname)
            self.set_active_groups([groupname])
            if managed_brushes:
                answer = dialogs.confirm_rewrite_group(
                    window, translate_group_name(groupname), translate_group_name(DELETED_BRUSH_GROUP))
                if answer == dialogs.CANCEL:
                    return
                elif answer == dialogs.OVERWRITE_THIS:
                    self.delete_group(groupname)
                elif answer == dialogs.DONT_OVERWRITE_THIS:
                    i = 0
                    old_groupname = groupname
                    while groupname in self.groups:
                        i += 1
                        groupname = old_groupname + '#%d' % i
                managed_brushes = self.get_group_brushes(groupname, make_active=True)

            final_groups.append(groupname)

            for brushname in brushes:
                # extract the brush from the zip
                assert (brushname + '.myb') in zip.namelist()
                # Support for utf-8 ZIP filenames that don't have the utf-8 bit set.
                brushname_utf8 = brushname.encode('utf-8')
                try:
                    myb_data = zip.read(brushname + '.myb')
                except KeyError:
                    myb_data = zip.read(brushname_utf8 + '.myb')
                try:
                    preview_data = zip.read(brushname + '_prev.png')
                except KeyError:
                    preview_data = zip.read(brushname_utf8 + '_prev.png')
                # in case we have imported that brush already in a previous group, but decided to rename it
                if brushname in renamed_brushes:
                    brushname = renamed_brushes[brushname]
                # possibly ask how to import the brush file (if we didn't already)
                b = self.get_brush_by_name(brushname)
                if brushname in new_brushes:
                    new_brushes.remove(brushname)
                    if b:
                        existing_preview_pixbuf = b.preview
                        if do_ask:
                            answer = dialogs.confirm_rewrite_brush(window, brushname, existing_preview_pixbuf, preview_data)
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
                            brushname = brushname_old + '#%d' % i
                            renamed_brushes[brushname_old] = brushname
                            b = self.get_brush_by_name(brushname)

                    if not b:
                        b = ManagedBrush(self, brushname)

                    # write to disk and reload brush (if overwritten)
                    prefix = b._get_fileprefix(saving=True)
                    myb_f = open(prefix + '.myb', 'w')
                    myb_f.write(myb_data)
                    myb_f.close()
                    preview_f = open(prefix + '_prev.png', 'wb')
                    preview_f.write(preview_data)
                    preview_f.close()
                    b.load()
                # finally, add it to the group
                if b not in managed_brushes:
                    managed_brushes.append(b)
                self.brushes_changed(managed_brushes)

        if DELETED_BRUSH_GROUP in self.groups:
            # remove deleted brushes that are in some group again
            self.delete_group(DELETED_BRUSH_GROUP)
        self.set_active_groups(final_groups)


    def export_group(self, group, filename):
        """Exports a group to a brushpack zipfile."""
        zip = zipfile.ZipFile(filename, mode='w')
        brushes = self.get_group_brushes(group)
        order_conf = 'Group: %s\n' % group.encode('utf-8')
        for brush in brushes:
            prefix = brush._get_fileprefix()
            zip.write(prefix + '.myb', brush.name + '.myb')
            zip.write(prefix + '_prev.png', brush.name + '_prev.png')
            order_conf += brush.name.encode('utf-8') + '\n'
        zip.writestr('order.conf', order_conf)
        zip.close()


    ## Brush lookup / access

    def get_brush_by_name(self, name):
        """Gets a ManagedBrush by its name.

        Slow method, should not be called too often.

        """
        # FIXME: speed up, use a dict.
        for group, brushes in self.groups.iteritems():
            for b in brushes:
                if b.name == name:
                    return b


    def is_in_brushlist(self, brush):
        """Returns whether this brush is in some brush group's list."""
        for group, brushes in self.groups.iteritems():
            if brush in brushes:
                return True
        return False

    def get_parent_brush(self, brush=None, brushinfo=None):
        """Gets the parent `ManagedBrush` for a brush or a `BrushInfo`.
        """
        if brush is not None:
            brushinfo = brush.brushinfo
        if brushinfo is None:
            raise RuntimeError, "One of `brush` or `brushinfo` must be defined."
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
        """Save the user's chose brush order to the config.
        """
        f = open(os.path.join(self.user_brushpath, 'order.conf'), 'w')
        f.write('# this file saves brush groups and order\n')
        for group, brushes in self.groups.iteritems():
            f.write('Group: %s\n' % group.encode('utf-8'))
            for b in brushes:
                f.write(b.name.encode('utf-8') + '\n')
        f.close()


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
            brush = self.get_parent_brush(brush=brush)
            if not brush:
                # no parent, select an empty brush instead
                brush = ManagedBrush(self)

        self.selected_brush = brush
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
        clone = ManagedBrush(self, name, persistent=False)
        clone.brushinfo = self.app.brush.clone()
        clone.preview = self.selected_brush.preview
        parent = self.selected_brush.name
        clone.brushinfo.set_string_property("parent_brush_name", parent)
        return clone

    ## Device-specific brushes

    def store_brush_for_device(self, device_name, managed_brush):
        """Records a brush as associated with an input device.

        :param device_name: name of an input device
        :type device_name: str
        :param managed_brush: the brush to associate
        :type managed_brush: MnagedBrush

        Normally the brush will be cloned first, since it will be given a new
        name. However, if the brush has a 'name' attribute of None, it will
        *not* be cloned and just modified in place and stored.

        """
        brush = managed_brush
        if brush.name is not None:
            brush = brush.clone()
        brush.name = unicode(_devbrush_quote(device_name))
        self.brush_by_device[device_name] = brush

    def fetch_brush_for_device(self, device_name):
        """Fetches the brush associated with an input device."""
        devbrush_name = _devbrush_quote(device_name)
        brush = self.brush_by_device.get(device_name, None)
        return brush

    def save_brushes_for_devices(self):
        """Saves the device/brush associations to disk."""
        for device_name, devbrush in self.brush_by_device.iteritems():
            devbrush.save()

    ## Brush history

    def _input_stroke_ended_cb(self, *junk):
        """Update brush usage history at the end of an input stroke."""
        # Remove instances of the working brush from the history
        b = self.app.brush
        b_parent = b.get_string_property("parent_brush_name")
        for i, h in enumerate(self.history):
            h_parent = h.brushinfo.get_string_property("parent_brush_name")
            # Possibly we should use a tighter equality check than this, but
            # then we'd need icons showing modifications from the parent.
            if b_parent == h_parent:
                del self.history[i]
                break
        # Append the working brush to the history, and trim it to length
        h = ManagedBrush(self, name=None, persistent=False)
        h.brushinfo = b.clone()
        h.preview = self.selected_brush.preview
        self.history.append(h)
        while len(self.history) > _BRUSH_HISTORY_SIZE:
            del self.history[0]
        # Rename the history brushes so they save to the right files.
        for i, h in enumerate(self.history):
            h.name = u"%s%d" % (_BRUSH_HISTORY_NAME_PREFIX, i)


    def save_brush_history(self):
        """Saves the brush usage history to disk."""
        for brush in self.history:
            brush.save()

    ## Brush groups

    def set_active_groups(self, groups):
        """Set active groups.

        :param groups: List of group names.
        :type group: list of str

        """
        self.active_groups = groups
        self.app.preferences['brushmanager.selected_groups'] = groups
        self.groups_changed()


    def get_group_brushes(self, group, make_active=False):
        """Get a group's brushes, optionally making the group active.

        If the group does not exist, it will be created.

        :param group: Name of the group to fetch
        :type group: str
        :param make_active: If true, add the group to the active groups list.
        :type make_active: bool
        :rtype: list of `ManagedBrush`es, owned by the BrushManager

        """
        if group not in self.groups:
            brushes = []
            self.groups[group] = brushes
            self.groups_changed()
            self.save_brushorder()
        if make_active and group not in self.active_groups:
            self.set_active_groups([group] + self.active_groups)
        return self.groups[group]


    def create_group(self, new_group, make_active=True):
        """Creates a new brush group, optionally making it active.

        :param group: Name of the group to create
        :type group: str
        :param make_active: If true, add the group to the active groups list.
        :type make_active: bool
        :rtype: empty list, owned by the BrushManager

        Returns the newly created group as a(n empty) list.

        """
        return self.get_group_brushes(new_group, make_active)


    def rename_group(self, old_group, new_group):
        """Renames a group.

        :param old_group: Name of the group to assign the new name to.
        :type old_group: str
        :param new_group: New name for the group.
        :type new_group: str

        """
        was_active = (old_group in self.active_groups)
        brushes = self.create_group(new_group, make_active=was_active)
        brushes += self.groups[old_group]
        self.delete_group(old_group)


    def delete_group(self, group):
        """Deletes a group.

        :param group: Name of the group to delete
        :type group: str

        Oprhaned brushes will be placed into `DELETED_BRUSH_GROUP`, which
        will be created if necessary.

        """

        homeless_brushes = self.groups[group]
        del self.groups[group]
        if group in self.active_groups:
            self.active_groups.remove(group)

        for brushes in self.groups.itervalues():
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
    """User-facing representation of a brush, associated with the BrushManager

    Managed brushes have a name, a preview image, and brush settings (which do
    not need to be loaded up front). They cannot be selected or painted with
    directly, but their settings can be loaded into the running app: see
    `Brushmanager.select_brush()`.

    """
    def __init__(self, brushmanager, name=None, persistent=False):
        super(ManagedBrush, self).__init__()
        self.bm = brushmanager
        self._preview = None
        self.name = name
        self._brushinfo = BrushInfo()

        #: If True, this brush is stored in the filesystem.
        self.persistent = persistent

        #: If True, this brush is fully initialized, ready to paint with.
        self.settings_loaded = False

        # Change detection for on-disk files.
        self.settings_mtime = None
        self.preview_mtime = None

        if persistent:
            # Files are loaded later, but throw an exception now if they
            # don't exist.
            self._get_fileprefix()


    ## Preview image: loaded on demand

    def get_preview(self):
        # load preview pixbuf on demand
        if self._preview is None and self.name:
            self._load_preview()
        if self._preview is None:
            # When does this happen?
            self.preview = gtk2compat.gdk.pixbuf.new(gdk.COLORSPACE_RGB,
                            False, 8, PREVIEW_W, PREVIEW_H)
            self.preview.fill(0xffffffff) # white
        return self._preview

    def set_preview(self, pixbuf):
        self._preview = pixbuf

    preview = property(get_preview, set_preview)


    ## Brush settings: loaded on demand

    def get_brushinfo(self):
        if self.persistent and not self.settings_loaded:
            self._load_settings()
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
        "Creates a new brush with all the settings of this brush, assigning it a new name"
        clone = ManagedBrush(self.bm)
        self.clone_into(clone, name=name)
        return clone

    def clone_into(self, target, name):
        "Copies all brush settings into another brush, giving it a new name"
        if not self.settings_loaded:   # XXX refactor
            self.load()
        target.brushinfo = self.brushinfo.clone()
        if self.bm.is_in_brushlist(self): # FIXME: get rid of this check!
            target.brushinfo.set_string_property("parent_brush_name", self.name)
        target.preview = self.preview
        target.name = name

    ## File save/load helpers

    def _get_fileprefix(self, saving=False):
        """Returns the filesystem prefix to use when saving or loading.

        :param saving: caller wants a prefix to save to
        :type saving: bool
        :rtype: unicode

        Files are stored with the returned prefix, with the extension ``.myb``
        for brush data and ``_prev.myb`` for preview images.  If `saving` is
        true, intermediate directories will be created, and the returned prefix
        will always contain the user brushpath. Otherwise the prefix you get
        depends on whether a stock brush exists and a user brush wit the same
        name does not. See also `delete_from_disk()`.

        """
        prefix = 'b'
        if os.path.realpath(self.bm.user_brushpath) == os.path.realpath(self.bm.stock_brushpath):
            # working directly on brush collection, use different prefix
            prefix = 's'

        # Construct a new, unique name if the brush is not yet named
        if not self.name:
            i = 0
            while 1:
                self.name = u'%s%03d' % (prefix, i)
                a = os.path.join(self.bm.user_brushpath, self.name + '.myb')
                b = os.path.join(self.bm.stock_brushpath, self.name + '.myb')
                if not os.path.isfile(a) and not os.path.isfile(b):
                    break
                i += 1
        assert isinstance(self.name, unicode)

        # Always save to the user brush path.
        prefix = os.path.join(self.bm.user_brushpath, self.name)
        if saving: 
            if '/' in self.name:
                d = os.path.dirname(prefix)
                if not os.path.isdir(d):
                    os.makedirs(d)
            return prefix

        # Loading: try user first, then stock
        if not os.path.isfile(prefix + '.myb'):
            prefix = os.path.join(self.bm.stock_brushpath, self.name)
        if not os.path.isfile(prefix + '.myb'):
            raise IOError, 'brush "' + self.name + '" not found'
        return prefix


    def _remember_mtimes(self):
        prefix = self._get_fileprefix()
        self.preview_mtime = os.path.getmtime(prefix + '_prev.png')
        self.settings_mtime = os.path.getmtime(prefix + '.myb')

    ## Saving and deleting

    def save(self):
        prefix = self._get_fileprefix(saving=True)

        if self.preview.get_has_alpha():
            # remove it (previous mypaint versions would display an empty image)
            w, h = PREVIEW_W, PREVIEW_H
            tmp = gtk2compat.gdk.pixbuf.new(gdk.COLORSPACE_RGB, False,
                                            8, w, h)
            tmp.fill(0xffffffff)
            self.preview.composite(tmp, 0, 0, w, h, 0, 0, 1, 1,
                                   gdk.INTERP_BILINEAR, 255)
            self.preview = tmp

        gtk2compat.gdk.pixbuf.save(self.preview, prefix + '_prev.png', 'png')
        brushinfo = self.brushinfo.clone()
        open(prefix + '.myb', 'w').write(brushinfo.save_to_string())
        self._remember_mtimes()


    def delete_from_disk(self):
        """Tries to delete the files for this brush from disk.

        :rtype: boolean

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
        pixbuf = gdk.pixbuf_new_from_file(filename)
        self._preview = pixbuf
        self._remember_mtimes()


    def _load_settings(self):
        """Loads the brush settings/dynamics from disk."""
        prefix = self._get_fileprefix()
        filename = prefix + '.myb'
        brushinfo_str = open(filename).read()
        try:
            self._brushinfo.load_from_string(brushinfo_str)
        except BrushInfo.ParseError, e:
            logger.warning('Failed to load brush %r: %s' % (filename, e))
            self._brushinfo.load_defaults()
        self._remember_mtimes()
        self.settings_loaded = True
        if self.bm.is_in_brushlist(self): # FIXME: get rid of this check
            self._brushinfo.set_string_property("parent_brush_name", None)
        self.persistent = True


    def _has_changed_on_disk(self):
        prefix = self._get_fileprefix()
        if self.preview_mtime != os.path.getmtime(prefix + '_prev.png'): return True
        if self.settings_mtime != os.path.getmtime(prefix + '.myb'): return True
        return False

    def reload_if_changed(self):
        if self.settings_mtime is None: return
        if self.preview_mtime is None: return
        if not self.name: return
        if not self._has_changed_on_disk(): return False
        logger.info('Brush %r has changed on disk, reloading it.'
                    % (self.name,))
        self.load()
        return True

if __name__ == '__main__':
    import doctest
    doctest.testmod()

