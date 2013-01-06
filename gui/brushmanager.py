# This file is part of MyPaint.
# Copyright (C) 2009 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""
This module does file management for brushes and brush groups.
"""

import pygtkcompat
import dialogs
import gtk
from gtk import gdk # only for gdk.pixbuf
from gettext import gettext as _
import os, zipfile
from os.path import basename
import urllib
from lib.brush import BrushInfo
from warnings import warn

preview_w = 128
preview_h = 128

DEFAULT_STARTUP_GROUP = 'set#2'  # Suggestion only (FIXME: no effect?)
DEFAULT_BRUSH = 'deevad/2B_pencil'  # TODO: phase out and use heuristics?
DEFAULT_ERASER = 'deevad/kneaded_eraser_large'  # TODO: ---------------"--------------
FOUND_BRUSHES_GROUP = 'lost&found'
DELETED_BRUSH_GROUP = 'deleted'
FAVORITES_BRUSH_GROUP = 'favorites'
DEVBRUSH_NAME_PREFIX = "devbrush_"
BRUSH_HISTORY_NAME_PREFIX = "history_"
BRUSH_HISTORY_SIZE = 5
NUM_BRUSHKEYS = 10

def devbrush_quote(device_name, prefix=DEVBRUSH_NAME_PREFIX):
    """
    Quotes an arbitrary device name for use as the basename of a
    device-specific brush.

        >>> devbrush_quote(u'Heavy Metal Umlaut D\u00ebvice')
        'devbrush_Heavy+Metal+Umlaut+D%C3%ABvice'
        >>> devbrush_quote(u'unsafe/device\u005Cname') # U+005C == backslash
        'devbrush_unsafe%2Fdevice%5Cname'

    Hopefully this is OK for Windows, UNIX and Mac OS X names.
    """
    device_name = unicode(device_name)
    u8bytes = device_name.encode("utf-8")
    quoted = urllib.quote_plus(u8bytes, safe='')
    return prefix + quoted

def devbrush_unquote(devbrush_name, prefix=DEVBRUSH_NAME_PREFIX):
    """
    Unquotes the basename of a devbrush for use when matching device names.

        >>> expected = "My sister was bitten by a m\u00f8\u00f8se..."
        >>> quoted = 'devbrush_My+sister+was+bitten+by+a+m%5Cu00f8%5Cu00f8se...'
        >>> devbrush_unquote(quoted) == expected
        True
    """
    devbrush_name = str(devbrush_name)
    assert devbrush_name.startswith(prefix)
    quoted = devbrush_name[len(prefix):]
    u8bytes = urllib.unquote_plus(quoted)
    return unicode(u8bytes.decode("utf-8"))

def translate_group_name(name):
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

def parse_order_conf(file_content):
    # parse order.conf file returing a dict like this:
    # {'group1' : ['brush1', 'brush2'], 'group2' : ['brush3']}
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
            print name + ': Warning: brush appears twice in the same group, ignored'
            continue
        groups[curr_group].append(name)
    return groups

class BrushManager:
    def __init__(self, stock_brushpath, user_brushpath, app):
        self.stock_brushpath = stock_brushpath
        self.user_brushpath = user_brushpath
        self.app = app

        self.selected_brush = None
        self.groups = {}
        self.contexts = []
        self.active_groups = []
        self.loaded_groups = []
        self.brush_by_device = {} # should be save/loaded too?
        self.selected_context = None

        self.selected_brush_observers = []
        self.groups_observers = [] # for both self.groups and self.active_groups
        self.brushes_observers = [] # for all brushlists inside groups

        if not os.path.isdir(self.user_brushpath):
            os.mkdir(self.user_brushpath)
        self.load_groups()

        # Retreive which groups were last open, or default to a nice/sane set.
        last_active_groups = self.app.preferences['brushmanager.selected_groups']
        if not last_active_groups:
            if DEFAULT_STARTUP_GROUP in self.groups:
                last_active_groups = [DEFAULT_STARTUP_GROUP]
            elif self.groups:
                group_names = self.groups.keys()
                group_names.sort()
                last_active_groups = [group_names[0]]
            else:
                last_active_groups = []
        for group in reversed(last_active_groups):
            if group in self.groups:
                brushes = self.get_group_brushes(group, make_active=True)

        self.brushes_observers.append(self.brushes_modified_cb)

        self.app.doc.input_stroke_ended_observers.append(self.input_stroke_ended_cb)

    def select_initial_brush(self):
        initial_brush = None
        # If we recorded which devbrush was last in use, restore it and assume
        # that most of the time the user will continue to work with the same
        # brush and its settings.
        last_used_devbrush = self.app.preferences.get('devbrush.last_used', None)
        initial_brush = self.brush_by_device.get(last_used_devbrush, None)
        # Otherwise, initialise from the old selected_brush setting
        if initial_brush is None:
            last_active_name = self.app.preferences['brushmanager.selected_brush']
            if last_active_name is not None:
                initial_brush = self.get_brush_by_name(last_active_name)
        # Fallback
        if initial_brush is None:
            initial_brush = self.get_default_brush()
        self.select_brush(initial_brush)

    def get_matching_brush(self, name=None, keywords=None,
                           favored_group=DEFAULT_STARTUP_GROUP,
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
        return self.get_matching_brush(name=DEFAULT_BRUSH,
                                keywords=["pencil", "charcoal", "sketch"])


    def get_default_eraser(self):
        """Returns a suitable default eraser brush."""
        return self.get_matching_brush(name=DEFAULT_ERASER,
                                keywords=["eraser", "kneaded", "smudge"],
                                fallback_eraser=1.0)



    def load_groups(self):
        self.contexts = [None for i in xrange(NUM_BRUSHKEYS)]
        self.history = [None for i in xrange(BRUSH_HISTORY_SIZE)]

        brush_by_name = {}
        def get_brush(name, **kwargs):
            if name not in brush_by_name:
                b = ManagedBrush(self, name, persistent=True, **kwargs)
                brush_by_name[name] = b
            return brush_by_name[name]

        def read_groups(filename):
            groups = {}
            if os.path.exists(filename):
                groups = parse_order_conf(open(filename).read())
                # replace brush names with ManagedBrush instances
                for group, names in groups.items():
                    brushes = []
                    for name in names:
                        try:
                            b = get_brush(name)
                        except IOError, e:
                            print e, '(removed from group)'
                            continue
                        brushes.append(b)
                    groups[group] = brushes
            return groups

        # tree-way-merge of brush groups (for upgrading)
        base  = read_groups(os.path.join(self.user_brushpath,  'order_default.conf'))
        our   = read_groups(os.path.join(self.user_brushpath,  'order.conf'))
        their = read_groups(os.path.join(self.stock_brushpath, 'order.conf'))

        if not our:
            # order.conf missing, restore stock order even if order_default.conf exists
            base = {}

        if base == their:
            self.groups = our
        else:
            print 'Merging upstream brush changes into your collection.'
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
        for name in listbrushes(self.stock_brushpath) + listbrushes(self.user_brushpath):
            if name.startswith('context'):
                b = get_brush(name)
                i = int(name[-2:])
                self.contexts[i] = b
            elif name.startswith(DEVBRUSH_NAME_PREFIX):
                b = get_brush(name)
                device_name = devbrush_unquote(name)
                self.brush_by_device[device_name] = b
            elif name.startswith(BRUSH_HISTORY_NAME_PREFIX):
                b = get_brush(name)
                i_str = name.replace(BRUSH_HISTORY_NAME_PREFIX, '')
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
        default_group = self.groups.get(DEFAULT_STARTUP_GROUP, None)

        # Otherwise, use the biggest group to minimise the chance
        # of repetition.
        if default_group is None:
            groups_by_len = [(len(g),n,g) for n,g in self.groups.items()]
            groups_by_len.sort()
            _len, _name, default_group = groups_by_len[-1]

        # Populate blank entries.
        for i in xrange(NUM_BRUSHKEYS):
            if self.contexts[i] is None:
                idx = (i+9) % 10 # keyboard order
                c_name = unicode('context%02d') % i
                c = ManagedBrush(self, name=c_name, persistent=False)
                group_idx = idx % len(default_group)
                b = default_group[group_idx]
                b.clone_into(c, c_name)
                self.contexts[i] = c
        for i in xrange(BRUSH_HISTORY_SIZE):
            if self.history[i] is None:
                h_name = unicode('%s%d') % (BRUSH_HISTORY_NAME_PREFIX, i)
                h = ManagedBrush(self, name=h_name, persistent=False)
                group_i = i % len(default_group)
                b = default_group[group_i]
                b.clone_into(h, h_name)
                self.history[i] = h

        # clean up legacy stuff
        fn = os.path.join(self.user_brushpath, 'deleted.conf')
        if os.path.exists(fn):
            os.remove(fn)

    def import_brushpack(self, path,  window):
        zip = zipfile.ZipFile(path)
        names = zip.namelist()
        # zipfile does utf-8 decoding on its own; this is just to make
        # sure we have only unicode objects as brush names.
        names = [s.decode('utf-8') for s in names]

        readme = None
        if 'readme.txt' in names:
            readme = zip.read('readme.txt')

        assert 'order.conf' in names, 'invalid brushpack: order.conf missing'
        groups = parse_order_conf(zip.read('order.conf'))

        new_brushes = []
        for brushes in groups.itervalues():
            for brush in brushes:
                if brush not in new_brushes:
                    new_brushes.append(brush)
        print len(new_brushes), 'different brushes found in order.conf of brushpack'

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
                    prefix = b.get_fileprefix(saving=True)
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
                for f in self.brushes_observers: f(managed_brushes)

        if DELETED_BRUSH_GROUP in self.groups:
            # remove deleted brushes that are in some group again
            self.delete_group(DELETED_BRUSH_GROUP)
        self.set_active_groups(final_groups)

    def export_group(self, group, filename):
        zip = zipfile.ZipFile(filename, mode='w')
        brushes = self.get_group_brushes(group)
        order_conf = 'Group: %s\n' % group.encode('utf-8')
        for brush in brushes:
            prefix = brush.get_fileprefix()
            zip.write(prefix + '.myb', brush.name + '.myb')
            zip.write(prefix + '_prev.png', brush.name + '_prev.png')
            order_conf += brush.name.encode('utf-8') + '\n'
        zip.writestr('order.conf', order_conf)
        zip.close()

    def get_brush_by_name(self, name):
        # slow method, should not be called too often
        # FIXME: speed up, use a dict.
        for group, brushes in self.groups.iteritems():
            for b in brushes:
                if b.name == name:
                    return b

    def brushes_modified_cb(self, brushes):
        self.save_brushorder()

    def save_brushorder(self):
        f = open(os.path.join(self.user_brushpath, 'order.conf'), 'w')
        f.write('# this file saves brush groups and order\n')
        for group, brushes in self.groups.iteritems():
            f.write('Group: %s\n' % group.encode('utf-8'))
            for b in brushes:
                f.write(b.name.encode('utf-8') + '\n')
        f.close()


    def input_stroke_ended_cb(self, *junk):
        """Update brush history at the end of an input stroke.
        """
        b = self.app.brush
        b_parent = b.get_string_property("parent_brush_name")
        for i, h in enumerate(self.history):
            h_parent = h.brushinfo.get_string_property("parent_brush_name")
            # Possibly we should use a tighter equality check than this, but
            # then we'd need icons showing modifications from the parent.
            if b_parent == h_parent:
                del self.history[i]
                break
        h = ManagedBrush(self, name=None, persistent=False)
        h.brushinfo = b.clone()
        h.preview = self.selected_brush.preview
        self.history.append(h)
        while len(self.history) > BRUSH_HISTORY_SIZE:
            del self.history[0]
        for i, h in enumerate(self.history):
            h.name = u"%s%d" % (BRUSH_HISTORY_NAME_PREFIX, i)

    def is_in_brushlist(self, brush):
        """Returns whether this brush is accessible through the brush selector."""
        for group, brushes in self.groups.iteritems():
            if brush in brushes:
                return True
        return False

    def select_brush(self, brush):
        """Selects a ManagedBrush, highlights it, & updates the live brush."""
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
        # Take care of updating the live brush, amongst other things
        for callback in self.selected_brush_observers:
            callback(brush, brushinfo)


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


    def clone_selected_brush(self, name):
        """
        Creates a new ManagedBrush based on the selected
        brush in the brushlist and the currently active lib.brush.
        """
        clone = ManagedBrush(self, name, persistent=False)
        clone.brushinfo = self.app.brush.clone()
        clone.preview = self.selected_brush.preview
        parent = self.selected_brush.name
        clone.brushinfo.set_string_property("parent_brush_name", parent)
        return clone

    def store_brush_for_device(self, device_name, managed_brush):
        """
        Records an existing ManagedBrush as associated with a given input device.

        Normally the brush will be cloned first, since it will be given a new
        name. However, if the ManagedBrush has a 'name' attribute of None, it
        will *not* be cloned and just modified in place and stored.
        """
        brush = managed_brush
        if brush.name is not None:
            brush = brush.clone()
        brush.name = unicode(devbrush_quote(device_name))
        self.brush_by_device[device_name] = brush

    def fetch_brush_for_device(self, device_name):
        """
        Fetches the brush associated with a particular input device name.
        """
        devbrush_name = devbrush_quote(device_name)
        brush = self.brush_by_device.get(device_name, None)
        return brush

    def save_brushes_for_devices(self):
        for device_name, devbrush in self.brush_by_device.iteritems():
            devbrush.save()

    def save_brush_history(self):
        for brush in self.history:
            brush.save()

    def set_active_groups(self, groups):
        """Set active groups."""
        self.active_groups = groups
        self.app.preferences['brushmanager.selected_groups'] = groups
        for f in self.groups_observers: f()

    def get_group_brushes(self, group, make_active=False):
        if group not in self.groups:
            brushes = []
            self.groups[group] = brushes
            for f in self.groups_observers: f()
            self.save_brushorder()
        if make_active and group not in self.active_groups:
            self.set_active_groups([group] + self.active_groups)
        return self.groups[group]

    def create_group(self, new_group, make_active=True):
        return self.get_group_brushes(new_group, make_active)

    def rename_group(self, old_group, new_group):
        was_active = (old_group in self.active_groups)
        brushes = self.create_group(new_group, make_active=was_active)
        brushes += self.groups[old_group]
        self.delete_group(old_group)

    def delete_group(self, group):
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
            for f in self.brushes_observers: f(deleted_brushes)
        for f in self.brushes_observers: f(homeless_brushes)
        for f in self.groups_observers: f()
        self.save_brushorder()


class ManagedBrush(object):
    '''Represents a brush, but cannot be selected or painted with directly.'''
    def __init__(self, brushmanager, name=None, persistent=False):
        self.bm = brushmanager
        self._preview = None
        self.name = name
        self._brushinfo = BrushInfo()
        self.persistent = persistent #: If True this brush is stored in the filesystem.
        self.settings_loaded = False  #: If True this brush is fully initialized, ready to paint with.

        self.settings_mtime = None
        self.preview_mtime = None

        if persistent:
            # we load the files later, but throw an exception now if they don't exist
            self.get_fileprefix()

    # load preview pixbuf on demand
    def get_preview(self):
        if self._preview is None and self.name:
            self._load_preview()
        if self._preview is None:
            # When does this happen?
            self.preview = pygtkcompat.gdk.pixbuf.new(gdk.COLORSPACE_RGB,
                            False, 8, preview_w, preview_h)
            self.preview.fill(0xffffffff) # white
        return self._preview
    def set_preview(self, pixbuf):
        self._preview = pixbuf
    preview = property(get_preview, set_preview)

    # load brush settings on demand
    def get_brushinfo(self):
        if self.persistent and not self.settings_loaded:
            self._load_settings()
        return self._brushinfo
    def set_brushinfo(self, brushinfo):
        self._brushinfo = brushinfo
    brushinfo = property(get_brushinfo, set_brushinfo)

    def get_display_name(self):
        """Gets a displayable name for the brush.
        """
        if self.bm.is_in_brushlist(self):  # FIXME: get rid of this check
            dname = self.name
        else:
            dname = self.brushinfo.get_string_property("parent_brush_name")
        if dname is None:
            return _("Unknown Brush")
        return dname.replace("_", " ")


    def get_fileprefix(self, saving=False):
        prefix = 'b'
        if os.path.realpath(self.bm.user_brushpath) == os.path.realpath(self.bm.stock_brushpath):
            # working directly on brush collection, use different prefix
            prefix = 's'

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
        prefix = os.path.join(self.bm.user_brushpath, self.name)
        if saving: 
            if '/' in self.name:
                d = os.path.dirname(prefix)
                if not os.path.isdir(d):
                    os.makedirs(d)
            return prefix
        if not os.path.isfile(prefix + '.myb'):
            prefix = os.path.join(self.bm.stock_brushpath, self.name)
        if not os.path.isfile(prefix + '.myb'):
            raise IOError, 'brush "' + self.name + '" not found'
        return prefix

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

    def delete_from_disk(self):
        prefix = os.path.join(self.bm.user_brushpath, self.name)
        if os.path.isfile(prefix + '.myb'):
            os.remove(prefix + '_prev.png')
            os.remove(prefix + '.myb')
            try:
                self.load()
            except IOError:
                return True # success
            else:
                return False # partial success, this brush was hiding a stock brush with the same name
        # stock brush cannot be deleted
        return False

    def remember_mtimes(self):
        prefix = self.get_fileprefix()
        self.preview_mtime = os.path.getmtime(prefix + '_prev.png')
        self.settings_mtime = os.path.getmtime(prefix + '.myb')

    def has_changed_on_disk(self):
        prefix = self.get_fileprefix()
        if self.preview_mtime != os.path.getmtime(prefix + '_prev.png'): return True
        if self.settings_mtime != os.path.getmtime(prefix + '.myb'): return True
        return False

    def save(self):
        prefix = self.get_fileprefix(saving=True)

        if self.preview.get_has_alpha():
            # remove it (previous mypaint versions would display an empty image)
            w, h = preview_w, preview_h
            tmp = pygtkcompat.gdk.pixbuf.new(gdk.COLORSPACE_RGB, False,
                                             8, w, h)
            tmp.fill(0xffffffff)
            self.preview.composite(tmp, 0, 0, w, h, 0, 0, 1, 1, gdk.INTERP_BILINEAR, 255)
            self.preview = tmp

        pygtkcompat.gdk.pixbuf.save(self.preview, prefix + '_prev.png', 'png')
        brushinfo = self.brushinfo.clone()
        open(prefix + '.myb', 'w').write(brushinfo.save_to_string())
        self.remember_mtimes()

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
        prefix = self.get_fileprefix()

        filename = prefix + '_prev.png'
        pixbuf = gdk.pixbuf_new_from_file(filename)
        self._preview = pixbuf
        self.remember_mtimes()

    def _load_settings(self):
        """Loads the brush settings/dynamics from disk."""
        prefix = self.get_fileprefix()
        filename = prefix + '.myb'
        brushinfo_str = open(filename).read()
        try:
            self._brushinfo.load_from_string(brushinfo_str)
        except BrushInfo.ParseError, e:
            print 'Failed to load brush %r: %s' % (filename, e)
            self._brushinfo.load_defaults()
        self.remember_mtimes()
        self.settings_loaded = True
        if self.bm.is_in_brushlist(self): # FIXME: get rid of this check
            self._brushinfo.set_string_property("parent_brush_name", None)
        self.persistent = True

    def reload_if_changed(self):
        if self.settings_mtime is None: return
        if self.preview_mtime is None: return
        if not self.name: return
        if not self.has_changed_on_disk(): return False
        print 'Brush "' + self.name + '" has changed on disk, reloading it.'
        self.load()
        return True

    def __repr__(self):
        if self._brushinfo.settings:
            return "<ManagedBrush %r p=%s>" % (self.name, self._brushinfo.get_string_property("parent_brush_name"))
        else:
            return "<ManagedBrush %r (settings not loaded yet)>" % self.name


if __name__ == '__main__':
    import doctest
    doctest.testmod()

