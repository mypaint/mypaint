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

import dialogs
import gtk
from gtk import gdk # only for gdk.pixbuf
from gettext import gettext as _
import os, zipfile
from os.path import basename
import urllib
import gobject
from lib.brush import BrushInfo

preview_w = 128
preview_h = 128

DEFAULT_STARTUP_GROUP = 'Deevad'  # Suggestion only
DEFAULT_BRUSH = 'deevad/artpen'  # TODO: phase out and use heruristics?
DEFAULT_ERASER = 'deevad/stick'  # TODO: ---------------"--------------
FOUND_BRUSHES_GROUP = 'lost&found'
DELETED_BRUSH_GROUP = 'deleted'
FAVORITES_BRUSH_GROUP = 'favorites'
DEVBRUSH_NAME_PREFIX = "devbrush_"

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
    d = {FOUND_BRUSHES_GROUP: _('lost&found'),
         DELETED_BRUSH_GROUP: _('deleted'),
         FAVORITES_BRUSH_GROUP: _('favorites'),
         'ink': _('ink'),
         'classic': _('classic'),
         'experimental': _('experimental'),
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
        self.contexts = [None for i in xrange(10)]

        brush_by_name = {}
        def get_brush(name):
            if name not in brush_by_name:
                b = ManagedBrush(self, name, persistent=True)
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
            b = get_brush(name)
            b.in_brushlist = True
            if name.startswith('context'):
                i = int(name[-2:])
                self.contexts[i] = b
                b.load_settings(retain_parent=True)
                b.in_brushlist = False
            elif name.startswith(DEVBRUSH_NAME_PREFIX):
                device_name = devbrush_unquote(name)
                self.brush_by_device[device_name] = b
                b.load_settings(retain_parent=True)
                b.in_brushlist = False
            if b.in_brushlist:
                if not [True for group in our.itervalues() if b in group]:
                    brushes = self.groups.setdefault(FOUND_BRUSHES_GROUP, [])
                    brushes.insert(0, b)

        # Sensible defaults for brushkeys: clone brushes 1 through 10 from the
        # default startup group if we need to and if we can.
        for i in xrange(10):
            if self.contexts[i] is not None:
                continue
            name = unicode('context%02d') % i
            c = ManagedBrush(self, name=name, persistent=False)
            group = self.groups.get(DEFAULT_STARTUP_GROUP, [])
            idx = (i+9) % 10 # keyboard order
            if idx < len(group):
                b = group[idx]
                b.clone_into(c, name)
            self.contexts[i] = c

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
                        b.load_preview()
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
                    b.in_brushlist = True
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

    def find_brushlist_ancestor(self, brush):
        """Finds the nearest ancestor of a ManagedBrush having in_brushlist

        Searches a brush's ancestry chain for something which can be
        highlighted for the user in the brushlist. Returns `brush`, one of its
        ancestors, or None if nothing suitable can be found.
        """
        while brush is not None:
            if brush.in_brushlist:
                return brush
            parent_name = brush.brushinfo.get_string_property("parent_brush_name")
            brush = self.get_brush_by_name(parent_name)
        return None

    def select_brush(self, brush):
        """Selects a ManagedBrush, highlights it, & updates the live brush."""
        if brush is None:
            brush = self.get_default_brush()
        self.selected_brush = brush
        if brush.persistent and not brush.settings_loaded:
            brush.load_settings()
        self.app.preferences['brushmanager.selected_brush'] = brush.name
        # Take care of updating the live brush, amongst other things
        for callback in self.selected_brush_observers:
            callback(brush)

    def clone_selected_brush(self, name):
        """
        Creates a new ManagedBrush based on the selected
        brush in the brushlist and the currently active lib.brush.
        """
        clone = ManagedBrush(self, name, persistent=False)
        clone.brushinfo = self.app.brush.clone()
        clone.preview = self.selected_brush.preview
        list_brush = self.find_brushlist_ancestor(self.selected_brush)
        if list_brush:
            parent = list_brush.name
        else:
            parent = None
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

    def set_active_groups(self, groups):
        """Set active groups, loading them first if neccesary."""
        for groupname in groups:
            if not groupname in self.loaded_groups:
                for brush in self.groups[groupname]:
                    brush.load_preview()
            self.loaded_groups.append(groupname)
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
        self.preview = None
        self.name = name
        self.brushinfo = BrushInfo()
        self.persistent = persistent
        """If True this brush is stored in the filesystem."""
        self.settings_loaded = False
        """If True this brush is fully initialized, ready to paint with."""
        self.in_brushlist = False
        """Set to True if this brush is known to be in the brushlist"""

        self.settings_mtime = None
        self.preview_mtime = None

        if persistent:
            # we load the files later, but throw an exception now if they don't exist
            self.get_fileprefix()

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
        if not self.settings_loaded:
            self.load()
        target.brushinfo = self.brushinfo.clone()
        list_brush = self.bm.find_brushlist_ancestor(self)
        if list_brush:
            parent = list_brush.name
        else:
            parent = None
        target.brushinfo.set_string_property("parent_brush_name", parent)
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
        if self.preview is None:
            self.preview = gdk.Pixbuf(gdk.COLORSPACE_RGB, False, 8, preview_w, preview_h)
            self.preview.fill(0xffffffff) # white
        self.preview.save(prefix + '_prev.png', 'png')
        brushinfo = self.brushinfo.clone()
        open(prefix + '.myb', 'w').write(brushinfo.save_to_string())
        self.remember_mtimes()

    def load(self, retain_parent=False):
        """Loads the brush's preview and settings from disk."""
        self.load_preview()
        self.load_settings(retain_parent)

    def load_preview(self):
        """Loads the brush preview as pixbuf into the brush."""
        prefix = self.get_fileprefix()

        filename = prefix + '_prev.png'
        pixbuf = gdk.pixbuf_new_from_file(filename)
        self.preview = pixbuf
        self.remember_mtimes()

    def load_settings(self, retain_parent=False):
        """Loads the brush settings/dynamics from disk."""
        prefix = self.get_fileprefix()
        filename = prefix + '.myb'
        brushinfo_str = open(filename).read()
        self.brushinfo.load_from_string(brushinfo_str)
        self.remember_mtimes()
        self.settings_loaded = True
        if not retain_parent:
            self.brushinfo.set_string_property("parent_brush_name", None)
        self.persistent = True

    def reload_if_changed(self):
        if self.settings_mtime is None: return
        if self.preview_mtime is None: return
        if not self.name: return
        if not self.has_changed_on_disk(): return False
        print 'Brush "' + self.name + '" has changed on disk, reloading it.'
        self.load()
        return True

    def __str__(self):
        if self.brushinfo.settings:
            return "<ManagedBrush %s p=%s>" % (self.name, self.brushinfo.get_string_property("parent_brush_name"))
        else:
            return "<ManagedBrush %s (settings not loaded yet)>" % self.name


if __name__ == '__main__':
    import doctest
    doctest.testmod()

