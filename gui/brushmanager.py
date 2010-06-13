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

from lib import brush
from gtk import gdk # only for gdk.pixbuf
from gettext import gettext as _
import os

preview_w = 128
preview_h = 128

DEFAULT_BRUSH_GROUP = 'default'
DELETED_BRUSH_GROUP = 'deleted'
FAVORITES_BRUSH_GROUP = 'favorites'

def translate_group_name(name):
    d = {DEFAULT_BRUSH_GROUP: _('default'),
         DELETED_BRUSH_GROUP: _('deleted'),
         FAVORITES_BRUSH_GROUP: _('favorites'),
         'ink': _('ink'),
         'classic': _('classic'),
         'experimental': _('experimental'),
         }
    return d.get(name, name)

class BrushManager:
    def __init__(self, stock_brushpath, user_brushpath):
        self.stock_brushpath = stock_brushpath
        self.user_brushpath = user_brushpath

        self.selected_brush = ManagedBrush(self)
        self.groups = {}
        self.contexts = []
        self.active_groups = []
        self.loaded_groups = []
        self.brush_by_device = {} # should be save/loaded too?
        self.selected_context = None

        self.selected_brush_observers = []
        self.groups_observers = [] # for both self.groups and self.active_groups
        self.brushes_observers = [] # for all bruslists inside groups

        if not os.path.isdir(self.user_brushpath):
            os.mkdir(self.user_brushpath)
        self.load_groups()

        # TODO: load from config instead
        if DEFAULT_BRUSH_GROUP in self.groups:
            brushes = self.get_group_brushes(DEFAULT_BRUSH_GROUP, make_active=True)
            if brushes:
                self.selected_brush = brushes[0]

        self.brushes_observers.append(self.brushes_modified_cb)


    def load_groups(self):
        for i in range(10):
            c = ManagedBrush(self)
            c.name = 'context%02d' % i
            self.contexts.append(c)

        brush_by_name = {}
        def get_brush(name):
            if name not in brush_by_name:
                b = ManagedBrush(self, name, persistent=True)
                brush_by_name[name] = b
            return brush_by_name[name]

        def read_groups(filename):
            groups = {}
            if os.path.exists(filename):
                curr_group = DEFAULT_BRUSH_GROUP
                for line in open(filename):
                    name = line.strip()
                    if name.startswith('#'):
                        continue
                    if name.startswith('Group: '):
                        curr_group = unicode(name[7:], 'utf-8')
                        if curr_group not in groups:
                            groups[curr_group] = []
                        continue
                    try:
                        b = get_brush(name)
                    except IOError, e:
                        print e, '(removed from group)'
                        continue
                    groups.setdefault(curr_group, [])
                    if b in groups[curr_group]:
                        print filename + ': Warning: brush appears twice in the same group, ignored'
                        continue
                    groups[curr_group].append(b)
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
            return [filename[:-4] for filename in os.listdir(path) if filename.endswith('.myb')]
        for name in listbrushes(self.stock_brushpath) + listbrushes(self.user_brushpath):
            b = get_brush(name)
            if name.startswith('context'):
                i = int(name[-2:])
                self.contexts[i] = b
                continue
            if not [True for group in our.itervalues() if b in group]:
                brushes = self.groups.setdefault(DEFAULT_BRUSH_GROUP, [])
                brushes.insert(0, b)

        # clean up legacy stuff
        fn = os.path.join(self.user_brushpath, 'deleted.conf')
        if os.path.exists(fn):
            os.remove(fn)

    def get_brush_by_name(self, name):
        # used only for testing
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
                f.write(b.name + '\n')
        f.close()

    def select_brush(self, brush):
        if brush is None:
            brush = ManagedBrush(self)
        if brush.persistent and not brush.settings_loaded:
            brush.load_settings()
        assert isinstance(brush, ManagedBrush)
        self.selected_brush = brush
        for callback in self.selected_brush_observers:
            callback(brush)

    def set_active_groups(self, groups):
        """Set active groups, loading them first if neccesary."""
        for groupname in groups:
            if not groupname in self.loaded_groups:
                for brush in self.groups[groupname]:
                    brush.load_preview()
            self.loaded_groups.append(groupname)
        self.active_groups = groups

    def get_group_brushes(self, group, make_active=False):
        if group not in self.groups:
            brushes = []
            self.groups[group] = brushes
            for f in self.groups_observers: f()
            self.save_brushorder()
        if make_active and group not in self.active_groups:
            self.set_active_groups([group] + self.active_groups)
            for f in self.groups_observers: f()
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


class ManagedBrush(brush.Brush):
    def __init__(self, brushmanager, name=None, persistent=False):
        brush.Brush.__init__(self)
        self.bm = brushmanager
        self.preview = None
        self.name = name
        self.persistent = persistent
        """If True this brush is stored in the filesystem and 
        not a context/picked brush."""
        self.settings_loaded = False
        """If True this brush is fully initialized, ready to paint with."""

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
                self.name = '%s%03d' % (prefix, i)
                a = os.path.join(self.bm.user_brushpath, self.name + '.myb')
                b = os.path.join(self.bm.stock_brushpath, self.name + '.myb')
                if not os.path.isfile(a) and not os.path.isfile(b):
                    break
                i += 1
        prefix = os.path.join(self.bm.user_brushpath, self.name)
        if saving: 
            return prefix
        if not os.path.isfile(prefix + '.myb'):
            prefix = os.path.join(self.bm.stock_brushpath, self.name)
        if not os.path.isfile(prefix + '.myb'):
            raise IOError, 'brush "' + self.name + '" not found'
        return prefix

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
        open(prefix + '.myb', 'w').write(self.save_to_string())
        self.remember_mtimes()

    def load(self):
        self.load_preview()
        self.load_settings()

    def load_preview(self):
        """Loads the brush preview as pixbuf into the brush."""
        prefix = self.get_fileprefix()

        filename = prefix + '_prev.png'
        pixbuf = gdk.pixbuf_new_from_file(filename)
        self.preview = pixbuf
        self.remember_mtimes()

    def load_settings(self):
        """Loads the brush settings/dynamics from disk."""
        prefix = self.get_fileprefix()
        filename = prefix + '.myb'
        errors = self.load_from_string(open(filename).read())
        if errors:
            print '%s:' % filename
            for line, reason in errors:
                print line
                print '==>', reason
            print

        self.remember_mtimes()
        self.settings_loaded = True

    def reload_if_changed(self):
        if self.settings_mtime is None: return
        if self.preview_mtime is None: return
        if not self.name: return
        if not self.has_changed_on_disk(): return False
        print 'Brush "' + self.name + '" has changed on disk, reloading it.'
        self.load()
        return True

