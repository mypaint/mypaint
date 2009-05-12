# This file is part of MyPaint.
# Copyright (C) 2007 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"interface to MyBrush; hiding some C implementation details"
# FIXME: bad file name, saying nothing about what's in here
# FIXME: should split brush_lowlevel into its own gtk-independent module
import mypaintlib
from brushlib import brushsettings
import gtk, string, os
import helpers

preview_w = 128
preview_h = 128
thumb_w = 64
thumb_h = 64

current_brushfile_version = 2

def pixbuf_scale_nostretch_centered(src, dst):
    scale_x = float(dst.get_width()) / src.get_width()
    scale_y = float(dst.get_height()) / src.get_height()
    offset_x = 0
    offset_y = 0
    if scale_x > scale_y: 
        scale = scale_y
        offset_x = (dst.get_width() - src.get_width() * scale) / 2
    else:
        scale = scale_x
        offset_y = (dst.get_height() - src.get_height() * scale) / 2

    src.scale(dst, 0, 0, dst.get_width(), dst.get_height(),
              offset_x, offset_y, scale, scale,
              gtk.gdk.INTERP_BILINEAR)

# points = [(x1, y1), (x2, y2), ...] (at least two points, or None)
class Setting:
    "a specific setting for a specific brush"
    def __init__(self, setting, parent_brush, observers):
        self.setting = setting
        self.brush = parent_brush
        self.observers = observers
        self.base_value = None
        self.set_base_value(setting.default)
        self.points = [[] for i in xrange(len(brushsettings.inputs))]
        if setting.cname == 'opaque_multiply':
            # make opaque depend on pressure by default
            for i in brushsettings.inputs:
                if i.name == 'pressure': break
            self.set_points(i, [(0.0, 0.0), (1.0, 1.0)])
    def set_base_value(self, value):
        if self.base_value == value: return
        self.base_value = value
        self.brush.set_base_value(self.setting.index, value)
        for f in self.observers: f()
    def has_only_base_value(self):
        for i in brushsettings.inputs:
            if self.has_input(i):
                return False
        return True
    def has_input(self, input):
        return self.points[input.index]
    def has_input_nonlinear(self, input):
        points = self.points[input.index]
        if not points: return False
        if len(points) > 2: return True
        # also if it is linear but the x-axis was changed (hm, bad function name)
        if abs(points[0][0] - input.soft_min) > 0.001: return True
        if abs(points[1][0] - input.soft_max) > 0.001: return True
        return False

    def set_points(self, input, points):
        assert len(points) != 1
        if self.points[input.index] == points: return
        #if len(points) > 2:
        #    print 'set_points[%s](%s, %s)' % (self.setting.cname, input.name, points)

        self.brush.set_mapping_n(self.setting.index, input.index, len(points))
        for i, (x, y) in enumerate(points):
            self.brush.set_mapping_point(self.setting.index, input.index, i, x, y)

        self.points[input.index] = points[:] # copy
        for f in self.observers: f()

    def copy_from(self, other):
        error = self.load_from_string(other.save_to_string(), version=current_brushfile_version)
        assert not error, error
    def save_to_string(self):
        s = str(self.base_value)
        for i in brushsettings.inputs:
            points = self.points[i.index]
            if points:
                s += ' | ' + i.name + ' ' + ', '.join(['(%f %f)' % xy for xy in points])
        return s
    def load_from_string(self, s, version):
        error = None
        parts = s.split('|')
        self.set_base_value(float(parts[0]))
        for i in brushsettings.inputs:
            self.set_points(i, [])
        for part in parts[1:]:
            command, args = part.strip().split(' ', 1)
            if version <= 1 and command == 'speed': command = 'speed1'
            i = brushsettings.inputs_dict.get(command)
            if i:
                if version <= 1:
                    points_old = [float(f) for f in args.split()]
                    points = [(0, 0)]
                    while points_old:
                        x = points_old.pop(0)
                        y = points_old.pop(0)
                        if x == 0: break
                        assert x > points[-1][0]
                        points.append((x, y))
                else:
                    points = []
                    for s in args.split(', '):
                        s = s.strip()
                        if not (s.startswith('(') and s.endswith(')') and ' ' in s):
                            return '(x y) expected, got "%s"' % s
                        s = s[1:-1]
                        try:
                            x, y = [float(ss) for ss in s.split(' ')]
                        except:
                            print s
                            raise
                        points.append((x, y))
                assert len(points) >= 2
                self.set_points(i, points)
            else:
                error = 'unknown input "%s"' % command
        return error
    def transform_y(self, func):
        # useful for migration from a earlier version
        self.set_base_value(func(self.base_value))
        for i in brushsettings.inputs:
            if not self.points[i.index]: continue
            points = self.points[i.index]
            points = [(x, func(y)) for x, y in points]
            self.set_points(i, points)

class Brush_Lowlevel(mypaintlib.Brush):
    def __init__(self):
        mypaintlib.Brush.__init__(self)
        self.settings_observers = []
        self.settings_observers_hidden = []
        self.settings = []
        for s in brushsettings.settings:
            self.settings.append(Setting(s, self, self.settings_observers))

        self.saved_string = None
        self.settings_observers.append(self.invalidate_saved_string)

    def invalidate_saved_string(self):
        self.saved_string = None

    def begin_atomic(self):
        self.settings_observers_hidden.append(self.settings_observers[:])
        del self.settings_observers[:]

    def end_atomic(self):
        self.settings_observers[:] = self.settings_observers_hidden.pop()
        for f in self.settings_observers: f()


    def get_stroke_bbox(self):
        bbox = self.stroke_bbox
        return bbox.x, bbox.y, bbox.w, bbox.h

    def setting_by_cname(self, cname):
        s = brushsettings.settings_dict[cname]
        return self.settings[s.index]

    def save_to_string(self):
        # OPTIMIZE: this cache could be more useful, the current "copy_settings_from()"
        #           brush selection mechanism invalidates it at every brush change
        if self.saved_string: return self.saved_string
        res  = '# mypaint brush file\n'
        res += '# you can edit this file and then select the brush in mypaint (again) to reload\n'
        res += 'version %d\n' % current_brushfile_version
        for s in brushsettings.settings:
            res += s.cname + ' ' + self.settings[s.index].save_to_string() + '\n'
        self.saved_string = res
        return res

    def load_from_string(self, s):
        self.begin_atomic()
        num_found = 0
        errors = []
        version = 1 # for files without a 'version' field
        for line in s.split('\n'):
            line = line.strip()
            if line.startswith('#'): continue
            if not line: continue
            try:
                command, rest = line.split(' ', 1)
                error = None

                if command in brushsettings.settings_dict:
                    setting = self.setting_by_cname(command)
                    error = setting.load_from_string(rest, version)
                elif command in brushsettings.settings_migrate:
                    command_new, transform_func = brushsettings.settings_migrate[command]
                    setting = self.setting_by_cname(command_new)
                    error = setting.load_from_string(rest, version)
                    if transform_func:
                        setting.transform_y(transform_func)
                elif command == 'version':
                    version = int(rest)
                    if version > current_brushfile_version:
                        error = 'this brush was saved with a more recent version of mypaint'
                elif version <= 1 and command == 'color':
                    self.set_color_rgb([int(s)/255.0 for s in rest.split()])
                elif version <= 1 and command == 'change_radius':
                    if rest != '0.0': error = 'change_radius is not supported any more'
                elif version <= 2 and command == 'adapt_color_from_image':
                    if rest != '0.0': error = 'adapt_color_from_image is obsolete, ignored; use smudge and smudge_length instead'
                elif version <= 1 and command == 'painting_time':
                    pass
                else:
                    error = 'unknown command, line ignored'

                if error:
                    errors.append((line, error))

            except Exception, e:
                errors.append((line, str(e)))
            else:
                num_found += 1
        if num_found == 0:
            errors.append(('', 'there was only garbage in this file, using defaults'))
        self.end_atomic()
        return errors

    def copy_settings_from(self, other):
        self.begin_atomic()
        for i, setting in enumerate(self.settings):
            setting.copy_from(other.settings[i])
        self.end_atomic()

    def get_color_hsv(self):
        h = self.setting_by_cname('color_h').base_value
        s = self.setting_by_cname('color_s').base_value
        v = self.setting_by_cname('color_v').base_value
        return (h, s, v)

    def set_color_hsv(self, hsv):
        self.begin_atomic()
        h, s, v = hsv
        self.setting_by_cname('color_h').set_base_value(h)
        self.setting_by_cname('color_s').set_base_value(s)
        self.setting_by_cname('color_v').set_base_value(v)
        self.end_atomic()

    def set_color_rgb(self, rgb):
        self.set_color_hsv(helpers.rgb_to_hsv(*rgb))

    def get_color_rgb(self):
        hsv = self.get_color_hsv()
        return helpers.hsv_to_rgb(*hsv)

    def is_eraser(self):
        return self.setting_by_cname('eraser').base_value > 0.9

class Brush(Brush_Lowlevel):
    def __init__(self, app):
        Brush_Lowlevel.__init__(self)
        self.app = app
        self.preview = None
        self.preview_thumb = None
        self.name = None
        self.preview_changed = True

        self.settings_mtime = None
        self.preview_mtime = None

    def get_fileprefix(self, saving=False):
        prefix = 'b'
        if os.path.realpath(self.app.user_brushpath) == os.path.realpath(self.app.stock_brushpath):
        #if os.path.samefile(self.app.user_brushpath, self.app.stock_brushpath):
            # working directly on brush collection, use different prefix
            prefix = 's'

        if not self.name:
            i = 0
            while 1:
                self.name = '%s%03d' % (prefix, i)
                a = os.path.join(self.app.user_brushpath,self.name + '.myb')
                b = os.path.join(self.app.stock_brushpath,self.name + '.myb')
                if not os.path.isfile(a) and not os.path.isfile(b):
                    break
                i += 1
        prefix = os.path.join(self.app.user_brushpath, self.name)
        if saving: 
            return prefix
        if not os.path.isfile(prefix + '.myb'):
            prefix = os.path.join(self.app.stock_brushpath,self.name)
        assert os.path.isfile(prefix + '.myb'), 'brush "' + self.name + '" not found'
        return prefix

    def delete_from_disk(self):
        prefix = os.path.join(self.app.user_brushpath, self.name)
        if os.path.isfile(prefix + '.myb'):
            os.remove(prefix + '_prev.png')
            os.remove(prefix + '.myb')

        prefix = os.path.join(self.app.stock_brushpath, self.name)
        if os.path.isfile(prefix + '.myb'):
            # user wants to delete a stock brush
            # cannot remove the file, manage blacklist instead
            filename = os.path.join(self.app.user_brushpath, 'deleted.conf')
            new = not os.path.isfile(filename)
            f = open(filename, 'a')
            if new:
                f.write('# list of stock brush names which you have deleted\n')
                f.write('# you can remove this file to get all of them back\n')
            f.write(self.name + '\n')
            f.close()

        self.preview_changed = True # need to recreate when saving

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
        if self.preview_changed:
            self.preview.save(prefix + '_prev.png', 'png')
            self.preview_changed = False
        open(prefix + '.myb', 'w').write(self.save_to_string())
        self.remember_mtimes()

    def load(self, name):
        self.name = name
        prefix = self.get_fileprefix()

        filename = prefix + '_prev.png'
        pixbuf = gtk.gdk.pixbuf_new_from_file(filename)
        self.update_preview(pixbuf)

        if prefix.startswith(self.app.user_brushpath):
            self.preview_changed = False
        else:
            # for saving, create the preview file even if not changed
            self.preview_changed = True

        filename = prefix + '.myb'
        errors = self.load_from_string(open(filename).read())
        if errors:
            print '%s:' % filename
            for line, reason in errors:
                print line
                print '==>', reason
            print

        self.remember_mtimes()

    def reload_if_changed(self):
        if self.settings_mtime is None: return
        if self.preview_mtime is None: return
        if not self.name: return
        prefix = self.get_fileprefix()
        if not self.has_changed_on_disk(): return False
        print 'Brush "' + self.name + '" has changed on disk, reloading it.'
        self.load(self.name)
        return True

    def update_preview(self, pixbuf):
        self.preview = pixbuf
        self.preview_thumb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, thumb_w, thumb_h)
        pixbuf_scale_nostretch_centered(src=pixbuf, dst=self.preview_thumb)
        self.preview_changed = True

