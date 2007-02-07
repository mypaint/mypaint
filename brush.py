"interface to MyBrush; hiding some C implementation details"
# FIXME: bad file name, saying nothing about what's in here
import mydrawwidget
import brushsettings
import gtk, string, os, colorsys
from helpers import clamp

thumb_w = 64 #128
thumb_h = 64 #128

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

# What do the points[] mean?
# Points = [x1, y1, x2, y2, x3, y3, x4, y4]
# with x_scale = max(x_i) and y_scale = max(y_i)
class Setting:
    "a specific setting for a specific brush"
    def __init__(self, setting, parent_brush):
        self.setting = setting
        self.brush = parent_brush
        self.set_base_value(setting.default)
        self.points = len(brushsettings.inputs) * [None]
        if setting.cname == 'opaque_multiply':
            # make opaque depend on pressure by default
            for i in brushsettings.inputs:
                if i.name == 'pressure': break
            self.set_points(i, [1.0, 1.0] + 3*[0.0, 0.0])
    def set_base_value(self, value):
        self.base_value = value
        self.brush.set_base_value(self.setting.index, value)
    def has_only_base_value(self):
        for i in brushsettings.inputs:
            if self.has_input(i):
                return False
        return True
    def has_input(self, input):
        points = self.points[input.index]
        return points is not None
    def has_input_nonlinear(self, input):
        points = self.points[input.index]
        if points is None:
            return False
        # also if it is linear but the x-axis was changed
        if points[0] != 1.0: return True
        # having one additional point is sufficient
        if points[2] == 0.0 and points[3] == 0.0: return False
        return True

    def set_points(self, input, points):
        if points is None:
            self.points[input.index] = None
            # set the first value to zero to disable it
            self.brush.set_mapping(self.setting.index, input.index, 0, 0)
        else:
            self.points[input.index] = points[:] # copy
            for j in xrange(8):
                self.brush.set_mapping(self.setting.index, input.index, j, points[j])
    def copy_from(self, other):
        error = self.load_from_string(other.save_to_string())
        assert not error
    def save_to_string(self):
        s = str(self.base_value)
        for i in brushsettings.inputs:
            points = self.points[i.index]
            if points:
                s += ' | ' + i.name + ' ' + ' '.join([str(f) for f in points])
        return s
    def load_from_string(self, s):
        error = None
        parts = s.split('|')
        self.set_base_value(float(parts[0]))
        for i in brushsettings.inputs:
            self.set_points(i, None)
        for part in parts[1:]:
            subparts = part.split()
            command, args = subparts[0], subparts[1:]
            if command == 'speed': command = 'speed1' # backwards compatibilty
            found = False
            for i in brushsettings.inputs:
                if command == i.name:
                    found = True
                    points = [float(f) for f in args]
                    self.set_points(i, points)
            if not found:
                error = 'unknown input "%s"' % command
        return error
    def transform_y(self, func):
        # useful for migration from a earlier version
        self.set_base_value(func(self.base_value))
        for i in brushsettings.inputs:
            if not self.points[i.index]: continue
            p = []
            for j, v in enumerate(self.points[i.index]):
                if j % 2 == 1: v = func(v)
                p.append(v)
            self.set_points(i, p)

class Brush_Lowlevel(mydrawwidget.MyBrush):
    def __init__(self):
        mydrawwidget.MyBrush.__init__(self)
        self.settings = []
        for s in brushsettings.settings:
            self.settings.append(Setting(s, self))
        self.painting_time = 0.0

    def setting_by_cname(self, cname):
        s = brushsettings.settings_dict[cname]
        return self.settings[s.index]

    def save_to_string(self):
        res  = '# mypaint brush file\n'
        for s in brushsettings.settings:
            res += s.cname + ' ' + self.settings[s.index].save_to_string() + '\n'
        return res

    def load_from_string(self, s):
        num_found = 0
        errors = []
        for line in s.split('\n'):
            line = line.strip()
            if line.startswith('#'): continue
            if not line: continue
            try:
                command, rest = line.split(' ', 1)
                error = None

                if command in brushsettings.settings_dict:
                    setting = self.setting_by_cname(command)
                    error = setting.load_from_string(rest)
                elif command in brushsettings.settings_migrate:
                    command_new, transform_func = brushsettings.settings_migrate[command]
                    setting = self.setting_by_cname(command_new)
                    error = setting.load_from_string(rest)
                    if transform_func:
                        setting.transform_y(transform_func)
                elif command == 'color': # obsolete
                    self.set_color_rgb([int(s)/255.0 for s in rest.split()])
                elif command == 'change_radius': # obsolete
                    if rest != '0.0': error = 'change_radius is not supported any more, use radius directly'
                #elif rest == '0.0':
                #    pass # silently ignore unknown/obsolete settings if they are zero
                elif command == 'painting_time': # obsolete
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
        return errors

    def copy_settings_from(self, other):
        for i, setting in enumerate(self.settings):
            setting.copy_from(other.settings[i])

    def get_color_hsv(self):
        h = self.setting_by_cname('color_h').base_value
        s = self.setting_by_cname('color_s').base_value
        v = self.setting_by_cname('color_v').base_value
        return (h, s, v)

    def set_color_hsv(self, hsv):
        h, s, v = hsv
        self.setting_by_cname('color_h').set_base_value(h)
        self.setting_by_cname('color_s').set_base_value(s)
        self.setting_by_cname('color_v').set_base_value(v)

    def set_color_rgb(self, rgb):
        for i in range(3): assert rgb[i] <= 1.0
        self.set_color_hsv(colorsys.rgb_to_hsv(*rgb))

    def get_color_rgb(self):
        hsv = self.get_color_hsv()
        hsv = [clamp(x, 0.0, 1.0) for x in hsv]
        return colorsys.hsv_to_rgb(*hsv)

    def invert_color(self):
        rgb = self.get_color_rgb()
        rgb = [1-x for x in rgb]
        self.set_color_rgb(rgb)


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

DrawWidget = mydrawwidget.MyDrawWidget
