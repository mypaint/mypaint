"interface to MyBrush; hiding some C implementation details"
# FIXME: bad file name, saying nothing about what's in here
import mydrawwidget
import brushsettings
import gtk, string, os

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

class Setting:
    "a specific setting for a specific brush"
    def __init__(self, setting, parent_brush):
        self.setting = setting
        self.brush = parent_brush
        self.set_base_value(setting.default)
        self.points = len(brushsettings.inputs) * [None]
    def set_base_value(self, value):
        self.base_value = value
        self.brush.set_base_value(self.setting.index, value)
    def set_points(self, input, points):
        if points is None:
            self.points[input.index] = None
            self.brush.remove_mapping(self.setting.index, input.index)
        else:
            self.points[input.index] = points[:] # copy
            for j in xrange(8):
                self.brush.set_mapping(self.setting.index, input.index, j, points[j])
    def copy_from(self, other):
        self.load_from_string(other.save_to_string())
    def save_to_string(self):
        s = '%f' % self.base_value
        for i in brushsettings.inputs:
            points = self.points[i.index]
            if points:
                s += ' | ' + i.name + ' ' + ' '.join([str(f) for f in points])
        return s
    def load_from_string(self, s):
        parts = s.split('|')
        self.set_base_value(float(parts[0]))
        for i in brushsettings.inputs:
            self.set_points(i, None)
        for part in parts[1:]:
            subparts = part.split()
            command, args = subparts[0], subparts[1:]
            for i in brushsettings.inputs:
                if command == i.name:
                    points = [float(f) for f in args]
                    self.set_points(i, points)
                    print 'loaded point data'

class Brush(mydrawwidget.MyBrush):
    def __init__(self):
        mydrawwidget.MyBrush.__init__(self)
        self.settings = []
        for s in brushsettings.settings:
            self.settings.append(Setting(s, self))
        self.color = [0, 0, 0]
        self.set_color(self.color)
        self.preview = None
        self.preview_thumb = None
        self.name = ''

    def get_fileprefix(self, path):
        if not os.path.isdir(path): os.mkdir(path)
        if not self.name:
            i = 0
            while 1:
                self.name = 'b%03d' % i
                if not os.path.isfile(path + self.name + '.myb'):
                    break
                i += 1
        return path + self.name
        
    def save(self, path):
        prefix = self.get_fileprefix(path)
        self.preview.save(prefix + '_prev.png', 'png')
        f = open(prefix + '.myb', 'w')
        f.write('# mypaint brush file\n')
        r, g, b = self.get_color()
        f.write('color %d %d %d\n' % (r, g, b))
        for s in brushsettings.settings:
            f.write(s.cname + ' ' + self.settings[s.index].save_to_string() + '\n')
        f.close()

    def load(self, path, name):
        self.name = name
        prefix = self.get_fileprefix(path)
        pixbuf = gtk.gdk.pixbuf_new_from_file(prefix + '_prev.png')
        self.update_preview(pixbuf)
        num_found = 0
        print 'parsing', prefix
        for line in open(prefix + '.myb').readlines():
            line = line.strip()
            if line.startswith('#'): continue
            try:
                command, rest = line.split(' ', 1)
                if command == 'color':
                    self.set_color([int(s) for s in rest.split()])
                else:
                    found = False
                    for s in brushsettings.settings:
                        if command == s.cname:
                            assert not found
                            found = True
                            self.settings[s.index].load_from_string(rest)
                    assert found, 'invalid setting'
            except None, e:
                print e
                print 'ignored line:'
                print line
            else:
                num_found += 1
        if num_found == 0:
            print 'there was only garbage in this file, using defaults'
        #TODO: load color

    def delete(self, path):
        prefix = self.get_fileprefix(path)
        os.remove(prefix + '_prev.png')
        os.remove(prefix + '.myb')

    def copy_settings_from(self, other):
        for s in brushsettings.settings:
            self.settings[s.index].copy_from(other.settings[s.index])
        self.color = other.color[:] # copy
        self.set_color(self.color)

    def get_color(self):
        return self.color[:] # copy

    def set_color(self, rgb):
        r, g, b = rgb
        self.color = rgb[:] # copy
        mydrawwidget.MyBrush.set_color(self, r, g, b)

    def invert_color(self):
        for i in range(3):
            self.color[i] = 255 - self.color[i]
        self.set_color(self.color)

    def update_preview(self, pixbuf):
        self.preview = pixbuf
        self.preview_thumb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, thumb_w, thumb_h)
        pixbuf_scale_nostretch_centered(src=pixbuf, dst=self.preview_thumb)

DrawWidget = mydrawwidget.MyDrawWidget
