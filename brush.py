"interface to MyBrush; hiding some C implementation details"
# FIXME: bad file name, saying nothing about what's in here
import mydrawwidget
from brushsettings import settings as brushsettings
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

class Brush(mydrawwidget.MyBrush):
    def __init__(self):
        mydrawwidget.MyBrush.__init__(self)
        for s in brushsettings:
            self.set_setting(s.index, s.default)
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
        for s in brushsettings:
            f.write('%s %f\n' % (s.cname, self.get_setting(s.index)))
        f.close()

    def load(self, path, name):
        self.name = name
        prefix = self.get_fileprefix(path)
        pixbuf = gtk.gdk.pixbuf_new_from_file(prefix + '_prev.png')
        self.update_preview(pixbuf)
        num_found = 0
        for line in open(prefix + '.myb').readlines():
            line = line.strip()
            if line.startswith('#'): continue
            try:
                parts = line.split()
                command = parts[0]
                args = parts[1:]
                if command == 'color':
                    self.set_color([int(s) for s in args])
                else:
                    found = False
                    for s in brushsettings:
                        if command == s.cname:
                            assert not found
                            found = True
                            self.set_setting(s.index, float(args[0]))
                    assert found, 'invalid setting'
            except e:
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
        for s in brushsettings:
            self.set_setting(s.index, other.get_setting(s.index))
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
