"interface to DrawWidget etc. hiding some C implementation details"
# FIXME: bad file name, saying nothing about what's in here
try:
    import mydrawwidget
    from brushsettings import settings as brushsettings
except ImportError:
    print "\nYou need to 'make' the C modules first.\n"
    raise

class Brush(mydrawwidget.MyBrush):
    def __init__(self):
        mydrawwidget.MyBrush.__init__(self)
        for s in brushsettings:
            self.set_setting(s.index, s.default)
        self.color = [0, 0, 0]
        self.set_color(self.color[0], self.color[1], self.color[2])
    def invert_color(self):
        for i in range(3):
            self.color[i] = 255 - self.color[i]
        self.set_color(self.color[0], self.color[1], self.color[2])
    # TODO: save/load to/from file

DrawWidget = mydrawwidget.MyDrawWidget
