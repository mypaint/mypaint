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
    def foo(self):
        print "blah - just an inheritance test"
    # TODO: save/load to/from file

DrawWidget = mydrawwidget.MyDrawWidget
