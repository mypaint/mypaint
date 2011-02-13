import numpy
import random
from lib import helpers

class ColorHistory (object):
    num_colors = 5
    default_colors = [(random.random(), random.random(), random.random()) for i in range(num_colors)]
    ## default to black-and-white
    default_colors[-1] = (0.0, 0.0, 0.0)
    default_colors[-2] = (1.0, 1.0, 1.0)

    def __init__(self, app):
        self.app = app
        self.colors = self.default_colors[:]
        self.last_color = self.colors[-1]
        self.color_pushed_observers = []
        saved = app.preferences.get('colorhistory.colors', [])
        for hsv, i in zip(saved, range(self.num_colors)):
            self.colors[i] = tuple(hsv)
            # tuple([helpers.clamp(c, 0.0, 1.0) for c in hsv])
        self.atomic = False # FIXME: bad name, it has nothing to do with atomic, right?

    @staticmethod
    def hsv_equal(a, b):
        # hack required because we somewhere have an rgb<-->hsv conversion roundtrip
        a_ = numpy.array(helpers.hsv_to_rgb(*a))
        b_ = numpy.array(helpers.hsv_to_rgb(*b))
        return ((a_ - b_)**2).sum() < (3*1.0/256)**2
    
    def push_color(self, color):
        if self.atomic:
            return
        for c in self.colors:
            if self.hsv_equal(c, color):
                self.colors.remove(c)
                break
        self.colors = (self.colors + [color])[-self.num_colors:]
        self.last_color = helpers.hsv_to_rgb(*color)
        self.app.preferences['colorhistory.colors'] = [tuple(hsv) for hsv in self.colors]
        for func in self.color_pushed_observers:
            func(color)
