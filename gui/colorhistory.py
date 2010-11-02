import numpy
import random
from lib import helpers

num_colors = 5

colors = [(random.random(), random.random(), random.random()) for i in range(num_colors)]
# default to black-and-white
colors[-1] = (0.0, 0.0, 0.0)
colors[-2] = (1.0, 1.0, 1.0)
last_color = colors[-1]
atomic = False # FIXME: bad name, it has nothing to do with atomic, right?

def hsv_equal(a, b):
    # hack required because we somewhere have an rgb<-->hsv conversion roundtrip
    a_ = numpy.array(helpers.hsv_to_rgb(*a))
    b_ = numpy.array(helpers.hsv_to_rgb(*b))
    return ((a_ - b_)**2).sum() < (3*1.0/256)**2

color_pushed_observers = []

def push_color(color):
    global colors, num_colors, atomic, last_color

    if atomic:
        return

    for c in colors:
        if hsv_equal(c,color):
            colors.remove(c)
            break
    colors = (colors + [color])[-num_colors:]
    last_color = helpers.hsv_to_rgb(*color)
    for func in color_pushed_observers:
        func(color)
