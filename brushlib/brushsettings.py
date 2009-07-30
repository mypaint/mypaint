# brushlib - The MyPaint Brush Library
# Copyright (C) 2007-2008 Martin Renold <martinxyz@gmx.ch>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY. See the COPYING file for more details.

"""Brush Settings / States

This is used to generate brushsettings.hpp (see generate.py)
It is also imported at runtime.
"""

inputs_list = [
    # name, hard minimum, soft minimum, normal[1], soft maximum, hard maximum, tooltip
    ['pressure', 0.0,  0.0,  0.4,  1.0, 1.0,  "The pressure reported by the tablet, between 0.0 and 1.0. If you use the mouse, it will be 0.5 when a button is pressed and 0.0 otherwise."],
    ['speed1',   None, 0.0,  0.5,  4.0, None, "How fast you currently move. This can change very quickly. Try 'print input values' from the 'help' menu to get a feeling for the range; negative values are rare but possible for very low speed."],
    ['speed2',   None, 0.0,  0.5,  4.0, None, "Same as speed1, but changes slower. Also look at the 'speed2 filter' setting."],
    ['random',   0.0,  0.0,  0.5,  1.0, 1.0,  "Fast random noise, changing at each evaluation. Evenly distributed between 0 and 1."],
    ['stroke',   0.0,  0.0,  0.5,  1.0, 1.0,  "This input slowly goes from zero to one while you draw a stroke. It can also be configured to jump back to zero periodically while you move. Look at the 'stroke duration' and 'stroke hold time' settings."],
    ['direction',0.0,  0.0,  0.0,  180.0, 180.0,  "The angle of the stroke, in degrees. The value will stay between 0.0 and 180.0, effectively ignoring turns of 180 degrees."],
    #['motion_strength',0.0,0.0,  0.0,  1.0, 1.0,  "[EXPERIMENTAL] Same as angle, but wraps at 180 degrees. The dynamics are shared with BRUSH_OFFSET_BY_SPEED_FILTER (FIXME: which is a bad thing)."],
    ['custom',   None,-2.0,  0.0, +2.0, None, "This is a user defined input. Look at the 'custom input' setting for details."],
    ]
    # [1] If, for example, the user increases the "by pressure" slider
    # in the "radius" control, then this should change the reaction to
    # pressure and not the "normal" radius. To implement this, we need
    # a guess what the user considers to be normal pressure.

settings_list = [
    # internal name, displayed name, constant, minimum, default, maximum, tooltip
    ['opaque', 'opaque', False, 0.0, 1.0, 1.0, "0 means brush is transparent, 1 fully visible\n(also known als alpha or opacity)"],
    ['opaque_multiply', 'opaque multiply', False, 0.0, 0.0, 1.0, "This gets multiplied with opaque. It is used for making opaque depend on pressure (or other inputs)."],
    ['opaque_linearize', 'opaque linearize', True, 0.0, 0.9, 2.0, "Correct the nonlinearity introduced by blending multiple dabs on top of each other. This correction should get you a linear (\"natural\") pressure response when pressure is mapped to opaque_multiply, as it is usually done. 0.9 is good for standard strokes, set it smaller if your brush scatters a lot, or higher if you use dabs_per_second.\n0.0 the opaque value above is for the individual dabs\n1.0 the opaque value above is for the final brush stroke, assuming each pixel gets (dabs_per_radius*2) brushdabs on average during a stroke"],
    ['radius_logarithmic', 'radius', False, -2.0, 2.0, 5.0, "basic brush radius (logarithmic)\n 0.7 means 2 pixels\n 3.0 means 20 pixels"],
    ['hardness', 'hardness', False, 0.0, 0.8, 1.0, "hard brush-circle borders (setting to zero will draw nothing)"],
    ['dabs_per_basic_radius', 'dabs per basic radius', True, 0.0, 0.0, 6.0, "how many dabs to draw while the pointer moves a distance of one brush radius (more precise: the base value of the radius)"],
    ['dabs_per_actual_radius', 'dabs per actual radius', True, 0.0, 2.0, 6.0, "same as above, but the radius actually drawn is used, which can change dynamically"],
    ['dabs_per_second', 'dabs per second', True, 0.0, 0.0, 80.0, "dabs to draw each second, no matter how far the pointer moves"],
    ['radius_by_random', 'radius by random', False, 0.0, 0.0, 1.5, "Alter the radius randomly each dab. You can also do this with the by_random input on the radius setting. If you do it here, there are two differences:\n1) the opaque value will be corrected such that a big-radius dabs is more transparent\n2) it will not change the actual radius seen by dabs_per_actual_radius"],
    ['speed1_slowness', 'speed1 filter', False, 0.0, 0.04, 0.2, "how slow the input speed1 is following the real speed\n0.0 change immediatly as your speed changes (not recommended, but try it)"],
    ['speed2_slowness', 'speed2 filter', False, 0.0, 0.8, 3.0, "same as 'speed1 slowness', but note that the range is different"],
    ['speed1_gamma', 'speed1 gamma', True, -8.0, 4.0, 8.0, "This changes the reaction of the speed1 input to extreme physical speed. You will see the difference best if speed1 is mapped to the radius.\n-8.0 very fast speed does not increase speed1 much more\n+8.0 very fast speed increases speed1 a lot\nFor very slow speed the opposite happens."],
    ['speed2_gamma', 'speed2 gamma', True, -8.0, 4.0, 8.0, "same as 'speed1 gamma' for speed2"],
    ['offset_by_random', 'jitter', False, 0.0, 0.0, 2.0, "add a random offset to the position where each dab is drawn\n 0.0 disabled\n 1.0 standard deviation is one basic radius away\n<0.0 negative values produce no jitter"],
    ['offset_by_speed', 'offset by speed', False, -3.0, 0.0, 3.0, "change position depending on pointer speed\n= 0 disable\n> 0 draw where the pointer moves to\n< 0 draw where the pointer comes from"],
    ['offset_by_speed_slowness', 'offset by speed filter', False, 0.0, 1.0, 15.0, "how slow the offset goes back to zero when the cursor stops moving"],
    ['slow_tracking', 'slow position tracking', True, 0.0, 0.0, 10.0, "Slowdown pointer tracking speed. 0 disables it, higher values remove more jitter in cursor movements. Useful for drawing smooth, comic-like outlines."],
    ['slow_tracking_per_dab', 'slow tracking per dab', False, 0.0, 0.0, 10.0, "Similar as above but at brushdab level (ignoring how much time has past, if brushdabs do not depend on time)"],
    ['tracking_noise', 'tracking noise', True, 0.0, 0.0, 12.0, "add randomness to the mouse pointer; this usually generates many small lines in random directions; maybe try this together with 'slow tracking'"],

    ['color_h', 'color hue', True, 0.0, 0.0, 1.0, "color hue"],
    ['color_s', 'color saturation', True, -0.5, 0.0, 1.5, "color saturation"],
    ['color_v', 'color value', True, -0.5, 0.0, 1.5, "color value (brightness, intensity)"],
    ['change_color_h', 'change color hue', False, -2.0, 0.0, 2.0, "Change color hue.\n-0.1 small clockwise color hue shift\n 0.0 disable\n 0.5 counterclockwise hue shift by 180 degrees"],
    ['change_color_l', 'change color lightness (HSL)', False, -2.0, 0.0, 2.0, "Change the color lightness (luminance) using the HSL color model.\n-1.0 blacker\n 0.0 disable\n 1.0 whiter"],
    ['change_color_hsl_s', 'change color satur. (HSL)', False, -2.0, 0.0, 2.0, "Change the color saturation using the HSL color model.\n-1.0 more grayish\n 0.0 disable\n 1.0 more saturated"],
    ['change_color_v', 'change color value (HSV)', False, -2.0, 0.0, 2.0, "Change the color value (brightness, intensity) using the HSV color model. HSV changes are applied before HSL.\n-1.0 darker\n 0.0 disable\n 1.0 brigher"],
    ['change_color_hsv_s', 'change color satur. (HSV)', False, -2.0, 0.0, 2.0, "Change the color saturation using the HSV color model. HSV changes are applied before HSL.\n-1.0 more grayish\n 0.0 disable\n 1.0 more saturated"],
    ['smudge', 'smudge', False, 0.0, 0.0, 1.0, "Paint with the smudge color instead of the brush color. The smudge color is slowly changed to the color you are painting on.\n 0.0 do not use the smudge color\n 0.5 mix the smudge color with the brush color\n 1.0 use only the smudge color"],
    ['smudge_length', 'smudge length', False, 0.0, 0.5, 1.0, "This controls how fast the smudge color becomes the color you are painting on.\n0.0 immediately change the smudge color\n1.0 never change the smudge color"],
    ['eraser', 'eraser', False, 0.0, 0.0, 1.0, "how much this tool behaves like an eraser\n 0.0 normal painting\n 1.0 standard eraser\n 0.5 pixels go towards 50% transparency"],

    ['stroke_treshold', 'stroke treshold', True, 0.0, 0.0, 0.5, "How much pressure is needed to start a stroke. This affects the stroke input only. Mypaint does not need a minimal pressure to start drawing."],
    ['stroke_duration_logarithmic', 'stroke duration', False, -1.0, 4.0, 7.0, "How far you have to move until the stroke input reaches 1.0. This value is logarithmic (negative values will not inverse the process)."],
    ['stroke_holdtime', 'stroke hold time', False, 0.0, 0.0, 10.0, "This defines how long the stroke input stays at 1.0. After that it will reset to 0.0 and start growing again, even if the stroke is not yet finished.\n2.0 means twice as long as it takes to go from 0.0 to 1.0\n9.9 and bigger stands for infinite"],
    ['custom_input', 'custom input', False, -5.0, 0.0, 5.0, "Set the custom input to this value. If it is slowed down, move it towards this value (see below). The idea is that you make this input depend on a mixture of pressure/speed/whatever, and then make other settings depend on this 'custom input' instead of repeating this combination everywhere you need it.\nIf you make it change 'by random' you can generate a slow (smooth) random input."],
    ['custom_input_slowness', 'custom input filter', False, 0.0, 0.0, 10.0, "How slow the custom input actually follows the desired value (the one above). This happens at brushdab level (ignoring how much time has past, if brushdabs do not depend on time).\n0.0 no slowdown (changes apply instantly)"],

    ['elliptical_dab_ratio', 'elliptical dab: ratio', False, 1.0, 1.0, 10.0, "aspect ratio of the dabs; must be >= 1.0, where 1.0 means a perfectly round dab. TODO: linearize? start at 0.0 maybe, or log?"],
    ['elliptical_dab_angle', 'elliptical dab: angle', False, 0.0, 90.0, 180.0, "this defines the angle by which eliptical dabs are tilted\n 0.0 horizontal dabs\n 45.0 45 degrees, turned clockwise\n 180.0 horizontal again"],
    ['direction_filter', 'direction filter', False, 0.0, 2.0, 10.0, "a low value will make the direction input adapt more quickly, a high value will make it smoother"],
    ]

settings_hidden = 'color_h color_s color_v'.split()

settings_migrate = {
    # old cname              new cname        scale function
    'color_hue'          : ('change_color_h', lambda y: y*64.0/360.0),
    'color_saturation'   : ('change_color_hsv_s', lambda y: y*128.0/256.0),
    'color_value'        : ('change_color_v', lambda y: y*128.0/256.0),
    'speed_slowness'     : ('speed1_slowness', None),
    'change_color_s'     : ('change_color_hsv_s', None),
    }

# the states are not (yet?) exposed to the user
# WARNING: only append to this list, for compatibility of replay files (brush.get_state() in stroke.py)
states_list = '''
# lowlevel
x, y
pressure
dist              # "distance" moved since last dab, a new dab is drawn at 1.0
actual_radius     # used by count_dabs_to, thus a state!

smudge_ra, smudge_ga, smudge_ba, smudge_a  # smudge color stored with premultiplied alpha

actual_x, actual_y  # for slow position
norm_dx_slow, norm_dy_slow # note: now this is dx/dt * (1/radius)

norm_speed1_slow, norm_speed2_slow

stroke, stroke_started # stroke_started is used as boolean

custom_input
rng_seed

actual_elliptical_dab_ratio, actual_elliptical_dab_angle # used by count_dabs_to

direction_dx, direction_dy
'''

class BrushInput:
    pass

inputs = []
inputs_dict = {}
for i_list in inputs_list:
    i = BrushInput()
    i.name, i.hard_min, i.soft_min, i.normal, i.soft_max, i.hard_max, i.tooltip = i_list
    i.index = len(inputs)
    inputs.append(i)
    inputs_dict[i.name] = i

class BrushSetting:
    pass

settings = []
settings_dict = {}
for s_list in settings_list:
    s = BrushSetting()
    s.cname, s.name, s.constant, s.min, s.default, s.max, s.tooltip = s_list
    s.index = len(settings)
    settings.append(s)
    settings_dict[s.cname] = s
    globals()[s.cname] = s

settings_visible = [s for s in settings if s.cname not in settings_hidden]

class BrushState:
    pass

states = []
for line in states_list.split('\n'):
    line = line.split('#')[0]
    for cname in line.split(','):
        cname = cname.strip()
        if not cname: continue
        st = BrushState()
        st.cname = cname
        st.index = len(states)
        states.append(st)
