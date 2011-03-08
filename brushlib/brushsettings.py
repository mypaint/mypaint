# brushlib - The MyPaint Brush Library
# Copyright (C) 2007-2011 Martin Renold <martinxyz@gmx.ch>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""Brush Settings / States

This is used to generate brushsettings.hpp (see generate.py)
It is also imported at runtime.
"""

from gettext import gettext as _

inputs_list = [
    # name, hard minimum, soft minimum, normal[1], soft maximum, hard maximum, displayed name, tooltip
    ['pressure', 0.0,  0.0,  0.4,  1.0, 1.0,  _("Pressure"), _("The pressure reported by the tablet, between 0.0 and 1.0. If you use the mouse, it will be 0.5 when a button is pressed and 0.0 otherwise.")],
    ['speed1',   None, 0.0,  0.5,  4.0, None, _("Fine speed"), _("How fast you currently move. This can change very quickly. Try 'print input values' from the 'help' menu to get a feeling for the range; negative values are rare but possible for very low speed.")],
    ['speed2',   None, 0.0,  0.5,  4.0, None, _("Gross speed"), _("Same as fine speed, but changes slower. Also look at the 'gross speed filter' setting.")],
    ['random',   0.0,  0.0,  0.5,  1.0, 1.0, _("Random"), _("Fast random noise, changing at each evaluation. Evenly distributed between 0 and 1.")],
    ['stroke',   0.0,  0.0,  0.5,  1.0, 1.0, _("Stroke"), _("This input slowly goes from zero to one while you draw a stroke. It can also be configured to jump back to zero periodically while you move. Look at the 'stroke duration' and 'stroke hold time' settings.")],
    ['direction',0.0,  0.0,  0.0,  180.0, 180.0, _("Direction"), _("The angle of the stroke, in degrees. The value will stay between 0.0 and 180.0, effectively ignoring turns of 180 degrees.")],
    ['tilt_declination',0.0,  0.0,  0.0,  90.0, 90.0,  _("Declination"), _("Declination of stylus tilt. 0 when stylus is parallel to tablet and 90.0 when it's perpendicular to tablet.")],
    ['tilt_ascension',-180.0,  -180.0,  0.0,  180.0, 180.0, _("Ascension"),  _("Right ascension of stylus tilt. 0 when stylus working end points to you, +90 when rotated 90 degrees clockwise, -90 when rotated 90 degrees counterclockwise.")],
    #['motion_strength',0.0,0.0,  0.0,  1.0, 1.0,  "[EXPERIMENTAL] Same as angle, but wraps at 180 degrees. The dynamics are shared with BRUSH_OFFSET_BY_SPEED_FILTER (FIXME: which is a bad thing)."],
    ['custom',   None,-2.0,  0.0, +2.0, None, _("Custom"), _("This is a user defined input. Look at the 'custom input' setting for details.")],
    ]
    # [1] If, for example, the user increases the "by pressure" slider
    # in the "radius" control, then this should change the reaction to
    # pressure and not the "normal" radius. To implement this, we need
    # a guess what the user considers to be normal pressure.

settings_list = [
    # internal name, displayed name, constant, minimum, default, maximum, tooltip
    ['opaque', _('Opacity'), False, 0.0, 1.0, 2.0, _("0 means brush is transparent, 1 fully visible\n(also known als alpha or opacity)")],
    ['opaque_multiply', _('Opacity multiply'), False, 0.0, 0.0, 2.0, _("This gets multiplied with opaque. You should only change the pressure input of this setting. Use 'opaque' instead to make opacity depend on speed.\nThis setting is responsible to stop painting when there is zero pressure. This is just a convention, the behaviour is identical to 'opaque'.")],
    ['opaque_linearize', _('Opacity linearize'), True, 0.0, 0.9, 2.0, _("Correct the nonlinearity introduced by blending multiple dabs on top of each other. This correction should get you a linear (\"natural\") pressure response when pressure is mapped to opaque_multiply, as it is usually done. 0.9 is good for standard strokes, set it smaller if your brush scatters a lot, or higher if you use dabs_per_second.\n0.0 the opaque value above is for the individual dabs\n1.0 the opaque value above is for the final brush stroke, assuming each pixel gets (dabs_per_radius*2) brushdabs on average during a stroke")],
    ['radius_logarithmic', _('Radius'), False, -2.0, 2.0, 5.0, _("basic brush radius (logarithmic)\n 0.7 means 2 pixels\n 3.0 means 20 pixels")],
    ['hardness', _('Hardness'), False, 0.0, 0.8, 1.0, _("hard brush-circle borders (setting to zero will draw nothing)")],
    ['dabs_per_basic_radius', _('Dabs per basic radius'), True, 0.0, 0.0, 6.0, _("how many dabs to draw while the pointer moves a distance of one brush radius (more precise: the base value of the radius)")],
    ['dabs_per_actual_radius', _('Dabs per actual radius'), True, 0.0, 2.0, 6.0, _("same as above, but the radius actually drawn is used, which can change dynamically")],
    ['dabs_per_second', _('Dabs per second'), True, 0.0, 0.0, 80.0, _("dabs to draw each second, no matter how far the pointer moves")],
    ['radius_by_random', _('Radius by random'), False, 0.0, 0.0, 1.5, _("Alter the radius randomly each dab. You can also do this with the by_random input on the radius setting. If you do it here, there are two differences:\n1) the opaque value will be corrected such that a big-radius dabs is more transparent\n2) it will not change the actual radius seen by dabs_per_actual_radius")],
    ['speed1_slowness', _('Fine speed filter'), False, 0.0, 0.04, 0.2, _("how slow the input fine speed is following the real speed\n0.0 change immediately as your speed changes (not recommended, but try it)")],
    ['speed2_slowness', _('Gross speed filter'), False, 0.0, 0.8, 3.0, _("same as 'fine speed filter', but note that the range is different")],
    ['speed1_gamma', _('Fine speed gamma'), True, -8.0, 4.0, 8.0, _("This changes the reaction of the 'fine speed' input to extreme physical speed. You will see the difference best if 'fine speed' is mapped to the radius.\n-8.0 very fast speed does not increase 'fine speed' much more\n+8.0 very fast speed increases 'fine speed' a lot\nFor very slow speed the opposite happens.")],
    ['speed2_gamma', _('Gross speed gamma'), True, -8.0, 4.0, 8.0, _("same as 'fine speed gamma' for gross speed")],
    ['offset_by_random', _('Jitter'), False, 0.0, 0.0, 2.0, _("add a random offset to the position where each dab is drawn\n 0.0 disabled\n 1.0 standard deviation is one basic radius away\n<0.0 negative values produce no jitter")],
    ['offset_by_speed', _('Offset by speed'), False, -3.0, 0.0, 3.0, _("change position depending on pointer speed\n= 0 disable\n> 0 draw where the pointer moves to\n< 0 draw where the pointer comes from")],
    ['offset_by_speed_slowness', _('Offset by speed filter'), False, 0.0, 1.0, 15.0, _("how slow the offset goes back to zero when the cursor stops moving")],
    ['slow_tracking', _('Slow position tracking'), True, 0.0, 0.0, 10.0, _("Slowdown pointer tracking speed. 0 disables it, higher values remove more jitter in cursor movements. Useful for drawing smooth, comic-like outlines.")],
    ['slow_tracking_per_dab', _('Slow tracking per dab'), False, 0.0, 0.0, 10.0, _("Similar as above but at brushdab level (ignoring how much time has past, if brushdabs do not depend on time)")],
    ['tracking_noise', _('Tracking noise'), True, 0.0, 0.0, 12.0, _("add randomness to the mouse pointer; this usually generates many small lines in random directions; maybe try this together with 'slow tracking'")],

    ['color_h', _('Color hue'), True, 0.0, 0.0, 1.0, _("color hue")],
    ['color_s', _('Color saturation'), True, -0.5, 0.0, 1.5, _("color saturation")],
    ['color_v', _('Color value'), True, -0.5, 0.0, 1.5, _("color value (brightness, intensity)")],
    ['change_color_h', _('Change color hue'), False, -2.0, 0.0, 2.0, _("Change color hue.\n-0.1 small clockwise color hue shift\n 0.0 disable\n 0.5 counterclockwise hue shift by 180 degrees")],
    ['change_color_l', _('Change color lightness (HSL)'), False, -2.0, 0.0, 2.0, _("Change the color lightness (luminance) using the HSL color model.\n-1.0 blacker\n 0.0 disable\n 1.0 whiter")],
    ['change_color_hsl_s', _('Change color satur. (HSL)'), False, -2.0, 0.0, 2.0, _("Change the color saturation using the HSL color model.\n-1.0 more grayish\n 0.0 disable\n 1.0 more saturated")],
    ['change_color_v', _('Change color value (HSV)'), False, -2.0, 0.0, 2.0, _("Change the color value (brightness, intensity) using the HSV color model. HSV changes are applied before HSL.\n-1.0 darker\n 0.0 disable\n 1.0 brigher")],
    ['change_color_hsv_s', _('Change color satur. (HSV)'), False, -2.0, 0.0, 2.0, _("Change the color saturation using the HSV color model. HSV changes are applied before HSL.\n-1.0 more grayish\n 0.0 disable\n 1.0 more saturated")],
    ['smudge', _('Smudge'), False, 0.0, 0.0, 1.0, _("Paint with the smudge color instead of the brush color. The smudge color is slowly changed to the color you are painting on.\n 0.0 do not use the smudge color\n 0.5 mix the smudge color with the brush color\n 1.0 use only the smudge color")],
    ['smudge_length', _('Smudge length'), False, 0.0, 0.5, 1.0, _("This controls how fast the smudge color becomes the color you are painting on.\n0.0 immediately change the smudge color\n1.0 never change the smudge color")],
    ['smudge_radius_log', _('Smudge radius'), False, -1.6, 0.0, 1.6, _("This modifies the radius of the circle where color is picked up for smudging.\n 0.0 use the brush radius \n-0.7 half the brush radius\n+0.7 twice the brush radius\n+1.6 five times the brush radius (slow)")],
    ['eraser', _('Eraser'), False, 0.0, 0.0, 1.0, _("how much this tool behaves like an eraser\n 0.0 normal painting\n 1.0 standard eraser\n 0.5 pixels go towards 50% transparency")],

    ['stroke_threshold', _('Stroke threshold'), True, 0.0, 0.0, 0.5, _("How much pressure is needed to start a stroke. This affects the stroke input only. Mypaint does not need a minimal pressure to start drawing.")],
    ['stroke_duration_logarithmic', _('Stroke duration'), False, -1.0, 4.0, 7.0, _("How far you have to move until the stroke input reaches 1.0. This value is logarithmic (negative values will not inverse the process).")],
    ['stroke_holdtime', _('Stroke hold time'), False, 0.0, 0.0, 10.0, _("This defines how long the stroke input stays at 1.0. After that it will reset to 0.0 and start growing again, even if the stroke is not yet finished.\n2.0 means twice as long as it takes to go from 0.0 to 1.0\n9.9 and bigger stands for infinite")],
    ['custom_input', _('Custom input'), False, -5.0, 0.0, 5.0, _("Set the custom input to this value. If it is slowed down, move it towards this value (see below). The idea is that you make this input depend on a mixture of pressure/speed/whatever, and then make other settings depend on this 'custom input' instead of repeating this combination everywhere you need it.\nIf you make it change 'by random' you can generate a slow (smooth) random input.")],
    ['custom_input_slowness', _('Custom input filter'), False, 0.0, 0.0, 10.0, _("How slow the custom input actually follows the desired value (the one above). This happens at brushdab level (ignoring how much time has past, if brushdabs do not depend on time).\n0.0 no slowdown (changes apply instantly)")],

    ['elliptical_dab_ratio', _('Elliptical dab: ratio'), False, 1.0, 1.0, 10.0, _("aspect ratio of the dabs; must be >= 1.0, where 1.0 means a perfectly round dab. TODO: linearize? start at 0.0 maybe, or log?")],
    ['elliptical_dab_angle', _('Elliptical dab: angle'), False, 0.0, 90.0, 180.0, _("this defines the angle by which eliptical dabs are tilted\n 0.0 horizontal dabs\n 45.0 45 degrees, turned clockwise\n 180.0 horizontal again")],
    ['direction_filter', _('Direction filter'), False, 0.0, 2.0, 10.0, _("a low value will make the direction input adapt more quickly, a high value will make it smoother")],

    ['lock_alpha', _('Lock alpha'), False, 0.0, 0.0, 1.0, _("do not modify the alpha channel of the layer (paint only where there is paint already)\n 0.0 normal painting\n 0.5 half of the paint gets applied normally\n 1.0 alpha channel fully locked")],
    ]

settings_hidden = 'color_h color_s color_v'.split()

settings_migrate = {
    # old cname              new cname        scale function
    'color_hue'          : ('change_color_h', lambda y: y*64.0/360.0),
    'color_saturation'   : ('change_color_hsv_s', lambda y: y*128.0/256.0),
    'color_value'        : ('change_color_v', lambda y: y*128.0/256.0),
    'speed_slowness'     : ('speed1_slowness', None),
    'change_color_s'     : ('change_color_hsv_s', None),
    'stroke_treshold'    : ('stroke_threshold', None),
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
declination, ascension
'''

class BrushInput:
    pass

inputs = []
inputs_dict = {}
for i_list in inputs_list:
    i = BrushInput()
    i.name, i.hard_min, i.soft_min, i.normal, i.soft_max, i.hard_max, i.dname, i.tooltip = i_list
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
