"""Brush Settings

This is used to generate brushsettings.h
It is also imported at runtime.
"""

inputs_list = [
    # name, default maximum, hard maximum, tooltip
    ['pressure', 1.0, True, "The pressure reported by the tablet, between 0.0 and 1.0. If you use the mouse, it will be 0.5 when a button is pressed and 0.0 otherwise."],
    ['speed', 5.0, False, "How fast you currently move. This can change very quickly. Try 'print input values' from the 'help' menu to get a feeling for the range."],
    ['speed2', 5.0, False, "Same as speed, but changes slower. Also look at the 'speed2 slowness' setting."],
    ['random', 1.0, False, "Random noise, changing at each evaluation. Equally distributed between 0 and 1."],
    ['stroke', 1.0, True, "This input slowly goes from zero to one while you draw a stroke. It can also be configured to jump back to zero periodically while you move. Look at the 'stroke duration' and 'stroke hold time' settings."],
    ['custom', 1.0, True, "This is a user defined input. See '' "],
    ]

settings_list = [
    # internal name, displayed name, constant, minimum, default, maximum, tooltip
    ['opaque', 'opaque', False, 0.0, 1.0, 1.0, "0 means brush is transparent, 1 fully visible"],
    ['opaque_multiply', 'opaque multiply', False, 0.0, 0.0, 1.0, "This gets multiplied with opaque. It is used for making opaque depend on pressure (or other inputs)."],
    ['radius_logarithmic', 'radius', False, -1.0, 2.0, 5.0, "basic brush radius (logarithmic)\n 0.7 means 2 pixels\n 3.0 means 20 pixels"],
    ['hardness', 'hardness', False, 0.0, 1.0, 1.0, "hard brush-circle borders (setting to zero will draw nothing; it's not implemented like in GIMP, I haven't figured out yet)"],
    ['dabs_per_basic_radius', 'dabs per basic radius', True, 0.0, 0.0, 5.0, "how many dabs to draw while the pointer moves a distance of one brush radius (more precise: the base value of the radius)"],
    ['dabs_per_actual_radius', 'dabs per actual radius', True, 0.0, 2.0, 5.0, "same as above, but the radius actually drawn is used, which can change dynamically"],
    ['dabs_per_second', 'dabs per second', True, 0.0, 0.0, 80.0, "dabs to draw each second, no matter how far the pointer moves"],
    ['radius_by_random', 'radius by random', False, 0.0, 0.0, 10.0, "alter the radius randomly each dab\n 0.0 disable\n 0.7 biggest radius is twice as large as smallest\n 3.0 biggest radius 20 times as large as smallest"],
    ['speed1_slowness', 'speed slowness', False, 0.0, 0.04, 0.2, "how slow the input speed is following the real speed\n0.0 change immediatly as your speed changes (not recommended, but try it)"],
    ['speed2_slowness', 'speed2 slowness', False, 0.0, 0.8, 3.0, "how slow the input speed2 is following the real speed\nsame as above, but note that the range is larger; this is supposed to be slower one of both speeds"],
    ['offset_by_random', 'offset by random', False, 0.0, 0.0, 2.0, "add randomness to the position where each dab is drawn\n 0.0 disabled\n 1.0 standard deviation is one basic radius away"],
    ['offset_by_speed', 'offset by speed', False, -3.0, 0.0, 3.0, "change position depending on pointer speed\n= 0 disable\n> 0 draw where the pointer moves to\n< 0 draw where the pointer comes from"],
    ['offset_by_speed_slowness', 'offset by speed slowness', False, 0.0, 1.0, 15.0, "how slow the offset goes back to zero when the cursor stops moving; 0 means there will never be any offset left"],
    ['saturation_slowdown', 'saturation slowdown', False, -1.0, 0.0, 1.0, "When painting black, it soon gets black completely. This setting controls how fast the final brush color is taken:\n 1.0 slowly\n 0.0 disable\n-1.0 even faster\nThis is nolinear and causes strange effects when it happens too fast. Set occupancy low enough to avoid this.\nFor example, a full-occupancy black stroke might get brighter over grey areas than over white ones.\nFIXME: this setting seems not to work as I expected. I reccomend to set it to zero for anything else than black/white drawing."],
    ['slow_tracking', 'slow position tracking', True, 0.0, 0.0, 10.0, "Slowdown pointer tracking speed. 0 disables it, higher values remove more jitter in cursor movements. Useful for drawing smooth, comic-like outlines."],
    ['slow_tracking_per_dab', 'slow tracking per dab', False, 0.0, 0.0, 10.0, "Similar as above but at brushdab level (ignoring how much time has past, if brushdabs do not depend on time)"],
    ['color_value', 'color brightness', False, -2.0, 0.0, 2.0, "change the color brightness (also known as intensity or value) depending from the choosen color\n-1.0 darker\n 0.0 disable\n 1.0 brigher"],
    ['color_saturation', 'color saturation', False, -2.0, 0.0, 2.0, "change the color saturation\n-1.0 more grayish\n 0.0 disable\n 1.0 more saturated"],
    ['color_hue', 'color hue', False, -2.0, 0.0, 2.0, "change color hue\n-1.0 clockwise color hue shift\n 0.0 disable\n 1.0 counterclockwise hue shift"],
    ['adapt_color_from_image', 'adapt color from image', False, 0.0, 0.0, 1.0, "slowly change the color to the one you're painting on (some kind of smudge tool)\nNote that this happens /before/ the hue/saturation/brighness adjustment below: you can get very different effects (eg brighten image) by combining with them."],
    ['change_radius', 'change radius', False, -1.0, 0.0, 1.0, "Modify the basic radius (the one above) permanently each dab. This will slowly increment/decrement its actual value, with all the consequences. The slider above will not change, but it should - FIXME: this is a small bug; also, the changed radius will never be saved."],
    ['stroke_treshold', 'stroke treshold', True, 0.0, 0.0, 0.5, "How much pressure is needed to start a stroke. This affects the stroke input only. Mypaint does not need a minimal pressure to start drawing."],
    ['stroke_duration_logarithmic', 'stroke duration', False, -1.0, 4.0, 7.0, "How far you have to move until the stroke input reaches 1.0. This value is logarithmic (negative values will not inverse the process)."],
    ['stroke_holdtime', 'stroke hold time', False, 0.0, 0.0, 10.0, "This defines how long the stroke input stays at 1.0. After that it will reset to 0.0 and start growing again, even if the stroke is not yet finished.\n2.0 means twice as long as it takes to go from 0.0 to 1.0\n9.9 and bigger stands for infinite"],
    ['custom_input', 'custom input', False, -5.0, 0.0, 5.0, "Set the custom input to this value. If it is slowed down, move it towards this value (see below). The idea is that you make this input depend on a mixture of pressure/speed/whatever, and then make other settings depend on this 'custom input' instead of repeating this combination everywhere you need it.\nIf you make it change 'by random' you can generate a slow (smooth) random input."],
    ['custom_input_slowness', 'custom input slowness', False, 0.0, 0.0, 10.0, "How slow the custom input actually follows the desired value (the one above). This happens at brushdab level (ignoring how much time has past, if brushdabs do not depend on time).\n0.0 no slowdown (changes apply instantly)"],
    ]

class BrushInput:
    pass

inputs = []
for i_list in inputs_list:
    i = BrushInput()
    i.name, i.max, i.hardmax, i.tooltip = i_list
    i.index = len(inputs)
    inputs.append(i)

class BrushSetting:
    pass

settings = []
for s_list in settings_list:
    s = BrushSetting()
    s.cname, s.name, s.constant, s.min, s.default, s.max, s.tooltip = s_list
    s.index = len(settings)
    settings.append(s)
