"Brush Settings (source for generated code, but also imported at runtime)"

inputs_list = [
    # name, default maximum, hard maximum (, TODO: tooltip)
    ['pressure', 1.0, True],
    ['speed', 5.0, False],
    # ...
    ]

settings_list = [
    # internal name, displayed name, special, minimum, default, maximum, tooltip
    ['opaque', 'opaque', None, 0.0, 1.0, 1.0, "0 means brush is transparent, 1 fully visible"],
    ['radius_logarithmic', 'radius', None, -0.5, 2.0, 5.0, "basic brush radius (logarithmic)\n 0.7 means 2 pixels\n 3.0 means 20 pixels"],
    ['hardness', 'hardness', None, 0.0, 1.0, 1.0, "hard brush-circle borders (setting to zero will draw nothing; it's not implemented like in GIMP, I haven't figured out yet)"],
    ['dabs_per_basic_radius', 'dabs per basic radius', None, 0.0, 0.0, 5.0, "dabs to draw while the pointer moves one brush radius\nthis is a constant, if you make it depend on something (pressure, speed, whatever) this will be ignored"],
    ['dabs_per_actual_radius', 'dabs per actual radius', None, 0.0, 2.0, 5.0, "same as above, but the radius actually drawn is used, which might change dynamically\nthis is a constant, if you make it depend on something (pressure, speed, whatever) this will be ignored"],
    ['dabs_per_second', 'dabs per second', None, 0.0, 0.0, 80.0, "dabs to draw each second, no matter how far the pointer moves\nthis is a constant, if you make it depend on something (pressure, speed, whatever) this will be ignored"],
    ['opaque_by_pressure', 'opaque by pressure', None, 0.0, 1.0, 5.0, "opaque above will get multiplyed by pressure times this value\nFIXME: this is really useless and has the same effect as oppaque."],
    ['radius_by_pressure', 'radius by pressure', None, -10.0, 0.1, 10.0, "how much more pressure will increase the radius\nwithout pressure, the radius is unchanged\n 0.0 disable\n 0.7 double radius at full pressure\n-0.7 half radius at full pressure\n3.0 20 times radius at full pressure"],
    ['radius_by_random', 'radius by random', None, 0.0, 0.0, 10.0, "alter the radius randomly each dab\n 0.0 disable\n 0.7 biggest radius is twice as large as smallest\n 3.0 biggest radius 20 times as large as smallest"],
    ['radius_by_speed', 'radius by speed', None, -10.0, 0.0, 10.0, "alter the radius depending on current speed; this is also affected by 'speed abs slowness' below, but not by 'speed slowness'"],
    ['rbs__speedabs_slowness', 'radius by speed: speed abs slowness', None, 0.0, 0.0, 10.0, "how slow to update the speed value\n0.0 change the radius immediatly as your speed changes"],
    ['offset_by_random', 'offset by random', None, 0.0, 0.0, 10.0, "add randomness to the position where the dab is drawn\n 0.0 disabled\n 1.0 standard derivation is one radius away (as set above, not the actual radius)"],
    ['offset_by_speed', 'offset by speed', None, -30.0, 0.0, 30.0, "change position depending on pointer speed\n= 0 disable\n> 0 draw where the pointer moves to\n< 0 draw where the pointer comes from"],
    ['obs__speed_slowness', 'offset by speed: speed slowness', None, 0.0, 0.0, 10.0, "use a short-term speed (0) or a long time average speed (big) for above"],
    ['obs__speedabs_slowness', 'offset by speed: speed abs slowness', None, 0.0, 0.0, 10.0, "how fast to adapt the absolut value of the speed (in contrast to the direction)"],
    ['saturation_slowdown', 'saturation slowdown', None, -1.0, 0.0, 1.0, "When painting black, it soon gets black completely. This setting controls how fast the final brush color is taken:\n 1.0 slowly\n 0.0 disable\n-1.0 even faster\nThis is nolinear and causes strange effects when it happens too fast. Set occupancy low enough to avoid this.\nFor example, a full-occupancy black stroke might get brighter over grey areas than over white ones.\nFIXME: this setting seems not to work as I expected. I reccomend to set it to zero for anything else than black/white drawing."],
    ['position_T', 'slow position', None, 0.0, 0.0, 10.0, "Slowdown pointer tracking speed. 0 disables it, higher values remove more jitter in cursor movements. Useful for drawing smooth, comic-like outlines."],
    ['position_T2', 'slow position 2', None, 0.0, 0.0, 10.0, "Similar as above but at brushdab level (ignoring how much time has past, if brushdabs do not depend on time)"],
    ['color_value_by_pressure', 'color brightness by pressure', None, -2.0, 0.0, 2.0, "change the color brightness (also known as intensity or value) depending on pressure\n-1.0 high pressure: darker\n 0.0 disable\n 1.0 high pressure: brigher"],
    ['color_value_by_random', 'color brightness by random', None, 0.0, 0.0, 1.0, "noisify the color brightness (also known as intensity or value)"],
    ['color_saturation_by_pressure', 'color saturation by pressure', None, -2.0, 0.0, 2.0, "change the color saturation depending on pressure\n-1.0 high pressure: grayish\n 0.0 disable\n 1.0 high pressure: saturated"],
    ['color_saturation_by_random', 'color saturation by random', None, 0.0, 0.0, 1.0, "noisify the color saturation"],
    ['color_hue_by_pressure', 'color hue by pressure', None, -2.0, 0.0, 2.0, "change color hue depending on pressure\n-1.0 high pressure: clockwise color hue shift\n 0.0 disable\n 1.0 high pressure: counterclockwise hue shift"],
    ['color_hue_by_random', 'color hue by random', None, 0.0, 0.0, 1.0, "noisify the color hue"],
    ['adapt_color_from_image', 'adapt color from image', None, 0.0, 0.0, 1.0, "slowly change the color to the one you're painting on (some kind of smudge tool)\nNote that this happens /before/ the hue/saturation/brighness adjustment below: you can get very different effects (eg brighten image) by combining with them."],
    ]

class BrushInput:
    pass

inputs = []
for i_list in inputs_list:
    i = BrushInput()
    i.name, i.max, i.hardmax = i_list
    i.index = len(inputs)
    inputs.append(i)

class BrushSetting:
    pass

settings = []
for s_list in settings_list:
    s = BrushSetting()
    s.cname, s.name, s.special, s.min, s.default, s.max, s.tooltip = s_list
    s.index = len(settings)
    settings.append(s)
