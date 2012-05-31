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

This is used to generate brushsettings.h (see generate.py)
It is also imported at runtime.
"""

import os, gettext


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

# Mapping between the the index of the parameter and the name
input_params = ["id", "hard_minimum", "soft_minimum", "normal", "soft_maximum", "hard_maximum", "displayed_name", "tooltip"]
settings_params = ["internal_name", "displayed_name", "constant", "minimum", "default", "maximum", "tooltip"]

def load_brush_definitions_from_json(json_string):

    import json
    document = json.loads(json_string)

    def convert_params_from_dict(dictionary, param_mapping):
        indexed_list = ["XXX" for i in param_mapping]
        for key, value in dictionary.items():
            param_index = param_mapping.index(key)
            indexed_list[param_index] = value

        return indexed_list

    inputs = [convert_params_from_dict(i, input_params) for i in document['inputs']]
    settings = [convert_params_from_dict(s, settings_params) for s in document['settings']]
    states = document['states']

    return (settings, inputs, states)

dir_of_this_file = os.path.abspath(os.path.dirname(__file__))
definition_path = os.path.join(dir_of_this_file, "brushsettings.json")

settings_list, inputs_list, states_list = load_brush_definitions_from_json(open(definition_path, "r").read())


class BrushInput:
    pass

inputs = []
inputs_dict = {}
for i_list in inputs_list:
    i = BrushInput()

    i.name, i.hard_min, i.soft_min, i.normal, i.soft_max, i.hard_max, i.dname, i.tooltip = i_list

    i.dname = gettext.dgettext("libmypaint", i.dname)
    i.tooltip = gettext.dgettext("libmypaint", i.tooltip)

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

    s.name = gettext.dgettext("libmypaint", s.name)
    s.tooltip = gettext.dgettext("libmypaint", s.tooltip)

    s.index = len(settings)
    settings.append(s)
    settings_dict[s.cname] = s

settings_visible = [s for s in settings if s.cname not in settings_hidden]

class BrushState:
    pass

states = []

for line in states_list:
    line = line.split('#')[0]
    for cname in line.split(','):
        cname = cname.strip()
        if not cname: continue
        st = BrushState()
        st.cname = cname
        st.index = len(states)
        states.append(st)
