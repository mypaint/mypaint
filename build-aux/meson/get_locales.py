#!/usr/bin/env python3

# This file is part of MyPaint.
# Copyright (C) 2024 the MyPaint project
#
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later
# version.

# This script prints a python-formatted list of locales that the build should use.
# The script receives two arguments. 1: locale_dir (path), 2: trtanslation_threshold (integer from 0-100)
# The os library validates argument 1, argument 2 should be validated by Meson before it calls me.

import os
import sys
import polib

# Definitions
locale_dir = sys.argv[1]
locales = [locale[:-3] for locale in os.listdir(locale_dir) if locale.endswith('.po')]
translation_threshold = float(sys.argv[2]) / float(100)

if translation_threshold == 0:
    print(*locales, sep=' ')
else:
    template_path = os.path.join(locale_dir, "mypaint.pot")
    total = len(polib.pofile(template_path))

    locale_completion = {}
    for locale in locales:
        locale_path = os.path.join(locale_dir, locale + ".po")
        locale_completion[locale] = len(polib.pofile(locale_path).translated_entries())

    # Some locales will always be included.
    for locale in ["en_CA", "en_GB"]:
        locale_completion[locale] = total

    good_locales = [k for k, v in locale_completion.items() if (v / total) >= translation_threshold]
    print(*good_locales, sep=' ')
