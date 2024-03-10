# This file is part of MyPaint.
# Copyright (C) 2017-2019 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Stuff for making names and keeping them unique."""

import re

from lib.gettext import C_
from lib.pycompat import unicode

UNIQUE_NAME_TEMPLATE = C_(
    "unique names: serial number needed: template",
    # TRANSLATORS: UNIQUE_NAME_TEMPLATE. Must match its regex.
    # TRANSLATORS: Leave this untranslated if you are unsure.
    # TRANSLATORS: Change only if your lang *REQUIRES* a different order or
    # TRANSLATORS: if raw digits and a space aren't enough.
    u'{name} {number}',
)

UNIQUE_NAME_REGEX = re.compile(C_(
    "unique names: regex matching a string with a serial number",
    # TRANSLATORS: UNIQUE_NAME_REGEX - regex for UNIQUE_NAME_TEMPLATE
    # TRANSLATORS: Must match its template (msgid: '{name} {number}')
    # TRANSLATORS: Leave this untranslated (or copy it) if you are unsure.
    u'^(?P<name>.*?)\\s+(?P<number>\\d+)$',
))


def make_unique_name(name, existing, start=1, always_number=None):
    """Ensures that a name is unique.

    :param unicode name: Name to be made unique.
    :param existing: An existing list or set of names.
    :type existing: anything supporting ``in``
    :param int start: starting number for numbering.
    :param unicode always_number: always number if name is this value.
    :returns: A unique name.
    :rtype: unicode

    >>> existing = set([u"abc 1", u"abc 2", u"abc"])
    >>> expected = u'abc 3'
    >>> make_unique_name(u"abc", existing) == expected
    True
    >>> expected not in existing
    True
    >>> make_unique_name(u"abc 1", existing) == expected  # still
    True

    Sometimes you may want a serial number every time if the given name
    is some specific value, normally a default. This allows your first
    item to be, for example, "Widget 1", not "Widget".

    >>> x1 = u'xyz 1'
    >>> make_unique_name(u"xyz", {}, start=1, always_number=u"xyz") == x1
    True
    >>> x2 = u'xyz 2'
    >>> make_unique_name(u"xyz", {}, start=2, always_number=u"xyz") == x2
    True

    """
    name = unicode(name)
    match = UNIQUE_NAME_REGEX.match(name)
    if match:
        base = match.group("name")
        num = int(match.group("number"))
    else:
        base = name
        num = max(0, int(start))
    force_numbering = (name == always_number)
    while (name in existing) or force_numbering:
        name = UNIQUE_NAME_TEMPLATE.format(name=base, number=num)
        num += 1
        force_numbering = False
    return name


assert UNIQUE_NAME_REGEX.match(
    UNIQUE_NAME_TEMPLATE.format(
        name="testing",
        number=12,
    )
), (
    "Translation error: lib.naming.UNIQUE_NAME_REGEX "
    "must match UNIQUE_NAME_TEMPLATE."
)
