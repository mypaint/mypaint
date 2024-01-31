# This file is part of MyPaint.
# Copyright (C) 2020 by the MyPaint Development Team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from __future__ import division, print_function

import logging

logger = logging.getLogger(__name__)


def validate(input_value, default_value, input_type, predicate, error_message):
    """ Validate user input, returning a default value on failed validations

    :param input_value: The value to validate
    :param default_value: The default value to return if validation fails
    :param input_type: Type or constructor taking the input value
    :param predicate: A unary truth-value function which is run against the
    output of the type / constructor applied to the input value.
    :param error_message: Error message to log on validation failure. The
    message may contain "{value}" to refer to the input value.
    :return: The converted input value if validation is successful, otherwise
    the given default value.
    """
    try:
        value = input_type(input_value)
        assert predicate(value)
        return value
    except Exception:
        logger.exception()
        logger.error(error_message.format(value=input_value))
        return default_value
