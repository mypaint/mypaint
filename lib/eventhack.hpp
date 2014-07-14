/*
 * This file is part of MyPaint.
 * Copyright (C) 2013-2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 */

// Workaround for the unhelpful event compression code present in certain GDK
// versions. For affected versions, if we want events delivered with a time
// granularity of less than the frame clock, we need the platform-specific
// code defined in eventhack.cpp.


#ifndef EVENTHACK_HPP
#define EVENTHACK_HPP

#include "Python.h"


// Adds an event filter for a Gdk window. gdk_window_add_filter() is not
// exported for use in PyGI code, but the interface here in analogous.
//
// The "data" arg must be a Python tuple (TDW, MODE) where MODE is an input
// mode object supporting a queue_evhack_position() method, e.g. a
// FreehandMode, and TDW is the TiledDrawWidget this filter is to be
// attached to.
//
// The event filter set up calls MODE.queue_evhack_position(x, y, t) for each
// received event prior to GTK3.8+ motion compression, where x and y are the
// position of the event, and t is its timestamp.

void
evhack_gdk_window_add_filter (PyObject *window, PyObject *data);

// Removes an event filter for a Gdk window. gdk_window_remove_filter() is not
// exported for use in PyGI code, but the interface here is analogous to it.
// The "data" arg must be the exact tuple passed to
// evhack_gdk_window_add_filter(), and tuple identity matters.

void
evhack_gdk_window_remove_filter (PyObject *window, PyObject *data);

#endif //EVENTHACK_HPP
