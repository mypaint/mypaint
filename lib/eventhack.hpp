/*
 * This file is part of MyPaint.
 * Copyright (C) 2013 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 */

#ifndef __HAVE_EVENTHACK
#define __HAVE_EVENTHACK

#include <gtk/gtk.h>
#include "Python.h"
#include <pygobject.h>


#if defined(GDK_WINDOWING_X11)

#include <gdk/gdkx.h>
#include <X11/extensions/XInput2.h>
#include <X11/Xlib.h>


/* _evhack_x11_event_filter:
 *
 * Event filter routine set up by evhack_gdk_window_add_filter().
 */

static GdkFilterReturn
_evhack_x11_event_filter (GdkXEvent *xevent_gdk,
                          GdkEvent *event_gdk,   // out, unused
                          gpointer data_ptr)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
#ifdef HEAVY_DEBUG
    assert(PySequence_Check((PyObject *)data_ptr));
#endif
    PyObject *tdw = NULL;
    PyObject *mode = NULL;
    if (! PyArg_ParseTuple((PyObject *)data_ptr, "OO", &tdw, &mode)) {
        PyGILState_Release(gstate);
        return GDK_FILTER_CONTINUE;
    }
    XEvent *xevent = (XEvent *) xevent_gdk;
    gdouble x = 0.0;
    gdouble y = 0.0;
    Time t = 0;
    gboolean captured = FALSE;
    if (xevent->type == MotionNotify) {
        // XXX: NOT TESTED (perhaps it could by turning off per-device events?)
        XMotionEvent *evt = (XMotionEvent *)xevent;
        x = (gdouble) evt->x;
        y = (gdouble) evt->y;
        t = (Time) evt->time;
        captured = TRUE;
    }
    else if (xevent->type == GenericEvent) {
        // Xi2 event
        XGenericEventCookie *cookie = &xevent->xcookie;
        XIEvent *ev = (XIEvent *) cookie->data;
        if (ev->evtype == XI_Motion) {
            XIDeviceEvent *xev = (XIDeviceEvent *) ev;
            x = (gdouble) xev->event_x;
            y = (gdouble) xev->event_y;
            t = (Time) xev->time;
            captured = TRUE;
        }
    }
    if (captured) {
        PyObject *result = PyObject_CallMethod(
            mode, "queue_evhack_position", "(Oddl)",
            tdw, x, y, t
        );
        if (result != NULL) {
            Py_DECREF(result);
        }
    }
    PyGILState_Release(gstate);
    return GDK_FILTER_CONTINUE;
}


#endif // defined(GDK_WINDOWING_X11)


#if defined(GDK_WINDOWING_WIN32)

/* TODO: Windows platform-specific code here */

#endif // defined(GDK_WINDOWING_WIN32)




/* evhack_gdk_window_add_filter:
 *
 * Adds an event filter for a TDW. gdk_window_add_filter() is not exported for
 * use in PyGI code, but the interface here in analogous.
 *
 * "data" must be a Python tuple (TDW, MODE) where MODE is an input mode object
 * supporting a queue_evhack_position() method, e.g. a FreehandOnlyMode, and
 * TDW is the TiledDrawWidget this filter is to be attached to.
 *
 * The event filter set up calls MODE.queue_evhack_position(x, y, t) for each
 * received event prior to GTK3.8+ motion compression, where x and y are the
 * position of the event, and t is its timestamp.
 */

void
evhack_gdk_window_add_filter (PyObject *window, PyObject *data)
{
    GdkWindow *win_gdk = GDK_WINDOW(((PyGObject *)window)->obj);
    GdkDisplay *display = gdk_window_get_display(win_gdk);
#ifdef GDK_WINDOWING_X11
    if (GDK_IS_X11_DISPLAY(display)) {
        Py_INCREF(data);
        gdk_window_add_filter(win_gdk, _evhack_x11_event_filter, data);
    }
#endif
}


/* evhack_gdk_window_remove_filter:
 *
 * Removes an event filter for a TDW. gdk_window_remove_filter() is not
 * exported for use in PyGI code, but the interface here in analogous.
 *
 * "data" must be the exact tuple passed to evhack_gdk_window_add_filter(), and
 * tuple identity matters.
 */

void
evhack_gdk_window_remove_filter (PyObject *window, PyObject *data)
{
    GdkWindow *win_gdk = GDK_WINDOW(((PyGObject *)window)->obj);
    GdkDisplay *display = gdk_window_get_display(win_gdk);
#ifdef GDK_WINDOWING_X11
    if (GDK_IS_X11_DISPLAY(display)) {
        gdk_window_remove_filter(win_gdk, _evhack_x11_event_filter, data);
        Py_DECREF(data);
    }
#endif
}

#endif // !defined(__HAVE_EVENTHACK)
