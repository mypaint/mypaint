#include <pygobject.h>

/* PLATFORM_WIN32 defined here */
#include "config.h"
 
void mydrawwidget_register_classes (PyObject *d); 
extern PyMethodDef mydrawwidget_functions[];
 
/* DL_EXPORT(void) */
/* DL_export not works with mingw32? it's cygwin specific ? */

#ifdef PLATFORM_WIN32
// the dll exports
#warning PLATFORM_WIN32 
#  define EXPORT __declspec(dllexport)
#else
#    define EXPORT
#endif

void EXPORT initmydrawwidget(void)
{
    PyObject *m, *d;
 
    init_pygobject ();
 
    m = Py_InitModule ("mydrawwidget", mydrawwidget_functions);
    d = PyModule_GetDict (m);
 
    mydrawwidget_register_classes (d);
 
    if (PyErr_Occurred ()) {
        Py_FatalError ("can't initialise module mydrawwidget");
    }
}
