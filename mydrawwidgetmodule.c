#include <pygobject.h>
 
void mydrawwidget_register_classes (PyObject *d); 
extern PyMethodDef mydrawwidget_functions[];
 
DL_EXPORT(void)
initmydrawwidget(void)
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
