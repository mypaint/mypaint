INCLUDES = $(PYTHON_INCLUDES) $(PYGTK_CFLAGS)

defsdir = $(datadir)/pygtk/2.0/defs
#pkgpyexecdir=$(pkglibdir)/pythonlib

# mydrawwidget binding
pkgpyexec_LTLIBRARIES = mydrawwidget.la

mydrawwidget_la_CFLAGS = $(MYPAINT_CFLAGS)

if PLATFORM_WIN32
no_undefined = -no-undefined
shared = -shared
avoid_version = -avoid-version
#windows_flags = -mno-cygwin -mms-bitfields -mwindows -mconsole
else
no_undefined =
shared =
avoid_version =
# windows_flags =
endif


mydrawwidget_la_LDFLAGS = $(shared) $(no_undefined) $(common_ldflags) -module  -export-symbols-regex initmydrawwidget $(avoid_version)
mydrawwidget_la_LIBADD = $(MYPAINT_LIBS) $(PYTHON_LIBS)

nodist_mydrawwidget_la_CFLAGS = $(MYPAINT_CFLAGS)
nodist_mydrawwidget_la_SOURCES = \
	mydrawwidget.defs.c 

mydrawwidget_la_SOURCES = \
	mydrawwidgetmodule.c \
	gtkmydrawwidget.c \
	gtkmybrush.c \
	gtkmysurface.c \
	gtkmysurfaceold.c \
	brush_dab.c \
	helpers.c \
	mapping.c \
	mymarshal.c \
	stroke_recording.c

CLEANFILES = mydrawwidget.defs.c mydrawwidget.defs  mydrawwidget.c brushsettings.h mymarshal.h mymarshal.c
EXTRA_DIST = mydrawwidget.override

# PYTHON PART
bin_SCRIPTS = mypaint

mypaintlibdir=$(pkgdatadir)/python

mypaintlib_PYTHON = \
        application.py \
        brush.py \
        brushselectionwindow.py \
        brushsettings.py \
        brushsettingswindow.py \
        colorselectionwindow.py \
        drawwindow.py \
        functionwindow.py \
        helpers.py \
        infinitemydrawwidget.py \
        layerswindow.py

mypaintbrushesdir=$(pkgdatadir)/brushes

mypaintbrushes_DATA = $(wildcard $(srcdir)/brushes/*)

mydrawwidget.defs: gtkmydrawwidget.h gtkmybrush.h gtkmysurface.h gtkmysurfaceold.h Makefile fix_generated_defs.py
	$(PYGTK_H2DEF) gtkmydrawwidget.h gtkmybrush.h gtkmysurface.h gtkmysurfaceold.h > mydrawwidget.defs
	./fix_generated_defs.py

mydrawwidget.defs.c: mydrawwidget.defs mydrawwidget.override
	$(PYGTK_CODEGEN) --prefix mydrawwidget \
	--register $(PYGTK_DEFSDIR)/gdk-types.defs \
	--register $(PYGTK_DEFSDIR)/gtk-types.defs \
	--override mydrawwidget.override \
	mydrawwidget.defs > mydrawwidget.defs.c

brushsettings.h:	generate.py brushsettings.py
	./generate.py

gtkmydrawwidget.c:	brushsettings.h gtkmydrawwidget.h

gtkmybrush.c:	brushsettings.h gtkmybrush.h

mapping.c:	 mapping.h

gtkmysurface.c:	gtkmysurface.h mymarshal.h

gtkmysurfaceold.c:	gtkmysurfaceold.h

mymarshal.h: mymarshal.list
	$(GLIB_GENMARSHAL) --prefix=mymarshal --header mymarshal.list > mymarshal.h

mymarshal.c: mymarshal.list
	$(GLIB_GENMARSHAL) --prefix=mymarshal --body mymarshal.list > mymarshal.c

