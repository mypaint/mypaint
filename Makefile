# not all dependencies are in here, you need to 'make clean' sometimes.

# FIXME: autodetect python version to build against (instead of 2.3)
PYTHON_VERSION = 2.3
PROFILE = #-g #-pg
CFLAGS = $(PROFILE) -O3 `pkg-config --cflags gtk+-2.0 pygtk-2.0` -Wall -Werror -I/usr/include/python$(PYTHON_VERSION) -I.
LDFLAGS = $(PROFILE) -O3 `pkg-config --libs gtk+-2.0 pygtk-2.0` -Wall -Werror
DEFSDIR = `pkg-config --variable=defsdir pygtk-2.0`

all:	checkdepend mydrawwidget.so


# some ugly quick dependency check
checkdepend:
	test -r /usr/include/python$(PYTHON_VERSION)/Python.h || (echo ; echo "/usr/include/python$(PYTHON_VERSION)/Python.h does not exist. Try 'make PYTHON_VERSION=2.4', or install the python developement files." ; exit 1)
.PHONY:	checkdepend

brushsettings.h:	generate.py brushsettings.py
	./generate.py

gtkmydrawwidget.o:	brushsettings.h gtkmydrawwidget.c gtkmydrawwidget.h
	cc $(CFLAGS) -c -o $@ gtkmydrawwidget.c

gtkmybrush.o:	brushsettings.h gtkmybrush.c gtkmybrush.h
	cc $(CFLAGS) -c -o $@ gtkmybrush.c

mapping.o:	mapping.c mapping.h
	cc $(CFLAGS) -c -o $@ mapping.c

gtkmysurface.o:	gtkmysurface.c gtkmysurface.h mymarshal.h
	cc $(CFLAGS) -c -o $@ gtkmysurface.c

gtkmysurfaceold.o:	gtkmysurfaceold.c gtkmysurfaceold.h
	cc $(CFLAGS) -c -o $@ gtkmysurfaceold.c

mymarshal.h: mymarshal.list
	glib-genmarshal --prefix=mymarshal --header mymarshal.list > mymarshal.h

mymarshal.c: mymarshal.list
	glib-genmarshal --prefix=mymarshal --body mymarshal.list > mymarshal.c

mymarshal.o:	mymarshal.h mymarshal.c
	cc $(CFLAGS) -c -o $@ mymarshal.c

clean:
	rm *.o *.so brushsettings.h mydrawwidget.defs mydrawwidget.defs.c mymarshal.c mymarshal.h

mydrawwidget.defs.c: mydrawwidget.defs mydrawwidget.override
	pygtk-codegen-2.0 --prefix mydrawwidget \
	--register $(DEFSDIR)/gdk-types.defs \
	--register $(DEFSDIR)/gtk-types.defs \
	--override mydrawwidget.override \
	mydrawwidget.defs > mydrawwidget.defs.c

mydrawwidget.defs: gtkmydrawwidget.h gtkmybrush.h gtkmysurface.h gtkmysurfaceold.h Makefile fix_generated_defs.py
	python /usr/share/pygtk/2.0/codegen/h2def.py gtkmydrawwidget.h gtkmybrush.h gtkmysurface.h gtkmysurfaceold.h > mydrawwidget.defs
	./fix_generated_defs.py

mydrawwidget.so: mydrawwidget.defs.c mydrawwidgetmodule.c gtkmydrawwidget.o gtkmybrush.o \
                 gtkmysurface.o gtkmysurfaceold.o brush_dab.o helpers.o mapping.o mymarshal.o
	$(CC) $(LDFLAGS) $(CFLAGS) -shared $^ -o $@

PREFIX=/usr/local
install: all
	install -d $(PREFIX)/lib/mypaint
	install *.py $(PREFIX)/lib/mypaint/
	install mydrawwidget.so $(PREFIX)/lib/mypaint/
	install -d $(PREFIX)/share/mypaint
	install -d $(PREFIX)/share/mypaint/brushes
	install brushes/*  $(PREFIX)/share/mypaint/brushes/
	install mypaint $(PREFIX)/bin/
	python -c "f = '$(PREFIX)/bin/mypaint'; s = open(f).read().replace('prefix = None', 'prefix = \"$(PREFIX)\"') ; open(f, 'w').write(s)"
