# not all dependencies are in here, you need to 'make clean' sometimes.

PROFILE = -g #-pg
CFLAGS = $(PROFILE) -O3 `pkg-config --cflags gtk+-2.0 pygtk-2.0` -Wall -Werror -I/usr/include/python2.3/ -I.
LDFLAGS = $(PROFILE) -O3 `pkg-config --libs gtk+-2.0 pygtk-2.0` -Wall -Werror
DEFSDIR = `pkg-config --variable=defsdir pygtk-2.0`

all:	mydrawwidget.so

gtkmybrush_settings.inc:	gtkmybrush.h generate.py
	./generate.py

gtkmybrush.o:	gtkmybrush_settings.inc gtkmybrush.c
	cc $(CFLAGS) -c -o $@ gtkmybrush.c

clean:
	rm *.o *.so gtkmybrush_settings.inc mydrawwidget.defs mydrawwidget.defs.c brushsettings.py

mydrawwidget.defs.c: mydrawwidget.defs mydrawwidget.override
	pygtk-codegen-2.0 --prefix mydrawwidget \
	--register $(DEFSDIR)/gdk-types.defs \
	--register $(DEFSDIR)/gtk-types.defs \
	--override mydrawwidget.override \
	mydrawwidget.defs > mydrawwidget.defs.c

mydrawwidget.defs: gtkmydrawwidget.h gtkmybrush.h surface.h
	/usr/share/pygtk/2.0/codegen/h2def.py gtkmydrawwidget.h gtkmybrush.h > mydrawwidget.defs

mydrawwidget.so: mydrawwidget.defs.c mydrawwidgetmodule.c gtkmydrawwidget.o surface.o gtkmybrush.o brush_dab.o helpers.o
	$(CC) $(LDFLAGS) $(CFLAGS) -shared $^ -o $@
