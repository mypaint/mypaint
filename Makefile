PROFILE = -g #-pg
CFLAGS = $(PROFILE) -O1 `pkg-config --cflags gtk+-2.0` -Wall -Werror
LDFLAGS = $(PROFILE) -O1 `pkg-config --libs gtk+-2.0` -Wall -Werror

all:	drawmain

drawmain:	surface.o drawmain.o brush.o helpers.o

brush_settings.inc:	brush.h generate.py
	./generate.py

brush.o:	brush_settings.inc brush.c
	cc $(CFLAGS) -c -o brush.o brush.c

clean:
	rm *.o drawmain brush_settings.inc
