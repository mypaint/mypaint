PROFILE = #-g -pg
CFLAGS = $(PROFILE) -O3 `gtk-config --cflags` -Wall -Werror -ansi
LDFLAGS = $(PROFILE) -O3 `gtk-config --libs` -Wall -Werror -ansi

all:	drawmain

drawmain:	surfacepaint.o drawmain.o
