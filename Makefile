PROFILE = #-g -pg
CFLAGS = $(PROFILE) -O3 `pkg-config --cflags gtk+-2.0` -Wall -Werror -ansi
LDFLAGS = $(PROFILE) -O3 `pkg-config --libs gtk+-2.0` -Wall -Werror -ansi

all:	drawmain

drawmain:	surfacepaint.o drawmain.o
