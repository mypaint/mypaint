PROFILE = -g -pg
CFLAGS = $(PROFILE) -O1 `pkg-config --cflags gtk+-2.0` -Wall -Werror
LDFLAGS = $(PROFILE) -O1 `pkg-config --libs gtk+-2.0` -Wall -Werror

all:	drawmain

drawmain:	surface.o drawmain.o brush.o

clean:
	rm *.o drawmain
