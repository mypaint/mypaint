PROFILE = -g -pg
CFLAGS = $(PROFILE) -O1 `pkg-config --cflags gtk+-2.0` -Wall -Werror
LDFLAGS = $(PROFILE) -O1 `pkg-config --libs gtk+-2.0` -Wall -Werror -lfann

all:	drawmain

drawmain:	surface.o drawmain.o brush.o nntrainer.c

clean:
	rm *.o drawmain #trainertest

trainertest: nntrainer.c
	gcc $(CFLAGS) $(LDFLAGS) -Dtest_nntrainer=main nntrainer.c -o trainertest

