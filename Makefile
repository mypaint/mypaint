PROFILE = -g -pg
CFLAGS = $(PROFILE) -O3 `pkg-config --cflags gtk+-2.0` -Wall -Werror -ansi
LDFLAGS = $(PROFILE) -O3 `pkg-config --libs gtk+-2.0` -Wall -Werror -ansi -lfann

all:	drawmain

drawmain:	surfacepaint.o drawmain.o neural.o nntrainer.c

clean:
	rm *.o drawmain trainertest

trainertest: nntrainer.c
	gcc $(CFLAGS) $(LDFLAGS) -Dtest_nntrainer=main nntrainer.c -o trainertest

