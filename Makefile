PROFILE = -g -pg
CFLAGS = $(PROFILE) -O3 `pkg-config --cflags gtk+-2.0` -Wall -Werror -ansi
LDFLAGS = $(PROFILE) -O3 `pkg-config --libs gtk+-2.0` -Wall -Werror -ansi -lfann

trainertest: nntrainer.c
	gcc $(CFLAGS) $(LDFLAGS) -Dtest_nntrainer=main nntrainer.c -o trainertest

all:	drawmain
clean:
	rm *.o drawmain

drawmain:	surfacepaint.o drawmain.o neural.o
