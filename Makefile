CFLAGS = -O3 `gtk-config --cflags` -Wall -Werror -ansi -pedantic
LDFLAGS = -O3 `gtk-config --libs` -Wall -Werror -ansi -pedantic

all:	drawmain

wetmain:	drawmain.o surfacepaint.o
