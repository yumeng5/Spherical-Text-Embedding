CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: jose

jose : jose.c
	$(CC) jose.c -o jose $(CFLAGS)

clean:
	rm -rf jose 
