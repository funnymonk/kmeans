OBJS=kmeans.o
SIMDOBJS=kmeans-simd.o
EXE=2d-example 8d-example-simd #example1# example2

CFLAGS=-g3 -O4 -std=c99 -mfma -mavx -mavx2

all: $(EXE)

clean:
	@rm -f *.o $(EXE)

example1: $(OBJS) example1.o
	$(CC) $(CFLAGS) $^ -o $@

example2: $(OBJS) example2.o
	$(CC) $(CFLAGS) $^ -o $@

2d-example: $(OBJS) 2d-example.o
	$(CC) $(CFLAGS) $^ -o $@

8d-example-simd: $(SIMDOBJS) 8d-example-simd.o
	$(CC) $(CFLAGS) $^ -o $@
