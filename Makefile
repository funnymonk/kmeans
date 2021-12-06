OBJS=kmeans.o
SIMDOBJS=kmeans-simd.o
EXE=8d-example 8d-example-simd 

CDBG +=
CFLAGS=-O3 -std=c99 -mfma -mavx -mavx2 -fopenmp
CFLAGS += $(CDBG)

all: $(EXE)

clean:
	@rm -f *.o $(EXE)

8d-example: $(OBJS) 8d-example.o
	$(CC) $(CFLAGS) $^ -o $@

8d-example-simd: $(SIMDOBJS) 8d-example-simd.o
	$(CC) $(CFLAGS) $^ -o $@
