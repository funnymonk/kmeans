OBJS=kmeans.o
EXE=2d-example #example1# example2

CFLAGS=-g3 -O0

all: $(EXE)

clean:
	@rm -f *.o $(EXE)

example1: $(OBJS) example1.o
	$(CC) $(CFLAGS) $^ -o $@

example2: $(OBJS) example2.o
	$(CC) $(CFLAGS) $^ -o $@

2d-example: $(OBJS) 2d-example.o
	$(CC) $(CFLAGS) $^ -o $@
