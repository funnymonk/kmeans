#!/bin/sh 

for thread in 1 2 4 8 10 16 24 40
do
for num_obj in 5600 6720 7840 8960 10752
do
make clean
sed "s/NUM_THREADS 1/NUM_THREADS $thread/" 8d-example-simd.c_orig > 8d-example-simd.c_temp 
sed "s/NUM_OBJS    ARRAY_LEN(dataset)/NUM_OBJS    $num_obj/" 8d-example-simd.c_temp > 8d-example-simd.c
#grep -rs "#define NUM_THREADS" 8d-example-simd.c 
#grep -rs "#define NUM_OBJS " 8d-example-simd.c 
rm -f 8d-example-simd.c_temp
make CDBG=-w
echo -n "num threads: $thread; dataset_size: $num_obj; cycles taken: "
./8d-example-simd | tail -n 1 | awk -F " " '{print $4}'
done
done



