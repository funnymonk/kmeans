#include <math.h>
#include <stdio.h>

#include "kmeans.h"
#include "dataset_test.h"

#define ARRAY_LEN(X)    (sizeof(X)/sizeof((X)[0]))

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

static float d_distance(const float* a, const float* b)
{
	float *da = (float*)a;
	float *db = (float*)b;
    float distance = 0;
    int i;

    for(i=0; i<DIM; i++){
        distance += ((da[i] - db[i]) * (da[i] - db[i]));
    }

    return distance;
}

static void d_centroid(float* objs, int * clusters, size_t num_objs, int cluster, float* centroid)
{
	int i, dim;
	int num_cluster = 0;
	float sum = 0;
	float *entry;
	float *dcentroid = (float*)centroid;

	if (num_objs <= 0) return;
    for(dim = 0; dim<DIM; dim++)
        dcentroid[dim] = 0.0;

	for (i = 0; i < num_objs; i++)
	{
        entry = objs + i*DIM;
		/* Only process objects of interest */
		if (clusters[i] != cluster)
			continue;

        for(dim = 0; dim<DIM; dim++){
            dcentroid[dim] += entry[dim];
        }

		num_cluster++;
	}

	if (num_cluster)
	{
#ifdef DEBUG_PRINT
        printf("Num cluster = %d, Sum = [%f, %f]\n", 
                num_cluster, dcentroid[0], dcentroid[1]);
#endif
        for(dim = 0; dim<DIM; dim++){
            dcentroid[dim] /= num_cluster;
        }
	}
	return;
}

int
main(int nargs, char **args)
{
    unsigned long long t0, t1, sum;
	float c[][8] = {
        {1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0},
        {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},
        {2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},
        {3.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},
        {4.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},
        {5.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},
        {6.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},
        {7.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},
    };
	kmeans_config config;
	kmeans_result result;
	int i, dim, num_in_0;

	config.num_objs = ARRAY_LEN(dataset);
	config.k = 8;
	config.max_iterations = 1000;
	config.distance_method = d_distance;
	config.centroid_method = d_centroid;

	config.clusters = malloc(config.num_objs * sizeof(int));

    config.centers = &c[0][0];
    config.objs = &dataset[0][0];

	/* run k-means */
    t0 = rdtsc();
	result = kmeans(&config);
    t1 = rdtsc();

    num_in_0 = 0;
	/* print result */
	for (i = 0; i < config.num_objs; i++)
	{
#ifdef DEBUG_PRINT
        printf("%d [%d]\n", i, config.clusters[i]);
#endif
        if(config.clusters[i] == 0)
            num_in_0++;
	}

	for (i = 0; i < config.k; i++){
        printf("Centroid %d [", i);
        float *center = config.centers + i*DIM;
        for(dim = 0; dim<DIM; dim++){
            printf("%f, ", center[dim]);
        }
        printf("]\n");
    }

    int fin_arr[DIM] = {0};
	/* print result */
	for (i = 0; i < config.num_objs; i++)
	{
        fin_arr[config.clusters[i]]++;
	}
    printf("Num in each :\n");
    int total=0;
    for(int j=0; j<DIM; j++){
        total += fin_arr[j];
        printf("%d : %d\n", j, fin_arr[j]);
    }

    printf("Took %d iterations, cycles = %ld, total = %d\n",
            config.total_iterations, t1 - t0, total);

	free(config.clusters);
}
