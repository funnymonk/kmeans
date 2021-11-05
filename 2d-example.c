#include <math.h>
#include <stdio.h>

#include "kmeans.h"
#include "dataset.h"

#define ARRAY_LEN(X)    (sizeof(X)/sizeof((X)[0]))
#define DIM (2)

static float d_distance(const Pointer a, const Pointer b)
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

static void d_centroid(const Pointer * objs, const int * clusters, size_t num_objs, int cluster, Pointer centroid)
{
	int i, dim;
	int num_cluster = 0;
	float sum = 0;
	float **floats = (float**)objs;
	float *dcentroid = (float*)centroid;

	if (num_objs <= 0) return;
    for(dim = 0; dim<DIM; dim++)
        dcentroid[dim] = 0.0;

	for (i = 0; i < num_objs; i++)
	{
		/* Only process objects of interest */
		if (clusters[i] != cluster)
			continue;

        for(dim = 0; dim<DIM; dim++){
            dcentroid[dim] += floats[i][dim];
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
	float c[2][2] = {{5.0, 1.0}, {6.0, 6.0}};
	kmeans_config config;
	kmeans_result result;
	int i, dim, num_in_0;

	config.num_objs = ARRAY_LEN(dataset);
	config.k = 2;
	config.max_iterations = 100;
	config.distance_method = d_distance;
	config.centroid_method = d_centroid;

	config.objs = calloc(config.num_objs, sizeof(Pointer));
	config.centers = calloc(config.k, sizeof(Pointer)); 
	config.clusters = calloc(config.num_objs, sizeof(int));

	/* populate objs */
	for (i = 0; i < config.num_objs; i++)
	{
		config.objs[i] = &dataset[i];
	}

	/* populate centroids */
	for (i = 0; i < config.k; i++)
	{
		config.centers[i] = &c[i];
	}

	/* run k-means */
	result = kmeans(&config);

    num_in_0 = 0;
	/* print result */
	for (i = 0; i < config.num_objs; i++)
	{
#ifdef DEBUG_PRINT
		if (config.objs[i])
			printf("%f [%d]\n", *(float*)(config.objs[i]), config.clusters[i]);
		else
			printf("NN Dafuq [%d]\n", config.clusters[i]);
#endif
        if(config.clusters[i] == 0)
            num_in_0++;
	}

	for (i = 0; i < config.k; i++){
        printf("Centroid %d [", i);
        for(dim = 0; dim<DIM; dim++){
            printf("%f, ", ((float**)(config.centers))[i][dim]);
        }
        printf("]\n");
    }
    printf("Took %d iterations, num in 0 = %d\n", 
            config.total_iterations, num_in_0);

	free(config.objs);
	free(config.clusters);
	free(config.centers);
}
