#include <math.h>
#include <stdio.h>

#include "kmeans.h"
//#include "dataset.h"
#include "dataset_8d.h"

#define ARRAY_LEN(X)    (sizeof(X)/sizeof((X)[0]))

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
	float c[2][8] = {
    {3.673064415363009871e+00,7.842842527838095101e+00,9.690638269043832409e+00,2.721014897817765288e+00,6.100443184789131834e+00,6.222161699622711595e+00,3.216597085650585441e+00,5.819027177689593877e+00},
    {5.454086487568165609e+00,9.674906510200042220e+00,2.205720657951358632e+00,3.023838573736853164e+00,6.111250410705403979e+00,9.669943546798972278e+00,9.296398730435974755e+00,8.188045578492848975e+00},
    };
	kmeans_config config;
	kmeans_result result;
	int i, dim, num_in_0;

	config.num_objs = ARRAY_LEN(dataset);
	config.k = 2;
	config.max_iterations = 1000;
	config.distance_method = d_distance;
	config.centroid_method = d_centroid;

	config.clusters = malloc(config.num_objs * sizeof(int));

    config.centers = &c[0][0];
    config.objs = &dataset[0][0];

	/* run k-means */
	result = kmeans(&config);

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
    printf("Took %d iterations, num in 0 = %d\n", 
            config.total_iterations, num_in_0);

	free(config.clusters);
}
