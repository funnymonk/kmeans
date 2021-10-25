#include <math.h>
#include <stdio.h>

#include "kmeans.h"


static float d_distance(const Pointer a, const Pointer b)
{
	float da = *((float*)a);
	float db = *((float*)b);
	return fabs(da - db);
}

static void d_centroid(const Pointer * objs, const int * clusters, size_t num_objs, int cluster, Pointer centroid)
{
	int i;
	int num_cluster = 0;
	float sum = 0;
	float **floats = (float**)objs;
	float *dcentroid = (float*)centroid;

	if (num_objs <= 0) return;

	for (i = 0; i < num_objs; i++)
	{
		/* Only process objects of interest */
		if (clusters[i] != cluster)
			continue;

		sum += *(floats[i]);
		num_cluster++;
	}
	if (num_cluster)
	{
		sum /= num_cluster;
		*dcentroid = sum;
	}
	return;
}

static int d_convergence(const Pointer *new, const Pointer *old, int len)
{
    int i;
    float **new_floats = (float**)new;
    float **old_floats = (float**)old;
    float sum_new=0.0;
    float sum_old=0.0;
    for(i=0; i<len; i++){
        sum_new += (*(new_floats[i]))*(*(new_floats[i]));
        sum_old += (*(old_floats[i]))*(*(old_floats[i]));
    }

    if(sum_new - sum_old < 0.0001){
        return 0;
    }

    return 1;
}

int
main(int nargs, char **args)
{
	float v[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
	float c[2] = {9.33, 5.14};
	float oc[2] = {0.0, 0.0};
	kmeans_config config;
	kmeans_result result;
	int i;

	config.num_objs = 10;
	config.k = 2;
	config.max_iterations = 100;
	config.distance_method = d_distance;
	config.centroid_method = d_centroid;
	config.convergence_method = d_convergence;

	config.objs = calloc(config.num_objs, sizeof(Pointer));
	config.centers = calloc(config.k, sizeof(Pointer)); 
	config.next_centers = calloc(config.k, sizeof(Pointer)); 
	config.clusters = calloc(config.num_objs, sizeof(int));

	/* populate objs */
	for (i = 0; i < config.num_objs; i++)
	{
		config.objs[i] = &v[i];
	}

	/* populate centroids */
	for (i = 0; i < config.k; i++)
	{
		config.centers[i] = &c[i];
		config.next_centers[i] = &oc[i];
	}

	/* run k-means */
	result = kmeans(&config);

	/* print result */
	for (i = 0; i < config.num_objs; i++)
	{
		if (config.objs[i])
			printf("%f [%d]\n", *(float*)(config.objs[i]), config.clusters[i]);
		else
			printf("NN Dafuq [%d]\n", config.clusters[i]);
	}

	for (i = 0; i < config.k; i++){
        printf("Centroid %d [%f]\n", i, *(float*)(config.next_centers[i]));
    }
    printf("Took %d iterations\n", config.total_iterations);

	free(config.objs);
	free(config.clusters);
	free(config.centers);

	return 0;
}

