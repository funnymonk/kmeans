/*-------------------------------------------------------------------------
*
* kmeans.c
*    Generic k-means implementation
*
* Copyright (c) 2016, Paul Ramsey <pramsey@cleverelephant.ca>
*
*------------------------------------------------------------------------*/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "kmeans-simd.h"

#ifdef KMEANS_THREADED
#include <pthread.h>
#endif

static void update_distances(kmeans_config *config)
{
    /* This handles computation for 2 cluster */
    (config->distance_method)(config);
}

static void update_distances_transpose(kmeans_config *config)
{
    /* This handles computation for 2 cluster */
    (config->transpose_method)(config->distance_transpose_arr, 
            config->distance_arr, config->num_objs*config->k, 
            config->num_objs, DIM,
            8, DIM*8);

}

#if 0
static void
update_r(kmeans_config *config)
{
	int i;

	for (i = 0; i < config->num_objs; i++)
	{
		double distance, curr_distance;
		int cluster, curr_cluster;
		float* obj;

		assert(config->objs != NULL);
		assert(config->num_objs > 0);
		assert(config->centers);
		assert(config->clusters);

		obj = config->objs + i*DIM;

		/* Initialize with distance to first cluster */
		curr_distance = (config->distance_method)(obj, config->centers);
		curr_cluster = 0;

		/* Check all other cluster centers and find the nearest */
		for (cluster = 1; cluster < config->k; cluster++)
		{
			distance = (config->distance_method)(obj, config->centers + cluster*DIM);
#ifdef DEBUG_PRINT
            printf("Distance of [%f, %f] from cluster %d = [%f, %f] = %f. Current : %f\n",
                    obj[0], obj[1], 
                    cluster,
                    (config->centers + cluster*DIM)[0],
                    (config->centers + cluster*DIM)[1],
                    distance,
                    curr_distance);
#endif
			if (distance < curr_distance)
			{
				curr_distance = distance;
				curr_cluster = cluster;
			}
		}

		/* Store the nearest cluster this object is in */
		config->clusters[i] = curr_cluster;
	}
}
#endif

static void update_centers(kmeans_config *config)
{
    /* Get masks for all points */
    (config->centroid_method)(config);
}

static void
update_means(kmeans_config *config)
{
    (config->means_method)(config);
}

kmeans_result
kmeans(kmeans_config *config)
{
	int iterations = 0;
	int *clusters_last;
	size_t clusters_sz = sizeof(int)*config->num_objs;

	assert(config);
	assert(config->objs);
	assert(config->num_objs);
	assert(config->distance_method);
	assert(config->centroid_method);
	assert(config->centers);
	assert(config->k);
	assert(config->clusters);
	assert(config->k <= config->num_objs);
    assert(config->transpose_arr == NULL);
    assert(config->distance_arr == NULL);
    assert(config->mask_arr == NULL);
    
    config->transpose_arr = malloc(config->num_objs * sizeof(float) * DIM);
    config->distance_arr = malloc(config->num_objs * sizeof(float) * config->k);
    config->distance_transpose_arr = malloc(config->num_objs * sizeof(float) * config->k);
    config->mask_arr = malloc(config->num_objs * sizeof(float) * DIM);

    (config->transpose_method)(config->transpose_arr, config->objs, config->num_objs,
            8, DIM*8,
            config->num_objs, DIM);

	/* Zero out cluster numbers, just in case user forgets */
	memset(config->clusters, 0, clusters_sz);

	/* Set default max iterations if necessary */
	if (!config->max_iterations)
		config->max_iterations = KMEANS_MAX_ITERATIONS;

	/*
	 * Previous cluster state array. At this time, r doesn't mean anything
	 * but it's ok
	 */
	clusters_last = kmeans_malloc(clusters_sz);

	while (1)
	{
        /* Compute distance of each point from the clusters */
		update_distances(config);

        /* transpose the distances */
        //update_distances_transpose(config);
        
        /* Compute new cluster assignments */
		update_centers(config);

        /* Recompute cluster centers */
		update_means(config);

		/*
		 * if all the cluster numbers are unchanged since last time,
		 * we are at a stable solution, so we can stop here
		 */

        /* NOTE : We don't care about convergence. Only max_iterations */

        iterations++;
		if (iterations >= config->max_iterations)
		{
			kmeans_free(clusters_last);
			config->total_iterations = iterations;
			return KMEANS_EXCEEDED_MAX_ITERATIONS;
		}
	}

	kmeans_free(clusters_last);
	config->total_iterations = iterations;
	return KMEANS_ERROR;
}
