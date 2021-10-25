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

#include "kmeans.h"

#ifdef KMEANS_THREADED
#include <pthread.h>
#endif

static void
update_r(kmeans_config *config)
{
	int i;

	for (i = 0; i < config->num_objs; i++)
	{
		double distance, curr_distance;
		int cluster, curr_cluster;
		Pointer obj;

		assert(config->objs != NULL);
		assert(config->num_objs > 0);
		assert(config->centers);
		assert(config->clusters);

		obj = config->objs[i];

		/* Initialize with distance to first cluster */
		curr_distance = (config->distance_method)(obj, config->centers[0]);
		curr_cluster = 0;

		/* Check all other cluster centers and find the nearest */
		for (cluster = 1; cluster < config->k; cluster++)
		{
			distance = (config->distance_method)(obj, config->centers[cluster]);
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

static void
update_means(kmeans_config *config)
{
	int i;

	for (i = 0; i < config->k; i++)
	{
		/* Update the centroid for this cluster */
		(config->centroid_method)(config->objs, config->clusters, config->num_objs, i, 
                config->next_centers[i]);
	}
}

static void swap_centres(kmeans_config *config)
{
    Pointer *tmp_center = config->centers;
    config->centers = config->next_centers;
    config->next_centers = tmp_center;
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
    assert(config->convergence_method);
	assert(config->centers);
	assert(config->next_centers);
	assert(config->k);
	assert(config->clusters);
	assert(config->k <= config->num_objs);

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
#if 0
		/* Store the previous state of the clustering */
		memcpy(clusters_last, config->clusters, clusters_sz);
#endif

		update_r(config);
		update_means(config);

		/*
		 * if all the cluster numbers are unchanged since last time,
		 * we are at a stable solution, so we can stop here
		 */

        /* NOTE : Scikit learn uses Frobenius Norm to determine convergence.
         * Replacing the memcmp with Frobenius norm */
#if 0
		if (memcmp(clusters_last, config->clusters, clusters_sz) == 0)
		{
			kmeans_free(clusters_last);
			config->total_iterations = iterations;
			return KMEANS_OK;
		}
#endif

        if((config->convergence_method)(config->centers, config->next_centers, config->k) == 0)
        {
			kmeans_free(clusters_last);
			config->total_iterations = iterations;
			return KMEANS_OK;
        }

		if (iterations++ > config->max_iterations)
		{
			kmeans_free(clusters_last);
			config->total_iterations = iterations;
			return KMEANS_EXCEEDED_MAX_ITERATIONS;
		}

        swap_centres(config);
	}

	kmeans_free(clusters_last);
	config->total_iterations = iterations;
	return KMEANS_ERROR;
}
