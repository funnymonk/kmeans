/*-------------------------------------------------------------------------
*
* kmeans.h
*    Generic k-means implementation
*
* Copyright (c) 2016, Paul Ramsey <pramsey@cleverelephant.ca>
*
*------------------------------------------------------------------------*/


#include <stdlib.h>

/*
* Simple k-means implementation for arbitrary data structures
*
* Since k-means partitions based on inter-object "distance" the same
* machinery can be used to support any object type that can calculate a
* "distance" between pairs.
*
* To use the k-means infrastructure, just fill out the kmeans_config
* structure and invoke the kmeans() function.
*/

/*
* Threaded calculation is available using pthreads, which practically
* means UNIX platforms only, unless you're building with a posix
* compatible environment.
*
* #define KMEANS_THREADED
*/

/*
* When clustering lists with NULL elements, they will get this as
* their cluster number. (All the other clusters will be non-negative)
*/
#define KMEANS_NULL_CLUSTER -1

/*
* If the algorithm doesn't converge within this number of iterations,
* it will return with a failure error code.
*/
#define KMEANS_MAX_ITERATIONS 1000

/*
* The code doesn't try to figure out how many threads to use, so
* best to set this to the number of cores you expect to have
* available. The threshold is the the value of k*n at which to
* move to multi-threading.
*/
#ifdef KMEANS_THREADED
#define KMEANS_THR_MAX 4
#define KMEANS_THR_THRESHOLD 250000
#endif

#define kmeans_malloc(size) malloc(size)
#define kmeans_free(ptr) free(ptr)

/* Our defines */
#define DIM (8)
#define DISTANCE_KERNEL_NUM_POINTS (7)

typedef enum {
	KMEANS_OK,
	KMEANS_EXCEEDED_MAX_ITERATIONS,
	KMEANS_ERROR
} kmeans_result;

/*
* Prototype for the distance calculating function
*/
extern struct kmeans_config;
typedef float (*kmeans_distance_method) (struct kmeans_config *config);

/*
* Prototype for the centroid calculating function
* @param objs the list of all objects in the calculation
* @param clusters the list of cluster numbers for each object
* @param num_objs the number of objects/cluster numbers in the previous arrays
* @param cluster the cluster number we are actually generating a centroid for here
* @param centroid the object to write the centroid result into (already allocated)
*/
typedef void (*kmeans_centroid_method) (float* objs, int * clusters, size_t num_objs, int cluster, float* centroid);
typedef int (*kmeans_convergence_method) (const float *new_centres, const float* old_centers, int num_centres);
typedef void (*kmeans_transpose_method) (float* transposed, float* objs, int num_objs);

typedef struct kmeans_config
{
	/* Function returns the "distance" between any pair of objects */
	kmeans_distance_method distance_method;

	/* Function returns the "centroid" of a collection of objects */
	kmeans_centroid_method centroid_method;

    /* Function returns 0 on convergence, else 1 */
    kmeans_convergence_method convergence_method;

    kmeans_transpose_method transpose_method;

	/* An array of objects to be analyzed. User allocates this array */
	/* and is responsible for freeing it. */
	/* For objects that are not capable of participating in the distance */
	/* calculations, but for which you still want included in the process */
	/* (for examples, database nulls, or geometry empties) use a NULL */
	/* value in this list. All NULL values will be returned in the */
	/* KMEANS_NULL_CLUSTER. */
	float* objs;

	/* Number of objects in the preceding array */
	size_t num_objs;

	/* An array of inital centers for the algorithm */
	/* Can be randomly assigned, or using proportions, */
	/* unfortunately the algorithm is sensitive to starting */
	/* points, so using a "better" set of starting points */
	/* might be wise. User allocates and is responsible for freeing. */
	float* centers;

	/* Number of means we are calculating, length of preceding array */
	unsigned int k;

	/* Maximum number of times to iterate the algorithm, or 0 for */
	/* library default */
	unsigned int max_iterations;

	/* Iteration counter */
	unsigned int total_iterations;

	/* Array to fill in with cluster numbers. User allocates and frees. */
	int * clusters;

    float *transpose_arr;
    float *distance_arr;
} kmeans_config;

/* This is where the magic happens. */
kmeans_result kmeans(kmeans_config *config);

