#include <math.h>
#include <stdio.h>
#include "immintrin.h"
#include <assert.h>

#include "kmeans-simd.h"
//#include "dataset.h"
//#include "dataset_8d.h"
#include "dataset_test.h"

#if 1
#define DEBUGD(X)\
    d = &X;\
    printf("Printing %s\n", #X); \
    for(int _i=0; _i<4; _i++)\
        printf("X[%d] = %f  ", _i, d[_i]);\
    printf("\n\n");
#define DEBUGS(X)\
    d = &X;\
    printf("Printing %s\n", #X); \
    for(int _i=0; _i<2; _i++)\
        printf("X[%d] = %f  ", _i, d[_i]);\
    printf("\n\n");
#define DEBUGF(X)\
    d = &X;\
    printf("Printing %s\n", #X); \
    for(int _i=0; _i<8; _i++)\
        printf("X[%d] = %f  ", _i, d[_i]);\
    printf("\n\n");
#else 
#define DEBUGD(X)
#define DEBUGS(X)
#endif
#define ARRAY_LEN(X)    (sizeof(X)/sizeof((X)[0]))

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


static void transpose_8(float *dest, float *src, int dst_offest)
{
    __m256i r0, r1, r2, r3, r4, r5, r6, r7;
    __m256i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    float *dest0, *dest1, *dest2,  *dest3,  *dest4,  *dest5,  *dest6,  *dest7;
    float *d;

    dest0 = dest;
    dest1 = dest + 1*dst_offest;
    dest2 = dest + 2*dst_offest;
    dest3 = dest + 3*dst_offest;
    dest4 = dest + 4*dst_offest;
    dest5 = dest + 5*dst_offest;
    dest6 = dest + 6*dst_offest;
    dest7 = dest + 7*dst_offest;

    r0 = _mm256_loadu_si256(src);
    r1 = _mm256_loadu_si256(src+8);
    r2 = _mm256_loadu_si256(src+16);
    r3 = _mm256_loadu_si256(src+24);
    r4 = _mm256_loadu_si256(src+32);
    r5 = _mm256_loadu_si256(src+40);
    r6 = _mm256_loadu_si256(src+48);
    r7 = _mm256_loadu_si256(src+56);

    tmp0 = _mm256_unpacklo_epi32(r0, r1);
    tmp1 = _mm256_unpackhi_epi32(r0, r1);
    tmp2 = _mm256_unpacklo_epi32(r2, r3);
    tmp3 = _mm256_unpackhi_epi32(r2, r3);
    tmp4 = _mm256_unpacklo_epi32(r4, r5);
    tmp5 = _mm256_unpackhi_epi32(r4, r5);
    tmp6 = _mm256_unpacklo_epi32(r6, r7);
    tmp7 = _mm256_unpackhi_epi32(r6, r7);

    r0 = _mm256_unpacklo_epi64(tmp0, tmp2);
    r1 = _mm256_unpackhi_epi64(tmp0, tmp2);
    r2 = _mm256_unpacklo_epi64(tmp1, tmp3);
    r3 = _mm256_unpackhi_epi64(tmp1, tmp3);
    r4 = _mm256_unpacklo_epi64(tmp4, tmp6);
    r5 = _mm256_unpackhi_epi64(tmp4, tmp6);
    r6 = _mm256_unpacklo_epi64(tmp5, tmp7);
    r7 = _mm256_unpackhi_epi64(tmp5, tmp7);

    tmp0 = _mm256_permute2f128_si256(r0, r4, 0x20);
    tmp1 = _mm256_permute2f128_si256(r1, r5, 0x20);
    tmp2 = _mm256_permute2f128_si256(r2, r6, 0x20);
    tmp3 = _mm256_permute2f128_si256(r3, r7, 0x20);
    tmp4 = _mm256_permute2f128_si256(r0, r4, 0x31);
    tmp5 = _mm256_permute2f128_si256(r1, r5, 0x31);
    tmp6 = _mm256_permute2f128_si256(r2, r6, 0x31);
    tmp7 = _mm256_permute2f128_si256(r3, r7, 0x31);

    _mm256_storeu_si256(dest0, tmp0);
    _mm256_storeu_si256(dest1, tmp1);
    _mm256_storeu_si256(dest2, tmp2);
    _mm256_storeu_si256(dest3, tmp3);
    _mm256_storeu_si256(dest4, tmp4);
    _mm256_storeu_si256(dest5, tmp5);
    _mm256_storeu_si256(dest6, tmp6);
    _mm256_storeu_si256(dest7, tmp7);

}

static void d_transpose(float *transpose, float *objs, int num_objs)
{
    /* Procedure : 
     * 1. Load 8 values
     * 2.
     * */
    float *dest = transpose;
    float *src = objs;
    int n = 0;
    int offset = num_objs; 
    
    while(n<num_objs/DIM){
        transpose_8(dest, src, offset);
        n++;
        src += DIM*8;
        dest += DIM;
    }

    for(int i=0; i<num_objs; i++){
        printf("row %d : ", i);
        for(int j=0; j<DIM; j++){
            printf("%f  ", *(transpose + i*DIM + j));
        }
        printf("\n");
    }


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
    config.transpose_method = d_transpose;

    printf("%d\n", config.num_objs);
	config.clusters = malloc(config.num_objs * sizeof(int));
    if(config.clusters == NULL){
        assert(1);
    }

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
    printf("Took %d iterations, cycles = %ld, num in 0 = %d\n", 
            config.total_iterations, t1 - t0, num_in_0);

	free(config.clusters);
}
