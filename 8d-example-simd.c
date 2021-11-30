#include <math.h>
#include <stdio.h>
#include "immintrin.h"
#include <assert.h>
#include <omp.h>

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


#define DEBUG
#ifdef DEBUG
#define dbg_printf  printf
#else
#define dbg_printf(...)
#endif

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

/* Unreadable transpose function. Works though :) */
static void transpose_8_kernel(float *dest, float *src, int src_offset, int dst_offest)
{
    register __m256i r0, r1, r2, r3, r4, r5, r6, r7;
    register __m256i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
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
    r1 = _mm256_loadu_si256(src+1*src_offset);
    r2 = _mm256_loadu_si256(src+2*src_offset);
    r3 = _mm256_loadu_si256(src+3*src_offset);
    r4 = _mm256_loadu_si256(src+4*src_offset);
    r5 = _mm256_loadu_si256(src+5*src_offset);
    r6 = _mm256_loadu_si256(src+6*src_offset);
    r7 = _mm256_loadu_si256(src+7*src_offset);

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

static void d_transpose(float *transpose, float *objs, int num_objs, 
        int src_offset, int src_increment, 
        int dst_offset, int dst_increment)
{

    float *dest;
    float *src;
    int n = 0;
    
    for(n=0; n<num_objs/DIM; n++){
        src = objs + n*src_increment;
        dest = transpose + n*dst_increment;
        transpose_8_kernel(dest, src, src_offset, dst_offset);
    }

}

/* Compute distance of 56 points(in 7 vectors) */
static void distance_7_kernel(float *dest, float *src, float *c_src, int src_offset)
{
    register __m256 c;
    register __m256 r0, r1, r2, r3, r4, r5, r6;
    register __m256 acc0, acc1, acc2, acc3, acc4, acc5, acc6;
    float *d;

    /* Init accumulaors */
    acc0 = _mm256_setzero_ps();
    acc1 = _mm256_setzero_ps();
    acc2 = _mm256_setzero_ps();
    acc3 = _mm256_setzero_ps();
    acc4 = _mm256_setzero_ps();
    acc5 = _mm256_setzero_ps();
    acc6 = _mm256_setzero_ps();

    for(int i=0; i<DIM; i++){
        c = _mm256_broadcast_ss(c_src + i);

        /* Load tranposed points */
        r0 = _mm256_loadu_ps(src);
        r1 = _mm256_loadu_ps(src+8);
        r2 = _mm256_loadu_ps(src+16);
        r3 = _mm256_loadu_ps(src+24);
        r4 = _mm256_loadu_ps(src+32);
        r5 = _mm256_loadu_ps(src+40);
        r6 = _mm256_loadu_ps(src+48);

        /* Get diff */
        r0 = _mm256_sub_ps(r0, c);
        r1 = _mm256_sub_ps(r1, c);
        r2 = _mm256_sub_ps(r2, c);
        r3 = _mm256_sub_ps(r3, c);
        r4 = _mm256_sub_ps(r4, c);
        r5 = _mm256_sub_ps(r5, c);
        r6 = _mm256_sub_ps(r6, c);

        /* Accumulate */
        acc0 = _mm256_fmadd_ps(r0, r0, acc0);
        acc1 = _mm256_fmadd_ps(r1, r1, acc1);
        acc2 = _mm256_fmadd_ps(r2, r2, acc2);
        acc3 = _mm256_fmadd_ps(r3, r3, acc3);
        acc4 = _mm256_fmadd_ps(r4, r4, acc4);
        acc5 = _mm256_fmadd_ps(r5, r5, acc5);
        acc6 = _mm256_fmadd_ps(r6, r6, acc6);

        /* Now we have distance^2 for 56 points in 1 dim 
         * Loop to get all dims */

        src += src_offset;
    }


    /* Now store out computed distance for this cluster */
    _mm256_storeu_ps(dest, acc0);
    _mm256_storeu_ps(dest+8, acc1);
    _mm256_storeu_ps(dest+16, acc2);
    _mm256_storeu_ps(dest+24, acc3);
    _mm256_storeu_ps(dest+32, acc4);
    _mm256_storeu_ps(dest+40, acc5);
    _mm256_storeu_ps(dest+48, acc6);


}

static float d_distance(kmeans_config *config)
{
    int i;

#pragma omp parallel for 
    for(int k=0; k < config->k; k++) {
        for(int j=0; j<config->num_objs/(DIM*DISTANCE_KERNEL_NUM_POINTS); j++){
            int src_offset = config->num_objs;
            float *src = config->transpose_arr + j*DIM*DISTANCE_KERNEL_NUM_POINTS;
            float *dest = config->distance_arr + k*config->num_objs + j*DIM*DISTANCE_KERNEL_NUM_POINTS;
            distance_7_kernel(dest, src, config->centers + k*DIM, src_offset);
        }
    }
}

/* Kernel to find the closest cluster (min operation)
 * It then outputs a mask vector, with the cluster it 
 * belongs to having the value 1.0, and remaining as 0 */
static void compare_8_kernel(float *dest, float *d_src, int d_offset)
{
    register __m256 r0, r1, r2, r3, r4, r5, r6, r7;
    register __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    register __m256 min;

    /* Load distances of 8 points from 8 clusters */
    r0 = _mm256_loadu_ps(d_src);
    r1 = _mm256_loadu_ps(d_src+1*d_offset);
    r2 = _mm256_loadu_ps(d_src+2*d_offset);
    r3 = _mm256_loadu_ps(d_src+3*d_offset);
    r4 = _mm256_loadu_ps(d_src+4*d_offset);
    r5 = _mm256_loadu_ps(d_src+5*d_offset);
    r6 = _mm256_loadu_ps(d_src+6*d_offset);
    r7 = _mm256_loadu_ps(d_src+7*d_offset);

    /* Find min */
    t0 = _mm256_min_ps(r0, r1); 
    t2 = _mm256_min_ps(r2, r3); 
    t4 = _mm256_min_ps(r4, r5); 
    t6 = _mm256_min_ps(r6, r7); 
    t0 = _mm256_min_ps(t0, t2); 
    t4 = _mm256_min_ps(t4, t6); 
    min = _mm256_min_ps(t0, t4); 

    /* Perform transpose of distances */
    t0 = (__m256)_mm256_unpacklo_epi32((__m256i)r0, (__m256i)r1);
    t1 = (__m256)_mm256_unpackhi_epi32((__m256i)r0, (__m256i)r1);
    t2 = (__m256)_mm256_unpacklo_epi32((__m256i)r2, (__m256i)r3);
    t3 = (__m256)_mm256_unpackhi_epi32((__m256i)r2, (__m256i)r3);
    t4 = (__m256)_mm256_unpacklo_epi32((__m256i)r4, (__m256i)r5);
    t5 = (__m256)_mm256_unpackhi_epi32((__m256i)r4, (__m256i)r5);
    t6 = (__m256)_mm256_unpacklo_epi32((__m256i)r6, (__m256i)r7);
    t7 = (__m256)_mm256_unpackhi_epi32((__m256i)r6, (__m256i)r7);

    r0 = (__m256)_mm256_unpacklo_epi64((__m256i)t0, (__m256i)t2);
    r1 = (__m256)_mm256_unpackhi_epi64((__m256i)t0, (__m256i)t2);
    r2 = (__m256)_mm256_unpacklo_epi64((__m256i)t1, (__m256i)t3);
    r3 = (__m256)_mm256_unpackhi_epi64((__m256i)t1, (__m256i)t3);
    r4 = (__m256)_mm256_unpacklo_epi64((__m256i)t4, (__m256i)t6);
    r5 = (__m256)_mm256_unpackhi_epi64((__m256i)t4, (__m256i)t6);
    r6 = (__m256)_mm256_unpacklo_epi64((__m256i)t5, (__m256i)t7);
    r7 = (__m256)_mm256_unpackhi_epi64((__m256i)t5, (__m256i)t7);

    t0 = (__m256)_mm256_permute2f128_si256((__m256i)r0, (__m256i)r4, 0x20);
    t1 = (__m256)_mm256_permute2f128_si256((__m256i)r1, (__m256i)r5, 0x20);
    t2 = (__m256)_mm256_permute2f128_si256((__m256i)r2, (__m256i)r6, 0x20);
    t3 = (__m256)_mm256_permute2f128_si256((__m256i)r3, (__m256i)r7, 0x20);
    t4 = (__m256)_mm256_permute2f128_si256((__m256i)r0, (__m256i)r4, 0x31);
    t5 = (__m256)_mm256_permute2f128_si256((__m256i)r1, (__m256i)r5, 0x31);
    t6 = (__m256)_mm256_permute2f128_si256((__m256i)r2, (__m256i)r6, 0x31);
    t7 = (__m256)_mm256_permute2f128_si256((__m256i)r3, (__m256i)r7, 0x31);

    r5 = _mm256_set1_ps(1.0);

    /* broadcast mins */
    r0 = _mm256_permutevar8x32_ps(min, _mm256_set1_epi32(0));
    r1 = _mm256_permutevar8x32_ps(min, _mm256_set1_epi32(1));
    r2 = _mm256_permutevar8x32_ps(min, _mm256_set1_epi32(2));
    r3 = _mm256_permutevar8x32_ps(min, _mm256_set1_epi32(3));
    r4 = _mm256_permutevar8x32_ps(min, _mm256_set1_epi32(4));

    /* Compare */
    r0 = _mm256_cmp_ps(t0, r0, _CMP_EQ_OQ);
    r1 = _mm256_cmp_ps(t1, r1, _CMP_EQ_OQ);
    r2 = _mm256_cmp_ps(t2, r2, _CMP_EQ_OQ);
    r3 = _mm256_cmp_ps(t3, r3, _CMP_EQ_OQ);
    r4 = _mm256_cmp_ps(t4, r4, _CMP_EQ_OQ);

    /* And */
    r0 = _mm256_and_ps(r0, r5);
    r1 = _mm256_and_ps(r1, r5);
    r2 = _mm256_and_ps(r2, r5);
    r3 = _mm256_and_ps(r3, r5);
    r4 = _mm256_and_ps(r4, r5);

    /* Store out */
    _mm256_storeu_ps(dest, r0);
    _mm256_storeu_ps(dest+1*8, r1);
    _mm256_storeu_ps(dest+2*8, r2);
    _mm256_storeu_ps(dest+3*8, r3);
    _mm256_storeu_ps(dest+4*8, r4);

    /* broadcast mins. r0, r1, r2 no longer needed */
    r0 = _mm256_permutevar8x32_ps(min, _mm256_set1_epi32(5));
    r1 = _mm256_permutevar8x32_ps(min, _mm256_set1_epi32(6));
    r2 = _mm256_permutevar8x32_ps(min, _mm256_set1_epi32(7));

    /* Compare */
    r0 = _mm256_cmp_ps(t5, r0, _CMP_EQ_OQ);
    r1 = _mm256_cmp_ps(t6, r1, _CMP_EQ_OQ);
    r2 = _mm256_cmp_ps(t7, r2, _CMP_EQ_OQ);

    /* And */
    r0 = _mm256_and_ps(r0, r5);
    r1 = _mm256_and_ps(r1, r5);
    r2 = _mm256_and_ps(r2, r5);

    /* Store out */
    _mm256_storeu_ps(dest+5*8, r0);
    _mm256_storeu_ps(dest+6*8, r1);
    _mm256_storeu_ps(dest+7*8, r2);
}

/* Compute new cluster assignments
 * Outputs a mask for assignments 
 * */
static void d_centroid(kmeans_config *config)
{
#pragma omp parallel for 
    for(int i=0; i<config->num_objs/CENTROID_KERNEL_NUM_POINTS; i++){
        int d_offset = config->num_objs;
        float *dest = config->mask_arr + i*CENTROID_KERNEL_NUM_POINTS*DIM;
        float *d_src = config->distance_arr + i*DIM;
        compare_8_kernel(dest, d_src, d_offset);
    }
}

static void d_means(kmeans_config *config)
{
    float *dest;
    register __m256 c0, c1, c2, c3, c4, c5, c6, c7;
    register __m256 macc;
    float *d;

    unsigned long long t0, t1;
        
    c0 = _mm256_setzero_ps();
    c1 = _mm256_setzero_ps();
    c2 = _mm256_setzero_ps();
    c3 = _mm256_setzero_ps();
    c4 = _mm256_setzero_ps();
    c5 = _mm256_setzero_ps();
    c6 = _mm256_setzero_ps();
    c7 = _mm256_setzero_ps();
    macc = _mm256_setzero_ps();

    /* Start of kernel
     * Each iteration of the loop accumulates
     * the 8 dimensions of a single point */
    for(int i=0; i<config->num_objs; i++){
        register __m256 r0, r1, r2, r3, r4;
        register __m256 mask;
        float *msrc = config->mask_arr + i*DIM;
        float *src = config->objs + i*DIM;

        /* Add dimensions for 1 point*/
        mask = _mm256_loadu_ps(msrc);

        r0 = _mm256_broadcast_ss(src);
        r1 = _mm256_broadcast_ss(src+1);
        r2 = _mm256_broadcast_ss(src+2);
        r3 = _mm256_broadcast_ss(src+3);
        r4 = _mm256_broadcast_ss(src+4);

        c0 = _mm256_fmadd_ps(r0, mask, c0);
        r0 = _mm256_broadcast_ss(src+5);
        c1 = _mm256_fmadd_ps(r1, mask, c1);
        r1 = _mm256_broadcast_ss(src+6);
        c2 = _mm256_fmadd_ps(r2, mask, c2);
        r2 = _mm256_broadcast_ss(src+7);
        c3 = _mm256_fmadd_ps(r3, mask, c3);
        c4 = _mm256_fmadd_ps(r4, mask, c4);
        c5 = _mm256_fmadd_ps(r0, mask, c5);
        c6 = _mm256_fmadd_ps(r1, mask, c6);
        c7 = _mm256_fmadd_ps(r2, mask, c7);

        macc = _mm256_add_ps(mask, macc);
    }

    /* Now we have sum of all dimensions of all points.
     * c0 stores sum of x dimensions of 8 clusters,
     * c1 stores sum of y dimensions of 8 clusters, and so on.
     * We take the transpose of c0-c7 so that now
     * c0 will store the new centroid coordinates of Centroid 0
     * c1 will store the new centroid coordinates of Centroid 1
     * and so on.(after division, of course)
     */
    __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    __m256 b0, b1, b2, b3, b4, b5, b6, b7;

    /* Take transpose */
    tmp0 = _mm256_unpacklo_ps(c0, c1);
    tmp1 = _mm256_unpackhi_ps(c0, c1);
    tmp2 = _mm256_unpacklo_ps(c2, c3);
    tmp3 = _mm256_unpackhi_ps(c2, c3);
    tmp4 = _mm256_unpacklo_ps(c4, c5);
    tmp5 = _mm256_unpackhi_ps(c4, c5);
    tmp6 = _mm256_unpacklo_ps(c6, c7);
    tmp7 = _mm256_unpackhi_ps(c6, c7);

    c0 = (__m256)_mm256_unpacklo_pd((__m256d)tmp0, (__m256d)tmp2);
    c1 = (__m256)_mm256_unpackhi_pd((__m256d)tmp0, (__m256d)tmp2);
    c2 = (__m256)_mm256_unpacklo_pd((__m256d)tmp1, (__m256d)tmp3);
    c3 = (__m256)_mm256_unpackhi_pd((__m256d)tmp1, (__m256d)tmp3);
    c4 = (__m256)_mm256_unpacklo_pd((__m256d)tmp4, (__m256d)tmp6);
    c5 = (__m256)_mm256_unpackhi_pd((__m256d)tmp4, (__m256d)tmp6);
    c6 = (__m256)_mm256_unpacklo_pd((__m256d)tmp5, (__m256d)tmp7);
    c7 = (__m256)_mm256_unpackhi_pd((__m256d)tmp5, (__m256d)tmp7);

    tmp0 = _mm256_permute2f128_ps(c0, c4, 0x20);
    tmp1 = _mm256_permute2f128_ps(c1, c5, 0x20);
    tmp2 = _mm256_permute2f128_ps(c2, c6, 0x20);
    tmp3 = _mm256_permute2f128_ps(c3, c7, 0x20);
    tmp4 = _mm256_permute2f128_ps(c0, c4, 0x31);
    tmp5 = _mm256_permute2f128_ps(c1, c5, 0x31);
    tmp6 = _mm256_permute2f128_ps(c2, c6, 0x31);
    tmp7 = _mm256_permute2f128_ps(c3, c7, 0x31);
    
    b0 = _mm256_permutevar8x32_ps(macc, _mm256_set1_epi32(0));
    b1 = _mm256_permutevar8x32_ps(macc, _mm256_set1_epi32(1));
    b2 = _mm256_permutevar8x32_ps(macc, _mm256_set1_epi32(2));
    b3 = _mm256_permutevar8x32_ps(macc, _mm256_set1_epi32(3));
    b4 = _mm256_permutevar8x32_ps(macc, _mm256_set1_epi32(4));

    /* If nothing assigned to cluster, don't divide(will end
     * up dividing by 0 otherwise)
     */
    d = &b0;
    if(d[0])
        tmp0 = _mm256_div_ps(tmp0, b0);

    d = &b1;
    if(d[0])
        tmp1 = _mm256_div_ps(tmp1, b1);

    d = &b2;
    if(d[0])
        tmp2 = _mm256_div_ps(tmp2, b2);

    d = &b3;
    if(d[0])
        tmp3 = _mm256_div_ps(tmp3, b3);

    d = &b4;
    if(d[0])
        tmp4 = _mm256_div_ps(tmp4, b4);

    b0 = _mm256_permutevar8x32_ps(macc, _mm256_set1_epi32(5));
    b1 = _mm256_permutevar8x32_ps(macc, _mm256_set1_epi32(6));
    b2 = _mm256_permutevar8x32_ps(macc, _mm256_set1_epi32(7));

    d = &b0;
    if(d[0])
        tmp5 = _mm256_div_ps(tmp5, b0);
    d = &b1;
    if(d[0])
        tmp6 = _mm256_div_ps(tmp6, b1);
    d = &b2;
    if(d[0])
        tmp7 = _mm256_div_ps(tmp7, b2);

    dest = config->centers;
    _mm256_storeu_ps(dest, tmp0);
    _mm256_storeu_ps(dest+8, tmp1);
    _mm256_storeu_ps(dest+16, tmp2);
    _mm256_storeu_ps(dest+24, tmp3);
    _mm256_storeu_ps(dest+32, tmp4);
    _mm256_storeu_ps(dest+40, tmp5);
    _mm256_storeu_ps(dest+48, tmp6);
    _mm256_storeu_ps(dest+56, tmp7);

}

void run_kmeans(void)
{
    /* Prepare configs, and call kmeans */
    unsigned long long t0, t1; 
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
	config.means_method = d_means;
    config.transpose_method = d_transpose;
    config.transpose_arr = NULL;
    config.distance_arr = NULL; 
    config.mask_arr = NULL; 

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

    int fin_arr[DIM] = {0};
	/* print result */
	for (i = 0; i < config.num_objs; i++)
	{
        int done = 0;
        for(int j=0; j<DIM; j++){
            if(config.mask_arr[i*DIM + j] == 1.0){
                assert(done == 0);
                done = 1;
                fin_arr[j]++;
            }
        }
	}

	for (i = 0; i < config.k; i++){
        dbg_printf("Centroid %d [", i);
        float *center = config.centers + i*DIM;
        for(dim = 0; dim<DIM; dim++){
            dbg_printf("%f, ", center[dim]);
        }
        dbg_printf("]\n");
    }
    dbg_printf("Num in each :\n");
    int total=0;
    for(int j=0; j<DIM; j++){
        total += fin_arr[j];
        dbg_printf("%d : %d\n", j, fin_arr[j]);
    }
    dbg_printf("Took %d iterations, cycles = %ld, total = %d\n", 
            config.total_iterations, t1 - t0, total);

    free(config.transpose_arr);
    free(config.distance_arr);
    free(config.distance_transpose_arr);
    free(config.mask_arr);
	free(config.clusters);
}

int
main(int nargs, char **args)
{
    unsigned long long t0, t1;
    t0 = rdtsc();
    for(int i=0; i<10; i++)
    {
        printf("iteration %d\n", i);
        run_kmeans();
    }
    t1 = rdtsc();
    printf("Took cycles = %ld\n", 
            t1 - t0);
}
