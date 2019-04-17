#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"
#include <omp.h>

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);

int find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float **pts,         /* [npts][nfeatures] */
                       int     npts)
{
    int index, i;
    float min_dist=FLT_MAX;
    /* find the cluster center id with min distance to pt */   
    for (i=0; i<npts; i++) {
        float dist;
	dist = euclid_dist_2(pt, pts[i], nfeatures);  /* no need square root */
       	 if (dist < min_dist) {
            min_dist = dist;
            index    = i;
         }
     }
    
    return(index);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline
float euclid_dist_2(float *pt1,
                    float *pt2,
                    int    numdims)
{
    int i;
    float ans=0.0;
//#pragma omp paralell for schedule(static)\
//    			reduction(+:ans)
    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);
    return(ans);
}


/*----< kmeans_clustering() >---------------------------------------------*/
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) /* out: [npoints] */
{

    int      i, j,k, n=0, index, loop=0;
    int     *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float    delta;
    float  **clusters;   /* out: [nclusters][nfeatures] */
    float  **new_centers;     /* [nclusters][nfeatures] */
    int nthreads;
    
    /*MY ALOCATION*/
    int **local_new_centers_len; //[nthreads][ncluster]
    float ***local_new_centers; // [nthreads][nclusters][nfeatures]
    nthreads= 8; //max number of threads

    /* allocate space for returning variable clusters[] */
    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    for (i=1; i<nclusters; i++)
        clusters[i] = clusters[i-1] + nfeatures;


    /* randomly pick cluster centers */   
    for (i=0; i<nclusters;i++) {  
	    for (j=0; j<nfeatures; j++)
            clusters[i][j] = feature[n][j];
	 n++;
    }

    for (i=0; i<npoints; i++)
		membership[i] = -1;
    /** Initalization of local varibales**/   
    local_new_centers_len    = (int**) malloc(nthreads * sizeof(int*));
    local_new_centers_len[0] = (int*)  calloc(nthreads*nclusters,sizeof(int));
    for (i=1; i<nthreads; i++)
         local_new_centers_len[i] = local_new_centers_len[i-1]+nclusters;

    /* local_new_centers is a 3D array */
    local_new_centers    =(float***)malloc(nthreads * sizeof(float**));
    local_new_centers[0] =(float**) malloc(nthreads * nclusters * sizeof(float*));
    for (i=1; i<nthreads; i++)
            local_new_centers[i] = local_new_centers[i-1] + nclusters;
    for (i=0; i<nthreads; i++) {
            for (j=0; j<nclusters; j++) {
                local_new_centers[i][j] = (float*)calloc(nfeatures, sizeof(float));
            }
        }
    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int*) calloc(nclusters, sizeof(int));

    new_centers    = (float**) malloc(nclusters *            sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;
 
 
    do {	   
        delta = 0.0;
#pragma omp parallel
	{
	int tid = omp_get_thread_num();
	nthreads= omp_get_num_threads();
	#pragma omp for\
		private(i,j,index)\
		firstprivate(npoints,nfeatures,nclusters)\
		schedule(static)\
		reduction(+:delta)
        for (i=0; i<npoints; i++) {
	        /* find the index of nestest cluster centers */
	        index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);
	        /* if membership changes, increase delta by 1 */
	        if (membership[i] != index) delta += 1.0;

	        /* assign the membership to object i */
	        
		membership[i] = index;

	        /* update new cluster centers : sum of objects located within */
		local_new_centers_len[tid][index]++;
                for (j=0; j<nfeatures; j++)
                        local_new_centers[tid][index][j] += feature[i][j];
        }
     }//END OF PRAGMA OMP PARALLEL
	
	//**Reduction in main thread	
      	for (i=0; i<nclusters; i++) {
              for (j=0; j<nthreads; j++) {
                  new_centers_len[i] += local_new_centers_len[j][i];
                  local_new_centers_len[j][i] = 0.0;
                  for (k=0; k<nfeatures; k++) {
                        new_centers[i][k] += local_new_centers[j][i][k];
                        local_new_centers[j][i][k] = 0.0;
                  }
              }
          }

	/* replace old cluster centers with new_centers */	
//#pragma omg parallel for collapse(2)\
//			schedule(static)
	for (i=0; i<nclusters; new_centers_len[i++]=0){  	     
	   for (j=0; j<nfeatures; j++) {
                if (new_centers_len[i] > 0)
					clusters[i][j] = new_centers[i][j] / new_centers_len[i];
				new_centers[i][j] = 0.0;   /* set back to 0 */
			}
	   }
        //delta /= npoints;
    } while (delta > threshold);
  
  
    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}

