# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <sys/time.h>
#define MAX 1024
int main (int argc, char *argv[] ) {
  struct timeval t1,t2;
  double elapsedTime;
  double *b;
  double d;
  int i;
  int it;
  int m;
  int n;
  double r;
  double t;
  double *x;
  double *xnew;
  gettimeofday(&t1,NULL);
  if (argc == 3) {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
  } else {
    m = 5000;
    n = 50000;
  }

  b = ( double * ) malloc ( n * sizeof ( double ) );
  x = ( double * ) malloc ( n * sizeof ( double ) );
  xnew = ( double * ) malloc ( n * sizeof ( double ) );

  printf ( "\n" );
  printf ( "JACOBI_OPENMP:\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "  Jacobi iteration to solve A*x=b.\n" );
  printf ( "\n" );
  printf ( "  Number of variables  N = %d\n", n );
  printf ( "  Number of iterations M = %d\n", m );

  printf ( "\n" );
  printf ( "  IT     l2(dX)    l2(resid)\n" );
  printf ( "\n" );
  srand(time(NULL));
/*
  Set up the right hand side.
*/
  for ( i = 0; i < n; i++ )
  { 
//    b[i] = ((rand()*1.0f)/RAND_MAX-0.5f)* MAX*2;
      b[i] = 0.0;
  }

  b[n-1] = ( double ) ( n + 1 );
/*
  Initialize the solution estimate to 0.
  Exact solution is (1,2,3,...,N).
*/
  for ( i = 0; i < n; i++ )
  {
    x[i] = 0.0;
  }

/*
  Iterate M times.
*/
  for ( it = 0; it < m; it++ )
  {
/*
  Jacobi update.
*/
#pragma omp parallel private(i,t)
{	  
    d=0.0;	
    #pragma omp for schedule(dynamic,256)\
			reduction(+:d)
    for ( i = 0; i < n; i++ )
    {
      xnew[i] = b[i];
      if ( 0 < i )
      {
        xnew[i] = xnew[i] + x[i-1];
      }
      if ( i < n - 1 )
      {
        xnew[i] = xnew[i] + x[i+1];
      }
      xnew[i] = xnew[i] / 2.0;
      d = d + pow ( x[i] - xnew[i], 2 );
    }
/*
  Difference.
*/
    r = 0.0;
#pragma omp for schedule(dynamic,256)\
    			reduction(+:r)
    for ( i = 0; i < n; i++ ){
/*
  Overwrite old solution.
*/
      x[i] = xnew[i];
    
/*
  Residual.
*/
      t = b[i] - 2.0 * xnew[i];
      if ( 0 < i )
      {
        t = t + xnew[i-1];
      }
      if ( i < n - 1 )
      {
        t = t + xnew[i+1];
      }
      r = r + t * t;
  }
#pragma omp single
  {
    if ( it < 10 || m - 10 < it )
    {
        printf ( "  %8d  %14.6g  %14.6g\n", it, sqrt ( d ), sqrt ( r ) );
    }
    if ( it == 9 )
    {
       printf ( "  Omitting intermediate results.\n" );
    }
  }
  }
}  
/*
  Write part of final estimate.
*/
  printf ( "\n" );
  printf ( "  Part of final solution estimate:\n" );
  printf ( "\n" );
  for ( i = 0; i < 10; i++ )
  {
    printf ( "  %8d  %14.6g\n", i, x[i] );
  }
  printf ( "...\n" );
  for ( i = n - 11; i < n; i++ )
  {
    printf ( "  %8d  %14.6g\n", i, x[i] );
  }
/*
  Free memory.
*/
  free ( b );
  free ( x );
  free ( xnew );

  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
  printf("ElapsedTime %lf ms.\n", elapsedTime);
  return 0;
}

