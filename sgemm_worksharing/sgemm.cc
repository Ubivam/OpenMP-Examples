#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <sys/time.h>



bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v)
{
  std::cerr << "Opening file:"<< fn << std::endl;
  std::fstream f(fn, std::fstream::in);
  if ( !f.good() ) {
    return false;
  }

  // Read # of rows and cols
  f >> nr_row;
  f >> nr_col;

  float data;
  std::cerr << "Matrix dimension: "<<nr_row<<"x"<<nr_col<<std::endl;
  while (f.good() ) {
    f >> data;
    v.push_back(data);
  }
  v.pop_back(); // remove the duplicated last element

}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col, std::vector<float>&v)
{
  std::cerr << "Opening file:"<< fn << " for write." << std::endl;
  std::fstream f(fn, std::fstream::out);
  if ( !f.good() ) {
    return false;
  }

  // Read # of rows and cols
  f << nr_row << " "<<nr_col<<" ";

  float data;
  std::cerr << "Matrix dimension: "<<nr_row<<"x"<<nr_col<<std::endl;
  for (int i = 0; i < v.size(); ++i) {
    f << v[i] << ' ';
  }
  f << "\n";
  return true;

}

/* 
 * Base C implementation of MM
 */

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

#pragma omp parallel for collapse(2)\
  			schedule(static,k)
  for (int mm = 0; mm <m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
       float c = 0.0f;
       for (int i = 0; i < k; ++i) {
         float a = A[mm + i * lda]; 
         float b = B[nn + i * ldb];
         c += a * b;
       }
       C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
}
int main (int argc, char *argv[]) {

  struct timeval t1,t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

   if (argc != 4)  
  {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
  }
  /* Read in data */
  // load A
  readColMajorMatrixFile(argv[1], matArow, matAcol, matA);

  // load B^T
  readColMajorMatrixFile(argv[2], matBcol, matBrow, matBT);

  // allocate space for C
  std::vector<float> matC(matArow*matBcol);

  // Use standard sgemm interface
  basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matC.front(), matArow);

  writeColMajorMatrixFile(argv[3], matArow, matBcol, matC); 
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec)*1000.0;
  elapsedTime += (t2.tv_usec - t1.tv_usec) /1000.0;
  std::cout << elapsedTime << "ms.\n";
  return 0;
}
